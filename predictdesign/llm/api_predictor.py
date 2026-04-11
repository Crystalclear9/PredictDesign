from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ExperimentConfig
from ..ctdg import ContinuousTimeDynamicGraph
from ..messages import Message
from ..prediction import (
    GraphActionType,
    PredictedGraphAction,
    PredictionRollout,
    PredictionSubgraphRollout,
)
from ..temporal_graph import TemporalEdge, TemporalGraph, TemporalNode
from ..types import ensure_tensor


CompletionFn = Callable[[str, str, ExperimentConfig], str]


class LLMApiGraphActionPredictor(nn.Module):
    supports_gradient_training = False

    def __init__(
        self,
        config: ExperimentConfig,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.completion_fn = completion_fn

    def predict_next_action(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
    ) -> PredictedGraphAction:
        return self.predict_action_set(temporal_graph, ctdg, observation_time)[0]

    def predict_action_set(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
    ) -> list[PredictedGraphAction]:
        prompt = self._build_user_prompt(
            temporal_graph=temporal_graph,
            ctdg=ctdg,
            observation_time=observation_time,
            horizon=1,
        )
        raw_response = self._complete(prompt)
        actions = self._parse_actions(
            raw_response=raw_response,
            observation_time=observation_time,
            temporal_graph=temporal_graph,
        )
        return actions or [self._no_op(observation_time)]

    def predict_rollout(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
        steps: int | None = None,
        time_schedule: list[float] | None = None,
    ) -> PredictionRollout:
        if time_schedule is not None:
            steps = len(time_schedule)
        else:
            steps = steps or self.config.prediction_horizon
        rollout_graph = temporal_graph.clone()
        rollout_ctdg = ctdg.clone_with_graph(rollout_graph)
        actions: list[PredictedGraphAction] = []
        for offset in range(steps):
            step_time = (
                time_schedule[offset]
                if time_schedule is not None
                else observation_time + float(offset + 1)
            )
            action = self.predict_next_action(
                temporal_graph=rollout_graph,
                ctdg=rollout_ctdg,
                observation_time=step_time,
            )
            actions.append(action)
            self.apply_action(
                action=action,
                temporal_graph=rollout_graph,
                ctdg=rollout_ctdg,
                update_state=True,
            )
        return PredictionRollout(actions=actions, temporal_graph=rollout_graph, ctdg=rollout_ctdg)

    def predict_subgraph_rollout(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
        steps: int | None = None,
        time_schedule: list[float] | None = None,
    ) -> PredictionSubgraphRollout:
        if time_schedule is not None:
            steps = len(time_schedule)
        else:
            steps = steps or self.config.prediction_horizon
        rollout_graph = temporal_graph.clone()
        rollout_ctdg = ctdg.clone_with_graph(rollout_graph)
        actions_by_step: list[list[PredictedGraphAction]] = []
        for offset in range(steps):
            step_time = (
                time_schedule[offset]
                if time_schedule is not None
                else observation_time + float(offset + 1)
            )
            prompt = self._build_user_prompt(
                temporal_graph=rollout_graph,
                ctdg=rollout_ctdg,
                observation_time=step_time,
                horizon=max(1, steps - offset),
            )
            raw_response = self._complete(prompt)
            action_set = self._parse_actions(
                raw_response=raw_response,
                observation_time=step_time,
                temporal_graph=rollout_graph,
            )
            if not action_set:
                action_set = [self._no_op(step_time)]
            actions_by_step.append(action_set)
            for action in action_set:
                self.apply_action(
                    action=action,
                    temporal_graph=rollout_graph,
                    ctdg=rollout_ctdg,
                    update_state=True,
                )
        return PredictionSubgraphRollout(
            actions_by_step=actions_by_step,
            temporal_graph=rollout_graph,
            ctdg=rollout_ctdg,
        )

    def apply_action(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        update_state: bool = False,
    ) -> None:
        generated_message: Message | None = None
        if action.action_type == GraphActionType.CREATE_EDGE:
            if action.source_node_id is None or action.target_node_id is None:
                return
            if temporal_graph.has_active_edge(
                action.source_node_id,
                action.target_node_id,
                action.effective_time,
            ):
                return
            temporal_graph.add_edge(
                TemporalEdge(
                    source_node_id=action.source_node_id,
                    target_node_id=action.target_node_id,
                    start_time=action.effective_time,
                    end_time=action.effective_time + self.config.prediction_edge_duration,
                )
            )
            if update_state:
                generated_message = self._build_rollout_message(action, temporal_graph, ctdg)
        elif action.action_type == GraphActionType.REMOVE_EDGE:
            if action.source_node_id is None or action.target_node_id is None:
                return
            temporal_graph.deactivate_edge(
                action.source_node_id,
                action.target_node_id,
                action.effective_time,
            )
            if update_state:
                generated_message = self._build_rollout_message(action, temporal_graph, ctdg)
        elif action.action_type == GraphActionType.ADD_NODE:
            role = action.role or "new_role"
            node_id = action.new_node_id or temporal_graph.generate_node_id(role)
            action.new_node_id = node_id
            temporal_graph.add_node(
                TemporalNode(
                    node_id=node_id,
                    role=role,
                    context=ensure_tensor(None, self.config.context_dim, self.device),
                )
            )
            ctdg.add_node(node_id)
            if update_state:
                generated_message = Message.build_query_message(
                    target_node_id=node_id,
                    time=action.effective_time,
                    context=self._role_seed_context(role),
                    context_dim=self.config.context_dim,
                    device=self.device,
                )
        if update_state and generated_message is not None:
            ctdg.ingest_messages([generated_message])

    def _complete(self, user_prompt: str) -> str:
        if self.completion_fn is not None:
            return self.completion_fn(self.config.llm_api.system_prompt, user_prompt, self.config)
        try:
            from openai import APITimeoutError, APIConnectionError, OpenAI, RateLimitError
        except ImportError as exc:
            raise ImportError(
                "openai is required for predictor_backend='llm_api'. Install it with `pip install openai`."
            ) from exc
        api_key = os.getenv("PREDICTDESIGN_LLM_API_KEY", self.config.llm_api.api_key)
        base_url = os.getenv("PREDICTDESIGN_LLM_BASE_URL", self.config.llm_api.base_url)
        model = os.getenv("PREDICTDESIGN_LLM_MODEL", self.config.llm_api.model)
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.config.llm_api.timeout)
        last_error: Exception | None = None
        for attempt in range(max(1, self.config.llm_api.max_retries + 1)):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.config.llm_api.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.llm_api.temperature,
                    max_tokens=self.config.llm_api.max_tokens,
                    stream=False,
                )
                content = response.choices[0].message.content
                return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_error = exc
                if attempt >= self.config.llm_api.max_retries:
                    break
                backoff = self.config.llm_api.retry_backoff_seconds * float(2**attempt)
                time.sleep(backoff)
        assert last_error is not None
        raise last_error

    def _build_user_prompt(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
        horizon: int,
    ) -> str:
        payload = {
            "observation_time": observation_time,
            "prediction_horizon": horizon,
            "candidate_new_roles": list(self.config.candidate_new_roles),
            "candidate_relation_types": list(self.config.candidate_relation_types),
            "graph_summary": {
                "node_count": len(temporal_graph.nodes),
                "active_edge_count": len(temporal_graph.active_edges(observation_time)),
                "structural_edge_count": len(temporal_graph.structural_edges),
            },
            "nodes": [
                {
                    "node_id": node_id,
                    "role": temporal_graph.nodes[node_id].role,
                    "current_output_text": str(temporal_graph.nodes[node_id].context_text or ""),
                    "current_output_summary": self._summarize_tensor(temporal_graph.nodes[node_id].context),
                    "current_state_summary": self._summarize_tensor(ctdg.get_state(node_id)),
                }
                for node_id in sorted(temporal_graph.nodes)
            ],
            "active_edges": [
                {
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "start_time": edge.start_time,
                    "end_time": edge.end_time,
                }
                for edge in temporal_graph.active_edges(observation_time)
            ],
            "structural_edges": [
                {"source_node_id": source_node_id, "target_node_id": target_node_id}
                for source_node_id, target_node_id in sorted(temporal_graph.structural_edges)
            ],
            "recent_messages": [
                {
                    "time": message.time,
                    "action": message.action.value,
                    "source_node_id": message.source_node_id,
                    "target_node_id": message.target_node_id,
                    "raw_text": str(message.metadata.get("raw_text", "")) if message.metadata else "",
                    "relation_type": str(message.metadata.get("relation_type", "")) if message.metadata else "",
                }
                for message in ctdg.message_history
            ],
            "response_format": {
                "predicted_count": "integer",
                "actions": [
                    {
                        "action_type": "create_edge | remove_edge | add_node | no_op",
                        "source_node_id": "optional string",
                        "target_node_id": "optional string",
                        "relation_type": "optional string from candidate_relation_types",
                        "role": "optional string from candidate_new_roles",
                        "new_node_id": "optional string",
                        "score": "optional float between 0 and 1",
                    }
                ],
            },
        }
        return (
            "Predict the next collaboration graph changes.\n"
            "Return JSON only. Do not include explanations.\n"
            "Use the current graph structure and each agent's current output text directly.\n"
            "No prompt-side truncation has been applied to node outputs or message texts.\n"
            "You may also use each agent's current output summary and current state summary as auxiliary signals.\n"
            "You must directly predict from the supplied collaboration graph, node roles, current node outputs, node states, and recent interaction contents.\n"
            "Do not assume hidden rules beyond the provided graph snapshot and node outputs.\n"
            "For each step, infer the most likely next collaboration subgraph changes.\n"
            "Prefer a small, high-confidence action set instead of many low-confidence actions.\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _parse_actions(
        self,
        raw_response: str,
        observation_time: float,
        temporal_graph: TemporalGraph,
    ) -> list[PredictedGraphAction]:
        payload = self._extract_json_payload(raw_response)
        if isinstance(payload, dict):
            raw_actions = payload.get("actions", [])
        elif isinstance(payload, list):
            raw_actions = payload
        else:
            raw_actions = []
        actions: list[PredictedGraphAction] = []
        for item in raw_actions:
            if not isinstance(item, dict):
                continue
            action = self._parse_action_item(item, observation_time)
            if action is None:
                continue
            if not self._is_valid_action(action, temporal_graph, observation_time):
                continue
            actions.append(action)
        if not actions and isinstance(payload, dict):
            predicted_count = int(payload.get("predicted_count", 0) or 0)
            if predicted_count <= 0:
                return [self._no_op(observation_time)]
        actions.sort(key=lambda item: item.score, reverse=True)
        return actions[: self.config.max_actions_per_step]

    def _parse_action_item(
        self,
        item: dict,
        observation_time: float,
    ) -> PredictedGraphAction | None:
        action_name = str(item.get("action_type", "no_op")).strip().lower()
        action_map = {
            "create_edge": GraphActionType.CREATE_EDGE,
            "remove_edge": GraphActionType.REMOVE_EDGE,
            "add_node": GraphActionType.ADD_NODE,
            "no_op": GraphActionType.NO_OP,
        }
        if action_name not in action_map:
            return None
        relation_type = item.get("relation_type")
        if relation_type not in self.config.candidate_relation_types:
            relation_type = (
                "communication"
                if action_map[action_name] in {GraphActionType.CREATE_EDGE, GraphActionType.REMOVE_EDGE}
                else None
            )
        role = item.get("role")
        if role not in self.config.candidate_new_roles:
            role = None
        try:
            score = float(item.get("score", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        return PredictedGraphAction(
            action_type=action_map[action_name],
            score=score,
            effective_time=observation_time,
            source_node_id=item.get("source_node_id"),
            target_node_id=item.get("target_node_id"),
            relation_type=relation_type,
            role=role,
            new_node_id=item.get("new_node_id"),
        )

    def _is_valid_action(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        observation_time: float,
    ) -> bool:
        if action.action_type == GraphActionType.NO_OP:
            return True
        if action.action_type == GraphActionType.ADD_NODE:
            return action.role is not None
        if action.source_node_id is None or action.target_node_id is None:
            return False
        if action.source_node_id == action.target_node_id:
            return False
        if action.source_node_id not in temporal_graph.nodes or action.target_node_id not in temporal_graph.nodes:
            return False
        if action.action_type == GraphActionType.CREATE_EDGE:
            return not temporal_graph.has_active_edge(
                action.source_node_id,
                action.target_node_id,
                observation_time,
            )
        if action.action_type == GraphActionType.REMOVE_EDGE:
            return temporal_graph.has_active_edge(
                action.source_node_id,
                action.target_node_id,
                observation_time,
            )
        return True

    def _extract_json_payload(self, raw_response: str):
        cleaned = raw_response.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        for candidate in (cleaned, *re.findall(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return {}

    def _summarize_tensor(self, tensor: torch.Tensor, top_k: int = 6) -> list[dict[str, float]]:
        flat = tensor.detach().float().flatten().cpu()
        if flat.numel() == 0:
            return []
        k = min(top_k, flat.numel())
        values, indices = torch.topk(flat.abs(), k=k)
        summary: list[dict[str, float]] = []
        for idx, magnitude in zip(indices.tolist(), values.tolist()):
            summary.append(
                {
                    "index": int(idx),
                    "value": round(float(flat[idx].item()), 6),
                    "magnitude": round(float(magnitude), 6),
                }
            )
        return summary

    def _build_rollout_message(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
    ) -> Message | None:
        if action.source_node_id is None or action.target_node_id is None:
            return None
        source_context = temporal_graph.nodes[action.source_node_id].context.to(self.device)
        target_context = temporal_graph.nodes[action.target_node_id].context.to(self.device)
        context = F.normalize(0.5 * (source_context + target_context), p=2, dim=0, eps=1e-6)
        message = Message.build_completion_message(
            time=action.effective_time,
            source_node_id=action.source_node_id,
            target_node_id=action.target_node_id,
            source_state=ctdg.get_state(action.source_node_id),
            target_state=ctdg.get_state(action.target_node_id),
            context=context,
            hidden_dim=self.config.hidden_dim,
            context_dim=self.config.context_dim,
            device=self.device,
        )
        if action.relation_type:
            message.metadata["relation_type"] = action.relation_type
        return message

    def _role_seed_context(self, role: str) -> torch.Tensor:
        seed = ensure_tensor(None, self.config.context_dim, self.device)
        if self.config.context_dim == 0:
            return seed
        role_hash = abs(hash(role)) % self.config.context_dim
        seed[role_hash] = 1.0
        return seed

    def _no_op(self, observation_time: float) -> PredictedGraphAction:
        return PredictedGraphAction(
            action_type=GraphActionType.NO_OP,
            score=1.0,
            effective_time=observation_time,
        )
