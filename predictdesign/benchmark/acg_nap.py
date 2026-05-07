from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..messages import Message
from ..prediction import GraphActionType, GraphPredictionContext, PredictedGraphAction
from ..temporal_graph import TemporalNode
from .local_results import DatasetCorpus
from .trainer import BenchmarkSplit, BenchmarkTrainer
from .types import BenchmarkEpisode, EpisodeStep


@dataclass(slots=True)
class ACGNapCleaningSummary:
    root_path: str
    source_file_count: int = 0
    raw_record_count: int = 0
    episode_count: int = 0
    removed_graph_profile_count: int = 0
    removed_prediction_query_count: int = 0
    removed_node_profile_count: int = 0
    removed_node_context_count: int = 0
    loaded_transition_candidate_count: int = 0
    loaded_node_latest_output_count: int = 0
    max_graph_profile_chars: int = 240
    max_node_text_chars: int = 480

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_path": self.root_path,
            "source_file_count": self.source_file_count,
            "raw_record_count": self.raw_record_count,
            "episode_count": self.episode_count,
            "removed_fields": {
                "graph.profile": self.removed_graph_profile_count,
                "prediction.query": self.removed_prediction_query_count,
                "graph.nodes.*.profile": self.removed_node_profile_count,
                "graph.nodes.*.context": self.removed_node_context_count,
            },
            "loaded_fields": {
                "graph.profile": self.removed_graph_profile_count,
                "prediction.query": self.removed_prediction_query_count,
                "graph.nodes.*.profile": self.removed_node_profile_count,
                "graph.nodes.*.context": self.removed_node_context_count,
                "graph.nodes.*.latest_output": self.loaded_node_latest_output_count,
                "prediction.transition_candidates": self.loaded_transition_candidate_count,
            },
            "retained_text_limits": {
                "graph_profile_chars": self.max_graph_profile_chars,
                "node_text_chars": self.max_node_text_chars,
            },
        }


@dataclass(slots=True)
class ACGNapCorpus:
    root_path: Path
    datasets: dict[str, DatasetCorpus]
    relation_types: tuple[str, ...]
    role_types: tuple[str, ...]
    cleaning_summary: dict[str, Any]

    def dataset_splits(self, trainer: BenchmarkTrainer) -> dict[str, BenchmarkSplit]:
        return {
            dataset_name: trainer.split_episodes(corpus.episodes)
            for dataset_name, corpus in self.datasets.items()
        }

    def combined_split(self, trainer: BenchmarkTrainer) -> BenchmarkSplit:
        train_episodes: list[BenchmarkEpisode] = []
        eval_episodes: list[BenchmarkEpisode] = []
        for split in self.dataset_splits(trainer).values():
            train_episodes.extend(split.train_episodes)
            eval_episodes.extend(split.eval_episodes)
        return BenchmarkSplit(train_episodes=train_episodes, eval_episodes=eval_episodes)


class ACGNapAdapter:
    def __init__(
        self,
        context_dim: int = 16,
        hidden_dim: int = 32,
        device: str = "cpu",
        max_graph_profile_chars: int = 240,
        max_node_text_chars: int = 480,
    ) -> None:
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_graph_profile_chars = max_graph_profile_chars
        self.max_node_text_chars = max_node_text_chars
        self.cleaning_summary = ACGNapCleaningSummary(
            root_path="",
            max_graph_profile_chars=max_graph_profile_chars,
            max_node_text_chars=max_node_text_chars,
        )
        self._seen_relation_types: set[str] = set()
        self._seen_role_types: set[str] = set()

    @property
    def relation_types(self) -> tuple[str, ...]:
        preferred = ["activate", "delegate", "delegate_return", "retry"]
        ordered = [item for item in preferred if item in self._seen_relation_types]
        remainder = sorted(self._seen_relation_types - set(ordered))
        return tuple(ordered + remainder)

    @property
    def role_types(self) -> tuple[str, ...]:
        preferred = [
            "planner",
            "coding_analyst",
            "coding_implementation",
            "coding_tester",
            "coding_debugger",
            "coding_reviewer",
            "researcher",
        ]
        ordered = [item for item in preferred if item in self._seen_role_types]
        remainder = sorted(self._seen_role_types - set(ordered))
        return tuple(ordered + remainder)

    def load_episode(self, source_path: str | Path, scenario: str) -> BenchmarkEpisode | None:
        path = Path(source_path)
        payloads = self._read_payloads(path)
        if not payloads:
            return None
        first_payload = payloads[0]
        workflow_id = str(first_payload.get("workflow_id") or path.stem)
        initial_nodes = self._build_initial_nodes(first_payload, scenario)
        initial_structural_edges = self._build_structural_edges(first_payload)
        initial_structural_edge_metadata = self._build_structural_edge_metadata(first_payload)
        initial_graph_context_text = self._bootstrap_query_text(first_payload, scenario=scenario)
        steps = self._build_steps(payloads, initial_nodes, scenario)
        if not steps:
            return None
        self._assign_future_targets(steps)
        self.cleaning_summary.episode_count += 1
        return BenchmarkEpisode(
            episode_id=workflow_id,
            dataset_name=scenario,
            initial_nodes=initial_nodes,
            initial_edges=[],
            initial_structural_edges=initial_structural_edges,
            initial_graph_context_text=initial_graph_context_text,
            initial_structural_edge_metadata=initial_structural_edge_metadata,
            steps=steps,
        )

    def load_candidate_episode(self, source_path: str | Path, scenario: str) -> BenchmarkEpisode | None:
        path = Path(source_path)
        payloads = self._read_payloads(path)
        if not payloads:
            return None
        first_payload = payloads[0]
        workflow_id = str(first_payload.get("workflow_id") or path.stem)
        initial_nodes = self._build_initial_nodes(first_payload, scenario)
        initial_structural_edges = self._build_structural_edges(first_payload)
        initial_structural_edge_metadata = self._build_structural_edge_metadata(first_payload)
        initial_graph_context_text = self._bootstrap_query_text(first_payload, scenario=scenario)
        steps = self._build_candidate_steps(payloads, initial_nodes, scenario)
        if not steps:
            return None
        self.cleaning_summary.episode_count += 1
        return BenchmarkEpisode(
            episode_id=workflow_id,
            dataset_name=scenario,
            initial_nodes=initial_nodes,
            initial_edges=[],
            initial_structural_edges=initial_structural_edges,
            initial_graph_context_text=initial_graph_context_text,
            initial_structural_edge_metadata=initial_structural_edge_metadata,
            steps=steps,
        )

    def _read_payloads(self, source_path: Path) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                payloads.append(payload)
                self.cleaning_summary.raw_record_count += 1
                if payload.get("graph", {}).get("profile"):
                    self.cleaning_summary.removed_graph_profile_count += 1
                if payload.get("prediction", {}).get("query"):
                    self.cleaning_summary.removed_prediction_query_count += 1
                self.cleaning_summary.loaded_transition_candidate_count += len(
                    payload.get("prediction", {}).get("transition_candidates") or []
                )
                for node_payload in (payload.get("graph", {}).get("nodes") or {}).values():
                    if node_payload.get("profile"):
                        self.cleaning_summary.removed_node_profile_count += 1
                    if node_payload.get("context"):
                        self.cleaning_summary.removed_node_context_count += 1
                    if node_payload.get("latest_output"):
                        self.cleaning_summary.loaded_node_latest_output_count += 1
        return payloads

    def _build_initial_nodes(
        self,
        payload: dict[str, Any],
        scenario: str,
    ) -> list[TemporalNode]:
        nodes_payload = payload.get("graph", {}).get("nodes") or {}
        nodes: list[TemporalNode] = []
        for node_id in sorted(nodes_payload):
            node_payload = nodes_payload[node_id]
            role = self._extract_role(node_id=node_id, node_payload=node_payload, scenario=scenario)
            context_text = self._node_context_text(node_id=node_id, node_payload=node_payload, role=role)
            node = TemporalNode.build(
                node_id=node_id,
                role=role,
                context=self._text_to_context(context_text),
                context_dim=self.context_dim,
                device=self.device,
            )
            node.context_text = context_text
            nodes.append(node)
            self._seen_role_types.add(role)
        return nodes

    def _build_structural_edges(self, payload: dict[str, Any]) -> list[tuple[str, str]]:
        transitions = payload.get("graph", {}).get("transitions") or []
        edges: set[tuple[str, str]] = set()
        for transition in transitions:
            tails = [str(item) for item in transition.get("tail") or []]
            heads = [str(item) for item in transition.get("head") or []]
            for source_node_id in tails:
                for target_node_id in heads:
                    if source_node_id == target_node_id:
                        continue
                    edges.add((source_node_id, target_node_id))
        return sorted(edges)

    def _build_structural_edge_metadata(
        self,
        payload: dict[str, Any],
    ) -> dict[tuple[str, str], list[dict[str, str]]]:
        metadata: dict[tuple[str, str], list[dict[str, str]]] = {}
        transitions = payload.get("graph", {}).get("transitions") or []
        for transition in transitions:
            relation = str(transition.get("type") or "").strip().lower()
            description = self._compact_text(
                str(transition.get("description") or ""),
                self.max_node_text_chars,
            )
            transition_id = str(transition.get("id") or "")
            tails = [str(item) for item in transition.get("tail") or []]
            heads = [str(item) for item in transition.get("head") or []]
            for source_node_id in tails:
                for target_node_id in heads:
                    if source_node_id == target_node_id:
                        continue
                    item = {
                        "relation_type": relation,
                        "description": description,
                        "transition_id": transition_id,
                    }
                    bucket = metadata.setdefault((source_node_id, target_node_id), [])
                    if item not in bucket:
                        bucket.append(item)
        return metadata

    def _build_steps(
        self,
        payloads: list[dict[str, Any]],
        initial_nodes: list[TemporalNode],
        scenario: str,
    ) -> list[EpisodeStep]:
        steps: list[EpisodeStep] = []
        bootstrap_payload = payloads[0]
        bootstrap_time = max(float(self._time_step(bootstrap_payload)) - 1.0, 0.0)
        bootstrap_query = self._bootstrap_query_text(bootstrap_payload, scenario=scenario)
        bootstrap_messages = self._build_query_messages(
            query_text=bootstrap_query,
            target_node_ids=[node.node_id for node in initial_nodes],
            time_value=bootstrap_time,
        )
        bootstrap_contexts = self._context_updates_from_payload(bootstrap_payload, scenario)
        bootstrap_action = PredictedGraphAction(
            action_type=GraphActionType.NO_OP,
            score=1.0,
            effective_time=bootstrap_time,
        )
        bootstrap_candidates = self._candidate_actions(bootstrap_payload, bootstrap_time)
        steps.append(
            EpisodeStep(
                observation_time=bootstrap_time,
                messages=bootstrap_messages,
                ground_truth_action=bootstrap_action,
                observed_actions=[bootstrap_action],
                candidate_actions=[self._clone_action(action) for action in bootstrap_candidates],
                context_updates={node_id: vector for node_id, vector, _ in bootstrap_contexts},
                context_text_updates={node_id: text for node_id, _, text in bootstrap_contexts},
                prediction_context=self._prediction_context(
                    bootstrap_payload,
                    scenario=scenario,
                    time_value=bootstrap_time,
                    candidate_actions=bootstrap_candidates,
                ),
            )
        )
        for payload in payloads:
            observation_time = float(self._time_step(payload))
            observed_actions = self._observed_actions(payload, observation_time)
            if not observed_actions:
                continue
            candidate_actions = self._candidate_actions(payload, observation_time)
            context_updates = self._context_updates_from_payload(payload, scenario)
            messages = [
                self._action_to_message(
                    action=action,
                    payload=payload,
                    time_value=observation_time,
                )
                for action in observed_actions
            ]
            steps.append(
                EpisodeStep(
                    observation_time=observation_time,
                    messages=messages,
                    ground_truth_action=observed_actions[0],
                    observed_actions=observed_actions,
                    candidate_actions=[self._clone_action(action) for action in candidate_actions],
                    context_updates={node_id: vector for node_id, vector, _ in context_updates},
                    context_text_updates={node_id: text for node_id, _, text in context_updates},
                    prediction_context=self._prediction_context(
                        payload,
                        scenario=scenario,
                        time_value=observation_time,
                        candidate_actions=candidate_actions,
                    ),
                )
            )
        return steps

    def _build_candidate_steps(
        self,
        payloads: list[dict[str, Any]],
        initial_nodes: list[TemporalNode],
        scenario: str,
    ) -> list[EpisodeStep]:
        steps: list[EpisodeStep] = []
        if not payloads:
            return steps
        query_text = self._bootstrap_query_text(payloads[0], scenario=scenario)
        query_messages = self._build_query_messages(
            query_text=query_text,
            target_node_ids=[node.node_id for node in initial_nodes],
            time_value=max(float(self._time_step(payloads[0])) - 1.0, 0.0),
        )
        for index, payload in enumerate(payloads):
            observation_time = float(self._time_step(payload))
            observed_actions = self._observed_actions(payload, observation_time)
            if not observed_actions:
                continue
            candidate_actions = self._candidate_actions(payload, observation_time)
            if not candidate_actions:
                candidate_actions = [self._clone_action(action) for action in observed_actions]
            context_updates = self._context_updates_from_payload(payload, scenario)
            messages: list[Message] = []
            if index == 0:
                messages.extend(query_messages)
            source_message = self._build_source_output_message(payload, observation_time)
            if source_message is not None:
                messages.append(source_message)
            steps.append(
                EpisodeStep(
                    observation_time=observation_time,
                    messages=messages,
                    ground_truth_action=self._clone_action(observed_actions[0]),
                    observed_actions=[self._clone_action(action) for action in observed_actions],
                    valid_next_actions=[self._clone_action(action) for action in observed_actions],
                    candidate_actions=[self._clone_action(action) for action in candidate_actions],
                    context_updates={node_id: vector for node_id, vector, _ in context_updates},
                    context_text_updates={node_id: text for node_id, _, text in context_updates},
                    prediction_context=self._prediction_context(
                        payload,
                        scenario=scenario,
                        time_value=observation_time,
                        candidate_actions=candidate_actions,
                    ),
                )
            )
        return steps

    def _time_step(self, payload: dict[str, Any]) -> float:
        return float((payload.get("time") or {}).get("step") or 0.0)

    def _context_updates_from_payload(
        self,
        payload: dict[str, Any],
        scenario: str,
    ) -> list[tuple[str, list[float], str]]:
        nodes_payload = payload.get("graph", {}).get("nodes") or {}
        updates: list[tuple[str, list[float], str]] = []
        for node_id in sorted(nodes_payload):
            node_payload = nodes_payload[node_id]
            role = self._extract_role(node_id=node_id, node_payload=node_payload, scenario=scenario)
            context_text = self._node_context_text(node_id=node_id, node_payload=node_payload, role=role)
            updates.append((node_id, self._text_to_context(context_text), context_text))
            self._seen_role_types.add(role)
        return updates

    def _observed_actions(
        self,
        payload: dict[str, Any],
        time_value: float,
    ) -> list[PredictedGraphAction]:
        prediction = payload.get("prediction") or {}
        label = prediction.get("label") or {}
        source_node_id = prediction.get("source")
        relation = str(label.get("relation") or "").strip().lower()
        targets = [str(target) for target in (label.get("targets") or []) if str(target)]
        if not relation or source_node_id is None or not targets:
            return []
        self._seen_relation_types.add(relation)
        actions = [
            PredictedGraphAction(
                action_type=GraphActionType.CREATE_EDGE,
                score=1.0,
                effective_time=time_value,
                source_node_id=str(source_node_id),
                target_node_id=target_node_id,
                relation_type=relation,
            )
            for target_node_id in targets
        ]
        for action in actions:
            action.metadata["description"] = self._transition_description(payload, action)
        return actions

    def _candidate_actions(
        self,
        payload: dict[str, Any],
        time_value: float,
    ) -> list[PredictedGraphAction]:
        prediction = payload.get("prediction") or {}
        source_node_id = prediction.get("source")
        if source_node_id is None:
            return []
        actions: list[PredictedGraphAction] = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in prediction.get("transition_candidates") or []:
            relation = str(candidate.get("relation") or "").strip().lower()
            if not relation:
                continue
            self._seen_relation_types.add(relation)
            for target_node_id in (candidate.get("targets") or []):
                target_node_id = str(target_node_id)
                key = (str(source_node_id), relation, target_node_id)
                if key in seen:
                    continue
                seen.add(key)
                actions.append(
                    PredictedGraphAction(
                        action_type=GraphActionType.CREATE_EDGE,
                        score=0.0,
                        effective_time=time_value,
                        source_node_id=str(source_node_id),
                        target_node_id=target_node_id,
                        relation_type=relation,
                        metadata={
                            "transition_id": str(candidate.get("transition_id") or ""),
                            "description": self._compact_text(
                                str(candidate.get("description") or ""),
                                self.max_node_text_chars,
                            ),
                        },
                    )
                )
        return actions

    def _prediction_context(
        self,
        payload: dict[str, Any],
        scenario: str,
        time_value: float,
        candidate_actions: list[PredictedGraphAction],
    ) -> GraphPredictionContext:
        prediction = payload.get("prediction") or {}
        source_node_id = prediction.get("source")
        graph_profile = self._compact_text(
            str(payload.get("graph", {}).get("profile") or ""),
            self.max_graph_profile_chars,
        )
        query_text = self._compact_text(
            str(prediction.get("query") or ""),
            self.max_node_text_chars,
        )
        source_output_text = ""
        nodes_payload = payload.get("graph", {}).get("nodes") or {}
        if source_node_id is not None and str(source_node_id) in nodes_payload:
            source_output_text = self._node_context_text(
                node_id=str(source_node_id),
                node_payload=nodes_payload[str(source_node_id)],
                role=self._extract_role(
                    node_id=str(source_node_id),
                    node_payload=nodes_payload[str(source_node_id)],
                    scenario=scenario,
                ),
            )
        return GraphPredictionContext(
            source_node_id=str(source_node_id) if source_node_id is not None else None,
            query_text=query_text,
            graph_profile_text=graph_profile,
            source_output_text=source_output_text,
            candidate_actions=[self._clone_action(action) for action in candidate_actions],
            metadata={
                "scenario": scenario,
                "workflow_id": str(payload.get("workflow_id") or ""),
                "sample_id": str(payload.get("sample_id") or ""),
                "time_step": str(time_value),
            },
        )

    def _build_source_output_message(
        self,
        payload: dict[str, Any],
        time_value: float,
    ) -> Message | None:
        prediction = payload.get("prediction") or {}
        source_node_id = prediction.get("source")
        nodes_payload = payload.get("graph", {}).get("nodes") or {}
        if source_node_id is None or str(source_node_id) not in nodes_payload:
            return None
        source_payload = nodes_payload[str(source_node_id)]
        raw_text = self._compact_text(
            str(source_payload.get("latest_output") or ""),
            self.max_node_text_chars,
        )
        if not raw_text:
            return None
        message = Message.build_completion_message(
            time=time_value,
            source_node_id=str(source_node_id),
            target_node_id=None,
            context=self._text_to_context(raw_text),
            hidden_dim=self.hidden_dim,
            context_dim=self.context_dim,
            device=self.device,
        )
        message.metadata["raw_text"] = raw_text
        return message

    def _action_to_message(
        self,
        action: PredictedGraphAction,
        payload: dict[str, Any],
        time_value: float,
    ) -> Message:
        source_node_id = action.source_node_id
        target_node_id = action.target_node_id
        if source_node_id is not None and source_node_id == target_node_id:
            message_target = None
        else:
            message_target = target_node_id
        raw_text = self._transition_description(payload, action)
        message = Message.build_completion_message(
            time=time_value,
            source_node_id=source_node_id,
            target_node_id=message_target,
            context=self._text_to_context(raw_text),
            hidden_dim=self.hidden_dim,
            context_dim=self.context_dim,
            device=self.device,
        )
        message.metadata["raw_text"] = raw_text
        if action.relation_type is not None:
            message.metadata["relation_type"] = action.relation_type
        if target_node_id is not None:
            message.metadata["target_node_id"] = target_node_id
        return message

    def _bootstrap_query_text(self, payload: dict[str, Any], scenario: str) -> str:
        workflow_id = str(payload.get("workflow_id") or scenario)
        graph_profile = self._compact_text(
            str(payload.get("graph", {}).get("profile") or ""),
            self.max_graph_profile_chars,
        )
        transition_catalog = self._transition_catalog_text(payload)
        parts = [f"{scenario} workflow {workflow_id}", graph_profile, transition_catalog]
        return self._compact_text(
            "\n".join(part for part in parts if part.strip()),
            self.max_graph_profile_chars + self.max_node_text_chars,
        )

    def _transition_catalog_text(self, payload: dict[str, Any]) -> str:
        snippets: list[str] = []
        for transition in (payload.get("graph", {}).get("transitions") or [])[:12]:
            relation = str(transition.get("type") or "").strip().lower()
            tails = ",".join(str(item) for item in transition.get("tail") or [])
            heads = ",".join(str(item) for item in transition.get("head") or [])
            description = self._compact_text(
                str(transition.get("description") or ""),
                160,
            )
            snippets.append(f"{relation}:{tails}->{heads}; {description}".strip())
        return self._compact_text(" | ".join(snippets), self.max_node_text_chars)

    def _build_query_messages(
        self,
        query_text: str,
        target_node_ids: list[str],
        time_value: float,
    ) -> list[Message]:
        if not query_text:
            return []
        context = self._text_to_context(query_text)
        messages: list[Message] = []
        for target_node_id in target_node_ids:
            message = Message.build_query_message(
                target_node_id=target_node_id,
                time=time_value,
                context=context,
                context_dim=self.context_dim,
                device=self.device,
            )
            message.metadata["raw_text"] = query_text
            message.metadata["query_text"] = query_text
            messages.append(message)
        return messages

    def _transition_description(
        self,
        payload: dict[str, Any],
        action: PredictedGraphAction,
    ) -> str:
        prediction = payload.get("prediction") or {}
        candidates = prediction.get("transition_candidates") or []
        for candidate in candidates:
            relation = str(candidate.get("relation") or "").strip().lower()
            targets = {str(target) for target in (candidate.get("targets") or [])}
            if (
                relation == (action.relation_type or "").strip().lower()
                and action.target_node_id in targets
            ):
                description = str(candidate.get("description") or "").strip()
                if description:
                    return self._compact_text(description, self.max_node_text_chars)
        transitions = payload.get("graph", {}).get("transitions") or []
        for transition in transitions:
            relation = str(transition.get("type") or "").strip().lower()
            tails = {str(item) for item in (transition.get("tail") or [])}
            heads = {str(item) for item in (transition.get("head") or [])}
            if (
                relation == (action.relation_type or "").strip().lower()
                and action.source_node_id in tails
                and action.target_node_id in heads
            ):
                description = str(transition.get("description") or "").strip()
                if description:
                    return self._compact_text(description, self.max_node_text_chars)
        if action.source_node_id and action.target_node_id:
            return f"{action.relation_type}:{action.source_node_id}->{action.target_node_id}"
        if action.source_node_id:
            return f"{action.relation_type}:{action.source_node_id}"
        return str(action.relation_type or "transition")

    def _extract_role(
        self,
        node_id: str,
        node_payload: dict[str, Any],
        scenario: str,
    ) -> str:
        if node_id.upper() == "PLANNER":
            return "planner"
        normalized_profile = str(node_payload.get("profile") or "").lower()
        if "runtime planner" in normalized_profile:
            return "planner"
        if scenario == "coding":
            coding_role_map = {
                "coding analyst": "coding_analyst",
                "implementation agent": "coding_implementation",
                "coding reviewer": "coding_reviewer",
                "coding tester": "coding_tester",
                "coding debugger": "coding_debugger",
            }
            for marker, role in coding_role_map.items():
                if marker in normalized_profile:
                    return role
            return "coding_agent"
        if scenario == "research":
            return "researcher"
        node_type = str(node_payload.get("type") or "agent").strip().lower()
        return node_type or "agent"

    def _node_context_text(
        self,
        node_id: str,
        node_payload: dict[str, Any],
        role: str,
    ) -> str:
        profile = self._compact_text(
            str(node_payload.get("profile") or ""),
            self.max_node_text_chars,
        )
        context = self._compact_text(
            str(node_payload.get("context") or ""),
            self.max_node_text_chars,
        )
        latest_output = self._compact_text(
            str(node_payload.get("latest_output") or ""),
            self.max_node_text_chars,
        )
        parts = [f"role={role}", f"node={node_id}"]
        if profile:
            parts.append(f"profile={profile}")
        if context:
            parts.append(f"context={context}")
        if latest_output:
            parts.append(f"latest_output={latest_output}")
        return self._compact_text("; ".join(parts), self.max_node_text_chars)

    def _compact_text(self, text: str, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if max_chars <= 0 or len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def _assign_future_targets(self, steps: list[EpisodeStep]) -> None:
        for index, step in enumerate(steps):
            if index + 1 < len(steps):
                step.valid_next_actions = [
                    self._clone_action(action)
                    for action in steps[index + 1].observed_actions
                ]
            else:
                step.valid_next_actions = [
                    PredictedGraphAction(
                        action_type=GraphActionType.NO_OP,
                        score=1.0,
                        effective_time=step.observation_time,
                    )
                ]
            step.ground_truth_action = self._clone_action(step.valid_next_actions[0])

    def _clone_action(self, action: PredictedGraphAction) -> PredictedGraphAction:
        return PredictedGraphAction(
            action_type=action.action_type,
            score=action.score,
            effective_time=action.effective_time,
            source_node_id=action.source_node_id,
            target_node_id=action.target_node_id,
            relation_type=action.relation_type,
            role=action.role,
            new_node_id=action.new_node_id,
            metadata=dict(action.metadata),
        )

    def _text_to_context(self, text: str) -> list[float]:
        buckets = [0.0] * self.context_dim
        if not text or self.context_dim <= 0:
            return buckets
        normalized = text.lower().strip()
        tokens = re.findall(r"[a-z0-9_]+", normalized)
        compact_text = re.sub(r"\s+", " ", normalized)

        def add_feature(feature: str, weight: float) -> None:
            if not feature:
                return
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.context_dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            buckets[index] += sign * weight

        for token in tokens:
            add_feature(f"tok:{token}", 1.0)
        for left, right in zip(tokens, tokens[1:]):
            add_feature(f"bigram:{left}|{right}", 0.75)
        for index in range(max(len(compact_text) - 2, 0)):
            trigram = compact_text[index : index + 3]
            add_feature(f"tri:{trigram}", 0.25)
        for index in range(max(len(tokens) - 2, 0)):
            trigram = "|".join(tokens[index : index + 3])
            add_feature(f"toktri:{trigram}", 0.6)

        keyword_weights = {
            "activate": 1.2,
            "delegate": 1.2,
            "retry": 1.2,
            "planner": 1.0,
            "research": 1.0,
            "coding": 1.0,
            "review": 1.0,
            "debug": 1.0,
            "test": 1.0,
            "tool": 0.8,
        }
        for keyword, weight in keyword_weights.items():
            if keyword in normalized:
                add_feature(f"kw:{keyword}", weight)

        if self.context_dim >= 4:
            buckets[-1] = min(len(tokens) / 32.0, 1.0)
            buckets[-2] = min(len(compact_text) / 256.0, 1.0)
            buckets[-3] = compact_text.count(":") / max(len(compact_text), 1)
            buckets[-4] = compact_text.count(";") / max(len(compact_text), 1)

        norm = sum(value * value for value in buckets) ** 0.5
        if norm > 0:
            buckets = [value / norm for value in buckets]
        return buckets


def load_acg_nap_corpus(
    root_path: str | Path,
    adapter: ACGNapAdapter,
    max_files_per_dataset: int | None = None,
) -> ACGNapCorpus:
    root = Path(root_path).resolve()
    adapter.cleaning_summary.root_path = str(root)
    scenario_dirs = {
        "coding": root / "coding",
        "research": root / "research",
    }
    datasets: dict[str, DatasetCorpus] = {}
    for dataset_name, scenario_dir in scenario_dirs.items():
        source_paths = sorted(scenario_dir.glob("*.jsonl")) if scenario_dir.exists() else []
        if max_files_per_dataset is not None and max_files_per_dataset > 0:
            source_paths = source_paths[:max_files_per_dataset]
        adapter.cleaning_summary.source_file_count += len(source_paths)
        episodes = [
            episode
            for episode in (
                adapter.load_episode(source_path, scenario=dataset_name)
                for source_path in source_paths
            )
            if episode is not None
        ]
        datasets[dataset_name] = DatasetCorpus(
            dataset_name=dataset_name,
            source_paths=source_paths,
            episodes=episodes,
        )
    missing = [
        dataset_name
        for dataset_name, corpus in datasets.items()
        if corpus.source_count == 0 or corpus.episode_count == 0
    ]
    if missing:
        joined = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"ACG-NAP data is incomplete under {root}. Missing usable episodes for: {joined}."
        )
    return ACGNapCorpus(
        root_path=root,
        datasets=datasets,
        relation_types=adapter.relation_types,
        role_types=adapter.role_types,
        cleaning_summary=adapter.cleaning_summary.to_dict(),
    )


def load_acg_nap_candidate_corpus(
    root_path: str | Path,
    adapter: ACGNapAdapter,
    max_files_per_dataset: int | None = None,
) -> ACGNapCorpus:
    root = Path(root_path).resolve()
    adapter.cleaning_summary.root_path = str(root)
    scenario_dirs = {
        "coding": root / "coding",
        "research": root / "research",
    }
    datasets: dict[str, DatasetCorpus] = {}
    for dataset_name, scenario_dir in scenario_dirs.items():
        source_paths = sorted(scenario_dir.glob("*.jsonl")) if scenario_dir.exists() else []
        if max_files_per_dataset is not None and max_files_per_dataset > 0:
            source_paths = source_paths[:max_files_per_dataset]
        adapter.cleaning_summary.source_file_count += len(source_paths)
        episodes = [
            episode
            for episode in (
                adapter.load_candidate_episode(source_path, scenario=dataset_name)
                for source_path in source_paths
            )
            if episode is not None
        ]
        datasets[dataset_name] = DatasetCorpus(
            dataset_name=dataset_name,
            source_paths=source_paths,
            episodes=episodes,
        )
    missing = [
        dataset_name
        for dataset_name, corpus in datasets.items()
        if corpus.source_count == 0 or corpus.episode_count == 0
    ]
    if missing:
        joined = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"ACG-NAP candidate data is incomplete under {root}. Missing usable episodes for: {joined}."
        )
    return ACGNapCorpus(
        root_path=root,
        datasets=datasets,
        relation_types=adapter.relation_types,
        role_types=adapter.role_types,
        cleaning_summary=adapter.cleaning_summary.to_dict(),
    )
