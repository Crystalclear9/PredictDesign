from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ..prediction import GraphActionType, GraphPredictionContext, PredictedGraphAction
from ..temporal_graph import TemporalGraph


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "next",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_DONE_TOKENS = {"complete", "completed", "done", "draft", "finish", "finished", "ready"}
_RETRY_TOKENS = {"bug", "error", "fail", "failed", "fix", "retry", "wrong"}


@dataclass(slots=True)
class ColdStartActionPriorScorer:
    """Deterministic action priors for cold-start inference.

    The scorer is deliberately non-parametric. It gives the model a useful
    starting policy before GNN/cross-encoder training has converged, while the
    learned scorer can still act as a residual once data is available.
    """

    source_match_bonus: float = 1.0
    structural_edge_bonus: float = 0.75
    relation_metadata_bonus: float = 0.75
    transition_id_bonus: float = 0.50
    description_match_bonus: float = 0.35
    relation_text_bonus: float = 0.55
    text_overlap_weight: float = 2.0
    target_mention_bonus: float = 0.35
    target_role_query_bonus: float = 1.25
    done_to_delegate_bonus: float = 0.30
    retry_keyword_bonus: float = 0.65
    retry_when_done_penalty: float = -0.50
    scenario_state_bonus: float = 6.0
    scenario_state_fallback_bonus: float = 2.0

    def candidate_score(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        prediction_context: GraphPredictionContext | None,
    ) -> float:
        if action.action_type != GraphActionType.CREATE_EDGE:
            return 0.0
        score = 0.0
        context_text = self._context_text(prediction_context)
        candidate_text = self._candidate_text(action, temporal_graph)
        context_tokens = self._tokens(context_text)
        candidate_tokens = self._tokens(candidate_text)
        relation = str(action.relation_type or "").strip().lower()

        if prediction_context and action.source_node_id == prediction_context.source_node_id:
            score += self.source_match_bonus

        if action.source_node_id and action.target_node_id:
            key = (action.source_node_id, action.target_node_id)
            if key in temporal_graph.structural_edges:
                score += self.structural_edge_bonus
            score += self._metadata_score(
                action=action,
                items=temporal_graph.structural_edge_metadata.get(key, []),
            )

        score += self.text_overlap_weight * self._overlap(context_tokens, candidate_tokens)
        if relation and relation in context_tokens.union(candidate_tokens):
            score += self.relation_text_bonus
        if action.target_node_id:
            target_text = self._node_text(action.target_node_id, temporal_graph)
            target_tokens = self._tokens(f"{action.target_node_id} {target_text}")
            if context_tokens.intersection(target_tokens):
                score += self.target_mention_bonus
            target = temporal_graph.nodes.get(action.target_node_id)
            if target is not None:
                score += self.target_role_query_bonus * self._role_query_match_score(
                    target.role,
                    context_text,
                    context_tokens,
                )

        source_output_tokens = self._tokens(prediction_context.source_output_text if prediction_context else "")
        if source_output_tokens.intersection(_DONE_TOKENS):
            if relation in {"review", "delegate", "delegate_return", "activate"} and action.source_node_id != action.target_node_id:
                score += self.done_to_delegate_bonus
            if relation == "retry":
                score += self.retry_when_done_penalty
        if relation == "retry" and context_tokens.intersection(_RETRY_TOKENS):
            score += self.retry_keyword_bonus

        score += self._scenario_state_score(
            action=action,
            temporal_graph=temporal_graph,
            prediction_context=prediction_context,
        )

        return score

    def _scenario_state_score(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        prediction_context: GraphPredictionContext | None,
    ) -> float:
        if prediction_context is None or action.target_node_id is None:
            return 0.0
        metadata = prediction_context.metadata or {}
        scenario = self._metadata_str(metadata, "scenario").lower()
        current = (
            self._metadata_str(metadata, "current_floor")
            or self._metadata_str(metadata, "floor_agent")
            or str(prediction_context.source_node_id or "")
        )
        lookahead = self._metadata_int(metadata, "lookahead", default=1)
        query_start = self._metadata_bool(metadata, "query_start") or self._metadata_bool(
            metadata, "is_query_start"
        )
        iteration_order = self._metadata_list(metadata.get("iteration_order"))
        if scenario in {"coding", "research"} or iteration_order:
            target = self._round_robin_target(
                iteration_order,
                current,
                lookahead,
                query_start=query_start,
            )
            if target and action.target_node_id == target:
                return self.scenario_state_bonus
            return 0.0

        phase = (
            self._metadata_str(metadata, "phase")
            or self._metadata_str(metadata, "scenario.werewolf.phase")
        ).lower()
        if scenario != "werewolf" and not phase:
            return 0.0
        alive = self._metadata_list(
            metadata.get("alive", metadata.get("scenario.werewolf.alive"))
        )
        alive_set = set(alive)
        if not alive_set:
            alive_set = set(temporal_graph.nodes)
        role_map = self._metadata_mapping(
            metadata.get("role_map", metadata.get("scenario.werewolf.role_map"))
        )
        active_roles = self._metadata_list(
            metadata.get(
                "active_roles",
                metadata.get("active_role", metadata.get("scenario.werewolf.active_roles")),
            )
        )
        if active_roles:
            role_targets = self._targets_for_roles(
                roles=active_roles,
                role_map=role_map,
                alive_set=alive_set,
                temporal_graph=temporal_graph,
            )
            if action.target_node_id in role_targets:
                return self.scenario_state_bonus
            if not role_targets and action.target_node_id in alive_set:
                return self.scenario_state_fallback_bonus
            return 0.0
        if phase == "day":
            speech_order = self._metadata_list(
                metadata.get("speech_order", metadata.get("scenario.werewolf.speech_order"))
            )
            target = self._round_robin_target(
                speech_order,
                current,
                lookahead,
                query_start=query_start,
            )
            if target and action.target_node_id == target:
                return self.scenario_state_bonus
            if not target and action.target_node_id in alive_set:
                return self.scenario_state_fallback_bonus
            return 0.0

        if phase in {"", "night"}:
            night_targets = self._werewolf_night_targets(role_map=role_map, alive_set=alive_set)
            if action.target_node_id in night_targets:
                return self.scenario_state_bonus
            if not night_targets and action.target_node_id in alive_set:
                return self.scenario_state_fallback_bonus
        return 0.0

    def _round_robin_target(
        self,
        iteration_order: list[str],
        current: str,
        lookahead: int,
        *,
        query_start: bool = False,
    ) -> str:
        if not iteration_order:
            return ""
        step = max(1, lookahead)
        if current not in iteration_order:
            if query_start:
                return iteration_order[(step - 1) % len(iteration_order)]
            return ""
        index = len(iteration_order) - 1 - iteration_order[::-1].index(current)
        return iteration_order[(index + step) % len(iteration_order)]

    def _werewolf_night_targets(
        self,
        *,
        role_map: Mapping[str, Any],
        alive_set: set[str],
    ) -> set[str]:
        active_roles = {"wolf", "seer", "witch", "guard"}
        return self._targets_for_roles(
            roles=list(active_roles),
            role_map=role_map,
            alive_set=alive_set,
            temporal_graph=None,
        )

    def _targets_for_roles(
        self,
        *,
        roles: list[str],
        role_map: Mapping[str, Any],
        alive_set: set[str],
        temporal_graph: TemporalGraph | None,
    ) -> set[str]:
        target_roles = {str(role).strip().lower() for role in roles if str(role).strip()}
        if "werewolf" in target_roles:
            target_roles.add("wolf")
        if "wolf" in target_roles:
            target_roles.add("werewolf")
        targets: set[str] = set()
        for role, members in role_map.items():
            if str(role).strip().lower() not in target_roles:
                continue
            for member in self._metadata_list(members):
                if member in alive_set:
                    targets.add(member)
        if targets or temporal_graph is None:
            return targets
        for node_id, node in temporal_graph.nodes.items():
            if node_id in alive_set and node.role.strip().lower() in target_roles:
                targets.add(node_id)
        return targets

    def _metadata_str(self, metadata: Mapping[str, Any], key: str) -> str:
        value = metadata.get(key, "")
        if isinstance(value, str):
            return value.strip()
        return str(value).strip() if value is not None else ""

    def _metadata_int(self, metadata: Mapping[str, Any], key: str, default: int) -> int:
        value = metadata.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _metadata_bool(self, metadata: Mapping[str, Any], key: str) -> bool:
        value = metadata.get(key, False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _metadata_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in re.split(r"[,| ]+", text) if part.strip()]
            return self._metadata_list(parsed)
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _metadata_mapping(self, value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, Mapping):
                return parsed
        return {}

    def edge_prior_matrix(
        self,
        temporal_graph: TemporalGraph,
        prediction_context: GraphPredictionContext | None,
        node_order: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        matrix = torch.zeros((len(node_order), len(node_order)), dtype=dtype, device=device)
        if not node_order:
            return matrix
        context_text = self._context_text(prediction_context)
        context_tokens = self._tokens(context_text)
        source_id = prediction_context.source_node_id if prediction_context else None
        for row, source_node_id in enumerate(node_order):
            for col, target_node_id in enumerate(node_order):
                score = 0.0
                if source_id is not None and source_node_id == source_id:
                    score += 0.60 * self.source_match_bonus
                key = (source_node_id, target_node_id)
                if key in temporal_graph.structural_edges:
                    score += 0.50 * self.structural_edge_bonus
                source_tokens = self._tokens(f"{source_node_id} {self._node_text(source_node_id, temporal_graph)}")
                target_tokens = self._tokens(f"{target_node_id} {self._node_text(target_node_id, temporal_graph)}")
                score += 0.35 * self._overlap(context_tokens, source_tokens)
                score += 0.45 * self._overlap(context_tokens, target_tokens)
                score += self.target_role_query_bonus * self._role_query_match_score(
                    temporal_graph.nodes[target_node_id].role,
                    context_text,
                    context_tokens,
                )
                score += self._scenario_state_score(
                    action=PredictedGraphAction(
                        action_type=GraphActionType.CREATE_EDGE,
                        score=0.0,
                        effective_time=0.0,
                        source_node_id=source_node_id,
                        target_node_id=target_node_id,
                        relation_type="activate",
                    ),
                    temporal_graph=temporal_graph,
                    prediction_context=prediction_context,
                )
                if score:
                    matrix[row, col] = float(score)
        return matrix

    def _metadata_score(
        self,
        action: PredictedGraphAction,
        items: list[dict[str, str]],
    ) -> float:
        score = 0.0
        relation = str(action.relation_type or "").strip().lower()
        transition_id = str(action.metadata.get("transition_id", "")).strip()
        description = str(action.metadata.get("description", "")).strip().lower()
        description_tokens = self._tokens(description)
        for item in items:
            item_relation = str(item.get("relation_type", "")).strip().lower()
            item_transition_id = str(item.get("transition_id", "")).strip()
            item_description = str(item.get("description", "")).strip().lower()
            if relation and item_relation == relation:
                score += self.relation_metadata_bonus
            if transition_id and item_transition_id == transition_id:
                score += self.transition_id_bonus
            if description and item_description:
                item_tokens = self._tokens(item_description)
                if description in item_description or item_description in description:
                    score += self.description_match_bonus
                else:
                    score += self.description_match_bonus * self._overlap(description_tokens, item_tokens)
        return score

    def _context_text(self, prediction_context: GraphPredictionContext | None) -> str:
        if prediction_context is None:
            return ""
        return "\n".join(
            part
            for part in (
                prediction_context.graph_profile_text,
                prediction_context.query_text,
                prediction_context.source_output_text,
                prediction_context.runtime_text,
            )
            if str(part).strip()
        )

    def _candidate_text(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
    ) -> str:
        parts = [
            action.relation_type or "",
            action.source_node_id or "",
            action.target_node_id or "",
            str(action.metadata.get("transition_id", "")),
            str(action.metadata.get("description", "")),
        ]
        if action.source_node_id:
            parts.append(self._node_text(action.source_node_id, temporal_graph))
        if action.target_node_id:
            parts.append(self._node_text(action.target_node_id, temporal_graph))
        return "\n".join(part for part in parts if str(part).strip())

    def _role_query_match_score(
        self,
        role: str,
        context_text: str,
        context_tokens: set[str],
    ) -> float:
        if not role or not context_tokens:
            return 0.0
        role_tokens = self._tokens(role)
        text = context_text.lower()
        phrase_aliases = {
            "wolf": (r"\bas (?:a |the )?werewolf\b", r"\bas (?:a |the )?wolf\b"),
            "werewolf": (r"\bas (?:a |the )?werewolf\b", r"\bas (?:a |the )?wolf\b"),
            "seer": (r"\bas (?:a |the )?seer\b",),
            "witch": (r"\bas (?:a |the )?witch\b",),
            "guard": (r"\bas (?:a |the )?guard\b",),
            "villager": (r"\bas (?:a |the )?villager\b",),
        }
        for token in role_tokens:
            for pattern in phrase_aliases.get(token, ()):
                if re.search(pattern, text):
                    return 2.0
        aliases = {
            "wolf": {"wolf", "werewolf", "werewolves"},
            "werewolf": {"wolf", "werewolf", "werewolves"},
            "seer": {"seer", "check", "checking"},
            "witch": {"witch", "poison", "antidote"},
            "guard": {"guard", "protect", "protection"},
            "villager": {"villager", "village"},
        }
        expanded = set(role_tokens)
        for token in list(role_tokens):
            expanded.update(aliases.get(token, set()))
        return 0.25 if expanded.intersection(context_tokens) else 0.0

    def _node_text(self, node_id: str, temporal_graph: TemporalGraph) -> str:
        node = temporal_graph.nodes.get(node_id)
        if node is None:
            return ""
        return f"{node.role}\n{node.context_text}"

    def _tokens(self, text: str) -> set[str]:
        tokens = {
            token
            for token in re.findall(r"[a-z0-9_]+", text.lower())
            if len(token) > 1 and token not in _STOPWORDS
        }
        expanded: set[str] = set()
        for token in tokens:
            expanded.add(token)
            expanded.update(part for part in token.split("_") if len(part) > 1)
        return expanded

    def _overlap(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / math.sqrt(len(left) * len(right))
