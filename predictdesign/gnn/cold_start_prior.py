from __future__ import annotations

import math
import re
from dataclasses import dataclass

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
    done_to_delegate_bonus: float = 0.30
    retry_keyword_bonus: float = 0.65
    retry_when_done_penalty: float = -0.50

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

        source_output_tokens = self._tokens(prediction_context.source_output_text if prediction_context else "")
        if source_output_tokens.intersection(_DONE_TOKENS):
            if relation in {"review", "delegate", "delegate_return", "activate"} and action.source_node_id != action.target_node_id:
                score += self.done_to_delegate_bonus
            if relation == "retry":
                score += self.retry_when_done_penalty
        if relation == "retry" and context_tokens.intersection(_RETRY_TOKENS):
            score += self.retry_keyword_bonus

        return score

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
