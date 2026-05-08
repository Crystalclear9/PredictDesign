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


@dataclass(slots=True)
class FewShotTransitionExample:
    source_role: str
    target_role: str
    relation_type: str
    text: str
    tokens: set[str]
    count: int = 1


class FewShotTransitionMemory:
    """Non-parametric transition memory for small-data scenarios."""

    def __init__(self, max_examples: int = 512) -> None:
        self.max_examples = max(1, int(max_examples))
        self.examples: list[FewShotTransitionExample] = []
        self._signature_to_index: dict[tuple[str, str, str, str], int] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def clear(self) -> None:
        self.examples.clear()
        self._signature_to_index.clear()

    def add(
        self,
        *,
        source_role: str,
        target_role: str,
        relation_type: str,
        text: str,
    ) -> None:
        relation = relation_type.strip().lower() or "unknown"
        source = source_role.strip().lower() or "unknown"
        target = target_role.strip().lower() or "unknown"
        normalized_text = self._normalize(text)
        signature = (source, target, relation, normalized_text[:240])
        if signature in self._signature_to_index:
            self.examples[self._signature_to_index[signature]].count += 1
            return
        if len(self.examples) >= self.max_examples:
            self.examples.pop(0)
            self._signature_to_index = {
                (
                    item.source_role,
                    item.target_role,
                    item.relation_type,
                    self._normalize(item.text)[:240],
                ): index
                for index, item in enumerate(self.examples)
            }
        example = FewShotTransitionExample(
            source_role=source,
            target_role=target,
            relation_type=relation,
            text=normalized_text,
            tokens=self._tokens(normalized_text),
        )
        self._signature_to_index[signature] = len(self.examples)
        self.examples.append(example)

    def candidate_score(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        prediction_context: GraphPredictionContext | None,
    ) -> float:
        if not self.examples or action.action_type != GraphActionType.CREATE_EDGE:
            return 0.0
        source_role, target_role = self._action_roles(action, temporal_graph)
        relation = str(action.relation_type or "").strip().lower()
        action_tokens = self._tokens(
            "\n".join(
                part
                for part in (
                    self._context_text(prediction_context),
                    relation,
                    str(action.source_node_id or ""),
                    str(action.target_node_id or ""),
                    str(action.metadata.get("transition_id", "")),
                    str(action.metadata.get("description", "")),
                    self._node_text(action.source_node_id, temporal_graph),
                    self._node_text(action.target_node_id, temporal_graph),
                )
                if str(part).strip()
            )
        )
        scores = [
            self._example_score(
                example=example,
                source_role=source_role,
                target_role=target_role,
                relation=relation,
                tokens=action_tokens,
            )
            for example in self.examples
        ]
        if not scores:
            return 0.0
        top_scores = sorted(scores, reverse=True)[:3]
        return sum(top_scores) / float(len(top_scores))

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
        if not self.examples or not node_order:
            return matrix
        context_tokens = self._tokens(self._context_text(prediction_context))
        for row, source_node_id in enumerate(node_order):
            for col, target_node_id in enumerate(node_order):
                source_role = self._node_role(source_node_id, temporal_graph)
                target_role = self._node_role(target_node_id, temporal_graph)
                pair_tokens = context_tokens.union(
                    self._tokens(
                        f"{source_node_id} {target_node_id} "
                        f"{self._node_text(source_node_id, temporal_graph)} "
                        f"{self._node_text(target_node_id, temporal_graph)}"
                    )
                )
                best = max(
                    self._example_score(
                        example=example,
                        source_role=source_role,
                        target_role=target_role,
                        relation="",
                        tokens=pair_tokens,
                    )
                    for example in self.examples
                )
                if best > 0:
                    matrix[row, col] = float(best)
        return matrix

    def _example_score(
        self,
        *,
        example: FewShotTransitionExample,
        source_role: str,
        target_role: str,
        relation: str,
        tokens: set[str],
    ) -> float:
        score = 0.0
        if source_role and source_role == example.source_role:
            score += 0.45
        if target_role and target_role == example.target_role:
            score += 0.55
        if relation and relation == example.relation_type:
            score += 1.00
        score += 1.50 * self._overlap(tokens, example.tokens)
        if example.count > 1:
            score *= 1.0 + min(math.log1p(example.count), 2.0) * 0.15
        return score

    def _action_roles(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
    ) -> tuple[str, str]:
        return (
            self._node_role(action.source_node_id, temporal_graph),
            self._node_role(action.target_node_id, temporal_graph),
        )

    def _node_role(self, node_id: str | None, temporal_graph: TemporalGraph) -> str:
        if node_id is None or node_id not in temporal_graph.nodes:
            return "unknown"
        return temporal_graph.nodes[node_id].role.strip().lower() or "unknown"

    def _node_text(self, node_id: str | None, temporal_graph: TemporalGraph) -> str:
        if node_id is None or node_id not in temporal_graph.nodes:
            return ""
        node = temporal_graph.nodes[node_id]
        return f"{node.role}\n{node.context_text}"

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

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()

    def _overlap(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / math.sqrt(len(left) * len(right))
