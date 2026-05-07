from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .ctdg import ContinuousTimeDynamicGraph
from .temporal_graph import TemporalGraph


class GraphActionType(str, Enum):
    CREATE_EDGE = "create_edge"
    REMOVE_EDGE = "remove_edge"
    ADD_NODE = "add_node"
    NO_OP = "no_op"


@dataclass(slots=True)
class PredictedGraphAction:
    action_type: GraphActionType
    score: float
    effective_time: float
    source_node_id: str | None = None
    target_node_id: str | None = None
    relation_type: str | None = None
    role: str | None = None
    new_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphPredictionContext:
    """Optional per-step context for cold-start and candidate-aware prediction."""

    source_node_id: str | None = None
    query_text: str = ""
    graph_profile_text: str = ""
    source_output_text: str = ""
    candidate_actions: list[PredictedGraphAction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def combined_text(self) -> str:
        parts = [
            self.graph_profile_text,
            self.query_text,
            self.source_output_text,
        ]
        for action in self.candidate_actions:
            description = str(action.metadata.get("description", "")).strip()
            if description:
                parts.append(description)
        return "\n".join(part for part in parts if part.strip())


@dataclass(slots=True)
class PredictionRollout:
    actions: list[PredictedGraphAction]
    temporal_graph: TemporalGraph
    ctdg: ContinuousTimeDynamicGraph


@dataclass(slots=True)
class PredictionSubgraphRollout:
    actions_by_step: list[list[PredictedGraphAction]]
    temporal_graph: TemporalGraph
    ctdg: ContinuousTimeDynamicGraph
