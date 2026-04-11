from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
