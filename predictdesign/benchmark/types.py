from __future__ import annotations

from dataclasses import dataclass, field

from ..messages import Message
from ..prediction import PredictedGraphAction
from ..temporal_graph import TemporalEdge, TemporalNode


@dataclass(slots=True)
class EpisodeStep:
    observation_time: float
    messages: list[Message]
    ground_truth_action: PredictedGraphAction
    observed_actions: list[PredictedGraphAction] = field(default_factory=list)
    valid_next_actions: list[PredictedGraphAction] = field(default_factory=list)
    context_updates: dict[str, list[float]] = field(default_factory=dict)
    context_text_updates: dict[str, str] = field(default_factory=dict)

    @property
    def supervision_actions(self) -> list[PredictedGraphAction]:
        return self.valid_next_actions or [self.ground_truth_action]


@dataclass(slots=True)
class BenchmarkEpisode:
    episode_id: str
    dataset_name: str
    initial_nodes: list[TemporalNode]
    initial_edges: list[TemporalEdge]
    steps: list[EpisodeStep]
    initial_structural_edges: list[tuple[str, str]] = field(default_factory=list)
