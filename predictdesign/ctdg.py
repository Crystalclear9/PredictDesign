from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
import copy

import torch

from .aggregation import ConcurrentMessageAggregator
from .messages import Message
from .state_update.base import BaseStateUpdater
from .temporal_graph import TemporalGraph
from .types import clone_tensor_dict


@dataclass(slots=True)
class StateRecord:
    time: float
    state: torch.Tensor


class ContinuousTimeDynamicGraph:
    def __init__(
        self,
        temporal_graph: TemporalGraph,
        message_aggregator: ConcurrentMessageAggregator,
        state_updater: BaseStateUpdater,
        hidden_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.temporal_graph = temporal_graph
        self.message_aggregator = message_aggregator
        self.state_updater = state_updater
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.current_states: dict[str, torch.Tensor] = {}
        self.state_history: dict[str, list[StateRecord]] = {}
        self.message_history: list[Message] = []
        self.synchronize_nodes()

    def synchronize_nodes(self) -> None:
        for node_id in self.temporal_graph.nodes:
            if node_id not in self.current_states:
                self.current_states[node_id] = torch.zeros(
                    self.hidden_dim,
                    dtype=torch.float32,
                    device=self.device,
                )
            if node_id not in self.state_history:
                self.state_history[node_id] = []

    def add_node(self, node_id: str) -> None:
        if node_id not in self.current_states:
            self.current_states[node_id] = torch.zeros(
                self.hidden_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self.state_history[node_id] = []

    def get_state(self, node_id: str) -> torch.Tensor:
        self.synchronize_nodes()
        return self.current_states[node_id]

    def ingest_messages(self, messages: list[Message]) -> None:
        if not messages:
            return
        self.synchronize_nodes()
        messages = sorted(messages, key=lambda item: item.time)
        self.message_history.extend(copy.deepcopy(messages))
        for timestamp, grouped in groupby(messages, key=lambda item: item.time):
            concurrent_messages = list(grouped)
            touched_nodes = sorted(
                {
                    node_id
                    for message in concurrent_messages
                    for node_id in (message.source_node_id, message.target_node_id)
                    if node_id is not None
                }
            )
            previous_states = clone_tensor_dict(self.current_states)
            updates: dict[str, torch.Tensor] = {}
            for node_id in touched_nodes:
                node_messages = [msg for msg in concurrent_messages if msg.touches_node(node_id)]
                if not node_messages:
                    continue
                aggregated_message = self.message_aggregator(
                    node_id=node_id,
                    messages=node_messages,
                    node_states=previous_states,
                    device=self.device,
                )
                node_context = self.temporal_graph.nodes[node_id].context.to(self.device)
                updates[node_id] = self.state_updater(
                    previous_state=previous_states[node_id],
                    node_context=node_context,
                    aggregated_message=aggregated_message,
                )
            for node_id, next_state in updates.items():
                self.current_states[node_id] = next_state
                self.state_history[node_id].append(
                    StateRecord(time=float(timestamp), state=next_state.detach().clone())
                )

    def clone_with_graph(self, temporal_graph: TemporalGraph) -> "ContinuousTimeDynamicGraph":
        cloned = ContinuousTimeDynamicGraph(
            temporal_graph=temporal_graph,
            message_aggregator=self.message_aggregator,
            state_updater=self.state_updater,
            hidden_dim=self.hidden_dim,
            device=self.device,
        )
        cloned.current_states = clone_tensor_dict(self.current_states)
        cloned.state_history = {
            node_id: [StateRecord(time=item.time, state=item.state.detach().clone()) for item in history]
            for node_id, history in self.state_history.items()
        }
        cloned.message_history = copy.deepcopy(self.message_history)
        cloned.synchronize_nodes()
        return cloned
