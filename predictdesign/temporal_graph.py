from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import TensorLike, ensure_tensor


@dataclass(slots=True)
class TemporalNode:
    node_id: str
    role: str
    context: torch.Tensor
    context_text: str = ""

    @classmethod
    def build(
        cls,
        node_id: str,
        role: str,
        context: TensorLike,
        context_dim: int,
        device: torch.device | str,
    ) -> "TemporalNode":
        return cls(
            node_id=node_id,
            role=role,
            context=ensure_tensor(context, context_dim, device),
            context_text="",
        )


@dataclass(slots=True)
class TemporalEdge:
    source_node_id: str
    target_node_id: str
    start_time: float
    end_time: float

    def is_active(self, time_value: float) -> bool:
        return self.start_time <= time_value <= self.end_time


class TemporalGraph:
    def __init__(self, context_dim: int, device: torch.device | str = "cpu") -> None:
        self.context_dim = context_dim
        self.device = torch.device(device)
        self.nodes: dict[str, TemporalNode] = {}
        self.edges: list[TemporalEdge] = []
        self.structural_edges: set[tuple[str, str]] = set()

    def add_node(self, node: TemporalNode) -> None:
        self.nodes[node.node_id] = TemporalNode(
            node_id=node.node_id,
            role=node.role,
            context=ensure_tensor(node.context, self.context_dim, self.device),
            context_text=str(getattr(node, "context_text", "") or ""),
        )

    def add_node_from_parts(
        self,
        node_id: str,
        role: str,
        context: TensorLike,
        context_text: str = "",
    ) -> None:
        node = TemporalNode.build(node_id, role, context, self.context_dim, self.device)
        node.context_text = str(context_text or "")
        self.add_node(node)

    def add_edge(self, edge: TemporalEdge) -> None:
        if edge.source_node_id not in self.nodes or edge.target_node_id not in self.nodes:
            raise KeyError("Both edge endpoints must exist in the temporal graph.")
        self.edges.append(edge)

    def add_structural_edge(self, source_node_id: str, target_node_id: str) -> None:
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            raise KeyError("Both structural edge endpoints must exist in the temporal graph.")
        if source_node_id == target_node_id:
            return
        self.structural_edges.add((source_node_id, target_node_id))

    def has_structural_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return (source_node_id, target_node_id) in self.structural_edges

    def update_node_context(
        self,
        node_id: str,
        context: TensorLike,
        context_text: str | None = None,
    ) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node_id: {node_id}")
        self.nodes[node_id].context = ensure_tensor(context, self.context_dim, self.device)
        if context_text is not None:
            self.nodes[node_id].context_text = str(context_text)

    def active_edges(self, time_value: float) -> list[TemporalEdge]:
        return [edge for edge in self.edges if edge.is_active(time_value)]

    def has_active_edge(self, source_node_id: str, target_node_id: str, time_value: float) -> bool:
        return any(
            edge.source_node_id == source_node_id
            and edge.target_node_id == target_node_id
            and edge.is_active(time_value)
            for edge in self.edges
        )

    def deactivate_edge(self, source_node_id: str, target_node_id: str, time_value: float) -> bool:
        for edge in reversed(self.edges):
            if (
                edge.source_node_id == source_node_id
                and edge.target_node_id == target_node_id
                and edge.is_active(time_value)
            ):
                edge.end_time = min(edge.end_time, time_value)
                return True
        return False

    def adjacency_matrix(
        self,
        time_value: float,
        node_order: list[str] | None = None,
        device: torch.device | str | None = None,
        include_structural: bool = True,
    ) -> torch.Tensor:
        node_order = node_order or sorted(self.nodes)
        device = torch.device(device or self.device)
        index = {node_id: idx for idx, node_id in enumerate(node_order)}
        adjacency = torch.zeros(
            (len(node_order), len(node_order)),
            dtype=torch.float32,
            device=device,
        )
        for edge in self.active_edges(time_value):
            if edge.source_node_id in index and edge.target_node_id in index:
                adjacency[index[edge.source_node_id], index[edge.target_node_id]] = 1.0
        if include_structural:
            for source_node_id, target_node_id in self.structural_edges:
                if source_node_id in index and target_node_id in index:
                    adjacency[index[source_node_id], index[target_node_id]] = 1.0
        return adjacency

    def temporal_edge_features(
        self,
        time_value: float,
        node_order: list[str] | None = None,
        device: torch.device | str | None = None,
        feature_dim: int = 4,
    ) -> torch.Tensor:
        node_order = node_order or sorted(self.nodes)
        device = torch.device(device or self.device)
        feature_tensor = torch.zeros(
            (len(node_order), len(node_order), feature_dim),
            dtype=torch.float32,
            device=device,
        )
        index = {node_id: idx for idx, node_id in enumerate(node_order)}
        for edge in self.edges:
            if edge.source_node_id not in index or edge.target_node_id not in index:
                continue
            row = index[edge.source_node_id]
            col = index[edge.target_node_id]
            duration = max(edge.end_time - edge.start_time, 0.0)
            elapsed = max(time_value - edge.start_time, 0.0)
            remaining = max(edge.end_time - time_value, 0.0)
            base_features = torch.tensor(
                [
                    1.0 if edge.is_active(time_value) else 0.0,
                    torch.log1p(torch.tensor(duration)).item(),
                    torch.log1p(torch.tensor(elapsed)).item(),
                    torch.log1p(torch.tensor(remaining)).item(),
                ],
                dtype=torch.float32,
                device=device,
            )
            feature_tensor[row, col, :4] = base_features
        if feature_dim >= 5:
            for source_node_id, target_node_id in self.structural_edges:
                if source_node_id not in index or target_node_id not in index:
                    continue
                row = index[source_node_id]
                col = index[target_node_id]
                feature_tensor[row, col, 4] = 1.0
        return feature_tensor

    def generate_node_id(self, role: str) -> str:
        normalized = role.strip().lower().replace(" ", "_") or "node"
        suffix = 1
        candidate = f"{normalized}_{suffix}"
        while candidate in self.nodes:
            suffix += 1
            candidate = f"{normalized}_{suffix}"
        return candidate

    def clone(self) -> "TemporalGraph":
        cloned = TemporalGraph(context_dim=self.context_dim, device=self.device)
        for node in self.nodes.values():
            cloned.add_node(node)
        for edge in self.edges:
            cloned.add_edge(
                TemporalEdge(
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    start_time=edge.start_time,
                    end_time=edge.end_time,
                )
            )
        for source_node_id, target_node_id in self.structural_edges:
            cloned.add_structural_edge(source_node_id, target_node_id)
        return cloned
