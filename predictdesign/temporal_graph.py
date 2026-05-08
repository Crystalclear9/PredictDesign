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

    def __post_init__(self) -> None:
        self.start_time = float(self.start_time)

    def is_active(self, time_value: float) -> bool:
        return self.start_time <= float(time_value)

    def elapsed_time(self, time_value: float) -> float:
        return max(float(time_value) - self.start_time, 0.0)


class TemporalGraph:
    def __init__(self, context_dim: int, device: torch.device | str = "cpu") -> None:
        self.context_dim = context_dim
        self.device = torch.device(device)
        self.nodes: dict[str, TemporalNode] = {}
        self.edges: list[TemporalEdge] = []
        self.structural_edges: set[tuple[str, str]] = set()
        self.structural_edge_metadata: dict[tuple[str, str], list[dict[str, str]]] = {}
        self.graph_context_text: str = ""

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

    def add_structural_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_type: str | None = None,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            raise KeyError("Both structural edge endpoints must exist in the temporal graph.")
        if source_node_id == target_node_id:
            return
        key = (source_node_id, target_node_id)
        self.structural_edges.add(key)
        if relation_type or description or metadata:
            item = {str(key): str(value) for key, value in (metadata or {}).items()}
            if relation_type is not None:
                item["relation_type"] = str(relation_type)
            if description is not None:
                item["description"] = str(description)
            bucket = self.structural_edge_metadata.setdefault(key, [])
            if item not in bucket:
                bucket.append(item)

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
        for index in range(len(self.edges) - 1, -1, -1):
            edge = self.edges[index]
            if (
                edge.source_node_id == source_node_id
                and edge.target_node_id == target_node_id
                and edge.is_active(time_value)
            ):
                self.edges.pop(index)
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

    def structural_adjacency_matrix(
        self,
        node_order: list[str] | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        node_order = node_order or sorted(self.nodes)
        device = torch.device(device or self.device)
        index = {node_id: idx for idx, node_id in enumerate(node_order)}
        adjacency = torch.zeros(
            (len(node_order), len(node_order)),
            dtype=torch.float32,
            device=device,
        )
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
            active_flag = 1.0 if edge.is_active(time_value) else 0.0
            elapsed = edge.elapsed_time(time_value)
            freshness = 1.0 / (1.0 + elapsed) if active_flag > 0.0 else 0.0
            start_scale = torch.log1p(torch.tensor(abs(edge.start_time))).item()
            base_features = torch.tensor(
                [
                    active_flag,
                    torch.log1p(torch.tensor(elapsed)).item(),
                    freshness,
                    start_scale,
                ],
                dtype=torch.float32,
                device=device,
            )
            feature_tensor[row, col, : min(feature_dim, 4)] = base_features[: min(feature_dim, 4)]
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
        cloned.graph_context_text = self.graph_context_text
        for node in self.nodes.values():
            cloned.add_node(node)
        for edge in self.edges:
            cloned.add_edge(
                TemporalEdge(
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    start_time=edge.start_time,
                )
            )
        for source_node_id, target_node_id in self.structural_edges:
            cloned.add_structural_edge(source_node_id, target_node_id)
        cloned.structural_edge_metadata = {
            key: [dict(item) for item in value]
            for key, value in self.structural_edge_metadata.items()
        }
        return cloned
