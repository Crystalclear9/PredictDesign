from __future__ import annotations

import torch
from torch import nn


class TemporalEdgeEncoder(nn.Module):
    def __init__(self, edge_feature_dim: int) -> None:
        super().__init__()
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_feature_dim),
            nn.ReLU(),
            nn.Linear(edge_feature_dim, 1),
        )
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_feature_dim),
            nn.ReLU(),
            nn.Linear(edge_feature_dim, 1),
        )

    def edge_weight_matrix(self, edge_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.edge_gate(edge_features)).squeeze(-1)

    def edge_bias_matrix(self, edge_features: torch.Tensor) -> torch.Tensor:
        return self.edge_bias(edge_features).squeeze(-1)


class DenseGCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.edge_encoder = TemporalEdgeEncoder(edge_feature_dim=edge_feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        identity = torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
        edge_weights = self.edge_encoder.edge_weight_matrix(edge_features)
        adjacency_hat = adjacency * (1.0 + edge_weights) + identity
        degree = adjacency_hat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized = adjacency_hat / degree
        return torch.relu(self.linear(normalized @ features))


class DenseGraphSAGELayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)
        self.edge_encoder = TemporalEdgeEncoder(edge_feature_dim=edge_feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        identity = torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
        edge_weights = self.edge_encoder.edge_weight_matrix(edge_features)
        adjacency_hat = adjacency * (1.0 + edge_weights) + identity
        degree = adjacency_hat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_mean = adjacency_hat @ features / degree
        return torch.relu(self.linear(torch.cat([features, neighbor_mean], dim=-1)))


class DenseGATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, edge_feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_source = nn.Parameter(torch.empty(output_dim))
        self.attn_target = nn.Parameter(torch.empty(output_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.edge_encoder = TemporalEdgeEncoder(edge_feature_dim=edge_feature_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.normal_(self.attn_source, mean=0.0, std=0.02)
        nn.init.normal_(self.attn_target, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.linear(features)
        source_scores = (projected * self.attn_source).sum(dim=-1, keepdim=True)
        target_scores = (projected * self.attn_target).sum(dim=-1, keepdim=True).transpose(0, 1)
        edge_bias = self.edge_encoder.edge_bias_matrix(edge_features)
        attention_scores = self.leaky_relu(source_scores + target_scores + edge_bias)
        mask = adjacency > 0
        mask = mask | torch.eye(adjacency.size(0), device=adjacency.device, dtype=torch.bool)
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(attention_scores, dim=-1)
        return torch.relu(attention @ projected)


class GNNBackbone(nn.Module):
    def __init__(
        self,
        layer_type: str,
        hidden_dim: int,
        num_layers: int,
        edge_feature_dim: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            if layer_type == "gcn":
                layers.append(DenseGCNLayer(hidden_dim, hidden_dim, edge_feature_dim=edge_feature_dim))
            elif layer_type == "graphsage":
                layers.append(
                    DenseGraphSAGELayer(hidden_dim, hidden_dim, edge_feature_dim=edge_feature_dim)
                )
            elif layer_type == "gat":
                layers.append(DenseGATLayer(hidden_dim, hidden_dim, edge_feature_dim=edge_feature_dim))
            else:
                raise ValueError(f"Unsupported layer_type: {layer_type}")
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = features
        for layer in self.layers:
            hidden = layer(hidden, adjacency, edge_features)
        return hidden
