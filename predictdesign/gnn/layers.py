from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm (LLaMA-style), matching RT paper architecture choice."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


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


class GatedMLP(nn.Module):
    """LLaMA-style gated MLP with SiLU activation, as used in RT paper."""

    def __init__(self, hidden_dim: int, intermediate_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        intermediate_dim = intermediate_dim or hidden_dim * 4
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)))


class RelationalAttentionLayer(nn.Module):
    """Relational Attention adapted from RT (ICLR 2026) for agent temporal graphs.

    Implements 4 attention sub-layers per block, inspired by RT's design:
    1. **Role Attention** (← RT Column Attention):
       Agents with the same role attend to each other.
    2. **Neighbor Attention** (← RT Neighbor Attention):
       Attention along temporal edges (adjacency-based), with edge bias from
       temporal edge features.
    3. **Full Attention**:
       Unrestricted bidirectional attention across all agents, providing
       full expressive power.
    4. **Feature Mixing** (← RT Feature Attention):
       Self-attention for each agent's own features — implemented as
       identity + gated residual since each agent is a single token.

    Uses RMSNorm (pre-norm) and Gated MLP with SiLU, matching LLaMA/RT arch.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # --- Role Attention ---
        self.role_norm = RMSNorm(hidden_dim)
        self.role_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.role_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.role_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.role_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # --- Neighbor Attention (with edge bias) ---
        self.neighbor_norm = RMSNorm(hidden_dim)
        self.neighbor_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbor_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbor_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbor_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Edge bias: project edge features to per-head bias
        self.edge_bias_proj = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )

        # --- Full Attention ---
        self.full_norm = RMSNorm(hidden_dim)
        self.full_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.full_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.full_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.full_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # --- Feature mixing (self-gate per node) ---
        self.feature_norm = RMSNorm(hidden_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # --- FFN ---
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = GatedMLP(hidden_dim, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def _multihead_attention(
        self,
        x: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        mask: torch.Tensor | None = None,
        edge_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard multi-head attention with optional mask and edge bias.

        Args:
            x: [N, D]
            mask: [N, N] boolean, True = can attend
            edge_bias: [N, N, num_heads] additive bias to attention scores
        Returns:
            [N, D]
        """
        n = x.size(0)
        q = q_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, Dk]
        k = k_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        v = v_proj(x).view(n, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [H, N, N]

        if edge_bias is not None:
            # edge_bias: [N, N, H] -> [H, N, N]
            scores = scores + edge_bias.permute(2, 0, 1)

        if mask is not None:
            # mask: [N, N] bool -> [1, N, N]
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        # Handle all-inf rows (isolated nodes)
        attn = attn.nan_to_num(nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [H, N, Dk]
        out = out.transpose(0, 1).contiguous().view(n, self.hidden_dim)  # [N, D]
        return out_proj(out)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        role_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [N, D] node feature matrix
            adjacency: [N, N] adjacency matrix (temporal + structural)
            edge_features: [N, N, E] temporal edge feature tensor
            role_indices: [N] integer tensor, nodes with same index have same role.
                         If None, role attention falls back to full attention.
        Returns:
            [N, D] updated node features
        """
        n = features.size(0)
        if n == 0:
            return features
        device = features.device

        # 1. Role Attention mask: same-role nodes can attend to each other
        if role_indices is not None:
            role_mask = role_indices.unsqueeze(0) == role_indices.unsqueeze(1)  # [N, N]
        else:
            role_mask = torch.ones(n, n, dtype=torch.bool, device=device)

        x = features
        # --- Role Attention ---
        residual = x
        x_normed = self.role_norm(x)
        x = residual + self.dropout(
            self._multihead_attention(x_normed, self.role_q, self.role_k, self.role_v, self.role_out, mask=role_mask)
        )

        # --- Neighbor Attention ---
        residual = x
        x_normed = self.neighbor_norm(x)
        # Neighbor mask: adjacency + self-loops
        neighbor_mask = (adjacency > 0) | torch.eye(n, dtype=torch.bool, device=device)
        edge_bias = self.edge_bias_proj(edge_features)  # [N, N, num_heads]
        x = residual + self.dropout(
            self._multihead_attention(
                x_normed, self.neighbor_q, self.neighbor_k, self.neighbor_v, self.neighbor_out,
                mask=neighbor_mask, edge_bias=edge_bias,
            )
        )

        # --- Full Attention ---
        residual = x
        x_normed = self.full_norm(x)
        x = residual + self.dropout(
            self._multihead_attention(x_normed, self.full_q, self.full_k, self.full_v, self.full_out)
        )

        # --- Feature Mixing (per-node self-gate) ---
        residual = x
        x_normed = self.feature_norm(x)
        transformed = self.feature_transform(x_normed)
        gate = self.feature_gate(torch.cat([x_normed, transformed], dim=-1))
        x = residual + self.dropout(gate * transformed)

        # --- FFN ---
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))

        return x


class GNNBackbone(nn.Module):
    def __init__(
        self,
        layer_type: str,
        hidden_dim: int,
        num_layers: int,
        edge_feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
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
            elif layer_type == "relational_transformer":
                layers.append(
                    RelationalAttentionLayer(
                        hidden_dim=hidden_dim,
                        edge_feature_dim=edge_feature_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                )
            else:
                raise ValueError(f"Unsupported layer_type: {layer_type}")
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        role_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = features
        for layer in self.layers:
            if self.layer_type == "relational_transformer":
                hidden = layer(hidden, adjacency, edge_features, role_indices=role_indices)
            else:
                hidden = layer(hidden, adjacency, edge_features)
        return hidden
