from __future__ import annotations

import math

import torch
from torch import nn

from .encoders import MessageEncoder
from .messages import Message


def _resolve_attention_heads(hidden_dim: int, requested_heads: int) -> int:
    heads = max(1, min(hidden_dim, requested_heads))
    while hidden_dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


class ConcurrentMessageAggregator(nn.Module):
    def __init__(
        self,
        message_encoder: MessageEncoder,
        reduce: str = "sum",
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if reduce not in {"sum", "mean", "attention"}:
            raise ValueError("reduce must be 'sum', 'mean', or 'attention'.")
        self.message_encoder = message_encoder
        self.reduce = reduce
        self.hidden_dim = int(message_encoder.hidden_dim)

        attention_heads = _resolve_attention_heads(self.hidden_dim, num_heads)
        self.message_self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.node_query_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        self.message_key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.message_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 5 + 3, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

    def forward(
        self,
        node_id: str,
        messages: list[Message],
        node_states: dict[str, torch.Tensor],
        device: torch.device | str,
    ) -> torch.Tensor:
        if not messages:
            sample_state = next(iter(node_states.values()), None)
            if sample_state is None:
                raise ValueError("node_states must contain at least one state tensor.")
            return torch.zeros_like(sample_state)

        device = torch.device(device)
        encoded = torch.stack(
            [
                self.message_encoder.encode_for_node(
                    node_id=node_id,
                    message=message,
                    node_states=node_states,
                    device=device,
                )
                for message in messages
            ],
            dim=0,
        )
        if self.reduce == "mean":
            return encoded.mean(dim=0)
        if self.reduce == "sum":
            return encoded.sum(dim=0)
        return self._attention_reduce(node_id=node_id, encoded=encoded, messages=messages, node_states=node_states)

    def _attention_reduce(
        self,
        node_id: str,
        encoded: torch.Tensor,
        messages: list[Message],
        node_states: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        contextualized, _ = self.message_self_attention(
            encoded.unsqueeze(0),
            encoded.unsqueeze(0),
            encoded.unsqueeze(0),
            need_weights=False,
        )
        contextualized = self.message_dropout(contextualized.squeeze(0))
        node_state = node_states.get(node_id)
        if node_state is None:
            node_state = torch.zeros(self.hidden_dim, dtype=contextualized.dtype, device=contextualized.device)
        else:
            node_state = node_state.to(device=contextualized.device, dtype=contextualized.dtype)
        query = self.node_query_projection(node_state)
        scores = (self.message_key_projection(contextualized) @ query) / math.sqrt(self.hidden_dim)
        weights = torch.softmax(scores, dim=0)
        attended = (weights.unsqueeze(-1) * contextualized).sum(dim=0)
        mean_value = contextualized.mean(dim=0)
        max_value = contextualized.max(dim=0).values
        variance = (
            contextualized.var(dim=0, unbiased=False)
            if contextualized.size(0) > 1
            else torch.zeros_like(mean_value)
        )
        source_fraction = sum(message.source_node_id == node_id for message in messages) / len(messages)
        target_fraction = sum(message.target_node_id == node_id for message in messages) / len(messages)
        stats = torch.tensor(
            [
                math.log1p(len(messages)),
                source_fraction,
                target_fraction,
            ],
            dtype=contextualized.dtype,
            device=contextualized.device,
        )
        return self.output_projection(
            torch.cat(
                [
                    attended,
                    mean_value,
                    max_value,
                    variance,
                    node_state,
                    stats,
                ],
                dim=-1,
            )
        )
