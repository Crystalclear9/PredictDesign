from __future__ import annotations

import torch
from torch import nn

from .encoders import MessageEncoder
from .messages import Message


class ConcurrentMessageAggregator(nn.Module):
    def __init__(self, message_encoder: MessageEncoder, reduce: str = "sum") -> None:
        super().__init__()
        if reduce not in {"sum", "mean"}:
            raise ValueError("reduce must be 'sum' or 'mean'.")
        self.message_encoder = message_encoder
        self.reduce = reduce

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
        return encoded.sum(dim=0)
