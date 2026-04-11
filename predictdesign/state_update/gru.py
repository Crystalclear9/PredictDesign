from __future__ import annotations

import torch
from torch import nn

from .base import BaseStateUpdater


class GRUStateUpdater(BaseStateUpdater):
    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__(context_dim=context_dim, hidden_dim=hidden_dim)
        self.gru_cell = nn.GRUCell(input_size=hidden_dim * 2, hidden_size=hidden_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(
        self,
        previous_state: torch.Tensor,
        node_context: torch.Tensor,
        aggregated_message: torch.Tensor,
    ) -> torch.Tensor:
        context_embedding = self.context_projection(node_context.to(previous_state.device))
        gru_input = self.input_projection(torch.cat([context_embedding, aggregated_message], dim=-1))
        return self.gru_cell(gru_input, previous_state)
