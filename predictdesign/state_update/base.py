from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseStateUpdater(nn.Module, ABC):
    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(
        self,
        previous_state: torch.Tensor,
        node_context: torch.Tensor,
        aggregated_message: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
