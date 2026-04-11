from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from .types import TensorLike, ensure_tensor


class MessageAction(str, Enum):
    QUERY_ARRIVAL = "query_arrival"
    NODE_COMPLETION = "node_completion"


@dataclass(slots=True)
class Message:
    source_node_id: str | None
    target_node_id: str | None
    time: float
    action: MessageAction
    source_state: torch.Tensor | None = None
    target_state: torch.Tensor | None = None
    context: torch.Tensor | None = None
    source_weight: float = 1.0
    target_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touches_node(self, node_id: str) -> bool:
        return self.source_node_id == node_id or self.target_node_id == node_id

    @classmethod
    def build_query_message(
        cls,
        target_node_id: str,
        time: float,
        context: TensorLike,
        context_dim: int,
        device: torch.device | str,
    ) -> "Message":
        return cls(
            source_node_id=None,
            target_node_id=target_node_id,
            time=time,
            action=MessageAction.QUERY_ARRIVAL,
            context=ensure_tensor(context, context_dim, device),
        )

    @classmethod
    def build_completion_message(
        cls,
        time: float,
        source_node_id: str | None = None,
        target_node_id: str | None = None,
        source_state: TensorLike = None,
        target_state: TensorLike = None,
        context: TensorLike = None,
        hidden_dim: int = 32,
        context_dim: int = 16,
        device: torch.device | str = "cpu",
    ) -> "Message":
        return cls(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            time=time,
            action=MessageAction.NODE_COMPLETION,
            source_state=ensure_tensor(source_state, hidden_dim, device)
            if source_state is not None
            else None,
            target_state=ensure_tensor(target_state, hidden_dim, device)
            if target_state is not None
            else None,
            context=ensure_tensor(context, context_dim, device) if context is not None else None,
        )
