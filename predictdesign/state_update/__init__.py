from __future__ import annotations

from .base import BaseStateUpdater
from .gru import GRUStateUpdater
from .mdp import MDPStateUpdater, MDPTransitionSummary


def build_state_updater(
    updater_type: str,
    context_dim: int,
    hidden_dim: int,
    latent_state_count: int,
    latent_action_count: int = 4,
) -> BaseStateUpdater:
    if updater_type == "gru":
        return GRUStateUpdater(context_dim=context_dim, hidden_dim=hidden_dim)
    if updater_type == "mdp":
        return MDPStateUpdater(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            latent_state_count=latent_state_count,
            latent_action_count=latent_action_count,
        )
    raise ValueError(f"Unsupported updater_type: {updater_type}")


__all__ = [
    "BaseStateUpdater",
    "GRUStateUpdater",
    "MDPStateUpdater",
    "MDPTransitionSummary",
    "build_state_updater",
]
