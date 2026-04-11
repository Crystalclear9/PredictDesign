from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .base import BaseStateUpdater


@dataclass(slots=True)
class MDPTransitionSummary:
    action_probs: torch.Tensor
    state_probs: torch.Tensor
    transition_matrix: torch.Tensor
    state_value: torch.Tensor


class MDPStateUpdater(BaseStateUpdater):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        latent_state_count: int,
        latent_action_count: int = 4,
    ) -> None:
        super().__init__(context_dim=context_dim, hidden_dim=hidden_dim)
        self.latent_action_count = latent_action_count
        self.latent_state_embeddings = nn.Parameter(
            torch.randn(latent_state_count, hidden_dim) * 0.05
        )
        self.latent_action_embeddings = nn.Parameter(
            torch.randn(latent_action_count, hidden_dim) * 0.05
        )
        self.policy_model = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_action_count),
        )
        self.transition_model = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_action_count * latent_state_count),
        )
        self.value_model = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_model = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.last_transition_summary: MDPTransitionSummary | None = None

    def step(
        self,
        previous_state: torch.Tensor,
        node_context: torch.Tensor,
        aggregated_message: torch.Tensor,
    ) -> tuple[torch.Tensor, MDPTransitionSummary]:
        context_embedding = self.context_projection(node_context.to(previous_state.device))
        mdp_input = torch.cat([previous_state, context_embedding, aggregated_message], dim=-1)
        action_logits = self.policy_model(mdp_input)
        action_probs = torch.softmax(action_logits, dim=-1)
        transition_logits = self.transition_model(mdp_input).view(
            self.latent_action_count, -1
        )
        transition_matrix = torch.softmax(transition_logits, dim=-1)
        state_probs = action_probs @ transition_matrix
        candidate_state = state_probs @ self.latent_state_embeddings
        action_context = action_probs @ self.latent_action_embeddings
        state_value = self.value_model(mdp_input).squeeze(-1)
        blended_candidate = 0.5 * (candidate_state + action_context)
        gate = self.gate_model(
            torch.cat([previous_state, blended_candidate, context_embedding, action_context], dim=-1)
        )
        value_scaled = torch.tanh(state_value).unsqueeze(-1)
        next_state = gate * blended_candidate * (1.0 + value_scaled) + (1.0 - gate) * previous_state
        summary = MDPTransitionSummary(
            action_probs=action_probs.detach().clone(),
            state_probs=state_probs.detach().clone(),
            transition_matrix=transition_matrix.detach().clone(),
            state_value=state_value.detach().clone(),
        )
        self.last_transition_summary = summary
        return next_state, summary

    def forward(
        self,
        previous_state: torch.Tensor,
        node_context: torch.Tensor,
        aggregated_message: torch.Tensor,
    ) -> torch.Tensor:
        next_state, _ = self.step(previous_state, node_context, aggregated_message)
        return next_state
