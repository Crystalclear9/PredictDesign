from __future__ import annotations

import torch
from torch import nn


class ColdStartInitializer(nn.Module):
    """Cold-start initialization for empty graphs, inspired by RT's task table prompting.

    When an agent graph has no nodes and no queries yet, this module provides
    meaningful initial embeddings based on:
    1. **Role prototypes**: learnable embeddings per candidate role
    2. **Task embedding**: encodes the task description (via SentenceTransformer)
    3. **Role-Task fusion**: combines role prototype with task embedding
    4. **Structural prior**: generates initial edge probability matrix based on
       role-pair compatibility

    This replaces the zero-vector initialization that causes the model to
    degrade to random guessing in cold-start scenarios.
    """

    def __init__(
        self,
        candidate_roles: tuple[str, ...],
        hidden_dim: int,
        text_encoder: nn.Module | None = None,
        st_dim: int = 384,
    ) -> None:
        super().__init__()
        self.candidate_roles = candidate_roles
        self.hidden_dim = hidden_dim
        self.role_to_index = {role: idx for idx, role in enumerate(candidate_roles)}

        # Learnable prototype per role
        self.role_prototypes = nn.Embedding(len(candidate_roles), hidden_dim)
        nn.init.normal_(self.role_prototypes.weight, mean=0.0, std=0.05)

        # Task description fusion
        task_input_dim = st_dim if text_encoder is not None else hidden_dim
        self.task_fusion = nn.Sequential(
            nn.Linear(task_input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Structural prior: predict initial edge probability between role pairs
        self.structural_prior_head = nn.Bilinear(hidden_dim, hidden_dim, 1)

        # Graph-level cold-start embedding (when no roles are specified either)
        self.fallback_graph_embedding = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.fallback_graph_embedding, mean=0.0, std=0.02)

        self.text_encoder = text_encoder

    def get_role_index(self, role: str) -> int:
        """Return index for a known role, or hash to a valid index for unknown roles."""
        if role in self.role_to_index:
            return self.role_to_index[role]
        # Fallback: hash unknown roles into the existing embedding space
        import hashlib
        digest = hashlib.sha256(role.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % len(self.candidate_roles)

    def initialize_state(
        self,
        role: str,
        device: torch.device | str = "cpu",
        task_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate initial hidden state for a single node given its role.

        Args:
            role: the agent's role name
            device: target device
            task_embedding: optional task description embedding from SentenceTransformer
        Returns:
            [hidden_dim] initial state vector
        """
        device = torch.device(device)
        role_idx = self.get_role_index(role)
        idx_tensor = torch.tensor(role_idx, dtype=torch.long, device=device)
        prototype = self.role_prototypes(idx_tensor)

        if task_embedding is not None:
            task_emb = task_embedding.to(device)
            fused = self.task_fusion(torch.cat([prototype, task_emb], dim=-1))
            return fused
        return prototype

    def initialize_states(
        self,
        roles: list[str],
        device: torch.device | str = "cpu",
        task_description: str | None = None,
    ) -> torch.Tensor:
        """Generate initial hidden states for a set of nodes.

        Args:
            roles: list of role names for each node
            device: target device
            task_description: optional task description text
        Returns:
            [N, hidden_dim] initial state matrix
        """
        device = torch.device(device)
        if not roles:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=device)

        # Encode task description if provided
        task_embedding = None
        if task_description and self.text_encoder is not None:
            task_embedding = self.text_encoder(task_description, device=device)

        states = []
        for role in roles:
            state = self.initialize_state(role, device=device, task_embedding=task_embedding)
            states.append(state)
        return torch.stack(states, dim=0)

    def initial_edge_priors(
        self,
        roles: list[str],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Generate initial edge probability matrix based on role compatibility.

        Args:
            roles: list of role names
            device: target device
        Returns:
            [N, N] edge probability matrix (sigmoid applied)
        """
        device = torch.device(device)
        n = len(roles)
        if n == 0:
            return torch.zeros((0, 0), dtype=torch.float32, device=device)

        indices = torch.tensor(
            [self.get_role_index(r) for r in roles], dtype=torch.long, device=device
        )
        embeddings = self.role_prototypes(indices)  # [N, D]

        # Bilinear scoring for all pairs
        source = embeddings.unsqueeze(1).expand(-1, n, -1)  # [N, N, D]
        target = embeddings.unsqueeze(0).expand(n, -1, -1)  # [N, N, D]
        scores = self.structural_prior_head(
            source.reshape(n * n, -1), target.reshape(n * n, -1)
        ).view(n, n)

        # Zero out self-loops
        diagonal_mask = torch.eye(n, dtype=torch.bool, device=device)
        scores = scores.masked_fill(diagonal_mask, float("-inf"))

        return torch.sigmoid(scores)

    def graph_embedding_cold(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return the fallback graph-level embedding for completely empty graphs."""
        return self.fallback_graph_embedding.to(torch.device(device))
