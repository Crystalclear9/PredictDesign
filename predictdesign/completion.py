from __future__ import annotations

import torch
from torch import nn


class NodeCompletionClassifier(nn.Module):
    """Lightweight classifier to detect whether an agent's output is complete.

    A tiny two-layer network that takes a node embedding and outputs a scalar
    probability P(output_complete | node_embedding) ∈ [0, 1].

    Integration with action scoring:
    - If completion_prob > threshold → node's output is done, it becomes
      available as a *source* for new edges (delegation, communication).
    - If completion_prob < threshold → node is still working, its create_edge
      source scores are penalized.

    The classifier is intentionally very small (~500 params for hidden_dim=32)
    to avoid interfering with the main prediction signal.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        bottleneck = max(min(hidden_dim // 2, 16), 4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict completion probability for each node.

        Args:
            node_embeddings: [N, D] or [D] node embedding(s)
        Returns:
            [N] or scalar completion probabilities in [0, 1]
        """
        logits = self.classifier(node_embeddings).squeeze(-1)
        return torch.sigmoid(logits)

    def completion_mask(
        self,
        node_embeddings: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Return boolean mask: True if the node's output is complete.

        Args:
            node_embeddings: [N, D]
            threshold: decision boundary
        Returns:
            [N] boolean tensor
        """
        probs = self.forward(node_embeddings)
        return probs >= threshold
