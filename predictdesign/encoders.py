from __future__ import annotations

import hashlib

import torch
from torch import nn

from .messages import Message, MessageAction
from .temporal_graph import TemporalNode
from .types import ensure_tensor


def stable_hash_index(value: str, bucket_count: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % bucket_count


class RoleEncoder(nn.Module):
    def __init__(self, role_dim: int, bucket_count: int) -> None:
        super().__init__()
        self.bucket_count = bucket_count
        self.embedding = nn.Embedding(bucket_count, role_dim)

    def forward(self, role: str, device: torch.device | str) -> torch.Tensor:
        index = stable_hash_index(role, self.bucket_count)
        token = torch.tensor(index, dtype=torch.long, device=device)
        return self.embedding(token)


class TimeEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.raw_dim = 9
        self.projection = nn.Linear(self.raw_dim, output_dim)
        frequencies = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time_value: float, device: torch.device | str) -> torch.Tensor:
        time_tensor = torch.tensor(float(time_value), dtype=torch.float32, device=device)
        angles = time_tensor * self.frequencies.to(device)
        raw = torch.cat(
            [
                time_tensor.view(1),
                torch.sin(angles),
                torch.cos(angles),
            ]
        )
        return self.projection(raw)


class SentenceTransformerEncoder(nn.Module):
    """Encode text using a frozen SentenceTransformer (MiniLMv2 by default, matching RT paper).

    ``model_name_or_path`` accepts either a HuggingFace model name (will be
    downloaded on first use) **or** a local absolute path to pre-downloaded
    weights so the system can work fully offline.
    """

    def __init__(
        self,
        output_dim: int,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        st_dim: int = 384,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.st_dim = st_dim
        self._model_path = model_name_or_path
        self._freeze = freeze
        self._st_model: object | None = None  # lazy-loaded

        # Projection from ST embedding dim to hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(st_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

        # Lightweight fallback for environments without sentence-transformers
        self._fallback_hash_dim = st_dim
        self._use_fallback = False

    def _ensure_model(self) -> None:
        if self._st_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self._model_path)
            if self._freeze:
                for param in self._st_model.parameters():
                    param.requires_grad = False
            self._use_fallback = False
        except (ImportError, OSError):
            self._use_fallback = True

    def _fallback_encode(self, text: str, device: torch.device) -> torch.Tensor:
        """Hash-based fallback when sentence-transformers is unavailable."""
        vector = torch.zeros(self._fallback_hash_dim, dtype=torch.float32, device=device)
        if not text or not text.strip():
            return vector
        normalized = text.strip().lower()
        tokens = normalized.split()
        for i, token in enumerate(tokens[:128]):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:8], "big") % self._fallback_hash_dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[idx] += sign * 1.0
        norm = vector.norm(p=2)
        if float(norm.item()) > 0:
            vector = vector / norm
        return vector

    def forward(self, text: str, device: torch.device | str) -> torch.Tensor:
        device = torch.device(device)
        self._ensure_model()

        if self._use_fallback:
            raw_embedding = self._fallback_encode(text, device)
        else:
            # SentenceTransformer.encode() internally uses torch.inference_mode(),
            # so we must .clone() to break out of that context before passing
            # through our trainable projection layer.
            embedding = self._st_model.encode(
                text or "",
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            raw_embedding = embedding.to(device=device, dtype=torch.float32).clone()

        return self.projection(raw_embedding)


class NodeFeatureEncoder(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        role_dim: int,
        role_hash_buckets: int,
        sentence_transformer_path: str = "all-MiniLM-L6-v2",
        sentence_transformer_dim: int = 384,
        sentence_transformer_freeze: bool = True,
    ) -> None:
        super().__init__()
        self.role_encoder = RoleEncoder(role_dim=role_dim, bucket_count=role_hash_buckets)
        self.text_encoder = SentenceTransformerEncoder(
            output_dim=hidden_dim,
            model_name_or_path=sentence_transformer_path,
            st_dim=sentence_transformer_dim,
            freeze=sentence_transformer_freeze,
        )
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.state_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 7 + role_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node: TemporalNode, history_state: torch.Tensor) -> torch.Tensor:
        device = history_state.device
        role_embedding = self.role_encoder(node.role, device=device)
        text_embedding = self.text_encoder(node.context_text, device=device)
        context_embedding = self.context_projection(node.context.to(device))
        state_embedding = self.state_projection(history_state)
        context_state_interaction = context_embedding * state_embedding
        context_state_delta = torch.abs(context_embedding - state_embedding)
        text_state_interaction = text_embedding * state_embedding
        text_context_interaction = text_embedding * context_embedding
        return self.output_projection(
            torch.cat(
                [
                    role_embedding,
                    text_embedding,
                    context_embedding,
                    state_embedding,
                    context_state_interaction,
                    context_state_delta,
                    text_state_interaction,
                    text_context_interaction,
                ],
                dim=-1,
            )
        )


class MessageEncoder(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        sentence_transformer_path: str = "all-MiniLM-L6-v2",
        sentence_transformer_dim: int = 384,
        sentence_transformer_freeze: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.text_encoder = SentenceTransformerEncoder(
            output_dim=hidden_dim,
            model_name_or_path=sentence_transformer_path,
            st_dim=sentence_transformer_dim,
            freeze=sentence_transformer_freeze,
        )
        self.relation_encoder = RoleEncoder(role_dim=hidden_dim, bucket_count=257)
        self.action_to_index = {action: idx for idx, action in enumerate(MessageAction)}
        self.action_embedding = nn.Embedding(len(MessageAction), hidden_dim)
        self.time_encoder = TimeEncoder(hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.direction_projection = nn.Linear(4, hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 10, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode_for_node(
        self,
        node_id: str,
        message: Message,
        node_states: dict[str, torch.Tensor],
        device: torch.device | str,
    ) -> torch.Tensor:
        device = torch.device(device)
        source_state = self._resolve_state(
            explicit_state=message.source_state,
            fallback_node_id=message.source_node_id,
            fallback_states=node_states,
            weight=message.source_weight,
            device=device,
        )
        target_state = self._resolve_state(
            explicit_state=message.target_state,
            fallback_node_id=message.target_node_id,
            fallback_states=node_states,
            weight=message.target_weight,
            device=device,
        )
        action_embedding = self.action_embedding(
            torch.tensor(self.action_to_index[message.action], dtype=torch.long, device=device)
        )
        time_embedding = self.time_encoder(message.time, device=device)
        context_embedding = self.context_projection(
            ensure_tensor(message.context, self.context_dim, device)
        )
        raw_text_embedding = self.text_encoder(
            str(message.metadata.get("raw_text", "")) if message.metadata else "",
            device=device,
        )
        relation_embedding = self.relation_encoder(
            str(message.metadata.get("relation_type", "")) if message.metadata else "",
            device=device,
        )
        direction_flags = torch.tensor(
            [
                1.0 if message.source_node_id == node_id else 0.0,
                1.0 if message.target_node_id == node_id else 0.0,
                1.0 if message.source_node_id is None else 0.0,
                1.0 if message.target_node_id is None else 0.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        direction_embedding = self.direction_projection(direction_flags)
        state_delta = source_state - target_state
        state_interaction = source_state * target_state
        return self.output_projection(
            torch.cat(
                [
                    source_state,
                    target_state,
                    state_delta,
                    state_interaction,
                    action_embedding,
                    time_embedding,
                    context_embedding,
                    raw_text_embedding,
                    relation_embedding,
                    direction_embedding,
                ],
                dim=-1,
            )
        )

    def _resolve_state(
        self,
        explicit_state: torch.Tensor | None,
        fallback_node_id: str | None,
        fallback_states: dict[str, torch.Tensor],
        weight: float,
        device: torch.device,
    ) -> torch.Tensor:
        if explicit_state is not None:
            state = explicit_state.to(device=device, dtype=torch.float32)
        elif fallback_node_id is not None and fallback_node_id in fallback_states:
            state = fallback_states[fallback_node_id].to(device=device, dtype=torch.float32)
        else:
            state = torch.zeros(self.hidden_dim, dtype=torch.float32, device=device)
        return state * float(weight)
