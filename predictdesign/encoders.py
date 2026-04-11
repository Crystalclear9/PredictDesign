from __future__ import annotations

import hashlib
import re

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


class TextSemanticEncoder(nn.Module):
    def __init__(self, output_dim: int, bucket_count: int = 512) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.bucket_count = bucket_count
        self.extra_feature_dim = 16
        self.projection = nn.Sequential(
            nn.Linear(bucket_count + self.extra_feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.keyword_groups = {
            "coordination": ("plan", "planner", "delegate", "assign", "coordinate", "task"),
            "analysis": ("analyze", "analysis", "reason", "evidence", "infer", "inspect"),
            "discussion": ("discussion", "speak", "speech", "communicate", "talk", "summary"),
            "research": ("research", "paper", "experiment", "baseline", "dataset", "metric"),
            "werewolf": ("werewolf", "wolf", "villager", "witch", "seer", "guard"),
            "decision": ("vote", "banish", "attack", "check", "protect", "save", "poison"),
        }

    def forward(self, text: str, device: torch.device | str) -> torch.Tensor:
        vector = self._vectorize(text, torch.device(device))
        return self.projection(vector)

    def _vectorize(self, text: str, device: torch.device) -> torch.Tensor:
        normalized = str(text or "").strip().lower()
        vector = torch.zeros(
            self.bucket_count + self.extra_feature_dim,
            dtype=torch.float32,
            device=device,
        )
        if not normalized:
            return vector
        tokens = re.findall(r"[a-z0-9_]+", normalized)
        compact = re.sub(r"\s+", " ", normalized)

        def add_feature(feature: str, weight: float) -> None:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.bucket_count
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign * weight

        for token in tokens:
            add_feature(f"tok:{token}", 1.0)
        for left, right in zip(tokens, tokens[1:]):
            add_feature(f"bigram:{left}|{right}", 0.8)
        for idx in range(max(len(compact) - 2, 0)):
            add_feature(f"tri:{compact[idx:idx + 3]}", 0.25)

        extra_offset = self.bucket_count
        for keyword_index, keywords in enumerate(self.keyword_groups.values()):
            vector[extra_offset + keyword_index] = min(
                1.0,
                sum(1 for keyword in keywords if keyword in normalized) / max(len(keywords), 1),
            )
        vector[extra_offset + 6] = min(len(tokens) / 128.0, 1.0)
        vector[extra_offset + 7] = min(len(compact) / 2048.0, 1.0)
        vector[extra_offset + 8] = compact.count("?") / max(len(compact), 1)
        vector[extra_offset + 9] = compact.count(":") / max(len(compact), 1)
        vector[extra_offset + 10] = compact.count("!") / max(len(compact), 1)
        vector[extra_offset + 11] = sum(token.isdigit() for token in tokens) / max(len(tokens), 1)
        vector[extra_offset + 12] = sum(token.isupper() for token in tokens) / max(len(tokens), 1)
        vector[extra_offset + 13] = min(len(set(tokens)) / max(len(tokens), 1), 1.0)
        vector[extra_offset + 14] = min(compact.count("\n") / 16.0, 1.0)
        vector[extra_offset + 15] = 1.0
        norm = vector.norm(p=2)
        if float(norm.item()) > 0:
            vector = vector / norm
        return vector


class NodeFeatureEncoder(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, role_dim: int, role_hash_buckets: int) -> None:
        super().__init__()
        self.role_encoder = RoleEncoder(role_dim=role_dim, bucket_count=role_hash_buckets)
        self.text_encoder = TextSemanticEncoder(output_dim=hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.state_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 7 + role_dim, hidden_dim),
            nn.ReLU(),
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
    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.text_encoder = TextSemanticEncoder(output_dim=hidden_dim)
        self.relation_encoder = RoleEncoder(role_dim=hidden_dim, bucket_count=257)
        self.action_to_index = {action: idx for idx, action in enumerate(MessageAction)}
        self.action_embedding = nn.Embedding(len(MessageAction), hidden_dim)
        self.time_encoder = TimeEncoder(hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.direction_projection = nn.Linear(4, hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 10, hidden_dim),
            nn.ReLU(),
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
