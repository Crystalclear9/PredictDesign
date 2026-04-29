from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re

from .messages import Message
from .temporal_graph import TemporalNode


ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "planner": ("planner", "plan", "planning", "规划", "计划", "协调", "coordinator"),
    "solver": ("solver", "executor", "coder", "developer", "执行", "编码", "开发", "求解"),
    "critic": ("critic", "reviewer", "judge", "评审", "审查", "评价", "质检"),
    "tool": ("tool", "searcher", "retriever", "工具", "检索", "搜索"),
    "researcher": ("researcher", "analyst", "研究", "分析", "调研"),
    "wolf": ("wolf", "werewolf", "狼人"),
    "villager": ("villager", "村民"),
    "seer": ("seer", "预言家"),
    "witch": ("witch", "女巫"),
    "guard": ("guard", "守卫"),
}


@dataclass(slots=True)
class QueryParseResult:
    nodes: list[TemporalNode]
    inferred_roles: list[str] = field(default_factory=list)
    unmatched_text: str = ""


class QueryParser:
    def __init__(
        self,
        context_dim: int,
        device: str = "cpu",
        default_role: str = "planner",
    ) -> None:
        self.context_dim = context_dim
        self.device = device
        self.default_role = default_role

    def parse(self, query_text: str) -> QueryParseResult:
        query_text = (query_text or "").strip()
        if not query_text:
            return QueryParseResult(nodes=[], unmatched_text="")

        explicit_nodes = self._parse_explicit_nodes(query_text)
        if explicit_nodes:
            return QueryParseResult(
                nodes=explicit_nodes,
                inferred_roles=[node.role for node in explicit_nodes],
                unmatched_text=query_text,
            )

        inferred_nodes = self._parse_role_mentions(query_text)
        if inferred_nodes:
            return QueryParseResult(
                nodes=inferred_nodes,
                inferred_roles=[node.role for node in inferred_nodes],
                unmatched_text=query_text,
            )

        fallback_node = TemporalNode.build(
            node_id=f"{self.default_role}_1",
            role=self.default_role,
            context=self._text_to_context(query_text),
            context_dim=self.context_dim,
            device=self.device,
        )
        return QueryParseResult(
            nodes=[fallback_node],
            inferred_roles=[self.default_role],
            unmatched_text=query_text,
        )

    def build_query_messages(
        self,
        query_text: str,
        target_node_ids: list[str],
        time_value: float = 0.0,
    ) -> list[Message]:
        context = self._text_to_context(query_text)
        messages: list[Message] = []
        for node_id in target_node_ids:
            message = Message.build_query_message(
                target_node_id=node_id,
                time=time_value,
                context=context,
                context_dim=self.context_dim,
                device=self.device,
            )
            message.metadata["raw_text"] = query_text
            message.metadata["query_text"] = query_text
            messages.append(message)
        return messages

    def _parse_explicit_nodes(self, query_text: str) -> list[TemporalNode]:
        patterns = (
            r"(?P<name>[A-Za-z][\w\-]*)\s*[:：]\s*(?P<role>[A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff ]{0,32})",
            r"(?P<name>[A-Za-z][\w\-]*)\s*\(\s*(?P<role>[A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff ]{0,32})\s*\)",
        )
        nodes: list[TemporalNode] = []
        seen_ids: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, query_text):
                name = match.group("name").strip()
                role_text = match.group("role").strip()
                canonical_role = self._canonical_role(role_text)
                if name in seen_ids:
                    continue
                seen_ids.add(name)
                context_window = self._match_context_window(query_text, match.start(), match.end())
                nodes.append(
                    TemporalNode.build(
                        node_id=name,
                        role=canonical_role,
                        context=self._text_to_context(context_window),
                        context_dim=self.context_dim,
                        device=self.device,
                    )
                )
        return nodes

    def _parse_role_mentions(self, query_text: str) -> list[TemporalNode]:
        nodes: list[TemporalNode] = []
        seen_roles: set[str] = set()
        lowered = query_text.lower()
        for canonical_role, aliases in ROLE_ALIASES.items():
            if any(alias.lower() in lowered for alias in aliases):
                if canonical_role in seen_roles:
                    continue
                seen_roles.add(canonical_role)
                nodes.append(
                    TemporalNode.build(
                        node_id=f"{canonical_role}_{len(nodes) + 1}",
                        role=canonical_role,
                        context=self._text_to_context(query_text),
                        context_dim=self.context_dim,
                        device=self.device,
                    )
                )
        return nodes

    def _canonical_role(self, role_text: str) -> str:
        lowered = role_text.lower()
        for canonical_role, aliases in ROLE_ALIASES.items():
            if lowered == canonical_role or any(alias.lower() == lowered for alias in aliases):
                return canonical_role
        return lowered.replace(" ", "_")

    def _match_context_window(self, text: str, start: int, end: int, radius: int = 48) -> str:
        left = max(0, start - radius)
        right = min(len(text), end + radius)
        return text[left:right].strip()

    def _text_to_context(self, text: str) -> list[float]:
        buckets = [0.0] * self.context_dim
        if not text or self.context_dim <= 0:
            return buckets

        normalized = text.lower().strip()
        tokens = re.findall(r"[a-z0-9_\u4e00-\u9fff]+", normalized)
        compact_text = re.sub(r"\s+", " ", normalized)

        def add_feature(feature: str, weight: float) -> None:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.context_dim
            sign = 1.0 if (digest[8] % 2 == 0) else -1.0
            buckets[index] += sign * weight

        for token in tokens:
            add_feature(f"tok:{token}", 1.0)
        for left, right in zip(tokens, tokens[1:]):
            add_feature(f"bigram:{left}|{right}", 0.75)
        for idx in range(max(len(compact_text) - 2, 0)):
            add_feature(f"tri:{compact_text[idx:idx + 3]}", 0.25)
        for canonical_role, aliases in ROLE_ALIASES.items():
            if canonical_role in normalized or any(alias.lower() in normalized for alias in aliases):
                add_feature(f"role:{canonical_role}", 1.5)

        if self.context_dim >= 2:
            buckets[-1] = min(len(tokens) / 32.0, 1.0)
            buckets[-2] = min(len(compact_text) / 256.0, 1.0)

        norm = sum(value * value for value in buckets) ** 0.5
        if norm > 0:
            buckets = [value / norm for value in buckets]
        return buckets
