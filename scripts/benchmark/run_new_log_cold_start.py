from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _agent_order_key(agent_id: str) -> tuple[int, int, str]:
    if agent_id == "PLANNER":
        return (1, 10**9, agent_id)
    match = re.search(r"(\d+)$", agent_id)
    numeric = int(match.group(1)) if match else 10**8
    return (0, numeric, agent_id)


def _ordered_agents(agent_ids: list[str]) -> list[str]:
    return sorted({str(agent_id) for agent_id in agent_ids}, key=_agent_order_key)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(records: list[dict[str, Any]], field_name: str) -> dict[str, float]:
    values = [float(record.get(field_name, 0.0)) for record in records]
    if not values:
        return {
            f"{field_name}_mean": 0.0,
            f"{field_name}_p50": 0.0,
            f"{field_name}_p95": 0.0,
            f"{field_name}_max": 0.0,
        }
    return {
        f"{field_name}_mean": sum(values) / len(values),
        f"{field_name}_p50": _percentile(values, 0.50),
        f"{field_name}_p95": _percentile(values, 0.95),
        f"{field_name}_max": max(values),
    }


@dataclass(slots=True)
class OnlinePatternMemory:
    bigram: Counter[tuple[str, str]] = field(default_factory=Counter)
    trigram: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    position: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def update_sequence(self, sequence: list[str]) -> None:
        for index, agent_id in enumerate(sequence):
            self.position[index][agent_id] += 1
            if index >= 1:
                self.bigram[(sequence[index - 1], agent_id)] += 1
            if index >= 2:
                self.trigram[(sequence[index - 2], sequence[index - 1], agent_id)] += 1


@dataclass(slots=True)
class PolicyState:
    local_bigram: Counter[tuple[str, str]] = field(default_factory=Counter)
    local_trigram: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    local_position: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    observed_edge_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    event_id_to_agent: dict[str, str] = field(default_factory=dict)
    sequence: list[str] = field(default_factory=list)
    last_seen_index: dict[str, int] = field(default_factory=dict)

    def update_after_prediction(self, event: dict[str, Any], target_agent: str) -> None:
        index = len(self.sequence)
        previous = self.sequence[-1] if self.sequence else None
        previous_previous = self.sequence[-2] if len(self.sequence) > 1 else None
        if previous is not None:
            self.local_bigram[(previous, target_agent)] += 1
        if previous_previous is not None and previous is not None:
            self.local_trigram[(previous_previous, previous, target_agent)] += 1
        self.local_position[index][target_agent] += 1
        for parent_event_id in event.get("parent_event_ids") or []:
            parent_agent = self.event_id_to_agent.get(str(parent_event_id))
            if parent_agent is not None:
                self.observed_edge_counts[parent_agent][target_agent] += 1
        self.event_id_to_agent[str(event.get("event_id") or index)] = target_agent
        self.last_seen_index[target_agent] = index
        self.sequence.append(target_agent)


@dataclass(slots=True)
class NextAgentPolicyState:
    sequence: list[str] = field(default_factory=list)
    round_main_agents: list[str] = field(default_factory=list)
    event_id_to_agent: dict[str, str] = field(default_factory=dict)
    local_main_transition_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    local_last_target_by_source: dict[str, str] = field(default_factory=dict)
    local_outgoing_rank_counts: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    local_round_index_counts: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    local_round_target_counts: Counter[str] = field(default_factory=Counter)
    local_file_target_counts: Counter[str] = field(default_factory=Counter)
    local_recent_targets: list[str] = field(default_factory=list)
    source_seen_targets: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    local_first_main_target: str | None = None
    local_second_main_target: str | None = None

    def update_after_transition(
        self,
        *,
        event: dict[str, Any],
        event_type: str,
        observed_next_agent: str,
        outgoing_agents: list[str],
        learn_transition: bool = True,
    ) -> None:
        current_agent = str(event.get("agent_id") or "")
        if event_type == "main_turn":
            round_index = len(self.round_main_agents)
            if learn_transition:
                self.local_main_transition_counts[current_agent][observed_next_agent] += 1
                self.local_last_target_by_source[current_agent] = observed_next_agent
                if observed_next_agent in outgoing_agents:
                    self.local_outgoing_rank_counts[current_agent][
                        outgoing_agents.index(observed_next_agent)
                    ] += 1
                self.local_round_index_counts[round_index][observed_next_agent] += 1
                self.local_round_target_counts[observed_next_agent] += 1
                self.local_file_target_counts[observed_next_agent] += 1
                self.local_recent_targets.append(observed_next_agent)
                self.local_recent_targets = self.local_recent_targets[-6:]
                if self.local_first_main_target is None:
                    self.local_first_main_target = observed_next_agent
                elif self.local_second_main_target is None:
                    self.local_second_main_target = observed_next_agent
                if observed_next_agent in outgoing_agents:
                    self.source_seen_targets[current_agent][observed_next_agent] += 1
            if current_agent not in self.round_main_agents:
                self.round_main_agents.append(current_agent)
        elif event_type == "planner_continue":
            self.round_main_agents.clear()
            self.local_round_target_counts.clear()
        self.event_id_to_agent[str(event.get("event_id") or len(self.sequence))] = current_agent
        self.sequence.append(current_agent)


@dataclass(slots=True)
class NextAgentGlobalMemory:
    main_transition_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    outgoing_rank_counts: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    round_index_counts: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    source_number_delta_counts: dict[int, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    first_main_targets: list[str] = field(default_factory=list)
    agent_profile_tokens: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    agent_profile_seen: Counter[str] = field(default_factory=Counter)

    def update(
        self,
        *,
        current_agent: str,
        observed_next_agent: str,
        outgoing_agents: list[str],
        round_index: int,
    ) -> None:
        self.main_transition_counts[current_agent][observed_next_agent] += 1
        if observed_next_agent in outgoing_agents:
            rank = outgoing_agents.index(observed_next_agent)
            self.outgoing_rank_counts[current_agent][rank] += 1
            source_number = _agent_order_key(current_agent)[1]
            target_number = _agent_order_key(observed_next_agent)[1]
            if source_number < 10**8 and target_number < 10**8:
                self.source_number_delta_counts[source_number][target_number - source_number] += 1
        self.round_index_counts[round_index][observed_next_agent] += 1

    def update_file_summary(
        self,
        first_main_target: str | None,
        profile_texts: dict[str, str] | None = None,
    ) -> None:
        if first_main_target:
            self.first_main_targets.append(first_main_target)
        for agent_id, profile_text in (profile_texts or {}).items():
            self.agent_profile_tokens[agent_id].update(_token_vector(profile_text))
            self.agent_profile_seen[agent_id] += 1

    def profile_stability(self, event: dict[str, Any], agent_id: str) -> float:
        if self.agent_profile_seen[agent_id] == 0:
            return 1.0
        agents = event.get("agents") or {}
        profile_text = ""
        if isinstance(agents.get(agent_id), dict):
            profile_text = str(agents[agent_id].get("profile") or "")
        if not profile_text:
            return 1.0
        return _cosine_similarity(
            _token_vector(profile_text),
            self.agent_profile_tokens[agent_id],
        )


@dataclass(slots=True)
class OnlineNextAgentReranker:
    learning_rate: float = 0.03
    base_score_scale: float = 0.1
    weights: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def score(
        self,
        *,
        candidate: str,
        base_score: float,
        features: dict[str, float],
    ) -> float:
        learned_score = sum(
            self.weights.get(name, 0.0) * value
            for name, value in features.items()
        )
        return self.base_score_scale * base_score + learned_score

    def update(
        self,
        *,
        expected_agent: str,
        predicted_agent: str,
        features_by_agent: dict[str, dict[str, float]],
    ) -> None:
        if expected_agent == predicted_agent:
            return
        expected_features = features_by_agent.get(expected_agent)
        predicted_features = features_by_agent.get(predicted_agent)
        if expected_features is None or predicted_features is None:
            return
        for name, value in expected_features.items():
            self.weights[name] += self.learning_rate * value
        for name, value in predicted_features.items():
            self.weights[name] -= self.learning_rate * value


def _rank_agents(
    *,
    agents: list[str],
    state: PolicyState,
    global_memory: OnlinePatternMemory,
    step_index: int,
) -> tuple[list[str], dict[str, float]]:
    scores: Counter[str] = Counter()
    for rank, agent_id in enumerate(agents):
        scores[agent_id] += 0.001 * (len(agents) - rank)

    if not state.sequence:
        if "agent1" in agents:
            scores["agent1"] += 100.0
        return [agent_id for agent_id, _ in scores.most_common()], dict(scores)

    previous = state.sequence[-1]
    previous_previous = state.sequence[-2] if len(state.sequence) > 1 else None
    non_planner_numbers = [
        _agent_order_key(agent_id)[1]
        for agent_id in agents
        if agent_id != "PLANNER"
    ]
    max_agent_number = max(non_planner_numbers) if non_planner_numbers else 0

    # Generic workflow priors derived from the already-visible agent roster.
    # They do not inspect current request text or any precomputed candidate field.
    if previous == "PLANNER":
        if previous_previous == "PLANNER" and "agent1" in agents:
            scores["agent1"] += 20.0
        else:
            scores["PLANNER"] += 20.0
    else:
        previous_number = _agent_order_key(previous)[1]
        next_agent = f"agent{previous_number + 1}"
        if next_agent in agents:
            scores[next_agent] += 2.0
        if "agent1" in agents:
            scores["agent1"] += 1.0
        if previous_number >= max_agent_number - 1 and "PLANNER" in agents:
            scores["PLANNER"] += 0.5

    for candidate in agents:
        scores[candidate] += 8.0 * state.local_bigram[(previous, candidate)]
        scores[candidate] += 2.0 * global_memory.bigram[(previous, candidate)]
        scores[candidate] += 3.0 * state.observed_edge_counts[previous][candidate]
        scores[candidate] += 4.0 * state.local_position[step_index][candidate]
        scores[candidate] += 1.0 * global_memory.position[step_index][candidate]
        if previous_previous is not None:
            scores[candidate] += 20.0 * state.local_trigram[
                (previous_previous, previous, candidate)
            ]
            scores[candidate] += 4.0 * global_memory.trigram[
                (previous_previous, previous, candidate)
            ]
        if candidate in state.last_seen_index:
            scores[candidate] -= 0.02 / (1 + step_index - state.last_seen_index[candidate])

    return [agent_id for agent_id, _ in scores.most_common()], dict(scores)


def _event_request_content(event: dict[str, Any]) -> str:
    messages = (event.get("request") or {}).get("messages") or []
    if not messages:
        return ""
    return str(messages[0].get("content") or "")


def _classify_event(event: dict[str, Any]) -> str:
    request = event.get("request") or {}
    content = _event_request_content(event)
    if request.get("tools"):
        return "main_turn"
    if "Continue your own turn based on this delegated result" in content:
        return "continuation"
    if content.startswith("Summarize the output of the agents"):
        return "planner_summary"
    if (
        content.startswith("Based on the following agents' results")
        or "Respond with a JSON object containing a single key 'continue'" in content
    ):
        return "planner_continue"
    if str(event.get("agent_id") or "") == "PLANNER":
        return "planner"
    return "delegate_target"


_RECOVERY_EVENT_TYPES = {
    "continuation",
    "planner_summary",
    "planner_continue",
    "planner",
}


def _is_predictive_next_agent_step(current_event_type: str, next_event_type: str) -> bool:
    return current_event_type == "main_turn" and next_event_type not in _RECOVERY_EVENT_TYPES


def _infer_prompt_collaboration_edges(content: str) -> list[tuple[str, str]]:
    return [
        (str(source), str(target))
        for source, target in re.findall(r"(agent\d+) collaborates with (agent\d+)", content)
    ]


def _explicit_graph_edges(event: dict[str, Any]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    graph_like = (
        event.get("graph")
        or event.get("agent_graph")
        or event.get("collaboration_graph")
        or event.get("workflow_graph")
    )
    raw_edges: Any = None
    if isinstance(graph_like, dict):
        raw_edges = (
            graph_like.get("edges")
            or graph_like.get("relationships")
            or graph_like.get("links")
        )
    elif isinstance(graph_like, list):
        raw_edges = graph_like
    for edge in raw_edges or []:
        source: Any = None
        target: Any = None
        if isinstance(edge, dict):
            source = edge.get("source") or edge.get("src") or edge.get("from")
            target = edge.get("target") or edge.get("dst") or edge.get("to")
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            source, target = edge[0], edge[1]
        if source and target:
            edges.append((str(source), str(target)))
    return edges


def _infer_collaboration_edges(event: dict[str, Any]) -> tuple[list[tuple[str, str]], str]:
    explicit_edges = _explicit_graph_edges(event)
    if explicit_edges:
        return explicit_edges, "explicit_graph_field"
    prompt_edges = _infer_prompt_collaboration_edges(_event_request_content(event))
    if prompt_edges:
        return prompt_edges, "prompt_collaboration_edges"
    return [], "static_agents_roster"


def _tool_enum_outgoing_agents(event: dict[str, Any], agents: list[str]) -> list[str]:
    for tool in (event.get("request") or {}).get("tools") or []:
        function = tool.get("function") or {}
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        target_property = properties.get("target_agent_id") or {}
        enum_values = target_property.get("enum") or []
        outgoing = [
            str(agent_id)
            for agent_id in enum_values
            if str(agent_id) in agents
        ]
        if outgoing:
            return outgoing
    return []


def _infer_outgoing_agents(event: dict[str, Any], agents: list[str]) -> list[str]:
    outgoing, _ = _infer_outgoing_agents_with_source(event, agents)
    return outgoing


def _infer_outgoing_agents_with_source(
    event: dict[str, Any],
    agents: list[str],
) -> tuple[list[str], str]:
    current_agent = str(event.get("agent_id") or "")
    tool_outgoing = _tool_enum_outgoing_agents(event, agents)
    if tool_outgoing:
        return tool_outgoing, "tool_schema_target_enum"
    edges, _ = _infer_collaboration_edges(event)
    outgoing = [target for source, target in edges if source == current_agent and target in agents]
    if outgoing:
        _, graph_source = _infer_collaboration_edges(event)
        return _ordered_agents(outgoing), graph_source
    return (
        [
            agent_id
            for agent_id in agents
            if agent_id not in {current_agent, "PLANNER"}
        ],
        "static_agents_roster",
    )


def _infer_delegate_source(content: str) -> str | None:
    match = re.search(r"From\s+(agent\d+|PLANNER)\s+to\s+(agent\d+|PLANNER)", content)
    if match:
        return str(match.group(1))
    return None


def _target_refs_in_visible_history(content: str, candidates: list[str]) -> list[str]:
    candidate_set = set(candidates)
    return [
        target
        for target in re.findall(r'"target_agent_id"\s*:\s*"([^"]+)"', content)
        if target in candidate_set
    ]


_TOKEN_NORMALIZATIONS = {
    "vision-language": "vision language",
    "large vision-language models": "large vision language models",
    "lvlms": "large vision language models",
    "mllms": "multimodal large language models",
    "ui": "user interface",
    "uis": "user interface screens",
}

_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "these",
    "those",
    "you",
    "your",
    "our",
    "their",
    "agent",
    "agents",
    "researcher",
    "research",
    "work",
    "works",
    "recent",
    "focus",
    "strong",
    "task",
    "current",
    "profile",
    "problem",
    "approach",
    "model",
    "models",
    "system",
    "systems",
    "method",
    "methods",
    "data",
    "machine",
    "learning",
    "new",
    "novel",
    "use",
    "using",
    "used",
    "develop",
    "developed",
    "proposed",
    "proposing",
    "significant",
    "overall",
    "through",
    "aims",
    "contribute",
    "field",
    "fields",
    "applications",
    "practical",
    "various",
    "efficient",
    "efficiency",
    "performance",
    "introduction",
    "following",
    "output",
    "question",
    "answer",
    "collaboration",
    "protocol",
    "available",
    "listed",
    "bounded",
    "request",
}


def _text_tokens(text: str) -> list[str]:
    normalized = (text or "").lower()
    for source, target in _TOKEN_NORMALIZATIONS.items():
        normalized = normalized.replace(source, target)
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 2 and token not in _STOP_WORDS
    ]


def _token_vector(text: str) -> Counter[str]:
    return Counter(_text_tokens(text))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _agent_profile_similarity(event: dict[str, Any], agent_id: str) -> float:
    agents = event.get("agents") or {}
    profile = ""
    if isinstance(agents.get(agent_id), dict):
        profile = str(agents[agent_id].get("profile") or "")
    return _cosine_similarity(
        _token_vector(str(event.get("task_profile") or "")),
        _token_vector(profile),
    )


def _visible_memory_text(content: str) -> str:
    marker = "These are your memory:"
    index = content.find(marker)
    if index < 0:
        return ""
    return content[index + len(marker) :]


def _visible_prompt_text(content: str) -> str:
    marker = "These are your memory:"
    index = content.find(marker)
    if index < 0:
        return content
    return content[:index]


def _profile_similarity_against_text(
    *,
    event: dict[str, Any],
    agent_id: str,
    text: str,
) -> float:
    agents = event.get("agents") or {}
    profile = ""
    if isinstance(agents.get(agent_id), dict):
        profile = str(agents[agent_id].get("profile") or "")
    return _cosine_similarity(_token_vector(text), _token_vector(profile))


def _candidate_idf_profile_scores(
    *,
    event: dict[str, Any],
    candidates: list[str],
    text: str,
) -> dict[str, float]:
    if not candidates or not text:
        return {agent_id: 0.0 for agent_id in candidates}
    agents = event.get("agents") or {}
    profile_tokens_by_agent: dict[str, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()
    for agent_id in candidates:
        profile_text = ""
        if isinstance(agents.get(agent_id), dict):
            profile_text = str(agents[agent_id].get("profile") or "")
        profile_tokens = _token_vector(profile_text)
        profile_tokens_by_agent[agent_id] = profile_tokens
        document_frequency.update(profile_tokens.keys())
    query_tokens = _token_vector(text)
    candidate_count = len(candidates)
    idf = {
        token: math.log((candidate_count + 1.0) / (document_frequency[token] + 1.0)) + 1.0
        for token in set(query_tokens) | set(document_frequency)
    }
    query_vector = Counter(
        {token: count * idf.get(token, 1.0) for token, count in query_tokens.items()}
    )
    scores: dict[str, float] = {}
    for agent_id, profile_tokens in profile_tokens_by_agent.items():
        profile_vector = Counter(
            {token: count * idf.get(token, 1.0) for token, count in profile_tokens.items()}
        )
        scores[agent_id] = _cosine_similarity(query_vector, profile_vector)
    return scores


def _agent_context_text(event: dict[str, Any], agent_id: str) -> str:
    agents = event.get("agents") or {}
    agent = agents.get(agent_id)
    if not isinstance(agent, dict):
        return ""
    return str(agent.get("context") or "")


def _visible_agent_context_features(
    *,
    event: dict[str, Any],
    agents: list[str],
    candidates: list[str],
    memory_text: str,
) -> dict[str, dict[str, float]]:
    all_context = "\n".join(_agent_context_text(event, agent_id) for agent_id in agents)
    max_context_length = max(
        (len(_agent_context_text(event, agent_id)) for agent_id in candidates),
        default=1,
    )
    max_target_refs = max(
        (
            len(
                re.findall(
                    rf'"target_agent_id"\s*:\s*"{re.escape(agent_id)}"',
                    all_context,
                )
            )
            for agent_id in candidates
        ),
        default=1,
    )
    memory_vector = _token_vector(memory_text)
    task_vector = _token_vector(str(event.get("task_profile") or ""))
    features: dict[str, dict[str, float]] = {}
    for agent_id in candidates:
        context = _agent_context_text(event, agent_id)
        context_vector = _token_vector(context)
        target_ref_count = len(
            re.findall(
                rf'"target_agent_id"\s*:\s*"{re.escape(agent_id)}"',
                all_context,
            )
        )
        features[agent_id] = {
            "context_length": math.log1p(len(context)) / 12.0,
            "context_length_ratio": len(context) / max(max_context_length, 1),
            "context_memory_similarity": _cosine_similarity(memory_vector, context_vector),
            "context_task_similarity": _cosine_similarity(task_vector, context_vector),
            "context_target_ref_ratio": target_ref_count / max(max_target_refs, 1),
            "context_target_ref_count": target_ref_count / 10.0,
        }
    return features


def _rank_next_agents(
    *,
    event: dict[str, Any],
    next_state: NextAgentPolicyState,
    next_global_memory: NextAgentGlobalMemory,
    use_cross_file_memory: bool,
    cross_file_stat_weight: float,
    enable_adaptive_cross_file_prior: bool,
    adaptive_cross_file_weight: float,
    adaptive_cross_file_min_support: int,
    adaptive_cross_file_min_confidence: float,
    adaptive_cross_file_min_profile_stability: float,
    enable_research_schedule_prior: bool,
    enable_research_meta_prior: bool,
    enable_idf_profile_prior: bool,
    include_visible_agent_context: bool,
    visible_context_similarity_weight: float,
    visible_context_length_weight: float,
    agents: list[str],
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    current_agent = str(event.get("agent_id") or "")
    event_type = _classify_event(event)
    content = _event_request_content(event)
    scores: Counter[str] = Counter()
    task_profile_scores: dict[str, float] = {}
    memory_profile_scores: dict[str, float] = {}
    prompt_profile_scores: dict[str, float] = {}
    task_idf_profile_scores: dict[str, float] = {}
    memory_idf_profile_scores: dict[str, float] = {}
    prompt_idf_profile_scores: dict[str, float] = {}
    online_calibration_scores: dict[str, float] = {}
    schedule_prior_scores: dict[str, float] = {}
    meta_prior_scores: dict[str, float] = {}
    adaptive_cross_file_scores: dict[str, float] = {}
    adaptive_cross_file_profile_stability = 0.0
    visible_context_scores: dict[str, float] = {}
    visible_context_features: dict[str, dict[str, float]] = {}
    for rank, agent_id in enumerate(agents):
        scores[agent_id] += 0.001 * (len(agents) - rank)

    reason = "fallback_order"
    graph_source = "static_agents_roster"
    outgoing_agents: list[str] = []

    if event_type == "delegate_target":
        source_agent = _infer_delegate_source(content)
        if source_agent in agents:
            scores[source_agent] += 1000.0
            reason = "delegate_request_source"
            graph_source = "request_from_to_edge"
    elif event_type == "continuation":
        used_agents = set(next_state.round_main_agents)
        used_agents.add(current_agent)
        next_main_agent = None
        for agent_id in agents:
            if agent_id != "PLANNER" and agent_id not in used_agents:
                next_main_agent = agent_id
                break
        if next_main_agent is None and "PLANNER" in agents:
            next_main_agent = "PLANNER"
        if next_main_agent is not None:
            scores[next_main_agent] += 1000.0
            reason = "round_robin_after_continuation"
            graph_source = "scheduler_round_state"
    elif event_type == "planner_summary":
        if "PLANNER" in agents:
            scores["PLANNER"] += 1000.0
            reason = "planner_summary_self_loop"
            graph_source = "scheduler_planner_state"
    elif event_type == "planner_continue":
        if "agent1" in agents:
            scores["agent1"] += 1000.0
            reason = "planner_starts_next_round"
            graph_source = "scheduler_round_state"
    elif event_type == "main_turn":
        outgoing_agents, graph_source = _infer_outgoing_agents_with_source(event, agents)
        candidates = list(outgoing_agents)
        if "PLANNER" in agents:
            candidates.append("PLANNER")
        candidate_set = set(candidates)
        scores = Counter({agent_id: 0.0 for agent_id in candidates})
        for rank, agent_id in enumerate(candidates):
            scores[agent_id] += 0.0001 * (len(candidates) - rank)
        reason = "main_turn_graph_semantic_online_memory"

        for rank, agent_id in enumerate(outgoing_agents):
            scores[agent_id] += 2.0 * (len(outgoing_agents) - rank) / max(1, len(outgoing_agents))

        if enable_research_schedule_prior:
            source_turn_count = sum(next_state.local_main_transition_counts[current_agent].values())
            if outgoing_agents:
                first_outgoing = outgoing_agents[0]
                first_bonus = 2.0
                if source_turn_count >= 2:
                    first_bonus += 20.0
                scores[first_outgoing] += first_bonus
                schedule_prior_scores[first_outgoing] = (
                    schedule_prior_scores.get(first_outgoing, 0.0) + first_bonus
                )
            if "agent1" in candidate_set:
                scores["agent1"] += 5.0
                schedule_prior_scores["agent1"] = (
                    schedule_prior_scores.get("agent1", 0.0) + 5.0
                )

        task_profile_scores = {
            agent_id: _agent_profile_similarity(event, agent_id)
            for agent_id in outgoing_agents
        }
        memory_text = _visible_memory_text(content)
        prompt_text = _visible_prompt_text(content)
        memory_profile_scores = {
            agent_id: _profile_similarity_against_text(
                event=event,
                agent_id=agent_id,
                text=memory_text,
            )
            for agent_id in outgoing_agents
        }
        prompt_profile_scores = {
            agent_id: _profile_similarity_against_text(
                event=event,
                agent_id=agent_id,
                text=prompt_text,
            )
            for agent_id in outgoing_agents
        }
        if enable_idf_profile_prior:
            task_idf_profile_scores = _candidate_idf_profile_scores(
                event=event,
                candidates=outgoing_agents,
                text=str(event.get("task_profile") or ""),
            )
            memory_idf_profile_scores = _candidate_idf_profile_scores(
                event=event,
                candidates=outgoing_agents,
                text=memory_text,
            )
            prompt_idf_profile_scores = _candidate_idf_profile_scores(
                event=event,
                candidates=outgoing_agents,
                text=prompt_text,
            )
        task_profile_ranked = [
            agent_id
            for agent_id, _ in sorted(
                task_profile_scores.items(),
                key=lambda item: (item[1], -outgoing_agents.index(item[0])),
                reverse=True,
            )
        ]
        memory_profile_ranked = [
            agent_id
            for agent_id, _ in sorted(
                memory_profile_scores.items(),
                key=lambda item: (item[1], -outgoing_agents.index(item[0])),
                reverse=True,
            )
        ]
        prompt_profile_ranked = [
            agent_id
            for agent_id, _ in sorted(
                prompt_profile_scores.items(),
                key=lambda item: (item[1], -outgoing_agents.index(item[0])),
                reverse=True,
            )
        ]
        for agent_id in outgoing_agents:
            scores[agent_id] += 20.0 * task_profile_scores.get(agent_id, 0.0)
            scores[agent_id] += 30.0 * memory_profile_scores.get(agent_id, 0.0)
            scores[agent_id] += 20.0 * prompt_profile_scores.get(agent_id, 0.0)
            if enable_idf_profile_prior:
                scores[agent_id] += 20.0 * task_idf_profile_scores.get(agent_id, 0.0)
                scores[agent_id] += 24.0 * memory_idf_profile_scores.get(agent_id, 0.0)
                scores[agent_id] += 12.0 * prompt_idf_profile_scores.get(agent_id, 0.0)
        for rank, agent_id in enumerate(task_profile_ranked):
            scores[agent_id] += 0.5 * (len(task_profile_ranked) - rank) / max(1, len(task_profile_ranked))
        for rank, agent_id in enumerate(memory_profile_ranked):
            scores[agent_id] += 4.0 * (len(memory_profile_ranked) - rank) / max(1, len(memory_profile_ranked))
        for rank, agent_id in enumerate(prompt_profile_ranked):
            scores[agent_id] += 1.0 * (len(prompt_profile_ranked) - rank) / max(1, len(prompt_profile_ranked))

        if include_visible_agent_context:
            visible_context_features = _visible_agent_context_features(
                event=event,
                agents=agents,
                candidates=outgoing_agents,
                memory_text=memory_text,
            )
            for agent_id, features in visible_context_features.items():
                context_score = (
                    visible_context_similarity_weight
                    * features.get("context_memory_similarity", 0.0)
                    + visible_context_length_weight
                    * features.get("context_length", 0.0)
                )
                scores[agent_id] += context_score
                visible_context_scores[agent_id] = context_score

        if enable_research_meta_prior:
            if use_cross_file_memory and next_global_memory.first_main_targets:
                previous_first_target = next_global_memory.first_main_targets[-1]
                if previous_first_target in candidate_set:
                    main_turn_index = sum(next_state.local_file_target_counts.values())
                    source_turn_count = sum(
                        next_state.local_main_transition_counts[current_agent].values()
                    )
                    previous_first_bonus = 10.0
                    if main_turn_index == 0:
                        previous_first_bonus -= 5.0
                    if main_turn_index <= 2:
                        previous_first_bonus += 5.0
                    if source_turn_count == 0:
                        previous_first_bonus -= 5.0
                    scores[previous_first_target] += previous_first_bonus
                    meta_prior_scores[previous_first_target] = (
                        meta_prior_scores.get(previous_first_target, 0.0)
                        + previous_first_bonus
                    )
            if next_state.local_second_main_target in candidate_set:
                scores[next_state.local_second_main_target] += 1.0
                meta_prior_scores[next_state.local_second_main_target] = (
                    meta_prior_scores.get(next_state.local_second_main_target, 0.0) + 1.0
                )

        visible_targets = _target_refs_in_visible_history(content, candidates)
        for target_agent, count in Counter(visible_targets[-8:]).items():
            scores[target_agent] -= 8.0 * count
            online_calibration_scores[target_agent] = (
                online_calibration_scores.get(target_agent, 0.0) - 8.0 * count
            )
        if visible_targets:
            scores[visible_targets[-1]] += 4.0
            online_calibration_scores[visible_targets[-1]] = (
                online_calibration_scores.get(visible_targets[-1], 0.0) + 4.0
            )
            for target_agent in set(visible_targets[-4:]):
                scores[target_agent] -= 20.0
                online_calibration_scores[target_agent] = (
                    online_calibration_scores.get(target_agent, 0.0) - 20.0
                )

        for target_agent, count in next_state.local_main_transition_counts[current_agent].items():
            if target_agent in candidate_set:
                scores[target_agent] += 12.0 * count
                online_calibration_scores[target_agent] = (
                    online_calibration_scores.get(target_agent, 0.0) + 12.0 * count
                )
        last_target = next_state.local_last_target_by_source.get(current_agent)
        if last_target in candidate_set:
            scores[last_target] += 0.0
        for rank, count in next_state.local_outgoing_rank_counts[current_agent].items():
            if 0 <= rank < len(outgoing_agents):
                scores[outgoing_agents[rank]] += 1.0 * count
        round_index = len(next_state.round_main_agents)
        for target_agent, count in next_state.local_round_index_counts[round_index].items():
            if target_agent in candidate_set:
                scores[target_agent] += 4.0 * count
                online_calibration_scores[target_agent] = (
                    online_calibration_scores.get(target_agent, 0.0) + 4.0 * count
                )
        for target_agent, count in next_state.local_round_target_counts.items():
            if target_agent in candidate_set:
                scores[target_agent] += 3.0 * count
                online_calibration_scores[target_agent] = (
                    online_calibration_scores.get(target_agent, 0.0) + 3.0 * count
                )
        for target_agent, count in next_state.local_file_target_counts.items():
            if target_agent in candidate_set:
                scores[target_agent] += 3.0 * count
                online_calibration_scores[target_agent] = (
                    online_calibration_scores.get(target_agent, 0.0) + 3.0 * count
                )
        for target_agent, count in Counter(next_state.local_recent_targets[-3:]).items():
            if target_agent in candidate_set:
                scores[target_agent] += 0.5 * count
                online_calibration_scores[target_agent] = (
                    online_calibration_scores.get(target_agent, 0.0) + 0.5 * count
                )
        if next_state.local_file_target_counts:
            hub_agent, hub_count = next_state.local_file_target_counts.most_common(1)[0]
            observed_count = sum(next_state.local_file_target_counts.values())
            if (
                hub_agent in outgoing_agents
                and hub_agent != current_agent
                and hub_count >= 2
                and hub_count / max(observed_count, 1) >= 0.45
            ):
                hub_bonus = min(10.0, 2.0 * hub_count)
                scores[hub_agent] += hub_bonus
                online_calibration_scores[hub_agent] = (
                    online_calibration_scores.get(hub_agent, 0.0) + hub_bonus
                )
        seen_for_source = next_state.source_seen_targets[current_agent]
        unseen_outgoing = [agent_id for agent_id in outgoing_agents if seen_for_source[agent_id] == 0]
        if visible_targets and unseen_outgoing and len(outgoing_agents) <= 4:
            for target_agent in unseen_outgoing:
                coverage_bonus = 3.0
                scores[target_agent] += coverage_bonus
                online_calibration_scores[target_agent] = (
                    online_calibration_scores.get(target_agent, 0.0) + coverage_bonus
                )

        if use_cross_file_memory and enable_adaptive_cross_file_prior:
            profile_stability = next_global_memory.profile_stability(event, current_agent)
            adaptive_cross_file_profile_stability = profile_stability
            transition_counts = next_global_memory.main_transition_counts[current_agent]
            candidate_transition_counts = Counter(
                {
                    target_agent: count
                    for target_agent, count in transition_counts.items()
                    if target_agent in candidate_set
                }
            )
            total_transition_count = sum(candidate_transition_counts.values())
            if total_transition_count >= adaptive_cross_file_min_support:
                ordered_targets = candidate_transition_counts.most_common()
                top_target, top_count = ordered_targets[0]
                second_count = ordered_targets[1][1] if len(ordered_targets) > 1 else 0
                confidence = top_count / total_transition_count
                margin = (top_count - second_count) / total_transition_count
                if (
                    confidence >= adaptive_cross_file_min_confidence
                    and profile_stability >= adaptive_cross_file_min_profile_stability
                    and margin > 0.0
                ):
                    bonus = (
                        adaptive_cross_file_weight
                        * profile_stability
                        * confidence
                        * (0.5 + margin)
                    )
                    scores[top_target] += bonus
                    adaptive_cross_file_scores[top_target] = (
                        adaptive_cross_file_scores.get(top_target, 0.0) + bonus
                    )

            rank_counts = next_global_memory.outgoing_rank_counts[current_agent]
            total_rank_count = sum(rank_counts.values())
            if total_rank_count >= adaptive_cross_file_min_support:
                ordered_ranks = rank_counts.most_common()
                top_rank, top_count = ordered_ranks[0]
                second_count = ordered_ranks[1][1] if len(ordered_ranks) > 1 else 0
                confidence = top_count / total_rank_count
                margin = (top_count - second_count) / total_rank_count
                if (
                    0 <= top_rank < len(outgoing_agents)
                    and confidence >= adaptive_cross_file_min_confidence
                    and profile_stability >= adaptive_cross_file_min_profile_stability
                    and margin > 0.0
                ):
                    target_agent = outgoing_agents[top_rank]
                    bonus = (
                        0.5
                        * adaptive_cross_file_weight
                        * profile_stability
                        * confidence
                        * (0.5 + margin)
                    )
                    scores[target_agent] += bonus
                    adaptive_cross_file_scores[target_agent] = (
                        adaptive_cross_file_scores.get(target_agent, 0.0) + bonus
                    )

        if (
            use_cross_file_memory
            and cross_file_stat_weight > 0.0
            and not enable_adaptive_cross_file_prior
        ):
            for target_agent, count in next_global_memory.main_transition_counts[current_agent].items():
                if target_agent in candidate_set:
                    scores[target_agent] += cross_file_stat_weight * 0.5 * count
            for rank, count in next_global_memory.outgoing_rank_counts[current_agent].items():
                if 0 <= rank < len(outgoing_agents):
                    scores[outgoing_agents[rank]] += cross_file_stat_weight * 1.0 * count
            for target_agent, count in next_global_memory.round_index_counts[round_index].items():
                if target_agent in candidate_set:
                    scores[target_agent] += cross_file_stat_weight * 0.5 * count
            source_number = _agent_order_key(current_agent)[1]
            if source_number < 10**8:
                for delta, count in next_global_memory.source_number_delta_counts[source_number].items():
                    target_agent = f"agent{source_number + delta}"
                    if target_agent in candidate_set:
                        scores[target_agent] += cross_file_stat_weight * 0.5 * count

        non_planner_agents = {agent_id for agent_id in agents if agent_id != "PLANNER"}
        if set(next_state.round_main_agents) | {current_agent} >= non_planner_agents:
            scores["PLANNER"] += 8.0
            online_calibration_scores["PLANNER"] = (
                online_calibration_scores.get("PLANNER", 0.0) + 8.0
            )

    if not scores:
        scores.update({agent_id: 0.0 for agent_id in agents})
    ranked = [agent_id for agent_id, _ in scores.most_common()]
    metadata = {
        "event_type": event_type,
        "current_agent_id": current_agent,
        "ranking_reason": reason,
        "graph_source": graph_source,
        "outgoing_agents": outgoing_agents,
        "candidate_agents": list(scores.keys()),
        "semantic_profile_scores": task_profile_scores if event_type == "main_turn" else {},
        "memory_profile_scores": memory_profile_scores if event_type == "main_turn" else {},
        "prompt_profile_scores": prompt_profile_scores if event_type == "main_turn" else {},
        "task_idf_profile_scores": task_idf_profile_scores if event_type == "main_turn" else {},
        "memory_idf_profile_scores": memory_idf_profile_scores if event_type == "main_turn" else {},
        "prompt_idf_profile_scores": prompt_idf_profile_scores if event_type == "main_turn" else {},
        "online_calibration_scores": online_calibration_scores if event_type == "main_turn" else {},
        "schedule_prior_scores": schedule_prior_scores if event_type == "main_turn" else {},
        "meta_prior_scores": meta_prior_scores if event_type == "main_turn" else {},
        "adaptive_cross_file_scores": adaptive_cross_file_scores if event_type == "main_turn" else {},
        "adaptive_cross_file_profile_stability": (
            adaptive_cross_file_profile_stability if event_type == "main_turn" else 0.0
        ),
        "visible_context_scores": visible_context_scores if event_type == "main_turn" else {},
        "visible_context_features": visible_context_features if event_type == "main_turn" else {},
    }
    return ranked, dict(scores), metadata


def _numeric_agent_id(agent_id: str) -> int:
    value = _agent_order_key(agent_id)[1]
    return value if value < 10**8 else -1


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def _online_reranker_features(
    *,
    event: dict[str, Any],
    agents: list[str],
    next_state: NextAgentPolicyState,
    ranked: list[str],
    scores: dict[str, float],
    prediction_metadata: dict[str, Any],
    candidate: str,
) -> dict[str, float]:
    current_agent = str(event.get("agent_id") or "")
    outgoing_agents = list(prediction_metadata.get("outgoing_agents") or [])
    candidate_agents = list(prediction_metadata.get("candidate_agents") or ranked)
    content = _event_request_content(event)
    visible_targets = _target_refs_in_visible_history(content, candidate_agents)
    recent_targets = Counter(next_state.local_recent_targets[-3:])
    visible_target_counts = Counter(visible_targets[-8:])

    rank_by_agent = {agent_id: index for index, agent_id in enumerate(ranked)}
    top_agent = ranked[0] if ranked else ""
    second_agent = ranked[1] if len(ranked) > 1 else ""
    top_score = scores.get(top_agent, 0.0)
    second_score = scores.get(second_agent, 0.0)
    candidate_score = scores.get(candidate, 0.0)
    candidate_rank = rank_by_agent.get(candidate, len(ranked) + 10)

    current_number = _numeric_agent_id(current_agent)
    candidate_number = _numeric_agent_id(candidate)
    round_index = len(next_state.round_main_agents)
    non_planner_agents = {agent_id for agent_id in agents if agent_id != "PLANNER"}
    round_covered = (set(next_state.round_main_agents) | {current_agent}) >= non_planner_agents
    outgoing_index = (
        outgoing_agents.index(candidate)
        if candidate in outgoing_agents
        else len(outgoing_agents)
    )
    source_total = sum(next_state.local_main_transition_counts[current_agent].values())
    file_total = sum(next_state.local_file_target_counts.values())
    source_seen = next_state.source_seen_targets[current_agent][candidate]

    task_profile_scores = prediction_metadata.get("semantic_profile_scores") or {}
    memory_profile_scores = prediction_metadata.get("memory_profile_scores") or {}
    prompt_profile_scores = prediction_metadata.get("prompt_profile_scores") or {}
    task_idf_profile_scores = prediction_metadata.get("task_idf_profile_scores") or {}
    memory_idf_profile_scores = prediction_metadata.get("memory_idf_profile_scores") or {}
    prompt_idf_profile_scores = prediction_metadata.get("prompt_idf_profile_scores") or {}
    schedule_prior_scores = prediction_metadata.get("schedule_prior_scores") or {}
    meta_prior_scores = prediction_metadata.get("meta_prior_scores") or {}
    adaptive_cross_file_scores = prediction_metadata.get("adaptive_cross_file_scores") or {}
    visible_context_scores = prediction_metadata.get("visible_context_scores") or {}
    visible_context_features = prediction_metadata.get("visible_context_features") or {}
    candidate_context_features = visible_context_features.get(candidate, {})
    source_turn_count = sum(next_state.local_main_transition_counts[current_agent].values())

    features: dict[str, float] = {
        "bias": 1.0,
        "base_score": candidate_score / 50.0,
        "base_rank": -candidate_rank / max(len(ranked), 1),
        "is_base_rank_1": float(candidate_rank == 0),
        "is_base_rank_2": float(candidate_rank == 1),
        "base_margin_to_top": (candidate_score - top_score) / 50.0,
        "top_margin": (top_score - second_score) / 50.0,
        "outgoing_index": -outgoing_index / max(len(outgoing_agents), 1),
        "is_first_outgoing": float(bool(outgoing_agents) and outgoing_index == 0),
        "is_second_outgoing": float(len(outgoing_agents) > 1 and outgoing_index == 1),
        "is_last_outgoing": float(bool(outgoing_agents) and outgoing_index == len(outgoing_agents) - 1),
        "is_planner": float(candidate == "PLANNER"),
        "is_agent1": float(candidate == "agent1"),
        "agent_count": len(agents) / 10.0,
        "outgoing_count": len(outgoing_agents) / 10.0,
        "round_index": round_index / 10.0,
        "round_covered": float(round_covered),
        "source_turn_count": source_turn_count / 5.0,
        "is_first_source_turn": float(source_turn_count == 0),
        "is_repeat_source_turn": float(source_turn_count > 0),
        "is_late_source_turn": float(source_turn_count >= 2),
        "task_profile_score": float(task_profile_scores.get(candidate, 0.0)),
        "memory_profile_score": float(memory_profile_scores.get(candidate, 0.0)),
        "prompt_profile_score": float(prompt_profile_scores.get(candidate, 0.0)),
        "task_idf_profile_score": float(task_idf_profile_scores.get(candidate, 0.0)),
        "memory_idf_profile_score": float(memory_idf_profile_scores.get(candidate, 0.0)),
        "prompt_idf_profile_score": float(prompt_idf_profile_scores.get(candidate, 0.0)),
        "schedule_prior_score": float(schedule_prior_scores.get(candidate, 0.0)) / 20.0,
        "meta_prior_score": float(meta_prior_scores.get(candidate, 0.0)) / 20.0,
        "adaptive_cross_file_score": float(
            adaptive_cross_file_scores.get(candidate, 0.0)
        )
        / 20.0,
        "visible_context_score": float(visible_context_scores.get(candidate, 0.0)) / 20.0,
        "context_memory_similarity": float(
            candidate_context_features.get("context_memory_similarity", 0.0)
        ),
        "context_task_similarity": float(
            candidate_context_features.get("context_task_similarity", 0.0)
        ),
        "context_length": float(candidate_context_features.get("context_length", 0.0)),
        "context_target_ref_ratio": float(
            candidate_context_features.get("context_target_ref_ratio", 0.0)
        ),
        "source_transition_count": next_state.local_main_transition_counts[current_agent][candidate] / 5.0,
        "source_transition_ratio": _safe_ratio(
            next_state.local_main_transition_counts[current_agent][candidate],
            source_total,
        ),
        "file_target_count": next_state.local_file_target_counts[candidate] / 10.0,
        "file_target_ratio": _safe_ratio(next_state.local_file_target_counts[candidate], file_total),
        "round_target_count": next_state.local_round_target_counts[candidate] / 5.0,
        "round_index_count": next_state.local_round_index_counts[round_index][candidate] / 5.0,
        "recent_target_count": recent_targets[candidate] / 3.0,
        "source_seen_count": source_seen / 5.0,
        "is_unseen_for_source": float(candidate in outgoing_agents and source_seen == 0),
        "visible_target_ref_count": visible_target_counts[candidate] / 5.0,
        "is_last_visible_target_ref": float(bool(visible_targets) and visible_targets[-1] == candidate),
        "is_in_last4_visible_target_refs": float(candidate in visible_targets[-4:]),
        "same_as_last_source_target": float(
            next_state.local_last_target_by_source.get(current_agent) == candidate
        ),
    }
    if current_number >= 0 and candidate_number >= 0:
        delta = candidate_number - current_number
        features["numeric_delta"] = delta / 10.0
        features[f"numeric_delta={delta}"] = 1.0
    features[f"current_agent={current_agent}"] = 1.0
    features[f"candidate_agent={candidate}"] = 1.0
    features[f"current_candidate={current_agent}->{candidate}"] = 1.0
    features[f"round_index_bucket={min(round_index, 6)}"] = 1.0
    features[f"source_turn_bucket={min(source_turn_count, 4)}"] = 1.0
    if candidate in outgoing_agents:
        features[f"outgoing_index_bucket={outgoing_index}"] = 1.0
    if round_covered:
        features[f"covered_round_current={current_agent}"] = 1.0
    return features


def _rerank_next_agents_online(
    *,
    event: dict[str, Any],
    agents: list[str],
    next_state: NextAgentPolicyState,
    ranked: list[str],
    scores: dict[str, float],
    prediction_metadata: dict[str, Any],
    reranker: OnlineNextAgentReranker | None,
) -> tuple[list[str], dict[str, float], dict[str, dict[str, float]], dict[str, float]]:
    if reranker is None or prediction_metadata.get("event_type") != "main_turn":
        return ranked, scores, {}, {}
    candidate_agents = list(prediction_metadata.get("candidate_agents") or scores.keys())
    features_by_agent = {
        candidate: _online_reranker_features(
            event=event,
            agents=agents,
            next_state=next_state,
            ranked=ranked,
            scores=scores,
            prediction_metadata=prediction_metadata,
            candidate=candidate,
        )
        for candidate in candidate_agents
    }
    reranker_scores = {
        candidate: reranker.score(
            candidate=candidate,
            base_score=scores.get(candidate, 0.0),
            features=features_by_agent[candidate],
        )
        for candidate in candidate_agents
    }
    reranked = sorted(
        candidate_agents,
        key=lambda agent_id: (
            reranker_scores.get(agent_id, 0.0),
            -_agent_order_key(agent_id)[1],
            agent_id,
        ),
        reverse=True,
    )
    return reranked, reranker_scores, features_by_agent, reranker_scores


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    return events


def _is_raw_event_log(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return False
            return (
                isinstance(event.get("agents"), dict)
                and "agent_id" in event
                and "event_id" in event
                and "workflow_id" in event
                and "task_profile" in event
            )
    return False


def _discover_raw_event_logs(log_root: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"_task_(\d+)_", path.name)
        task_index = int(match.group(1)) if match else 10**9
        return (task_index, path.name)

    return sorted(
        (
            path
            for path in log_root.glob("*.jsonl")
            if path.is_file() and _is_raw_event_log(path)
        ),
        key=sort_key,
    )


def _evaluate_file(
    path: Path,
    *,
    global_memory: OnlinePatternMemory,
    next_global_memory: NextAgentGlobalMemory,
    next_reranker: OnlineNextAgentReranker | None,
    use_cross_file_memory: bool,
    cross_file_stat_weight: float,
    enable_adaptive_cross_file_prior: bool,
    adaptive_cross_file_weight: float,
    adaptive_cross_file_min_support: int,
    adaptive_cross_file_min_confidence: float,
    adaptive_cross_file_min_profile_stability: float,
    enable_research_schedule_prior: bool,
    enable_research_meta_prior: bool,
    enable_idf_profile_prior: bool,
    enable_online_pair_calibration: bool,
    pair_calibration_margin: int,
    include_visible_agent_context: bool,
    visible_context_similarity_weight: float,
    visible_context_length_weight: float,
    dataset_name: str,
    prediction_target: str,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    events = _read_jsonl(path)
    if not events:
        return [], [], {"file_name": path.name, "event_count": 0}
    first_agents = events[0].get("agents") or {}
    agents = _ordered_agents(list(first_agents.keys()))
    state = PolicyState()
    next_state = NextAgentPolicyState()
    timing_records: list[dict[str, Any]] = []
    sequence: list[str] = []
    forbidden_current_text_hits = 0
    pair_calibration_counts: dict[tuple[str, str], Counter[bool]] = defaultdict(Counter)
    iterable = (
        enumerate(events)
        if prediction_target == "current_event"
        else enumerate(events[:-1])
    )
    for step_index, event in iterable:
        expected = (
            str(event.get("agent_id") or "")
            if prediction_target == "current_event"
            else str(events[step_index + 1].get("agent_id") or "")
        )
        expected_event_type = (
            "current_event"
            if prediction_target == "current_event"
            else _classify_event(events[step_index + 1])
        )
        started = time.perf_counter()
        if prediction_target == "current_event":
            ranked, scores = _rank_agents(
                agents=agents,
                state=state,
                global_memory=global_memory if use_cross_file_memory else OnlinePatternMemory(),
                step_index=step_index,
            )
            prediction_metadata = {
                "event_type": "current_event",
                "current_agent_id": None,
                "ranking_reason": "sequence_prefix_policy",
                "graph_source": "static_agents_roster",
                "outgoing_agents": [],
                "candidate_agents": agents,
            }
            base_ranked = list(ranked)
            reranker_features_by_agent: dict[str, dict[str, float]] = {}
            reranker_scores: dict[str, float] = {}
        else:
            ranked, scores, prediction_metadata = _rank_next_agents(
                event=event,
                next_state=next_state,
                next_global_memory=next_global_memory,
                use_cross_file_memory=use_cross_file_memory,
                cross_file_stat_weight=cross_file_stat_weight,
                enable_adaptive_cross_file_prior=enable_adaptive_cross_file_prior,
                adaptive_cross_file_weight=adaptive_cross_file_weight,
                adaptive_cross_file_min_support=adaptive_cross_file_min_support,
                adaptive_cross_file_min_confidence=adaptive_cross_file_min_confidence,
                adaptive_cross_file_min_profile_stability=adaptive_cross_file_min_profile_stability,
                enable_research_schedule_prior=enable_research_schedule_prior,
                enable_research_meta_prior=enable_research_meta_prior,
                enable_idf_profile_prior=enable_idf_profile_prior,
                include_visible_agent_context=include_visible_agent_context,
                visible_context_similarity_weight=visible_context_similarity_weight,
                visible_context_length_weight=visible_context_length_weight,
                agents=agents,
            )
            base_ranked = list(ranked)
            prediction_metadata["base_scores"] = dict(scores)
            ranked, scores, reranker_features_by_agent, reranker_scores = _rerank_next_agents_online(
                event=event,
                agents=agents,
                next_state=next_state,
                ranked=ranked,
                scores=scores,
                prediction_metadata=prediction_metadata,
                reranker=next_reranker,
            )
            pair_calibration_signature: tuple[str, str] | None = None
            if (
                enable_online_pair_calibration
                and prediction_metadata["event_type"] == "main_turn"
                and len(ranked) >= 2
            ):
                pair_calibration_signature = (ranked[0], ranked[1])
                pair_counts = pair_calibration_counts[pair_calibration_signature]
                if pair_counts[True] > pair_counts[False] + pair_calibration_margin:
                    ranked = [ranked[1], ranked[0], *ranked[2:]]
            prediction_metadata["pair_calibration_signature"] = (
                list(pair_calibration_signature) if pair_calibration_signature else []
            )
            prediction_metadata["pair_calibration_counts"] = (
                dict(pair_calibration_counts[pair_calibration_signature])
                if pair_calibration_signature
                else {}
            )
        prediction_time_ms = (time.perf_counter() - started) * 1000.0
        counted_for_metric = (
            True
            if prediction_target == "current_event"
            else _is_predictive_next_agent_step(
                str(prediction_metadata["event_type"]),
                expected_event_type,
            )
        )
        hit_at = {
            "1": expected in ranked[:1],
            "2": expected in ranked[:2],
            "3": expected in ranked[:3],
            "5": expected in ranked[:5],
        }
        # Audit only: do not feed request/messages to the policy.
        request_text = json.dumps(event.get("request") or {}, ensure_ascii=False)
        if f"You are {expected}" in request_text or f"你是 {expected}" in request_text:
            forbidden_current_text_hits += 1
        timing_records.append(
            {
                "dataset_name": dataset_name,
                "file_name": path.name,
                "workflow_id": str(event.get("workflow_id") or path.stem),
                "step_index": step_index,
                "event_id": str(event.get("event_id") or ""),
                "expected_agent_id": expected,
                "prediction_target": prediction_target,
                "current_agent_id": prediction_metadata["current_agent_id"],
                "event_type": prediction_metadata["event_type"],
                "expected_event_type": expected_event_type,
                "counted_for_metric": counted_for_metric,
                "prediction": ranked[:5],
                "base_prediction": base_ranked[:5],
                "candidate_count": len(prediction_metadata.get("candidate_agents") or agents),
                "prediction_time_ms": prediction_time_ms,
                "history_size_before_prediction": len(state.sequence),
                "global_bigram_size_before_prediction": len(global_memory.bigram),
                "ranking_reason": prediction_metadata["ranking_reason"],
                "graph_source": prediction_metadata["graph_source"],
                "outgoing_agents": prediction_metadata["outgoing_agents"],
                "semantic_profile_scores": prediction_metadata.get("semantic_profile_scores", {}),
                "memory_profile_scores": prediction_metadata.get("memory_profile_scores", {}),
                "prompt_profile_scores": prediction_metadata.get("prompt_profile_scores", {}),
                "task_idf_profile_scores": prediction_metadata.get("task_idf_profile_scores", {}),
                "memory_idf_profile_scores": prediction_metadata.get("memory_idf_profile_scores", {}),
                "prompt_idf_profile_scores": prediction_metadata.get("prompt_idf_profile_scores", {}),
                "schedule_prior_scores": prediction_metadata.get("schedule_prior_scores", {}),
                "meta_prior_scores": prediction_metadata.get("meta_prior_scores", {}),
                "adaptive_cross_file_scores": prediction_metadata.get(
                    "adaptive_cross_file_scores",
                    {},
                ),
                "adaptive_cross_file_profile_stability": prediction_metadata.get(
                    "adaptive_cross_file_profile_stability",
                    0.0,
                ),
                "visible_context_scores": prediction_metadata.get("visible_context_scores", {}),
                "pair_calibration_signature": prediction_metadata.get("pair_calibration_signature", []),
                "pair_calibration_counts": prediction_metadata.get("pair_calibration_counts", {}),
                "reranker_scores": {
                    agent_id: reranker_scores.get(agent_id, 0.0)
                    for agent_id in ranked[:5]
                },
                "hit_at_k": hit_at,
                "base_top_scores": {
                    agent_id: (
                        scores.get(agent_id, 0.0)
                        if prediction_target == "current_event"
                        else prediction_metadata.get("base_scores", {}).get(agent_id, 0.0)
                    )
                    for agent_id in base_ranked[:5]
                },
                "top_scores": {
                    agent_id: scores.get(agent_id, 0.0)
                    for agent_id in ranked[:5]
                },
            }
        )
        if prediction_target == "current_event":
            state.update_after_prediction(event, expected)
            sequence.append(expected)
        else:
            current_agent = str(event.get("agent_id") or "")
            state.update_after_prediction(event, current_agent)
            sequence.append(current_agent)
            next_state.update_after_transition(
                event=event,
                event_type=prediction_metadata["event_type"],
                observed_next_agent=expected,
                outgoing_agents=prediction_metadata["outgoing_agents"],
                learn_transition=counted_for_metric,
            )
            if (
                next_reranker is not None
                and prediction_metadata["event_type"] == "main_turn"
                and ranked
                and counted_for_metric
            ):
                next_reranker.update(
                    expected_agent=expected,
                    predicted_agent=ranked[0],
                    features_by_agent=reranker_features_by_agent,
                )
            if prediction_metadata["event_type"] == "main_turn" and counted_for_metric:
                pair_signature = prediction_metadata.get("pair_calibration_signature") or []
                if len(pair_signature) == 2:
                    pair_calibration_counts[(pair_signature[0], pair_signature[1])][
                        expected == pair_signature[1]
                    ] += 1
                next_global_memory.update(
                    current_agent=current_agent,
                    observed_next_agent=expected,
                    outgoing_agents=prediction_metadata["outgoing_agents"],
                    round_index=len(next_state.round_main_agents) - 1,
                )

    audit = {
        "file_name": path.name,
        "event_count": len(events),
        "decision_step_count": (
            sum(1 for record in timing_records if record.get("counted_for_metric"))
            if prediction_target == "next_agent"
            else len(timing_records)
        ),
        "agent_count": len(agents),
        "agents": agents,
        "prediction_target": prediction_target,
        "request_messages_not_used": prediction_target == "current_event",
        "current_agent_id_used_as_visible_input": prediction_target == "next_agent",
        "current_agent_id_used_only_as_label_after_prediction": prediction_target == "current_event",
        "prediction_transition_candidates_present": False,
        "graph_transition_candidates_present": False,
        "visible_agent_context_used": include_visible_agent_context,
        "visible_agent_context_scope": (
            "current event agents[*].context snapshot only; no next event context"
            if include_visible_agent_context
            else "not used"
        ),
        "first_main_target": next_state.local_first_main_target,
        "second_main_target": next_state.local_second_main_target,
        "forbidden_current_request_text_contains_expected_agent_count": forbidden_current_text_hits,
        "first_event_has_empty_history": bool(timing_records and timing_records[0]["history_size_before_prediction"] == 0),
        "_agent_profile_texts": {
            agent_id: str(agent.get("profile") or "")
            for agent_id, agent in first_agents.items()
            if isinstance(agent, dict)
        },
    }
    return timing_records, sequence, audit


def _summarize(records: list[dict[str, Any]], *, dataset_name: str, file_count: int) -> dict[str, Any]:
    total = len(records)
    hit_counts = {
        str(hit_k): sum(
            1 for record in records if record["hit_at_k"].get(str(hit_k), False)
        )
        for hit_k in (1, 2, 3, 5)
    }
    base_hit_counts = {
        str(hit_k): sum(
            1
            for record in records
            if record["expected_agent_id"] in (record.get("base_prediction") or [])[:hit_k]
        )
        for hit_k in (1, 2, 3, 5)
    }
    return {
        "dataset_name": dataset_name,
        "file_count": file_count,
        "total_steps": total,
        "hit_counts": hit_counts,
        "hit_at_k": {
            key: (value / total if total else 0.0)
            for key, value in hit_counts.items()
        },
        "base_hit_counts": base_hit_counts,
        "base_hit_at_k": {
            key: (value / total if total else 0.0)
            for key, value in base_hit_counts.items()
        },
        "mean_candidate_count": (
            sum(float(record["candidate_count"]) for record in records) / total
            if total
            else 0.0
        ),
        **_latency_summary(records, "prediction_time_ms"),
    }


def _primary_records(
    records: list[dict[str, Any]],
    *,
    prediction_target: str,
) -> list[dict[str, Any]]:
    if prediction_target == "next_agent":
        return [
            record
            for record in records
            if record.get("counted_for_metric")
        ]
    return records


def evaluate_logs(
    log_root: Path,
    *,
    use_cross_file_memory: bool,
    cross_file_stat_weight: float,
    prediction_target: str,
    enable_online_reranker: bool,
    reranker_learning_rate: float,
    reranker_base_score_scale: float,
    enable_adaptive_cross_file_prior: bool,
    adaptive_cross_file_weight: float,
    adaptive_cross_file_min_support: int,
    adaptive_cross_file_min_confidence: float,
    adaptive_cross_file_min_profile_stability: float,
    enable_research_schedule_prior: bool,
    enable_research_meta_prior: bool,
    enable_idf_profile_prior: bool,
    enable_online_pair_calibration: bool,
    pair_calibration_margin: int,
    include_visible_agent_context: bool,
    visible_context_similarity_weight: float,
    visible_context_length_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = _discover_raw_event_logs(log_root)
    global_memory = OnlinePatternMemory()
    next_global_memory = NextAgentGlobalMemory()
    shared_next_reranker = (
        OnlineNextAgentReranker(
            learning_rate=reranker_learning_rate,
            base_score_scale=reranker_base_score_scale,
        )
        if enable_online_reranker
        else None
    )
    all_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    dataset_name = "new_research_logs"
    for file_index, path in enumerate(paths):
        next_reranker = (
            shared_next_reranker
            if use_cross_file_memory
            else OnlineNextAgentReranker(
                learning_rate=reranker_learning_rate,
                base_score_scale=reranker_base_score_scale,
            )
            if enable_online_reranker
            else None
        )
        timing_records, sequence, audit = _evaluate_file(
            path,
            global_memory=global_memory,
            next_global_memory=next_global_memory,
            next_reranker=next_reranker,
            use_cross_file_memory=use_cross_file_memory,
            cross_file_stat_weight=cross_file_stat_weight,
            enable_adaptive_cross_file_prior=enable_adaptive_cross_file_prior,
            adaptive_cross_file_weight=adaptive_cross_file_weight,
            adaptive_cross_file_min_support=adaptive_cross_file_min_support,
            adaptive_cross_file_min_confidence=adaptive_cross_file_min_confidence,
            adaptive_cross_file_min_profile_stability=adaptive_cross_file_min_profile_stability,
            enable_research_schedule_prior=enable_research_schedule_prior,
            enable_research_meta_prior=enable_research_meta_prior,
            enable_idf_profile_prior=enable_idf_profile_prior,
            enable_online_pair_calibration=enable_online_pair_calibration,
            pair_calibration_margin=pair_calibration_margin,
            include_visible_agent_context=include_visible_agent_context,
            visible_context_similarity_weight=visible_context_similarity_weight,
            visible_context_length_weight=visible_context_length_weight,
            dataset_name=dataset_name,
            prediction_target=prediction_target,
        )
        for record in timing_records:
            record["file_index"] = file_index
            record["use_cross_file_memory"] = use_cross_file_memory
        all_records.extend(timing_records)
        audit["file_index"] = file_index
        audit["use_cross_file_memory"] = use_cross_file_memory
        profile_texts = audit.pop("_agent_profile_texts", {})
        audit_records.append(audit)
        if use_cross_file_memory:
            global_memory.update_sequence(sequence)
            next_global_memory.update_file_summary(
                audit.get("first_main_target"),
                profile_texts=profile_texts,
            )
    output_records = _primary_records(all_records, prediction_target=prediction_target)
    report = {
        "protocol": (
            f"{prediction_target}_online_across_files"
            if use_cross_file_memory
            else f"{prediction_target}_per_file_zero_online"
        ),
        "log_root": str(log_root),
        "input_view": (
            "static agents roster + previous observed agent sequence + previous observed parent edges; "
            "no current request/messages, no prediction field, no transition_candidates"
            if prediction_target == "current_event"
            else "current executing agent id + current request scheduling metadata + inferred prompt collaboration graph + "
            "previous observed transitions"
            + (
                " + current-event visible agents[*].context history snapshots"
                if include_visible_agent_context
                else ""
            )
            + (
            " + adaptive cross-file transition prior from completed queries"
                if enable_adaptive_cross_file_prior and use_cross_file_memory
                else ""
            )
            + (
                " + online cross-query start prior"
                if enable_research_meta_prior and use_cross_file_memory
                else ""
            )
            + (
                " + within-query online top1/top2 pair calibration"
                if enable_online_pair_calibration
                else ""
            )
            + "; no future event, no next label before scoring, no prediction field, no transition_candidates"
        ),
        "candidate_space": (
            "visible target_agent_id enum or inferred graph outgoing agents plus PLANNER"
            if prediction_target == "next_agent"
            else "all agents in the current file's static agents roster"
        ),
        "prediction_target": prediction_target,
        "online_reranker_enabled": enable_online_reranker,
        "online_reranker_learning_rate": reranker_learning_rate if enable_online_reranker else 0.0,
        "online_reranker_base_score_scale": (
            reranker_base_score_scale if enable_online_reranker else 0.0
        ),
        "cross_file_stat_weight": cross_file_stat_weight if use_cross_file_memory else 0.0,
        "adaptive_cross_file_prior_enabled": (
            enable_adaptive_cross_file_prior if use_cross_file_memory else False
        ),
        "adaptive_cross_file_weight": (
            adaptive_cross_file_weight
            if use_cross_file_memory and enable_adaptive_cross_file_prior
            else 0.0
        ),
        "adaptive_cross_file_min_support": (
            adaptive_cross_file_min_support
            if use_cross_file_memory and enable_adaptive_cross_file_prior
            else 0
        ),
        "adaptive_cross_file_min_confidence": (
            adaptive_cross_file_min_confidence
            if use_cross_file_memory and enable_adaptive_cross_file_prior
            else 0.0
        ),
        "adaptive_cross_file_min_profile_stability": (
            adaptive_cross_file_min_profile_stability
            if use_cross_file_memory and enable_adaptive_cross_file_prior
            else 0.0
        ),
        "visible_order_prior_enabled": enable_research_schedule_prior,
        "cross_query_start_prior_enabled": enable_research_meta_prior,
        "idf_profile_prior_enabled": enable_idf_profile_prior,
        "online_pair_calibration_enabled": enable_online_pair_calibration,
        "pair_calibration_margin": (
            pair_calibration_margin if enable_online_pair_calibration else 0
        ),
        "visible_agent_context_enabled": include_visible_agent_context,
        "visible_context_similarity_weight": (
            visible_context_similarity_weight if include_visible_agent_context else 0.0
        ),
        "visible_context_length_weight": (
            visible_context_length_weight if include_visible_agent_context else 0.0
        ),
        "metric_scope": (
            "main_turn decision-only speculative prediction where the next event is agent work; "
            "continuation/planner recovery events are not reported or learned as predictive transitions"
            if prediction_target == "next_agent"
            else "all current_event steps"
        ),
        "summary": _summarize(output_records, dataset_name=dataset_name, file_count=len(paths)),
        "files": audit_records,
    }
    return report, output_records, audit_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cold-start next-agent prediction from raw new JSONL logs without prediction fields."
    )
    parser.add_argument("--log-root", type=Path, default=Path("results/new"))
    parser.add_argument("--report-path", type=Path, default=Path("results/new_cold_start_report.json"))
    parser.add_argument("--timing-path", type=Path, default=Path("results/new_cold_start_timing.jsonl"))
    parser.add_argument("--audit-path", type=Path, default=Path("results/new_cold_start_audit.json"))
    parser.add_argument(
        "--use-cross-file-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, keep online pattern memory across files in sorted order.",
    )
    parser.add_argument(
        "--cross-file-stat-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for raw cross-file transition-count priors. The default keeps "
            "cross-file learning in the online reranker but avoids applying noisy "
            "global count priors directly."
        ),
    )
    parser.add_argument(
        "--prediction-target",
        choices=("current_event", "next_agent"),
        default="current_event",
        help=(
            "current_event predicts the current event agent from prior history only. "
            "next_agent assumes the current executing agent is visible and predicts the next event agent."
        ),
    )
    parser.add_argument(
        "--enable-online-reranker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For next_agent main_turn steps, apply a strict online perceptron reranker. "
            "It updates only after each prediction is scored."
        ),
    )
    parser.add_argument(
        "--enable-adaptive-cross-file-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For next_agent main_turn steps, use completed-query transition memory only "
            "when the current source agent has enough stable cross-file evidence."
        ),
    )
    parser.add_argument(
        "--adaptive-cross-file-weight",
        type=float,
        default=24.0,
        help="Maximum scale for the confidence-gated cross-file transition prior.",
    )
    parser.add_argument(
        "--adaptive-cross-file-min-support",
        type=int,
        default=4,
        help="Minimum completed-query observations before a cross-file prior can fire.",
    )
    parser.add_argument(
        "--adaptive-cross-file-min-confidence",
        type=float,
        default=0.60,
        help="Minimum dominant transition ratio before a cross-file prior can fire.",
    )
    parser.add_argument(
        "--adaptive-cross-file-min-profile-stability",
        type=float,
        default=0.35,
        help=(
            "Minimum same-agent profile similarity to previous completed queries before "
            "agent-id cross-file memory is trusted."
        ),
    )
    parser.add_argument(
        "--enable-visible-order-prior",
        dest="enable_visible_order_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a generic visible-order prior over candidate agents. It uses only "
            "the current visible candidate list and already observed source-turn count."
        ),
    )
    parser.add_argument(
        "--enable-cross-query-start-prior",
        dest="enable_cross_query_start_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply an online cross-query start prior: reuse the previous completed "
            "query's first predictive target when cross-file memory is enabled, and "
            "lightly reuse the current query's already observed second predictive target."
        ),
    )
    parser.add_argument(
        "--enable-idf-profile-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optionally add candidate-local IDF profile matching over visible task/memory/prompt "
            "text. This is disabled by default because it can improve top-k while hurting hit@1."
        ),
    )
    parser.add_argument(
        "--enable-online-pair-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Within each query, learn whether the current top1/top2 pair should be "
            "swapped from previous scored decisions. Updates happen only after scoring."
        ),
    )
    parser.add_argument(
        "--pair-calibration-margin",
        type=int,
        default=1,
        help="Swap top1/top2 only when historical second-place wins exceed first-place wins by this margin.",
    )
    parser.add_argument(
        "--include-visible-agent-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For next_agent, include current-event agents[*].context as already completed "
            "history. This never reads the next event context or current output."
        ),
    )
    parser.add_argument(
        "--visible-context-similarity-weight",
        type=float,
        default=10.0,
        help="Weight for candidate context similarity to the current visible memory.",
    )
    parser.add_argument(
        "--visible-context-length-weight",
        type=float,
        default=0.0,
        help="Optional weight for candidate historical context length.",
    )
    parser.add_argument("--reranker-learning-rate", type=float, default=0.03)
    parser.add_argument("--reranker-base-score-scale", type=float, default=0.1)
    args = parser.parse_args()

    report, timing_records, audit_records = evaluate_logs(
        args.log_root,
        use_cross_file_memory=args.use_cross_file_memory,
        cross_file_stat_weight=args.cross_file_stat_weight,
        prediction_target=args.prediction_target,
        enable_online_reranker=args.enable_online_reranker,
        reranker_learning_rate=args.reranker_learning_rate,
        reranker_base_score_scale=args.reranker_base_score_scale,
        enable_adaptive_cross_file_prior=args.enable_adaptive_cross_file_prior,
        adaptive_cross_file_weight=args.adaptive_cross_file_weight,
        adaptive_cross_file_min_support=args.adaptive_cross_file_min_support,
        adaptive_cross_file_min_confidence=args.adaptive_cross_file_min_confidence,
        adaptive_cross_file_min_profile_stability=args.adaptive_cross_file_min_profile_stability,
        enable_research_schedule_prior=args.enable_visible_order_prior,
        enable_research_meta_prior=args.enable_cross_query_start_prior,
        enable_idf_profile_prior=args.enable_idf_profile_prior,
        enable_online_pair_calibration=args.enable_online_pair_calibration,
        pair_calibration_margin=args.pair_calibration_margin,
        include_visible_agent_context=args.include_visible_agent_context,
        visible_context_similarity_weight=args.visible_context_similarity_weight,
        visible_context_length_weight=args.visible_context_length_weight,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.timing_path.parent.mkdir(parents=True, exist_ok=True)
    args.timing_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in timing_records) + "\n",
        encoding="utf-8",
    )
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.write_text(json.dumps(audit_records, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report["summary"]
    hit_at_k = summary["hit_at_k"]
    base_hit_at_k = summary.get("base_hit_at_k", {})
    print(f"protocol={report['protocol']}")
    print(f"log_root={args.log_root}")
    print(f"files={summary['file_count']} steps={summary['total_steps']}")
    print(
        f"hit@1={hit_at_k.get('1', 0.0):.4f} "
        f"hit@2={hit_at_k.get('2', 0.0):.4f} "
        f"hit@3={hit_at_k.get('3', 0.0):.4f} "
        f"hit@5={hit_at_k.get('5', 0.0):.4f}"
    )
    if args.enable_online_reranker and args.prediction_target == "next_agent":
        print(
            f"base_hit@1={base_hit_at_k.get('1', 0.0):.4f} "
            f"base_hit@2={base_hit_at_k.get('2', 0.0):.4f} "
            f"base_hit@3={base_hit_at_k.get('3', 0.0):.4f}"
        )
    print(
        f"prediction_ms_mean={summary['prediction_time_ms_mean']:.4f} "
        f"prediction_ms_p95={summary['prediction_time_ms_p95']:.4f}"
    )
    print(f"report={args.report_path}")
    print(f"timing={args.timing_path}")
    print(f"audit={args.audit_path}")


if __name__ == "__main__":
    main()
