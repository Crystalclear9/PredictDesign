from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign import (  # noqa: E402
    ExperimentConfig,
    GraphActionType,
    GraphPredictionContext,
    Message,
    PredictedGraphAction,
    PredictDesignSystem,
    TemporalNode,
)


SCHEDULER_ID = "SCHEDULER"


@dataclass(slots=True)
class OnlineStep:
    scenario: str
    query_id: str
    bucket: str
    time_value: float
    context: GraphPredictionContext
    expected_actions: list[PredictedGraphAction]


@dataclass(slots=True)
class XAIRecord:
    query_id: str
    expected: str
    predicted: str
    correct: bool
    total_score: float
    graph_residual_score: float
    edge_logit_with_priors: float
    zero_shot_prior: float
    zero_shot_contribution: float
    few_shot_prior: float
    few_shot_contribution: float
    runtime_score: float


def _sync_device(device: str | torch.device) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def _time_call(device: str | torch.device, fn):
    _sync_device(device)
    start = time.perf_counter()
    value = fn()
    _sync_device(device)
    return value, (time.perf_counter() - start) * 1000.0


def _resolve_device(requested_device: str, *, require_cuda: bool = False) -> str:
    requested = str(requested_device or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if require_cuda:
            raise RuntimeError(
                "--require-cuda was set, but this Python environment is using a CPU-only torch build."
            )
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment is using a CPU-only torch build."
        )
    return requested_device


def _vector(text: str, dim: int) -> list[float]:
    values = [0.0 for _ in range(dim)]
    for token in str(text or "").lower().split()[:128]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign
    norm = sum(value * value for value in values) ** 0.5
    if norm > 0:
        values = [value / norm for value in values]
    return values


def _compact(text: str, limit: int = 1400) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return data


def _relation_types() -> tuple[str, ...]:
    return (
        "activate",
        "communication",
        "delegate",
        "delegate_return",
        "retry",
        "review",
        "werewolf_vote",
        "werewolf_attack",
        "guard_action",
        "seer_check",
        "witch_save",
        "witch_poison",
    )


def _config(context_dim: int, hidden_dim: int, device: str) -> ExperimentConfig:
    return ExperimentConfig(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        gnn_type="gcn",
        device=device,
        sentence_transformer_path="__missing_vendor_online_model__",
        candidate_relation_types=_relation_types(),
        candidate_new_roles=("planner", "worker"),
        enable_add_node_prediction=False,
        use_context_conditioning=False,
        use_candidate_cross_encoder=False,
        use_structural_candidate_priors=True,
        use_zero_shot_action_priors=True,
        use_few_shot_transition_memory=True,
        use_online_few_shot_updates=True,
        learned_candidate_score_weight=0.0,
        candidate_text_score_weight=0.0,
        zero_shot_prior_weight=5.0,
        few_shot_memory_weight=6.0,
        context_source_bias_weight=4.0,
        runtime_message_candidate_score=0.0,
        no_directed_message_noop_bias=0.0,
        prediction_horizon=2,
    )


def _agent_order(payload: dict[str, Any]) -> list[str]:
    return [str(agent.get("agent_id")) for agent in payload.get("agents", []) if agent.get("agent_id")]


def _agent_profile(payload: dict[str, Any], agent_id: str) -> str:
    for agent in payload.get("agents", []):
        if str(agent.get("agent_id")) == agent_id:
            return _compact(str(agent.get("profile") or agent.get("type") or agent_id))
    return agent_id


def _agent_role(payload: dict[str, Any], agent_id: str) -> str:
    for agent in payload.get("agents", []):
        if str(agent.get("agent_id")) == agent_id:
            role = str(agent.get("type") or "").strip()
            return role or "agent"
    return "agent"


def _coding_research_steps(
    payloads: list[dict[str, Any]],
    *,
    scene_config: dict[str, Any],
    scenario: str,
) -> list[OnlineStep]:
    steps: list[OnlineStep] = []
    time_value = 1.0
    order = _agent_order(scene_config)
    if not order:
        return steps
    config_task = _compact(str((scene_config.get("task") or {}).get("content") or ""))
    for query_index, payload in enumerate(payloads, start=1):
        task_text = _compact(
            str((payload.get("task") or {}).get("content") or config_task)
        )
        context = GraphPredictionContext(
            source_node_id=SCHEDULER_ID,
            query_text=task_text,
            graph_profile_text="",
            candidate_actions=[],
            metadata={},
        ).query_time_view(include_candidate_actions=False)
        steps.append(
            OnlineStep(
                scenario=scenario,
                query_id=str(payload.get("task_id") or query_index),
                bucket="query_only",
                time_value=time_value,
                context=context,
                expected_actions=[
                    PredictedGraphAction(
                        action_type=GraphActionType.CREATE_EDGE,
                        score=1.0,
                        effective_time=time_value,
                        source_node_id=SCHEDULER_ID,
                        target_node_id=order[0],
                        relation_type="activate",
                    )
                ],
            )
        )
        time_value += 1.0
    return steps


def _parse_werewolf_roles(config_path: Path) -> list[str]:
    roles: list[str] = []
    in_roles = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "roles:":
            in_roles = True
            continue
        if in_roles and line.startswith("- "):
            roles.append(line[2:].strip())
            continue
        if in_roles and line and not line.startswith("#"):
            break
    return roles


def _werewolf_steps(
    roles: list[str],
    *,
    limit: int,
    prompt_dir: Path,
) -> list[OnlineStep]:
    agent_ids = [f"agent{index}" for index in range(1, len(roles) + 1)]
    role_by_agent = dict(zip(agent_ids, roles))
    role_map: dict[str, list[str]] = {}
    for agent_id, role in role_by_agent.items():
        role_map.setdefault(role, []).append(agent_id)
    stages = [
        ("wolf", _prompt_text(prompt_dir / "werewolf_action.yaml")),
        ("seer", _prompt_text(prompt_dir / "seer_prompt.yaml")),
        ("witch", _prompt_text(prompt_dir / "witch_prompt.yaml")),
        ("guard", _prompt_text(prompt_dir / "guard_prompt.yaml")),
    ]
    steps: list[OnlineStep] = []
    time_value = 1.0
    while len(steps) < limit:
        for role, query_text in stages:
            targets = role_map.get(role, [])
            if not targets:
                continue
            steps.append(
                _werewolf_step(
                    time_value=time_value,
                    query_id=f"night-{int(time_value)}",
                    bucket="query_only",
                    query_text=query_text,
                    source_node_id=SCHEDULER_ID,
                    expected_targets=targets,
                )
            )
            time_value += 1.0
            if len(steps) >= limit:
                return steps
    return steps


def _prompt_text(path: Path) -> str:
    data = _load_yaml(path)
    return _compact(str(data.get("user") or data.get("system") or path.stem))


def _werewolf_step(
    *,
    time_value: float,
    query_id: str,
    bucket: str,
    query_text: str,
    source_node_id: str,
    expected_targets: list[str],
) -> OnlineStep:
    context = GraphPredictionContext(
        source_node_id=source_node_id,
        query_text=query_text,
        graph_profile_text="",
        candidate_actions=[],
        metadata={},
    ).query_time_view(include_candidate_actions=False)
    return OnlineStep(
        scenario="werewolf",
        query_id=query_id,
        bucket=bucket,
        time_value=time_value,
        context=context,
        expected_actions=[
            PredictedGraphAction(
                action_type=GraphActionType.CREATE_EDGE,
                score=1.0,
                effective_time=time_value,
                source_node_id=source_node_id,
                target_node_id=target,
                relation_type="activate",
            )
            for target in expected_targets
        ],
    )


def _build_system_for_config(
    scene_config: dict[str, Any],
    *,
    config: ExperimentConfig,
) -> PredictDesignSystem:
    profile_by_agent: dict[str, str] = {}
    role_by_agent: dict[str, str] = {}
    structural_edges: list[tuple[str, str]] = []
    order = _agent_order(scene_config)
    for agent_id in order:
        profile_by_agent[agent_id] = _agent_profile(scene_config, agent_id)
        role_by_agent[agent_id] = _agent_role(scene_config, agent_id)
    for source, target, _ in scene_config.get("relationships", []) or []:
        source_id = str(source)
        target_id = str(target)
        if source_id != target_id:
            structural_edges.append((source_id, target_id))
    nodes = [
        _node(
            SCHEDULER_ID,
            "scheduler",
            "Runtime scheduler that activates the next agent.",
            config.context_dim,
            device=config.device,
        )
    ]
    for agent_id in order:
        nodes.append(
            _node(
                agent_id,
                role_by_agent.get(agent_id, "agent"),
                profile_by_agent.get(agent_id, agent_id),
                config.context_dim,
                device=config.device,
            )
        )
    system = PredictDesignSystem(config=config)
    all_structural_edges = [(SCHEDULER_ID, agent_id) for agent_id in order]
    all_structural_edges.extend(structural_edges)
    system.initialize_graph(nodes=nodes, structural_edges=all_structural_edges)
    return system


def _build_werewolf_system(roles: list[str], *, config: ExperimentConfig) -> PredictDesignSystem:
    nodes = [
        _node(
            SCHEDULER_ID,
            "scheduler",
            "Runtime scheduler that activates the next werewolf actor.",
            config.context_dim,
            device=config.device,
        )
    ]
    for index, role in enumerate(roles, start=1):
        nodes.append(
            _node(
                f"agent{index}",
                role,
                f"Werewolf player role={role}",
                config.context_dim,
                device=config.device,
            )
        )
    system = PredictDesignSystem(config=config)
    structural_edges = [(SCHEDULER_ID, node.node_id) for node in nodes if node.node_id != SCHEDULER_ID]
    system.initialize_graph(nodes=nodes, structural_edges=structural_edges)
    return system


def _node(
    node_id: str,
    role: str,
    text: str,
    context_dim: int,
    *,
    device: str | torch.device,
) -> TemporalNode:
    node = TemporalNode.build(
        node_id=node_id,
        role=role,
        context=_vector(f"{node_id} {role} {text}", context_dim),
        context_dim=context_dim,
        device=device,
    )
    node.context_text = _compact(text)
    return node


def _evaluate_steps(
    system: PredictDesignSystem,
    steps: list[OnlineStep],
    *,
    hit_ks: tuple[int, ...] = (1, 2, 3),
    collect_xai: bool = False,
    speculative_steps: int = 2,
    latency_warmup_steps: int = 2,
    include_timing_records: bool = True,
    measure_latency: bool = True,
) -> dict[str, Any]:
    totals = {k: 0 for k in hit_ks}
    hits = {k: 0 for k in hit_ks}
    bucket_totals: dict[str, int] = {}
    bucket_hits: dict[str, int] = {}
    examples: list[str] = []
    xai_records: list[XAIRecord] = []
    timing_records: list[dict[str, Any]] = []
    updates = 0
    system.eval()
    with torch.no_grad():
        for step in steps:
            _assert_query_only_context(step.context)
            if measure_latency:
                predictions, prediction_time_ms = _time_call(
                    system.device,
                    lambda: system.predictor.predict_action_set(
                        temporal_graph=system.temporal_graph,
                        ctdg=system.ctdg,
                        observation_time=step.time_value,
                        prediction_context=step.context,
                    ),
                )
            else:
                predictions = system.predictor.predict_action_set(
                    temporal_graph=system.temporal_graph,
                    ctdg=system.ctdg,
                    observation_time=step.time_value,
                    prediction_context=step.context,
                )
                prediction_time_ms = 0.0
            rollout_action_count = 0
            speculative_rollout_time_ms = 0.0
            speculative_actions_by_step: list[list[PredictedGraphAction]] = []
            if measure_latency and speculative_steps > 0:
                rollout, speculative_rollout_time_ms = _time_call(
                    system.device,
                    lambda: system.predict_speculative_action_sets(
                        observation_time=step.time_value,
                        steps=speculative_steps,
                        prediction_context=step.context,
                    ),
                )
                speculative_actions_by_step = rollout.actions_by_step
                rollout_action_count = sum(len(actions) for actions in speculative_actions_by_step)
            if collect_xai:
                xai_records.append(_xai_record(system, step, predictions[:1]))
            for k in hit_ks:
                totals[k] += 1
                if _window_matches(predictions[:k], step.expected_actions):
                    hits[k] += 1
            bucket_totals[step.bucket] = bucket_totals.get(step.bucket, 0) + 1
            if _window_matches(predictions[:1], step.expected_actions):
                bucket_hits[step.bucket] = bucket_hits.get(step.bucket, 0) + 1
            if len(examples) < 4:
                expected = ",".join(action.target_node_id or "" for action in step.expected_actions)
                top = predictions[0] if predictions else None
                top_text = f"{top.source_node_id}->{top.target_node_id}:{top.relation_type}" if top else "none"
                examples.append(f"{step.query_id}/{step.bucket}: expected={expected} top1={top_text}")
            observed_actions = step.expected_actions[:1]
            if measure_latency:
                added_updates, memory_update_time_ms = _time_call(
                    system.device,
                    lambda: system.record_observed_actions(
                        observed_actions,
                        prediction_context=step.context,
                    ),
                )
                _, state_update_time_ms = _time_call(
                    system.device,
                    lambda: [
                        _ingest_observed_message(system, observed, step)
                        for observed in observed_actions
                    ],
                )
            else:
                added_updates = system.record_observed_actions(
                    observed_actions,
                    prediction_context=step.context,
                )
                for observed in observed_actions:
                    _ingest_observed_message(system, observed, step)
                memory_update_time_ms = 0.0
                state_update_time_ms = 0.0
            updates += int(added_updates)
            prediction_action_count = max(len(predictions), 1)
            rollout_action_denominator = max(rollout_action_count, 1)
            timing_records.append(
                {
                    "query_id": step.query_id,
                    "bucket": step.bucket,
                    "expected_actions": [_action_text(action) for action in step.expected_actions],
                    "predicted_actions": [
                        {
                            "rank": rank,
                            "action": _action_text(action),
                            "score": float(action.score),
                            "matches_expected": any(
                                _actions_match(action, expected)
                                for expected in step.expected_actions
                            ),
                            "prediction_time_ms": prediction_time_ms,
                            "amortized_prediction_time_ms": (
                                prediction_time_ms / prediction_action_count
                            ),
                        }
                        for rank, action in enumerate(predictions, start=1)
                    ],
                    "speculative_actions_by_step": [
                        [
                            {
                                "rank": rank,
                                "action": _action_text(action),
                                "score": float(action.score),
                                "amortized_speculative_rollout_time_ms": (
                                    speculative_rollout_time_ms / rollout_action_denominator
                                ),
                            }
                            for rank, action in enumerate(actions, start=1)
                        ]
                        for actions in speculative_actions_by_step
                    ],
                    "prediction_action_count": len(predictions),
                    "prediction_time_ms": prediction_time_ms,
                    "amortized_prediction_action_time_ms": (
                        prediction_time_ms / prediction_action_count
                    ),
                    "speculative_rollout_steps": speculative_steps,
                    "speculative_rollout_action_count": rollout_action_count,
                    "speculative_rollout_time_ms": speculative_rollout_time_ms,
                    "amortized_speculative_action_time_ms": (
                        speculative_rollout_time_ms / rollout_action_denominator
                    ),
                    "memory_update_time_ms": memory_update_time_ms,
                    "state_update_time_ms": state_update_time_ms,
                    "total_single_step_overhead_ms": (
                        prediction_time_ms + memory_update_time_ms + state_update_time_ms
                    ),
                    "total_speculative_overhead_ms": (
                        speculative_rollout_time_ms + memory_update_time_ms + state_update_time_ms
                    ),
                }
            )
    timing_summary = _summarize_timing(
        timing_records,
        latency_warmup_steps=latency_warmup_steps,
    )
    return {
        "steps": len(steps),
        "hit": {f"hit@{k}": hits[k] / totals[k] if totals[k] else 0.0 for k in hit_ks},
        "bucket_hit@1": {
            bucket: bucket_hits.get(bucket, 0) / total if total else 0.0
            for bucket, total in sorted(bucket_totals.items())
        },
        "online_memory_updates": updates,
        "examples": examples,
        "xai": _summarize_xai(xai_records) if collect_xai else {},
        "timing": timing_summary,
        "timing_records": timing_records if include_timing_records else [],
    }


def _xai_record(
    system: PredictDesignSystem,
    step: OnlineStep,
    predictions: list[PredictedGraphAction],
) -> XAIRecord:
    predicted = predictions[0] if predictions else None
    expected = step.expected_actions[0] if step.expected_actions else None
    score_parts = _score_parts_for_action(system, step, predicted)
    return XAIRecord(
        query_id=step.query_id,
        expected=_action_text(expected),
        predicted=_action_text(predicted),
        correct=predicted is not None
        and expected is not None
        and _actions_match(predicted, expected),
        total_score=float(predicted.score) if predicted is not None else 0.0,
        graph_residual_score=score_parts["graph_residual_score"],
        edge_logit_with_priors=score_parts["edge_logit_with_priors"],
        zero_shot_prior=score_parts["zero_shot_prior"],
        zero_shot_contribution=score_parts["zero_shot_contribution"],
        few_shot_prior=score_parts["few_shot_prior"],
        few_shot_contribution=score_parts["few_shot_contribution"],
        runtime_score=score_parts["runtime_score"],
    )


def _score_parts_for_action(
    system: PredictDesignSystem,
    step: OnlineStep,
    action: PredictedGraphAction | None,
) -> dict[str, float]:
    empty = {
        "graph_residual_score": 0.0,
        "edge_logit_with_priors": 0.0,
        "zero_shot_prior": 0.0,
        "zero_shot_contribution": 0.0,
        "few_shot_prior": 0.0,
        "few_shot_contribution": 0.0,
        "runtime_score": 0.0,
    }
    if action is None or action.action_type != GraphActionType.CREATE_EDGE:
        return empty
    bundle = system.predictor.score_action_space(
        temporal_graph=system.temporal_graph,
        ctdg=system.ctdg,
        observation_time=step.time_value,
        prediction_context=step.context,
    )
    if action.source_node_id not in bundle.node_order or action.target_node_id not in bundle.node_order:
        return empty
    row = bundle.node_order.index(str(action.source_node_id))
    col = bundle.node_order.index(str(action.target_node_id))
    edge_logit = float(bundle.create_scores[row, col].detach().cpu().item())
    if action.relation_type in system.config.candidate_relation_types and bundle.relation_logits.numel() > 0:
        relation_index = system.config.candidate_relation_types.index(str(action.relation_type))
        edge_logit += float(bundle.relation_logits[row, col, relation_index].detach().cpu().item())
    zero_shot_prior = 0.0
    if system.config.use_zero_shot_action_priors:
        zero_matrix = system.predictor.cold_start_prior_scorer.edge_prior_matrix(
            temporal_graph=system.temporal_graph,
            prediction_context=step.context,
            node_order=bundle.node_order,
            device=bundle.create_scores.device,
            dtype=bundle.create_scores.dtype,
        )
        zero_shot_prior = float(zero_matrix[row, col].detach().cpu().item())
    few_shot_prior = 0.0
    if len(system.predictor.few_shot_memory) > 0:
        few_matrix = system.predictor.few_shot_memory.edge_prior_matrix(
            temporal_graph=system.temporal_graph,
            prediction_context=step.context,
            node_order=bundle.node_order,
            device=bundle.create_scores.device,
            dtype=bundle.create_scores.dtype,
        )
        few_shot_prior = float(few_matrix[row, col].detach().cpu().item())
    zero_shot_contribution = system.config.zero_shot_prior_weight * zero_shot_prior
    few_shot_contribution = system.config.few_shot_memory_weight * few_shot_prior
    residual = edge_logit - zero_shot_contribution - few_shot_contribution
    runtime_score = (
        system.config.runtime_message_candidate_score
        if str(action.metadata.get("source", "")) == "runtime_message"
        else 0.0
    )
    return {
        "graph_residual_score": residual,
        "edge_logit_with_priors": edge_logit,
        "zero_shot_prior": zero_shot_prior,
        "zero_shot_contribution": zero_shot_contribution,
        "few_shot_prior": few_shot_prior,
        "few_shot_contribution": few_shot_contribution,
        "runtime_score": runtime_score,
    }


def _summarize_xai(records: list[XAIRecord]) -> dict[str, Any]:
    if not records:
        return {}
    correct = [record for record in records if record.correct]
    wrong = [record for record in records if not record.correct]
    return {
        "record_count": len(records),
        "correct_count": len(correct),
        "wrong_count": len(wrong),
        "accuracy": len(correct) / len(records),
        "mean_total_score": _mean(record.total_score for record in records),
        "mean_graph_residual_score": _mean(record.graph_residual_score for record in records),
        "mean_edge_logit_with_priors": _mean(record.edge_logit_with_priors for record in records),
        "mean_zero_shot_prior": _mean(record.zero_shot_prior for record in records),
        "mean_weighted_zero_shot_contribution": _mean(
            record.zero_shot_contribution for record in records
        ),
        "mean_few_shot_prior": _mean(record.few_shot_prior for record in records),
        "mean_weighted_few_shot_contribution": _mean(
            record.few_shot_contribution for record in records
        ),
        "mean_runtime_score": _mean(record.runtime_score for record in records),
        "top_wrong_examples": [
            {
                "query_id": record.query_id,
                "expected": record.expected,
                "predicted": record.predicted,
                "total_score": record.total_score,
                "graph_residual_score": record.graph_residual_score,
                "edge_logit_with_priors": record.edge_logit_with_priors,
                "zero_shot_prior": record.zero_shot_prior,
                "zero_shot_contribution": record.zero_shot_contribution,
                "few_shot_prior": record.few_shot_prior,
                "few_shot_contribution": record.few_shot_contribution,
                "runtime_score": record.runtime_score,
            }
            for record in wrong[:8]
        ],
        "sample_correct_examples": [
            {
                "query_id": record.query_id,
                "expected": record.expected,
                "predicted": record.predicted,
                "total_score": record.total_score,
                "graph_residual_score": record.graph_residual_score,
                "edge_logit_with_priors": record.edge_logit_with_priors,
                "zero_shot_prior": record.zero_shot_prior,
                "zero_shot_contribution": record.zero_shot_contribution,
                "few_shot_prior": record.few_shot_prior,
                "few_shot_contribution": record.few_shot_contribution,
            }
            for record in correct[:8]
        ],
    }


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _latency_stats(values: list[float]) -> dict[str, float]:
    items = sorted(float(value) for value in values)
    if not items:
        return {
            "count": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "stdev_ms": 0.0,
        }
    return {
        "count": float(len(items)),
        "mean_ms": sum(items) / len(items),
        "median_ms": statistics.median(items),
        "p90_ms": _percentile(items, 0.90),
        "p95_ms": _percentile(items, 0.95),
        "p99_ms": _percentile(items, 0.99),
        "min_ms": items[0],
        "max_ms": items[-1],
        "stdev_ms": statistics.pstdev(items) if len(items) > 1 else 0.0,
    }


def _summarize_timing(
    records: list[dict[str, Any]],
    *,
    latency_warmup_steps: int,
) -> dict[str, Any]:
    measured = records[max(int(latency_warmup_steps), 0):]
    if not measured:
        measured = records

    def values(key: str) -> list[float]:
        return [float(record.get(key, 0.0)) for record in measured]

    total_prediction_ms = sum(values("prediction_time_ms"))
    total_speculative_ms = sum(values("speculative_rollout_time_ms"))
    total_single_step_overhead_ms = sum(values("total_single_step_overhead_ms"))
    total_speculative_overhead_ms = sum(values("total_speculative_overhead_ms"))
    total_prediction_actions = sum(
        int(record.get("prediction_action_count", 0)) for record in measured
    )
    total_speculative_actions = sum(
        int(record.get("speculative_rollout_action_count", 0)) for record in measured
    )
    return {
        "warmup_steps_excluded": min(max(int(latency_warmup_steps), 0), len(records)),
        "measured_steps": len(measured),
        "prediction_time": _latency_stats(values("prediction_time_ms")),
        "amortized_prediction_action_time": _latency_stats(
            values("amortized_prediction_action_time_ms")
        ),
        "speculative_rollout_time": _latency_stats(values("speculative_rollout_time_ms")),
        "amortized_speculative_action_time": _latency_stats(
            values("amortized_speculative_action_time_ms")
        ),
        "memory_update_time": _latency_stats(values("memory_update_time_ms")),
        "state_update_time": _latency_stats(values("state_update_time_ms")),
        "total_single_step_overhead": _latency_stats(values("total_single_step_overhead_ms")),
        "total_speculative_overhead": _latency_stats(values("total_speculative_overhead_ms")),
        "throughput": {
            "prediction_steps_per_second": (
                len(measured) * 1000.0 / total_prediction_ms
                if total_prediction_ms > 0
                else 0.0
            ),
            "speculative_rollouts_per_second": (
                len(measured) * 1000.0 / total_speculative_ms
                if total_speculative_ms > 0
                else 0.0
            ),
            "prediction_actions_per_second": (
                total_prediction_actions * 1000.0 / total_prediction_ms
                if total_prediction_ms > 0
                else 0.0
            ),
            "speculative_actions_per_second": (
                total_speculative_actions * 1000.0 / total_speculative_ms
                if total_speculative_ms > 0
                else 0.0
            ),
        },
        "totals": {
            "prediction_time_ms": total_prediction_ms,
            "speculative_rollout_time_ms": total_speculative_ms,
            "single_step_overhead_ms": total_single_step_overhead_ms,
            "speculative_overhead_ms": total_speculative_overhead_ms,
            "prediction_action_count": total_prediction_actions,
            "speculative_action_count": total_speculative_actions,
        },
        "speedup_accounting_hint": (
            "net_saved_time = avoided_downstream_runtime_ms - total_speculative_overhead_ms. "
            "For single-step prefetch decisions, use total_single_step_overhead_ms instead."
        ),
    }


def _action_text(action: PredictedGraphAction | None) -> str:
    if action is None:
        return "none"
    return f"{action.source_node_id}->{action.target_node_id}:{action.relation_type}"


def _context_without_query(context: GraphPredictionContext) -> GraphPredictionContext:
    return GraphPredictionContext(
        source_node_id=context.source_node_id,
        query_text="",
        graph_profile_text=context.graph_profile_text,
        source_output_text="",
        runtime_text="",
        candidate_actions=[],
        metadata={},
    )


def _clone_steps_without_query(steps: list[OnlineStep]) -> list[OnlineStep]:
    return [
        OnlineStep(
            scenario=step.scenario,
            query_id=step.query_id,
            bucket=step.bucket,
            time_value=step.time_value,
            context=_context_without_query(step.context),
            expected_actions=[
                PredictedGraphAction(
                    action_type=action.action_type,
                    score=action.score,
                    effective_time=action.effective_time,
                    source_node_id=action.source_node_id,
                    target_node_id=action.target_node_id,
                    relation_type=action.relation_type,
                    role=action.role,
                    new_node_id=action.new_node_id,
                    metadata=dict(action.metadata),
                )
                for action in step.expected_actions
            ],
        )
        for step in steps
    ]


def _apply_ablation(config: ExperimentConfig, ablation: str) -> None:
    if ablation == "full":
        return
    if ablation == "no_query":
        return
    if ablation == "no_zero_shot_prior":
        config.use_zero_shot_action_priors = False
        config.zero_shot_prior_weight = 0.0
        config.zero_shot_action_type_boost = 0.0
        return
    if ablation == "no_online_memory":
        config.use_few_shot_transition_memory = False
        config.use_online_few_shot_updates = False
        config.few_shot_memory_weight = 0.0
        return
    if ablation == "no_runtime_features":
        config.use_runtime_context_features = False
        return
    if ablation == "no_structural_prior":
        config.use_structural_candidate_priors = False
        config.candidate_structural_prior_weight = 0.0
        return
    raise ValueError(f"Unknown ablation: {ablation}")


def _build_runs(
    *,
    vendor_root: Path,
    queries: int,
    config: ExperimentConfig,
) -> list[tuple[str, int, PredictDesignSystem, list[OnlineStep]]]:
    marble_root = vendor_root / "benchmarks" / "marble"
    coding_rows = _load_jsonl(marble_root / "metadata" / "coding" / "coding_main.jsonl", queries)
    research_rows = _load_jsonl(marble_root / "metadata" / "research" / "research_main.jsonl", queries)
    coding_config = _load_yaml(marble_root / "core" / "configs" / "coding_config" / "coding_config.yaml")
    research_config = _load_yaml(marble_root / "core" / "configs" / "research_config" / "research_config.yaml")
    werewolf_config = _load_yaml(
        marble_root / "core" / "configs" / "werewolf_config" / "werewolf_config.yaml"
    )
    werewolf_roles = _parse_werewolf_roles(
        marble_root / "core" / "configs" / "werewolf_config" / "werewolf_config.yaml"
    )
    werewolf_roles = list(werewolf_config.get("roles") or werewolf_roles)
    return [
        (
            "coding",
            len(coding_rows),
            _build_system_for_config(coding_config, config=config),
            _coding_research_steps(
                coding_rows,
                scene_config=coding_config,
                scenario="coding",
            ),
        ),
        (
            "research",
            len(research_rows),
            _build_system_for_config(research_config, config=config),
            _coding_research_steps(
                research_rows,
                scene_config=research_config,
                scenario="research",
            ),
        ),
        (
            "werewolf",
            queries,
            _build_werewolf_system(werewolf_roles, config=config),
            _werewolf_steps(
                werewolf_roles,
                limit=queries,
                prompt_dir=marble_root / "core" / "agent" / "werewolf_prompts",
            ),
        ),
    ]


def _evaluate_ablation(
    *,
    vendor_root: Path,
    queries: int,
    base_config: ExperimentConfig,
    ablation: str,
    speculative_steps: int,
    latency_warmup_steps: int,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    _apply_ablation(config, ablation)
    scenario_metrics: dict[str, Any] = {}
    for scenario, _, system, steps in _build_runs(vendor_root=vendor_root, queries=queries, config=config):
        eval_steps = _clone_steps_without_query(steps) if ablation == "no_query" else steps
        scenario_metrics[scenario] = _evaluate_steps(
            system,
            eval_steps,
            speculative_steps=0,
            latency_warmup_steps=latency_warmup_steps,
            include_timing_records=False,
            measure_latency=False,
        )
    mean_hit1 = sum(
        item["hit"]["hit@1"] for item in scenario_metrics.values()
    ) / max(len(scenario_metrics), 1)
    return {
        "mean_hit@1": mean_hit1,
        "scenarios": scenario_metrics,
    }


def _interpret_xai(
    summaries: list[dict[str, Any]],
    ablations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mean_hit1 = sum(item["hit"]["hit@1"] for item in summaries) / max(len(summaries), 1)
    mean_zero_shot = _mean(
        item.get("xai", {}).get("mean_zero_shot_prior", 0.0) for item in summaries
    )
    mean_few_shot = _mean(
        item.get("xai", {}).get("mean_few_shot_prior", 0.0) for item in summaries
    )
    mean_zero_shot_contribution = _mean(
        item.get("xai", {}).get("mean_weighted_zero_shot_contribution", 0.0)
        for item in summaries
    )
    mean_few_shot_contribution = _mean(
        item.get("xai", {}).get("mean_weighted_few_shot_contribution", 0.0)
        for item in summaries
    )
    mean_graph_residual = _mean(
        item.get("xai", {}).get("mean_graph_residual_score", 0.0) for item in summaries
    )
    deltas = {
        name: result["mean_hit@1"] - mean_hit1
        for name, result in ablations.items()
    }
    strongest_dependency = min(deltas.items(), key=lambda item: item[1])[0] if deltas else ""
    return {
        "strongest_dependency": strongest_dependency,
        "ablation_hit@1_delta": deltas,
        "score_component_means": {
            "graph_residual_score": mean_graph_residual,
            "zero_shot_prior": mean_zero_shot,
            "zero_shot_contribution": mean_zero_shot_contribution,
            "few_shot_prior": mean_few_shot,
            "few_shot_contribution": mean_few_shot_contribution,
        },
        "summary": (
            "The online cold-start predictor mainly ranks scheduler-to-agent edges by "
            "query text, static vendor graph roles, deterministic cold-start priors, "
            "and online few-shot transition memory accumulated only after each observed query."
        ),
    }


def _assert_query_only_context(context: GraphPredictionContext) -> None:
    if context.source_node_id != SCHEDULER_ID:
        raise ValueError("query-only context must use the scheduler as the source")
    if context.source_output_text.strip():
        raise ValueError("query-only context leaked source_output_text")
    if context.runtime_text.strip():
        raise ValueError("query-only context leaked runtime_text")
    if context.candidate_actions:
        raise ValueError("query-only context leaked candidate_actions")
    if context.metadata:
        raise ValueError("query-only context leaked scheduler metadata")
    if context.graph_profile_text.strip():
        raise ValueError("query-only context leaked graph_profile_text")
    if context.combined_text().strip() != context.query_text.strip():
        raise ValueError("query-only context combined text is not exactly the query text")


def _ingest_observed_message(
    system: PredictDesignSystem,
    action: PredictedGraphAction,
    step: OnlineStep,
) -> None:
    if action.target_node_id not in system.temporal_graph.nodes:
        return
    context_text = f"observed {step.scenario} {step.query_id} activated {action.target_node_id}"
    system.update_node_context(
        action.target_node_id,
        _vector(context_text, system.config.context_dim),
        text=context_text,
    )
    message = Message.build_completion_message(
        time=step.time_value,
        source_node_id=action.source_node_id,
        target_node_id=action.target_node_id,
        context=_vector(context_text, system.config.context_dim),
        hidden_dim=system.config.hidden_dim,
        context_dim=system.config.context_dim,
        device=system.device,
    )
    message.metadata["relation_type"] = str(action.relation_type or "activate")
    message.metadata["raw_text"] = context_text
    message.metadata["available_for_prediction"] = True
    system.ingest_messages([message])


def _window_matches(
    predicted_actions: list[PredictedGraphAction],
    expected_actions: list[PredictedGraphAction],
) -> bool:
    return any(_actions_match(predicted, expected) for predicted in predicted_actions for expected in expected_actions)


def _actions_match(predicted: PredictedGraphAction, expected: PredictedGraphAction) -> bool:
    if predicted.action_type != expected.action_type:
        return False
    if predicted.action_type == GraphActionType.CREATE_EDGE:
        return (
            predicted.source_node_id == expected.source_node_id
            and predicted.target_node_id == expected.target_node_id
        )
    return predicted.action_type == expected.action_type


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict online cold-start evaluation from vendor scenario definitions only."
    )
    parser.add_argument("--vendor-root", type=Path, default=PROJECT_ROOT / "vendor" / "prefetch-kv-mas")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--speculative-steps",
        type=int,
        default=2,
        help="Number of intra-query speculative rollout steps to time per online query.",
    )
    parser.add_argument(
        "--latency-warmup-steps",
        type=int,
        default=2,
        help="Exclude this many first steps from latency summaries while still recording them.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast unless this Python environment can use CUDA.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "vendor_online_cold_start_xai.json",
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Print post-hoc expected/top-1 examples after evaluation. Disabled by default.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = _resolve_device(args.device, require_cuda=args.require_cuda)
    config = _config(context_dim=args.context_dim, hidden_dim=args.hidden_dim, device=device)
    runs = _build_runs(vendor_root=args.vendor_root, queries=args.queries, config=config)
    print(
        "mode=query_only_online no_prediction_fields=True no_candidates=True "
        "no_current_output=True no_scheduler_metadata=True no_old_logs=True "
        f"input_context=query_text_only static_graph=vendor_config metric=target_hit device={device}"
    )
    summaries: list[dict[str, Any]] = []
    for scenario, query_count, system, steps in runs:
        summary = _evaluate_steps(
            system,
            steps,
            collect_xai=True,
            speculative_steps=args.speculative_steps,
            latency_warmup_steps=args.latency_warmup_steps,
            include_timing_records=True,
        )
        summary["scenario"] = scenario
        summary["queries"] = query_count
        summaries.append(summary)
        hit = summary["hit"]
        buckets = summary["bucket_hit@1"]
        timing = summary["timing"]
        prediction_mean = timing["prediction_time"]["mean_ms"]
        prediction_p95 = timing["prediction_time"]["p95_ms"]
        speculative_mean = timing["speculative_rollout_time"]["mean_ms"]
        overhead_mean = timing["total_speculative_overhead"]["mean_ms"]
        print(
            f"{scenario:<9} queries={query_count:<3} steps={summary['steps']:<3} "
            f"hit@1={hit['hit@1']:.3f} hit@2={hit['hit@2']:.3f} hit@3={hit['hit@3']:.3f} "
            f"query_hit@1={buckets.get('query_only', 0.0):.3f} "
            f"online_updates={summary['online_memory_updates']} "
            f"pred_mean_ms={prediction_mean:.3f} pred_p95_ms={prediction_p95:.3f} "
            f"rollout{args.speculative_steps}_mean_ms={speculative_mean:.3f} "
            f"spec_overhead_mean_ms={overhead_mean:.3f}"
        )
        if args.show_examples:
            for example in summary["examples"][:2]:
                print(f"  {example}")
    mean_hit1 = sum(item["hit"]["hit@1"] for item in summaries) / max(len(summaries), 1)
    ablations = {
        name: _evaluate_ablation(
            vendor_root=args.vendor_root,
            queries=args.queries,
            base_config=config,
            ablation=name,
            speculative_steps=args.speculative_steps,
            latency_warmup_steps=args.latency_warmup_steps,
        )
        for name in (
            "no_query",
            "no_zero_shot_prior",
            "no_online_memory",
            "no_runtime_features",
            "no_structural_prior",
        )
    }
    report = {
        "mode": "query_only_online",
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "queries_per_scenario": args.queries,
        "leakage_guard": "passed",
        "scenario_results": summaries,
        "mean_hit@1": mean_hit1,
        "ablations": ablations,
        "interpretation": _interpret_xai(summaries, ablations),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"mean_hit@1={mean_hit1:.3f}")
    for name, result in ablations.items():
        delta = result["mean_hit@1"] - mean_hit1
        print(f"xai_ablation={name} mean_hit@1={result['mean_hit@1']:.3f} delta={delta:.3f}")
    print(f"report={args.report_path}")
    print("leakage_guard=passed")


if __name__ == "__main__":
    main()
