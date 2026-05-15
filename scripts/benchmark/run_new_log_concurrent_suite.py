from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_new_log_cold_start import (
    NextAgentGlobalMemory,
    OnlinePatternMemory,
    _discover_raw_event_logs,
    _latency_summary,
    _primary_records,
    _summarize,
)
from scripts.benchmark.run_new_log_concurrent_batches import (
    _apply_transition_updates,
    _evaluate_active_batch,
    evaluate_concurrent_batches,
)


_RECOVERY_EVENT_TYPES = {
    "continuation",
    "planner_summary",
    "planner_continue",
    "planner",
}

_COMPONENT_SCORE_FIELDS = [
    ("semantic_profile", "semantic_profile_scores"),
    ("memory_profile", "memory_profile_scores"),
    ("prompt_profile", "prompt_profile_scores"),
    ("schedule_prior", "schedule_prior_scores"),
    ("role_workflow_prior", "role_workflow_prior_scores"),
    ("meta_prior", "meta_prior_scores"),
    ("adaptive_cross_file", "adaptive_cross_file_scores"),
    ("contextual_cross_file", "contextual_cross_file_scores"),
    ("episodic_cross_file", "episodic_cross_file_scores"),
    ("profile_signature_transition", "profile_signature_transition_scores"),
    ("roster_position_transition", "roster_position_transition_scores"),
    ("first_target_profile", "first_target_profile_scores"),
    ("visible_context", "visible_context_scores"),
    ("base_total", "base_top_scores"),
    ("final_total", "top_scores"),
]

_EXPERT_ADVICE_FIELDS = [
    ("base_total", "base_top_scores"),
    ("final_total", "top_scores"),
    ("role_workflow_prior", "role_workflow_prior_scores"),
    ("adaptive_cross_file", "adaptive_cross_file_scores"),
    ("contextual_cross_file", "contextual_cross_file_scores"),
    ("profile_signature_transition", "profile_signature_transition_scores"),
    ("roster_position_transition", "roster_position_transition_scores"),
    ("schedule_prior", "schedule_prior_scores"),
    ("meta_prior", "meta_prior_scores"),
    ("semantic_profile", "semantic_profile_scores"),
    ("memory_profile", "memory_profile_scores"),
    ("prompt_profile", "prompt_profile_scores"),
    ("visible_context", "visible_context_scores"),
    ("first_target_profile", "first_target_profile_scores"),
]


def _expected_agents(record: dict[str, Any]) -> set[str]:
    expected = record.get("expected_agent_ids")
    if isinstance(expected, list) and expected:
        return {str(agent_id) for agent_id in expected if str(agent_id)}
    single = str(record.get("expected_agent_id") or "")
    return {single} if single else set()


def _default_report_path(root: Path, batch_sizes: list[int], policy_mode: str) -> Path:
    suffix = "_".join(f"b{batch_size}" for batch_size in batch_sizes)
    return root / f"next_agent_concurrent_suite_{policy_mode}_{suffix}_report.json"


def _default_timing_path(root: Path, batch_sizes: list[int], policy_mode: str) -> Path:
    suffix = "_".join(f"b{batch_size}" for batch_size in batch_sizes)
    return root / f"next_agent_concurrent_suite_{policy_mode}_{suffix}_timing.jsonl"


def _default_convergence_plot_path(root: Path, batch_sizes: list[int], policy_mode: str) -> Path:
    suffix = "_".join(f"b{batch_size}" for batch_size in batch_sizes)
    return root / f"next_agent_concurrent_suite_{policy_mode}_{suffix}_convergence.svg"


def _policy_config(policy_mode: str) -> dict[str, Any]:
    base = {
        "use_cross_file_memory": True,
        "cross_file_stat_weight": 0.0,
        "enable_adaptive_cross_file_prior": True,
        "enable_profile_signature_transition_prior": False,
        "enable_roster_position_transition_prior": False,
        "enable_episodic_cross_file_prior": False,
        "adaptive_cross_file_weight": 30.0,
        "adaptive_cross_file_min_support": 1,
        "adaptive_cross_file_min_confidence": 0.4,
        "adaptive_cross_file_min_profile_stability": 0.65,
        "enable_graph_order_prior": True,
        "enable_role_workflow_prior": True,
        "enable_local_transition_memory": True,
        "online_evidence_mode": "heuristic",
        "candidate_scope": "visible_graph",
        "online_feedback_scope": "event",
        "enable_research_schedule_prior": True,
        "enable_research_meta_prior": True,
        "enable_profile_similarity_prior": True,
        "profile_similarity_mode": "full",
        "enable_idf_profile_prior": False,
        "enable_online_pair_calibration": True,
        "pair_calibration_margin": 1,
        "include_visible_agent_context": True,
        "visible_context_similarity_weight": 10.0,
        "visible_context_length_weight": 0.0,
        "agent_id_view": "original",
        "agent_id_salt": "",
    }
    if policy_mode == "full":
        return base
    if policy_mode == "balanced":
        return {
            **base,
            "include_visible_agent_context": False,
            "visible_context_similarity_weight": 0.0,
        }
    if policy_mode == "compact":
        return {
            **base,
            "profile_similarity_mode": "task",
            "include_visible_agent_context": False,
            "visible_context_similarity_weight": 0.0,
        }
    if policy_mode == "fast":
        return {
            **base,
            "enable_profile_similarity_prior": False,
            "profile_similarity_mode": "full",
            "adaptive_cross_file_min_profile_stability": 0.0,
            "include_visible_agent_context": False,
            "visible_context_similarity_weight": 0.0,
        }
    if policy_mode == "strict_online":
        return {
            **base,
            "enable_graph_order_prior": False,
            "enable_role_workflow_prior": False,
            "online_evidence_mode": "transition_only",
            "online_feedback_scope": "query",
            "enable_research_schedule_prior": False,
            "enable_research_meta_prior": False,
            "enable_profile_similarity_prior": False,
            "profile_similarity_mode": "full",
            "enable_idf_profile_prior": False,
            "enable_online_pair_calibration": False,
            "adaptive_cross_file_min_profile_stability": 0.0,
            "include_visible_agent_context": False,
            "visible_context_similarity_weight": 0.0,
            "visible_context_length_weight": 0.0,
        }
    if policy_mode == "strict_all_agents":
        return {
            **base,
            "enable_graph_order_prior": False,
            "enable_role_workflow_prior": False,
            "online_evidence_mode": "transition_only",
            "candidate_scope": "all_agents",
            "online_feedback_scope": "query",
            "enable_research_schedule_prior": False,
            "enable_research_meta_prior": False,
            "enable_profile_similarity_prior": False,
            "profile_similarity_mode": "full",
            "enable_idf_profile_prior": False,
            "enable_online_pair_calibration": False,
            "adaptive_cross_file_min_profile_stability": 0.0,
            "include_visible_agent_context": False,
            "visible_context_similarity_weight": 0.0,
            "visible_context_length_weight": 0.0,
        }
    if policy_mode == "strict_no_memory":
        return {
            **_policy_config("strict_online"),
            "use_cross_file_memory": False,
            "enable_adaptive_cross_file_prior": False,
        }
    if policy_mode == "strict_no_memory_all_agents":
        return {
            **_policy_config("strict_all_agents"),
            "use_cross_file_memory": False,
            "enable_adaptive_cross_file_prior": False,
        }
    if policy_mode == "strict_profile_online":
        return {
            **_policy_config("strict_online"),
            "enable_adaptive_cross_file_prior": False,
            "enable_profile_signature_transition_prior": True,
            "enable_roster_position_transition_prior": True,
            "adaptive_cross_file_weight": 45.0,
        }
    if policy_mode == "strict_profile_event_online":
        return {
            **_policy_config("strict_profile_online"),
            "online_feedback_scope": "event",
        }
    if policy_mode == "skeptical_profile_event_online":
        return {
            **_policy_config("strict_profile_event_online"),
            "candidate_scope": "all_agents",
            "enable_roster_position_transition_prior": False,
            "enable_local_transition_memory": False,
        }
    if policy_mode == "semantic_skeptical_profile_event_online":
        return {
            **_policy_config("skeptical_profile_event_online"),
            "enable_profile_similarity_prior": True,
            "profile_similarity_mode": "task",
        }
    if policy_mode == "structural_event_online":
        return {
            **_policy_config("strict_profile_event_online"),
            "enable_graph_order_prior": True,
        }
    if policy_mode == "strict_all_agents_profile_online":
        return {
            **_policy_config("strict_all_agents"),
            "enable_adaptive_cross_file_prior": False,
            "enable_profile_signature_transition_prior": True,
            "enable_roster_position_transition_prior": True,
            "adaptive_cross_file_weight": 45.0,
        }
    if policy_mode == "strict_all_agents_profile_event_online":
        return {
            **_policy_config("strict_all_agents_profile_online"),
            "online_feedback_scope": "event",
        }
    if policy_mode == "structural_all_agents_event_online":
        return {
            **_policy_config("strict_all_agents_profile_event_online"),
            "enable_graph_order_prior": True,
        }
    if policy_mode == "strict_id_permutation":
        return {
            **_policy_config("strict_online"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "strict_all_agents_id_permutation":
        return {
            **_policy_config("strict_all_agents"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "strict_profile_id_permutation":
        return {
            **_policy_config("strict_profile_online"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "strict_profile_event_id_permutation":
        return {
            **_policy_config("strict_profile_event_online"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "skeptical_profile_event_id_permutation":
        return {
            **_policy_config("skeptical_profile_event_online"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "semantic_skeptical_profile_event_id_permutation":
        return {
            **_policy_config("semantic_skeptical_profile_event_online"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "structural_event_id_permutation":
        return {
            **_policy_config("structural_event_online"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "strict_all_agents_profile_id_permutation":
        return {
            **_policy_config("strict_all_agents_profile_online"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "strict_all_agents_profile_event_id_permutation":
        return {
            **_policy_config("strict_all_agents_profile_event_online"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "structural_all_agents_event_id_permutation":
        return {
            **_policy_config("structural_all_agents_event_online"),
            "agent_id_view": "per_file_permutation",
        }
    if policy_mode == "robust":
        return {
            **base,
            "enable_episodic_cross_file_prior": True,
            "enable_profile_signature_transition_prior": True,
            "enable_roster_position_transition_prior": True,
            "include_visible_agent_context": False,
            "visible_context_similarity_weight": 0.0,
        }
    raise ValueError(f"unknown policy mode: {policy_mode}")


def _policy_claim(policy_mode: str) -> dict[str, str]:
    if policy_mode == "strict_online":
        return {
            "claim": "online_learning_with_visible_candidate_scope",
            "reporting_rule": (
                "May be reported as strict online-learning accuracy: hand-written role "
                "workflow, schedule/meta, profile/context similarity, and graph-order "
                "priors are disabled; online evidence is transition-only; feedback "
                "updates are delayed until each query completes. The visible graph/tool "
                "candidate scope must be disclosed because it can strongly narrow the "
                "search space."
            ),
        }
    if policy_mode == "strict_all_agents":
        return {
            "claim": "online_learning_no_candidate_narrowing",
            "reporting_rule": (
                "May be reported as the stricter all-agent ablation: hand-written role "
                "workflow, schedule/meta, profile/context similarity, graph-order "
                "priors, and visible graph/tool-schema candidate narrowing are disabled; "
                "online evidence is transition-only and feedback updates are delayed "
                "until each query completes."
            ),
        }
    if policy_mode in {"strict_no_memory", "strict_no_memory_all_agents"}:
        return {
            "claim": "zero_cross_query_memory_ablation",
            "reporting_rule": (
                "Audit ablation only: no completed-query memory is available, so this "
                "measures candidate scope and deterministic fallback/order effects rather "
                "than online learning."
            ),
        }
    if policy_mode in {"strict_profile_event_online", "strict_all_agents_profile_event_online"}:
        return {
            "claim": "query_internal_profile_conditioned_online_learning",
            "reporting_rule": (
                "May be reported as query-internal online prediction: cross-query raw "
                "agent-id transition counts are disabled, completed-query memory uses "
                "visible profile/roster signatures, and local same-query transition "
                "memory is updated only after each earlier predicted step has actually "
                "occurred."
            ),
        }
    if policy_mode == "skeptical_profile_event_online":
        return {
            "claim": "skeptical_profile_only_online_learning",
            "reporting_rule": (
                "Most conservative non-oracle profile-only mode: graph-order, roster "
                "position memory, visible graph candidate narrowing, raw cross-query "
                "agent-id transitions, and raw local same-query agent-id transitions "
                "are disabled. It keeps only visible profile-signature memory and "
                "all-agent candidate ranking."
            ),
        }
    if policy_mode == "semantic_skeptical_profile_event_online":
        return {
            "claim": "semantic_skeptical_profile_online_learning",
            "reporting_rule": (
                "Skeptical profile-only mode plus visible task-profile to candidate-profile "
                "token similarity. It still disables graph-order, roster-position memory, "
                "visible graph candidate narrowing, raw cross-query agent-id transitions, "
                "and raw local same-query agent-id transition memory."
            ),
        }
    if policy_mode in {"structural_event_online", "structural_all_agents_event_online"}:
        return {
            "claim": "query_internal_structural_online_learning",
            "reporting_rule": (
                "May be reported as a deployable structural online result: it uses "
                "visible graph/tool-schema candidate order, visible profile/roster "
                "completed-query memory, and same-query local feedback only after "
                "earlier predicted steps have occurred. It is not pure online-memory-only."
            ),
        }
    if policy_mode in {"strict_profile_online", "strict_all_agents_profile_online"}:
        return {
            "claim": "profile_conditioned_online_learning",
            "reporting_rule": (
                "May be reported as a profile-conditioned online-memory result: raw "
                "agent-id transition counts are disabled, completed-query transitions "
                "are aggregated by visible source/target profile signatures and "
                "visible worker roster positions, and feedback is delayed until each "
                "query completes."
            ),
        }
    if policy_mode in {"strict_id_permutation", "strict_all_agents_id_permutation"}:
        return {
            "claim": "agent_id_permutation_counterfactual",
            "reporting_rule": (
                "Audit ablation only: worker agent ids are consistently permuted inside "
                "each query file with a different deterministic permutation per file. A "
                "large drop means the original result depends on fixed cross-query agent "
                "identity/role alignment."
            ),
        }
    if policy_mode in {
        "strict_profile_event_id_permutation",
        "strict_all_agents_profile_event_id_permutation",
    }:
        return {
            "claim": "query_internal_profile_conditioned_agent_id_permutation_counterfactual",
            "reporting_rule": (
                "Audit for query-internal profile/roster-conditioned online learning "
                "under per-file agent-id permutation. Cross-query raw agent numbering "
                "cannot be used; same-query local transition updates are allowed only "
                "after earlier steps have occurred."
            ),
        }
    if policy_mode == "skeptical_profile_event_id_permutation":
        return {
            "claim": "skeptical_profile_only_agent_id_permutation_counterfactual",
            "reporting_rule": (
                "Most conservative anti-cheat audit: per-file agent-id permutation, "
                "all-agent candidate ranking, no graph-order, no roster-position "
                "memory, no raw cross-query agent-id memory, and no raw local same-query "
                "agent-id transition memory."
            ),
        }
    if policy_mode == "semantic_skeptical_profile_event_id_permutation":
        return {
            "claim": "semantic_skeptical_agent_id_permutation_counterfactual",
            "reporting_rule": (
                "Skeptical anti-cheat audit plus visible task/profile semantic matching. "
                "Per-file agent-id permutation and all-agent candidate ranking remain "
                "enabled; structural order and raw transition shortcuts remain disabled."
            ),
        }
    if policy_mode in {
        "structural_event_id_permutation",
        "structural_all_agents_event_id_permutation",
    }:
        return {
            "claim": "structural_agent_id_permutation_counterfactual",
            "reporting_rule": (
                "Audit for deployable structural online prediction under per-file "
                "agent-id permutation. Cross-query raw agent numbering cannot be used; "
                "visible graph/tool-schema order and visible roster positions remain "
                "available and must be disclosed."
            ),
        }
    if policy_mode in {
        "strict_profile_id_permutation",
        "strict_all_agents_profile_id_permutation",
    }:
        return {
            "claim": "profile_conditioned_agent_id_permutation_counterfactual",
            "reporting_rule": (
                "Audit for profile-conditioned online learning under per-file agent-id "
                "permutation. The query remains internally consistent, but fixed raw "
                "agent numbering cannot be used across queries."
            ),
        }
    return {
        "claim": "structural_prior_baseline",
        "reporting_rule": (
            "Do not report as online-learning-only accuracy. This mode may include "
            "static role/profile/schedule/context/cascade priors or structural ablations."
        ),
    }


def _record_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("file_name", "")),
        int(record.get("step_index") or 0),
        str(record.get("event_id", "")),
    )


def _records_summary(
    records: list[dict[str, Any]],
    *,
    dataset_name: str,
    file_count: int,
) -> dict[str, Any]:
    total = len(records)
    hit_counts = _hit_counts(records)
    base_hit_counts = {
        str(k): sum(
            1
            for record in records
            if _expected_agents(record).intersection((record.get("base_prediction") or [])[:k])
        )
        for k in (1, 2, 3, 5)
    }
    return {
        "dataset_name": dataset_name,
        "file_count": file_count,
        "total_steps": total,
        "hit_counts": hit_counts,
        "hit_at_k": _hit_rates(hit_counts, total),
        "base_hit_counts": base_hit_counts,
        "base_hit_at_k": _hit_rates(base_hit_counts, total),
        "mean_candidate_count": (
            sum(float(record.get("candidate_count", 0.0)) for record in records) / total
            if total
            else 0.0
        ),
        **_latency_summary(records, "prediction_time_ms"),
    }


def _merge_cascade_records(
    *,
    fast_records: list[dict[str, Any]],
    slow_records: list[dict[str, Any]],
    margin_threshold: float,
    slow_policy_mode: str,
) -> list[dict[str, Any]]:
    slow_by_key = {_record_key(record): record for record in slow_records}
    merged: list[dict[str, Any]] = []
    for fast_record in fast_records:
        slow_record = slow_by_key.get(_record_key(fast_record))
        fast_margin = _score_margin(fast_record, 0, 1)
        use_slow = slow_record is not None and fast_margin <= margin_threshold
        source_record = slow_record if use_slow else fast_record
        record = dict(source_record)
        record["cascade_enabled"] = True
        record["cascade_stage"] = slow_policy_mode if use_slow else "fast"
        record["cascade_fast_margin"] = fast_margin
        record["cascade_margin_threshold"] = margin_threshold
        record["cascade_slow_policy_mode"] = slow_policy_mode
        record["cascade_fast_prediction_time_ms"] = float(
            fast_record.get("prediction_time_ms", 0.0)
        )
        record["cascade_slow_prediction_time_ms"] = (
            float(slow_record.get("prediction_time_ms", 0.0))
            if use_slow and slow_record is not None
            else 0.0
        )
        record["prediction_time_ms"] = (
            record["cascade_fast_prediction_time_ms"]
            + record["cascade_slow_prediction_time_ms"]
        )
        record["fast_prediction"] = fast_record.get("prediction", [])
        record["fast_top_scores"] = fast_record.get("top_scores", {})
        if slow_record is not None:
            record["slow_prediction"] = slow_record.get("prediction", [])
        merged.append(record)
    return merged


def _cascade_stage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    stage_counts = Counter(str(record.get("cascade_stage", "")) for record in records)
    return {
        "stage_counts": dict(stage_counts),
        "slow_stage_rate": (
            sum(count for stage, count in stage_counts.items() if stage != "fast") / total
            if total
            else 0.0
        ),
        "fast_stage_rate": stage_counts.get("fast", 0) / total if total else 0.0,
    }


def _cascade_report_from_records(
    *,
    fast_report: dict[str, Any],
    slow_report: dict[str, Any],
    records: list[dict[str, Any]],
    policy_mode: str,
    margin_threshold: float,
    slow_policy_mode: str,
) -> dict[str, Any]:
    batch_records = []
    records_by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_batch[int(record.get("batch_index") or 0)].append(record)
    fast_batches = {
        int(batch.get("batch_index") or 0): batch for batch in fast_report.get("batches", [])
    }
    for batch_index in sorted(records_by_batch):
        batch_records_for_index = records_by_batch[batch_index]
        fast_batch = fast_batches.get(batch_index, {})
        file_names = fast_batch.get("file_names") or sorted(
            {str(record.get("file_name", "")) for record in batch_records_for_index}
        )
        batch_records.append(
            {
                "batch_index": batch_index,
                "file_count": len(file_names),
                "completed_queries_before_batch": fast_batch.get(
                    "completed_queries_before_batch",
                    0,
                ),
                "batch_wall_time_ms": 0.0,
                "file_names": file_names,
                "summary": _records_summary(
                    batch_records_for_index,
                    dataset_name="new_research_logs",
                    file_count=len(file_names),
                ),
                "cascade_stage_summary": _cascade_stage_summary(batch_records_for_index),
            }
        )
    return {
        "protocol": f"next_agent_active_batch_snapshot_online_{policy_mode}",
        "log_root": fast_report["log_root"],
        "configured_batch_size": fast_report["configured_batch_size"],
        "batch_count": fast_report["batch_count"],
        "concurrent_active_replay": True,
        "input_view": (
            "cascade: run fast graph/role/schedule/online-memory policy first; "
            f"if top1-top2 margin <= {margin_threshold}, run {slow_policy_mode} and use "
            "that prediction. Both stages use only prediction-time visible inputs."
        ),
        "candidate_space": fast_report.get("candidate_space"),
        "prediction_target": "next_agent",
        "policy_mode": policy_mode,
        "cascade_margin_threshold": margin_threshold,
        "cascade_slow_policy_mode": slow_policy_mode,
        "cascade_stage_summary": _cascade_stage_summary(records),
        "fast_policy_config": _policy_config("fast"),
        "slow_policy_config": _policy_config(slow_policy_mode),
        "metric_scope": fast_report.get("metric_scope"),
        "leakage_control": (
            "Cascade selection uses only the fast-stage score margin before the true "
            "next agent is observed. Slow-stage labels are scored before any online update."
        ),
        "summary": _records_summary(
            records,
            dataset_name="new_research_logs",
            file_count=fast_report["summary"]["file_count"],
        ),
        "batches": batch_records,
        "files": fast_report.get("files", []),
        "fast_summary": fast_report.get("summary", {}),
        "slow_summary": slow_report.get("summary", {}),
    }


def _component_ranking(record: dict[str, Any], field_name: str) -> list[str]:
    prediction = list(record.get("prediction") or [])
    scores = record.get(field_name) or {}
    active = [
        candidate
        for candidate in prediction
        if abs(float(scores.get(candidate, 0.0) or 0.0)) > 1e-12
    ]
    if not active:
        return []
    rank_by_agent = {agent_id: index for index, agent_id in enumerate(prediction)}
    return sorted(
        active,
        key=lambda agent_id: (
            float(scores.get(agent_id, 0.0) or 0.0),
            -rank_by_agent.get(agent_id, len(prediction)),
        ),
        reverse=True,
    )


def _online_expert_context_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("scenario_name", "")),
        str(record.get("current_agent_id", "")),
    )


def _expert_advice_prediction(
    record: dict[str, Any],
    *,
    expert_weights: dict[str, float],
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    base_prediction = list(record.get("prediction") or [])
    if not base_prediction:
        return [], {}, {}
    rank_by_agent = {agent_id: index for index, agent_id in enumerate(base_prediction)}
    scores: Counter[str] = Counter(
        {
            agent_id: 0.001 * (len(base_prediction) - index)
            for index, agent_id in enumerate(base_prediction)
        }
    )
    expert_top_agents: dict[str, str] = {}
    for expert_name, field_name in _EXPERT_ADVICE_FIELDS:
        ranking = _component_ranking(record, field_name)
        if not ranking:
            continue
        expert_top_agents[expert_name] = ranking[0]
        weight = expert_weights.get(expert_name, 1.0)
        for rank, agent_id in enumerate(ranking[:3]):
            scores[agent_id] += weight / (rank + 1)
    ranked = sorted(
        base_prediction,
        key=lambda agent_id: (
            scores.get(agent_id, 0.0),
            -rank_by_agent.get(agent_id, len(base_prediction)),
        ),
        reverse=True,
    )
    return ranked, dict(scores), expert_top_agents


def _apply_online_expert_advice(
    records: list[dict[str, Any]],
    *,
    decay: float = 0.90,
    reward: float = 0.05,
    min_context_steps: int = 20,
) -> list[dict[str, Any]]:
    weights_by_context: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: 1.0)
    )
    seen_by_context: Counter[tuple[str, str]] = Counter()
    output: list[dict[str, Any]] = []
    records_by_wave: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_wave[_record_wave_key(record)].append(record)

    for wave_key in sorted(records_by_wave):
        wave_records = sorted(
            records_by_wave[wave_key],
            key=lambda record: (
                int(record.get("batch_position") or 0),
                str(record.get("file_name", "")),
            ),
        )
        wave_outputs: list[dict[str, Any]] = []
        for record in wave_records:
            context_key = _online_expert_context_key(record)
            new_record = dict(record)
            new_record["expert_advice_base_prediction"] = list(record.get("prediction") or [])
            new_record["expert_advice_base_top_scores"] = dict(record.get("top_scores") or {})
            new_record["online_expert_advice_enabled"] = True
            new_record["online_expert_advice_context"] = list(context_key)
            new_record["online_expert_advice_seen_before"] = seen_by_context[context_key]
            if seen_by_context[context_key] >= min_context_steps:
                ranked, scores, expert_top_agents = _expert_advice_prediction(
                    record,
                    expert_weights=weights_by_context[context_key],
                )
                new_record["prediction"] = ranked[:5]
                new_record["top_scores"] = {
                    agent_id: scores.get(agent_id, 0.0) for agent_id in ranked[:5]
                }
                new_record["online_expert_advice_applied"] = True
                new_record["online_expert_advice_expert_top_agents"] = expert_top_agents
            else:
                new_record["online_expert_advice_applied"] = False
                new_record["online_expert_advice_expert_top_agents"] = {}
            expected = _expected_agents(new_record)
            prediction = list(new_record.get("prediction") or [])
            new_record["hit_at_k"] = {
                str(k): bool(expected.intersection(prediction[:k]))
                for k in (1, 2, 3, 5)
            }
            new_record["target_recall_at_k"] = {
                str(k): _target_recall(expected, prediction, k)
                for k in (1, 2, 3, 5)
            }
            wave_outputs.append(new_record)
            output.append(new_record)

        for record in wave_outputs:
            context_key = _online_expert_context_key(record)
            expected = _expected_agents(record)
            original_record = dict(record)
            original_record["prediction"] = list(
                record.get("expert_advice_base_prediction") or record.get("prediction") or []
            )
            for expert_name, field_name in _EXPERT_ADVICE_FIELDS:
                ranking = _component_ranking(original_record, field_name)
                if not ranking:
                    continue
                if expected.intersection(ranking[:1]):
                    weights_by_context[context_key][expert_name] *= 1.0 + reward
                else:
                    weights_by_context[context_key][expert_name] *= decay
            seen_by_context[context_key] += 1
    return output


def _expert_advice_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    enabled_records = [
        record for record in records if record.get("online_expert_advice_enabled")
    ]
    applied_records = [
        record for record in enabled_records if record.get("online_expert_advice_applied")
    ]
    base_hit_counts = {
        str(k): sum(
            1
            for record in enabled_records
            if _expected_agents(record).intersection(
                (record.get("expert_advice_base_prediction") or [])[:k]
            )
        )
        for k in (1, 2, 3, 5)
    }
    return {
        "enabled": bool(enabled_records),
        "definition": (
            "Online expert-advice reranking over no-leak score components. Expert "
            "weights are keyed by scenario and current source agent, used before "
            "the current wave is labeled, then updated after that wave is scored."
        ),
        "configured_decay": 0.90,
        "configured_reward": 0.05,
        "configured_min_context_steps": 20,
        "records": len(enabled_records),
        "applied_records": len(applied_records),
        "base_hit_at_k": _hit_rates(base_hit_counts, len(enabled_records)),
        "final_hit_at_k": _hit_rates(_hit_counts(enabled_records), len(enabled_records)),
    }


def _method_references() -> list[dict[str, str]]:
    return [
        {
            "name": "Weighted Majority / prediction with expert advice",
            "use_in_project": (
                "Motivates expert_cascade: combine no-leak score components online "
                "and update weights only after feedback arrives."
            ),
            "url": "https://onlineprediction.cs.rhul.ac.uk/index.html?n=Main.WeightedMajorityAlgorithm",
        },
        {
            "name": "Adaptive conformal inference under distribution shift",
            "use_in_project": (
                "Motivates adaptive branch sets and sustained coverage tracking under "
                "online distribution shift."
            ),
            "url": "https://papers.nips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html",
        },
        {
            "name": "Selective classification / risk-coverage tradeoff",
            "use_in_project": (
                "Frames strict top1, prediction sets, branch precision, and coverage "
                "as separate operating points."
            ),
            "url": "https://arxiv.org/abs/1705.08500",
        },
        {
            "name": "FrugalML cost-aware prediction cascades",
            "use_in_project": (
                "Motivates cheap-first cascade routing and cost-normalized utility."
            ),
            "url": "https://arxiv.org/abs/2006.07512",
        },
    ]


def _external_data_source_candidates() -> list[dict[str, str]]:
    return [
        {
            "name": "MultiAgentBench",
            "why_relevant": (
                "Multi-agent collaboration benchmark; useful for adding more scenarios "
                "with explicit agent interaction structure."
            ),
            "integration_note": (
                "Convert each interaction trace into raw event JSONL with agent_id, "
                "event_id, parent_event_ids, agents, task_profile, and request fields."
            ),
            "url": "https://github.com/MultiagentBench/MARBLE",
        },
        {
            "name": "AgentBench",
            "why_relevant": (
                "Agent evaluation tasks with tool use and environment feedback; useful "
                "for broader workflow priors beyond coding/research."
            ),
            "integration_note": (
                "Map environment steps and tool calls into current_event or next_agent "
                "episodes; avoid using final answers as prediction inputs."
            ),
            "url": "https://github.com/THUDM/AgentBench",
        },
        {
            "name": "SWE-bench",
            "why_relevant": (
                "Software-engineering issue-to-patch tasks; useful for coding-agent "
                "workflow traces and role transitions."
            ),
            "integration_note": (
                "Generate multi-agent runs over SWE-bench tasks, then log only visible "
                "query/profile/tool-schema information before each prediction."
            ),
            "url": "https://www.swebench.com/",
        },
        {
            "name": "ToolBench",
            "why_relevant": (
                "Tool-use benchmark with API/tool selection structure; useful for "
                "candidate construction and tool-routing priors."
            ),
            "integration_note": (
                "Convert tool invocation trajectories into agent/tool transition events; "
                "do not include oracle tool labels before scoring."
            ),
            "url": "https://github.com/OpenBMB/ToolBench",
        },
        {
            "name": "WebArena",
            "why_relevant": (
                "Web task benchmark for autonomous agents; useful for long-horizon "
                "stateful workflows and online feedback."
            ),
            "integration_note": (
                "Treat browser actions, observations, and assistant turns as visible "
                "history; keep post-action page results out of the current prediction."
            ),
            "url": "https://webarena.dev/",
        },
        {
            "name": "tau-bench",
            "why_relevant": (
                "Tool-agent-user interaction benchmark; useful for realistic service "
                "workflow transitions and policy evaluation."
            ),
            "integration_note": (
                "Map user/tool/agent turns to event JSONL and evaluate next-agent or "
                "next-tool prediction under online replay."
            ),
            "url": "https://github.com/sierra-research/tau-bench",
        },
    ]


def _hit_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(k): sum(
            1
            for record in records
            if _expected_agents(record).intersection((record.get("prediction") or [])[:k])
        )
        for k in (1, 2, 3, 5)
    }


def _hit_rates(hit_counts: dict[str, int], total: int) -> dict[str, float]:
    return {key: (value / total if total else 0.0) for key, value in hit_counts.items()}


def _label_shuffle_negative_control(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "definition": "No records.",
            "hit_at_k": {str(k): 0.0 for k in (1, 2, 3, 5)},
            "total_steps": 0,
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("scenario_name", ""))].append(record)
    shuffled_expected_by_key: dict[tuple[str, int, str], set[str]] = {}
    for scenario_name, scenario_records in grouped.items():
        ordered = sorted(scenario_records, key=_record_key)
        expected_sets = [_expected_agents(record) for record in ordered]
        seed = int.from_bytes(
            hashlib.sha256(f"label-shuffle:{scenario_name}".encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        shuffled = list(expected_sets)
        random.Random(seed).shuffle(shuffled)
        if len(shuffled) > 1 and all(
            shuffled[index] == expected_sets[index] for index in range(len(shuffled))
        ):
            shuffled = shuffled[1:] + shuffled[:1]
        for record, expected in zip(ordered, shuffled):
            shuffled_expected_by_key[_record_key(record)] = expected

    hit_counts = {
        str(k): sum(
            1
            for record in records
            if shuffled_expected_by_key.get(_record_key(record), set()).intersection(
                (record.get("prediction") or [])[:k]
            )
        )
        for k in (1, 2, 3, 5)
    }
    return {
        "definition": (
            "Negative control: expected target sets are deterministically shuffled "
            "within each scenario after predictions are made. A high score here would "
            "suggest that the metric or candidate distribution is suspicious."
        ),
        "total_steps": len(records),
        "hit_counts": hit_counts,
        "hit_at_k": _hit_rates(hit_counts, len(records)),
    }


def _target_recall(expected: set[str], prediction: list[str], k: int) -> float:
    if not expected:
        return 0.0
    return len(expected.intersection(prediction[:k])) / len(expected)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return {
        "low": max(0.0, (centre - radius) / denominator),
        "high": min(1.0, (centre + radius) / denominator),
    }


def _hit_confidence_intervals(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    hit_counts = _hit_counts(records)
    per_scenario = {}
    for scenario_name in sorted({str(record.get("scenario_name", "")) for record in records}):
        scenario_records = [
            record for record in records if str(record.get("scenario_name", "")) == scenario_name
        ]
        scenario_total = len(scenario_records)
        scenario_hit_counts = _hit_counts(scenario_records)
        per_scenario[scenario_name] = {
            str(k): _wilson_interval(scenario_hit_counts[str(k)], scenario_total)
            for k in (1, 2, 3, 5)
        }
    return {
        "definition": (
            "Wilson 95% confidence intervals over strict predictive steps. These "
            "show uncertainty from limited step counts and should be considered when "
            "comparing small cold-start runs."
        ),
        "step_micro": {
            str(k): _wilson_interval(hit_counts[str(k)], total) for k in (1, 2, 3, 5)
        },
        "per_scenario": per_scenario,
    }


def _macro_mean(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {str(k): 0.0 for k in (1, 2, 3, 5)}
    return {
        str(k): mean(value[str(k)] for value in values)
        for k in (1, 2, 3, 5)
    }


def _query_macro(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["scenario_name"]), str(record["file_name"]))].append(record)
    query_rates = [
        _hit_rates(_hit_counts(query_records), len(query_records))
        for query_records in grouped.values()
        if query_records
    ]
    return {
        "query_count": len(query_rates),
        "hit_at_k": _macro_mean(query_rates),
    }


def _batch_macro(scenario_reports: list[dict[str, Any]]) -> dict[str, Any]:
    batch_rates: list[dict[str, float]] = []
    for scenario_report in scenario_reports:
        for batch in scenario_report["batches"]:
            if batch["summary"]["total_steps"] > 0:
                batch_rates.append(batch["summary"]["hit_at_k"])
    return {
        "batch_count": len(batch_rates),
        "hit_at_k": _macro_mean(batch_rates),
    }


def _target_set_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "target_set_size_distribution": {},
            "multi_target_step_count": 0,
            "target_recall_at_k": {str(k): 0.0 for k in (1, 2, 3, 5)},
        }
    size_counts = Counter(len(_expected_agents(record)) for record in records)
    recall_at_k = {
        str(k): mean(
            float((record.get("target_recall_at_k") or {}).get(str(k), 0.0))
            for record in records
        )
        for k in (1, 2, 3, 5)
    }
    return {
        "definition": (
            "expected_agent_ids is the set of non-recovery child targets spawned by the "
            "current main_turn when available; otherwise it falls back to the next "
            "non-recovery event. hit@k is true if top-k intersects this set."
        ),
        "target_set_size_distribution": {
            str(size): count for size, count in sorted(size_counts.items())
        },
        "multi_target_step_count": sum(
            count for size, count in size_counts.items() if size > 1
        ),
        "target_recall_at_k": recall_at_k,
    }


def _query_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("scenario_name", "")), str(record.get("file_name", "")))].append(record)
    query_rows: list[dict[str, Any]] = []
    for (scenario_name, file_name), query_records in grouped.items():
        if not query_records:
            continue
        ordered_records = sorted(query_records, key=lambda record: int(record.get("step_index") or 0))
        first_record = ordered_records[0]
        hit_counts = _hit_counts(ordered_records)
        total_steps = len(ordered_records)
        query_rows.append(
            {
                "scenario_name": scenario_name,
                "file_name": file_name,
                "query_order": int(first_record.get("completed_queries_before_batch") or 0)
                + int(first_record.get("batch_position") or 0)
                + 1,
                "steps": total_steps,
                "hit_counts": hit_counts,
                "hit_at_1": hit_counts["1"] / total_steps if total_steps else 0.0,
                "prediction_time_ms_sum": sum(
                    float(record.get("prediction_time_ms", 0.0))
                    for record in ordered_records
                ),
            }
        )
    query_rows.sort(key=lambda row: (row["query_order"], row["scenario_name"], row["file_name"]))
    return query_rows


def _cumulative_query_curve(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_steps = 0
    total_hit1 = 0
    total_latency = 0.0
    curve: list[dict[str, Any]] = []
    for index, row in enumerate(query_rows, start=1):
        total_steps += int(row["steps"])
        total_hit1 += int(row["hit_counts"]["1"])
        total_latency += float(row["prediction_time_ms_sum"])
        curve.append(
            {
                "queries_seen": index,
                "steps_seen": total_steps,
                "cumulative_hit_at_1": total_hit1 / total_steps if total_steps else 0.0,
                "cumulative_prediction_ms_mean": (
                    total_latency / total_steps if total_steps else 0.0
                ),
            }
        )
    return curve


def _checkpoint_curve(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checkpoints = {1, 2, 3, 5, 10, 15, 20, 30, 50, 100}
    return [
        row
        for row in curve
        if row["queries_seen"] in checkpoints or row["queries_seen"] == len(curve)
    ]


def _first_sustained_within_final(
    curve: list[dict[str, Any]],
    *,
    tolerance: float,
    min_queries_seen: int = 5,
) -> int | None:
    if not curve:
        return None
    final = float(curve[-1]["cumulative_hit_at_1"])
    for index, row in enumerate(curve):
        if int(row["queries_seen"]) < min_queries_seen:
            continue
        suffix = curve[index:]
        if all(
            abs(float(item["cumulative_hit_at_1"]) - final) <= tolerance
            for item in suffix
        ):
            return int(row["queries_seen"])
    return None


def _first_sustained_above(
    curve: list[dict[str, Any]],
    *,
    threshold: float,
    min_queries_seen: int = 1,
) -> int | None:
    for index, row in enumerate(curve):
        if int(row["queries_seen"]) < min_queries_seen:
            continue
        suffix = curve[index:]
        if all(float(item["cumulative_hit_at_1"]) >= threshold for item in suffix):
            return int(row["queries_seen"])
    return None


def _rolling_query_windows(
    query_rows: list[dict[str, Any]],
    *,
    window_size: int,
) -> list[dict[str, Any]]:
    if window_size <= 0:
        return []
    windows = []
    for end_index in range(window_size, len(query_rows) + 1):
        window = query_rows[end_index - window_size : end_index]
        steps = sum(int(row["steps"]) for row in window)
        hits = sum(int(row["hit_counts"]["1"]) for row in window)
        latency = sum(float(row["prediction_time_ms_sum"]) for row in window)
        windows.append(
            {
                "end_query": end_index,
                "window_size": window_size,
                "steps": steps,
                "hit_at_1": hits / steps if steps else 0.0,
                "prediction_ms_mean": latency / steps if steps else 0.0,
            }
        )
    return windows


def _convergence_summary_for_rows(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    curve = _cumulative_query_curve(query_rows)
    if not curve:
        return {
            "query_count": 0,
            "final_hit_at_1": 0.0,
            "stable_after_queries": {},
            "sustained_above_threshold_after_queries": {},
            "rolling_window": {},
        }
    window_size = min(5, len(query_rows))
    rolling = _rolling_query_windows(query_rows, window_size=window_size)
    rolling_values = [float(row["hit_at_1"]) for row in rolling]
    final_row = curve[-1]
    return {
        "query_count": len(query_rows),
        "steps": int(final_row["steps_seen"]),
        "final_hit_at_1": float(final_row["cumulative_hit_at_1"]),
        "mean_prediction_ms": float(final_row["cumulative_prediction_ms_mean"]),
        "stable_after_queries": {
            "within_2pp_of_final_for_all_later": _first_sustained_within_final(
                curve,
                tolerance=0.02,
            ),
            "within_5pp_of_final_for_all_later": _first_sustained_within_final(
                curve,
                tolerance=0.05,
            ),
        },
        "sustained_above_threshold_after_queries": {
            "70pct": _first_sustained_above(curve, threshold=0.70),
            "75pct": _first_sustained_above(curve, threshold=0.75),
            "80pct": _first_sustained_above(curve, threshold=0.80),
        },
        "rolling_window": {
            "window_size_queries": window_size,
            "final_window_hit_at_1": rolling[-1]["hit_at_1"] if rolling else 0.0,
            "min_window_hit_at_1": min(rolling_values) if rolling_values else 0.0,
            "max_window_hit_at_1": max(rolling_values) if rolling_values else 0.0,
            "windows": rolling,
        },
    }


def _query_learning_curve(records: list[dict[str, Any]]) -> dict[str, Any]:
    query_rows = _query_rows(records)
    curve = _cumulative_query_curve(query_rows)
    checkpoints = _checkpoint_curve(curve)
    final_hit_at_1 = curve[-1]["cumulative_hit_at_1"] if curve else 0.0
    per_scenario = {}
    for scenario_name in sorted({row["scenario_name"] for row in query_rows}):
        scenario_query_rows = [row for row in query_rows if row["scenario_name"] == scenario_name]
        scenario_steps = sum(int(row["steps"]) for row in scenario_query_rows)
        scenario_hit1 = sum(int(row["hit_counts"]["1"]) for row in scenario_query_rows)
        scenario_latency = sum(float(row["prediction_time_ms_sum"]) for row in scenario_query_rows)
        per_scenario[scenario_name] = {
            "query_count": len(scenario_query_rows),
            "steps": scenario_steps,
            "final_hit_at_1": scenario_hit1 / scenario_steps if scenario_steps else 0.0,
            "prediction_ms_mean": scenario_latency / scenario_steps if scenario_steps else 0.0,
        }
    return {
        "definition": (
            "Queries are ordered by completed_queries_before_batch + batch_position. "
            "For batch_size=1 this is a one-request-at-a-time online learning curve. "
            "Convergence is reported separately using sustained stability criteria, "
            "not by the first point that happens to be near the final average."
        ),
        "query_count": len(query_rows),
        "final_hit_at_1": final_hit_at_1,
        "checkpoints": checkpoints,
        "all_query_points": curve,
        "per_scenario": per_scenario,
    }


def _online_convergence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    query_rows = _query_rows(records)
    per_scenario = {}
    for scenario_name in sorted({row["scenario_name"] for row in query_rows}):
        scenario_rows = [row for row in query_rows if row["scenario_name"] == scenario_name]
        per_scenario[scenario_name] = _convergence_summary_for_rows(scenario_rows)
    return {
        "definition": (
            "A query count is considered stable only if every later cumulative hit@1 "
            "stays within the stated tolerance of the final cumulative hit@1. Threshold "
            "fields answer when the cumulative curve first stays above 70/75/80% for "
            "the rest of the replay. Null means the available queries do not support "
            "that convergence claim."
        ),
        "overall": _convergence_summary_for_rows(query_rows),
        "per_scenario": per_scenario,
    }


def _validate_records(
    records: list[dict[str, Any]],
    scenario_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    recomputed_hit_counts = _hit_counts(records)
    stored_hit_counts = {
        str(k): sum(1 for record in records if record["hit_at_k"].get(str(k), False))
        for k in (1, 2, 3, 5)
    }
    invalid_metric_records = [
        record
        for record in records
        if (
            not record.get("counted_for_metric")
            or record.get("event_type") != "main_turn"
            or int(record.get("expected_agent_count") or len(_expected_agents(record))) == 0
        )
    ]
    all_agent_profile_scope_mismatches = [
        {
            "scenario_name": record.get("scenario_name"),
            "file_name": record.get("file_name"),
            "step_index": record.get("step_index"),
            "candidate_count": record.get("candidate_count"),
            "profile_score_candidate_count": len(
                record.get("profile_score_candidates") or []
            ),
        }
        for record in records
        if record.get("candidate_scope") == "all_agents"
        and record.get("semantic_profile_scores")
        and len(record.get("profile_score_candidates") or [])
        < int(record.get("candidate_count") or 0)
    ]
    candidate_flagged_files = []
    hidden_transition_updates_exported = False
    for scenario_report in scenario_reports:
        for file_audit in scenario_report["files"]:
            if (
                file_audit.get("prediction_transition_candidates_present")
                or file_audit.get("graph_transition_candidates_present")
            ):
                candidate_flagged_files.append(
                    {
                        "scenario_name": scenario_report["scenario_name"],
                        "file_name": file_audit.get("file_name"),
                    }
                )
            hidden_transition_updates_exported = (
                hidden_transition_updates_exported
                or "_transition_updates" in file_audit
                or "_agent_profile_texts" in file_audit
            )
    return {
        "stored_hit_counts": stored_hit_counts,
        "recomputed_hit_counts": recomputed_hit_counts,
        "stored_matches_recomputed": stored_hit_counts == recomputed_hit_counts,
        "invalid_metric_record_count": len(invalid_metric_records),
        "candidate_flagged_file_count": len(candidate_flagged_files),
        "candidate_flagged_files": candidate_flagged_files[:20],
        "all_agent_profile_scope_mismatch_count": len(
            all_agent_profile_scope_mismatches
        ),
        "all_agent_profile_scope_mismatches": all_agent_profile_scope_mismatches[:20],
        "hidden_training_update_fields_exported": hidden_transition_updates_exported,
        "strict_records_only": True,
    }


def _leakage_audit_summary(
    records: list[dict[str, Any]],
    scenario_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = _validate_records(records, scenario_reports)
    forbidden_request_hits = 0
    visible_context_enabled_files = 0
    for scenario_report in scenario_reports:
        for file_audit in scenario_report.get("files", []):
            forbidden_request_hits += int(
                file_audit.get(
                    "forbidden_current_request_text_contains_expected_agent_count",
                    0,
                )
                or 0
            )
            visible_context_enabled_files += int(
                bool(file_audit.get("visible_agent_context_used"))
            )
    expert_enabled = any(record.get("online_expert_advice_enabled") for record in records)
    expert_base_preserved = all(
        bool(record.get("expert_advice_base_prediction"))
        for record in records
        if record.get("online_expert_advice_enabled")
    )
    expert_applied_before_min_context = [
        record
        for record in records
        if record.get("online_expert_advice_applied")
        and int(record.get("online_expert_advice_seen_before") or 0) < 20
    ]
    return {
        "verdict": (
            "no_known_label_leakage_detected"
            if validation["stored_matches_recomputed"]
            and validation["invalid_metric_record_count"] == 0
            and validation["candidate_flagged_file_count"] == 0
            and validation["all_agent_profile_scope_mismatch_count"] == 0
            and not validation["hidden_training_update_fields_exported"]
            and not expert_applied_before_min_context
            else "audit_failed_or_requires_manual_review"
        ),
        "forbidden_inputs": [
            "current_or_future_agent_output",
            "old_Predict_entries",
            "prediction.transition_candidates",
            "label_derived_candidate_sets",
            "scheduler_recovery_as_model_accuracy",
            "same_wave_labels_for_concurrent_queries",
        ],
        "allowed_inputs": [
            "current_executing_agent_id",
            "visible_agent_profiles_and_roles",
            "visible_tool_schema_or_inferred_graph",
            "completed_history_before_prediction",
            "completed_query_memory_before_batch",
            "optional_current_event_visible_context_snapshot",
        ],
        "metric_validation": validation,
        "forbidden_current_request_text_hits": forbidden_request_hits,
        "visible_context_enabled_file_count": visible_context_enabled_files,
        "online_expert_advice_enabled": expert_enabled,
        "online_expert_advice_base_prediction_preserved": expert_base_preserved,
        "online_expert_advice_applied_before_min_context_count": len(
            expert_applied_before_min_context
        ),
        "online_update_order": (
            "predict first; score current wave; update online memories and expert "
            "weights only after the wave labels are observed"
        ),
    }


def _score_at(record: dict[str, Any], rank_index: int) -> float:
    prediction = record.get("prediction") or []
    if rank_index >= len(prediction):
        return 0.0
    scores = record.get("top_scores") or record.get("base_top_scores") or {}
    try:
        return float(scores.get(prediction[rank_index], 0.0))
    except (TypeError, ValueError):
        return 0.0


def _score_margin(record: dict[str, Any], left_index: int, right_index: int) -> float:
    prediction = record.get("prediction") or []
    if right_index >= len(prediction):
        return float("inf")
    return _score_at(record, left_index) - _score_at(record, right_index)


def _selected_branch_count(
    record: dict[str, Any],
    *,
    fixed_k: int | None = None,
    top2_margin_threshold: float | None = None,
    top3_margin_threshold: float | None = None,
) -> int:
    prediction = record.get("prediction") or []
    if not prediction:
        return 0
    max_width = min(len(prediction), 5)
    if fixed_k is not None:
        return min(max(fixed_k, 1), max_width)
    width = 1
    if (
        max_width >= 2
        and top2_margin_threshold is not None
        and _score_margin(record, 0, 1) <= top2_margin_threshold
    ):
        width = 2
    if (
        width >= 2
        and max_width >= 3
        and top3_margin_threshold is not None
        and _score_margin(record, 1, 2) <= top3_margin_threshold
    ):
        width = 3
    return width


def _branch_policy_metrics(
    records: list[dict[str, Any]],
    *,
    name: str,
    description: str,
    fixed_k: int | None = None,
    top2_margin_threshold: float | None = None,
    top3_margin_threshold: float | None = None,
) -> dict[str, Any]:
    total = len(records)
    selected_counts = [
        _selected_branch_count(
            record,
            fixed_k=fixed_k,
            top2_margin_threshold=top2_margin_threshold,
            top3_margin_threshold=top3_margin_threshold,
        )
        for record in records
    ]
    hits = sum(
        1
        for record, selected_count in zip(records, selected_counts)
        if _expected_agents(record).intersection((record.get("prediction") or [])[:selected_count])
    )
    target_recalls = []
    branch_precisions = []
    set_f1s = []
    jaccards = []
    for record, selected_count in zip(records, selected_counts):
        expected = _expected_agents(record)
        selected = set((record.get("prediction") or [])[:selected_count])
        intersection = expected.intersection(selected)
        recall = len(intersection) / len(expected) if expected else 0.0
        precision = len(intersection) / len(selected) if selected else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        union = expected.union(selected)
        jaccard = len(intersection) / len(union) if union else 0.0
        target_recalls.append(recall)
        branch_precisions.append(precision)
        set_f1s.append(f1)
        jaccards.append(jaccard)
    per_scenario: dict[str, dict[str, Any]] = {}
    scenario_names = sorted({str(record.get("scenario_name", "")) for record in records})
    for scenario_name in scenario_names:
        scenario_records = [
            record for record in records if str(record.get("scenario_name", "")) == scenario_name
        ]
        scenario_total = len(scenario_records)
        scenario_counts = [
            _selected_branch_count(
                record,
                fixed_k=fixed_k,
                top2_margin_threshold=top2_margin_threshold,
                top3_margin_threshold=top3_margin_threshold,
            )
            for record in scenario_records
        ]
        scenario_hits = sum(
            1
            for record, selected_count in zip(scenario_records, scenario_counts)
            if _expected_agents(record).intersection(
                (record.get("prediction") or [])[:selected_count]
            )
        )
        scenario_recalls = []
        scenario_precisions = []
        scenario_f1s = []
        for record, selected_count in zip(scenario_records, scenario_counts):
            expected = _expected_agents(record)
            selected = set((record.get("prediction") or [])[:selected_count])
            intersection = expected.intersection(selected)
            recall = len(intersection) / len(expected) if expected else 0.0
            precision = len(intersection) / len(selected) if selected else 0.0
            scenario_recalls.append(recall)
            scenario_precisions.append(precision)
            scenario_f1s.append(
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        scenario_average_branches = mean(scenario_counts) if scenario_counts else 0.0
        scenario_coverage = scenario_hits / scenario_total if scenario_total else 0.0
        per_scenario[scenario_name] = {
            "steps": scenario_total,
            "coverage": scenario_coverage,
            "target_recall": mean(scenario_recalls) if scenario_recalls else 0.0,
            "branch_precision": mean(scenario_precisions) if scenario_precisions else 0.0,
            "set_f1": mean(scenario_f1s) if scenario_f1s else 0.0,
            "average_branches": scenario_average_branches,
        }
    average_branches = mean(selected_counts) if selected_counts else 0.0
    coverage = hits / total if total else 0.0
    extra_branch_rate = max(0.0, average_branches - 1.0)
    return {
        "name": name,
        "description": description,
        "fixed_k": fixed_k,
        "top2_margin_threshold": top2_margin_threshold,
        "top3_margin_threshold": top3_margin_threshold,
        "steps": total,
        "coverage": coverage,
        "target_recall": mean(target_recalls) if target_recalls else 0.0,
        "branch_precision": mean(branch_precisions) if branch_precisions else 0.0,
        "set_f1": mean(set_f1s) if set_f1s else 0.0,
        "set_jaccard": mean(jaccards) if jaccards else 0.0,
        "average_branches": average_branches,
        "extra_branch_rate_vs_top1": extra_branch_rate,
        "mean_nonmatching_branches": max(0.0, average_branches - coverage),
        "cost_normalized_utility": {
            str(cost_ratio): coverage - cost_ratio * extra_branch_rate
            for cost_ratio in (0.05, 0.10, 0.25, 0.50)
        },
        "per_scenario": per_scenario,
    }


def _best_margin_policies_under_budgets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    budgets = [1.10, 1.15, 1.20, 1.30, 1.50, 2.00]
    top2_thresholds = [0, 2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    top3_thresholds: list[float | None] = [None, 0, 2, 5, 8, 10, 15, 20, 30]
    candidates: list[dict[str, Any]] = []
    for top2_threshold in top2_thresholds:
        for top3_threshold in top3_thresholds:
            candidates.append(
                _branch_policy_metrics(
                    records,
                    name=(
                        f"margin_top2_{top2_threshold}"
                        if top3_threshold is None
                        else f"margin_top2_{top2_threshold}_top3_{top3_threshold}"
                    ),
                    description=(
                        "Diagnostic threshold search over current evaluation records; "
                        "use only as a calibration candidate, not as a separate model score."
                    ),
                    top2_margin_threshold=float(top2_threshold),
                    top3_margin_threshold=(
                        float(top3_threshold) if top3_threshold is not None else None
                    ),
                )
            )
    best: list[dict[str, Any]] = []
    for budget in budgets:
        feasible = [
            candidate
            for candidate in candidates
            if candidate["average_branches"] <= budget + 1e-9
        ]
        if not feasible:
            continue
        winner = max(
            feasible,
            key=lambda item: (
                item["coverage"],
                -item["average_branches"],
                item["top2_margin_threshold"] or 0.0,
                item["top3_margin_threshold"] or -1.0,
            ),
        )
        best.append(
            {
                "max_average_branches": budget,
                "name": winner["name"],
                "top2_margin_threshold": winner["top2_margin_threshold"],
                "top3_margin_threshold": winner["top3_margin_threshold"],
                "coverage": winner["coverage"],
                "average_branches": winner["average_branches"],
                "extra_branch_rate_vs_top1": winner["extra_branch_rate_vs_top1"],
                "per_scenario": winner["per_scenario"],
            }
        )
    return best


def _adaptive_speculation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    policies = [
        _branch_policy_metrics(
            records,
            name="fixed_top1",
            description="Single speculative branch; identical to strict hit@1.",
            fixed_k=1,
        ),
        _branch_policy_metrics(
            records,
            name="fixed_top2",
            description="Always execute the top two ranked branches.",
            fixed_k=2,
        ),
        _branch_policy_metrics(
            records,
            name="fixed_top3",
            description="Always execute the top three ranked branches.",
            fixed_k=3,
        ),
        _branch_policy_metrics(
            records,
            name="balanced_margin_top2_10",
            description="Execute top2 only when top1-top2 score margin is at most 10.",
            top2_margin_threshold=10.0,
        ),
        _branch_policy_metrics(
            records,
            name="balanced_margin_top2_15",
            description="Execute top2 only when top1-top2 score margin is at most 15.",
            top2_margin_threshold=15.0,
        ),
        _branch_policy_metrics(
            records,
            name="balanced_margin_top2_25",
            description="Execute top2 only when top1-top2 score margin is at most 25.",
            top2_margin_threshold=25.0,
        ),
        _branch_policy_metrics(
            records,
            name="aggressive_margin_top2_80_top3_5",
            description=(
                "Execute top2 for broad uncertainty, and top3 only when top2-top3 "
                "is nearly tied."
            ),
            top2_margin_threshold=80.0,
            top3_margin_threshold=5.0,
        ),
    ]
    return {
        "definition": (
            "Adaptive speculative coverage counts the true next agent as covered when it is "
            "inside the selected top-k branch set. It is a system coverage metric, not hit@1."
        ),
        "branch_cost_model": (
            "The predictor already computes a top-5 ranking in one call, so changing top-k "
            "does not materially change prediction_time_ms. The main added cost is downstream "
            "speculative work, approximated by average_branches and extra_branch_rate_vs_top1."
        ),
        "policies": policies,
        "best_under_average_branch_budgets": _best_margin_policies_under_budgets(records),
    }


def _record_wave_key(
    record: dict[str, Any],
    *,
    history_scope: str = "scenario",
) -> tuple[str, int, int]:
    scope_key = (
        "__pooled__"
        if history_scope == "pooled"
        else str(record.get("scenario_name", ""))
    )
    return (
        scope_key,
        int(record.get("batch_index") or 0),
        int(record.get("step_index") or 0),
    )


def _record_order_key(
    record: dict[str, Any],
    *,
    history_scope: str = "scenario",
) -> tuple[str, int, int, int, str]:
    scope_key = (
        "__pooled__"
        if history_scope == "pooled"
        else str(record.get("scenario_name", ""))
    )
    return (
        scope_key,
        int(record.get("batch_index") or 0),
        int(record.get("step_index") or 0),
        int(record.get("batch_position") or 0),
        str(record.get("file_name", "")),
    )


def _policy_result_for_threshold(
    record: dict[str, Any],
    *,
    top2_margin_threshold: float,
) -> dict[str, float]:
    selected_count = _selected_branch_count(
        record,
        top2_margin_threshold=top2_margin_threshold,
    )
    expected = _expected_agents(record)
    selected = set((record.get("prediction") or [])[:selected_count])
    intersection = expected.intersection(selected)
    coverage = float(bool(intersection))
    recall = len(intersection) / len(expected) if expected else 0.0
    precision = len(intersection) / len(selected) if selected else 0.0
    return {
        "selected_count": float(selected_count),
        "coverage": coverage,
        "target_recall": recall,
        "branch_precision": precision,
    }


def _choose_online_margin_threshold(
    history: list[dict[str, Any]],
    *,
    target_coverage: float,
    max_average_branches: float,
    default_threshold: float,
    min_calibration_steps: int,
) -> float:
    if len(history) < min_calibration_steps:
        return default_threshold
    thresholds = [0, 2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150]
    candidates = []
    for threshold in thresholds:
        results = [
            _policy_result_for_threshold(record, top2_margin_threshold=float(threshold))
            for record in history
        ]
        average_branches = mean(result["selected_count"] for result in results)
        coverage = mean(result["coverage"] for result in results)
        candidates.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "average_branches": average_branches,
            }
        )
    feasible = [
        candidate
        for candidate in candidates
        if candidate["coverage"] >= target_coverage
        and candidate["average_branches"] <= max_average_branches + 1e-9
    ]
    if feasible:
        winner = min(
            feasible,
            key=lambda item: (
                item["average_branches"],
                abs(item["coverage"] - target_coverage),
                item["threshold"],
            ),
        )
        return float(winner["threshold"])
    under_budget = [
        candidate
        for candidate in candidates
        if candidate["average_branches"] <= max_average_branches + 1e-9
    ]
    if under_budget:
        winner = max(
            under_budget,
            key=lambda item: (
                item["coverage"],
                -item["average_branches"],
                -item["threshold"],
            ),
        )
        return float(winner["threshold"])
    return default_threshold


def _online_adaptive_branch_policy(
    records: list[dict[str, Any]],
    *,
    name: str,
    target_coverage: float,
    max_average_branches: float,
    default_threshold: float = 10.0,
    min_calibration_steps: int = 25,
    history_scope: str = "scenario",
) -> dict[str, Any]:
    history_by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_results: list[dict[str, Any]] = []
    records_by_wave: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(
        records,
        key=lambda item: _record_order_key(item, history_scope=history_scope),
    ):
        records_by_wave[_record_wave_key(record, history_scope=history_scope)].append(record)
    for wave_key in sorted(records_by_wave):
        scope_key = wave_key[0]
        history = history_by_scope[scope_key]
        threshold = _choose_online_margin_threshold(
            history,
            target_coverage=target_coverage,
            max_average_branches=max_average_branches,
            default_threshold=default_threshold,
            min_calibration_steps=min_calibration_steps,
        )
        wave_records = records_by_wave[wave_key]
        for record in wave_records:
            result = _policy_result_for_threshold(
                record,
                top2_margin_threshold=threshold,
            )
            selected_results.append(
                {
                    **result,
                    "scenario_name": str(record.get("scenario_name", "")),
                    "history_scope_key": scope_key,
                    "threshold": threshold,
                }
            )
        history.extend(wave_records)
    if not selected_results:
        return {
            "name": name,
            "coverage": 0.0,
            "average_branches": 0.0,
            "extra_branch_rate_vs_top1": 0.0,
        }
    per_scenario = {}
    for scenario_name in sorted({row["scenario_name"] for row in selected_results}):
        scenario_rows = [
            row for row in selected_results if row["scenario_name"] == scenario_name
        ]
        per_scenario[scenario_name] = {
            "steps": len(scenario_rows),
            "coverage": mean(row["coverage"] for row in scenario_rows),
            "target_recall": mean(row["target_recall"] for row in scenario_rows),
            "branch_precision": mean(row["branch_precision"] for row in scenario_rows),
            "average_branches": mean(row["selected_count"] for row in scenario_rows),
        }
    average_branches = mean(row["selected_count"] for row in selected_results)
    coverage = mean(row["coverage"] for row in selected_results)
    extra_branch_rate = max(0.0, average_branches - 1.0)
    return {
        "name": name,
        "definition": (
            "Online adaptive top2 thresholding. The threshold is chosen from prior "
            "scored steps only; all records in the same concurrent wave use the "
            "pre-wave threshold."
        ),
        "history_scope": history_scope,
        "target_coverage": target_coverage,
        "max_average_branches": max_average_branches,
        "default_threshold": default_threshold,
        "min_calibration_steps": min_calibration_steps,
        "steps": len(selected_results),
        "coverage": coverage,
        "target_recall": mean(row["target_recall"] for row in selected_results),
        "branch_precision": mean(row["branch_precision"] for row in selected_results),
        "average_branches": average_branches,
        "extra_branch_rate_vs_top1": extra_branch_rate,
        "cost_normalized_utility": {
            str(cost_ratio): coverage - cost_ratio * extra_branch_rate
            for cost_ratio in (0.05, 0.10, 0.25, 0.50)
        },
        "threshold_usage": dict(
            Counter(str(row["threshold"]) for row in selected_results)
        ),
        "per_scenario": per_scenario,
    }


def _online_adaptive_branch_summary(
    records: list[dict[str, Any]],
    *,
    history_scope: str = "scenario",
) -> dict[str, Any]:
    return {
        "definition": (
            "Deployable online prediction-set policies. They adapt branch width using "
            "only previous scored steps, unlike post-hoc oracle diagnostics."
        ),
        "history_scope": history_scope,
        "policies": [
            _online_adaptive_branch_policy(
                records,
                name="online_target80_budget1.20",
                target_coverage=0.80,
                max_average_branches=1.20,
                history_scope=history_scope,
            ),
            _online_adaptive_branch_policy(
                records,
                name="online_target85_budget1.50",
                target_coverage=0.85,
                max_average_branches=1.50,
                default_threshold=15.0,
                history_scope=history_scope,
            ),
        ],
    }


def _component_top_agent(record: dict[str, Any], field_name: str) -> str | None:
    prediction = record.get("prediction") or []
    scores = record.get(field_name) or {}
    active_candidates = [
        candidate
        for candidate in prediction
        if abs(float(scores.get(candidate, 0.0) or 0.0)) > 1e-12
    ]
    if not active_candidates:
        return None
    rank_by_agent = {agent_id: index for index, agent_id in enumerate(prediction)}
    return max(
        active_candidates,
        key=lambda agent_id: (
            float(scores.get(agent_id, 0.0) or 0.0),
            -rank_by_agent.get(agent_id, len(prediction)),
        ),
    )


def _margin_bucket(record: dict[str, Any]) -> str:
    margin = _score_margin(record, 0, 1)
    if margin == float("inf"):
        return "single_candidate"
    for upper in (0, 2, 5, 10, 15, 25, 40, 60, 80, 120):
        if margin <= upper:
            return f"<= {upper}"
    return "> 120"


def _diagnostic_hit_rate(records: list[dict[str, Any]], k: int) -> float:
    if not records:
        return 0.0
    return sum(
        1
        for record in records
        if _expected_agents(record).intersection((record.get("prediction") or [])[:k])
    ) / len(records)


def _feasibility_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    component_stats: list[dict[str, Any]] = []
    component_oracle_hits = 0
    component_unanimous_wrong = 0
    component_no_active = 0
    for name, field_name in _COMPONENT_SCORE_FIELDS:
        active = 0
        hits = 0
        for record in records:
            top_agent = _component_top_agent(record, field_name)
            if top_agent is None:
                continue
            active += 1
            hits += int(top_agent in _expected_agents(record))
        component_stats.append(
            {
                "name": name,
                "score_field": field_name,
                "active_steps": active,
                "active_hit_at_1": hits / active if active else 0.0,
                "all_step_hit_contribution": hits / total if total else 0.0,
            }
        )

    for record in records:
        expected = _expected_agents(record)
        expert_predictions = [
            top_agent
            for _, field_name in _COMPONENT_SCORE_FIELDS
            if (top_agent := _component_top_agent(record, field_name)) is not None
        ]
        if not expert_predictions:
            component_no_active += 1
            continue
        if expected.intersection(expert_predictions):
            component_oracle_hits += 1
        if len(set(expert_predictions)) == 1 and expert_predictions[0] not in expected:
            component_unanimous_wrong += 1

    margin_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        margin_groups[_margin_bucket(record)].append(record)
    margin_buckets = []
    for bucket, bucket_records in sorted(margin_groups.items()):
        margin_buckets.append(
            {
                "bucket": bucket,
                "steps": len(bucket_records),
                "hit_at_1": _diagnostic_hit_rate(bucket_records, 1),
                "hit_at_2": _diagnostic_hit_rate(bucket_records, 2),
                "hit_at_3": _diagnostic_hit_rate(bucket_records, 3),
            }
        )

    per_scenario = {}
    for scenario_name in sorted({str(record.get("scenario_name", "")) for record in records}):
        scenario_records = [
            record for record in records if str(record.get("scenario_name", "")) == scenario_name
        ]
        scenario_total = len(scenario_records)
        scenario_oracle_hits = 0
        for record in scenario_records:
            expected = _expected_agents(record)
            expert_predictions = [
                top_agent
                for _, field_name in _COMPONENT_SCORE_FIELDS
                if (top_agent := _component_top_agent(record, field_name)) is not None
            ]
            scenario_oracle_hits += int(bool(expected.intersection(expert_predictions)))
        per_scenario[scenario_name] = {
            "steps": scenario_total,
            "strict_hit_at_1": _diagnostic_hit_rate(scenario_records, 1),
            "candidate_top2_oracle": _diagnostic_hit_rate(scenario_records, 2),
            "candidate_top3_oracle": _diagnostic_hit_rate(scenario_records, 3),
            "component_oracle": (
                scenario_oracle_hits / scenario_total if scenario_total else 0.0
            ),
        }

    return {
        "definition": (
            "Diagnostics are post-hoc upper-bound probes over no-leak score components. "
            "They are not deployable accuracy and must not be reported as model hit@1."
        ),
        "strict_hit_at_1": _diagnostic_hit_rate(records, 1),
        "candidate_top2_oracle": _diagnostic_hit_rate(records, 2),
        "candidate_top3_oracle": _diagnostic_hit_rate(records, 3),
        "candidate_top5_oracle": _diagnostic_hit_rate(records, 5),
        "component_oracle": component_oracle_hits / total if total else 0.0,
        "component_unanimous_wrong_rate": (
            component_unanimous_wrong / total if total else 0.0
        ),
        "component_no_active_steps": component_no_active,
        "component_stats": sorted(
            component_stats,
            key=lambda item: (item["active_hit_at_1"], item["active_steps"]),
            reverse=True,
        ),
        "margin_buckets": margin_buckets,
        "per_scenario": per_scenario,
    }


def _wave_latency_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record.get("scenario_name", "")),
                int(record.get("suite_batch_size") or record.get("configured_batch_size") or 0),
                int(record.get("batch_index") or 0),
                int(record.get("step_index") or 0),
            )
        ].append(record)
    wave_records = []
    for group_records in grouped.values():
        prediction_times = [
            float(record.get("prediction_time_ms", 0.0)) for record in group_records
        ]
        if not prediction_times:
            continue
        wave_records.append(
            {
                "serial_prediction_wave_ms": sum(prediction_times),
                "parallel_prediction_wave_ms": max(prediction_times),
                "wave_size": len(prediction_times),
            }
        )
    return {
        "wave_count": len(wave_records),
        "mean_wave_size": (
            sum(float(record["wave_size"]) for record in wave_records) / len(wave_records)
            if wave_records
            else 0.0
        ),
        **_latency_summary(wave_records, "serial_prediction_wave_ms"),
        **_latency_summary(wave_records, "parallel_prediction_wave_ms"),
    }


def _svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _polyline_points(
    points: list[tuple[float, float]],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    if not points:
        return ""
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    rendered = []
    for x_value, y_value in points:
        x = left + (x_value - x_min) / x_span * width
        y = top + (y_max - y_value) / y_span * height
        rendered.append(f"{x:.2f},{y:.2f}")
    return " ".join(rendered)


def _write_convergence_svg(report: dict[str, Any], path: Path) -> None:
    width = 980
    height = 620
    left = 80
    right = 220
    top = 70
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c"]
    series: list[dict[str, Any]] = []
    max_query = 1
    y_values = []
    for index, batch in enumerate(report.get("batches", [])):
        batch_size = batch.get("batch_size")
        curve = (
            batch.get("aggregate", {})
            .get("query_learning_curve", {})
            .get("all_query_points", [])
        )
        points = [
            (
                float(point.get("queries_seen", 0.0)),
                float(point.get("cumulative_hit_at_1", 0.0)),
            )
            for point in curve
        ]
        if not points:
            continue
        max_query = max(max_query, int(max(x for x, _ in points)))
        y_values.extend(y for _, y in points)
        series.append(
            {
                "label": f"batch {batch_size}",
                "points": points,
                "color": colors[index % len(colors)],
                "final": points[-1][1],
            }
        )
    if not series:
        return
    y_min = max(0.0, min(y_values) - 0.05)
    y_max = min(1.0, max(y_values) + 0.05)
    y_min = min(y_min, 0.60)
    y_max = max(y_max, 0.85)

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#111827} .small{font-size:12px;fill:#4b5563} .axis{stroke:#111827;stroke-width:1.2} .grid{stroke:#e5e7eb;stroke-width:1}</style>',
        f'<text x="{left}" y="35" font-size="22" font-weight="700">Online convergence curve: cumulative hit@1</text>',
        f'<text x="{left}" y="55" class="small">policy={_svg_escape(report.get("policy_mode", ""))}; root={_svg_escape(report.get("root", ""))}</text>',
    ]
    for tick in range(0, 6):
        ratio = tick / 5
        y = top + (1 - ratio) * plot_height
        value = y_min + ratio * (y_max - y_min)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="grid"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="small">{value * 100:.0f}%</text>')
    for tick in range(0, 6):
        ratio = tick / 5
        x = left + ratio * plot_width
        value = 1 + ratio * (max_query - 1)
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" class="grid"/>')
        elements.append(f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" class="small">{value:.0f}</text>')
    elements.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>')
    elements.append(f'<text x="{left + plot_width / 2:.2f}" y="{height - 28}" text-anchor="middle" class="small">queries seen</text>')
    elements.append(f'<text x="24" y="{top + plot_height / 2:.2f}" transform="rotate(-90 24 {top + plot_height / 2:.2f})" text-anchor="middle" class="small">cumulative hit@1</text>')
    for item in series:
        points_attr = _polyline_points(
            item["points"],
            x_min=1.0,
            x_max=float(max_query),
            y_min=y_min,
            y_max=y_max,
            left=float(left),
            top=float(top),
            width=float(plot_width),
            height=float(plot_height),
        )
        elements.append(
            f'<polyline points="{points_attr}" fill="none" stroke="{item["color"]}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        final_x, final_y = item["points"][-1]
        cx = left + (final_x - 1.0) / max(max_query - 1, 1) * plot_width
        cy = top + (y_max - final_y) / max(y_max - y_min, 1e-9) * plot_height
        elements.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="{item["color"]}"/>')
    legend_x = left + plot_width + 35
    legend_y = top + 20
    for index, item in enumerate(series):
        y = legend_y + index * 28
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{item["color"]}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 34}" y="{y + 4}" class="small">{_svg_escape(item["label"])} final {item["final"] * 100:.2f}%</text>')
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _task_index_from_path(path: Path) -> int:
    match = re.search(r"_task_(\d+)_", path.name)
    return int(match.group(1)) if match else 10**9


def _batched_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _pooled_query_rows(
    *,
    root: Path,
    scenarios: list[str],
    query_order_seed: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_name in scenarios:
        log_root = root / scenario_name
        for path in _discover_raw_event_logs(log_root):
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "path": path,
                    "task_index": _task_index_from_path(path),
                }
            )
    rows.sort(
        key=lambda row: (
            int(row["task_index"]),
            str(row["scenario_name"]),
            str(row["path"].name),
        )
    )
    if query_order_seed:
        seed = int.from_bytes(
            hashlib.sha256(
                f"pooled-query-order:{query_order_seed}:{root}".encode("utf-8")
            ).digest()[:8],
            byteorder="big",
            signed=False,
        )
        random.Random(seed).shuffle(rows)
    return rows


def evaluate_pooled_concurrent_batches(
    *,
    root: Path,
    scenarios: list[str],
    batch_size: int,
    query_order_seed: str = "",
    **policy_config: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rows = _pooled_query_rows(
        root=root,
        scenarios=scenarios,
        query_order_seed=query_order_seed,
    )
    path_to_scenario = {str(row["path"]): str(row["scenario_name"]) for row in rows}
    global_memory = OnlinePatternMemory()
    next_global_memory = NextAgentGlobalMemory()
    all_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    completed_queries_before_batch = 0
    dataset_name = "new_research_logs_pooled"

    for batch_index, batch_rows in enumerate(_batched_rows(rows, batch_size)):
        batch_paths = [row["path"] for row in batch_rows]
        batch_started = time.perf_counter()
        memory_snapshot = copy.deepcopy(global_memory)
        next_memory_snapshot = copy.deepcopy(next_global_memory)
        timing_records, active_audits, batch_outputs = _evaluate_active_batch(
            batch_paths,
            batch_index=batch_index,
            first_file_index=batch_index * batch_size,
            memory_snapshot=memory_snapshot,
            next_memory_snapshot=next_memory_snapshot,
            dataset_name=dataset_name,
            completed_queries_before_batch=completed_queries_before_batch,
            configured_batch_size=batch_size,
            **policy_config,
        )
        for record in timing_records:
            scenario_name = path_to_scenario.get(str(record.get("source_log_path", "")), "")
            record["scenario_name"] = scenario_name
            record["suite_batch_size"] = batch_size
            record["scenario_replay_mode"] = "pooled"
        for audit in active_audits:
            scenario_name = path_to_scenario.get(str(audit.get("source_log_path", "")), "")
            audit["scenario_name"] = scenario_name
            audit["scenario_replay_mode"] = "pooled"

        all_records.extend(timing_records)
        audit_records.extend(active_audits)

        batch_wall_time_ms = (time.perf_counter() - batch_started) * 1000.0
        current_primary_records = _primary_records(
            timing_records,
            prediction_target="next_agent",
        )
        batch_records.append(
            {
                "batch_index": batch_index,
                "file_count": len(batch_paths),
                "scenario_names": sorted({str(row["scenario_name"]) for row in batch_rows}),
                "completed_queries_before_batch": completed_queries_before_batch,
                "batch_wall_time_ms": batch_wall_time_ms,
                "file_names": [path.name for path in batch_paths],
                "summary": _summarize(
                    current_primary_records,
                    dataset_name=dataset_name,
                    file_count=len(batch_paths),
                ),
            }
        )

        if bool(policy_config.get("use_cross_file_memory", True)):
            for output in batch_outputs:
                global_memory.update_sequence(output["sequence"])
                _apply_transition_updates(next_global_memory, output["transition_updates"])
                next_global_memory.update_file_summary(
                    output["audit"].get("first_main_target"),
                    profile_texts=output["profile_texts"],
                    task_profile_text=output["task_profile_text"],
                )
        completed_queries_before_batch += len(batch_paths)

    output_records = _primary_records(all_records, prediction_target="next_agent")
    report = {
        "protocol": "next_agent_active_batch_snapshot_online_pooled_scenarios",
        "log_root": str(root),
        "scenarios": scenarios,
        "configured_batch_size": batch_size,
        "batch_count": len(batch_records),
        "concurrent_active_replay": True,
        "scenario_replay_mode": "pooled",
        "input_view": (
            "pooled scenario replay: all scenario folders share one chronological "
            "online memory; batch members share one pre-batch memory snapshot"
        ),
        "candidate_space": (
            "defined by policy_config candidate_scope; no scenario label is passed "
            "to the ranking function"
        ),
        "prediction_target": "next_agent",
        "policy_config": policy_config,
        "query_order_seed": query_order_seed,
        "metric_scope": (
            "main_turn decision-only speculative prediction where the next event is "
            "agent work; continuation/planner recovery events are not reported or learned"
        ),
        "summary": _summarize(
            output_records,
            dataset_name=dataset_name,
            file_count=len(rows),
        ),
        "batches": batch_records,
        "files": audit_records,
    }
    return report, output_records, audit_records


def _scenario_reports_from_pooled(
    *,
    pooled_report: dict[str, Any],
    records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
    scenarios: list[str],
) -> list[dict[str, Any]]:
    scenario_reports: list[dict[str, Any]] = []
    for scenario_name in scenarios:
        scenario_records = [
            record
            for record in records
            if str(record.get("scenario_name", "")) == scenario_name
        ]
        scenario_audits = [
            audit
            for audit in audit_records
            if str(audit.get("scenario_name", "")) == scenario_name
        ]
        batches: list[dict[str, Any]] = []
        for batch in pooled_report.get("batches", []):
            batch_index = int(batch.get("batch_index") or 0)
            batch_records = [
                record
                for record in scenario_records
                if int(record.get("batch_index") or 0) == batch_index
            ]
            if not batch_records:
                continue
            file_names = sorted({str(record.get("file_name", "")) for record in batch_records})
            batches.append(
                {
                    "batch_index": batch_index,
                    "scenario_name": scenario_name,
                    "file_count": len(file_names),
                    "completed_queries_before_batch": batch.get(
                        "completed_queries_before_batch",
                        0,
                    ),
                    "batch_wall_time_ms": batch.get("batch_wall_time_ms", 0.0),
                    "file_names": file_names,
                    "summary": _records_summary(
                        batch_records,
                        dataset_name="new_research_logs_pooled",
                        file_count=len(file_names),
                    ),
                }
            )
        file_count = len({str(record.get("file_name", "")) for record in scenario_records})
        scenario_reports.append(
            {
                "protocol": pooled_report.get("protocol"),
                "scenario_name": scenario_name,
                "log_root": str(Path(str(pooled_report.get("log_root", ""))) / scenario_name),
                "configured_batch_size": pooled_report.get("configured_batch_size"),
                "batch_count": len(batches),
                "concurrent_active_replay": True,
                "scenario_replay_mode": "pooled",
                "summary": _records_summary(
                    scenario_records,
                    dataset_name="new_research_logs_pooled",
                    file_count=file_count,
                ),
                "batches": batches,
                "files": scenario_audits,
            }
        )
    return scenario_reports


def _aggregate_for_batch(
    *,
    batch_size: int,
    records: list[dict[str, Any]],
    scenario_reports: list[dict[str, Any]],
    scenario_replay_mode: str,
) -> dict[str, Any]:
    total_steps = len(records)
    total_hit_counts = _hit_counts(records)
    scenario_hit_rates = [report["summary"]["hit_at_k"] for report in scenario_reports]
    return {
        "batch_size": batch_size,
        "scenario_replay_mode": scenario_replay_mode,
        "scenario_count": len(scenario_reports),
        "file_count": sum(report["summary"]["file_count"] for report in scenario_reports),
        "total_steps": total_steps,
        "step_micro": {
            "hit_counts": total_hit_counts,
            "hit_at_k": _hit_rates(total_hit_counts, total_steps),
        },
        "scenario_macro": {
            "hit_at_k": _macro_mean(scenario_hit_rates),
        },
        "query_macro": _query_macro(records),
        "batch_macro": _batch_macro(scenario_reports),
        "latency": _latency_summary(records, "prediction_time_ms"),
        "wave_latency": _wave_latency_summary(records),
        "adaptive_speculation": _adaptive_speculation_summary(records),
        "online_adaptive_branching": _online_adaptive_branch_summary(
            records,
            history_scope=(
                "pooled" if scenario_replay_mode == "pooled" else "scenario"
            ),
        ),
        "online_expert_advice": _expert_advice_summary(records),
        "feasibility_diagnostics": _feasibility_diagnostics(records),
        "target_set_metrics": _target_set_summary(records),
        "query_learning_curve": _query_learning_curve(records),
        "online_convergence": _online_convergence_summary(records),
        "confidence_intervals": _hit_confidence_intervals(records),
        "label_shuffle_negative_control": _label_shuffle_negative_control(records),
        "leakage_audit": _leakage_audit_summary(records, scenario_reports),
        "metric_validation": _validate_records(records, scenario_reports),
    }


def evaluate_suite(
    *,
    root: Path,
    scenarios: list[str],
    batch_sizes: list[int],
    policy_mode: str,
    cascade_margin_threshold: float = 150.0,
    cascade_slow_policy_mode: str = "balanced",
    agent_id_salt: str = "",
    query_order_seed: str = "",
    scenario_replay_mode: str = "separate",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_records: list[dict[str, Any]] = []
    batch_reports: list[dict[str, Any]] = []
    policy_config: dict[str, Any]
    if policy_mode in {"cascade", "expert_cascade"}:
        policy_config = {
            "fast": _policy_config("fast"),
            "slow": _policy_config(cascade_slow_policy_mode),
            "cascade_margin_threshold": cascade_margin_threshold,
            "cascade_slow_policy_mode": cascade_slow_policy_mode,
            "online_expert_advice": policy_mode == "expert_cascade",
        }
    else:
        policy_config = _policy_config(policy_mode)
    if agent_id_salt:
        if policy_mode in {"cascade", "expert_cascade"}:
            policy_config["fast"]["agent_id_salt"] = agent_id_salt
            policy_config["slow"]["agent_id_salt"] = agent_id_salt
        else:
            policy_config["agent_id_salt"] = agent_id_salt

    for batch_size in batch_sizes:
        scenario_reports: list[dict[str, Any]] = []
        batch_records: list[dict[str, Any]] = []
        if scenario_replay_mode == "pooled":
            if policy_mode in {"cascade", "expert_cascade"}:
                raise ValueError("pooled scenario replay is not implemented for cascade modes")
            pooled_report, records, audit_records = evaluate_pooled_concurrent_batches(
                root=root,
                scenarios=scenarios,
                batch_size=batch_size,
                query_order_seed=query_order_seed,
                **policy_config,
            )
            scenario_reports = _scenario_reports_from_pooled(
                pooled_report=pooled_report,
                records=records,
                audit_records=audit_records,
                scenarios=scenarios,
            )
            batch_records.extend(records)
        elif scenario_replay_mode == "separate":
            for scenario_name in scenarios:
                log_root = root / scenario_name
                if policy_mode in {"cascade", "expert_cascade"}:
                    fast_report, fast_records, _ = evaluate_concurrent_batches(
                        log_root,
                        batch_size=batch_size,
                        query_order_seed=query_order_seed,
                        **policy_config["fast"],
                    )
                    slow_report, slow_records, _ = evaluate_concurrent_batches(
                        log_root,
                        batch_size=batch_size,
                        query_order_seed=query_order_seed,
                        **policy_config["slow"],
                    )
                    records = _merge_cascade_records(
                        fast_records=fast_records,
                        slow_records=slow_records,
                        margin_threshold=cascade_margin_threshold,
                        slow_policy_mode=cascade_slow_policy_mode,
                    )
                    for record in records:
                        record["scenario_name"] = scenario_name
                        record["suite_batch_size"] = batch_size
                        record["scenario_replay_mode"] = "separate"
                    if policy_mode == "expert_cascade":
                        records = _apply_online_expert_advice(records)
                    report = _cascade_report_from_records(
                        fast_report=fast_report,
                        slow_report=slow_report,
                        records=records,
                        policy_mode=policy_mode,
                        margin_threshold=cascade_margin_threshold,
                        slow_policy_mode=cascade_slow_policy_mode,
                    )
                    report["online_expert_advice"] = _expert_advice_summary(records)
                else:
                    report, records, _ = evaluate_concurrent_batches(
                        log_root,
                        batch_size=batch_size,
                        query_order_seed=query_order_seed,
                        **policy_config,
                    )
                report["scenario_name"] = scenario_name
                report["scenario_replay_mode"] = "separate"
                for record in records:
                    record["scenario_name"] = scenario_name
                    record["suite_batch_size"] = batch_size
                    record["scenario_replay_mode"] = "separate"
                for batch in report["batches"]:
                    batch["scenario_name"] = scenario_name
                scenario_reports.append(report)
                batch_records.extend(records)
        else:
            raise ValueError(f"unknown scenario_replay_mode: {scenario_replay_mode}")
        suite_records.extend(batch_records)
        batch_reports.append(
            {
                "batch_size": batch_size,
                "scenario_replay_mode": scenario_replay_mode,
                "scenarios": [
                    {
                        "scenario_name": report["scenario_name"],
                        "protocol": report["protocol"],
                        "concurrent_active_replay": report.get("concurrent_active_replay", False),
                        "log_root": report["log_root"],
                        "summary": report["summary"],
                        "batch_count": report["batch_count"],
                        "cascade_stage_summary": report.get("cascade_stage_summary", {}),
                        "scenario_replay_mode": report.get(
                            "scenario_replay_mode",
                            scenario_replay_mode,
                        ),
                    }
                    for report in scenario_reports
                ],
                "aggregate": _aggregate_for_batch(
                    batch_size=batch_size,
                    records=batch_records,
                    scenario_reports=scenario_reports,
                    scenario_replay_mode=scenario_replay_mode,
                ),
            }
        )

    report = {
        "protocol": "next_agent_active_batch_snapshot_online_suite",
        "root": str(root),
        "scenarios": scenarios,
        "batch_sizes": batch_sizes,
        "policy_mode": policy_mode,
        "scenario_replay_mode": scenario_replay_mode,
        "policy_claim": _policy_claim(policy_mode),
        "policy_config": policy_config,
        "agent_id_salt": agent_id_salt,
        "query_order_seed": query_order_seed,
        "cascade_margin_threshold": (
            cascade_margin_threshold if policy_mode in {"cascade", "expert_cascade"} else None
        ),
        "cascade_slow_policy_mode": (
            cascade_slow_policy_mode if policy_mode in {"cascade", "expert_cascade"} else None
        ),
        "metric_scope": (
            "Strict next-agent speculative steps only: current event is main_turn and the next "
            "event is real agent work. Scheduler recovery events are excluded."
        ),
        "aggregation_definitions": {
            "step_micro": "sum correct strict predictive steps across all scenarios / sum strict predictive steps",
            "scenario_macro": "mean of per-scenario hit rates, with each scenario weighted equally",
            "query_macro": "mean of per-query hit rates, excluding queries with zero strict predictive steps",
            "batch_macro": "mean of per-batch hit rates, with each concurrent batch weighted equally",
        },
        "method_references": _method_references(),
        "external_data_source_candidates": _external_data_source_candidates(),
        "batches": batch_reports,
    }
    return report, suite_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run concurrent batch next-agent evaluation over multiple scenario folders."
    )
    parser.add_argument("--root", type=Path, default=Path("results/research"))
    parser.add_argument("--scenarios", nargs="+", default=["coding", "research"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 3, 4])
    parser.add_argument(
        "--policy-mode",
        choices=(
            "full",
            "balanced",
            "compact",
            "fast",
            "strict_online",
            "strict_all_agents",
            "strict_no_memory",
            "strict_no_memory_all_agents",
            "strict_profile_online",
            "strict_all_agents_profile_online",
            "strict_profile_event_online",
            "strict_all_agents_profile_event_online",
            "skeptical_profile_event_online",
            "semantic_skeptical_profile_event_online",
            "structural_event_online",
            "structural_all_agents_event_online",
            "strict_id_permutation",
            "strict_all_agents_id_permutation",
            "strict_profile_id_permutation",
            "strict_all_agents_profile_id_permutation",
            "strict_profile_event_id_permutation",
            "strict_all_agents_profile_event_id_permutation",
            "skeptical_profile_event_id_permutation",
            "semantic_skeptical_profile_event_id_permutation",
            "structural_event_id_permutation",
            "structural_all_agents_event_id_permutation",
            "robust",
            "cascade",
            "expert_cascade",
        ),
        default="strict_online",
        help=(
            "full keeps all visible features; balanced disables visible agent-context "
            "similarity; compact keeps only task-profile similarity; fast disables "
            "profile/text similarity and trusts only graph, role, schedule, and "
            "online transition memory; strict_online removes hand-written role, "
            "schedule, profile, visible-context, and graph-order priors so only "
            "visible graph candidates plus observed online transitions remain; "
            "strict_all_agents also removes visible graph/tool-schema candidate "
            "narrowing and ranks every visible agent; "
            "strict_no_memory variants disable completed-query memory; "
            "strict_profile_online variants replace raw agent-id transition memory "
            "with profile/roster-conditioned completed-query memory; "
            "strict_profile_event variants additionally update local memory after "
            "each already-observed step inside the same query; "
            "skeptical_profile_event disables graph order, roster positions, candidate "
            "narrowing, and raw local transition memory; "
            "semantic_skeptical_profile_event adds visible task/profile token matching "
            "while keeping those skeptical exclusions; "
            "structural_event variants add the visible graph/tool-schema candidate "
            "order prior as a deployable structural feature; "
            "strict_id_permutation variants consistently rename worker ids inside "
            "each file to audit dependence on fixed agent numbering; "
            "robust adds an episodic completed-query profile/position prior; "
            "cascade runs fast first and escalates to a slower policy only for "
            "low-margin predictions; expert_cascade adds a strict online "
            "expert-advice reranker on top of cascade."
        ),
    )
    parser.add_argument("--cascade-margin-threshold", type=float, default=150.0)
    parser.add_argument(
        "--cascade-slow-policy-mode",
        choices=("balanced", "full", "compact", "robust"),
        default="balanced",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--timing-path", type=Path, default=None)
    parser.add_argument("--convergence-plot-path", type=Path, default=None)
    parser.add_argument(
        "--agent-id-salt",
        default="",
        help="Optional salt for deterministic per-file agent-id permutation audits.",
    )
    parser.add_argument(
        "--query-order-seed",
        default="",
        help="Optional deterministic shuffle seed for query-file replay order.",
    )
    parser.add_argument(
        "--scenario-replay-mode",
        choices=("separate", "pooled"),
        default="separate",
        help=(
            "separate resets online memory for each scenario folder; pooled mixes all "
            "scenario folders into one replay with one shared online memory and no "
            "scenario label passed to the predictor."
        ),
    )
    args = parser.parse_args()

    report_path = args.report_path or _default_report_path(
        args.root,
        args.batch_sizes,
        args.policy_mode,
    )
    timing_path = args.timing_path or _default_timing_path(
        args.root,
        args.batch_sizes,
        args.policy_mode,
    )
    convergence_plot_path = args.convergence_plot_path or _default_convergence_plot_path(
        args.root,
        args.batch_sizes,
        args.policy_mode,
    )
    report, records = evaluate_suite(
        root=args.root,
        scenarios=args.scenarios,
        batch_sizes=args.batch_sizes,
        policy_mode=args.policy_mode,
        cascade_margin_threshold=args.cascade_margin_threshold,
        cascade_slow_policy_mode=args.cascade_slow_policy_mode,
        agent_id_salt=args.agent_id_salt,
        query_order_seed=args.query_order_seed,
        scenario_replay_mode=args.scenario_replay_mode,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_convergence_svg(report, convergence_plot_path)

    print(f"protocol={report['protocol']}")
    print(f"root={args.root}")
    print(f"policy_mode={args.policy_mode}")
    print(f"scenario_replay_mode={args.scenario_replay_mode}")
    for batch_report in report["batches"]:
        aggregate = batch_report["aggregate"]
        step_micro = aggregate["step_micro"]["hit_at_k"]
        scenario_macro = aggregate["scenario_macro"]["hit_at_k"]
        query_macro = aggregate["query_macro"]["hit_at_k"]
        batch_macro = aggregate["batch_macro"]["hit_at_k"]
        validation = aggregate["metric_validation"]
        adaptive_policies = {
            policy["name"]: policy
            for policy in aggregate["adaptive_speculation"]["policies"]
        }
        balanced_policy = adaptive_policies["balanced_margin_top2_15"]
        aggressive_policy = adaptive_policies["aggressive_margin_top2_80_top3_5"]
        wave_latency = aggregate["wave_latency"]
        feasibility = aggregate["feasibility_diagnostics"]
        negative_control = aggregate["label_shuffle_negative_control"]
        convergence = aggregate["online_convergence"]["overall"]
        online_branch_policies = {
            policy["name"]: policy
            for policy in aggregate["online_adaptive_branching"]["policies"]
        }
        online_target80 = online_branch_policies["online_target80_budget1.20"]
        print(
            f"batch_size={aggregate['batch_size']} "
            f"steps={aggregate['total_steps']} "
            f"step_micro_hit@1={step_micro['1']:.4f} "
            f"scenario_macro_hit@1={scenario_macro['1']:.4f} "
            f"query_macro_hit@1={query_macro['1']:.4f} "
            f"batch_macro_hit@1={batch_macro['1']:.4f} "
            f"valid={validation['stored_matches_recomputed']} "
            f"invalid_records={validation['invalid_metric_record_count']}"
        )
        print(
            f"  balanced_top2_margin<=15 coverage={balanced_policy['coverage']:.4f} "
            f"avg_branches={balanced_policy['average_branches']:.3f} "
            f"extra_branches={balanced_policy['extra_branch_rate_vs_top1']:.3f}"
        )
        print(
            f"  aggressive_top2_margin<=80_top3_margin<=5 "
            f"coverage={aggressive_policy['coverage']:.4f} "
            f"avg_branches={aggressive_policy['average_branches']:.3f} "
            f"extra_branches={aggressive_policy['extra_branch_rate_vs_top1']:.3f}"
        )
        print(
            f"  wave_prediction_serial_mean={wave_latency['serial_prediction_wave_ms_mean']:.4f}ms "
            f"parallel_mean={wave_latency['parallel_prediction_wave_ms_mean']:.4f}ms"
        )
        print(
            f"  online_target80_budget1.20 coverage={online_target80['coverage']:.4f} "
            f"avg_branches={online_target80['average_branches']:.3f} "
            f"stable_pm5pp_after={convergence['stable_after_queries']['within_5pp_of_final_for_all_later']}"
        )
        print(
            f"  feasibility top2_oracle={feasibility['candidate_top2_oracle']:.4f} "
            f"top3_oracle={feasibility['candidate_top3_oracle']:.4f} "
            f"component_oracle={feasibility['component_oracle']:.4f}"
        )
        print(
            f"  label_shuffle_negative_control_hit@1="
            f"{negative_control['hit_at_k']['1']:.4f}"
        )
        expert_summary = aggregate.get("online_expert_advice") or {}
        if expert_summary.get("enabled"):
            print(
                f"  online_expert_advice applied={expert_summary['applied_records']} "
                f"base_hit@1={expert_summary['base_hit_at_k']['1']:.4f} "
                f"final_hit@1={expert_summary['final_hit_at_k']['1']:.4f}"
            )
        if args.policy_mode in {"cascade", "expert_cascade"}:
            stage_summary = {}
            for scenario in batch_report["scenarios"]:
                scenario_summary = scenario.get("cascade_stage_summary") or {}
                for stage, count in (scenario_summary.get("stage_counts") or {}).items():
                    stage_summary[stage] = stage_summary.get(stage, 0) + count
            print(f"  cascade_stage_counts={stage_summary}")
    print(f"report={report_path}")
    print(f"timing={timing_path}")
    print(f"convergence_plot={convergence_plot_path}")


if __name__ == "__main__":
    main()
