from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_new_log_cold_start import (
    NextAgentGlobalMemory,
    NextAgentPolicyState,
    OnlinePatternMemory,
    PolicyState,
    _agent_profile_text,
    _agent_roster_position,
    _apply_agent_id_view,
    _classify_event,
    _discover_raw_event_logs,
    _event_request_content,
    _evaluate_file,
    _expected_next_agent_targets,
    _is_predictive_next_agent_step,
    _primary_records,
    _rank_next_agents,
    _read_jsonl,
    _ordered_agents,
    _request_contains_exact_agent_marker,
    _summarize,
    _target_set_hit,
    _target_set_recall,
)


def _batched(paths: list[Path], batch_size: int) -> list[list[Path]]:
    return [paths[index : index + batch_size] for index in range(0, len(paths), batch_size)]


def _apply_query_order_view(paths: list[Path], *, seed_text: str) -> list[Path]:
    if not seed_text:
        return paths
    seed = int.from_bytes(
        hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _default_output_path(log_root: Path, batch_size: int, suffix: str) -> Path:
    return log_root / f"next_agent_concurrent_b{batch_size}_{suffix}"


def _apply_transition_updates(
    next_global_memory: NextAgentGlobalMemory,
    transition_updates: list[dict[str, Any]],
) -> None:
    for update in transition_updates:
        next_global_memory.update(
            current_agent=str(update["current_agent"]),
            observed_next_agent=str(update["observed_next_agent"]),
            outgoing_agents=[str(agent_id) for agent_id in update.get("outgoing_agents", [])],
            round_index=int(update.get("round_index", 0)),
            source_turn_count=int(update.get("source_turn_count", 0)),
            current_agent_profile_text=str(update.get("current_agent_profile_text", "")),
            observed_next_agent_profile_text=str(
                update.get("observed_next_agent_profile_text", "")
            ),
            task_profile_text=str(update.get("task_profile_text", "")),
            current_agent_roster_position=(
                int(update["current_agent_roster_position"])
                if update.get("current_agent_roster_position") is not None
                else None
            ),
            observed_next_agent_roster_position=(
                int(update["observed_next_agent_roster_position"])
                if update.get("observed_next_agent_roster_position") is not None
                else None
            ),
        )


def _evaluate_active_batch(
    batch_paths: list[Path],
    *,
    batch_index: int,
    first_file_index: int,
    memory_snapshot: OnlinePatternMemory,
    next_memory_snapshot: NextAgentGlobalMemory,
    use_cross_file_memory: bool,
    cross_file_stat_weight: float,
    enable_adaptive_cross_file_prior: bool,
    enable_profile_signature_transition_prior: bool,
    enable_roster_position_transition_prior: bool,
    enable_episodic_cross_file_prior: bool,
    adaptive_cross_file_weight: float,
    adaptive_cross_file_min_support: int,
    adaptive_cross_file_min_confidence: float,
    adaptive_cross_file_min_profile_stability: float,
    enable_graph_order_prior: bool,
    enable_role_workflow_prior: bool,
    enable_local_transition_memory: bool,
    online_evidence_mode: str,
    candidate_scope: str,
    online_feedback_scope: str,
    enable_research_schedule_prior: bool,
    enable_research_meta_prior: bool,
    enable_profile_similarity_prior: bool,
    profile_similarity_mode: str,
    enable_idf_profile_prior: bool,
    enable_online_pair_calibration: bool,
    pair_calibration_margin: int,
    include_visible_agent_context: bool,
    visible_context_similarity_weight: float,
    visible_context_length_weight: float,
    agent_id_view: str,
    agent_id_salt: str,
    dataset_name: str,
    completed_queries_before_batch: int,
    configured_batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    active_queries: list[dict[str, Any]] = []
    for batch_position, path in enumerate(batch_paths):
        events = _read_jsonl(path)
        events = _apply_agent_id_view(
            events,
            agent_id_view=agent_id_view,
            seed_text=path.name,
            agent_id_salt=agent_id_salt,
        )
        first_agents = (events[0].get("agents") or {}) if events else {}
        agents = _ordered_agents(list(first_agents.keys()))
        active_queries.append(
            {
                "path": path,
                "events": events,
                "agents": agents,
                "first_agents": first_agents,
                "state": PolicyState(),
                "next_state": NextAgentPolicyState(),
                "next_memory": copy.deepcopy(next_memory_snapshot),
                "step_index": 0,
                "timing_records": [],
                "sequence": [],
                "transition_updates": [],
                "pair_calibration_counts": defaultdict(Counter),
                "forbidden_current_text_hits": 0,
                "file_index": first_file_index + batch_position,
                "batch_position": batch_position,
            }
        )

    while True:
        progressed = False
        for query in active_queries:
            events = query["events"]
            step_index = query["step_index"]
            if step_index >= max(len(events) - 1, 0):
                continue
            progressed = True
            event = events[step_index]
            expected_agent_ids, expected_target_source = _expected_next_agent_targets(
                events,
                step_index,
            )
            expected = expected_agent_ids[0] if expected_agent_ids else ""
            expected_event_type = _classify_event(events[step_index + 1])
            next_state: NextAgentPolicyState = query["next_state"]
            state: PolicyState = query["state"]
            pair_calibration_counts = query["pair_calibration_counts"]
            started = time.perf_counter()
            ranked, scores, prediction_metadata = _rank_next_agents(
                event=event,
                next_state=next_state,
                next_global_memory=query["next_memory"],
                use_cross_file_memory=use_cross_file_memory,
                cross_file_stat_weight=cross_file_stat_weight,
                enable_adaptive_cross_file_prior=enable_adaptive_cross_file_prior,
                enable_profile_signature_transition_prior=enable_profile_signature_transition_prior,
                enable_roster_position_transition_prior=enable_roster_position_transition_prior,
                enable_episodic_cross_file_prior=enable_episodic_cross_file_prior,
                adaptive_cross_file_weight=adaptive_cross_file_weight,
                adaptive_cross_file_min_support=adaptive_cross_file_min_support,
                adaptive_cross_file_min_confidence=adaptive_cross_file_min_confidence,
                adaptive_cross_file_min_profile_stability=adaptive_cross_file_min_profile_stability,
                enable_graph_order_prior=enable_graph_order_prior,
                enable_role_workflow_prior=enable_role_workflow_prior,
                enable_local_transition_memory=enable_local_transition_memory,
                online_evidence_mode=online_evidence_mode,
                candidate_scope=candidate_scope,
                enable_research_schedule_prior=enable_research_schedule_prior,
                enable_research_meta_prior=enable_research_meta_prior,
                enable_profile_similarity_prior=enable_profile_similarity_prior,
                profile_similarity_mode=profile_similarity_mode,
                enable_idf_profile_prior=enable_idf_profile_prior,
                include_visible_agent_context=include_visible_agent_context,
                visible_context_similarity_weight=visible_context_similarity_weight,
                visible_context_length_weight=visible_context_length_weight,
                agents=query["agents"],
            )
            base_ranked = list(ranked)
            prediction_metadata["base_scores"] = dict(scores)
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
                str(prediction_metadata["event_type"]) == "main_turn"
                and bool(expected_agent_ids)
            )
            hit_at = {
                "1": _target_set_hit(expected_agent_ids, ranked, 1),
                "2": _target_set_hit(expected_agent_ids, ranked, 2),
                "3": _target_set_hit(expected_agent_ids, ranked, 3),
                "5": _target_set_hit(expected_agent_ids, ranked, 5),
            }
            target_recall_at = {
                "1": _target_set_recall(expected_agent_ids, ranked, 1),
                "2": _target_set_recall(expected_agent_ids, ranked, 2),
                "3": _target_set_recall(expected_agent_ids, ranked, 3),
                "5": _target_set_recall(expected_agent_ids, ranked, 5),
            }
            request_text = json.dumps(event.get("request") or {}, ensure_ascii=False)
            if _request_contains_exact_agent_marker(request_text, expected):
                query["forbidden_current_text_hits"] += 1
            record = {
                "dataset_name": dataset_name,
                "file_name": query["path"].name,
                "source_log_path": str(query["path"]),
                "workflow_id": str(event.get("workflow_id") or query["path"].stem),
                "step_index": step_index,
                "event_id": str(event.get("event_id") or ""),
                "expected_agent_id": expected,
                "expected_agent_ids": expected_agent_ids,
                "expected_agent_count": len(expected_agent_ids),
                "expected_target_source": expected_target_source,
                "prediction_target": "next_agent",
                "agent_id_view": agent_id_view,
                "agent_id_salt": agent_id_salt,
                "current_agent_id": prediction_metadata["current_agent_id"],
                "event_type": prediction_metadata["event_type"],
                "expected_event_type": expected_event_type,
                "counted_for_metric": counted_for_metric,
                "prediction": ranked[:5],
                "base_prediction": base_ranked[:5],
                "candidate_count": len(prediction_metadata.get("candidate_agents") or query["agents"]),
                "prediction_time_ms": prediction_time_ms,
                "history_size_before_prediction": len(state.sequence),
                "global_bigram_size_before_prediction": len(memory_snapshot.bigram),
                "ranking_reason": prediction_metadata["ranking_reason"],
                "graph_source": prediction_metadata["graph_source"],
                "candidate_scope": prediction_metadata.get("candidate_scope", candidate_scope),
                "outgoing_agents": prediction_metadata["outgoing_agents"],
                "profile_score_candidates": prediction_metadata.get(
                    "profile_score_candidates",
                    [],
                ),
                "semantic_profile_scores": prediction_metadata.get("semantic_profile_scores", {}),
                "memory_profile_scores": prediction_metadata.get("memory_profile_scores", {}),
                "prompt_profile_scores": prediction_metadata.get("prompt_profile_scores", {}),
                "task_idf_profile_scores": prediction_metadata.get("task_idf_profile_scores", {}),
                "memory_idf_profile_scores": prediction_metadata.get("memory_idf_profile_scores", {}),
                "prompt_idf_profile_scores": prediction_metadata.get("prompt_idf_profile_scores", {}),
                "schedule_prior_scores": prediction_metadata.get("schedule_prior_scores", {}),
                "role_workflow_prior_scores": prediction_metadata.get("role_workflow_prior_scores", {}),
                "meta_prior_scores": prediction_metadata.get("meta_prior_scores", {}),
                "adaptive_cross_file_scores": prediction_metadata.get("adaptive_cross_file_scores", {}),
                "contextual_cross_file_scores": prediction_metadata.get("contextual_cross_file_scores", {}),
                "episodic_cross_file_scores": prediction_metadata.get("episodic_cross_file_scores", {}),
                "profile_signature_transition_scores": prediction_metadata.get(
                    "profile_signature_transition_scores",
                    {},
                ),
                "roster_position_transition_scores": prediction_metadata.get(
                    "roster_position_transition_scores",
                    {},
                ),
                "first_target_profile_scores": prediction_metadata.get("first_target_profile_scores", {}),
                "adaptive_cross_file_profile_stability": prediction_metadata.get(
                    "adaptive_cross_file_profile_stability",
                    0.0,
                ),
                "visible_context_scores": prediction_metadata.get("visible_context_scores", {}),
                "source_turn_count": prediction_metadata.get("source_turn_count", 0),
                "round_index": prediction_metadata.get("round_index", 0),
                "pair_calibration_signature": prediction_metadata.get("pair_calibration_signature", []),
                "pair_calibration_counts": prediction_metadata.get("pair_calibration_counts", {}),
                "reranker_scores": {},
                "hit_at_k": hit_at,
                "target_recall_at_k": target_recall_at,
                "base_top_scores": {
                    agent_id: prediction_metadata.get("base_scores", {}).get(agent_id, 0.0)
                    for agent_id in base_ranked[:5]
                },
                "top_scores": {agent_id: scores.get(agent_id, 0.0) for agent_id in ranked[:5]},
                "file_index": query["file_index"],
                "batch_index": batch_index,
                "batch_position": query["batch_position"],
                "configured_batch_size": configured_batch_size,
                "concurrent_batch_size": len(batch_paths),
                "completed_queries_before_batch": completed_queries_before_batch,
                "use_cross_file_memory": use_cross_file_memory,
                "concurrent_active_replay": True,
            }
            query["timing_records"].append(record)

            current_agent = str(event.get("agent_id") or "")
            state.update_after_prediction(event, current_agent)
            query["sequence"].append(current_agent)
            if online_feedback_scope == "event":
                next_state.update_after_transition(
                    event=event,
                    event_type=prediction_metadata["event_type"],
                    observed_next_agent=expected,
                    outgoing_agents=prediction_metadata["outgoing_agents"],
                    learn_transition=counted_for_metric,
                )
            if prediction_metadata["event_type"] == "main_turn" and counted_for_metric:
                pair_signature = prediction_metadata.get("pair_calibration_signature") or []
                if online_feedback_scope == "event" and len(pair_signature) == 2:
                    pair_calibration_counts[(pair_signature[0], pair_signature[1])][
                        expected == pair_signature[1]
                    ] += 1
                transition_update = {
                    "current_agent": current_agent,
                    "observed_next_agent": expected,
                    "outgoing_agents": prediction_metadata["outgoing_agents"],
                    "round_index": prediction_metadata.get("round_index", 0),
                    "source_turn_count": prediction_metadata.get("source_turn_count", 0),
                    "current_agent_profile_text": _agent_profile_text(event, current_agent),
                    "observed_next_agent_profile_text": _agent_profile_text(event, expected),
                    "task_profile_text": str(event.get("task_profile") or ""),
                    "current_agent_roster_position": _agent_roster_position(
                        event,
                        current_agent,
                    ),
                    "observed_next_agent_roster_position": _agent_roster_position(
                        event,
                        expected,
                    ),
                }
                query["transition_updates"].append(transition_update)
                if online_feedback_scope == "event":
                    query["next_memory"].update(**transition_update)

            query["step_index"] += 1
        if not progressed:
            break

    timing_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    batch_outputs: list[dict[str, Any]] = []
    for query in active_queries:
        events = query["events"]
        first_agents = query["first_agents"]
        timing_records.extend(query["timing_records"])
        audit = {
            "file_name": query["path"].name,
            "source_log_path": str(query["path"]),
            "event_count": len(events),
            "decision_step_count": sum(
                1 for record in query["timing_records"] if record.get("counted_for_metric")
            ),
            "agent_count": len(query["agents"]),
            "agents": query["agents"],
            "prediction_target": "next_agent",
            "agent_id_view": agent_id_view,
            "agent_id_salt": agent_id_salt,
            "request_messages_not_used": False,
            "current_agent_id_used_as_visible_input": True,
            "current_agent_id_used_only_as_label_after_prediction": False,
            "prediction_transition_candidates_present": False,
            "graph_transition_candidates_present": False,
            "visible_agent_context_used": include_visible_agent_context,
            "visible_agent_context_scope": (
                "current event agents[*].context snapshot only; no next event context"
                if include_visible_agent_context
                else "not used"
            ),
            "first_main_target": query["next_state"].local_first_main_target,
            "second_main_target": query["next_state"].local_second_main_target,
            "forbidden_current_request_text_contains_expected_agent_count": query[
                "forbidden_current_text_hits"
            ],
            "first_event_has_empty_history": bool(
                query["timing_records"]
                and query["timing_records"][0]["history_size_before_prediction"] == 0
            ),
            "file_index": query["file_index"],
            "batch_index": batch_index,
            "batch_position": query["batch_position"],
            "configured_batch_size": configured_batch_size,
            "concurrent_batch_size": len(batch_paths),
            "completed_queries_before_batch": completed_queries_before_batch,
            "use_cross_file_memory": use_cross_file_memory,
            "concurrent_active_replay": True,
        }
        profile_texts = {
            agent_id: str(agent.get("profile") or "")
            for agent_id, agent in first_agents.items()
            if isinstance(agent, dict)
        }
        task_profile_text = str(events[0].get("task_profile") or "") if events else ""
        audit_records.append(audit)
        batch_outputs.append(
            {
                "sequence": query["sequence"],
                "audit": audit,
                "profile_texts": profile_texts,
                "task_profile_text": task_profile_text,
                "transition_updates": query["transition_updates"],
            }
        )
    return timing_records, audit_records, batch_outputs


def evaluate_concurrent_batches(
    log_root: Path,
    *,
    batch_size: int,
    use_cross_file_memory: bool,
    cross_file_stat_weight: float,
    enable_adaptive_cross_file_prior: bool,
    enable_profile_signature_transition_prior: bool,
    enable_roster_position_transition_prior: bool,
    enable_episodic_cross_file_prior: bool,
    adaptive_cross_file_weight: float,
    adaptive_cross_file_min_support: int,
    adaptive_cross_file_min_confidence: float,
    adaptive_cross_file_min_profile_stability: float,
    enable_graph_order_prior: bool,
    enable_role_workflow_prior: bool,
    enable_local_transition_memory: bool,
    online_evidence_mode: str,
    candidate_scope: str,
    online_feedback_scope: str,
    enable_research_schedule_prior: bool,
    enable_research_meta_prior: bool,
    enable_profile_similarity_prior: bool,
    profile_similarity_mode: str,
    enable_idf_profile_prior: bool,
    enable_online_pair_calibration: bool,
    pair_calibration_margin: int,
    include_visible_agent_context: bool,
    visible_context_similarity_weight: float,
    visible_context_length_weight: float,
    agent_id_view: str,
    agent_id_salt: str,
    query_order_seed: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    paths = _apply_query_order_view(
        _discover_raw_event_logs(log_root),
        seed_text=f"{query_order_seed}:{log_root}" if query_order_seed else "",
    )
    global_memory = OnlinePatternMemory()
    next_global_memory = NextAgentGlobalMemory()
    all_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    dataset_name = "new_research_logs"
    completed_queries_before_batch = 0

    for batch_index, batch_paths in enumerate(_batched(paths, batch_size)):
        batch_started = time.perf_counter()
        memory_snapshot = copy.deepcopy(global_memory)
        next_memory_snapshot = copy.deepcopy(next_global_memory)
        batch_record_start = len(all_records)
        timing_records, active_audits, batch_outputs = _evaluate_active_batch(
            batch_paths,
            batch_index=batch_index,
            first_file_index=batch_index * batch_size,
            memory_snapshot=memory_snapshot,
            next_memory_snapshot=next_memory_snapshot,
            use_cross_file_memory=use_cross_file_memory,
            cross_file_stat_weight=cross_file_stat_weight,
            enable_adaptive_cross_file_prior=enable_adaptive_cross_file_prior,
            enable_profile_signature_transition_prior=enable_profile_signature_transition_prior,
            enable_roster_position_transition_prior=enable_roster_position_transition_prior,
            enable_episodic_cross_file_prior=enable_episodic_cross_file_prior,
            adaptive_cross_file_weight=adaptive_cross_file_weight,
            adaptive_cross_file_min_support=adaptive_cross_file_min_support,
            adaptive_cross_file_min_confidence=adaptive_cross_file_min_confidence,
            adaptive_cross_file_min_profile_stability=adaptive_cross_file_min_profile_stability,
            enable_graph_order_prior=enable_graph_order_prior,
            enable_role_workflow_prior=enable_role_workflow_prior,
            enable_local_transition_memory=enable_local_transition_memory,
            online_evidence_mode=online_evidence_mode,
            candidate_scope=candidate_scope,
            online_feedback_scope=online_feedback_scope,
            enable_research_schedule_prior=enable_research_schedule_prior,
            enable_research_meta_prior=enable_research_meta_prior,
            enable_profile_similarity_prior=enable_profile_similarity_prior,
            profile_similarity_mode=profile_similarity_mode,
            enable_idf_profile_prior=enable_idf_profile_prior,
            enable_online_pair_calibration=enable_online_pair_calibration,
            pair_calibration_margin=pair_calibration_margin,
            include_visible_agent_context=include_visible_agent_context,
            visible_context_similarity_weight=visible_context_similarity_weight,
            visible_context_length_weight=visible_context_length_weight,
            agent_id_view=agent_id_view,
            agent_id_salt=agent_id_salt,
            dataset_name=dataset_name,
            completed_queries_before_batch=completed_queries_before_batch,
            configured_batch_size=batch_size,
        )
        all_records.extend(timing_records)
        audit_records.extend(active_audits)

        batch_wall_time_ms = (time.perf_counter() - batch_started) * 1000.0
        current_batch_records = all_records[batch_record_start:]
        current_primary_records = _primary_records(
            current_batch_records,
            prediction_target="next_agent",
        )
        batch_summary = _summarize(
            current_primary_records,
            dataset_name=dataset_name,
            file_count=len(batch_paths),
        )
        batch_records.append(
            {
                "batch_index": batch_index,
                "file_count": len(batch_paths),
                "completed_queries_before_batch": completed_queries_before_batch,
                "batch_wall_time_ms": batch_wall_time_ms,
                "file_names": [path.name for path in batch_paths],
                "summary": batch_summary,
            }
        )

        if use_cross_file_memory:
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
    non_online_prior_flags = {
        "graph_order_prior": bool(enable_graph_order_prior),
        "role_workflow_prior": bool(enable_role_workflow_prior),
        "visible_order_prior": bool(enable_research_schedule_prior),
        "cross_query_start_prior": bool(enable_research_meta_prior),
        "profile_similarity_prior": bool(enable_profile_similarity_prior),
        "idf_profile_prior": bool(enable_idf_profile_prior),
        "visible_agent_context": bool(include_visible_agent_context),
        "profile_signature_transition_prior": bool(enable_profile_signature_transition_prior),
        "roster_position_transition_prior": bool(enable_roster_position_transition_prior),
        "episodic_profile_position_prior": bool(enable_episodic_cross_file_prior),
        "local_transition_memory": bool(enable_local_transition_memory),
        "heuristic_online_evidence": online_evidence_mode != "transition_only",
        "event_feedback_within_query": online_feedback_scope == "event",
    }
    strict_online_compatible = not any(non_online_prior_flags.values())
    profile_conditioned_compatible = (
        (
            bool(enable_profile_signature_transition_prior)
            or bool(enable_roster_position_transition_prior)
        )
        and not any(
            value
            for key, value in non_online_prior_flags.items()
            if key
            not in {
                "profile_signature_transition_prior",
                "roster_position_transition_prior",
                "local_transition_memory",
            }
        )
    )
    input_features = [
        "current executing agent id",
    ]
    if candidate_scope == "visible_graph":
        input_features.append("visible graph/tool-schema candidate set")
    elif candidate_scope == "all_agents":
        input_features.append("all visible agents as candidates")
    elif candidate_scope == "all_worker_agents":
        input_features.append("all visible non-PLANNER agents as candidates")
    if online_evidence_mode == "transition_only":
        input_features.append("online current_agent->next_agent transition counts")
    else:
        input_features.append("online transition and position/frequency heuristics")
    if online_feedback_scope == "query":
        input_features.append("feedback applied only after each query completes")
    else:
        input_features.append("feedback applied after each scored event")
    if use_cross_file_memory and enable_adaptive_cross_file_prior:
        input_features.append("completed-query transition memory before the current batch")
    if enable_graph_order_prior:
        input_features.append("static graph candidate-order prior")
    if enable_role_workflow_prior:
        input_features.append("hand-written role workflow prior")
    if enable_research_schedule_prior:
        input_features.append("visible order/schedule prior")
    if enable_research_meta_prior:
        input_features.append("cross-query start/meta prior")
    if enable_profile_similarity_prior:
        input_features.append(f"profile similarity prior ({profile_similarity_mode})")
    if enable_idf_profile_prior:
        input_features.append("IDF profile prior")
    if include_visible_agent_context:
        input_features.append("current-event visible agents[*].context snapshot")
    if enable_episodic_cross_file_prior:
        input_features.append("episodic profile/position prior")
    if enable_profile_signature_transition_prior:
        input_features.append("profile-signature transition memory")
    if enable_roster_position_transition_prior:
        input_features.append("visible roster-position transition memory")
    if enable_local_transition_memory:
        input_features.append("local same-query current_agent->next_agent transition memory")
    else:
        input_features.append("no local raw agent-id transition memory")
    report = {
        "protocol": "next_agent_active_batch_snapshot_online",
        "log_root": str(log_root),
        "configured_batch_size": batch_size,
        "batch_count": len(batch_records),
        "concurrent_active_replay": True,
        "policy_claim": {
            "claim": (
                "online_learning_no_candidate_narrowing"
                if strict_online_compatible and candidate_scope == "all_agents"
                else "online_learning_with_visible_candidate_scope"
                if strict_online_compatible
                else "profile_conditioned_online_memory"
                if profile_conditioned_compatible
                else "structural_prior_baseline"
            ),
            "reporting_rule": (
                "May be reported as strict online-learning accuracy, but "
                "candidate_scope must be disclosed because visible graph/tool-schema "
                "candidates can strongly narrow the search space."
                if strict_online_compatible
                else (
                    "May be reported as profile/roster-conditioned online memory: raw "
                    "cross-query agent-id transition counts are disabled and completed "
                    "queries are aggregated by visible profile signatures and/or "
                    "visible worker roster positions."
                )
                if profile_conditioned_compatible
                else (
                    "Do not report as online-learning-only accuracy because at least "
                    "one structural/static prior is enabled."
                )
            ),
        },
        "non_online_prior_flags": non_online_prior_flags,
        "online_evidence_mode": online_evidence_mode,
        "online_feedback_scope": online_feedback_scope,
        "candidate_scope": candidate_scope,
        "agent_id_view": agent_id_view,
        "agent_id_salt": agent_id_salt,
        "query_order_seed": query_order_seed,
        "input_view": "; ".join(input_features)
        + (
            "; batch members share one pre-batch memory snapshot and cannot learn "
            "from each other before the batch is scored"
        ),
        "candidate_space": (
            "all visible agents from the current file's static agents roster"
            if candidate_scope == "all_agents"
            else "all visible non-PLANNER agents from the current file's static agents roster"
            if candidate_scope == "all_worker_agents"
            else "visible target_agent_id enum or inferred graph outgoing agents plus PLANNER"
        ),
        "prediction_target": "next_agent",
        "online_reranker_enabled": False,
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
        "visible_graph_order_prior_enabled": enable_graph_order_prior,
        "visible_order_prior_enabled": enable_research_schedule_prior,
        "visible_role_workflow_prior_enabled": enable_role_workflow_prior,
        "local_transition_memory_enabled": enable_local_transition_memory,
        "contextual_cross_file_prior_enabled": (
            enable_adaptive_cross_file_prior if use_cross_file_memory else False
        ),
        "profile_signature_transition_prior_enabled": (
            enable_profile_signature_transition_prior if use_cross_file_memory else False
        ),
        "roster_position_transition_prior_enabled": (
            enable_roster_position_transition_prior if use_cross_file_memory else False
        ),
        "episodic_cross_file_prior_enabled": (
            enable_episodic_cross_file_prior if use_cross_file_memory else False
        ),
        "cross_query_start_prior_enabled": enable_research_meta_prior,
        "profile_similarity_prior_enabled": enable_profile_similarity_prior,
        "profile_similarity_mode": (
            profile_similarity_mode if enable_profile_similarity_prior else "off"
        ),
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
        ),
        "leakage_control": (
            "For each batch, all files are scored from deep copies of the same pre-batch memory. "
            "Observed labels and transition updates are applied to global memory only after every "
            "file in the batch has been scored."
        ),
        "summary": _summarize(output_records, dataset_name=dataset_name, file_count=len(paths)),
        "batches": batch_records,
        "files": audit_records,
    }
    return report, output_records, audit_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate strict next-agent prediction when several queries enter concurrently. "
            "Each batch is scored from one pre-batch memory snapshot."
        )
    )
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--timing-path", type=Path, default=None)
    parser.add_argument("--audit-path", type=Path, default=None)
    parser.add_argument(
        "--use-cross-file-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cross-file-stat-weight", type=float, default=0.0)
    parser.add_argument(
        "--enable-adaptive-cross-file-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-episodic-cross-file-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use completed-query profile/position transition examples as an additional "
            "no-leak prior. Updates happen only after a batch has been scored."
        ),
    )
    parser.add_argument(
        "--enable-profile-signature-transition-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Learn source-profile-signature to target-profile-signature transitions "
            "from completed queries, without raw cross-query agent-id transitions."
        ),
    )
    parser.add_argument(
        "--enable-roster-position-transition-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Learn visible worker-roster-position transitions from completed queries. "
            "This uses structural order in agents, not raw agent-id labels."
        ),
    )
    parser.add_argument("--adaptive-cross-file-weight", type=float, default=30.0)
    parser.add_argument("--adaptive-cross-file-min-support", type=int, default=1)
    parser.add_argument("--adaptive-cross-file-min-confidence", type=float, default=0.4)
    parser.add_argument("--adaptive-cross-file-min-profile-stability", type=float, default=0.0)
    parser.add_argument(
        "--enable-visible-order-prior",
        dest="enable_visible_order_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-graph-order-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a static bonus to earlier visible graph/tool-schema candidates. "
            "Disable for strict online-learning evaluation."
        ),
    )
    parser.add_argument(
        "--enable-role-workflow-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply hand-written role workflow priors. This is a structural-prior "
            "ablation, not online learning."
        ),
    )
    parser.add_argument(
        "--enable-local-transition-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use already observed same-query current_agent->next_agent transition "
            "counts after those steps have occurred. Disable for skeptical audits."
        ),
    )
    parser.add_argument(
        "--online-evidence-mode",
        choices=("transition_only", "heuristic"),
        default="transition_only",
        help=(
            "transition_only uses only learned current_agent->next_agent transition "
            "counts; heuristic also enables visible-history and position/frequency "
            "heuristics."
        ),
    )
    parser.add_argument(
        "--online-feedback-scope",
        choices=("query", "event"),
        default="query",
        help=(
            "query freezes online memory inside each query and updates only after the "
            "query completes; event updates after each scored event."
        ),
    )
    parser.add_argument(
        "--candidate-scope",
        choices=("visible_graph", "all_agents", "all_worker_agents"),
        default="visible_graph",
        help=(
            "visible_graph uses the current visible tool-schema/graph outgoing set; "
            "all_agents disables that candidate narrowing and ranks every visible "
            "agent in the file; all_worker_agents ranks every visible non-PLANNER agent."
        ),
    )
    parser.add_argument(
        "--agent-id-view",
        choices=("original", "per_file_permutation"),
        default="original",
        help=(
            "original keeps raw agent ids. per_file_permutation consistently renames "
            "worker agent ids inside each query file, with a different deterministic "
            "permutation per file, to audit dependence on fixed cross-query agent ids."
        ),
    )
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
        "--enable-cross-query-start-prior",
        dest="enable_cross_query_start_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-profile-similarity-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--profile-similarity-mode",
        choices=("full", "task"),
        default="full",
    )
    parser.add_argument(
        "--enable-idf-profile-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-online-pair-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pair-calibration-margin", type=int, default=1)
    parser.add_argument(
        "--include-visible-agent-context",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--visible-context-similarity-weight", type=float, default=10.0)
    parser.add_argument("--visible-context-length-weight", type=float, default=0.0)
    args = parser.parse_args()

    report_path = args.report_path or _default_output_path(
        args.log_root,
        args.batch_size,
        "report.json",
    )
    timing_path = args.timing_path or _default_output_path(
        args.log_root,
        args.batch_size,
        "timing.jsonl",
    )
    audit_path = args.audit_path or _default_output_path(
        args.log_root,
        args.batch_size,
        "audit.json",
    )

    report, timing_records, audit_records = evaluate_concurrent_batches(
        args.log_root,
        batch_size=args.batch_size,
        use_cross_file_memory=args.use_cross_file_memory,
        cross_file_stat_weight=args.cross_file_stat_weight,
        enable_adaptive_cross_file_prior=args.enable_adaptive_cross_file_prior,
        enable_profile_signature_transition_prior=args.enable_profile_signature_transition_prior,
        enable_roster_position_transition_prior=args.enable_roster_position_transition_prior,
        enable_episodic_cross_file_prior=args.enable_episodic_cross_file_prior,
        adaptive_cross_file_weight=args.adaptive_cross_file_weight,
        adaptive_cross_file_min_support=args.adaptive_cross_file_min_support,
        adaptive_cross_file_min_confidence=args.adaptive_cross_file_min_confidence,
        adaptive_cross_file_min_profile_stability=args.adaptive_cross_file_min_profile_stability,
        enable_graph_order_prior=args.enable_graph_order_prior,
        enable_role_workflow_prior=args.enable_role_workflow_prior,
        enable_local_transition_memory=args.enable_local_transition_memory,
        online_evidence_mode=args.online_evidence_mode,
        candidate_scope=args.candidate_scope,
        online_feedback_scope=args.online_feedback_scope,
        enable_research_schedule_prior=args.enable_visible_order_prior,
        enable_research_meta_prior=args.enable_cross_query_start_prior,
        enable_profile_similarity_prior=args.enable_profile_similarity_prior,
        profile_similarity_mode=args.profile_similarity_mode,
        enable_idf_profile_prior=args.enable_idf_profile_prior,
        enable_online_pair_calibration=args.enable_online_pair_calibration,
        pair_calibration_margin=args.pair_calibration_margin,
        include_visible_agent_context=args.include_visible_agent_context,
        visible_context_similarity_weight=args.visible_context_similarity_weight,
        visible_context_length_weight=args.visible_context_length_weight,
        agent_id_view=args.agent_id_view,
        agent_id_salt=args.agent_id_salt,
        query_order_seed=args.query_order_seed,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in timing_records) + "\n",
        encoding="utf-8",
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_records, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report["summary"]
    hit_at_k = summary["hit_at_k"]
    print(f"protocol={report['protocol']}")
    print(f"log_root={args.log_root}")
    print(
        f"batch_size={args.batch_size} batches={report['batch_count']} "
        f"files={summary['file_count']} steps={summary['total_steps']}"
    )
    print(
        f"hit@1={hit_at_k.get('1', 0.0):.4f} "
        f"hit@2={hit_at_k.get('2', 0.0):.4f} "
        f"hit@3={hit_at_k.get('3', 0.0):.4f} "
        f"hit@5={hit_at_k.get('5', 0.0):.4f}"
    )
    print(
        f"prediction_ms_mean={summary['prediction_time_ms_mean']:.4f} "
        f"prediction_ms_p95={summary['prediction_time_ms_p95']:.4f} "
        f"prediction_ms_max={summary['prediction_time_ms_max']:.4f}"
    )
    print(f"report={report_path}")
    print(f"timing={timing_path}")
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()
