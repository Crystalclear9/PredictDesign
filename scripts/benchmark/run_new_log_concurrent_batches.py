from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_new_log_cold_start import (
    NextAgentGlobalMemory,
    OnlinePatternMemory,
    _discover_raw_event_logs,
    _evaluate_file,
    _primary_records,
    _summarize,
)


def _batched(paths: list[Path], batch_size: int) -> list[list[Path]]:
    return [paths[index : index + batch_size] for index in range(0, len(paths), batch_size)]


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
        )


def evaluate_concurrent_batches(
    log_root: Path,
    *,
    batch_size: int,
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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    paths = _discover_raw_event_logs(log_root)
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
        batch_outputs: list[dict[str, Any]] = []
        batch_record_start = len(all_records)

        for batch_position, path in enumerate(batch_paths):
            file_index = batch_index * batch_size + batch_position
            timing_records, sequence, audit = _evaluate_file(
                path,
                global_memory=copy.deepcopy(memory_snapshot),
                next_global_memory=copy.deepcopy(next_memory_snapshot),
                next_reranker=None,
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
                prediction_target="next_agent",
                collect_transition_updates=True,
            )

            for record in timing_records:
                record["file_index"] = file_index
                record["batch_index"] = batch_index
                record["batch_position"] = batch_position
                record["configured_batch_size"] = batch_size
                record["concurrent_batch_size"] = len(batch_paths)
                record["completed_queries_before_batch"] = completed_queries_before_batch
                record["use_cross_file_memory"] = use_cross_file_memory
            all_records.extend(timing_records)

            profile_texts = audit.pop("_agent_profile_texts", {})
            transition_updates = audit.pop("_transition_updates", [])
            audit["file_index"] = file_index
            audit["batch_index"] = batch_index
            audit["batch_position"] = batch_position
            audit["configured_batch_size"] = batch_size
            audit["concurrent_batch_size"] = len(batch_paths)
            audit["completed_queries_before_batch"] = completed_queries_before_batch
            audit["use_cross_file_memory"] = use_cross_file_memory
            audit_records.append(audit)
            batch_outputs.append(
                {
                    "sequence": sequence,
                    "audit": audit,
                    "profile_texts": profile_texts,
                    "transition_updates": transition_updates,
                }
            )

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
                )

        completed_queries_before_batch += len(batch_paths)

    output_records = _primary_records(all_records, prediction_target="next_agent")
    report = {
        "protocol": "next_agent_batch_snapshot_online",
        "log_root": str(log_root),
        "configured_batch_size": batch_size,
        "batch_count": len(batch_records),
        "input_view": (
            "current executing agent id + current request scheduling metadata + inferred prompt "
            "collaboration graph + previous observed transitions + current-event visible "
            "agents[*].context history snapshots + visible profile-derived role workflow prior + "
            "adaptive/contextual cross-file transition prior from queries completed before the "
            "batch + online cross-query start prior from completed queries before the batch + "
            "within-query online top1/top2 pair calibration; batch "
            "members share one pre-batch memory snapshot and cannot learn from each other before scoring"
        ),
        "candidate_space": "visible target_agent_id enum or inferred graph outgoing agents plus PLANNER",
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
        "visible_order_prior_enabled": enable_research_schedule_prior,
        "visible_role_workflow_prior_enabled": True,
        "contextual_cross_file_prior_enabled": (
            enable_adaptive_cross_file_prior if use_cross_file_memory else False
        ),
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
    parser.add_argument("--adaptive-cross-file-weight", type=float, default=30.0)
    parser.add_argument("--adaptive-cross-file-min-support", type=int, default=1)
    parser.add_argument("--adaptive-cross-file-min-confidence", type=float, default=0.4)
    parser.add_argument("--adaptive-cross-file-min-profile-stability", type=float, default=0.65)
    parser.add_argument(
        "--enable-visible-order-prior",
        dest="enable_visible_order_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-cross-query-start-prior",
        dest="enable_cross_query_start_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-idf-profile-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-online-pair-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pair-calibration-margin", type=int, default=1)
    parser.add_argument(
        "--include-visible-agent-context",
        action=argparse.BooleanOptionalAction,
        default=True,
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
