from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark.run_new_log_cold_start import _latency_summary
from scripts.benchmark.run_new_log_concurrent_batches import evaluate_concurrent_batches


_RECOVERY_EVENT_TYPES = {
    "continuation",
    "planner_summary",
    "planner_continue",
    "planner",
}


def _default_report_path(root: Path, batch_sizes: list[int]) -> Path:
    suffix = "_".join(f"b{batch_size}" for batch_size in batch_sizes)
    return root / f"next_agent_concurrent_suite_{suffix}_report.json"


def _default_timing_path(root: Path, batch_sizes: list[int]) -> Path:
    suffix = "_".join(f"b{batch_size}" for batch_size in batch_sizes)
    return root / f"next_agent_concurrent_suite_{suffix}_timing.jsonl"


def _hit_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(k): sum(
            1 for record in records if record["expected_agent_id"] in record["prediction"][:k]
        )
        for k in (1, 2, 3, 5)
    }


def _hit_rates(hit_counts: dict[str, int], total: int) -> dict[str, float]:
    return {key: (value / total if total else 0.0) for key, value in hit_counts.items()}


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
            or record.get("expected_event_type") in _RECOVERY_EVENT_TYPES
        )
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
        "hidden_training_update_fields_exported": hidden_transition_updates_exported,
        "strict_records_only": True,
    }


def _aggregate_for_batch(
    *,
    batch_size: int,
    records: list[dict[str, Any]],
    scenario_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    total_steps = len(records)
    total_hit_counts = _hit_counts(records)
    scenario_hit_rates = [report["summary"]["hit_at_k"] for report in scenario_reports]
    return {
        "batch_size": batch_size,
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
        "metric_validation": _validate_records(records, scenario_reports),
    }


def evaluate_suite(
    *,
    root: Path,
    scenarios: list[str],
    batch_sizes: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_records: list[dict[str, Any]] = []
    batch_reports: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        scenario_reports: list[dict[str, Any]] = []
        batch_records: list[dict[str, Any]] = []
        for scenario_name in scenarios:
            log_root = root / scenario_name
            report, records, _ = evaluate_concurrent_batches(
                log_root,
                batch_size=batch_size,
                use_cross_file_memory=True,
                cross_file_stat_weight=0.0,
                enable_adaptive_cross_file_prior=True,
                adaptive_cross_file_weight=30.0,
                adaptive_cross_file_min_support=1,
                adaptive_cross_file_min_confidence=0.4,
                adaptive_cross_file_min_profile_stability=0.65,
                enable_research_schedule_prior=True,
                enable_research_meta_prior=True,
                enable_idf_profile_prior=False,
                enable_online_pair_calibration=True,
                pair_calibration_margin=1,
                include_visible_agent_context=True,
                visible_context_similarity_weight=10.0,
                visible_context_length_weight=0.0,
            )
            report["scenario_name"] = scenario_name
            for record in records:
                record["scenario_name"] = scenario_name
                record["suite_batch_size"] = batch_size
            for batch in report["batches"]:
                batch["scenario_name"] = scenario_name
            scenario_reports.append(report)
            batch_records.extend(records)
        suite_records.extend(batch_records)
        batch_reports.append(
            {
                "batch_size": batch_size,
                "scenarios": [
                    {
                        "scenario_name": report["scenario_name"],
                        "log_root": report["log_root"],
                        "summary": report["summary"],
                        "batch_count": report["batch_count"],
                    }
                    for report in scenario_reports
                ],
                "aggregate": _aggregate_for_batch(
                    batch_size=batch_size,
                    records=batch_records,
                    scenario_reports=scenario_reports,
                ),
            }
        )

    report = {
        "protocol": "next_agent_batch_snapshot_online_suite",
        "root": str(root),
        "scenarios": scenarios,
        "batch_sizes": batch_sizes,
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
        "batches": batch_reports,
    }
    return report, suite_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run concurrent batch next-agent evaluation over multiple scenario folders."
    )
    parser.add_argument("--root", type=Path, default=Path("results/research"))
    parser.add_argument("--scenarios", nargs="+", default=["coding", "research"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--timing-path", type=Path, default=None)
    args = parser.parse_args()

    report_path = args.report_path or _default_report_path(args.root, args.batch_sizes)
    timing_path = args.timing_path or _default_timing_path(args.root, args.batch_sizes)
    report, records = evaluate_suite(
        root=args.root,
        scenarios=args.scenarios,
        batch_sizes=args.batch_sizes,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    print(f"protocol={report['protocol']}")
    print(f"root={args.root}")
    for batch_report in report["batches"]:
        aggregate = batch_report["aggregate"]
        step_micro = aggregate["step_micro"]["hit_at_k"]
        scenario_macro = aggregate["scenario_macro"]["hit_at_k"]
        query_macro = aggregate["query_macro"]["hit_at_k"]
        batch_macro = aggregate["batch_macro"]["hit_at_k"]
        validation = aggregate["metric_validation"]
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
    print(f"report={report_path}")
    print(f"timing={timing_path}")


if __name__ == "__main__":
    main()
