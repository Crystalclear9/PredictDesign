from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load_report(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_markdown_table(title: str, rows: list[dict[str, Any]]) -> str:
    hit_ks = rows[0].get("hit_ks", [1, 3, 5]) if rows else [1, 3, 5]
    hit_ks = [int(value) for value in hit_ks]
    lines = [
        f"## {title}",
        "",
        (
            "| Aggregation | State | GNN | "
            + " | ".join(f"Hit@{hit_k}" for hit_k in hit_ks)
            + " | "
            + " | ".join(f"One-Step Hit@{hit_k}" for hit_k in hit_ks)
            + " | Rollout Exact Acc | Rollout Subgraph Acc | Subgraph P | Subgraph R | Subgraph F1 | Top1 Hits / Total |"
        ),
        (
            "|---|---|---|"
            + "---:|" * len(hit_ks)
            + "---:|" * len(hit_ks)
            + "---:|---:|---:|---:|---:|---:|"
        ),
    ]
    ordered = sorted(
        rows,
        key=lambda item: (
            item["message_reduce"],
            item["state_updater"],
            item["gnn_type"],
        ),
    )
    for item in ordered:
        hit_at_k = item.get("hit_at_k", {})
        one_step_hit_at_k = item.get("one_step_hit_at_k", {})
        hit_values = [
            f'{float(hit_at_k.get(str(hit_k), item.get("accuracy", 0.0))):.4f}'
            for hit_k in hit_ks
        ]
        one_step_hit_values = [
            f'{float(one_step_hit_at_k.get(str(hit_k), item.get("one_step_accuracy", 0.0))):.4f}'
            for hit_k in hit_ks
        ]
        rollout_exact_accuracy = f'{float(item.get("rollout_exact_accuracy", 0.0)):.4f}'
        rollout_subgraph_accuracy = f'{float(item.get("rollout_subgraph_accuracy", 0.0)):.4f}'
        subgraph_precision = f'{float(item.get("subgraph_precision", 0.0)):.4f}'
        subgraph_recall = f'{float(item.get("subgraph_recall", 0.0)):.4f}'
        subgraph_f1 = f'{float(item.get("subgraph_f1", 0.0)):.4f}'
        ratio = f'{item["correct_steps"]} / {item["total_steps"]}'
        lines.append(
            f'| {item["message_reduce"]} | {item["state_updater"]} | {item["gnn_type"]} | '
            + " | ".join(hit_values)
            + " | "
            + " | ".join(one_step_hit_values)
            + f" | {rollout_exact_accuracy} | {rollout_subgraph_accuracy} | {subgraph_precision} | {subgraph_recall} | {subgraph_f1} | {ratio} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_combined_markdown(reports: dict[str, list[dict[str, Any]]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in reports.values():
        for item in rows:
            grouped[item["dataset_name"]].append(item)

    lines = [
        "# PredictDesign Full-Fidelity Hit@K Tables",
        "",
        f"Generated at: {_now_iso()}",
        "",
    ]
    for dataset_name in sorted(grouped):
        lines.append(_format_markdown_table(dataset_name, grouped[dataset_name]))
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor full-fidelity MultiAgentBench runs and render final tables.")
    parser.add_argument("--research-report", type=str, required=True)
    parser.add_argument("--werewolf-report", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--status-json", type=str, required=True)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    research_report = Path(args.research_report).resolve()
    werewolf_report = Path(args.werewolf_report).resolve()
    output_md = Path(args.output_md).resolve()
    status_json = Path(args.status_json).resolve()
    log_file = Path(args.log_file).resolve() if args.log_file else None
    output_md.parent.mkdir(parents=True, exist_ok=True)
    status_json.parent.mkdir(parents=True, exist_ok=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        print(message, flush=True)
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    while True:
        reports = {
            "research": _load_report(research_report),
            "werewolf": _load_report(werewolf_report),
        }
        status_payload = {
            "updated_at": _now_iso(),
            "research_report": str(research_report),
            "research_ready": reports["research"] is not None,
            "werewolf_report": str(werewolf_report),
            "werewolf_ready": reports["werewolf"] is not None,
            "output_md": str(output_md),
        }
        status_json.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        emit(
            f"[{status_payload['updated_at']}] "
            f"research_ready={status_payload['research_ready']} "
            f"werewolf_ready={status_payload['werewolf_ready']}",
        )

        if reports["research"] is not None and reports["werewolf"] is not None:
            output_md.write_text(
                _render_combined_markdown(
                    {
                        "research": reports["research"] or [],
                        "werewolf": reports["werewolf"] or [],
                    }
                ),
                encoding="utf-8",
            )
            emit(f"[{_now_iso()}] final tables written to {output_md}")
            break

        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    main()
