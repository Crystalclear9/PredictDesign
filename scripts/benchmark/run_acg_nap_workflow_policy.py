from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark.acg_nap_workflow_policy import (  # noqa: E402
    evaluate_acg_nap_workflow_policy,
)
from predictdesign.paths import ACG_NAP_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a strict non-leaking ACG-NAP workflow/candidate policy. "
            "This does not modify or train the GNN predictor."
        )
    )
    parser.add_argument("--acg-nap-root", type=Path, default=ACG_NAP_ROOT)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["coding", "research"],
        help="ACG-NAP scenario folders to evaluate.",
    )
    parser.add_argument("--max-files-per-dataset", type=int, default=15)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help=(
            "Episode split fraction used only to choose the holdout files. "
            "The policy itself does not train on the train split."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report path. Omit it to avoid writing benchmark artifacts.",
    )
    args = parser.parse_args()

    root = args.acg_nap_root.resolve()
    results = evaluate_acg_nap_workflow_policy(
        root,
        scenarios=tuple(args.scenarios),
        max_files_per_dataset=args.max_files_per_dataset,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps([item.to_dict() for item in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        "mode=strict_workflow_candidate_policy "
        "uses=query_text,candidate_actions,node_profiles,previous_observed_actions "
        "no_node_context=True no_source_output=True no_latest_output=True "
        "no_runtime_text=True no_current_label=True no_future_label=True "
        "model_unchanged=True"
    )
    print(f"acg_nap_root={root}")
    print(f"report={args.report_path if args.report_path is not None else 'disabled'}")
    for item in results:
        print(
            f"{item.dataset_name:<8} steps={item.total_steps:<4} "
            f"hit@1={item.hit_at_1:.4f} hit@3={item.hit_at_3:.4f} hit@5={item.hit_at_5:.4f}"
        )


if __name__ == "__main__":
    main()
