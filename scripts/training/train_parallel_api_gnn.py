from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark import BenchmarkEvaluator, MultiAgentBenchAdapter, load_parallel_api_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train PredictDesign GNN variants on results/parallel_api_test with a fixed 80/20 holdout split."
        )
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "parallel_api_test"),
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "parallel_api_test" / "gnn_holdout_report.json"),
    )
    parser.add_argument(
        "--split-summary-path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "parallel_api_test" / "gnn_holdout_split_summary.json"),
    )
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--message-reduce-modes", nargs="*", default=["attention"])
    parser.add_argument("--state-updaters", nargs="*", default=["gru", "mdp"])
    parser.add_argument(
        "--gnn-types",
        nargs="*",
        default=["hybrid", "relational_transformer", "gcn", "graphsage", "gat"],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sentence-transformer-path", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--sentence-transformer-dim", type=int, default=384)
    parser.add_argument(
        "--sentence-transformer-freeze",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    adapter = MultiAgentBenchAdapter(
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    corpus = load_parallel_api_corpus(args.results_root, adapter)
    evaluator = BenchmarkEvaluator(
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_fraction=args.train_fraction,
        seed=args.seed,
        sentence_transformer_path=args.sentence_transformer_path,
        sentence_transformer_dim=args.sentence_transformer_dim,
        sentence_transformer_freeze=args.sentence_transformer_freeze,
    )

    dataset_splits = corpus.dataset_splits(evaluator.trainer)
    combined_split = corpus.combined_split(evaluator.trainer)
    combination_specs = evaluator.combination_specs(
        message_reduce_modes=tuple(args.message_reduce_modes),
        state_updaters=tuple(args.state_updaters),
        gnn_types=tuple(args.gnn_types),
    )

    report_records = []
    for spec in combination_specs:
        system = evaluator.build_system(spec)
        evaluator.fit_system(
            system,
            combined_split.train_episodes,
            use_eval_for_training=False,
        )
        report_records.append(
            evaluator.evaluate_system(
                dataset_name="parallel_api_test_all",
                system=system,
                eval_episodes=combined_split.eval_episodes,
                spec=spec,
                train_episode_count=len(combined_split.train_episodes),
                eval_episode_count=len(combined_split.eval_episodes),
                cv_fold_count=1,
            )
        )
        for dataset_name, split in dataset_splits.items():
            report_records.append(
                evaluator.evaluate_system(
                    dataset_name=dataset_name,
                    system=system,
                    eval_episodes=split.eval_episodes,
                    spec=spec,
                    train_episode_count=len(combined_split.train_episodes),
                    eval_episode_count=len(split.eval_episodes),
                    cv_fold_count=1,
                )
            )

    split_summary = {
        "results_root": str(corpus.root_path),
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "combined": {
            "train_episode_count": len(combined_split.train_episodes),
            "eval_episode_count": len(combined_split.eval_episodes),
            "train_episode_ids": [episode.episode_id for episode in combined_split.train_episodes],
            "eval_episode_ids": [episode.episode_id for episode in combined_split.eval_episodes],
        },
        "datasets": {
            dataset_name: {
                "source_count": corpus.datasets[dataset_name].source_count,
                "episode_count": corpus.datasets[dataset_name].episode_count,
                "train_episode_count": len(split.train_episodes),
                "eval_episode_count": len(split.eval_episodes),
                "train_episode_ids": [episode.episode_id for episode in split.train_episodes],
                "eval_episode_ids": [episode.episode_id for episode in split.eval_episodes],
            }
            for dataset_name, split in dataset_splits.items()
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_report(report_path, report_records)

    split_summary_path = Path(args.split_summary_path)
    split_summary_path.parent.mkdir(parents=True, exist_ok=True)
    split_summary_path.write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"report={report_path}")
    print(f"split_summary={split_summary_path}")
    print(
        "combined_split "
        f"train={len(combined_split.train_episodes)} "
        f"eval={len(combined_split.eval_episodes)}"
    )
    for dataset_name, split in dataset_splits.items():
        print(
            f"{dataset_name} "
            f"train={len(split.train_episodes)} "
            f"eval={len(split.eval_episodes)}"
        )
    for item in report_records:
        print(
            f"{item.dataset_name}\t{item.message_reduce}\t{item.state_updater}\t"
            f"{item.gnn_type}\thit@1={item.hit_at_k.get('1', 0.0):.4f}\t"
            f"one_step@1={item.one_step_hit_at_k.get('1', 0.0):.4f}\t"
            f"subgraph_f1={item.subgraph_f1:.4f}"
        )


if __name__ == "__main__":
    main()

