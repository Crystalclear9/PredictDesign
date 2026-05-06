from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark import BenchmarkEvaluator, MultiAgentBenchAdapter, load_parallel_api_corpus
from predictdesign.benchmark.evaluator import CombinationSpec


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained PredictDesign GNN checkpoint and evaluate it on a results root "
            "containing coding/research/werewolf outputs."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "external_hitk_report.json"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)

    adapter = MultiAgentBenchAdapter(
        context_dim=int(checkpoint["context_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        device=args.device,
    )
    corpus = load_parallel_api_corpus(args.results_root, adapter)
    evaluator = BenchmarkEvaluator(
        context_dim=int(checkpoint["context_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        device=args.device,
        train_epochs=0,
        seed=int(checkpoint.get("seed", 7)),
        sentence_transformer_path=str(checkpoint.get("sentence_transformer_path", "all-MiniLM-L6-v2")),
        sentence_transformer_dim=int(checkpoint.get("sentence_transformer_dim", 384)),
        sentence_transformer_freeze=bool(checkpoint.get("sentence_transformer_freeze", True)),
    )
    spec = CombinationSpec(**checkpoint["spec"])
    system = evaluator.build_system(spec)
    system.load_state_dict(checkpoint["state_dict"])
    system.eval()

    train_episode_count = int(checkpoint.get("train_episode_count", 0))
    results = []
    combined_episodes = []

    for dataset_name, dataset in corpus.datasets.items():
        combined_episodes.extend(dataset.episodes)
        results.append(
            evaluator.evaluate_system(
                dataset_name=dataset_name,
                system=system,
                eval_episodes=dataset.episodes,
                spec=spec,
                train_episode_count=train_episode_count,
                eval_episode_count=len(dataset.episodes),
                cv_fold_count=1,
            )
        )

    results.append(
        evaluator.evaluate_system(
            dataset_name="all_external_results",
            system=system,
            eval_episodes=combined_episodes,
            spec=spec,
            train_episode_count=train_episode_count,
            eval_episode_count=len(combined_episodes),
            cv_fold_count=1,
        )
    )

    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_report(report_path, results)

    print(f"report={report_path}")
    for item in results:
        hit_metrics = " ".join(
            f"hit@{hit_k}={float(item.hit_at_k.get(str(hit_k), 0.0)):.4f}"
            for hit_k in item.hit_ks
        )
        print(
            f"{item.dataset_name}\t{item.message_reduce}\t{item.state_updater}\t"
            f"{item.gnn_type}\t{hit_metrics}\t"
            f"top1_hits={item.correct_steps}/{item.total_steps}"
        )
    print(json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

