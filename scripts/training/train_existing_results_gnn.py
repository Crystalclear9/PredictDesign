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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a single PredictDesign GNN configuration on all existing scenario logs "
            "under a results root and save a reusable checkpoint."
        )
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "parallel_api_test"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "parallel_api_test" / "trained_gnn_checkpoint.pt"),
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "parallel_api_test" / "trained_gnn_checkpoint_summary.json"),
    )
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--message-reduce-mode", type=str, default="attention")
    parser.add_argument("--state-updater", type=str, default="gru")
    parser.add_argument("--gnn-type", type=str, default="hybrid")
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
        seed=args.seed,
        sentence_transformer_path=args.sentence_transformer_path,
        sentence_transformer_dim=args.sentence_transformer_dim,
        sentence_transformer_freeze=args.sentence_transformer_freeze,
    )
    spec = evaluator.combination_specs(
        message_reduce_modes=(args.message_reduce_mode,),
        state_updaters=(args.state_updater,),
        gnn_types=(args.gnn_type,),
    )[0]
    system = evaluator.build_system(spec)

    training_episodes = []
    dataset_summary: dict[str, dict[str, int | list[str]]] = {}
    for dataset_name, dataset in corpus.datasets.items():
        training_episodes.extend(dataset.episodes)
        dataset_summary[dataset_name] = {
            "source_count": dataset.source_count,
            "episode_count": dataset.episode_count,
            "episode_ids": [episode.episode_id for episode in dataset.episodes],
        }

    evaluator.fit_system(system, training_episodes, use_eval_for_training=False)

    checkpoint_path = Path(args.checkpoint_path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": system.state_dict(),
        "spec": asdict(spec),
        "train_results_root": str(Path(args.results_root).resolve()),
        "context_dim": args.context_dim,
        "hidden_dim": args.hidden_dim,
        "device": args.device,
        "sentence_transformer_path": args.sentence_transformer_path,
        "sentence_transformer_dim": args.sentence_transformer_dim,
        "sentence_transformer_freeze": args.sentence_transformer_freeze,
        "train_epochs": args.train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_episode_count": len(training_episodes),
        "train_episode_ids": [episode.episode_id for episode in training_episodes],
        "dataset_summary": dataset_summary,
        "trainer_summary": (
            asdict(evaluator.trainer.last_fit_summary)
            if evaluator.trainer.last_fit_summary is not None
            else None
        ),
    }
    torch.save(payload, checkpoint_path)

    summary_path = Path(args.summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "state_dict"
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"checkpoint={checkpoint_path}")
    print(f"summary={summary_path}")
    print(f"train_episodes={len(training_episodes)}")
    for dataset_name, dataset in corpus.datasets.items():
        print(
            f"{dataset_name}\tsources={dataset.source_count}\tepisodes={dataset.episode_count}"
        )


if __name__ == "__main__":
    main()

