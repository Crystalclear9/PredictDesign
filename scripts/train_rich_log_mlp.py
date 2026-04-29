from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark.rich_log import train_mlp_on_rich_log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a lightweight MLP baseline from a PredictDesign rich JSONL log."
    )
    parser.add_argument("--log-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results" / "rich_log_mlp"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=384,
        help="Only used as the fallback text-vector dimension if sentence-transformers cannot be loaded.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=0,
        help="Legacy single-width override. Set 0 to use the automatic hidden-dimension heuristic.",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit hidden dimensions, for example: --hidden-dims 1024 512",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--label-mode",
        choices=("action_type", "action_signature"),
        default="action_type",
        help="action_type is the stable preliminary check; action_signature is stricter.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--sentence-transformer-model",
        type=str,
        default="all-MiniLM-L6-v2",
    )
    parser.add_argument("--sentence-transformer-batch-size", type=int, default=64)
    parser.add_argument("--repeat-count", type=int, default=5)
    args = parser.parse_args()

    result = train_mlp_on_rich_log(
        log_path=args.log_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        train_fraction=args.train_fraction,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        label_mode=args.label_mode,
        device=args.device,
        sentence_transformer_model=args.sentence_transformer_model,
        sentence_transformer_batch_size=args.sentence_transformer_batch_size,
        hidden_dims=tuple(args.hidden_dims) if args.hidden_dims else None,
        repeat_count=args.repeat_count,
    )
    print(result.report_path)
    print(result.chart_path)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
