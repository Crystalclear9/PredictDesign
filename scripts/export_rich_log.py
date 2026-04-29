from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark.multiagentbench import MultiAgentBenchAdapter
from predictdesign.benchmark.rich_log import train_mlp_on_rich_log, write_rich_log


def _write_original_manifest(
    output_dir: Path,
    *,
    coding_outputs: list[str],
    research_outputs: list[str],
    werewolf_dirs: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "original_logs_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "coding_output_jsonl": coding_outputs,
                "research_output_jsonl": research_outputs,
                "werewolf_checkpoint_dirs": werewolf_dirs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export coding/research/werewolf outputs to a rich JSONL log, optionally with an MLP check."
    )
    parser.add_argument("--coding-output-jsonl", nargs="*", default=[])
    parser.add_argument("--research-output-jsonl", nargs="*", default=[])
    parser.add_argument("--werewolf-checkpoint-dir", nargs="*", default=[])
    parser.add_argument("--output-log", type=str, default=str(PROJECT_ROOT / "results" / "rich_log.jsonl"))
    parser.add_argument(
        "--log-version",
        choices=("rich", "original", "both"),
        default="rich",
        help="rich saves the detailed normalized JSONL; original saves a manifest pointing to raw scenario outputs.",
    )
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-mlp", action="store_true")
    parser.add_argument("--mlp-output-dir", type=str, default=str(PROJECT_ROOT / "results" / "rich_log_mlp"))
    parser.add_argument("--mlp-max-samples", type=int, default=100)
    parser.add_argument("--mlp-epochs", type=int, default=60)
    parser.add_argument("--mlp-feature-dim", type=int, default=256)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mlp-label-mode",
        choices=("action_type", "action_signature"),
        default="action_type",
    )
    args = parser.parse_args()

    adapter = MultiAgentBenchAdapter(
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    episodes = []
    for path in args.coding_output_jsonl:
        episodes.extend(adapter.load_coding_from_output_jsonl(path))
    for path in args.research_output_jsonl:
        episodes.extend(adapter.load_research_from_output_jsonl(path))
    for path in args.werewolf_checkpoint_dir:
        episodes.extend(adapter.load_werewolf_from_checkpoints(path))
    if not episodes:
        raise SystemExit(
            "No inputs were provided. Pass at least one --coding-output-jsonl, "
            "--research-output-jsonl, or --werewolf-checkpoint-dir."
        )
    if args.train_mlp and args.log_version == "original":
        raise SystemExit("--train-mlp requires --log-version rich or --log-version both.")

    output_log = Path(args.output_log).resolve()
    manifest_path = None
    rich_result = None
    if args.log_version in {"original", "both"}:
        manifest_path = _write_original_manifest(
            output_log.parent,
            coding_outputs=args.coding_output_jsonl,
            research_outputs=args.research_output_jsonl,
            werewolf_dirs=args.werewolf_checkpoint_dir,
        )
        print(json.dumps({"original_logs_manifest": str(manifest_path)}, indent=2, ensure_ascii=False))
    if args.log_version in {"rich", "both"}:
        rich_result = write_rich_log(
            output_log,
            episodes,
            context_dim=args.context_dim,
            device=args.device,
        )
        print(json.dumps(asdict(rich_result), indent=2, ensure_ascii=False))

    if args.train_mlp:
        assert rich_result is not None
        mlp_result = train_mlp_on_rich_log(
            log_path=rich_result.path,
            output_dir=args.mlp_output_dir,
            max_samples=args.mlp_max_samples,
            feature_dim=args.mlp_feature_dim,
            hidden_dim=args.mlp_hidden_dim,
            epochs=args.mlp_epochs,
            learning_rate=args.mlp_learning_rate,
            seed=args.seed,
            label_mode=args.mlp_label_mode,
            device=args.mlp_device,
        )
        print(json.dumps(asdict(mlp_result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
