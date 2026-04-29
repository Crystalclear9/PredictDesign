from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark.evaluator import BenchmarkEvaluator
from predictdesign.benchmark.multiagentbench import MultiAgentBenchAdapter
from predictdesign.benchmark.rich_log import train_mlp_on_rich_log, write_rich_log
from predictdesign.config import LLMApiConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PredictDesign combinations on MultiAgentBench outputs."
    )
    parser.add_argument("--coding-output-jsonl", type=str, default=None)
    parser.add_argument("--research-output-jsonl", type=str, default=None)
    parser.add_argument("--werewolf-checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--report-path",
        type=str,
        default="/data0/fanjiarun/PredictDesign/results/multiagentbench_accuracy_report.json",
    )
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--gnn-types", nargs="*", default=["gcn", "graphsage", "gat"])
    parser.add_argument("--hit-k-values", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--rich-log-path", type=str, default=None)
    parser.add_argument("--train-rich-log-mlp", action="store_true")
    parser.add_argument("--mlp-output-dir", type=str, default=None)
    parser.add_argument("--mlp-max-samples", type=int, default=100)
    parser.add_argument("--mlp-epochs", type=int, default=60)
    parser.add_argument(
        "--mlp-label-mode",
        choices=("action_type", "action_signature"),
        default="action_type",
    )
    parser.add_argument("--llm-api-key", type=str, default=LLMApiConfig().api_key)
    parser.add_argument("--llm-base-url", type=str, default=LLMApiConfig().base_url)
    parser.add_argument("--llm-model", type=str, default=LLMApiConfig().model)
    parser.add_argument("--llm-temperature", type=float, default=LLMApiConfig().temperature)
    parser.add_argument("--llm-max-tokens", type=int, default=LLMApiConfig().max_tokens)
    parser.add_argument("--llm-timeout", type=float, default=LLMApiConfig().timeout)
    args = parser.parse_args()

    adapter = MultiAgentBenchAdapter(context_dim=args.context_dim)
    evaluator = BenchmarkEvaluator(
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_fraction=args.train_fraction,
        hit_k_values=tuple(args.hit_k_values),
        llm_api_config=LLMApiConfig(
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
        ),
    )
    all_results = []
    all_episodes = []

    if args.coding_output_jsonl:
        episodes = adapter.load_coding_from_output_jsonl(args.coding_output_jsonl)
        all_episodes.extend(episodes)
        all_results.extend(
            evaluator.evaluate_dataset(
                "coding",
                episodes,
                gnn_types=tuple(args.gnn_types),
            )
        )
    if args.research_output_jsonl:
        episodes = adapter.load_research_from_output_jsonl(args.research_output_jsonl)
        all_episodes.extend(episodes)
        all_results.extend(
            evaluator.evaluate_dataset(
                "research",
                episodes,
                gnn_types=tuple(args.gnn_types),
            )
        )
    if args.werewolf_checkpoint_dir:
        episodes = adapter.load_werewolf_from_checkpoints(args.werewolf_checkpoint_dir)
        all_episodes.extend(episodes)
        all_results.extend(
            evaluator.evaluate_dataset(
                "werewolf",
                episodes,
                gnn_types=tuple(args.gnn_types),
            )
        )

    if not all_results:
        raise SystemExit(
            "No benchmark inputs were provided. Pass --coding-output-jsonl, "
            "--research-output-jsonl and/or --werewolf-checkpoint-dir."
        )

    report_path = Path(args.report_path)
    evaluator.save_report(report_path, all_results)
    print(report_path)
    rich_log_path = args.rich_log_path
    if args.train_rich_log_mlp and rich_log_path is None:
        rich_log_path = str(report_path.parent / "rich_log.jsonl")
    if rich_log_path:
        rich_result = write_rich_log(
            rich_log_path,
            all_episodes,
            context_dim=args.context_dim,
            device="cpu",
        )
        print(
            f"rich_log={rich_result.path} records={rich_result.record_count} "
            f"episodes={rich_result.episode_count}"
        )
        if args.train_rich_log_mlp:
            mlp_output_dir = args.mlp_output_dir or str(report_path.parent / "rich_log_mlp")
            mlp_result = train_mlp_on_rich_log(
                log_path=rich_result.path,
                output_dir=mlp_output_dir,
                max_samples=args.mlp_max_samples,
                epochs=args.mlp_epochs,
                label_mode=args.mlp_label_mode,
            )
            print(f"mlp_report={mlp_result.report_path}")
            print(f"mlp_chart={mlp_result.chart_path}")
    for item in all_results:
        hit_metrics = " ".join(
            f"hit@{hit_k}={float(item.hit_at_k.get(str(hit_k), 0.0)):.4f}"
            for hit_k in item.hit_ks
        )
        print(
            f"{item.dataset_name}\t{item.message_reduce}\t{item.state_updater}\t"
            f"{item.gnn_type}\t{hit_metrics}\t"
            f"top1_hits={item.correct_steps}/{item.total_steps}"
        )


if __name__ == "__main__":
    main()
