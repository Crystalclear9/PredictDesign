from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark.evaluator import BenchmarkEvaluator
from predictdesign.benchmark.multiagentbench import MultiAgentBenchAdapter
from predictdesign.config import LLMApiConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PredictDesign combinations on MultiAgentBench outputs."
    )
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

    if args.research_output_jsonl:
        all_results.extend(
            evaluator.evaluate_dataset(
                "research",
                adapter.load_research_from_output_jsonl(args.research_output_jsonl),
                gnn_types=tuple(args.gnn_types),
            )
        )
    if args.werewolf_checkpoint_dir:
        all_results.extend(
            evaluator.evaluate_dataset(
                "werewolf",
                adapter.load_werewolf_from_checkpoints(args.werewolf_checkpoint_dir),
                gnn_types=tuple(args.gnn_types),
            )
        )

    if not all_results:
        raise SystemExit(
            "No benchmark inputs were provided. Pass --research-output-jsonl and/or --werewolf-checkpoint-dir."
        )

    report_path = Path(args.report_path)
    evaluator.save_report(report_path, all_results)
    print(report_path)
    for item in all_results:
        print(
            f"{item.dataset_name}\t{item.message_reduce}\t{item.state_updater}\t"
            f"{item.gnn_type}\tacc={item.accuracy:.4f}\t{item.correct_steps}/{item.total_steps}"
        )


if __name__ == "__main__":
    main()
