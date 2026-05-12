from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark import ACGNapAdapter, BenchmarkEvaluator, load_acg_nap_corpus
from predictdesign.benchmark.evaluator import CombinationSpec
from predictdesign.benchmark.trainer import BenchmarkTrainer
from predictdesign.paths import ACG_NAP_ROOT, RESULTS_ROOT
from predictdesign.prediction import GraphPredictionContext, PredictedGraphAction


@dataclass(slots=True)
class XAIRecord:
    dataset_name: str
    episode_id: str
    step_index: int
    expected: str
    predicted: str
    correct: bool
    score: float
    learned_score: float
    prior_score: float
    few_shot_score: float
    description: str


def _resolve_device(requested_device: str, *, require_cuda: bool = False) -> str:
    requested = str(requested_device or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if require_cuda:
            raise RuntimeError(
                "--require-cuda was set, but this Python environment is using a CPU-only torch build."
            )
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment is using a CPU-only torch build."
        )
    return requested_device


def _action_signature(action: PredictedGraphAction | None) -> str:
    if action is None:
        return "NONE"
    if action.source_node_id and action.target_node_id:
        relation = action.relation_type or ""
        return f"{action.source_node_id}->{action.target_node_id}:{relation}"
    return action.action_type.value


def _actions_match(predicted: PredictedGraphAction, expected: PredictedGraphAction) -> bool:
    return (
        predicted.action_type == expected.action_type
        and predicted.source_node_id == expected.source_node_id
        and predicted.target_node_id == expected.target_node_id
        and (predicted.relation_type or "") == (expected.relation_type or "")
    )


def _clone_context(
    context: GraphPredictionContext | None,
    *,
    include_query: bool = True,
    include_descriptions: bool = True,
    include_candidates: bool = True,
) -> GraphPredictionContext | None:
    if context is None:
        return None
    cloned_candidates: list[PredictedGraphAction] = []
    if include_candidates:
        for action in context.candidate_actions:
            metadata = dict(action.metadata)
            if not include_descriptions:
                metadata.pop("description", None)
            cloned_candidates.append(
                PredictedGraphAction(
                    action_type=action.action_type,
                    score=action.score,
                    effective_time=action.effective_time,
                    source_node_id=action.source_node_id,
                    target_node_id=action.target_node_id,
                    relation_type=action.relation_type,
                    role=action.role,
                    new_node_id=action.new_node_id,
                    metadata=metadata,
                )
            )
    return GraphPredictionContext(
        source_node_id=context.source_node_id,
        query_text=context.query_text if include_query else "",
        graph_profile_text=context.graph_profile_text,
        source_output_text="",
        runtime_text="",
        candidate_actions=cloned_candidates,
        metadata=dict(context.metadata),
    )


def _best_candidate_from_bundle(bundle) -> tuple[PredictedGraphAction | None, int | None]:
    if not bundle.candidate_actions or bundle.candidate_scores is None:
        return None, None
    index = int(torch.argmax(bundle.candidate_scores).item())
    return bundle.candidate_actions[index], index


def _score_at(bundle, index: int | None, values: torch.Tensor | None) -> float:
    if index is None or values is None or values.numel() <= index:
        return 0.0
    return float(values[index].detach().cpu().item())


def _candidate_hit(
    system,
    episode,
    *,
    context_mode: str,
    disable_few_shot: bool = False,
) -> tuple[dict[int, int], int, list[XAIRecord]]:
    trainer = BenchmarkTrainer(epochs=0)
    system.initialize_graph(
        nodes=episode.initial_nodes,
        edges=episode.initial_edges,
        structural_edges=episode.initial_structural_edges,
        graph_context_text=episode.initial_graph_context_text,
        structural_edge_metadata=episode.initial_structural_edge_metadata,
    )
    previous_few_shot_weight = system.config.few_shot_memory_weight
    if disable_few_shot:
        system.config.few_shot_memory_weight = 0.0
    hits = {1: 0, 3: 0, 5: 0}
    total = 0
    records: list[XAIRecord] = []
    try:
        for step_index, step in enumerate(episode.steps):
            context = _context_for_mode(step.prediction_context, context_mode)
            bundle = system.predictor.score_action_space(
                temporal_graph=system.temporal_graph,
                ctdg=system.ctdg,
                observation_time=step.observation_time,
                prediction_context=context,
            )
            expected_actions = step.supervision_actions
            ranked = []
            if bundle.candidate_actions and bundle.candidate_scores is not None:
                order = torch.argsort(bundle.candidate_scores, descending=True).tolist()
                ranked = [bundle.candidate_actions[index] for index in order]
            total += 1
            for hit_k in hits:
                if any(
                    _actions_match(predicted, expected)
                    for predicted in ranked[:hit_k]
                    for expected in expected_actions
                ):
                    hits[hit_k] += 1
            predicted, index = _best_candidate_from_bundle(bundle)
            expected = expected_actions[0] if expected_actions else None
            score = _score_at(bundle, index, bundle.candidate_scores)
            prior = _score_at(bundle, index, bundle.candidate_prior_scores)
            few_shot = _score_at(bundle, index, bundle.candidate_few_shot_scores)
            learned = score - system.config.zero_shot_prior_weight * prior
            learned -= system.config.few_shot_memory_weight * few_shot
            records.append(
                XAIRecord(
                    dataset_name=episode.dataset_name,
                    episode_id=episode.episode_id,
                    step_index=step_index,
                    expected=_action_signature(expected),
                    predicted=_action_signature(predicted),
                    correct=(
                        predicted is not None
                        and expected is not None
                        and _actions_match(predicted, expected)
                    ),
                    score=score,
                    learned_score=learned,
                    prior_score=prior,
                    few_shot_score=few_shot,
                    description=str(predicted.metadata.get("description", "")) if predicted else "",
                )
            )
            trainer.apply_observed_step(system, step, update_memory=True)
    finally:
        system.config.few_shot_memory_weight = previous_few_shot_weight
    return hits, total, records


def _context_for_mode(
    context: GraphPredictionContext | None,
    mode: str,
) -> GraphPredictionContext | None:
    if mode == "full":
        return _clone_context(context)
    if mode == "no_query":
        return _clone_context(context, include_query=False)
    if mode == "no_candidate_descriptions":
        return _clone_context(context, include_descriptions=False)
    if mode == "no_candidates":
        return _clone_context(context, include_candidates=False)
    raise ValueError(f"Unknown XAI context mode: {mode}")


def _summarize_records(records: list[XAIRecord]) -> dict[str, Any]:
    correct = [record for record in records if record.correct]
    wrong = [record for record in records if not record.correct]
    relation_counts = Counter(record.predicted.split(":")[-1] for record in records)
    return {
        "record_count": len(records),
        "correct_count": len(correct),
        "wrong_count": len(wrong),
        "accuracy": len(correct) / len(records) if records else 0.0,
        "mean_total_score": _mean(record.score for record in records),
        "mean_learned_score": _mean(record.learned_score for record in records),
        "mean_zero_shot_prior": _mean(record.prior_score for record in records),
        "mean_few_shot_score": _mean(record.few_shot_score for record in records),
        "mean_correct_margin": _mean(record.score for record in correct),
        "mean_wrong_margin": _mean(record.score for record in wrong),
        "predicted_relation_counts": dict(relation_counts),
        "top_wrong_examples": [
            {
                "dataset": record.dataset_name,
                "episode": record.episode_id,
                "step_index": record.step_index,
                "expected": record.expected,
                "predicted": record.predicted,
                "score": record.score,
                "learned_score": record.learned_score,
                "zero_shot_prior": record.prior_score,
                "few_shot_score": record.few_shot_score,
                "description": record.description,
            }
            for record in wrong[:20]
        ],
    }


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run XAI analysis for a trained ACG-NAP PredictDesign checkpoint."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=RESULTS_ROOT / "acg_nap" / "gnn_holdout_checkpoint.pt",
    )
    parser.add_argument("--acg-nap-root", type=Path, default=ACG_NAP_ROOT)
    parser.add_argument("--output-path", type=Path, default=RESULTS_ROOT / "acg_nap" / "gnn_xai_report.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--max-files-per-dataset", type=int, default=0)
    args = parser.parse_args()

    device = _resolve_device(args.device, require_cuda=args.require_cuda)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    adapter = ACGNapAdapter(
        context_dim=int(checkpoint["context_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        device=device,
        max_graph_profile_chars=int(checkpoint.get("max_graph_profile_chars", 240)),
        max_node_text_chars=int(checkpoint.get("max_node_text_chars", 480)),
        include_latest_output_in_node_context=bool(checkpoint.get("include_latest_output", False)),
        include_source_output_messages=bool(checkpoint.get("include_source_output_messages", False)),
    )
    corpus = load_acg_nap_corpus(
        args.acg_nap_root,
        adapter,
        max_files_per_dataset=(args.max_files_per_dataset or checkpoint.get("max_files_per_dataset") or None),
    )
    evaluator = BenchmarkEvaluator(
        context_dim=int(checkpoint["context_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        candidate_new_roles=tuple(checkpoint["candidate_new_roles"]),
        candidate_relation_types=tuple(checkpoint["candidate_relation_types"]),
        allow_self_loop_prediction=bool(checkpoint.get("allow_self_loop_prediction", True)),
        device=device,
        train_epochs=0,
        train_fraction=float(checkpoint.get("train_fraction", 0.8)),
        seed=int(checkpoint.get("seed", 7)),
        sentence_transformer_path=str(checkpoint.get("sentence_transformer_path", "__missing_sentence_transformer_model__")),
        sentence_transformer_dim=int(checkpoint.get("sentence_transformer_dim", 384)),
        sentence_transformer_freeze=bool(checkpoint.get("sentence_transformer_freeze", True)),
    )
    spec = CombinationSpec(**checkpoint["spec"])
    system = evaluator.build_system(spec)
    system.load_state_dict(checkpoint["state_dict"])
    system.to(device)
    system.eval()
    combined_split = corpus.combined_split(evaluator.trainer)

    modes = ["full", "no_query", "no_candidate_descriptions", "no_candidates"]
    mode_metrics: dict[str, dict[str, Any]] = {}
    all_records: list[XAIRecord] = []
    with torch.no_grad():
        for mode in modes:
            mode_hits = {1: 0, 3: 0, 5: 0}
            mode_total = 0
            mode_records: list[XAIRecord] = []
            for episode in combined_split.eval_episodes:
                hits, total, records = _candidate_hit(system, episode, context_mode=mode)
                for hit_k, count in hits.items():
                    mode_hits[hit_k] += count
                mode_total += total
                mode_records.extend(records)
            mode_metrics[mode] = {
                "total_steps": mode_total,
                "hit_at_k": {
                    str(hit_k): mode_hits[hit_k] / mode_total if mode_total else 0.0
                    for hit_k in sorted(mode_hits)
                },
                "summary": _summarize_records(mode_records),
            }
            if mode == "full":
                all_records = mode_records

        few_shot_disabled_hits = {1: 0, 3: 0, 5: 0}
        few_shot_disabled_total = 0
        for episode in combined_split.eval_episodes:
            hits, total, _ = _candidate_hit(
                system,
                episode,
                context_mode="full",
                disable_few_shot=True,
            )
            for hit_k, count in hits.items():
                few_shot_disabled_hits[hit_k] += count
            few_shot_disabled_total += total

    baseline_hit_1 = mode_metrics["full"]["hit_at_k"]["1"]
    ablations = {
        mode: {
            "hit_at_1_delta": mode_metrics[mode]["hit_at_k"]["1"] - baseline_hit_1,
            "hit_at_k": mode_metrics[mode]["hit_at_k"],
        }
        for mode in modes
        if mode != "full"
    }
    ablations["few_shot_weight_zero"] = {
        "hit_at_1_delta": (
            few_shot_disabled_hits[1] / few_shot_disabled_total if few_shot_disabled_total else 0.0
        )
        - baseline_hit_1,
        "hit_at_k": {
            str(hit_k): (
                few_shot_disabled_hits[hit_k] / few_shot_disabled_total
                if few_shot_disabled_total
                else 0.0
            )
            for hit_k in sorted(few_shot_disabled_hits)
        },
    }
    report = {
        "checkpoint_path": str(args.checkpoint_path),
        "device": device,
        "eval_episode_count": len(combined_split.eval_episodes),
        "model_spec": checkpoint["spec"],
        "baseline": mode_metrics["full"],
        "ablations": ablations,
        "interpretation": {
            "most_important_signal": min(
                ablations.items(),
                key=lambda item: item[1]["hit_at_1_delta"],
            )[0]
            if ablations
            else "",
            "score_components": mode_metrics["full"]["summary"],
            "note": (
                "learned_score is total_score minus configured zero-shot and few-shot contributions. "
                "Ablation deltas are relative to the full context run."
            ),
        },
        "records_sample": [
            {
                "dataset": record.dataset_name,
                "episode": record.episode_id,
                "step_index": record.step_index,
                "expected": record.expected,
                "predicted": record.predicted,
                "correct": record.correct,
                "score": record.score,
                "learned_score": record.learned_score,
                "zero_shot_prior": record.prior_score,
                "few_shot_score": record.few_shot_score,
            }
            for record in all_records[:100]
        ],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"xai_report={args.output_path}")
    print(f"device={device}")
    print(f"baseline_hit@1={baseline_hit_1:.4f}")
    for name, item in ablations.items():
        print(
            f"ablation={name} hit@1={item['hit_at_k']['1']:.4f} "
            f"delta={item['hit_at_1_delta']:.4f}"
        )


if __name__ == "__main__":
    main()
