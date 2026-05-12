from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign import (
    ACGNapAdapter,
    ExperimentConfig,
    PredictDesignSystem,
    PredictedGraphAction,
    load_acg_nap_candidate_corpus,
)
from predictdesign.paths import ACG_NAP_ROOT, RESULTS_ROOT
from predictdesign.benchmark.trainer import BenchmarkTrainer, bootstrap_few_shot_transition_memory


def _multi_target_log_loss(logits: torch.Tensor, target_indices: list[int]) -> torch.Tensor:
    if not target_indices:
        return logits.new_tensor(0.0)
    log_probs = F.log_softmax(logits, dim=0)
    index_tensor = torch.tensor(target_indices, dtype=torch.long, device=logits.device)
    return -torch.logsumexp(log_probs.index_select(0, index_tensor), dim=0)


def _synchronize_if_cuda(system: PredictDesignSystem) -> None:
    if system.device.type == "cuda":
        torch.cuda.synchronize(system.device)


def _seed_everything(seed: int, *, device: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if str(device).lower().startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(records: list[dict[str, object]], wall_time_s: float) -> dict[str, float | int]:
    evaluated_records = [record for record in records if bool(record.get("evaluated"))]
    feature_cache_total_ms = sum(float(record.get("feature_cache_time_ms", 0.0)) for record in records)

    def summarize(prefix: str, values: list[float]) -> dict[str, float]:
        if not values:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_p50": 0.0,
                f"{prefix}_p95": 0.0,
                f"{prefix}_max": 0.0,
            }
        return {
            f"{prefix}_mean": sum(values) / len(values),
            f"{prefix}_p50": _percentile(values, 0.50),
            f"{prefix}_p95": _percentile(values, 0.95),
            f"{prefix}_max": max(values),
        }

    summary: dict[str, float | int] = {
        "wall_time_s": wall_time_s,
        "raw_step_count": len(records),
        "evaluated_step_count": len(evaluated_records),
        "feature_cache_total_ms": feature_cache_total_ms,
        "feature_cache_amortized_ms_mean": (
            feature_cache_total_ms / len(evaluated_records) if evaluated_records else 0.0
        ),
    }
    for field_name in (
        "prediction_time_ms",
        "prediction_score_time_ms",
        "prediction_rank_time_ms",
        "observed_update_time_ms",
        "context_update_time_ms",
        "message_ingest_time_ms",
        "memory_update_time_ms",
        "action_apply_time_ms",
        "online_step_overhead_ms",
        "total_step_overhead_ms",
        "candidate_count",
        "few_shot_memory_size_before_prediction",
        "few_shot_memory_size_after_update",
    ):
        values = [float(record.get(field_name, 0.0)) for record in evaluated_records]
        summary.update(summarize(field_name, values))
    return summary


def _few_shot_memory_size(system: PredictDesignSystem) -> int:
    if hasattr(system.predictor, "few_shot_memory_size"):
        return int(system.predictor.few_shot_memory_size())
    return 0


def _reset_few_shot_memory(system: PredictDesignSystem) -> None:
    if hasattr(system.predictor, "reset_few_shot_memory"):
        system.predictor.reset_few_shot_memory()


def _audit_episodes(
    episodes,
    *,
    dataset_name: str,
    allow_oracle_candidate_fallback: bool,
    candidate_source: str,
) -> dict[str, object]:
    step_count = 0
    evaluated_step_count = 0
    candidate_missing_count = 0
    expected_missing_from_candidates_count = 0
    source_output_leak_count = 0
    runtime_text_leak_count = 0
    suspicious_predict_text_count = 0
    context_candidate_count = 0
    candidate_description_count = 0
    candidate_transition_id_count = 0
    metadata_keys: set[str] = set()
    for episode in episodes:
        for step in episode.steps:
            step_count += 1
            context = step.prediction_context
            if context is not None:
                if str(context.source_output_text or "").strip():
                    source_output_leak_count += 1
                if str(context.runtime_text or "").strip():
                    runtime_text_leak_count += 1
                combined = context.combined_text().lower()
                if "predict" in combined:
                    suspicious_predict_text_count += 1
                context_candidate_count += len(context.candidate_actions)
                for action in context.candidate_actions:
                    metadata_keys.update(str(key) for key in action.metadata)
                    if str(action.metadata.get("description", "")).strip():
                        candidate_description_count += 1
                    if str(action.metadata.get("transition_id", "")).strip():
                        candidate_transition_id_count += 1
            candidates = list(step.candidate_actions or [])
            if not candidates:
                candidate_missing_count += 1
                if not allow_oracle_candidate_fallback:
                    continue
            expected_actions = step.observed_actions or [step.ground_truth_action]
            if not candidates:
                continue
            evaluated_step_count += 1
            if not any(
                candidate.action_type == expected.action_type
                and candidate.source_node_id == expected.source_node_id
                and candidate.target_node_id == expected.target_node_id
                and candidate.relation_type == expected.relation_type
                for candidate in candidates
                for expected in expected_actions
            ):
                expected_missing_from_candidates_count += 1
    return {
        "dataset_name": dataset_name,
        "candidate_source": candidate_source,
        "step_count": step_count,
        "evaluated_candidate_step_count": evaluated_step_count,
        "candidate_missing_count": candidate_missing_count,
        "expected_missing_from_candidates_count": expected_missing_from_candidates_count,
        "oracle_candidate_fallback_enabled": allow_oracle_candidate_fallback,
        "source_output_leak_count": source_output_leak_count,
        "runtime_text_leak_count": runtime_text_leak_count,
        "suspicious_predict_text_count": suspicious_predict_text_count,
        "prediction_context_candidate_count": context_candidate_count,
        "candidate_description_count": candidate_description_count,
        "candidate_transition_id_count": candidate_transition_id_count,
        "prediction_context_metadata_keys": sorted(metadata_keys),
        "notes": [
            "label/observed_actions are used only for loss and hit@k after scoring.",
            "candidate_actions are treated as the static action space; candidate descriptions and transition ids are model inputs unless role_prompt_query_only is enabled.",
            "candidate_source=graph_transitions_by_source does not read prediction.transition_candidates.",
        ],
    }


@dataclass(slots=True)
class CandidateEvalResult:
    dataset_name: str
    message_reduce: str
    state_updater: str
    gnn_type: str
    total_steps: int
    hit_ks: tuple[int, ...]
    hit_counts: dict[str, int]
    hit_at_k: dict[str, float]
    train_episode_count: int
    eval_episode_count: int
    train_wall_time_s: float = 0.0
    timing_summary: dict[str, float | int] = field(default_factory=dict)


class ACGNapCandidateTrainer:
    def __init__(
        self,
        epochs: int = 5,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-4,
        seed: int = 7,
        hit_k_values: tuple[int, ...] = (1, 3, 5),
        allow_oracle_candidate_fallback: bool = False,
        progress_interval: int = 1,
        cache_static_node_features: bool = False,
        bootstrap_few_shot_memory: bool = True,
        eval_memory_mode: str = "trained",
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.hit_k_values = hit_k_values
        self.allow_oracle_candidate_fallback = allow_oracle_candidate_fallback
        self.progress_interval = max(0, int(progress_interval))
        self.cache_static_node_features = cache_static_node_features
        if eval_memory_mode not in {"trained", "empty"}:
            raise ValueError("eval_memory_mode must be 'trained' or 'empty'.")
        self.bootstrap_few_shot_memory = bootstrap_few_shot_memory
        self.eval_memory_mode = eval_memory_mode

    def fit(self, system: PredictDesignSystem, episodes) -> None:
        if self.bootstrap_few_shot_memory and episodes:
            bootstrap_few_shot_transition_memory(system, episodes)
        elif not self.bootstrap_few_shot_memory:
            _reset_few_shot_memory(system)
        if not episodes or self.epochs <= 0:
            return
        torch.manual_seed(self.seed)
        optimizer = torch.optim.AdamW(
            system.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        system.train()
        fit_started = time.perf_counter()
        for epoch_idx in range(self.epochs):
            epoch_started = time.perf_counter()
            shuffled = list(episodes)
            random.Random(self.seed + epoch_idx).shuffle(shuffled)
            epoch_losses: list[float] = []
            for episode in shuffled:
                loss_value = self._fit_episode(system, episode, optimizer)
                epoch_losses.append(loss_value)
            _synchronize_if_cuda(system)
            if self.progress_interval and (
                epoch_idx == 0
                or epoch_idx + 1 == self.epochs
                or (epoch_idx + 1) % self.progress_interval == 0
            ):
                epoch_time_s = time.perf_counter() - epoch_started
                elapsed_s = time.perf_counter() - fit_started
                mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
                print(
                    "train_progress "
                    f"epoch={epoch_idx + 1}/{self.epochs} "
                    f"episodes={len(shuffled)} "
                    f"mean_loss={mean_loss:.6f} "
                    f"epoch_time_s={epoch_time_s:.3f} "
                    f"elapsed_s={elapsed_s:.3f} "
                    f"device={system.device}",
                    flush=True,
                )

    def _fit_episode(self, system: PredictDesignSystem, episode, optimizer) -> float:
        system.initialize_graph(
            nodes=episode.initial_nodes,
            edges=episode.initial_edges,
            structural_edges=episode.initial_structural_edges,
            graph_context_text=episode.initial_graph_context_text,
            structural_edge_metadata=episode.initial_structural_edge_metadata,
        )
        optimizer.zero_grad(set_to_none=True)
        node_feature_cache, _ = self._build_node_feature_cache(system)
        episode_loss = next(system.parameters()).new_tensor(0.0)
        contributing_steps = 0
        for step in episode.steps:
            score_vector, target_indices, candidate_actions = self._candidate_score_vector(
                system,
                step,
                node_feature_cache=node_feature_cache,
            )
            self._apply_observed_after_prediction(system, step)
            if score_vector is None or not target_indices:
                self._detach_ctdg_state(system)
                continue
            episode_loss = episode_loss + _multi_target_log_loss(score_vector, target_indices)
            contributing_steps += 1
            self._detach_ctdg_state(system)
        if contributing_steps <= 0:
            return 0.0
        normalized_loss = episode_loss / float(contributing_steps)
        normalized_loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
        optimizer.step()
        return float(normalized_loss.detach().item())

    @torch.no_grad()
    def evaluate(
        self,
        system: PredictDesignSystem,
        episodes,
        dataset_name: str,
        timing_records: list[dict[str, object]] | None = None,
    ) -> CandidateEvalResult:
        previous_mode = system.training
        memory_snapshot = (
            system.predictor.snapshot_few_shot_memory()
            if hasattr(system.predictor, "snapshot_few_shot_memory")
            else None
        )
        if self.eval_memory_mode == "empty":
            _reset_few_shot_memory(system)
        system.eval()
        try:
            eval_started = time.perf_counter()
            total_steps = 0
            hit_counts = {str(hit_k): 0 for hit_k in self.hit_k_values}
            local_timing_records: list[dict[str, object]] = []
            for episode in episodes:
                system.initialize_graph(
                    nodes=episode.initial_nodes,
                    edges=episode.initial_edges,
                    structural_edges=episode.initial_structural_edges,
                    graph_context_text=episode.initial_graph_context_text,
                    structural_edge_metadata=episode.initial_structural_edge_metadata,
                )
                node_feature_cache, feature_cache_time_ms = self._build_node_feature_cache(system)
                for step_index, step in enumerate(episode.steps):
                    memory_size_before_prediction = _few_shot_memory_size(system)
                    _synchronize_if_cuda(system)
                    prediction_score_started = time.perf_counter()
                    score_vector, target_indices, candidate_actions = self._candidate_score_vector(
                        system,
                        step,
                        node_feature_cache=node_feature_cache,
                    )
                    _synchronize_if_cuda(system)
                    prediction_score_time_ms = (time.perf_counter() - prediction_score_started) * 1000.0

                    prediction_rank_started = time.perf_counter()
                    candidate_count = len(candidate_actions)
                    hit_flags = {str(hit_k): False for hit_k in self.hit_k_values}
                    evaluated = bool(score_vector is not None and target_indices and candidate_actions)
                    if evaluated:
                        ranked_indices = torch.argsort(score_vector, descending=True).tolist()
                        for hit_k in self.hit_k_values:
                            top_k = ranked_indices[:hit_k]
                            if any(index in target_indices for index in top_k):
                                hit_counts[str(hit_k)] += 1
                                hit_flags[str(hit_k)] = True
                        total_steps += 1
                    _synchronize_if_cuda(system)
                    prediction_rank_time_ms = (time.perf_counter() - prediction_rank_started) * 1000.0
                    prediction_time_ms = prediction_score_time_ms + prediction_rank_time_ms

                    update_breakdown = self._apply_observed_after_prediction_timed(system, step)
                    memory_size_after_update = _few_shot_memory_size(system)
                    update_time_ms = update_breakdown["observed_update_time_ms"]
                    online_step_overhead_ms = prediction_time_ms + update_time_ms
                    total_step_overhead_ms = online_step_overhead_ms + (
                        feature_cache_time_ms if step_index == 0 else 0.0
                    )

                    timing_record = {
                        "dataset_name": dataset_name,
                        "episode_id": str(episode.episode_id),
                        "step_index": step_index,
                        "observation_time": float(step.observation_time),
                        "evaluated": evaluated,
                        "candidate_count": candidate_count,
                        "target_count": len(target_indices),
                        "prediction_time_ms": prediction_time_ms,
                        "prediction_score_time_ms": prediction_score_time_ms,
                        "prediction_rank_time_ms": prediction_rank_time_ms,
                        "observed_update_time_ms": update_time_ms,
                        **update_breakdown,
                        "feature_cache_time_ms": feature_cache_time_ms if step_index == 0 else 0.0,
                        "online_step_overhead_ms": online_step_overhead_ms,
                        "total_step_overhead_ms": total_step_overhead_ms,
                        "message_count": len(step.messages),
                        "observed_action_count": len(step.observed_actions),
                        "context_update_count": len(step.context_updates),
                        "few_shot_memory_size_before_prediction": memory_size_before_prediction,
                        "few_shot_memory_size_after_update": memory_size_after_update,
                        "hit_at_k": hit_flags,
                    }
                    local_timing_records.append(timing_record)
                    if score_vector is None or not target_indices or not candidate_actions:
                        continue
            eval_wall_time_s = time.perf_counter() - eval_started
            if timing_records is not None:
                timing_records.extend(local_timing_records)
            hit_at_k = {
                key: (value / total_steps) if total_steps else 0.0
                for key, value in hit_counts.items()
            }
            return CandidateEvalResult(
                dataset_name=dataset_name,
                message_reduce=system.config.concurrent_update_mode,
                state_updater=system.config.state_updater_type,
                gnn_type=system.config.gnn_type,
                total_steps=total_steps,
                hit_ks=self.hit_k_values,
                hit_counts=hit_counts,
                hit_at_k=hit_at_k,
                train_episode_count=0,
                eval_episode_count=len(episodes),
                timing_summary=_timing_summary(local_timing_records, eval_wall_time_s),
            )
        finally:
            if memory_snapshot is not None and hasattr(system.predictor, "restore_few_shot_memory"):
                system.predictor.restore_few_shot_memory(memory_snapshot)
            if previous_mode:
                system.train()

    def _candidate_score_vector(
        self,
        system: PredictDesignSystem,
        step,
        node_feature_cache: tuple[list[str], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, list[int], list[PredictedGraphAction]]:
        candidate_actions = list(step.candidate_actions or [])
        if not candidate_actions and self.allow_oracle_candidate_fallback:
            candidate_actions = list(step.observed_actions or [step.ground_truth_action])
        if not candidate_actions:
            return None, [], []
        bundle = system.predictor.score_action_space(
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
            observation_time=step.observation_time,
            prediction_context=step.prediction_context,
            node_feature_cache=node_feature_cache,
        )
        if not bundle.node_order:
            return None, [], []
        if bundle.candidate_actions and bundle.candidate_scores is not None:
            filtered_actions = list(bundle.candidate_actions)
            target_indices = [
                idx
                for idx, action in enumerate(filtered_actions)
                if self._action_matches_any(action, step.observed_actions or [step.ground_truth_action])
            ]
            return bundle.candidate_scores, target_indices, filtered_actions
        relation_index = {
            relation_type: index
            for index, relation_type in enumerate(system.config.candidate_relation_types)
        }
        scores: list[torch.Tensor] = []
        filtered_actions: list[PredictedGraphAction] = []
        target_indices: list[int] = []
        for action in candidate_actions:
            if (
                action.source_node_id is None
                or action.target_node_id is None
                or action.source_node_id not in bundle.node_order
                or action.target_node_id not in bundle.node_order
                or action.relation_type not in relation_index
            ):
                continue
            row = bundle.node_order.index(action.source_node_id)
            col = bundle.node_order.index(action.target_node_id)
            if (
                not system.config.allow_self_loop_prediction
                and action.source_node_id == action.target_node_id
            ):
                continue
            relation_idx = relation_index[action.relation_type]
            score = bundle.create_scores[row, col] + bundle.relation_logits[row, col, relation_idx]
            filtered_actions.append(action)
            scores.append(score)
        if not scores:
            return None, [], []
        score_vector = torch.stack(scores)
        for idx, action in enumerate(filtered_actions):
            if self._action_matches_any(action, step.observed_actions or [step.ground_truth_action]):
                target_indices.append(idx)
        return score_vector, target_indices, filtered_actions

    def _build_node_feature_cache(
        self,
        system: PredictDesignSystem,
    ) -> tuple[tuple[list[str], torch.Tensor] | None, float]:
        if (
            not self.cache_static_node_features
            or not hasattr(system.predictor, "build_node_feature_cache")
        ):
            return None, 0.0
        _synchronize_if_cuda(system)
        started = time.perf_counter()
        cache = system.predictor.build_node_feature_cache(
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
        )
        _synchronize_if_cuda(system)
        return cache, (time.perf_counter() - started) * 1000.0

    def _apply_context_updates(self, system: PredictDesignSystem, step) -> None:
        for node_id, context in step.context_updates.items():
            system.update_node_context(
                node_id,
                context,
                text=step.context_text_updates.get(node_id),
            )

    def _apply_observed_after_prediction(self, system: PredictDesignSystem, step) -> None:
        self._apply_context_updates(system, step)
        system.ingest_messages(step.messages)
        system.record_observed_actions(
            step.observed_actions,
            prediction_context=step.prediction_context,
        )
        self._apply_actions(system, step.observed_actions)

    def _apply_observed_after_prediction_timed(
        self,
        system: PredictDesignSystem,
        step,
    ) -> dict[str, float | int]:
        _synchronize_if_cuda(system)
        total_started = time.perf_counter()

        context_started = time.perf_counter()
        self._apply_context_updates(system, step)
        _synchronize_if_cuda(system)
        context_update_time_ms = (time.perf_counter() - context_started) * 1000.0

        message_started = time.perf_counter()
        system.ingest_messages(step.messages)
        _synchronize_if_cuda(system)
        message_ingest_time_ms = (time.perf_counter() - message_started) * 1000.0

        memory_started = time.perf_counter()
        memory_added_count = system.record_observed_actions(
            step.observed_actions,
            prediction_context=step.prediction_context,
        )
        _synchronize_if_cuda(system)
        memory_update_time_ms = (time.perf_counter() - memory_started) * 1000.0

        action_started = time.perf_counter()
        self._apply_actions(system, step.observed_actions)
        _synchronize_if_cuda(system)
        action_apply_time_ms = (time.perf_counter() - action_started) * 1000.0

        observed_update_time_ms = (time.perf_counter() - total_started) * 1000.0
        return {
            "observed_update_time_ms": observed_update_time_ms,
            "context_update_time_ms": context_update_time_ms,
            "message_ingest_time_ms": message_ingest_time_ms,
            "memory_update_time_ms": memory_update_time_ms,
            "action_apply_time_ms": action_apply_time_ms,
            "memory_added_count": memory_added_count,
        }

    def _detach_ctdg_state(self, system: PredictDesignSystem) -> None:
        system.ctdg.current_states = {
            node_id: state.detach()
            for node_id, state in system.ctdg.current_states.items()
        }

    def _apply_actions(self, system: PredictDesignSystem, actions: list[PredictedGraphAction]) -> None:
        for action in actions:
            system.predictor.apply_action(
                action=action,
                temporal_graph=system.temporal_graph,
                ctdg=system.ctdg,
                update_state=False,
            )

    def _action_matches_any(
        self,
        action: PredictedGraphAction,
        expected_actions: list[PredictedGraphAction],
    ) -> bool:
        return any(
            action.action_type == expected.action_type
            and action.source_node_id == expected.source_node_id
            and action.target_node_id == expected.target_node_id
            and action.relation_type == expected.relation_type
            for expected in expected_actions
        )


def build_system(args, candidate_roles, candidate_relations) -> PredictDesignSystem:
    if str(args.device).lower().startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    config = ExperimentConfig(
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        concurrent_update_mode=args.message_reduce_mode,
        state_updater_type=args.state_updater,
        gnn_type=args.gnn_type,
        candidate_new_roles=candidate_roles,
        candidate_relation_types=candidate_relations,
        allow_self_loop_prediction=args.allow_self_loops,
        device=args.device,
        sentence_transformer_path=args.sentence_transformer_path,
        sentence_transformer_dim=args.sentence_transformer_dim,
        sentence_transformer_freeze=args.sentence_transformer_freeze,
        use_runtime_context_features=not args.role_prompt_query_only,
        use_zero_shot_action_priors=not args.gpu_only_learned_scoring,
        use_few_shot_transition_memory=not args.gpu_only_learned_scoring,
        use_online_few_shot_updates=not args.gpu_only_learned_scoring,
    )
    return PredictDesignSystem(config=config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate PredictDesign on acg_nap as a current-step candidate-ranking task."
    )
    parser.add_argument("--acg-nap-root", type=str, default=str(ACG_NAP_ROOT))
    parser.add_argument("--report-path", type=str, default=str(RESULTS_ROOT / "acg_nap" / "candidate_report.json"))
    parser.add_argument("--split-summary-path", type=str, default=str(RESULTS_ROOT / "acg_nap" / "candidate_split_summary.json"))
    parser.add_argument("--cleaning-summary-path", type=str, default=str(RESULTS_ROOT / "acg_nap" / "candidate_cleaning_summary.json"))
    parser.add_argument(
        "--timing-path",
        type=str,
        default="",
        help="Optional JSONL path for per-step evaluation timing records.",
    )
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="hybrid",
        choices=["gcn", "graphsage", "gat", "relational_transformer", "hybrid"],
    )
    parser.add_argument("--state-updater", type=str, default="gru")
    parser.add_argument("--message-reduce-mode", type=str, default="attention")
    parser.add_argument("--sentence-transformer-path", type=str, required=True)
    parser.add_argument("--sentence-transformer-dim", type=int, default=384)
    parser.add_argument("--sentence-transformer-freeze", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-self-loops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-oracle-candidate-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, missing candidate sets are filled with the current label. "
            "This is an oracle upper-bound mode and is disabled by default."
        ),
    )
    parser.add_argument("--max-graph-profile-chars", type=int, default=240)
    parser.add_argument("--max-node-text-chars", type=int, default=480)
    parser.add_argument("--max-files-per-dataset", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument(
        "--gpu-only-learned-scoring",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable CPU-heavy token/set based zero-shot and few-shot priors so "
            "the ranking score is produced by learned CUDA modules plus tensor "
            "edge/relation heads. This is useful for role/prompt/query-only "
            "ablation timing."
        ),
    )
    parser.add_argument(
        "--role-prompt-query-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run an ablation where the model view contains only current query text "
            "and static agent role/profile prompts. It removes graph profile, "
            "runtime context, latest/source outputs, candidate descriptions, "
            "candidate actions in prediction_context, structural workflow metadata, "
            "and automatic runtime feature text."
        ),
    )
    parser.add_argument(
        "--bootstrap-few-shot-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preload online few-shot transition memory from the training split before "
            "gradient training/evaluation. Disable this for a stricter from-zero "
            "online cold-start protocol."
        ),
    )
    parser.add_argument(
        "--eval-memory-mode",
        choices=["trained", "empty"],
        default="trained",
        help=(
            "'trained' evaluates from the post-training few-shot memory snapshot; "
            "'empty' clears few-shot memory before each eval segment, then still "
            "updates it online after each prediction."
        ),
    )
    parser.add_argument(
        "--audit-path",
        type=str,
        default="",
        help="Optional JSON path with leakage/candidate coverage audit statistics.",
    )
    parser.add_argument(
        "--candidate-source",
        choices=[
            "graph_transitions_by_source",
            "graph_transitions_all_edges",
            "prediction_transition_candidates",
        ],
        default="graph_transitions_by_source",
        help=(
            "How to build the candidate action space. The default uses only "
            "graph.transitions filtered by current source. "
            "prediction_transition_candidates is the old strong/oracle-like "
            "benchmark field and should only be used for legacy comparisons."
        ),
    )
    args = parser.parse_args()
    _seed_everything(args.seed, device=args.device)

    adapter = ACGNapAdapter(
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        max_graph_profile_chars=args.max_graph_profile_chars,
        max_node_text_chars=args.max_node_text_chars,
        allow_oracle_candidate_fallback=args.allow_oracle_candidate_fallback,
        role_prompt_query_only=args.role_prompt_query_only,
        candidate_source=args.candidate_source,
    )
    corpus = load_acg_nap_candidate_corpus(
        args.acg_nap_root,
        adapter,
        max_files_per_dataset=(args.max_files_per_dataset or None),
    )
    split_trainer = BenchmarkTrainer(train_fraction=args.train_fraction, seed=args.seed)
    dataset_splits = corpus.dataset_splits(split_trainer)
    combined_split = corpus.combined_split(split_trainer)

    system = build_system(
        args,
        candidate_roles=corpus.role_types or ("planner", "researcher"),
        candidate_relations=corpus.relation_types or ("activate", "delegate", "delegate_return", "retry"),
    )
    trainer = ACGNapCandidateTrainer(
        epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        allow_oracle_candidate_fallback=args.allow_oracle_candidate_fallback,
        progress_interval=args.progress_interval,
        cache_static_node_features=args.role_prompt_query_only,
        bootstrap_few_shot_memory=args.bootstrap_few_shot_memory,
        eval_memory_mode=args.eval_memory_mode,
    )
    train_started = time.perf_counter()
    trainer.fit(system, combined_split.train_episodes)
    _synchronize_if_cuda(system)
    train_wall_time_s = time.perf_counter() - train_started

    report_records: list[CandidateEvalResult] = []
    timing_records: list[dict[str, object]] = []
    all_result = trainer.evaluate(
        system,
        combined_split.eval_episodes,
        dataset_name="acg_nap_all",
        timing_records=timing_records,
    )
    all_result.train_episode_count = len(combined_split.train_episodes)
    all_result.train_wall_time_s = train_wall_time_s
    report_records.append(all_result)
    for dataset_name, split in dataset_splits.items():
        item = trainer.evaluate(
            system,
            split.eval_episodes,
            dataset_name=dataset_name,
            timing_records=timing_records,
        )
        item.train_episode_count = len(combined_split.train_episodes)
        item.train_wall_time_s = train_wall_time_s
        report_records.append(item)

    split_summary = {
        "acg_nap_root": str(corpus.root_path),
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "candidate_relation_types": list(corpus.relation_types),
        "candidate_new_roles": list(corpus.role_types),
        "strict_no_oracle_candidate_fallback": not args.allow_oracle_candidate_fallback,
        "role_prompt_query_only": args.role_prompt_query_only,
        "gpu_only_learned_scoring": args.gpu_only_learned_scoring,
        "candidate_source": args.candidate_source,
        "bootstrap_few_shot_memory": args.bootstrap_few_shot_memory,
        "eval_memory_mode": args.eval_memory_mode,
        "post_training_few_shot_memory_size": _few_shot_memory_size(system),
        "train_wall_time_s": train_wall_time_s,
        "timing_path": args.timing_path,
        "audit_path": args.audit_path,
        "input_view": (
            "query_text + static graph.nodes.*.profile(role/prompt) only"
            if args.role_prompt_query_only
            else "strict query-time candidate view"
        ),
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
    report_path.write_text(
        json.dumps([asdict(item) for item in report_records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    split_summary_path = Path(args.split_summary_path)
    split_summary_path.parent.mkdir(parents=True, exist_ok=True)
    split_summary_path.write_text(json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    cleaning_summary_path = Path(args.cleaning_summary_path)
    cleaning_summary_path.parent.mkdir(parents=True, exist_ok=True)
    cleaning_summary_path.write_text(json.dumps(corpus.cleaning_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.audit_path:
        audit_records = [
            _audit_episodes(
                combined_split.eval_episodes,
                dataset_name="acg_nap_all",
                allow_oracle_candidate_fallback=args.allow_oracle_candidate_fallback,
                candidate_source=args.candidate_source,
            ),
            *[
                _audit_episodes(
                    split.eval_episodes,
                    dataset_name=dataset_name,
                    allow_oracle_candidate_fallback=args.allow_oracle_candidate_fallback,
                    candidate_source=args.candidate_source,
                )
                for dataset_name, split in dataset_splits.items()
            ],
        ]
        audit_path = Path(args.audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.timing_path:
        timing_path = Path(args.timing_path)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in timing_records) + "\n",
            encoding="utf-8",
        )

    print(f"report={report_path}")
    print(f"split_summary={split_summary_path}")
    print(f"cleaning_summary={cleaning_summary_path}")
    if args.audit_path:
        print(f"audit={args.audit_path}")
    if args.timing_path:
        print(f"timing={args.timing_path}")
    print(f"train_wall_time_s={train_wall_time_s:.3f}")
    print(
        f"bootstrap_few_shot_memory={args.bootstrap_few_shot_memory} "
        f"eval_memory_mode={args.eval_memory_mode} "
        f"candidate_source={args.candidate_source} "
        f"post_training_few_shot_memory_size={_few_shot_memory_size(system)}"
    )
    print(f"combined_split train={len(combined_split.train_episodes)} eval={len(combined_split.eval_episodes)}")
    for dataset_name, split in dataset_splits.items():
        print(f"{dataset_name} train={len(split.train_episodes)} eval={len(split.eval_episodes)}")
    for item in report_records:
        print(
            f"{item.dataset_name}\t{item.message_reduce}\t{item.state_updater}\t{item.gnn_type}\t"
            f"hit@1={item.hit_at_k.get('1', 0.0):.4f}\t"
            f"hit@3={item.hit_at_k.get('3', 0.0):.4f}\t"
            f"hit@5={item.hit_at_k.get('5', 0.0):.4f}\t"
            f"pred_ms_mean={item.timing_summary.get('prediction_time_ms_mean', 0.0):.3f}\t"
            f"pred_ms_p95={item.timing_summary.get('prediction_time_ms_p95', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()

