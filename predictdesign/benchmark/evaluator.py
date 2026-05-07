from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path

from ..config import ExperimentConfig, LLMApiConfig
from ..experiment import PredictDesignSystem
from ..prediction import GraphActionType, PredictedGraphAction
from .trainer import BenchmarkTrainer
from .types import BenchmarkEpisode


@dataclass(slots=True)
class CombinationResult:
    dataset_name: str
    message_reduce: str
    state_updater: str
    gnn_type: str
    total_steps: int
    correct_steps: int
    accuracy: float
    hit_ks: tuple[int, ...]
    hit_counts: dict[str, int]
    hit_at_k: dict[str, float]
    train_episode_count: int
    eval_episode_count: int
    one_step_correct_steps: int = 0
    one_step_accuracy: float = 0.0
    one_step_hit_counts: dict[str, int] = field(default_factory=dict)
    one_step_hit_at_k: dict[str, float] = field(default_factory=dict)
    rollout_total_actions: int = 0
    rollout_exact_correct_actions: int = 0
    rollout_exact_accuracy: float = 0.0
    rollout_subgraph_correct_actions: int = 0
    rollout_subgraph_accuracy: float = 0.0
    subgraph_precision: float = 0.0
    subgraph_recall: float = 0.0
    subgraph_f1: float = 0.0
    cv_fold_count: int = 1


@dataclass(slots=True)
class CombinationSpec:
    reduce_mode: str
    updater_type: str
    gnn_type: str
    display_reduce_mode: str
    display_updater_type: str


@dataclass(slots=True)
class _EvaluationTally:
    total_steps: int
    hit_counts: dict[int, int]
    one_step_hit_counts: dict[int, int]
    rollout_total_actions: int
    rollout_exact_correct_actions: int
    rollout_subgraph_correct_actions: int
    predicted_total: int
    matched_truth_total: int
    truth_total: int

    @classmethod
    def empty(cls, hit_ks: tuple[int, ...]) -> "_EvaluationTally":
        return cls(
            total_steps=0,
            hit_counts={hit_k: 0 for hit_k in hit_ks},
            one_step_hit_counts={hit_k: 0 for hit_k in hit_ks},
            rollout_total_actions=0,
            rollout_exact_correct_actions=0,
            rollout_subgraph_correct_actions=0,
            predicted_total=0,
            matched_truth_total=0,
            truth_total=0,
        )

    def merge(self, other: "_EvaluationTally") -> None:
        self.total_steps += other.total_steps
        self.rollout_total_actions += other.rollout_total_actions
        self.rollout_exact_correct_actions += other.rollout_exact_correct_actions
        self.rollout_subgraph_correct_actions += other.rollout_subgraph_correct_actions
        self.predicted_total += other.predicted_total
        self.matched_truth_total += other.matched_truth_total
        self.truth_total += other.truth_total
        for hit_k, count in other.hit_counts.items():
            self.hit_counts[hit_k] = self.hit_counts.get(hit_k, 0) + count
        for hit_k, count in other.one_step_hit_counts.items():
            self.one_step_hit_counts[hit_k] = self.one_step_hit_counts.get(hit_k, 0) + count


class BenchmarkEvaluator:
    def __init__(
        self,
        context_dim: int = 16,
        hidden_dim: int = 32,
        candidate_new_roles: tuple[str, ...] = ("planner", "solver", "critic", "tool"),
        candidate_relation_types: tuple[str, ...] = (
            "communication",
            "delegation",
            "banishment_vote",
            "werewolf_vote",
            "werewolf_attack",
            "guard_action",
            "seer_check",
            "witch_save",
            "witch_poison",
        ),
        allow_self_loop_prediction: bool = False,
        device: str = "cpu",
        train_epochs: int = 20,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-4,
        train_fraction: float = 0.8,
        seed: int = 7,
        cv_folds: int = 5,
        first_step_loss_weight: float = 3.0,
        hit_k_values: tuple[int, ...] = (1, 3, 5),
        llm_api_config: LLMApiConfig | None = None,
        llm_completion_fn=None,
        sentence_transformer_path: str = "all-MiniLM-L6-v2",
        sentence_transformer_dim: int = 384,
        sentence_transformer_freeze: bool = True,
    ) -> None:
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.candidate_new_roles = candidate_new_roles
        self.candidate_relation_types = candidate_relation_types
        self.allow_self_loop_prediction = allow_self_loop_prediction
        self.device = device
        self.seed = seed
        self.cv_folds = cv_folds
        normalized_hit_ks = tuple(dict.fromkeys(int(k) for k in hit_k_values))
        if not normalized_hit_ks or any(k <= 0 for k in normalized_hit_ks):
            raise ValueError("hit_k_values must contain positive integers.")
        self.hit_k_values = normalized_hit_ks
        self.primary_hit_k = self.hit_k_values[0]
        self.llm_api_config = llm_api_config or LLMApiConfig()
        self.llm_completion_fn = llm_completion_fn
        self.sentence_transformer_path = sentence_transformer_path
        self.sentence_transformer_dim = sentence_transformer_dim
        self.sentence_transformer_freeze = sentence_transformer_freeze
        self.trainer = BenchmarkTrainer(
            epochs=train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_fraction=train_fraction,
            seed=seed,
            first_step_loss_weight=first_step_loss_weight,
        )

    def combination_specs(
        self,
        message_reduce_modes: tuple[str, ...] = ("attention",),
        state_updaters: tuple[str, ...] = ("gru", "mdp"),
        gnn_types: tuple[str, ...] = ("gcn", "graphsage", "gat"),
    ) -> list[CombinationSpec]:
        specs: list[CombinationSpec] = []
        for gnn_type in gnn_types:
            if gnn_type == "llm_api":
                specs.append(
                    CombinationSpec(
                        reduce_mode="attention",
                        updater_type="gru",
                        gnn_type="llm_api",
                        display_reduce_mode="llm_api",
                        display_updater_type="llm_api",
                    )
                )
                continue
            for reduce_mode, updater_type in itertools.product(message_reduce_modes, state_updaters):
                specs.append(
                    CombinationSpec(
                        reduce_mode=reduce_mode,
                        updater_type=updater_type,
                        gnn_type=gnn_type,
                        display_reduce_mode=reduce_mode,
                        display_updater_type=updater_type,
                    )
                )
        return specs

    def build_system(self, spec: CombinationSpec) -> PredictDesignSystem:
        return self._build_system(spec.reduce_mode, spec.updater_type, spec.gnn_type)

    def fit_system(
        self,
        system: PredictDesignSystem,
        train_episodes: list[BenchmarkEpisode],
        eval_episodes: list[BenchmarkEpisode] | None = None,
        use_eval_for_training: bool = False,
    ) -> None:
        self.trainer.fit(
            system,
            train_episodes,
            eval_episodes=eval_episodes if use_eval_for_training else None,
        )

    def evaluate_system(
        self,
        dataset_name: str,
        system: PredictDesignSystem,
        eval_episodes: list[BenchmarkEpisode],
        spec: CombinationSpec,
        train_episode_count: int,
        eval_episode_count: int,
        cv_fold_count: int = 1,
    ) -> CombinationResult:
        tally = self._evaluate_episodes(system, eval_episodes)
        return self._build_result(
            dataset_name=dataset_name,
            spec=spec,
            tally=tally,
            train_episode_count=train_episode_count,
            eval_episode_count=eval_episode_count,
            cv_fold_count=cv_fold_count,
        )

    def evaluate_dataset(
        self,
        dataset_name: str,
        episodes: list[BenchmarkEpisode],
        message_reduce_modes: tuple[str, ...] = ("attention",),
        state_updaters: tuple[str, ...] = ("gru", "mdp"),
        gnn_types: tuple[str, ...] = ("gcn", "graphsage", "gat"),
        split_strategy: str = "holdout",
        train_episodes: list[BenchmarkEpisode] | None = None,
        eval_episodes: list[BenchmarkEpisode] | None = None,
        use_eval_for_training: bool | None = None,
    ) -> list[CombinationResult]:
        specs = self.combination_specs(
            message_reduce_modes=message_reduce_modes,
            state_updaters=state_updaters,
            gnn_types=gnn_types,
        )
        split_sets, train_count, eval_count, cv_fold_count = self._resolve_split_sets(
            episodes=episodes,
            split_strategy=split_strategy,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
        )
        if use_eval_for_training is None:
            use_eval_for_training = split_strategy == "cross_validation"
        results: list[CombinationResult] = []
        for spec in specs:
            aggregate_tally = _EvaluationTally.empty(self.hit_k_values)
            for split_train_episodes, split_eval_episodes in split_sets:
                system = self.build_system(spec)
                self.fit_system(
                    system,
                    split_train_episodes,
                    eval_episodes=split_eval_episodes,
                    use_eval_for_training=use_eval_for_training,
                )
                aggregate_tally.merge(self._evaluate_episodes(system, split_eval_episodes))
            results.append(
                self._build_result(
                    dataset_name=dataset_name,
                    spec=spec,
                    tally=aggregate_tally,
                    train_episode_count=train_count,
                    eval_episode_count=eval_count,
                    cv_fold_count=cv_fold_count,
                )
            )
        return results

    def save_report(self, output_path: str | Path, results: list[CombinationResult]) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _resolve_split_sets(
        self,
        episodes: list[BenchmarkEpisode],
        split_strategy: str,
        train_episodes: list[BenchmarkEpisode] | None,
        eval_episodes: list[BenchmarkEpisode] | None,
    ) -> tuple[list[tuple[list[BenchmarkEpisode], list[BenchmarkEpisode]]], int, int, int]:
        if train_episodes is not None or eval_episodes is not None:
            if train_episodes is None or eval_episodes is None:
                raise ValueError("train_episodes and eval_episodes must be provided together.")
            return [(train_episodes, eval_episodes)], len(train_episodes), len(eval_episodes), 1
        if split_strategy == "holdout":
            split = self.trainer.split_episodes(episodes)
            return (
                [(split.train_episodes, split.eval_episodes)],
                len(split.train_episodes),
                len(split.eval_episodes),
                1,
            )
        if split_strategy == "cross_validation":
            folds = self._episode_folds(episodes)
            average_train_count = round(sum(len(train) for train, _ in folds) / len(folds)) if folds else 0
            average_eval_count = round(sum(len(eval_) for _, eval_ in folds) / len(folds)) if folds else 0
            return folds, average_train_count, average_eval_count, len(folds)
        raise ValueError("split_strategy must be 'holdout' or 'cross_validation'.")

    def _build_result(
        self,
        dataset_name: str,
        spec: CombinationSpec,
        tally: _EvaluationTally,
        train_episode_count: int,
        eval_episode_count: int,
        cv_fold_count: int,
    ) -> CombinationResult:
        precision = (
            tally.matched_truth_total / tally.predicted_total
            if tally.predicted_total
            else 0.0
        )
        recall = (tally.matched_truth_total / tally.truth_total) if tally.truth_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        hit_rates = {
            str(hit_k): (tally.hit_counts[hit_k] / tally.total_steps) if tally.total_steps else 0.0
            for hit_k in self.hit_k_values
        }
        one_step_hit_rates = {
            str(hit_k): (
                tally.one_step_hit_counts[hit_k] / tally.total_steps
            ) if tally.total_steps else 0.0
            for hit_k in self.hit_k_values
        }
        primary_hit_count = tally.hit_counts[self.primary_hit_k]
        primary_one_step_hit_count = tally.one_step_hit_counts[self.primary_hit_k]
        return CombinationResult(
            dataset_name=dataset_name,
            message_reduce=spec.display_reduce_mode,
            state_updater=spec.display_updater_type,
            gnn_type=spec.gnn_type,
            total_steps=tally.total_steps,
            correct_steps=primary_hit_count,
            accuracy=hit_rates[str(self.primary_hit_k)],
            hit_ks=self.hit_k_values,
            hit_counts={str(hit_k): tally.hit_counts[hit_k] for hit_k in self.hit_k_values},
            hit_at_k=hit_rates,
            train_episode_count=train_episode_count,
            eval_episode_count=eval_episode_count,
            one_step_correct_steps=primary_one_step_hit_count,
            one_step_accuracy=one_step_hit_rates[str(self.primary_hit_k)],
            one_step_hit_counts={
                str(hit_k): tally.one_step_hit_counts[hit_k]
                for hit_k in self.hit_k_values
            },
            one_step_hit_at_k=one_step_hit_rates,
            rollout_total_actions=tally.rollout_total_actions,
            rollout_exact_correct_actions=tally.rollout_exact_correct_actions,
            rollout_exact_accuracy=(
                tally.rollout_exact_correct_actions / tally.rollout_total_actions
            ) if tally.rollout_total_actions else 0.0,
            rollout_subgraph_correct_actions=tally.rollout_subgraph_correct_actions,
            rollout_subgraph_accuracy=(
                tally.rollout_subgraph_correct_actions / tally.rollout_total_actions
            ) if tally.rollout_total_actions else 0.0,
            subgraph_precision=precision,
            subgraph_recall=recall,
            subgraph_f1=f1,
            cv_fold_count=cv_fold_count,
        )

    def _evaluate_episodes(
        self,
        system: PredictDesignSystem,
        episodes: list[BenchmarkEpisode],
    ) -> _EvaluationTally:
        tally = _EvaluationTally.empty(self.hit_k_values)
        system.eval()
        for episode in episodes:
            system.initialize_graph(
                nodes=episode.initial_nodes,
                edges=episode.initial_edges,
                structural_edges=episode.initial_structural_edges,
                graph_context_text=episode.initial_graph_context_text,
                structural_edge_metadata=episode.initial_structural_edge_metadata,
            )
            for step_index, step in enumerate(episode.steps):
                self.trainer._apply_context_updates(system, step)
                system.ingest_messages(step.messages)
                self.trainer._apply_actions(system, step.observed_actions)
                available_future_steps = min(
                    system.config.prediction_horizon,
                    max(len(episode.steps) - step_index - 1, 0),
                )
                if available_future_steps <= 0:
                    continue
                future_targets = self._future_rollout_targets(
                    episode=episode,
                    step_index=step_index,
                    horizon=system.config.prediction_horizon,
                )[:available_future_steps]
                future_times = [time_value for time_value, _, _ in future_targets]
                prediction_context_schedule = [
                    prediction_context for _, _, prediction_context in future_targets
                ]
                rollout = system.predictor.predict_subgraph_rollout(
                    temporal_graph=system.temporal_graph,
                    ctdg=system.ctdg,
                    observation_time=step.observation_time,
                    time_schedule=future_times,
                    prediction_context_schedule=prediction_context_schedule,
                )
                predicted_action_windows = rollout.actions_by_step
                future_action_windows = [actions for _, actions, _ in future_targets]
                future_union = self._flatten_action_windows(future_action_windows)
                first_predicted_window = predicted_action_windows[0] if predicted_action_windows else []
                predicted_union = self._flatten_action_windows(predicted_action_windows)
                for hit_k in self.hit_k_values:
                    if self._window_matches_any(
                        self._top_k_actions(first_predicted_window, hit_k),
                        future_action_windows[0],
                    ):
                        tally.one_step_hit_counts[hit_k] += 1
                    if self._window_matches_any(
                        self._top_k_actions(predicted_union, hit_k),
                        future_union,
                    ):
                        tally.hit_counts[hit_k] += 1
                tally.total_steps += 1
                tally.rollout_total_actions += sum(len(window) for window in predicted_action_windows)
                tally.predicted_total += len(predicted_union)
                tally.truth_total += len(future_union)
                tally.matched_truth_total += self._matched_action_count(predicted_union, future_union)
                for predicted_actions, expected_actions in zip(predicted_action_windows, future_action_windows):
                    tally.rollout_exact_correct_actions += self._matched_action_count(
                        predicted_actions,
                        expected_actions,
                    )
                    tally.rollout_subgraph_correct_actions += self._matched_action_count(
                        predicted_actions,
                        future_union,
                    )
        return tally

    def _build_system(
        self,
        reduce_mode: str,
        updater_type: str,
        gnn_type: str,
    ) -> PredictDesignSystem:
        config = ExperimentConfig(
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            concurrent_update_mode=reduce_mode,
            state_updater_type=updater_type,
            gnn_type=gnn_type,
            predictor_backend="llm_api" if gnn_type == "llm_api" else "gnn",
            candidate_new_roles=self.candidate_new_roles,
            candidate_relation_types=self.candidate_relation_types,
            allow_self_loop_prediction=self.allow_self_loop_prediction,
            device=self.device,
            sentence_transformer_path=self.sentence_transformer_path,
            sentence_transformer_dim=self.sentence_transformer_dim,
            sentence_transformer_freeze=self.sentence_transformer_freeze,
            llm_api=self.llm_api_config,
        )
        return PredictDesignSystem(config=config, llm_completion_fn=self.llm_completion_fn)

    def _future_rollout_targets(
        self,
        episode: BenchmarkEpisode,
        step_index: int,
        horizon: int,
    ):
        return self.trainer._future_rollout_targets(episode, step_index, horizon)

    def _episode_folds(
        self,
        episodes: list[BenchmarkEpisode],
    ) -> list[tuple[list[BenchmarkEpisode], list[BenchmarkEpisode]]]:
        if not episodes:
            return []
        if len(episodes) == 1:
            return [(episodes, episodes)]
        shuffled = list(episodes)
        random.Random(self.seed).shuffle(shuffled)
        fold_count = max(2, min(self.cv_folds, len(shuffled)))
        fold_size = ceil(len(shuffled) / fold_count)
        folds: list[tuple[list[BenchmarkEpisode], list[BenchmarkEpisode]]] = []
        for fold_index in range(fold_count):
            start = fold_index * fold_size
            end = min(len(shuffled), start + fold_size)
            eval_episodes = shuffled[start:end]
            if not eval_episodes:
                continue
            train_episodes = shuffled[:start] + shuffled[end:]
            if not train_episodes:
                train_episodes = eval_episodes
            folds.append((train_episodes, eval_episodes))
        return folds

    def _flatten_action_windows(
        self,
        action_windows: list[list[PredictedGraphAction]],
    ) -> list[PredictedGraphAction]:
        flattened: list[PredictedGraphAction] = []
        seen: set[tuple[str, str | None, str | None, str | None, str | None]] = set()
        for actions in action_windows:
            for action in actions:
                key = (
                    action.action_type.value,
                    action.source_node_id,
                    action.target_node_id,
                    action.relation_type,
                    action.role,
                )
                if key in seen:
                    continue
                seen.add(key)
                flattened.append(action)
        return flattened

    def _top_k_actions(
        self,
        actions: list[PredictedGraphAction],
        k: int,
    ) -> list[PredictedGraphAction]:
        if k <= 0:
            return []
        return actions[:k]

    def _actions_match_any(
        self,
        predicted: PredictedGraphAction,
        expected_actions: list[PredictedGraphAction],
    ) -> bool:
        return any(self._actions_match(predicted, expected) for expected in expected_actions)

    def _window_matches_any(
        self,
        predicted_actions: list[PredictedGraphAction],
        expected_actions: list[PredictedGraphAction],
    ) -> bool:
        return any(self._actions_match_any(action, expected_actions) for action in predicted_actions)

    def _matched_action_count(
        self,
        predicted_actions: list[PredictedGraphAction],
        expected_actions: list[PredictedGraphAction],
    ) -> int:
        remaining = list(expected_actions)
        match_count = 0
        for predicted in predicted_actions:
            for index, expected in enumerate(remaining):
                if self._actions_match(predicted, expected):
                    match_count += 1
                    remaining.pop(index)
                    break
        return match_count

    def _actions_match(
        self,
        predicted: PredictedGraphAction,
        expected: PredictedGraphAction,
    ) -> bool:
        if predicted.action_type != expected.action_type:
            return False
        if predicted.action_type in {GraphActionType.CREATE_EDGE, GraphActionType.REMOVE_EDGE}:
            return (
                predicted.source_node_id == expected.source_node_id
                and predicted.target_node_id == expected.target_node_id
                and predicted.relation_type == expected.relation_type
            )
        if predicted.action_type == GraphActionType.ADD_NODE:
            return predicted.role == expected.role
        return True
