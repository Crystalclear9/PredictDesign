from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

from ..config import ExperimentConfig, LLMApiConfig
from ..experiment import PredictDesignSystem
from ..prediction import GraphActionType, PredictedGraphAction
from .types import BenchmarkEpisode
from .trainer import BenchmarkTrainer


@dataclass(slots=True)
class CombinationResult:
    dataset_name: str
    message_reduce: str
    state_updater: str
    gnn_type: str
    total_steps: int
    correct_steps: int
    accuracy: float
    train_episode_count: int
    eval_episode_count: int
    one_step_correct_steps: int = 0
    one_step_accuracy: float = 0.0
    rollout_total_actions: int = 0
    rollout_exact_correct_actions: int = 0
    rollout_exact_accuracy: float = 0.0
    rollout_subgraph_correct_actions: int = 0
    rollout_subgraph_accuracy: float = 0.0
    subgraph_precision: float = 0.0
    subgraph_recall: float = 0.0
    subgraph_f1: float = 0.0
    cv_fold_count: int = 1


class BenchmarkEvaluator:
    def __init__(
        self,
        context_dim: int = 16,
        hidden_dim: int = 32,
        candidate_new_roles: tuple[str, ...] = ("planner", "solver", "critic", "tool"),
        device: str = "cpu",
        train_epochs: int = 20,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-4,
        train_fraction: float = 0.8,
        seed: int = 7,
        cv_folds: int = 5,
        first_step_loss_weight: float = 3.0,
        llm_api_config: LLMApiConfig | None = None,
        llm_completion_fn=None,
    ) -> None:
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.candidate_new_roles = candidate_new_roles
        self.device = device
        self.seed = seed
        self.cv_folds = cv_folds
        self.llm_api_config = llm_api_config or LLMApiConfig()
        self.llm_completion_fn = llm_completion_fn
        self.trainer = BenchmarkTrainer(
            epochs=train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_fraction=train_fraction,
            seed=seed,
            first_step_loss_weight=first_step_loss_weight,
        )

    def evaluate_dataset(
        self,
        dataset_name: str,
        episodes: list[BenchmarkEpisode],
        message_reduce_modes: tuple[str, ...] = ("sum", "mean"),
        state_updaters: tuple[str, ...] = ("gru", "mdp"),
        gnn_types: tuple[str, ...] = ("gcn", "graphsage", "gat"),
    ) -> list[CombinationResult]:
        results: list[CombinationResult] = []
        folds = self._episode_folds(episodes)
        average_train_count = round(sum(len(train) for train, _ in folds) / len(folds)) if folds else 0
        average_eval_count = round(sum(len(eval_) for _, eval_ in folds) / len(folds)) if folds else 0
        combinations: list[tuple[str, str, str, str, str]] = []
        for gnn_type in gnn_types:
            if gnn_type == "llm_api":
                combinations.append(("sum", "gru", gnn_type, "llm_api", "llm_api"))
                continue
            for reduce_mode, updater_type in itertools.product(message_reduce_modes, state_updaters):
                combinations.append((reduce_mode, updater_type, gnn_type, reduce_mode, updater_type))

        for reduce_mode, updater_type, gnn_type, display_reduce_mode, display_updater_type in combinations:
            correct = 0
            total = 0
            one_step_correct = 0
            rollout_total = 0
            rollout_exact_correct = 0
            rollout_subgraph_correct = 0
            predicted_total = 0
            matched_truth_total = 0
            truth_total = 0
            for train_episodes, eval_episodes in folds:
                system = self._build_system(reduce_mode, updater_type, gnn_type)
                self.trainer.fit(system, train_episodes)
                system.eval()
                for episode in eval_episodes:
                    system.initialize_graph(
                        nodes=episode.initial_nodes,
                        edges=episode.initial_edges,
                        structural_edges=episode.initial_structural_edges,
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
                        future_times = [time_value for time_value, _ in future_targets]
                        rollout = system.predictor.predict_subgraph_rollout(
                            temporal_graph=system.temporal_graph,
                            ctdg=system.ctdg,
                            observation_time=step.observation_time,
                            time_schedule=future_times,
                        )
                        predicted_action_windows = rollout.actions_by_step
                        future_action_windows = [actions for _, actions in future_targets]
                        future_union = self._flatten_action_windows(future_action_windows)
                        first_predicted_window = predicted_action_windows[0] if predicted_action_windows else []
                        if self._window_matches_any(first_predicted_window, future_action_windows[0]):
                            one_step_correct += 1
                        predicted_union = self._flatten_action_windows(predicted_action_windows)
                        if self._window_matches_any(predicted_union, future_union):
                            correct += 1
                        total += 1
                        rollout_total += sum(len(window) for window in predicted_action_windows)
                        predicted_total += len(predicted_union)
                        truth_total += len(future_union)
                        matched_truth_total += self._matched_action_count(predicted_union, future_union)
                        for predicted_actions, expected_actions in zip(predicted_action_windows, future_action_windows):
                            rollout_exact_correct += self._matched_action_count(predicted_actions, expected_actions)
                            rollout_subgraph_correct += self._matched_action_count(predicted_actions, future_union)
            precision = (matched_truth_total / predicted_total) if predicted_total else 0.0
            recall = (matched_truth_total / truth_total) if truth_total else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            results.append(
                CombinationResult(
                    dataset_name=dataset_name,
                    message_reduce=display_reduce_mode,
                    state_updater=display_updater_type,
                    gnn_type=gnn_type,
                    total_steps=total,
                    correct_steps=correct,
                    accuracy=(correct / total) if total else 0.0,
                    train_episode_count=average_train_count,
                    eval_episode_count=average_eval_count,
                    one_step_correct_steps=one_step_correct,
                    one_step_accuracy=(one_step_correct / total) if total else 0.0,
                    rollout_total_actions=rollout_total,
                    rollout_exact_correct_actions=rollout_exact_correct,
                    rollout_exact_accuracy=(rollout_exact_correct / rollout_total) if rollout_total else 0.0,
                    rollout_subgraph_correct_actions=rollout_subgraph_correct,
                    rollout_subgraph_accuracy=(
                        rollout_subgraph_correct / rollout_total
                    ) if rollout_total else 0.0,
                    subgraph_precision=precision,
                    subgraph_recall=recall,
                    subgraph_f1=f1,
                    cv_fold_count=len(folds),
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
            device=self.device,
            llm_api=self.llm_api_config,
        )
        return PredictDesignSystem(config=config, llm_completion_fn=self.llm_completion_fn)

    def _future_rollout_targets(
        self,
        episode: BenchmarkEpisode,
        step_index: int,
        horizon: int,
    ) -> list[tuple[float, list[PredictedGraphAction]]]:
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
        seen: set[tuple[str, str | None, str | None, str | None]] = set()
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
