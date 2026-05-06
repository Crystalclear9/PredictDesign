from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import random

import torch
import torch.nn.functional as F

from ..config import ExperimentConfig
from ..experiment import PredictDesignSystem
from ..messages import Message, MessageAction
from ..prediction import GraphActionType, PredictedGraphAction
from .types import BenchmarkEpisode, EpisodeStep


def _focal_loss(
    logits: torch.Tensor,
    target_indices: list[int],
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss variant of multi-target cross-entropy.

    Focuses training on hard-to-classify samples by down-weighting
    easy examples: FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    if not target_indices:
        return logits.new_tensor(0.0)
    probs = torch.softmax(logits, dim=0)
    index_tensor = torch.tensor(target_indices, dtype=torch.long, device=logits.device)
    target_probs = probs.index_select(0, index_tensor)
    target_prob = target_probs.sum().clamp(min=1e-8)
    focal_weight = (1.0 - target_prob) ** gamma
    return -focal_weight * torch.log(target_prob)


@dataclass(slots=True)
class BenchmarkSplit:
    train_episodes: list[BenchmarkEpisode]
    eval_episodes: list[BenchmarkEpisode]


@dataclass(slots=True)
class TrainingSummary:
    epochs_completed: int
    best_eval_accuracy: float
    last_epoch_loss: float
    train_episode_count: int
    eval_episode_count: int


class BenchmarkTrainer:
    def __init__(
        self,
        epochs: int = 20,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-4,
        train_fraction: float = 0.8,
        seed: int = 7,
        first_step_loss_weight: float = 3.0,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_fraction = train_fraction
        self.seed = seed
        self.first_step_loss_weight = first_step_loss_weight
        self.last_fit_summary: TrainingSummary | None = None

    def split_episodes(self, episodes: list[BenchmarkEpisode]) -> BenchmarkSplit:
        if not episodes:
            return BenchmarkSplit(train_episodes=[], eval_episodes=[])
        if len(episodes) == 1:
            return BenchmarkSplit(train_episodes=episodes, eval_episodes=episodes)
        shuffled = list(episodes)
        random.Random(self.seed).shuffle(shuffled)
        train_count = max(1, min(len(episodes) - 1, ceil(len(episodes) * self.train_fraction)))
        return BenchmarkSplit(
            train_episodes=shuffled[:train_count],
            eval_episodes=shuffled[train_count:],
        )

    def fit(
        self,
        system: PredictDesignSystem,
        episodes: list[BenchmarkEpisode],
        config: ExperimentConfig | None = None,
        eval_episodes: list[BenchmarkEpisode] | None = None,
    ) -> None:
        if not episodes or self.epochs <= 0:
            return
        if not getattr(system.predictor, "supports_gradient_training", True):
            return
        cfg = config or system.config
        torch.manual_seed(self.seed)
        optimizer = torch.optim.AdamW(
            system.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Warmup scheduler
        warmup_epochs = max(1, int(self.epochs * cfg.warmup_fraction))
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        system.train()
        best_state: dict[str, torch.Tensor] | None = None
        best_eval_accuracy = float("-inf")
        stale_epochs = 0
        epochs_completed = 0
        last_epoch_loss = 0.0
        for epoch_idx in range(self.epochs):
            epoch_losses: list[float] = []
            epoch_rng = random.Random(self.seed + epoch_idx)
            for episode in self._epoch_episodes(episodes, cfg, epoch_idx):
                epoch_losses.append(self._fit_episode(system, episode, optimizer, cfg, epoch_rng))
            scheduler.step()
            epochs_completed = epoch_idx + 1
            last_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            if eval_episodes:
                eval_accuracy = self.evaluate_next_step_accuracy(system, eval_episodes)
                if eval_accuracy > best_eval_accuracy + 1e-6:
                    best_eval_accuracy = eval_accuracy
                    stale_epochs = 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in system.state_dict().items()
                    }
                else:
                    stale_epochs += 1
                    if (
                        epochs_completed >= cfg.min_training_epochs
                        and stale_epochs >= cfg.early_stopping_patience
                    ):
                        break
        if best_state is not None:
            system.load_state_dict(best_state)
        if best_eval_accuracy == float("-inf"):
            best_eval_accuracy = 0.0
        self.last_fit_summary = TrainingSummary(
            epochs_completed=epochs_completed,
            best_eval_accuracy=best_eval_accuracy,
            last_epoch_loss=last_epoch_loss,
            train_episode_count=len(episodes),
            eval_episode_count=len(eval_episodes or []),
        )

    def _fit_episode(
        self,
        system: PredictDesignSystem,
        episode: BenchmarkEpisode,
        optimizer: torch.optim.Optimizer,
        config: ExperimentConfig | None = None,
        rng: random.Random | None = None,
    ) -> float:
        cfg = config or system.config
        system.initialize_graph(
            nodes=episode.initial_nodes,
            edges=episode.initial_edges,
            structural_edges=episode.initial_structural_edges,
        )
        optimizer.zero_grad(set_to_none=True)
        episode_loss = next(system.parameters()).new_tensor(0.0)
        contributing_steps = 0
        for step_index, step in enumerate(episode.steps):
            self._apply_context_updates(system, step)
            step_messages = self._augment_messages(step.messages, cfg, rng)
            system.ingest_messages(step_messages)
            self._apply_actions(system, step.observed_actions)
            rollout_targets = self._future_rollout_targets(
                episode=episode,
                step_index=step_index,
                horizon=system.config.prediction_horizon,
            )
            loss = self._rollout_loss(system, rollout_targets, cfg)
            if float(loss.detach().item()) <= 0:
                self._detach_ctdg_state(system)
                continue
            episode_loss = episode_loss + loss
            contributing_steps += 1
            self._detach_ctdg_state(system)
        if contributing_steps <= 0:
            return 0.0
        normalized_loss = episode_loss / float(contributing_steps)
        normalized_loss.backward()
        if cfg.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=cfg.gradient_clip_norm)
        optimizer.step()
        return float(normalized_loss.detach().item())

    def _rollout_loss(
        self,
        system: PredictDesignSystem,
        rollout_targets: list[tuple[float, list[PredictedGraphAction]]],
        config: ExperimentConfig | None = None,
    ) -> torch.Tensor:
        if not rollout_targets:
            return next(system.parameters()).new_tensor(0.0)
        cfg = config or system.config
        rollout_graph = system.temporal_graph.clone()
        rollout_ctdg = system.ctdg.clone_with_graph(rollout_graph)
        total_loss = next(system.parameters()).new_tensor(0.0)
        total_weight = 0.0
        for step_offset, (observation_time, actions) in enumerate(rollout_targets):
            step_weight = self.first_step_loss_weight if step_offset == 0 else 1.0
            total_loss = total_loss + step_weight * self._single_time_loss(
                system=system,
                temporal_graph=rollout_graph,
                ctdg=rollout_ctdg,
                actions=actions,
                observation_time=observation_time,
                config=cfg,
            )
            total_weight += step_weight
            self._apply_actions(
                system,
                actions,
                temporal_graph=rollout_graph,
                ctdg=rollout_ctdg,
                update_state=True,
            )
        return total_loss / max(total_weight, 1.0)

    def _single_time_loss(
        self,
        system: PredictDesignSystem,
        temporal_graph,
        ctdg,
        actions: list[PredictedGraphAction],
        observation_time: float,
        config: ExperimentConfig | None = None,
    ) -> torch.Tensor:
        cfg = config or system.config
        bundle = system.predictor.score_action_space(
            temporal_graph=temporal_graph,
            ctdg=ctdg,
            observation_time=observation_time,
        )
        action_logits = system.predictor.action_type_logits(bundle)
        action_targets = {
            GraphActionType.CREATE_EDGE: 0,
            GraphActionType.REMOVE_EDGE: 1,
            GraphActionType.ADD_NODE: 2,
            GraphActionType.NO_OP: 3,
        }
        target_indices = sorted({action_targets[action.action_type] for action in actions})
        # Focal loss or standard multi-target loss
        if cfg.use_focal_loss:
            total_loss = _focal_loss(action_logits, target_indices, gamma=cfg.focal_loss_gamma)
        else:
            total_loss = self._multi_target_log_loss(action_logits, target_indices)
        non_noop_count = min(
            sum(1 for action in actions if action.action_type != GraphActionType.NO_OP),
            system.config.max_actions_per_step,
        )
        count_target = torch.tensor(
            [non_noop_count],
            dtype=torch.long,
            device=action_logits.device,
        )
        total_loss = total_loss + F.cross_entropy(bundle.count_logits.unsqueeze(0), count_target)
        create_loss = self._edge_pair_set_loss(
            bundle,
            actions,
            GraphActionType.CREATE_EDGE,
            allow_self_loops=system.config.allow_self_loop_prediction,
        )
        if create_loss is not None:
            total_loss = total_loss + create_loss
        remove_loss = self._edge_pair_set_loss(
            bundle,
            actions,
            GraphActionType.REMOVE_EDGE,
            allow_self_loops=system.config.allow_self_loop_prediction,
        )
        if remove_loss is not None:
            total_loss = total_loss + remove_loss
        relation_loss = self._relation_loss(bundle, system, actions)
        if relation_loss is not None:
            total_loss = total_loss + relation_loss
        role_loss = self._role_set_loss(bundle.role_logits, system, actions)
        if role_loss is not None:
            total_loss = total_loss + role_loss
        return total_loss

    def _future_rollout_targets(
        self,
        episode: BenchmarkEpisode,
        step_index: int,
        horizon: int,
    ) -> list[tuple[float, list[PredictedGraphAction]]]:
        targets: list[tuple[float, list[PredictedGraphAction]]] = []
        current_time = episode.steps[step_index].observation_time
        for offset in range(1, horizon + 1):
            future_index = step_index + offset
            if future_index < len(episode.steps):
                future_step = episode.steps[future_index]
                targets.append((future_step.observation_time, future_step.supervision_actions))
            else:
                targets.append(
                    (
                        current_time + float(offset),
                        [
                            PredictedGraphAction(
                                action_type=GraphActionType.NO_OP,
                                score=1.0,
                                effective_time=current_time + float(offset),
                            )
                        ],
                    )
                )
        return targets

    def _multi_target_log_loss(
        self,
        logits: torch.Tensor,
        target_indices: list[int],
    ) -> torch.Tensor:
        if not target_indices:
            return logits.new_tensor(0.0)
        log_probs = F.log_softmax(logits, dim=0)
        index_tensor = torch.tensor(target_indices, dtype=torch.long, device=logits.device)
        return -torch.logsumexp(log_probs.index_select(0, index_tensor), dim=0)

    def _edge_pair_set_loss(
        self,
        bundle,
        actions: list[PredictedGraphAction],
        action_type: GraphActionType,
        allow_self_loops: bool = False,
    ) -> torch.Tensor | None:
        size = len(bundle.node_order)
        if size == 0:
            return None
        if action_type == GraphActionType.CREATE_EDGE:
            mask = bundle.create_valid_mask.bool()
            score_matrix = bundle.create_scores
        else:
            mask = bundle.remove_valid_mask.bool()
            score_matrix = bundle.remove_scores
        if not allow_self_loops:
            diagonal = torch.eye(size, dtype=torch.bool, device=bundle.adjacency.device)
            mask = mask & ~diagonal
        if not bool(mask.any().item()):
            return None
        valid_targets: list[int] = []
        for action in actions:
            if action.action_type != action_type:
                continue
            if action.source_node_id is None or action.target_node_id is None:
                continue
            if action.source_node_id not in bundle.node_order or action.target_node_id not in bundle.node_order:
                continue
            source_index = bundle.node_order.index(action.source_node_id)
            target_index = bundle.node_order.index(action.target_node_id)
            if not bool(mask[source_index, target_index].item()):
                continue
            valid_targets.append(source_index * size + target_index)
        if not valid_targets:
            return None
        masked_scores = score_matrix.masked_fill(~mask, -1e9).reshape(size * size)
        return self._multi_target_log_loss(masked_scores, sorted(set(valid_targets)))

    def _role_set_loss(
        self,
        role_logits: torch.Tensor,
        system: PredictDesignSystem,
        actions: list[PredictedGraphAction],
    ) -> torch.Tensor | None:
        valid_roles = [
            system.config.candidate_new_roles.index(action.role)
            for action in actions
            if action.action_type == GraphActionType.ADD_NODE
            and action.role in system.config.candidate_new_roles
        ]
        if not valid_roles:
            return None
        return self._multi_target_log_loss(role_logits, sorted(set(valid_roles)))

    def _relation_loss(
        self,
        bundle,
        system: PredictDesignSystem,
        actions: list[PredictedGraphAction],
    ) -> torch.Tensor | None:
        relation_losses: list[torch.Tensor] = []
        relation_types = system.config.candidate_relation_types
        for action in actions:
            if action.action_type not in {GraphActionType.CREATE_EDGE, GraphActionType.REMOVE_EDGE}:
                continue
            if action.source_node_id is None or action.target_node_id is None:
                continue
            if action.relation_type not in relation_types:
                continue
            if action.source_node_id not in bundle.node_order or action.target_node_id not in bundle.node_order:
                continue
            row = bundle.node_order.index(action.source_node_id)
            col = bundle.node_order.index(action.target_node_id)
            logits = bundle.relation_logits[row, col].unsqueeze(0)
            target = torch.tensor(
                [relation_types.index(action.relation_type)],
                dtype=torch.long,
                device=logits.device,
            )
            relation_losses.append(F.cross_entropy(logits, target))
        if not relation_losses:
            return None
        return torch.stack(relation_losses).mean()

    def _apply_context_updates(self, system: PredictDesignSystem, step: EpisodeStep) -> None:
        for node_id, context in step.context_updates.items():
            system.update_node_context(
                node_id,
                context,
                text=step.context_text_updates.get(node_id),
            )

    def _detach_ctdg_state(self, system: PredictDesignSystem) -> None:
        system.ctdg.current_states = {
            node_id: state.detach()
            for node_id, state in system.ctdg.current_states.items()
        }

    def _epoch_episodes(
        self,
        episodes: list[BenchmarkEpisode],
        config: ExperimentConfig,
        epoch_idx: int,
    ) -> list[BenchmarkEpisode]:
        if not config.shuffle_train_episodes or len(episodes) <= 1:
            return list(episodes)
        shuffled = list(episodes)
        random.Random(self.seed + epoch_idx).shuffle(shuffled)
        return shuffled

    def _augment_messages(
        self,
        messages: list[Message],
        config: ExperimentConfig,
        rng: random.Random | None = None,
    ) -> list[Message]:
        if config.training_message_dropout <= 0 or len(messages) <= 1:
            return messages
        rng = rng or random
        kept: list[Message] = []
        for message in messages:
            if message.action == MessageAction.QUERY_ARRIVAL:
                kept.append(message)
                continue
            if rng.random() >= config.training_message_dropout:
                kept.append(message)
        return kept or messages[:1]

    @torch.no_grad()
    def evaluate_next_step_accuracy(
        self,
        system: PredictDesignSystem,
        episodes: list[BenchmarkEpisode],
    ) -> float:
        if not episodes:
            return 0.0
        previous_mode = system.training
        system.eval()
        correct = 0
        total = 0
        for episode in episodes:
            system.initialize_graph(
                nodes=episode.initial_nodes,
                edges=episode.initial_edges,
                structural_edges=episode.initial_structural_edges,
            )
            for step_index, step in enumerate(episode.steps):
                self._apply_context_updates(system, step)
                system.ingest_messages(step.messages)
                self._apply_actions(system, step.observed_actions)
                rollout_targets = self._future_rollout_targets(episode, step_index=step_index, horizon=1)
                if not rollout_targets:
                    continue
                observation_time, expected_actions = rollout_targets[0]
                predicted_actions = system.predictor.predict_action_set(
                    temporal_graph=system.temporal_graph,
                    ctdg=system.ctdg,
                    observation_time=observation_time,
                )
                if self._actions_match_any(predicted_actions[:1], expected_actions):
                    correct += 1
                total += 1
        if previous_mode:
            system.train()
        return (correct / total) if total else 0.0

    def _actions_match_any(
        self,
        predicted_actions: list[PredictedGraphAction],
        expected_actions: list[PredictedGraphAction],
    ) -> bool:
        return any(
            self._actions_match(predicted, expected)
            for predicted in predicted_actions
            for expected in expected_actions
        )

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

    def _apply_actions(
        self,
        system: PredictDesignSystem,
        actions: list[PredictedGraphAction],
        temporal_graph=None,
        ctdg=None,
        update_state: bool = False,
    ) -> None:
        temporal_graph = temporal_graph or system.temporal_graph
        ctdg = ctdg or system.ctdg
        for action in actions:
            system.predictor.apply_action(
                action=action,
                temporal_graph=temporal_graph,
                ctdg=ctdg,
                update_state=update_state,
            )
