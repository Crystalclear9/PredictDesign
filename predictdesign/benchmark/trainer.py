from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F

from ..experiment import PredictDesignSystem
from ..prediction import GraphActionType, PredictedGraphAction
from .types import BenchmarkEpisode, EpisodeStep


@dataclass(slots=True)
class BenchmarkSplit:
    train_episodes: list[BenchmarkEpisode]
    eval_episodes: list[BenchmarkEpisode]


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

    def split_episodes(self, episodes: list[BenchmarkEpisode]) -> BenchmarkSplit:
        if not episodes:
            return BenchmarkSplit(train_episodes=[], eval_episodes=[])
        if len(episodes) == 1:
            return BenchmarkSplit(train_episodes=episodes, eval_episodes=episodes)
        train_count = max(1, min(len(episodes) - 1, ceil(len(episodes) * self.train_fraction)))
        return BenchmarkSplit(
            train_episodes=episodes[:train_count],
            eval_episodes=episodes[train_count:],
        )

    def fit(self, system: PredictDesignSystem, episodes: list[BenchmarkEpisode]) -> None:
        if not episodes or self.epochs <= 0:
            return
        if not getattr(system.predictor, "supports_gradient_training", True):
            return
        torch.manual_seed(self.seed)
        optimizer = torch.optim.AdamW(
            system.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        system.train()
        for _ in range(self.epochs):
            for episode in episodes:
                self._fit_episode(system, episode, optimizer)

    def _fit_episode(
        self,
        system: PredictDesignSystem,
        episode: BenchmarkEpisode,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        system.initialize_graph(
            nodes=episode.initial_nodes,
            edges=episode.initial_edges,
            structural_edges=episode.initial_structural_edges,
        )
        for step_index, step in enumerate(episode.steps):
            self._apply_context_updates(system, step)
            system.ingest_messages(step.messages)
            self._apply_actions(system, step.observed_actions)
            rollout_targets = self._future_rollout_targets(
                episode=episode,
                step_index=step_index,
                horizon=system.config.prediction_horizon,
            )
            loss = self._rollout_loss(system, rollout_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self._detach_ctdg_state(system)

    def _rollout_loss(
        self,
        system: PredictDesignSystem,
        rollout_targets: list[tuple[float, list[PredictedGraphAction]]],
    ) -> torch.Tensor:
        if not rollout_targets:
            return next(system.parameters()).new_tensor(0.0)
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
    ) -> torch.Tensor:
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
        create_loss = self._edge_pair_set_loss(bundle, actions, GraphActionType.CREATE_EDGE)
        if create_loss is not None:
            total_loss = total_loss + create_loss
        remove_loss = self._edge_pair_set_loss(bundle, actions, GraphActionType.REMOVE_EDGE)
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
