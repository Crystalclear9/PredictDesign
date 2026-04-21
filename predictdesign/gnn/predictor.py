from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..completion import NodeCompletionClassifier
from ..config import ExperimentConfig
from ..ctdg import ContinuousTimeDynamicGraph
from ..encoders import SentenceTransformerEncoder, stable_hash_index
from ..messages import Message
from ..prediction import (
    GraphActionType,
    PredictedGraphAction,
    PredictionRollout,
    PredictionSubgraphRollout,
)
from ..temporal_graph import TemporalEdge, TemporalGraph, TemporalNode
from ..types import ensure_tensor
from .cold_start import ColdStartInitializer
from .layers import GNNBackbone


@dataclass(slots=True)
class ActionScoreBundle:
    node_order: list[str]
    adjacency: torch.Tensor
    create_scores: torch.Tensor
    remove_scores: torch.Tensor
    create_valid_mask: torch.Tensor
    remove_valid_mask: torch.Tensor
    relation_logits: torch.Tensor
    role_logits: torch.Tensor
    count_logits: torch.Tensor
    no_op_logit: torch.Tensor
    graph_embedding: torch.Tensor
    completion_scores: torch.Tensor | None = None


class GraphActionPredictor(nn.Module):
    def __init__(self, config: ExperimentConfig, node_feature_encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.node_feature_encoder = node_feature_encoder
        self.gnn_backbone = GNNBackbone(
            layer_type=config.gnn_type,
            hidden_dim=config.hidden_dim,
            num_layers=config.gnn_layers,
            edge_feature_dim=config.temporal_edge_dim,
            num_heads=config.rt_num_heads,
            dropout=config.rt_dropout,
        )
        self.create_source = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.create_target = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.remove_source = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.remove_target = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.create_edge_bias = nn.Sequential(
            nn.Linear(config.temporal_edge_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.remove_edge_bias = nn.Sequential(
            nn.Linear(config.temporal_edge_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.relation_pair_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 4 + config.temporal_edge_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, len(config.candidate_relation_types)),
        )
        self.relation_context_embedding = nn.Embedding(
            len(config.candidate_relation_types),
            config.context_dim,
        )
        self.graph_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.add_node_head = nn.Linear(config.hidden_dim, len(config.candidate_new_roles))
        self.action_count_head = nn.Linear(config.hidden_dim, config.max_actions_per_step + 1)
        self.no_op_head = nn.Linear(config.hidden_dim, 1)
        self.empty_graph_embedding = nn.Parameter(torch.zeros(config.hidden_dim))

        # Attention pooling for graph-level embedding
        self.attn_pool_query = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        self.attn_pool_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Cold start initializer
        self._text_encoder_for_cold_start: SentenceTransformerEncoder | None = None
        if config.use_cold_start:
            self._text_encoder_for_cold_start = SentenceTransformerEncoder(
                output_dim=config.sentence_transformer_dim,
                model_name_or_path=config.sentence_transformer_path,
                st_dim=config.sentence_transformer_dim,
                freeze=config.sentence_transformer_freeze,
            )
            self.cold_start = ColdStartInitializer(
                candidate_roles=config.candidate_new_roles,
                hidden_dim=config.hidden_dim,
                text_encoder=self._text_encoder_for_cold_start,
                st_dim=config.sentence_transformer_dim,
            )
        else:
            self.cold_start = None

        # Completion classifier
        if config.use_completion_detection:
            self.completion_classifier = NodeCompletionClassifier(config.hidden_dim)
        else:
            self.completion_classifier = None

    def _build_role_indices(
        self,
        node_order: list[str],
        temporal_graph: TemporalGraph,
    ) -> torch.Tensor:
        """Build integer role index tensor for RelationalAttentionLayer."""
        role_to_idx: dict[str, int] = {}
        indices: list[int] = []
        for node_id in node_order:
            role = temporal_graph.nodes[node_id].role
            if role not in role_to_idx:
                role_to_idx[role] = len(role_to_idx)
            indices.append(role_to_idx[role])
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def encode_graph(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        node_order = sorted(temporal_graph.nodes)
        if not node_order:
            empty_matrix = torch.zeros((0, 0), dtype=torch.float32, device=self.device)
            empty_edge_features = torch.zeros(
                (0, 0, self.config.temporal_edge_dim),
                dtype=torch.float32,
                device=self.device,
            )
            # Cold-start: use initializer if available
            if self.cold_start is not None:
                cold_embedding = self.cold_start.graph_embedding_cold(device=self.device)
                empty_embeddings = cold_embedding.unsqueeze(0)  # [1, D]
            else:
                empty_embeddings = torch.zeros(
                    (0, self.config.hidden_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
            return node_order, empty_matrix, empty_edge_features, empty_embeddings
        features = torch.stack(
            [
                self.node_feature_encoder(
                    temporal_graph.nodes[node_id],
                    ctdg.get_state(node_id).to(self.device),
                )
                for node_id in node_order
            ],
            dim=0,
        )
        active_adjacency = temporal_graph.adjacency_matrix(
            time_value=observation_time,
            node_order=node_order,
            device=self.device,
            include_structural=False,
        )
        message_passing_adjacency = temporal_graph.adjacency_matrix(
            time_value=observation_time,
            node_order=node_order,
            device=self.device,
            include_structural=True,
        )
        edge_features = temporal_graph.temporal_edge_features(
            time_value=observation_time,
            node_order=node_order,
            device=self.device,
            feature_dim=self.config.temporal_edge_dim,
        )
        edge_features = self._inject_semantic_edge_features(
            edge_features=edge_features,
            node_order=node_order,
            temporal_graph=temporal_graph,
            ctdg=ctdg,
        )
        # Build role indices for RT
        role_indices = self._build_role_indices(node_order, temporal_graph)
        encoded = self.gnn_backbone(
            features, message_passing_adjacency, edge_features, role_indices=role_indices,
        )
        return node_order, active_adjacency, edge_features, encoded

    def graph_embedding_from_encoded(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if node_embeddings.numel() == 0:
            if self.cold_start is not None:
                return self.graph_projection(
                    self.cold_start.graph_embedding_cold(device=self.device)
                )
            return self.graph_projection(self.empty_graph_embedding)
        # Attention pooling (learned query attends to node embeddings)
        keys = self.attn_pool_key(node_embeddings)  # [N, D]
        attn_scores = (keys @ self.attn_pool_query) / math.sqrt(self.config.hidden_dim)  # [N]
        attn_weights = torch.softmax(attn_scores, dim=0)  # [N]
        pooled = (attn_weights.unsqueeze(-1) * node_embeddings).sum(dim=0)  # [D]
        return self.graph_projection(pooled)

    def predict_next_action(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
    ) -> PredictedGraphAction:
        action_set = self.predict_action_set(
            temporal_graph=temporal_graph,
            ctdg=ctdg,
            observation_time=observation_time,
        )
        return action_set[0]

    def predict_action_set(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
    ) -> list[PredictedGraphAction]:
        score_bundle = self.score_action_space(
            temporal_graph=temporal_graph,
            ctdg=ctdg,
            observation_time=observation_time,
        )
        action_logits = self.action_type_logits(score_bundle)
        predicted_count = int(score_bundle.count_logits.argmax().item())
        no_op_action = PredictedGraphAction(
            action_type=GraphActionType.NO_OP,
            score=float(action_logits[3].item()),
            effective_time=observation_time,
        )
        candidates = self._candidate_actions(score_bundle, observation_time, action_logits)
        if predicted_count <= 0:
            best_non_noop_logit = float(action_logits[:3].max().item())
            if candidates and best_non_noop_logit > float(action_logits[3].item()):
                predicted_count = 1
            else:
                return [no_op_action]
        if not candidates:
            return [no_op_action]
        selected: list[PredictedGraphAction] = []
        seen: set[tuple[str, str | None, str | None, str | None]] = set()
        for action in candidates:
            key = self._action_key(action)
            if key in seen:
                continue
            seen.add(key)
            selected.append(action)
            if len(selected) >= predicted_count:
                break
        return selected or [no_op_action]

    def score_action_space(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
    ) -> ActionScoreBundle:
        node_order, adjacency, edge_features, node_embeddings = self.encode_graph(
            temporal_graph=temporal_graph,
            ctdg=ctdg,
            observation_time=observation_time,
        )
        graph_embedding = self.graph_embedding_from_encoded(node_embeddings)

        # Number of actual nodes (cold start may add virtual embeddings)
        n_actual = len(node_order)

        # Use only actual node embeddings for edge scoring
        actual_embeddings = node_embeddings[:n_actual] if node_embeddings.size(0) > n_actual else node_embeddings

        # Handle empty graph edge scoring
        if n_actual == 0 or adjacency.numel() == 0:
            create_scores = torch.zeros((0, 0), dtype=torch.float32, device=self.device)
            remove_scores = torch.zeros((0, 0), dtype=torch.float32, device=self.device)
        else:
            create_scores = (
                self.create_source(actual_embeddings) @ self.create_target(actual_embeddings).T
                + self.create_edge_bias(edge_features).squeeze(-1)
            )
            remove_scores = (
                self.remove_source(actual_embeddings) @ self.remove_target(actual_embeddings).T
                + self.remove_edge_bias(edge_features).squeeze(-1)
            )

        create_valid_mask = adjacency == 0
        remove_valid_mask = adjacency > 0
        pair_features = self._pair_feature_tensor(actual_embeddings, edge_features)
        relation_logits = self.relation_pair_head(pair_features)
        relation_logits = relation_logits + self._relation_role_priors(
            node_order=node_order,
            temporal_graph=temporal_graph,
        )

        # Completion detection
        completion_scores: torch.Tensor | None = None
        if self.completion_classifier is not None and n_actual > 0:
            completion_scores = self.completion_classifier(actual_embeddings)  # [N]

        return ActionScoreBundle(
            node_order=node_order,
            adjacency=adjacency,
            create_scores=create_scores,
            remove_scores=remove_scores,
            create_valid_mask=create_valid_mask,
            remove_valid_mask=remove_valid_mask,
            relation_logits=relation_logits,
            role_logits=self.add_node_head(graph_embedding),
            count_logits=self.action_count_head(graph_embedding),
            no_op_logit=self.no_op_head(graph_embedding).view(()),
            graph_embedding=graph_embedding,
            completion_scores=completion_scores,
        )

    def action_type_logits(self, score_bundle: ActionScoreBundle) -> torch.Tensor:
        return torch.stack(
            [
                self._pooled_edge_logit(score_bundle.create_scores, score_bundle.create_valid_mask),
                self._pooled_edge_logit(score_bundle.remove_scores, score_bundle.remove_valid_mask),
                torch.logsumexp(score_bundle.role_logits, dim=0),
                score_bundle.no_op_logit,
            ]
        )

    def predict_rollout(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
        steps: int | None = None,
        time_schedule: list[float] | None = None,
    ) -> PredictionRollout:
        if time_schedule is not None:
            steps = len(time_schedule)
        else:
            steps = steps or self.config.prediction_horizon
        rollout_graph = temporal_graph.clone()
        rollout_ctdg = ctdg.clone_with_graph(rollout_graph)
        actions: list[PredictedGraphAction] = []
        for offset in range(steps):
            step_time = (
                time_schedule[offset]
                if time_schedule is not None
                else observation_time + float(offset + 1)
            )
            action = self.predict_next_action(
                temporal_graph=rollout_graph,
                ctdg=rollout_ctdg,
                observation_time=step_time,
            )
            actions.append(action)
            self.apply_action(
                action=action,
                temporal_graph=rollout_graph,
                ctdg=rollout_ctdg,
                update_state=True,
            )
        return PredictionRollout(actions=actions, temporal_graph=rollout_graph, ctdg=rollout_ctdg)

    def predict_subgraph_rollout(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
        steps: int | None = None,
        time_schedule: list[float] | None = None,
    ) -> PredictionSubgraphRollout:
        if time_schedule is not None:
            steps = len(time_schedule)
        else:
            steps = steps or self.config.prediction_horizon
        rollout_graph = temporal_graph.clone()
        rollout_ctdg = ctdg.clone_with_graph(rollout_graph)
        actions_by_step: list[list[PredictedGraphAction]] = []
        for offset in range(steps):
            step_time = (
                time_schedule[offset]
                if time_schedule is not None
                else observation_time + float(offset + 1)
            )
            action_set = self.predict_action_set(
                temporal_graph=rollout_graph,
                ctdg=rollout_ctdg,
                observation_time=step_time,
            )
            actions_by_step.append(action_set)
            for action in action_set:
                self.apply_action(
                    action=action,
                    temporal_graph=rollout_graph,
                    ctdg=rollout_ctdg,
                    update_state=True,
                )
        return PredictionSubgraphRollout(
            actions_by_step=actions_by_step,
            temporal_graph=rollout_graph,
            ctdg=rollout_ctdg,
        )

    def apply_action(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        update_state: bool = False,
    ) -> None:
        generated_message: Message | None = None
        if action.action_type == GraphActionType.CREATE_EDGE:
            if action.source_node_id is None or action.target_node_id is None:
                return
            if temporal_graph.has_active_edge(
                action.source_node_id, action.target_node_id, action.effective_time
            ):
                return
            temporal_graph.add_edge(
                TemporalEdge(
                    source_node_id=action.source_node_id,
                    target_node_id=action.target_node_id,
                    start_time=action.effective_time,
                    end_time=action.effective_time + self.config.prediction_edge_duration,
                )
            )
            if update_state:
                generated_message = self._build_rollout_message(
                    action=action,
                    temporal_graph=temporal_graph,
                    ctdg=ctdg,
                )
        elif action.action_type == GraphActionType.REMOVE_EDGE:
            if action.source_node_id is None or action.target_node_id is None:
                return
            temporal_graph.deactivate_edge(
                action.source_node_id,
                action.target_node_id,
                action.effective_time,
            )
            if update_state:
                generated_message = self._build_rollout_message(
                    action=action,
                    temporal_graph=temporal_graph,
                    ctdg=ctdg,
                )
        elif action.action_type == GraphActionType.ADD_NODE:
            role = action.role or "new_role"
            node_id = action.new_node_id or temporal_graph.generate_node_id(role)
            action.new_node_id = node_id
            temporal_graph.add_node(
                TemporalNode(
                    node_id=node_id,
                    role=role,
                    context=ensure_tensor(None, self.config.context_dim, self.device),
                )
            )
            ctdg.add_node(node_id)
            # Cold-start: initialize with role prototype instead of zero
            if self.cold_start is not None:
                init_state = self.cold_start.initialize_state(role, device=self.device)
                ctdg.current_states[node_id] = init_state.detach()
            if update_state:
                generated_message = Message.build_query_message(
                    target_node_id=node_id,
                    time=action.effective_time,
                    context=self._role_seed_context(role),
                    context_dim=self.config.context_dim,
                    device=self.device,
                )
        if update_state and generated_message is not None:
            ctdg.ingest_messages([generated_message])

    def _best_edge_action(
        self,
        action_type: GraphActionType,
        node_order: list[str],
        adjacency: torch.Tensor,
        scores: torch.Tensor,
        effective_time: float,
        valid_mask: torch.Tensor,
    ) -> PredictedGraphAction:
        if scores.numel() == 0:
            return PredictedGraphAction(
                action_type=action_type,
                score=-math.inf,
                effective_time=effective_time,
            )
        invalid_mask = ~valid_mask.bool()
        diagonal = torch.eye(adjacency.size(0), dtype=torch.bool, device=adjacency.device)
        masked_scores = scores.masked_fill(invalid_mask | diagonal, float("-inf"))
        best_flat_index = int(masked_scores.argmax().item())
        best_score = float(masked_scores.flatten()[best_flat_index].item())
        if not math.isfinite(best_score):
            return PredictedGraphAction(
                action_type=action_type,
                score=-math.inf,
                effective_time=effective_time,
            )
        row = best_flat_index // masked_scores.size(1)
        col = best_flat_index % masked_scores.size(1)
        return PredictedGraphAction(
            action_type=action_type,
            score=best_score,
            effective_time=effective_time,
            source_node_id=node_order[row],
            target_node_id=node_order[col],
        )

    def _candidate_actions(
        self,
        score_bundle: ActionScoreBundle,
        effective_time: float,
        action_logits: torch.Tensor,
    ) -> list[PredictedGraphAction]:
        candidates: list[PredictedGraphAction] = []
        type_log_probs = torch.log_softmax(action_logits, dim=0)

        # Apply completion-aware score adjustment to create_scores
        adjusted_create_scores = score_bundle.create_scores
        if score_bundle.completion_scores is not None and adjusted_create_scores.numel() > 0:
            # Completed nodes are more likely to be sources (they're done, can delegate)
            # Incomplete nodes get penalized as sources
            completion = score_bundle.completion_scores  # [N]
            source_bonus = (completion * 2.0 - 1.0).unsqueeze(1)  # [N, 1]
            adjusted_create_scores = adjusted_create_scores + source_bonus

        candidates.extend(
            self._edge_candidates(
                action_type=GraphActionType.CREATE_EDGE,
                node_order=score_bundle.node_order,
                scores=adjusted_create_scores,
                valid_mask=score_bundle.create_valid_mask,
                relation_logits=score_bundle.relation_logits,
                effective_time=effective_time,
                action_type_log_prob=float(type_log_probs[0].item()),
            )
        )
        candidates.extend(
            self._edge_candidates(
                action_type=GraphActionType.REMOVE_EDGE,
                node_order=score_bundle.node_order,
                scores=score_bundle.remove_scores,
                valid_mask=score_bundle.remove_valid_mask,
                relation_logits=score_bundle.relation_logits,
                effective_time=effective_time,
                action_type_log_prob=float(type_log_probs[1].item()),
            )
        )
        if len(self.config.candidate_new_roles) > 0:
            role_log_probs = torch.log_softmax(score_bundle.role_logits, dim=0)
            for role_index, role_name in enumerate(self.config.candidate_new_roles):
                candidates.append(
                    PredictedGraphAction(
                        action_type=GraphActionType.ADD_NODE,
                        score=float(type_log_probs[2].item() + role_log_probs[role_index].item()),
                        effective_time=effective_time,
                        role=role_name,
                    )
                )
        candidates.sort(key=lambda action: action.score, reverse=True)
        return candidates

    def _edge_candidates(
        self,
        action_type: GraphActionType,
        node_order: list[str],
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        relation_logits: torch.Tensor,
        effective_time: float,
        action_type_log_prob: float,
    ) -> list[PredictedGraphAction]:
        if scores.numel() == 0:
            return []
        diagonal = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        mask = valid_mask.bool() & ~diagonal
        if not bool(mask.any().item()):
            return []
        flat_scores = scores.masked_fill(~mask, float("-inf")).reshape(-1)
        valid_flat_mask = mask.reshape(-1)
        pair_log_probs = torch.full_like(flat_scores, float("-inf"))
        pair_log_probs[valid_flat_mask] = F.log_softmax(flat_scores[valid_flat_mask], dim=0)
        joint_log_probs = pair_log_probs.clone()
        joint_log_probs[valid_flat_mask] = joint_log_probs[valid_flat_mask] + action_type_log_prob
        top_k = min(self.config.max_actions_per_step, int(mask.sum().item()))
        top_values, top_indices = torch.topk(joint_log_probs, k=top_k)
        actions: list[PredictedGraphAction] = []
        size = scores.size(0)
        for value, flat_index in zip(top_values.tolist(), top_indices.tolist()):
            if not math.isfinite(value):
                continue
            row = flat_index // size
            col = flat_index % size
            relation_index = int(relation_logits[row, col].argmax().item())
            relation_log_probs = torch.log_softmax(relation_logits[row, col], dim=0)
            actions.append(
                PredictedGraphAction(
                    action_type=action_type,
                    score=float(value + relation_log_probs[relation_index].item()),
                    effective_time=effective_time,
                    source_node_id=node_order[row],
                    target_node_id=node_order[col],
                    relation_type=self.config.candidate_relation_types[relation_index],
                )
            )
        return actions

    def _action_key(
        self,
        action: PredictedGraphAction,
    ) -> tuple[str, str | None, str | None, str | None, str | None]:
        return (
            action.action_type.value,
            action.source_node_id,
            action.target_node_id,
            action.relation_type,
            action.role,
        )

    def _pooled_edge_logit(self, scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return valid_mask.new_tensor(-1e9, dtype=torch.float32)
        diagonal = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        mask = valid_mask.bool() & ~diagonal
        if not bool(mask.any().item()):
            return scores.new_tensor(-1e9)
        return torch.logsumexp(scores.masked_select(mask), dim=0)

    def _pair_feature_tensor(
        self,
        node_embeddings: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if node_embeddings.numel() == 0:
            return torch.zeros(
                (0, 0, self.config.hidden_dim * 4 + self.config.temporal_edge_dim),
                dtype=torch.float32,
                device=self.device,
            )
        source = node_embeddings.unsqueeze(1).expand(-1, node_embeddings.size(0), -1)
        target = node_embeddings.unsqueeze(0).expand(node_embeddings.size(0), -1, -1)
        return torch.cat(
            [
                source,
                target,
                source * target,
                torch.abs(source - target),
                edge_features,
            ],
            dim=-1,
        )

    def _inject_semantic_edge_features(
        self,
        edge_features: torch.Tensor,
        node_order: list[str],
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
    ) -> torch.Tensor:
        if edge_features.numel() == 0 or edge_features.size(-1) <= 5:
            return edge_features
        contexts = torch.stack(
            [temporal_graph.nodes[node_id].context.to(self.device) for node_id in node_order],
            dim=0,
        )
        states = torch.stack(
            [ctdg.get_state(node_id).to(self.device) for node_id in node_order],
            dim=0,
        )
        projected_contexts = self.node_feature_encoder.context_projection(contexts)
        projected_states = self.node_feature_encoder.state_projection(states)
        normalized_contexts = F.normalize(projected_contexts, p=2, dim=-1, eps=1e-6)
        normalized_states = F.normalize(projected_states, p=2, dim=-1, eps=1e-6)
        context_similarity = normalized_contexts @ normalized_contexts.T
        state_similarity = normalized_states @ normalized_states.T
        context_distance = torch.cdist(projected_contexts, projected_contexts, p=2)
        context_proximity = torch.exp(-context_distance / max(float(self.config.hidden_dim) ** 0.5, 1.0))
        role_match = torch.tensor(
            [
                [
                    1.0 if temporal_graph.nodes[source_id].role == temporal_graph.nodes[target_id].role else 0.0
                    for target_id in node_order
                ]
                for source_id in node_order
            ],
            dtype=edge_features.dtype,
            device=self.device,
        )
        semantic_stack = torch.stack(
            [
                context_similarity,
                state_similarity,
                context_proximity,
                role_match,
            ],
            dim=-1,
        )
        semantic_dim = min(edge_features.size(-1) - 5, semantic_stack.size(-1))
        if semantic_dim > 0:
            edge_features[:, :, 5 : 5 + semantic_dim] = semantic_stack[:, :, :semantic_dim]
        return edge_features

    def _relation_role_priors(
        self,
        node_order: list[str],
        temporal_graph: TemporalGraph,
    ) -> torch.Tensor:
        relation_count = len(self.config.candidate_relation_types)
        priors = torch.zeros(
            (len(node_order), len(node_order), relation_count),
            dtype=torch.float32,
            device=self.device,
        )
        relation_index = {
            relation_type: index
            for index, relation_type in enumerate(self.config.candidate_relation_types)
        }
        for row, source_id in enumerate(node_order):
            source_role = temporal_graph.nodes[source_id].role.lower()
            for col, target_id in enumerate(node_order):
                target_role = temporal_graph.nodes[target_id].role.lower()
                same_node = source_id == target_id
                priors[row, col, relation_index["communication"]] = 0.4 if not same_node else -3.0
                priors[row, col, relation_index["delegation"]] = (
                    0.8
                    if source_role in {"planner", "critic", "leader", "manager"}
                    and not same_node
                    else -1.5
                )
                priors[row, col, relation_index["banishment_vote"]] = 0.7 if not same_node else -3.0
                priors[row, col, relation_index["werewolf_vote"]] = (
                    1.4 if source_role in {"wolf", "werewolf"} and not same_node else -4.0
                )
                priors[row, col, relation_index["werewolf_attack"]] = (
                    1.6
                    if source_role in {"wolf", "werewolf"}
                    and target_role not in {"wolf", "werewolf"}
                    and not same_node
                    else -4.0
                )
                priors[row, col, relation_index["guard_action"]] = (
                    1.5 if source_role == "guard" and not same_node else -4.0
                )
                priors[row, col, relation_index["seer_check"]] = (
                    1.5 if source_role == "seer" and not same_node else -4.0
                )
                priors[row, col, relation_index["witch_save"]] = (
                    1.2 if source_role == "witch" and not same_node else -4.0
                )
                priors[row, col, relation_index["witch_poison"]] = (
                    1.2 if source_role == "witch" and not same_node else -4.0
                )
        return priors

    def _build_rollout_message(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
    ) -> Message | None:
        if action.source_node_id is None or action.target_node_id is None:
            return None
        if action.source_node_id not in temporal_graph.nodes or action.target_node_id not in temporal_graph.nodes:
            return None
        source_context = temporal_graph.nodes[action.source_node_id].context.to(self.device)
        target_context = temporal_graph.nodes[action.target_node_id].context.to(self.device)
        context = 0.5 * (source_context + target_context)
        if action.relation_type in self.config.candidate_relation_types:
            relation_index = self.config.candidate_relation_types.index(action.relation_type)
            relation_context = self.relation_context_embedding.weight[relation_index]
            context = F.normalize(context + relation_context, p=2, dim=0, eps=1e-6)
        message = Message.build_completion_message(
            time=action.effective_time,
            source_node_id=action.source_node_id,
            target_node_id=action.target_node_id,
            source_state=ctdg.get_state(action.source_node_id),
            target_state=ctdg.get_state(action.target_node_id),
            context=context,
            hidden_dim=self.config.hidden_dim,
            context_dim=self.config.context_dim,
            device=self.device,
        )
        if action.relation_type:
            message.metadata["relation_type"] = action.relation_type
        return message

    def _role_seed_context(self, role: str) -> torch.Tensor:
        seed = ensure_tensor(None, self.config.context_dim, self.device)
        if self.config.context_dim == 0:
            return seed
        role_hash = abs(hash(role)) % self.config.context_dim
        seed[role_hash] = 1.0
        return seed
