from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from ..completion import NodeCompletionClassifier
from ..config import ExperimentConfig
from ..ctdg import ContinuousTimeDynamicGraph
from ..encoders import SentenceTransformerEncoder, stable_hash_index
from ..messages import Message
from ..prediction import (
    GraphPredictionContext,
    GraphActionType,
    PredictedGraphAction,
    PredictionRollout,
    PredictionSubgraphRollout,
)
from ..temporal_graph import TemporalEdge, TemporalGraph, TemporalNode
from ..types import ensure_tensor
from .cold_start import ColdStartInitializer
from .cold_start_prior import ColdStartActionPriorScorer
from .few_shot_memory import FewShotTransitionMemory
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
    prediction_context: GraphPredictionContext | None = None
    context_embedding: torch.Tensor | None = None
    action_type_context_logits: torch.Tensor | None = None
    candidate_actions: list[PredictedGraphAction] | None = None
    candidate_scores: torch.Tensor | None = None
    candidate_prior_scores: torch.Tensor | None = None
    candidate_few_shot_scores: torch.Tensor | None = None
    runtime_directed_message_count: int = 0


class GraphActionPredictor(nn.Module):
    supports_gradient_training = True

    def __init__(self, config: ExperimentConfig, node_feature_encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.node_feature_encoder = node_feature_encoder
        self.cold_start_prior_scorer = ColdStartActionPriorScorer()
        self.few_shot_memory = FewShotTransitionMemory(
            max_examples=config.few_shot_memory_max_examples,
        )
        self.node_feature_dropout = nn.Dropout(config.training_node_feature_dropout)
        self.edge_feature_dropout = nn.Dropout(config.training_edge_feature_dropout)
        self.edge_mask_rate = config.training_edge_mask_rate
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
        self.context_node_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.context_node_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Sigmoid(),
        )
        self.context_graph_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.context_graph_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Sigmoid(),
        )
        self.action_context_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 4),
        )
        self.relation_hidden_embedding = nn.Embedding(
            len(config.candidate_relation_types),
            config.hidden_dim,
        )
        self.candidate_edge_projection = nn.Linear(config.temporal_edge_dim, config.hidden_dim)
        self.candidate_cross_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 9, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(config.rt_dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

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

    def reset_few_shot_memory(self) -> None:
        self.few_shot_memory.clear()

    def add_few_shot_transition(
        self,
        *,
        source_role: str,
        target_role: str,
        relation_type: str,
        text: str,
        source_node_id: str = "",
        target_node_id: str = "",
    ) -> None:
        self.few_shot_memory.add(
            source_role=source_role,
            target_role=target_role,
            relation_type=relation_type,
            text=text,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
        )

    def few_shot_memory_size(self) -> int:
        return len(self.few_shot_memory)

    def snapshot_few_shot_memory(self):
        return self.few_shot_memory.snapshot()

    def restore_few_shot_memory(self, snapshot) -> None:
        self.few_shot_memory.restore(snapshot)

    def initialize_ctdg_states(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        graph_context_text: str = "",
    ) -> None:
        """Seed existing nodes from role, node text, graph profile, and structural priors."""
        if self.cold_start is None or not temporal_graph.nodes:
            return
        structural_text_by_node: dict[str, list[str]] = {}
        for (source_id, target_id), items in temporal_graph.structural_edge_metadata.items():
            for item in items:
                relation = str(item.get("relation_type", "")).strip()
                description = str(item.get("description", "")).strip()
                text = " ".join(part for part in (relation, description) if part)
                if not text:
                    continue
                structural_text_by_node.setdefault(source_id, []).append(text)
                structural_text_by_node.setdefault(target_id, []).append(text)
        for node_id, node in temporal_graph.nodes.items():
            node_text = "\n".join(
                part
                for part in (
                    graph_context_text,
                    node.role,
                    node.context_text,
                    "\n".join(structural_text_by_node.get(node_id, [])[:8]),
                )
                if str(part).strip()
            )
            ctdg.current_states[node_id] = self.cold_start.initialize_state_from_text(
                node.role,
                node_text,
                device=self.device,
            )

    def encode_graph(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
        node_feature_cache: tuple[list[str], torch.Tensor] | None = None,
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
        features = None
        if node_feature_cache is not None:
            cached_order, cached_features = node_feature_cache
            if list(cached_order) == node_order:
                features = cached_features.to(self.device)
        if features is None:
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
        structural_adjacency = temporal_graph.structural_adjacency_matrix(
            node_order=node_order,
            device=self.device,
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
        features = self._regularize_node_features(features)
        active_adjacency, message_passing_adjacency, edge_features = self._regularize_graph_inputs(
            active_adjacency=active_adjacency,
            structural_adjacency=structural_adjacency,
            edge_features=edge_features,
        )
        # Build role indices for RT
        role_indices = self._build_role_indices(node_order, temporal_graph)
        encoded = self.gnn_backbone(
            features, message_passing_adjacency, edge_features, role_indices=role_indices,
        )
        return node_order, active_adjacency, edge_features, encoded

    def build_node_feature_cache(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
    ) -> tuple[list[str], torch.Tensor]:
        node_order = sorted(temporal_graph.nodes)
        if not node_order:
            return node_order, torch.zeros(
                (0, self.config.hidden_dim),
                dtype=torch.float32,
                device=self.device,
            )
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
        return node_order, features

    def _regularize_node_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features
        regularized = self.node_feature_dropout(features)
        if not self.training or self.config.training_node_mask_rate <= 0:
            return regularized
        keep_mask = (
            torch.rand((features.size(0), 1), device=features.device) >= self.config.training_node_mask_rate
        ).to(dtype=features.dtype)
        keep_prob = max(1.0 - self.config.training_node_mask_rate, 1e-6)
        return regularized * keep_mask / keep_prob

    def _regularize_graph_inputs(
        self,
        active_adjacency: torch.Tensor,
        structural_adjacency: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        message_passing_adjacency = torch.maximum(active_adjacency, structural_adjacency)
        if not self.training:
            return active_adjacency, message_passing_adjacency, edge_features
        if self.edge_mask_rate > 0 and active_adjacency.numel() > 0:
            active_mask = active_adjacency > 0
            structural_mask = structural_adjacency > 0
            droppable_mask = active_mask & ~structural_mask
            if bool(droppable_mask.any().item()):
                keep_mask = (
                    torch.rand_like(active_adjacency) >= self.edge_mask_rate
                ).to(dtype=active_adjacency.dtype)
                protected_keep_mask = torch.where(
                    droppable_mask,
                    keep_mask,
                    torch.ones_like(keep_mask),
                )
                message_passing_adjacency = torch.maximum(
                    active_adjacency * protected_keep_mask,
                    structural_adjacency,
                )
                edge_features = edge_features * protected_keep_mask.unsqueeze(-1)
        edge_features = self.edge_feature_dropout(edge_features)
        return active_adjacency, message_passing_adjacency, edge_features

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
        prediction_context: GraphPredictionContext | None = None,
    ) -> PredictedGraphAction:
        action_set = self.predict_action_set(
            temporal_graph=temporal_graph,
            ctdg=ctdg,
            observation_time=observation_time,
            prediction_context=prediction_context,
        )
        return action_set[0]

    def predict_action_set(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
        prediction_context: GraphPredictionContext | None = None,
    ) -> list[PredictedGraphAction]:
        score_bundle = self.score_action_space(
            temporal_graph=temporal_graph,
            ctdg=ctdg,
            observation_time=observation_time,
            prediction_context=prediction_context,
        )
        action_logits = self.action_type_logits(score_bundle)
        predicted_count = int(score_bundle.count_logits.argmax().item())
        no_op_action = PredictedGraphAction(
            action_type=GraphActionType.NO_OP,
            score=float(action_logits[3].item()),
            effective_time=observation_time,
        )
        candidates = self._candidate_actions(score_bundle, observation_time, action_logits)
        best_non_noop_logit = float(action_logits[:3].max().item())
        if float(action_logits[3].item()) >= best_non_noop_logit:
            return [no_op_action]
        if predicted_count <= 0:
            if candidates and best_non_noop_logit > float(action_logits[3].item()):
                predicted_count = 1
            else:
                return [no_op_action]
        if not candidates:
            return [no_op_action]
        selected: list[PredictedGraphAction] = []
        seen: set[tuple[str, str | None, str | None, str | None, str | None]] = set()
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
        prediction_context: GraphPredictionContext | None = None,
        node_feature_cache: tuple[list[str], torch.Tensor] | None = None,
    ) -> ActionScoreBundle:
        prediction_context = self._runtime_enriched_prediction_context(
            temporal_graph=temporal_graph,
            observation_time=observation_time,
            prediction_context=prediction_context,
        )
        node_order, adjacency, edge_features, node_embeddings = self.encode_graph(
            temporal_graph=temporal_graph,
            ctdg=ctdg,
            observation_time=observation_time,
            node_feature_cache=node_feature_cache,
        )
        # Number of actual nodes (cold start may add virtual embeddings)
        n_actual = len(node_order)

        # Use only actual node embeddings for edge scoring
        actual_embeddings = node_embeddings[:n_actual] if node_embeddings.size(0) > n_actual else node_embeddings
        graph_source_embeddings = actual_embeddings if n_actual > 0 else node_embeddings
        graph_embedding = self.graph_embedding_from_encoded(graph_source_embeddings)
        context_embedding = self._context_text_embedding(prediction_context)
        if self.config.use_context_conditioning and context_embedding is not None:
            if n_actual > 0:
                actual_embeddings = self._condition_node_embeddings(
                    node_embeddings=actual_embeddings,
                    context_embedding=context_embedding,
                )
                graph_embedding = self.graph_embedding_from_encoded(actual_embeddings)
            graph_embedding = self._condition_graph_embedding(
                graph_embedding=graph_embedding,
                context_embedding=context_embedding,
            )

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
            create_scores = self._apply_prediction_context_bias(
                scores=create_scores,
                node_order=node_order,
                node_embeddings=actual_embeddings,
                graph_embedding=graph_embedding,
                prediction_context=prediction_context,
                context_embedding=context_embedding,
            )
            create_scores = self._apply_zero_shot_create_priors(
                scores=create_scores,
                temporal_graph=temporal_graph,
                prediction_context=prediction_context,
                node_order=node_order,
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

        (
            candidate_actions,
            candidate_scores,
            candidate_prior_scores,
            candidate_few_shot_scores,
        ) = self._score_prediction_context_candidates(
            prediction_context=prediction_context,
            temporal_graph=temporal_graph,
            node_order=node_order,
            create_scores=create_scores,
            create_valid_mask=create_valid_mask,
            relation_logits=relation_logits,
            edge_features=edge_features,
            node_embeddings=actual_embeddings,
            graph_embedding=graph_embedding,
            context_embedding=context_embedding,
        )
        (
            runtime_message_actions,
            runtime_message_scores,
        ) = self._score_runtime_message_candidates(
            ctdg=ctdg,
            temporal_graph=temporal_graph,
            observation_time=observation_time,
            device=graph_embedding.device,
            dtype=graph_embedding.dtype,
        )
        if runtime_message_actions:
            candidate_actions = [*runtime_message_actions, *(candidate_actions or [])]
            candidate_scores = self._concat_optional_scores(
                runtime_message_scores,
                candidate_scores,
            )
            runtime_prior_scores = torch.full(
                (len(runtime_message_actions),),
                self.config.runtime_message_candidate_score,
                dtype=graph_embedding.dtype,
                device=graph_embedding.device,
            )
            candidate_prior_scores = self._concat_optional_scores(
                runtime_prior_scores,
                candidate_prior_scores,
            )
            runtime_few_shot_scores = torch.zeros(
                len(runtime_message_actions),
                dtype=graph_embedding.dtype,
                device=graph_embedding.device,
            )
            candidate_few_shot_scores = self._concat_optional_scores(
                runtime_few_shot_scores,
                candidate_few_shot_scores,
            )
        count_logits = self.action_count_head(graph_embedding)
        if candidate_actions:
            count_logits = count_logits.clone()
            count_logits[0] = count_logits[0] - 2.0
            count_logits[1] = count_logits[1] + 2.0
            candidate_count = min(len(candidate_actions), self.config.max_actions_per_step)
            if candidate_count > 1:
                count_logits[candidate_count] = count_logits[candidate_count] + 0.5
        action_type_context_logits = None
        if self.config.use_context_conditioning and context_embedding is not None:
            action_type_context_logits = self.action_context_head(
                torch.cat([graph_embedding, context_embedding], dim=-1)
            )

        return ActionScoreBundle(
            node_order=node_order,
            adjacency=adjacency,
            create_scores=create_scores,
            remove_scores=remove_scores,
            create_valid_mask=create_valid_mask,
            remove_valid_mask=remove_valid_mask,
            relation_logits=relation_logits,
            role_logits=self.add_node_head(graph_embedding),
            count_logits=count_logits,
            no_op_logit=self.no_op_head(graph_embedding).view(()),
            graph_embedding=graph_embedding,
            completion_scores=completion_scores,
            prediction_context=prediction_context,
            context_embedding=context_embedding,
            action_type_context_logits=action_type_context_logits,
            candidate_actions=candidate_actions,
            candidate_scores=candidate_scores,
            candidate_prior_scores=candidate_prior_scores,
            candidate_few_shot_scores=candidate_few_shot_scores,
            runtime_directed_message_count=len(runtime_message_actions),
        )

    def _apply_prediction_context_bias(
        self,
        scores: torch.Tensor,
        node_order: list[str],
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        prediction_context: GraphPredictionContext | None,
        context_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction_context is None or scores.numel() == 0:
            return scores
        adjusted = scores
        if prediction_context.source_node_id in node_order:
            source_index = node_order.index(str(prediction_context.source_node_id))
            source_bias = torch.zeros_like(adjusted)
            source_bias[source_index, :] = self.config.context_source_bias_weight
            adjusted = adjusted + source_bias
        if context_embedding is None or node_embeddings.numel() == 0:
            return adjusted
        normalized_nodes = F.normalize(node_embeddings, p=2, dim=-1, eps=1e-6)
        normalized_context = F.normalize(context_embedding, p=2, dim=0, eps=1e-6)
        node_context_scores = normalized_nodes @ normalized_context
        graph_context_score = torch.dot(
            F.normalize(graph_embedding, p=2, dim=0, eps=1e-6),
            normalized_context,
        )
        return (
            adjusted
            + 0.35 * node_context_scores.unsqueeze(1)
            + 0.15 * node_context_scores.unsqueeze(0)
            + 0.10 * graph_context_score
        )

    def _condition_node_embeddings(
        self,
        node_embeddings: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if node_embeddings.numel() == 0:
            return node_embeddings
        expanded_context = context_embedding.unsqueeze(0).expand_as(node_embeddings)
        fusion_features = torch.cat(
            [
                node_embeddings,
                expanded_context,
                node_embeddings * expanded_context,
                torch.abs(node_embeddings - expanded_context),
            ],
            dim=-1,
        )
        gate = self.context_node_gate(fusion_features)
        delta = self.context_node_fusion(fusion_features)
        return node_embeddings + gate * delta

    def _condition_graph_embedding(
        self,
        graph_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        fusion_features = torch.cat([graph_embedding, context_embedding], dim=-1)
        gate = self.context_graph_gate(fusion_features)
        delta = self.context_graph_fusion(fusion_features)
        return graph_embedding + gate * delta

    def _apply_zero_shot_create_priors(
        self,
        scores: torch.Tensor,
        temporal_graph: TemporalGraph,
        prediction_context: GraphPredictionContext | None,
        node_order: list[str],
    ) -> torch.Tensor:
        if scores.numel() == 0 or (
            prediction_context is not None and prediction_context.candidate_actions
        ):
            return scores
        adjusted = scores
        if (
            self.config.use_zero_shot_action_priors
            and prediction_context is not None
            and self.config.zero_shot_prior_weight > 0
        ):
            prior_matrix = self.cold_start_prior_scorer.edge_prior_matrix(
                temporal_graph=temporal_graph,
                prediction_context=prediction_context,
                node_order=node_order,
                device=scores.device,
                dtype=scores.dtype,
            )
            adjusted = adjusted + self.config.zero_shot_prior_weight * prior_matrix
        if self._uses_few_shot_memory() and self.config.few_shot_memory_weight > 0:
            memory_matrix = self.few_shot_memory.edge_prior_matrix(
                temporal_graph=temporal_graph,
                prediction_context=prediction_context,
                node_order=node_order,
                device=scores.device,
                dtype=scores.dtype,
            )
            adjusted = adjusted + self.config.few_shot_memory_weight * memory_matrix
        return adjusted

    def _context_text_embedding(
        self,
        prediction_context: GraphPredictionContext | None,
    ) -> torch.Tensor | None:
        if prediction_context is None:
            return None
        context_text = prediction_context.combined_text().strip()
        if not context_text:
            return None
        return self.node_feature_encoder.text_encoder(context_text, device=self.device)

    def _runtime_enriched_prediction_context(
        self,
        temporal_graph: TemporalGraph,
        observation_time: float,
        prediction_context: GraphPredictionContext | None,
    ) -> GraphPredictionContext | None:
        if prediction_context is None or not self.config.use_runtime_context_features:
            return prediction_context
        runtime_text = self._runtime_context_text(
            temporal_graph=temporal_graph,
            observation_time=observation_time,
            prediction_context=prediction_context,
        )
        if not runtime_text:
            return prediction_context
        existing = str(prediction_context.runtime_text or "").strip()
        merged_runtime_text = "\n".join(
            part for part in (existing, runtime_text) if part
        )
        if merged_runtime_text == existing:
            return prediction_context
        return replace(prediction_context, runtime_text=merged_runtime_text)

    def _runtime_context_text(
        self,
        temporal_graph: TemporalGraph,
        observation_time: float,
        prediction_context: GraphPredictionContext,
    ) -> str:
        parts = [
            f"observation_time={observation_time}",
            f"node_count={len(temporal_graph.nodes)}",
            f"active_edge_count={len(temporal_graph.active_edges(observation_time))}",
            f"structural_edge_count={len(temporal_graph.structural_edges)}",
            f"few_shot_memory_size={len(self.few_shot_memory)}",
            f"candidate_count={len(prediction_context.candidate_actions)}",
        ]
        if prediction_context.source_node_id in temporal_graph.nodes:
            source = temporal_graph.nodes[str(prediction_context.source_node_id)]
            parts.append(f"source_role={source.role}")
            if source.context_text:
                parts.append(f"source_context={source.context_text}")
        metadata_parts = [
            f"{key}={value}"
            for key, value in sorted(prediction_context.metadata.items())
            if str(value).strip()
        ]
        if metadata_parts:
            parts.append("metadata " + " ".join(metadata_parts))
        max_edges = self.config.runtime_context_max_edges
        if max_edges > 0:
            edge_texts: list[str] = []
            for edge in temporal_graph.active_edges(observation_time)[-max_edges:]:
                if edge.source_node_id not in temporal_graph.nodes or edge.target_node_id not in temporal_graph.nodes:
                    continue
                source = temporal_graph.nodes[edge.source_node_id]
                target = temporal_graph.nodes[edge.target_node_id]
                edge_texts.append(f"{source.role}:{edge.source_node_id}->{target.role}:{edge.target_node_id}")
            if edge_texts:
                parts.append("recent_active_edges=" + " ".join(edge_texts))
        return "\n".join(part for part in parts if str(part).strip())

    def _score_prediction_context_candidates(
        self,
        prediction_context: GraphPredictionContext | None,
        temporal_graph: TemporalGraph,
        node_order: list[str],
        create_scores: torch.Tensor,
        create_valid_mask: torch.Tensor,
        relation_logits: torch.Tensor,
        edge_features: torch.Tensor,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        context_embedding: torch.Tensor | None,
    ) -> tuple[
        list[PredictedGraphAction] | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if (
            prediction_context is None
            or not prediction_context.candidate_actions
            or create_scores.numel() == 0
        ):
            return None, None, None, None
        relation_index = {
            relation_type: index
            for index, relation_type in enumerate(self.config.candidate_relation_types)
        }
        scored_actions: list[PredictedGraphAction] = []
        scores: list[torch.Tensor] = []
        prior_scores: list[torch.Tensor] = []
        few_shot_scores: list[torch.Tensor] = []
        for action in prediction_context.candidate_actions:
            if action.action_type != GraphActionType.CREATE_EDGE:
                continue
            if (
                action.source_node_id is None
                or action.target_node_id is None
                or action.source_node_id not in node_order
                or action.target_node_id not in node_order
                or action.relation_type not in relation_index
            ):
                continue
            row = node_order.index(action.source_node_id)
            col = node_order.index(action.target_node_id)
            if not self.config.allow_self_loop_prediction and row == col:
                continue
            if not bool(create_valid_mask[row, col].item()):
                continue
            relation_idx = relation_index[action.relation_type]
            learned_score = create_scores[row, col] + relation_logits[row, col, relation_idx]
            text_embedding = self._candidate_text_embedding(
                action=action,
                prediction_context=prediction_context,
            )
            learned_score = learned_score + self._candidate_text_score(
                row=row,
                col=col,
                node_embeddings=node_embeddings,
                graph_embedding=graph_embedding,
                text_embedding=text_embedding,
            )
            if self.config.use_candidate_cross_encoder:
                learned_score = learned_score + self._candidate_cross_encoder_score(
                    action=action,
                    row=row,
                    col=col,
                    relation_idx=relation_idx,
                    edge_features=edge_features,
                    node_embeddings=node_embeddings,
                    graph_embedding=graph_embedding,
                    context_embedding=context_embedding,
                    text_embedding=text_embedding,
                )
            if self.config.use_structural_candidate_priors:
                learned_score = learned_score + self._structural_candidate_bonus(
                    action=action,
                    temporal_graph=temporal_graph,
                )
            candidate_score = self.config.learned_candidate_score_weight * learned_score
            prior_score = self._zero_shot_candidate_prior_score(
                action=action,
                temporal_graph=temporal_graph,
                prediction_context=prediction_context,
                graph_embedding=graph_embedding,
            )
            candidate_score = candidate_score + self.config.zero_shot_prior_weight * prior_score
            few_shot_score = self._few_shot_candidate_score(
                action=action,
                temporal_graph=temporal_graph,
                prediction_context=prediction_context,
                graph_embedding=graph_embedding,
            )
            candidate_score = candidate_score + self.config.few_shot_memory_weight * few_shot_score
            scored_actions.append(action)
            scores.append(candidate_score)
            prior_scores.append(prior_score)
            few_shot_scores.append(few_shot_score)
        if not scores:
            return None, None, None, None
        return (
            scored_actions,
            torch.stack(scores),
            torch.stack(prior_scores),
            torch.stack(few_shot_scores),
        )

    def _zero_shot_candidate_prior_score(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        prediction_context: GraphPredictionContext | None,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.use_zero_shot_action_priors:
            return graph_embedding.new_tensor(0.0)
        score = self.cold_start_prior_scorer.candidate_score(
            action=action,
            temporal_graph=temporal_graph,
            prediction_context=prediction_context,
        )
        return graph_embedding.new_tensor(float(score))

    def _few_shot_candidate_score(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
        prediction_context: GraphPredictionContext | None,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if not self._uses_few_shot_memory():
            return graph_embedding.new_tensor(0.0)
        score = self.few_shot_memory.candidate_score(
            action=action,
            temporal_graph=temporal_graph,
            prediction_context=prediction_context,
        )
        return graph_embedding.new_tensor(float(score))

    def _uses_few_shot_memory(self) -> bool:
        return (
            self.config.use_few_shot_transition_memory
            and len(self.few_shot_memory) > 0
        )

    def _candidate_text_embedding(
        self,
        action: PredictedGraphAction,
        prediction_context: GraphPredictionContext,
    ) -> torch.Tensor | None:
        text = "\n".join(
            part
            for part in (
                prediction_context.query_text,
                str(action.metadata.get("description", "")),
                str(action.metadata.get("transition_id", "")),
            )
            if part.strip()
        )
        if not text.strip():
            return None
        return self.node_feature_encoder.text_encoder(text, device=self.device)

    def _candidate_text_score(
        self,
        row: int,
        col: int,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        text_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if text_embedding is None or node_embeddings.numel() == 0:
            return graph_embedding.new_tensor(0.0)
        pair_embedding = node_embeddings[row] + node_embeddings[col] + graph_embedding
        return self.config.candidate_text_score_weight * torch.dot(
            F.normalize(pair_embedding, p=2, dim=0, eps=1e-6),
            F.normalize(text_embedding, p=2, dim=0, eps=1e-6),
        )

    def _candidate_cross_encoder_score(
        self,
        action: PredictedGraphAction,
        row: int,
        col: int,
        relation_idx: int,
        edge_features: torch.Tensor,
        node_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        context_embedding: torch.Tensor | None,
        text_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if node_embeddings.numel() == 0:
            return graph_embedding.new_tensor(0.0)
        zero = graph_embedding.new_zeros(self.config.hidden_dim)
        source_embedding = node_embeddings[row]
        target_embedding = node_embeddings[col]
        context_vector = context_embedding if context_embedding is not None else zero
        text_vector = text_embedding if text_embedding is not None else zero
        relation_vector = self.relation_hidden_embedding(
            torch.tensor(relation_idx, dtype=torch.long, device=self.device)
        )
        edge_vector = self.candidate_edge_projection(edge_features[row, col])
        candidate_features = torch.cat(
            [
                source_embedding,
                target_embedding,
                graph_embedding,
                context_vector,
                text_vector,
                relation_vector,
                source_embedding * target_embedding,
                torch.abs(source_embedding - target_embedding),
                edge_vector,
            ],
            dim=-1,
        )
        return self.candidate_cross_encoder(candidate_features).view(())

    def _structural_candidate_bonus(
        self,
        action: PredictedGraphAction,
        temporal_graph: TemporalGraph,
    ) -> torch.Tensor:
        device_value = self.empty_graph_embedding.new_tensor(0.0)
        if action.source_node_id is None or action.target_node_id is None:
            return device_value
        key = (action.source_node_id, action.target_node_id)
        bonus = 0.0
        if key in temporal_graph.structural_edges:
            bonus += 0.25
        metadata_items = temporal_graph.structural_edge_metadata.get(key, [])
        action_relation = str(action.relation_type or "").strip().lower()
        action_transition = str(action.metadata.get("transition_id", "")).strip()
        action_description = str(action.metadata.get("description", "")).strip().lower()
        for item in metadata_items:
            relation = str(item.get("relation_type", "")).strip().lower()
            transition_id = str(item.get("transition_id", "")).strip()
            description = str(item.get("description", "")).strip().lower()
            if action_relation and relation == action_relation:
                bonus += 0.50
            if action_transition and transition_id == action_transition:
                bonus += 0.35
            if action_description and description and (
                action_description in description or description in action_description
            ):
                bonus += 0.20
        return device_value + self.config.candidate_structural_prior_weight * bonus

    def action_type_logits(self, score_bundle: ActionScoreBundle) -> torch.Tensor:
        create_logit = self._pooled_edge_logit(
            score_bundle.create_scores,
            score_bundle.create_valid_mask,
        )
        if score_bundle.candidate_scores is not None and score_bundle.candidate_scores.numel() > 0:
            create_logit = torch.logaddexp(
                create_logit,
                torch.logsumexp(score_bundle.candidate_scores, dim=0)
                + self.config.candidate_action_type_boost,
            )
        if self._has_cold_start_create_signal(score_bundle):
            create_logit = create_logit + self.config.zero_shot_action_type_boost
        add_node_logit = (
            torch.logsumexp(score_bundle.role_logits, dim=0)
            if self.config.enable_add_node_prediction and score_bundle.role_logits.numel() > 0
            else create_logit.new_tensor(float("-inf"))
        )
        logits = torch.stack(
            [
                create_logit + self.config.create_action_bias,
                self._pooled_edge_logit(score_bundle.remove_scores, score_bundle.remove_valid_mask)
                + self.config.remove_action_bias,
                add_node_logit + self.config.add_node_action_bias,
                score_bundle.no_op_logit + self.config.no_op_action_bias,
            ]
        )
        if score_bundle.action_type_context_logits is not None:
            logits = logits + score_bundle.action_type_context_logits
        if (
            score_bundle.runtime_directed_message_count <= 0
            and not score_bundle.candidate_actions
            and self.config.no_directed_message_noop_bias > 0
        ):
            logits = logits.clone()
            logits[3] = logits[3] + self.config.no_directed_message_noop_bias
        return logits

    def _score_runtime_message_candidates(
        self,
        *,
        ctdg: ContinuousTimeDynamicGraph,
        temporal_graph: TemporalGraph,
        observation_time: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[list[PredictedGraphAction], torch.Tensor]:
        if self.config.runtime_message_candidate_score <= 0:
            return [], torch.zeros(0, dtype=dtype, device=device)
        actions: list[PredictedGraphAction] = []
        scores: list[float] = []
        seen: set[tuple[str, str, str]] = set()
        tolerance = self.config.runtime_message_time_tolerance
        recent_messages = sorted(
            (
                message
                for message in ctdg.message_history
                if self._message_visible_for_prediction(
                    message=message,
                    observation_time=observation_time,
                    tolerance=tolerance,
                )
            ),
            key=lambda message: float(message.time),
        )
        for order_index, message in enumerate(recent_messages):
            if message.source_node_id is None or message.target_node_id is None:
                continue
            if message.source_node_id == message.target_node_id:
                continue
            if message.source_node_id not in temporal_graph.nodes:
                continue
            if message.target_node_id not in temporal_graph.nodes:
                continue
            relation_type = self._message_relation_type(message)
            key = (message.source_node_id, message.target_node_id, relation_type)
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                PredictedGraphAction(
                    action_type=GraphActionType.CREATE_EDGE,
                    score=self.config.runtime_message_candidate_score,
                    effective_time=observation_time,
                    source_node_id=message.source_node_id,
                    target_node_id=message.target_node_id,
                    relation_type=relation_type,
                    metadata={
                        "source": "runtime_message",
                        "message_time": str(message.time),
                        "raw_text": str(message.metadata.get("raw_text", "")),
                    },
                )
            )
            scores.append(
                self.config.runtime_message_candidate_score
                - (0.01 * order_index)
                - abs(float(message.time) - float(observation_time))
            )
        return actions, torch.tensor(scores, dtype=dtype, device=device)

    def _message_visible_for_prediction(
        self,
        *,
        message: Message,
        observation_time: float,
        tolerance: float,
    ) -> bool:
        message_time = float(message.time)
        observation_time = float(observation_time)
        if message_time < observation_time:
            return observation_time - message_time <= tolerance
        if message_time == observation_time:
            if self.config.allow_same_time_runtime_messages:
                return True
            return bool(
                message.metadata.get("available_for_prediction")
                or message.metadata.get("observed_before_prediction")
            )
        return False

    def _message_relation_type(self, message: Message) -> str:
        if message.metadata and message.metadata.get("relation_type"):
            relation_type = str(message.metadata["relation_type"]).strip()
            if relation_type:
                return relation_type
        raw_text = str((message.metadata or {}).get("raw_text", "")).lower()
        for prefix, relation_type in (
            ("banishment_vote:", "banishment_vote"),
            ("werewolf_vote:", "werewolf_vote"),
            ("werewolf_final_target:", "werewolf_attack"),
            ("guard_action:", "guard_action"),
            ("seer_check:", "seer_check"),
            ("witch_save:", "witch_save"),
            ("witch_poison:", "witch_poison"),
        ):
            if prefix in raw_text:
                return relation_type
        if "delegate" in raw_text or "assign" in raw_text:
            return "delegation"
        return "communication"

    def _concat_optional_scores(
        self,
        left: torch.Tensor,
        right: torch.Tensor | None,
    ) -> torch.Tensor:
        if right is None or right.numel() == 0:
            return left
        if left.numel() == 0:
            return right
        return torch.cat([left, right.to(device=left.device, dtype=left.dtype)], dim=0)

    def _has_cold_start_create_signal(self, score_bundle: ActionScoreBundle) -> bool:
        if self.config.zero_shot_action_type_boost <= 0:
            return False
        if score_bundle.candidate_prior_scores is not None and score_bundle.candidate_prior_scores.numel() > 0:
            if bool((score_bundle.candidate_prior_scores > 0).any().item()):
                return True
        if (
            score_bundle.candidate_few_shot_scores is not None
            and score_bundle.candidate_few_shot_scores.numel() > 0
        ):
            if bool((score_bundle.candidate_few_shot_scores > 0).any().item()):
                return True
        return (
            self.config.use_zero_shot_action_priors
            and score_bundle.prediction_context is not None
            and not score_bundle.prediction_context.candidate_actions
        )

    def predict_rollout(
        self,
        temporal_graph: TemporalGraph,
        ctdg: ContinuousTimeDynamicGraph,
        observation_time: float,
        steps: int | None = None,
        time_schedule: list[float] | None = None,
        prediction_context_schedule: list[GraphPredictionContext | None] | None = None,
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
                prediction_context=(
                    prediction_context_schedule[offset]
                    if prediction_context_schedule and offset < len(prediction_context_schedule)
                    else None
                ),
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
        prediction_context_schedule: list[GraphPredictionContext | None] | None = None,
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
                prediction_context=(
                    prediction_context_schedule[offset]
                    if prediction_context_schedule and offset < len(prediction_context_schedule)
                    else None
                ),
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
        diagonal = self._diagonal_mask(adjacency.size(0), device=adjacency.device)
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

        context_candidates = self._context_ranked_actions(
            score_bundle=score_bundle,
            effective_time=effective_time,
            type_log_probs=type_log_probs,
        )
        if context_candidates and not self.config.include_graph_candidates_with_context_candidates:
            return context_candidates
        candidates.extend(context_candidates)

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
        if self.config.enable_add_node_prediction and len(self.config.candidate_new_roles) > 0:
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
        return self._deduplicate_sorted_actions(candidates)

    def _context_ranked_actions(
        self,
        score_bundle: ActionScoreBundle,
        effective_time: float,
        type_log_probs: torch.Tensor,
    ) -> list[PredictedGraphAction]:
        if (
            not score_bundle.candidate_actions
            or score_bundle.candidate_scores is None
            or score_bundle.candidate_scores.numel() == 0
        ):
            return []
        ranked: list[PredictedGraphAction] = []
        for action, score in zip(score_bundle.candidate_actions, score_bundle.candidate_scores):
            if action.action_type == GraphActionType.CREATE_EDGE:
                type_index = 0
            elif action.action_type == GraphActionType.REMOVE_EDGE:
                type_index = 1
            elif action.action_type == GraphActionType.ADD_NODE:
                type_index = 2
            else:
                type_index = 3
            ranked.append(
                PredictedGraphAction(
                    action_type=action.action_type,
                    score=float((score + type_log_probs[type_index]).item()),
                    effective_time=effective_time,
                    source_node_id=action.source_node_id,
                    target_node_id=action.target_node_id,
                    relation_type=action.relation_type,
                    role=action.role,
                    new_node_id=action.new_node_id,
                    metadata=dict(action.metadata),
                )
            )
        ranked.sort(key=lambda action: action.score, reverse=True)
        return ranked

    def _deduplicate_sorted_actions(
        self,
        actions: list[PredictedGraphAction],
    ) -> list[PredictedGraphAction]:
        actions.sort(key=lambda action: action.score, reverse=True)
        deduped: list[PredictedGraphAction] = []
        seen: set[tuple[str, str | None, str | None, str | None, str | None]] = set()
        for action in actions:
            key = self._action_key(action)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(action)
        return deduped

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
        diagonal = self._diagonal_mask(scores.size(0), device=scores.device)
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
        diagonal = self._diagonal_mask(scores.size(0), device=scores.device)
        mask = valid_mask.bool() & ~diagonal
        if not bool(mask.any().item()):
            return scores.new_tensor(-1e9)
        return torch.logsumexp(scores.masked_select(mask), dim=0)

    def _diagonal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        if self.config.allow_self_loop_prediction:
            return torch.zeros((size, size), dtype=torch.bool, device=device)
        return torch.eye(size, dtype=torch.bool, device=device)

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
                if "communication" in relation_index:
                    priors[row, col, relation_index["communication"]] = (
                        0.4 if not same_node else -3.0
                    )
                if "activate" in relation_index:
                    priors[row, col, relation_index["activate"]] = (
                        1.0 if not same_node else -2.0
                    )
                if "delegation" in relation_index:
                    priors[row, col, relation_index["delegation"]] = (
                        0.8
                        if source_role in {"planner", "critic", "leader", "manager"} and not same_node
                        else -1.5
                    )
                if "delegate" in relation_index:
                    priors[row, col, relation_index["delegate"]] = (
                        0.8
                        if source_role in {"planner", "critic", "leader", "manager", "coding_analyst"}
                        and not same_node
                        else -1.5
                    )
                if "delegate_return" in relation_index:
                    priors[row, col, relation_index["delegate_return"]] = (
                        0.7 if not same_node else -2.0
                    )
                if "retry" in relation_index:
                    priors[row, col, relation_index["retry"]] = 1.2 if same_node else -2.5
                if "banishment_vote" in relation_index:
                    priors[row, col, relation_index["banishment_vote"]] = (
                        0.7 if not same_node else -3.0
                    )
                if "werewolf_vote" in relation_index:
                    priors[row, col, relation_index["werewolf_vote"]] = (
                        1.4 if source_role in {"wolf", "werewolf"} and not same_node else -4.0
                    )
                if "werewolf_attack" in relation_index:
                    priors[row, col, relation_index["werewolf_attack"]] = (
                        1.6
                        if source_role in {"wolf", "werewolf"}
                        and target_role not in {"wolf", "werewolf"}
                        and not same_node
                        else -4.0
                    )
                if "guard_action" in relation_index:
                    priors[row, col, relation_index["guard_action"]] = (
                        1.5 if source_role == "guard" and not same_node else -4.0
                    )
                if "seer_check" in relation_index:
                    priors[row, col, relation_index["seer_check"]] = (
                        1.5 if source_role == "seer" and not same_node else -4.0
                    )
                if "witch_save" in relation_index:
                    priors[row, col, relation_index["witch_save"]] = (
                        1.2 if source_role == "witch" and not same_node else -4.0
                    )
                if "witch_poison" in relation_index:
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
