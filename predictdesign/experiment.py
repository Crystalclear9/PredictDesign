from __future__ import annotations

import torch
from torch import nn

from .aggregation import ConcurrentMessageAggregator
from .config import ExperimentConfig
from .ctdg import ContinuousTimeDynamicGraph
from .encoders import MessageEncoder, NodeFeatureEncoder
from .gnn import GraphActionPredictor
from .messages import Message
from .prediction import (
    GraphActionType,
    GraphPredictionContext,
    PredictedGraphAction,
    PredictionRollout,
    PredictionSubgraphRollout,
)
from .query_parser import QueryParseResult, QueryParser
from .llm import LLMApiGraphActionPredictor
from .state_update import build_state_updater
from .temporal_graph import TemporalEdge, TemporalGraph, TemporalNode
from .types import TensorLike


class PredictDesignSystem(nn.Module):
    def __init__(
        self,
        config: ExperimentConfig | None = None,
        message_aggregator: ConcurrentMessageAggregator | None = None,
        state_updater: nn.Module | None = None,
        predictor: nn.Module | None = None,
        query_parser: QueryParser | None = None,
        llm_completion_fn=None,
    ) -> None:
        super().__init__()
        self.config = config or ExperimentConfig()
        self.config.validate()
        self.device = torch.device(self.config.device)
        self.message_encoder = MessageEncoder(
            context_dim=self.config.context_dim,
            hidden_dim=self.config.hidden_dim,
            sentence_transformer_path=self.config.sentence_transformer_path,
            sentence_transformer_dim=self.config.sentence_transformer_dim,
            sentence_transformer_freeze=self.config.sentence_transformer_freeze,
        )
        self.message_aggregator = message_aggregator or ConcurrentMessageAggregator(
            message_encoder=self.message_encoder,
            reduce=self.config.concurrent_update_mode,
            num_heads=self.config.aggregator_num_heads,
            dropout=self.config.aggregator_dropout,
        )
        self.state_updater = state_updater or build_state_updater(
            updater_type=self.config.state_updater_type,
            context_dim=self.config.context_dim,
            hidden_dim=self.config.hidden_dim,
            latent_state_count=self.config.latent_state_count,
            latent_action_count=self.config.latent_action_count,
        )
        self.node_feature_encoder = NodeFeatureEncoder(
            context_dim=self.config.context_dim,
            hidden_dim=self.config.hidden_dim,
            role_dim=self.config.role_dim,
            role_hash_buckets=self.config.role_hash_buckets,
            sentence_transformer_path=self.config.sentence_transformer_path,
            sentence_transformer_dim=self.config.sentence_transformer_dim,
            sentence_transformer_freeze=self.config.sentence_transformer_freeze,
        )
        if predictor is not None:
            self.predictor = predictor
        elif self.config.predictor_backend == "llm_api" or self.config.gnn_type == "llm_api":
            self.predictor = LLMApiGraphActionPredictor(
                config=self.config,
                completion_fn=llm_completion_fn,
            )
        else:
            self.predictor = GraphActionPredictor(
                config=self.config,
                node_feature_encoder=self.node_feature_encoder,
            )
        self.query_parser = query_parser or QueryParser(
            context_dim=self.config.context_dim,
            device=str(self.device),
        )
        self.active_prediction_context: GraphPredictionContext | None = None
        self.temporal_graph = TemporalGraph(
            context_dim=self.config.context_dim,
            device=self.device,
        )
        self.ctdg = ContinuousTimeDynamicGraph(
            temporal_graph=self.temporal_graph,
            message_aggregator=self.message_aggregator,
            state_updater=self.state_updater,
            hidden_dim=self.config.hidden_dim,
            device=self.device,
        )
        self.to(self.device)

    def initialize_graph(
        self,
        nodes: list[TemporalNode] | None = None,
        edges: list[TemporalEdge] | None = None,
        structural_edges: list[tuple[str, str]] | None = None,
        graph_context_text: str = "",
        structural_edge_metadata: dict[tuple[str, str], list[dict[str, str]]] | None = None,
    ) -> None:
        self.temporal_graph = TemporalGraph(
            context_dim=self.config.context_dim,
            device=self.device,
        )
        self.active_prediction_context = None
        self.temporal_graph.graph_context_text = str(graph_context_text or "")
        for node in nodes or []:
            self.temporal_graph.add_node(node)
        for edge in edges or []:
            self.temporal_graph.add_edge(edge)
        edge_metadata = structural_edge_metadata or {}
        for source_node_id, target_node_id in structural_edges or []:
            self.temporal_graph.add_structural_edge(source_node_id, target_node_id)
            metadata_items = edge_metadata.get((source_node_id, target_node_id), [])
            if metadata_items:
                self.temporal_graph.structural_edge_metadata[
                    (source_node_id, target_node_id)
                ] = [dict(item) for item in metadata_items]
        self.ctdg = ContinuousTimeDynamicGraph(
            temporal_graph=self.temporal_graph,
            message_aggregator=self.message_aggregator,
            state_updater=self.state_updater,
            hidden_dim=self.config.hidden_dim,
            device=self.device,
        )
        if hasattr(self.predictor, "initialize_ctdg_states"):
            self.predictor.initialize_ctdg_states(
                temporal_graph=self.temporal_graph,
                ctdg=self.ctdg,
                graph_context_text=self.temporal_graph.graph_context_text,
            )

    def initialize_from_query(
        self,
        query_text: str,
        nodes: list[TemporalNode] | None = None,
        edges: list[TemporalEdge] | None = None,
        structural_edges: list[tuple[str, str]] | None = None,
        inject_query_message: bool = True,
        query_time: float = 0.0,
    ) -> QueryParseResult:
        parse_result = self.query_parser.parse(query_text)
        merged_nodes = self._merge_nodes(nodes or [], parse_result.nodes)
        self.initialize_graph(
            nodes=merged_nodes,
            edges=edges,
            structural_edges=structural_edges,
            graph_context_text=query_text,
        )
        if inject_query_message and merged_nodes:
            self.ingest_messages(
                self.query_parser.build_query_messages(
                    query_text=query_text,
                    target_node_ids=[node.node_id for node in merged_nodes],
                    time_value=query_time,
                )
            )
        return parse_result

    def add_node(self, node_id: str, role: str, context: TensorLike = None) -> None:
        self.temporal_graph.add_node(
            TemporalNode.build(
                node_id=node_id,
                role=role,
                context=context,
                context_dim=self.config.context_dim,
                device=self.device,
            )
        )
        self.ctdg.add_node(node_id)

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        start_time: float,
    ) -> None:
        self.temporal_graph.add_edge(
            TemporalEdge(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                start_time=start_time,
            )
        )

    def update_node_context(
        self,
        node_id: str,
        context: TensorLike,
        text: str | None = None,
    ) -> None:
        self.temporal_graph.update_node_context(node_id, context, context_text=text)

    def ingest_messages(self, messages: list[Message]) -> None:
        self.ctdg.ingest_messages(messages)

    def predict_next_steps(
        self,
        observation_time: float,
        steps: int | None = None,
        prediction_context: GraphPredictionContext | None = None,
        prediction_context_schedule: list[GraphPredictionContext | None] | None = None,
    ) -> PredictionRollout:
        context = prediction_context if prediction_context is not None else self.active_prediction_context
        return self.predictor.predict_rollout(
            temporal_graph=self.temporal_graph,
            ctdg=self.ctdg,
            observation_time=observation_time,
            steps=steps,
            prediction_context_schedule=(
                prediction_context_schedule
                if prediction_context_schedule is not None
                else self._default_prediction_context_schedule(context, steps)
            ),
        )

    def predict_speculative_action_sets(
        self,
        observation_time: float,
        steps: int | None = None,
        prediction_context: GraphPredictionContext | None = None,
        prediction_context_schedule: list[GraphPredictionContext | None] | None = None,
    ) -> PredictionSubgraphRollout:
        context = prediction_context if prediction_context is not None else self.active_prediction_context
        return self.predictor.predict_subgraph_rollout(
            temporal_graph=self.temporal_graph,
            ctdg=self.ctdg,
            observation_time=observation_time,
            steps=steps,
            prediction_context_schedule=(
                prediction_context_schedule
                if prediction_context_schedule is not None
                else self._default_prediction_context_schedule(context, steps)
            ),
        )

    def process_query_runtime_update(
        self,
        *,
        observation_time: float,
        prediction_context: GraphPredictionContext | None = None,
        messages: list[Message] | None = None,
        context_updates: dict[str, TensorLike] | None = None,
        context_text_updates: dict[str, str] | None = None,
        observed_actions: list[PredictedGraphAction] | None = None,
        runtime_text: str = "",
        steps: int | None = None,
        apply_observed_actions: bool = True,
        update_memory: bool = True,
        update_state_from_actions: bool = False,
    ) -> PredictionSubgraphRollout:
        base_context = prediction_context if prediction_context is not None else self.active_prediction_context
        enriched_context = self._merge_prediction_context_runtime(
            prediction_context=base_context,
            runtime_text=runtime_text,
        )
        self.active_prediction_context = enriched_context
        text_updates = context_text_updates or {}
        for node_id, context in (context_updates or {}).items():
            self.update_node_context(
                node_id,
                context,
                text=text_updates.get(node_id),
            )
        if messages:
            for message in messages:
                message.metadata.setdefault("available_for_prediction", True)
            self.ingest_messages(messages)
        actions = observed_actions or []
        if update_memory:
            self.record_observed_actions(
                actions,
                prediction_context=enriched_context,
                extra_text=runtime_text,
            )
        if apply_observed_actions:
            for action in actions:
                self.predictor.apply_action(
                    action=action,
                    temporal_graph=self.temporal_graph,
                    ctdg=self.ctdg,
                    update_state=update_state_from_actions,
                )
        return self.predict_speculative_action_sets(
            observation_time=observation_time,
            steps=steps,
            prediction_context=enriched_context,
        )

    def clear_active_prediction_context(self) -> None:
        self.active_prediction_context = None

    def record_observed_actions(
        self,
        actions: list[PredictedGraphAction],
        prediction_context: GraphPredictionContext | None = None,
        extra_text: str = "",
    ) -> int:
        if not (
            self.config.use_few_shot_transition_memory
            and self.config.use_online_few_shot_updates
            and hasattr(self.predictor, "add_few_shot_transition")
        ):
            return 0
        added = 0
        for action in actions:
            if action.action_type != GraphActionType.CREATE_EDGE:
                continue
            if action.source_node_id not in self.temporal_graph.nodes:
                continue
            if action.target_node_id not in self.temporal_graph.nodes:
                continue
            source = self.temporal_graph.nodes[action.source_node_id]
            target = self.temporal_graph.nodes[action.target_node_id]
            self.predictor.add_few_shot_transition(
                source_role=source.role,
                target_role=target.role,
                relation_type=str(action.relation_type or ""),
                text=self._observed_action_memory_text(
                    action=action,
                    prediction_context=prediction_context,
                    extra_text=extra_text,
                ),
                source_node_id=action.source_node_id,
                target_node_id=action.target_node_id,
            )
            added += 1
        return added

    def _observed_action_memory_text(
        self,
        action: PredictedGraphAction,
        prediction_context: GraphPredictionContext | None,
        extra_text: str = "",
    ) -> str:
        source_text = self._node_memory_text(action.source_node_id)
        target_text = self._node_memory_text(action.target_node_id)
        metadata_text = " ".join(
            str(action.metadata.get(key, "")).strip()
            for key in ("transition_id", "description", "label", "name")
            if str(action.metadata.get(key, "")).strip()
        )
        context_text = prediction_context.combined_text() if prediction_context else ""
        return "\n".join(
            part
            for part in (
                self.temporal_graph.graph_context_text,
                context_text,
                str(action.relation_type or ""),
                metadata_text,
                source_text,
                target_text,
                extra_text,
            )
            if str(part).strip()
        )

    def _node_memory_text(self, node_id: str | None) -> str:
        if node_id is None or node_id not in self.temporal_graph.nodes:
            return ""
        node = self.temporal_graph.nodes[node_id]
        return f"{node.node_id} {node.role} {node.context_text}"

    def _default_prediction_context_schedule(
        self,
        prediction_context: GraphPredictionContext | None,
        steps: int | None,
    ) -> list[GraphPredictionContext | None] | None:
        if prediction_context is None:
            return None
        step_count = steps or self.config.prediction_horizon
        if self.config.reuse_current_context_for_speculative_rollout:
            return [prediction_context for _ in range(step_count)]
        return [prediction_context]

    def _merge_prediction_context_runtime(
        self,
        prediction_context: GraphPredictionContext | None,
        runtime_text: str,
    ) -> GraphPredictionContext | None:
        if prediction_context is None:
            if not runtime_text.strip():
                return None
            return GraphPredictionContext(runtime_text=runtime_text.strip())
        merged_runtime = self._merge_runtime_text(
            str(prediction_context.runtime_text or ""),
            runtime_text,
        )
        return GraphPredictionContext(
            source_node_id=prediction_context.source_node_id,
            query_text=prediction_context.query_text,
            graph_profile_text=prediction_context.graph_profile_text,
            source_output_text=prediction_context.source_output_text,
            runtime_text=merged_runtime,
            candidate_actions=[
                PredictedGraphAction(
                    action_type=action.action_type,
                    score=action.score,
                    effective_time=action.effective_time,
                    source_node_id=action.source_node_id,
                    target_node_id=action.target_node_id,
                    relation_type=action.relation_type,
                    role=action.role,
                    new_node_id=action.new_node_id,
                    metadata=dict(action.metadata),
                )
                for action in prediction_context.candidate_actions
            ],
            metadata=dict(prediction_context.metadata),
        )

    def _merge_runtime_text(self, *items: str) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for item in items:
            for line in str(item or "").splitlines():
                cleaned = line.strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                lines.append(cleaned)
        return "\n".join(lines)

    def _merge_nodes(
        self,
        user_nodes: list[TemporalNode],
        parsed_nodes: list[TemporalNode],
    ) -> list[TemporalNode]:
        merged: dict[str, TemporalNode] = {
            node.node_id: node
            for node in user_nodes
        }
        for node in parsed_nodes:
            merged.setdefault(node.node_id, node)
        return [merged[node_id] for node_id in sorted(merged)]
