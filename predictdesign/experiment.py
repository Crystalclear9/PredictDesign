from __future__ import annotations

import torch
from torch import nn

from .aggregation import ConcurrentMessageAggregator
from .config import ExperimentConfig
from .ctdg import ContinuousTimeDynamicGraph
from .encoders import MessageEncoder, NodeFeatureEncoder
from .gnn import GraphActionPredictor
from .messages import Message
from .prediction import PredictionRollout
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
    ) -> None:
        self.temporal_graph = TemporalGraph(
            context_dim=self.config.context_dim,
            device=self.device,
        )
        for node in nodes or []:
            self.temporal_graph.add_node(node)
        for edge in edges or []:
            self.temporal_graph.add_edge(edge)
        for source_node_id, target_node_id in structural_edges or []:
            self.temporal_graph.add_structural_edge(source_node_id, target_node_id)
        self.ctdg = ContinuousTimeDynamicGraph(
            temporal_graph=self.temporal_graph,
            message_aggregator=self.message_aggregator,
            state_updater=self.state_updater,
            hidden_dim=self.config.hidden_dim,
            device=self.device,
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
        end_time: float,
    ) -> None:
        self.temporal_graph.add_edge(
            TemporalEdge(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                start_time=start_time,
                end_time=end_time,
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
    ) -> PredictionRollout:
        return self.predictor.predict_rollout(
            temporal_graph=self.temporal_graph,
            ctdg=self.ctdg,
            observation_time=observation_time,
            steps=steps,
        )

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
