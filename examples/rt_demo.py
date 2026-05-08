"""Compare Relational Transformer and Hybrid GNN on one toy episode.

Run:
    python examples/rt_demo.py

The script uses the local fallback text encoder, so it is safe to run offline.
It also performs one tiny supervised update to show the training API shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign import (
    BenchmarkEpisode,
    BenchmarkTrainer,
    EpisodeStep,
    ExperimentConfig,
    GraphActionType,
    GraphPredictionContext,
    PredictDesignSystem,
    PredictedGraphAction,
    TemporalNode,
)
from predictdesign.messages import Message


FALLBACK_ST_MODEL = "__fallback_sentence_transformer__"


def make_config(gnn_type: str) -> ExperimentConfig:
    return ExperimentConfig(
        context_dim=8,
        hidden_dim=16,
        gnn_type=gnn_type,
        gnn_layers=1,
        rt_num_heads=4,
        prediction_horizon=1,
        candidate_new_roles=("planner", "coder", "reviewer"),
        candidate_relation_types=("activate", "review", "retry"),
        sentence_transformer_path=FALLBACK_ST_MODEL,
        allow_self_loop_prediction=False,
    )


def make_episode() -> BenchmarkEpisode:
    review_action = PredictedGraphAction(
        action_type=GraphActionType.CREATE_EDGE,
        score=1.0,
        effective_time=2.0,
        source_node_id="coder",
        target_node_id="reviewer",
        relation_type="review",
        metadata={
            "transition_id": "code_to_review",
            "description": "coder sends implementation to reviewer",
        },
    )
    return BenchmarkEpisode(
        episode_id="toy-review",
        dataset_name="examples",
        initial_nodes=[
            TemporalNode.build("planner", "planner", [1, 0, 0, 0, 0, 0, 0, 0], 8, "cpu"),
            TemporalNode.build("coder", "coder", [0, 1, 0, 0, 0, 0, 0, 0], 8, "cpu"),
            TemporalNode.build("reviewer", "reviewer", [0, 0, 1, 0, 0, 0, 0, 0], 8, "cpu"),
        ],
        initial_edges=[],
        initial_structural_edges=[("planner", "coder"), ("coder", "reviewer")],
        initial_graph_context_text="Plan, implement, review.",
        initial_structural_edge_metadata={
            ("coder", "reviewer"): [
                {
                    "relation_type": "review",
                    "transition_id": "code_to_review",
                    "description": "coder sends implementation to reviewer",
                }
            ]
        },
        steps=[
            EpisodeStep(
                observation_time=1.0,
                messages=[
                    Message.build_completion_message(
                        time=1.0,
                        source_node_id="planner",
                        target_node_id="coder",
                        context=[0.8, 0.2, 0, 0, 0, 0, 0, 0],
                        hidden_dim=16,
                        context_dim=8,
                        device="cpu",
                    )
                ],
                ground_truth_action=PredictedGraphAction(
                    action_type=GraphActionType.NO_OP,
                    score=1.0,
                    effective_time=1.0,
                ),
                observed_actions=[],
                context_text_updates={"coder": "Implementation draft is ready."},
                context_updates={"coder": [0, 1, 0, 0, 0, 0, 0, 0]},
            ),
            EpisodeStep(
                observation_time=2.0,
                messages=[],
                ground_truth_action=PredictedGraphAction(
                    action_type=GraphActionType.NO_OP,
                    score=1.0,
                    effective_time=2.0,
                ),
                observed_actions=[],
                valid_next_actions=[review_action],
                prediction_context=GraphPredictionContext(
                    source_node_id="coder",
                    query_text="Implementation is done. Choose the next edge.",
                    graph_profile_text="Plan, implement, review.",
                    source_output_text="Implementation draft is ready.",
                    candidate_actions=[
                        review_action,
                        PredictedGraphAction(
                            action_type=GraphActionType.CREATE_EDGE,
                            score=0.0,
                            effective_time=2.0,
                            source_node_id="coder",
                            target_node_id="coder",
                            relation_type="retry",
                            metadata={"transition_id": "retry_code", "description": "coder retries"},
                        ),
                    ],
                ),
            )
        ],
    )


def predict_once(system: PredictDesignSystem, episode: BenchmarkEpisode) -> PredictedGraphAction:
    step = episode.steps[0]
    target_step = episode.steps[1]
    system.initialize_graph(
        nodes=episode.initial_nodes,
        edges=episode.initial_edges,
        structural_edges=episode.initial_structural_edges,
        graph_context_text=episode.initial_graph_context_text,
        structural_edge_metadata=episode.initial_structural_edge_metadata,
    )
    for node_id, context in step.context_updates.items():
        system.update_node_context(node_id, context, text=step.context_text_updates.get(node_id))
    system.ingest_messages(step.messages)
    return system.predictor.predict_next_action(
        temporal_graph=system.temporal_graph,
        ctdg=system.ctdg,
        observation_time=2.0,
        prediction_context=target_step.prediction_context,
    )


def main() -> None:
    torch.manual_seed(11)
    episode = make_episode()
    for gnn_type in ("relational_transformer", "hybrid"):
        system = PredictDesignSystem(config=make_config(gnn_type))
        before = predict_once(system, episode)
        print(
            f"{gnn_type}: before training -> {before.action_type.value} "
            f"{before.source_node_id}->{before.target_node_id} relation={before.relation_type}"
        )
        trainer = BenchmarkTrainer(epochs=1, learning_rate=1e-3)
        trainer.fit(system, [episode])
        after = predict_once(system, episode)
        print(
            f"{gnn_type}: after one tiny update -> {after.action_type.value} "
            f"{after.source_node_id}->{after.target_node_id} relation={after.relation_type}"
        )


if __name__ == "__main__":
    main()
