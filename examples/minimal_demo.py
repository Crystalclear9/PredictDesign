"""Minimal offline demo for the current hybrid predictor.

Run:
    python examples/minimal_demo.py

This example is intentionally small and does not require external model
downloads. The ``__fallback_sentence_transformer__`` sentinel activates the
local hash text encoder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign import (
    ExperimentConfig,
    GraphActionType,
    GraphPredictionContext,
    PredictDesignSystem,
    PredictedGraphAction,
    TemporalNode,
)
from predictdesign.messages import Message


FALLBACK_ST_MODEL = "__fallback_sentence_transformer__"


def describe_action(prefix: str, action: PredictedGraphAction) -> None:
    print(
        f"{prefix}: type={action.action_type.value} score={action.score:.4f} "
        f"source={action.source_node_id} target={action.target_node_id} "
        f"relation={action.relation_type} role={action.role}"
    )


def main() -> None:
    torch.manual_seed(7)
    config = ExperimentConfig(
        context_dim=8,
        hidden_dim=16,
        gnn_type="hybrid",
        gnn_layers=1,
        rt_num_heads=4,
        prediction_horizon=2,
        candidate_new_roles=("planner", "coder", "reviewer"),
        candidate_relation_types=("activate", "delegate", "review", "retry"),
        sentence_transformer_path=FALLBACK_ST_MODEL,
    )
    system = PredictDesignSystem(config=config)
    system.initialize_graph(
        nodes=[
            TemporalNode.build("planner", "planner", [1, 0, 0, 0, 0, 0, 0, 0], 8, "cpu"),
            TemporalNode.build("coder", "coder", [0, 1, 0, 0, 0, 0, 0, 0], 8, "cpu"),
            TemporalNode.build("reviewer", "reviewer", [0, 0, 1, 0, 0, 0, 0, 0], 8, "cpu"),
        ],
        structural_edges=[("planner", "coder"), ("coder", "reviewer")],
        graph_context_text="Build a feature, implement it, then review it.",
        structural_edge_metadata={
            ("planner", "coder"): [
                {
                    "relation_type": "activate",
                    "transition_id": "plan_to_code",
                    "description": "planner activates coder for implementation",
                }
            ],
            ("coder", "reviewer"): [
                {
                    "relation_type": "review",
                    "transition_id": "code_to_review",
                    "description": "coder asks reviewer to inspect the patch",
                }
            ],
        },
    )
    system.update_node_context("planner", [1, 0, 0, 0, 0, 0, 0, 0], text="Plan is ready.")
    system.update_node_context("coder", [0, 1, 0, 0, 0, 0, 0, 0], text="Implementation draft is ready.")
    message = Message.build_completion_message(
        time=1.0,
        source_node_id="planner",
        target_node_id="coder",
        context=[0.8, 0.2, 0, 0, 0, 0, 0, 0],
        hidden_dim=16,
        context_dim=8,
        device="cpu",
    )
    message.metadata["raw_text"] = "Please implement the planned change."
    message.metadata["relation_type"] = "activate"
    system.ingest_messages([message])

    prediction_context = GraphPredictionContext(
        source_node_id="coder",
        query_text="The coder finished a draft. Pick the next workflow transition.",
        graph_profile_text=system.temporal_graph.graph_context_text,
        source_output_text="Implementation draft is ready.",
        candidate_actions=[
            PredictedGraphAction(
                action_type=GraphActionType.CREATE_EDGE,
                score=0.0,
                effective_time=2.0,
                source_node_id="coder",
                target_node_id="reviewer",
                relation_type="review",
                metadata={
                    "transition_id": "code_to_review",
                    "description": "coder asks reviewer to inspect the patch",
                },
            ),
            PredictedGraphAction(
                action_type=GraphActionType.CREATE_EDGE,
                score=0.0,
                effective_time=2.0,
                source_node_id="coder",
                target_node_id="coder",
                relation_type="retry",
                metadata={"transition_id": "retry_code", "description": "coder retries implementation"},
            ),
        ],
    )

    actions = system.predictor.predict_action_set(
        temporal_graph=system.temporal_graph,
        ctdg=system.ctdg,
        observation_time=2.0,
        prediction_context=prediction_context,
    )
    print("Hybrid candidate-aware prediction")
    for index, action in enumerate(actions, start=1):
        describe_action(f"action[{index}]", action)


if __name__ == "__main__":
    main()
