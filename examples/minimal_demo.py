from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign import ExperimentConfig, PredictDesignSystem, TemporalEdge, TemporalNode
from predictdesign.messages import Message


def main() -> None:
    torch.manual_seed(7)
    config = ExperimentConfig(
        context_dim=8,
        hidden_dim=16,
        concurrent_update_mode="mean",
        state_updater_type="gru",
        gnn_type="graphsage",
        prediction_horizon=3,
        candidate_new_roles=("planner", "coder", "reviewer"),
    )
    system = PredictDesignSystem(config=config)
    system.initialize_graph(
        nodes=[
            TemporalNode.build("user_proxy", "planner", [1, 0, 0, 0, 1, 0, 0, 0], 8, "cpu"),
            TemporalNode.build("engineer", "coder", [0, 1, 0, 0, 0, 1, 0, 0], 8, "cpu"),
            TemporalNode.build("critic", "reviewer", [0, 0, 1, 0, 0, 0, 1, 0], 8, "cpu"),
        ],
        edges=[
            TemporalEdge("user_proxy", "engineer", 0.0, 5.0),
            TemporalEdge("engineer", "critic", 1.0, 5.0),
        ],
    )
    system.ingest_messages(
        [
            Message.build_query_message(
                target_node_id="user_proxy",
                time=0.5,
                context=[0.2] * 8,
                context_dim=8,
                device="cpu",
            ),
            Message.build_completion_message(
                time=2.0,
                source_node_id="engineer",
                target_node_id="critic",
                context=[0.3] * 8,
                hidden_dim=16,
                context_dim=8,
                device="cpu",
            ),
            Message.build_completion_message(
                time=2.0,
                source_node_id="critic",
                target_node_id="user_proxy",
                context=[0.1] * 8,
                hidden_dim=16,
                context_dim=8,
                device="cpu",
            ),
        ]
    )
    rollout = system.predict_next_steps(observation_time=2.0, steps=3)
    for step, action in enumerate(rollout.actions, start=1):
        print(
            f"step={step} action={action.action_type.value} score={action.score:.4f} "
            f"source={action.source_node_id} target={action.target_node_id} "
            f"role={action.role} new_node={action.new_node_id}"
        )


if __name__ == "__main__":
    main()
