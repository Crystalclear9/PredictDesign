"""RT Demo — 演示 Relational Transformer + 冷启动 + 完成检测 + 训练一步。

运行方式:
    python examples/rt_demo.py
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
    PredictDesignSystem,
    TemporalEdge,
    TemporalNode,
)
from predictdesign.messages import Message


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        context_dim=16,
        hidden_dim=32,
        gnn_type="relational_transformer",
        rt_num_heads=4,
        gnn_layers=2,
        use_cold_start=True,
        use_completion_detection=True,
        use_focal_loss=True,
        focal_loss_gamma=2.0,
        gradient_clip_norm=1.0,
        warmup_fraction=0.1,
        candidate_new_roles=("planner", "coder", "reviewer", "tool"),
        prediction_horizon=3,
    )


def demo_cold_start(system: PredictDesignSystem) -> None:
    """1. 空图冷启动：无节点时仍能预测。"""
    print("\n=== 1. Cold-Start (empty graph) ===")
    system.initialize_graph()
    rollout = system.predict_next_steps(observation_time=0.0, steps=1)
    action = rollout.actions[0]
    print(f"  action={action.action_type.value}  score={action.score:.4f}  role={action.role}")


def demo_with_nodes(system: PredictDesignSystem) -> None:
    """2. 有节点的 RT 推理 + 完成检测。"""
    print("\n=== 2. RT Inference with nodes ===")
    system.initialize_graph(
        nodes=[
            TemporalNode.build("planner_0", "planner", context=None, context_dim=16, device="cpu"),
            TemporalNode.build("coder_0", "coder", context=None, context_dim=16, device="cpu"),
            TemporalNode.build("reviewer_0", "reviewer", context=None, context_dim=16, device="cpu"),
        ],
        edges=[
            TemporalEdge("planner_0", "coder_0", 0.0, 5.0),
            TemporalEdge("coder_0", "reviewer_0", 1.0, 5.0),
        ],
    )
    system.ingest_messages([
        Message.build_query_message(
            target_node_id="planner_0",
            time=0.5,
            context=None,
            context_dim=16,
            device="cpu",
        ),
        Message.build_completion_message(
            time=2.0,
            source_node_id="coder_0",
            target_node_id="reviewer_0",
            context=None,
            hidden_dim=32,
            context_dim=16,
            device="cpu",
        ),
    ])
    rollout = system.predict_next_steps(observation_time=2.0, steps=3)
    for i, action in enumerate(rollout.actions, 1):
        print(
            f"  step={i}  action={action.action_type.value}  score={action.score:.4f}"
            f"  src={action.source_node_id}  tgt={action.target_node_id}  role={action.role}"
        )


def demo_train_step(system: PredictDesignSystem) -> None:
    """3. 单步梯度训练演示。"""
    print("\n=== 3. One Training Step ===")
    from predictdesign.benchmark.types import BenchmarkEpisode, EpisodeStep
    from predictdesign.benchmark.trainer import BenchmarkTrainer
    from predictdesign.prediction import PredictedGraphAction, GraphActionType

    episode = BenchmarkEpisode(
        initial_nodes=[
            TemporalNode.build("a", "planner", context=None, context_dim=16, device="cpu"),
            TemporalNode.build("b", "coder", context=None, context_dim=16, device="cpu"),
        ],
        initial_edges=[],
        initial_structural_edges=[],
        steps=[
            EpisodeStep(
                observation_time=1.0,
                messages=[],
                ground_truth_action=PredictedGraphAction(
                    action_type=GraphActionType.CREATE_EDGE,
                    score=1.0,
                    effective_time=1.0,
                    source_node_id="a",
                    target_node_id="b",
                ),
                observed_actions=[],
                valid_next_actions=[PredictedGraphAction(
                    action_type=GraphActionType.CREATE_EDGE,
                    score=1.0,
                    effective_time=1.0,
                    source_node_id="a",
                    target_node_id="b",
                )],
                context_updates={},
                context_text_updates={},
            ),
        ],
    )
    trainer = BenchmarkTrainer(epochs=2, learning_rate=1e-3)
    before = sum(p.sum().item() for p in system.parameters())
    trainer.fit(system, [episode])
    after = sum(p.sum().item() for p in system.parameters())
    print(f"  param sum before={before:.4f}  after={after:.4f}  (should differ → gradients flowed)")


def main() -> None:
    torch.manual_seed(42)
    print("Building PredictDesignSystem with Relational Transformer...")
    config = make_config()
    system = PredictDesignSystem(config=config)
    print(f"  backbone: {config.gnn_type}, heads={config.rt_num_heads}, layers={config.gnn_layers}")
    print(f"  cold_start={config.use_cold_start}, completion={config.use_completion_detection}")

    demo_cold_start(system)
    demo_with_nodes(system)
    demo_train_step(system)

    print("\nAll demos completed successfully.")


if __name__ == "__main__":
    main()
