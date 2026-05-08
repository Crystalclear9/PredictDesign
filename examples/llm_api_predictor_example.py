"""Offline-safe LLM API predictor example.

Run with a fake completion function:
    python examples/llm_api_predictor_example.py

To use a real OpenAI-compatible endpoint, set these environment variables and
pass ``--real``:
    PREDICTDESIGN_LLM_API_KEY
    PREDICTDESIGN_LLM_BASE_URL
    PREDICTDESIGN_LLM_MODEL
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign import (
    ExperimentConfig,
    GraphActionType,
    GraphPredictionContext,
    LLMApiConfig,
    PredictDesignSystem,
    PredictedGraphAction,
    TemporalNode,
)


def fake_completion(system_prompt: str, user_prompt: str, config: ExperimentConfig) -> str:
    print("Fake LLM received prompt characters:", len(user_prompt))
    return """
    {
      "predicted_count": 1,
      "actions": [
        {
          "action_type": "create_edge",
          "source_node_id": "planner",
          "target_node_id": "solver",
          "relation_type": "delegate",
          "score": 0.92
        }
      ]
    }
    """


def build_system(use_real_api: bool) -> PredictDesignSystem:
    config = ExperimentConfig(
        context_dim=8,
        hidden_dim=16,
        gnn_type="llm_api",
        predictor_backend="llm_api",
        candidate_new_roles=("planner", "solver", "critic"),
        candidate_relation_types=("delegate", "review", "retry"),
        llm_api=LLMApiConfig(
            api_key=os.getenv("PREDICTDESIGN_LLM_API_KEY", ""),
            base_url=os.getenv("PREDICTDESIGN_LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
            model=os.getenv("PREDICTDESIGN_LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"),
            temperature=0.1,
            max_tokens=1024,
            timeout=120.0,
        ),
    )
    completion_fn = None if use_real_api else fake_completion
    system = PredictDesignSystem(config=config, llm_completion_fn=completion_fn)
    system.initialize_graph(
        nodes=[
            TemporalNode.build("planner", "planner", [1, 0, 0, 0, 0, 0, 0, 0], 8, "cpu"),
            TemporalNode.build("solver", "solver", [0, 1, 0, 0, 0, 0, 0, 0], 8, "cpu"),
        ],
        graph_context_text="Planner delegates work to solver, then solver returns progress.",
    )
    return system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Call the configured real LLM API.")
    args = parser.parse_args()

    system = build_system(use_real_api=args.real)
    prediction_context = GraphPredictionContext(
        source_node_id="planner",
        query_text="Pick the next collaboration graph action.",
        graph_profile_text=system.temporal_graph.graph_context_text,
        candidate_actions=[
            PredictedGraphAction(
                action_type=GraphActionType.CREATE_EDGE,
                score=0.0,
                effective_time=1.0,
                source_node_id="planner",
                target_node_id="solver",
                relation_type="delegate",
                metadata={"description": "planner delegates implementation to solver"},
            )
        ],
    )
    action = system.predictor.predict_next_action(
        temporal_graph=system.temporal_graph,
        ctdg=system.ctdg,
        observation_time=1.0,
        prediction_context=prediction_context,
    )
    print(
        f"action={action.action_type.value} source={action.source_node_id} "
        f"target={action.target_node_id} relation={action.relation_type} score={action.score:.3f}"
    )


if __name__ == "__main__":
    main()
