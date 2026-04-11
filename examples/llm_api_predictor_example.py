from predictdesign import ExperimentConfig, LLMApiConfig, PredictDesignSystem, TemporalNode


def build_system_with_llm_api() -> PredictDesignSystem:
    config = ExperimentConfig(
        context_dim=64,
        hidden_dim=64,
        gnn_type="llm_api",
        predictor_backend="llm_api",
        llm_api=LLMApiConfig(
            api_key="to_fill_in",
            base_url="https://api.siliconflow.cn/v1",
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            max_tokens=1024,
            timeout=120.0,
        ),
    )
    system = PredictDesignSystem(config=config)
    system.initialize_graph(
        nodes=[
            TemporalNode.build("agent1", "planner", [0.0] * 64, 64, "cpu"),
            TemporalNode.build("agent2", "solver", [0.0] * 64, 64, "cpu"),
        ]
    )
    return system


if __name__ == "__main__":
    system = build_system_with_llm_api()
    rollout = system.predict_next_steps(observation_time=1.0, steps=1)
    print(rollout.actions)
