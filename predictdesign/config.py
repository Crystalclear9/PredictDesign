from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LLMApiConfig:
    api_key: str = "sk-atbmbuobehrzbiagahmshyrdksemgfkztnljehdnojlakwkq"
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/Qwen3-32B"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 600.0
    max_retries: int = 5
    retry_backoff_seconds: float = 5.0
    system_prompt: str = (
        "You are a temporal graph prediction engine. "
        "Given the current collaboration graph, node roles, node output texts, node states, "
        "and recent interaction records, predict the next-step or next-few-step graph actions. "
        "Return JSON only."
    )


@dataclass(slots=True)
class ExperimentConfig:
    context_dim: int = 16
    hidden_dim: int = 32
    role_dim: int = 16
    role_hash_buckets: int = 257
    latent_state_count: int = 8
    latent_action_count: int = 4
    gnn_layers: int = 2
    prediction_horizon: int = 3
    prediction_edge_duration: float = 1.0
    max_actions_per_step: int = 8
    temporal_edge_dim: int = 9
    concurrent_update_mode: str = "sum"
    state_updater_type: str = "gru"
    gnn_type: str = "gcn"
    predictor_backend: str = "gnn"
    device: str = "cpu"
    candidate_new_roles: tuple[str, ...] = field(
        default_factory=lambda: ("planner", "solver", "critic", "tool")
    )
    candidate_relation_types: tuple[str, ...] = field(
        default_factory=lambda: (
            "communication",
            "delegation",
            "banishment_vote",
            "werewolf_vote",
            "werewolf_attack",
            "guard_action",
            "seer_check",
            "witch_save",
            "witch_poison",
        )
    )
    llm_api: LLMApiConfig = field(default_factory=LLMApiConfig)

    def validate(self) -> None:
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.role_dim <= 0:
            raise ValueError("role_dim must be positive.")
        if self.latent_state_count <= 1:
            raise ValueError("latent_state_count must be greater than one.")
        if self.latent_action_count <= 1:
            raise ValueError("latent_action_count must be greater than one.")
        if self.gnn_layers <= 0:
            raise ValueError("gnn_layers must be positive.")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive.")
        if self.prediction_edge_duration <= 0:
            raise ValueError("prediction_edge_duration must be positive.")
        if self.max_actions_per_step <= 0:
            raise ValueError("max_actions_per_step must be positive.")
        if self.temporal_edge_dim < 9:
            raise ValueError("temporal_edge_dim must be at least 9.")
        if self.concurrent_update_mode not in {"sum", "mean"}:
            raise ValueError("concurrent_update_mode must be 'sum' or 'mean'.")
        if self.state_updater_type not in {"gru", "mdp"}:
            raise ValueError("state_updater_type must be 'gru' or 'mdp'.")
        if self.gnn_type not in {"gcn", "graphsage", "gat", "llm_api"}:
            raise ValueError("gnn_type must be one of: gcn, graphsage, gat, llm_api.")
        if self.predictor_backend not in {"gnn", "llm_api"}:
            raise ValueError("predictor_backend must be 'gnn' or 'llm_api'.")
        if not self.candidate_new_roles:
            raise ValueError("candidate_new_roles must not be empty.")
        if not self.candidate_relation_types:
            raise ValueError("candidate_relation_types must not be empty.")
        if not self.llm_api.base_url:
            raise ValueError("llm_api.base_url must not be empty.")
        if not self.llm_api.model:
            raise ValueError("llm_api.model must not be empty.")
        if self.llm_api.max_tokens <= 0:
            raise ValueError("llm_api.max_tokens must be positive.")
        if self.llm_api.timeout <= 0:
            raise ValueError("llm_api.timeout must be positive.")
        if self.llm_api.max_retries < 0:
            raise ValueError("llm_api.max_retries must be non-negative.")
        if self.llm_api.retry_backoff_seconds < 0:
            raise ValueError("llm_api.retry_backoff_seconds must be non-negative.")
