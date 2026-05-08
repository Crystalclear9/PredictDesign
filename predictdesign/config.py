from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LLMApiConfig:
    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
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
    max_actions_per_step: int = 8
    temporal_edge_dim: int = 9
    allow_self_loop_prediction: bool = False
    concurrent_update_mode: str = "attention"
    aggregator_num_heads: int = 4
    aggregator_dropout: float = 0.1
    state_updater_type: str = "gru"
    gnn_type: str = "gcn"
    predictor_backend: str = "gnn"
    device: str = "cpu"
    # Relational Transformer settings
    rt_num_heads: int = 4
    rt_dropout: float = 0.1
    # SentenceTransformer settings
    sentence_transformer_path: str = "all-MiniLM-L6-v2"
    sentence_transformer_dim: int = 384
    sentence_transformer_freeze: bool = True
    # Cold start
    use_cold_start: bool = True
    # Strong predictor features
    use_context_conditioning: bool = True
    use_candidate_cross_encoder: bool = True
    use_structural_candidate_priors: bool = True
    context_source_bias_weight: float = 2.0
    candidate_text_score_weight: float = 0.35
    candidate_structural_prior_weight: float = 0.75
    candidate_action_type_boost: float = 1.5
    # Completion detection
    use_completion_detection: bool = True
    completion_threshold: float = 0.5
    # Training improvements
    use_focal_loss: bool = True
    focal_loss_gamma: float = 2.0
    gradient_clip_norm: float = 1.0
    warmup_fraction: float = 0.2
    training_node_feature_dropout: float = 0.15
    training_edge_feature_dropout: float = 0.10
    training_edge_mask_rate: float = 0.08
    training_node_mask_rate: float = 0.05
    training_message_dropout: float = 0.10
    min_training_epochs: int = 5
    early_stopping_patience: int = 8
    shuffle_train_episodes: bool = True
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
        if self.max_actions_per_step <= 0:
            raise ValueError("max_actions_per_step must be positive.")
        if self.temporal_edge_dim < 9:
            raise ValueError("temporal_edge_dim must be at least 9.")
        if self.concurrent_update_mode not in {"sum", "mean", "attention"}:
            raise ValueError("concurrent_update_mode must be 'sum', 'mean', or 'attention'.")
        if self.aggregator_num_heads <= 0:
            raise ValueError("aggregator_num_heads must be positive.")
        if self.aggregator_dropout < 0 or self.aggregator_dropout >= 1:
            raise ValueError("aggregator_dropout must be in [0, 1).")
        if self.state_updater_type not in {"gru", "mdp"}:
            raise ValueError("state_updater_type must be 'gru' or 'mdp'.")
        if self.gnn_type not in {"gcn", "graphsage", "gat", "relational_transformer", "hybrid", "llm_api"}:
            raise ValueError(
                "gnn_type must be one of: gcn, graphsage, gat, relational_transformer, hybrid, llm_api."
            )
        if self.predictor_backend not in {"gnn", "llm_api"}:
            raise ValueError("predictor_backend must be 'gnn' or 'llm_api'.")
        if not self.candidate_new_roles:
            raise ValueError("candidate_new_roles must not be empty.")
        if not self.candidate_relation_types:
            raise ValueError("candidate_relation_types must not be empty.")
        if self.rt_num_heads <= 0:
            raise ValueError("rt_num_heads must be positive.")
        if self.gnn_type in {"relational_transformer", "hybrid"} and self.hidden_dim % self.rt_num_heads != 0:
            raise ValueError("hidden_dim must be divisible by rt_num_heads for relational_transformer or hybrid.")
        if self.rt_dropout < 0 or self.rt_dropout >= 1:
            raise ValueError("rt_dropout must be in [0, 1).")
        if self.context_source_bias_weight < 0:
            raise ValueError("context_source_bias_weight must be non-negative.")
        if self.candidate_text_score_weight < 0:
            raise ValueError("candidate_text_score_weight must be non-negative.")
        if self.candidate_structural_prior_weight < 0:
            raise ValueError("candidate_structural_prior_weight must be non-negative.")
        if self.candidate_action_type_boost < 0:
            raise ValueError("candidate_action_type_boost must be non-negative.")
        if self.training_node_feature_dropout < 0 or self.training_node_feature_dropout >= 1:
            raise ValueError("training_node_feature_dropout must be in [0, 1).")
        if self.training_edge_feature_dropout < 0 or self.training_edge_feature_dropout >= 1:
            raise ValueError("training_edge_feature_dropout must be in [0, 1).")
        if self.training_edge_mask_rate < 0 or self.training_edge_mask_rate >= 1:
            raise ValueError("training_edge_mask_rate must be in [0, 1).")
        if self.training_node_mask_rate < 0 or self.training_node_mask_rate >= 1:
            raise ValueError("training_node_mask_rate must be in [0, 1).")
        if self.training_message_dropout < 0 or self.training_message_dropout >= 1:
            raise ValueError("training_message_dropout must be in [0, 1).")
        if self.min_training_epochs <= 0:
            raise ValueError("min_training_epochs must be positive.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
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
