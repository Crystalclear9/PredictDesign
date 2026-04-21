# PredictDesign

> **v0.2.0** — 多智能体协作时序图动作预测框架，现已集成 [Relational Transformer (RT, ICLR 2026)](https://openreview.net/forum?id=rpPtgMC5s9) 架构、SentenceTransformer 文本编码、冷启动初始化和节点完成检测。

---

## 核心能力

| 模块 | 描述 |
|------|------|
| **Temporal Graph** | 带起止时间的时序边 + 角色/上下文节点 |
| **CTDG** | 每节点维护历史状态，支持并发消息聚合（sum/mean）|
| **State Updater** | `GRU` / `MDP` 两种可切换的状态更新器 |
| **Relational Transformer** ⭐ | 四种注意力掩码（role/neighbor/full/feature）+ GatedMLP + RMSNorm |
| **SentenceTransformer 编码** ⭐ | 冻结 MiniLMv2 + 可训练投影层替代 hash 编码 |
| **Cold Start** ⭐ | 空图时的角色原型 + 任务嵌入 + 初始边权先验 |
| **Completion Detection** ⭐ | 每节点轻量二分类器，判断 agent 输出是否结束 |
| **Focal Loss + Warmup** ⭐ | 训练时支持焦点损失、梯度裁剪、线性 warmup |
| **LLM API Predictor** | 可用 LLM API 替换 GNN 做预测 |

⭐ v0.2.0 新增

---

## 目录结构

```text
PredictDesign/
├── predictdesign/
│   ├── __init__.py               # 公开 API
│   ├── config.py                 # ExperimentConfig（所有超参）
│   ├── experiment.py             # PredictDesignSystem（顶层门面）
│   ├── temporal_graph.py         # TemporalGraph / TemporalNode / TemporalEdge
│   ├── ctdg.py                   # ContinuousTimeDynamicGraph
│   ├── encoders.py               # SentenceTransformerEncoder / NodeFeatureEncoder
│   ├── completion.py             # NodeCompletionClassifier ⭐
│   ├── aggregation.py            # ConcurrentMessageAggregator
│   ├── messages.py               # Message / MessageAction
│   ├── prediction.py             # PredictedGraphAction / PredictionRollout
│   ├── query_parser.py           # QueryParser（自然语言→初始图）
│   ├── types.py
│   ├── gnn/
│   │   ├── layers.py             # RelationalAttentionLayer / GNNBackbone ⭐
│   │   ├── cold_start.py         # ColdStartInitializer ⭐
│   │   └── predictor.py          # GraphActionPredictor
│   ├── state_update/             # GRUStateUpdater / MDPStateUpdater
│   ├── benchmark/                # BenchmarkTrainer / BenchmarkEvaluator
│   └── llm/                      # LLMApiGraphActionPredictor
├── examples/
│   ├── minimal_demo.py           # 基础 GNN 用法
│   └── rt_demo.py                # RT + 冷启动 + 完成检测示例 ⭐
├── scripts/                      # 评测脚本（MultiAgentBench）
├── results/                      # 运行结果
├── tests/
│   └── test_predictdesign.py
└── pyproject.toml
```

---

## 快速安装

```bash
pip install -e .
# 首次运行会自动下载 all-MiniLM-L6-v2（约 90MB）
```

依赖：`torch>=2.0`，`sentence-transformers>=2.2.0`

---

## 快速上手

### 最简配置（向下兼容）

```python
from predictdesign import ExperimentConfig, PredictDesignSystem

system = PredictDesignSystem(config=ExperimentConfig(
    gnn_type="graphsage",
    candidate_new_roles=("planner", "coder", "reviewer"),
))
rollout = system.predict_next_steps(observation_time=1.0, steps=3)
```

### RT + 冷启动（推荐）

```python
from predictdesign import ExperimentConfig, PredictDesignSystem

config = ExperimentConfig(
    gnn_type="relational_transformer",   # RT backbone
    rt_num_heads=4,
    use_cold_start=True,                 # 空图冷启动
    use_completion_detection=True,       # 节点完成检测
    use_focal_loss=True,                 # 训练用焦点损失
    gradient_clip_norm=1.0,
    candidate_new_roles=("planner", "coder", "reviewer", "tool"),
)
system = PredictDesignSystem(config=config)

# 冷启动：空图时仍可预测 ADD_NODE
rollout = system.predict_next_steps(observation_time=0.0, steps=1)
print(rollout.actions[0].action_type)  # GraphActionType.ADD_NODE
```

完整可运行示例见 [`examples/rt_demo.py`](examples/rt_demo.py)。

---

## 模型架构（v0.2.0）

```
文本/上下文
    ↓ SentenceTransformerEncoder（冻结 MiniLMv2 + 可训练 Linear）
节点特征 ──────────────┐
角色嵌入（hash）       │
时间编码               │
上下文维度             ↓
              NodeFeatureEncoder → [N, D]
                        ↓
          RelationalAttentionLayer × L
          ├── Role Attention     （同角色节点）
          ├── Neighbor Attention  （有边相连的节点）
          ├── Full Attention      （全节点）
          └── Feature Attention   （cosine top-k neighbour）
          + GatedMLP (SiLU) + RMSNorm (pre-norm)
                        ↓
            Attention Pooling → graph_embedding [D]
            ├── add_node_head      → 角色分布
            ├── action_count_head → 动作数量
            └── no_op_head
                        ↓
      NodeCompletionClassifier → completion_scores [N] ∈ [0,1]
      （影响 CREATE_EDGE 的 source 评分）
```

### 冷启动路径

空图（无节点）时：
1. `ColdStartInitializer.graph_embedding_cold()` 返回任务感知的图级嵌入，不再退化到零向量
2. 预测头仍能正常预测 `ADD_NODE`
3. ADD_NODE 执行后，新节点 CTDG 状态由角色原型初始化（非零）

---

## 配置参数速查

```python
ExperimentConfig(
    # --- 架构 ---
    gnn_type = "relational_transformer",  # "gcn" | "graphsage" | "gat" | "relational_transformer"
    rt_num_heads = 4,
    rt_num_layers = 2,                    # 同 gnn_layers
    rt_dropout = 0.1,

    # --- SentenceTransformer ---
    sentence_transformer_path = "all-MiniLM-L6-v2",  # 或本地路径
    sentence_transformer_dim = 384,
    sentence_transformer_freeze = True,

    # --- 冷启动 ---
    use_cold_start = True,

    # --- 完成检测 ---
    use_completion_detection = True,

    # --- 训练 ---
    use_focal_loss = True,
    focal_loss_gamma = 2.0,
    gradient_clip_norm = 1.0,
    warmup_fraction = 0.1,

    # --- 状态更新 ---
    state_updater_type = "gru",   # "gru" | "mdp"
    concurrent_update_mode = "mean",  # "sum" | "mean"
)
```

---

## 运行测试

```bash
# 直接用 Python（规避 hydra/antlr 版本冲突）
python -c "import predictdesign; print('OK')"
python examples/rt_demo.py
python examples/minimal_demo.py
```

---

## 后续扩展建议

- **接入自定义 ST 模型**：将 `sentence_transformer_path` 改为本地路径
- **替换 GNN**：只需在 `gnn/layers.py` 里新增一个 `nn.Module` 并在 `GNNBackbone` 注册
- **接入 LLM**：设置 `gnn_type="llm_api"` 并传入 `llm_completion_fn`
- **做消融实验**：通过 `ExperimentConfig` 字段独立开关每个模块
