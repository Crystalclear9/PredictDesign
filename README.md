# PredictDesign

PredictDesign 是一个面向多智能体协作过程的时序图预测实验框架。它把一次多智能体任务执行过程建模为连续时间动态图，学习在某个观察时刻之后应该新增、删除哪些协作边，或是否需要新增节点。

当前代码重点支持三类工作：

- 图预测模型：TemporalGraph、CTDG 状态更新、Hybrid GNN / Relational Transformer、冷启动、候选动作重排、节点完成检测。
- benchmark 适配：ACG-NAP、MultiAgentBench / MARBLE、本地 rich log。
- 训练与评估：GNN holdout / candidate rerank、rich log MLP、LLM API predictor、批量 benchmark runner。

当前版本：`0.2.0`

## 快速运行

### 1. 安装

要求 Python `>=3.11`。

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

最小导入验证：

```powershell
python -c "import predictdesign; print('OK')"
```

### 2. 跑离线示例

这些示例默认使用本地 hash text encoder，不会下载 SentenceTransformer 模型。

```powershell
python examples\minimal_demo.py
python examples\rt_demo.py
python examples\llm_api_predictor_example.py
```

示例说明已经集中在本文档的“运行示例”部分。

### 3. 跑测试

```powershell
python tests\test_predictdesign.py
```

可选：

```powershell
pytest
```

### 4. 清理缓存

预览：

```powershell
python scripts\cleanup_workspace.py
```

执行：

```powershell
python scripts\cleanup_workspace.py --execute
```

## 项目目录

```text
PredictDesign/
|- predictdesign/            # 可复用库代码
|  |- benchmark/             # benchmark 适配、rich log、训练/评估封装
|  |- gnn/                   # GNN/RT/hybrid 图骨干、冷启动、动作预测器
|  |- llm/                   # OpenAI-compatible LLM predictor
|  `- state_update/          # GRU / MDP 状态更新器
|- examples/                 # 离线可运行示例
|- scripts/                  # 命令行入口与运维脚本
|  |- benchmark/             # benchmark runner 与 rich log 导出
|  |- training/              # MLP/GNN 训练与评估
|  `- ops/                   # 清理、监控、长任务 shell launcher
|- tests/                    # 单元测试与回归测试
|- data/                     # 本地数据目录，例如 data/acg_nap/
|- results/                  # 实验输出、报告、归档结果
|- docs/                     # 维护说明
|- vendor/                   # 第三方 benchmark 代码
|- README.md
`- pyproject.toml
```

目录约定：

- `predictdesign/` 只放库代码，不写训练产物。
- `examples/` 只放小型可运行示例，不放缓存和结果。
- `scripts/benchmark/` 放 benchmark 运行与日志导出。
- `scripts/training/` 放训练、评估、checkpoint/result 读取。
- `scripts/ops/` 放清理、监控和长任务启动器。
- `results/` 放报告、图表、运行输出。
- `data/` 放本地数据，不提交大体积数据。
- 路径常量集中在 [predictdesign/paths.py](predictdesign/paths.py)。

## 代码框架

### 核心数据流

```text
raw benchmark logs / local records
        |
        v
BenchmarkEpisode / EpisodeStep
        |
        v
PredictDesignSystem.initialize_graph(...)
        |
        +--> TemporalGraph: nodes, temporal edges, structural edges, metadata
        +--> CTDG: continuous-time node states and message history
        |
        v
GraphActionPredictor / LLMApiGraphActionPredictor
        |
        v
PredictedGraphAction / PredictionRollout / PredictionSubgraphRollout
        |
        v
BenchmarkTrainer / BenchmarkEvaluator / scripts
```

### 关键模块

- [predictdesign/temporal_graph.py](predictdesign/temporal_graph.py)
  定义 `TemporalNode`、`TemporalEdge`、`TemporalGraph`。图中同时保存时序边、结构边、结构边 metadata 和图级上下文文本。

- [predictdesign/ctdg.py](predictdesign/ctdg.py)
  连续时间动态图状态容器，保存节点 hidden state、message history 和 state history。

- [predictdesign/messages.py](predictdesign/messages.py)
  定义 query/completion message。message 会被编码后用于更新 CTDG 节点状态。

- [predictdesign/encoders.py](predictdesign/encoders.py)
  文本、角色、时间、节点和消息编码器。SentenceTransformer 不可用或使用 fallback sentinel 时，会走本地 hash encoder。

- [predictdesign/prediction.py](predictdesign/prediction.py)
  定义 `GraphActionType`、`PredictedGraphAction`、`GraphPredictionContext`、rollout 结果对象。

- [predictdesign/experiment.py](predictdesign/experiment.py)
  顶层 `PredictDesignSystem`，负责组装图、CTDG、state updater、predictor 和 query parser。

- [predictdesign/gnn/layers.py](predictdesign/gnn/layers.py)
  GCN、GraphSAGE、GAT、Relational Transformer 和 `HybridGraphLayer`。

- [predictdesign/gnn/predictor.py](predictdesign/gnn/predictor.py)
  图动作预测器。它负责图编码、候选动作打分、action type/count 打分、completion-aware 调整和 rollout apply。

- [predictdesign/benchmark/](predictdesign/benchmark)
  负责把不同数据源变成统一的 `BenchmarkEpisode`，并提供训练、评估、rich log 输出。

### 训练样本结构

`BenchmarkEpisode` 是训练/评估的主样本：

- `initial_nodes`：初始节点。
- `initial_edges`：初始时序边。
- `initial_structural_edges`：workflow 或 benchmark 给出的结构边。
- `initial_graph_context_text`：图级任务描述。
- `initial_structural_edge_metadata`：transition relation、transition id、description。
- `steps`：按时间排列的 `EpisodeStep`。

`EpisodeStep` 包含：

- `observation_time`：观察时刻。
- `messages`：当前时刻被 CTDG ingest 的消息。
- `observed_actions`：真实发生的图动作。
- `valid_next_actions`：下一步监督动作，支持 parallel valid actions。
- `candidate_actions`：候选动作列表。
- `context_updates` / `context_text_updates`：节点上下文更新。
- `prediction_context`：当前 source/query/profile/latest output/candidate transition 等条件信息。

## 当前推荐模型

默认推荐强模型是：

```python
ExperimentConfig(
    gnn_type="hybrid",
    use_context_conditioning=True,
    use_candidate_cross_encoder=True,
    use_structural_candidate_priors=True,
)
```

`hybrid` 每层融合四种图视角：

- GCN：稳定局部聚合。
- GraphSAGE：自身状态与邻居状态对比。
- GAT：可学习邻居注意力。
- Relational Transformer：角色注意力、结构邻居注意力、全局注意力和 gated MLP。

预测器额外使用：

- 上下文条件化：`GraphPredictionContext` 会门控更新节点 embedding、图 embedding 和 action type logits。
- 候选 cross-encoder：对 source/target/graph/context/text/relation/edge 特征联合打分。
- 结构先验：结构 transition metadata 与候选动作匹配时会加分。
- 冷启动初始化：已有节点不再从纯零 state 开始，而是融合 role、节点文本、图级 profile 和结构边描述。

这些能力是模型能力层面的增强；实际收益仍应通过 holdout 或交叉验证确认。

## SentenceTransformer fallback

如果 `sentence_transformer_path` 或 rich log MLP 的 `sentence_transformer_model` 使用下面前缀，代码会直接启用本地 hash text encoder，不会访问 HuggingFace：

- `__missing_*`
- `__fallback_*`
- `fallback_hash_*`

示例：

```python
config = ExperimentConfig(
    sentence_transformer_path="__fallback_sentence_transformer__",
)
```

真实实验建议传入可访问的模型名或本地模型路径，例如：

```powershell
python scripts\training\train_acg_nap_gnn.py --sentence-transformer-path C:\models\all-MiniLM-L6-v2
```

## 运行示例

### Minimal Hybrid Demo

```powershell
python examples\minimal_demo.py
```

演示内容：

- 初始化三节点 workflow。
- 注入图级上下文和结构边 metadata。
- 构造 `GraphPredictionContext`。
- 使用 `hybrid` + candidate cross-encoder 输出候选动作排序。

### RT / Hybrid 对比

```powershell
python examples\rt_demo.py
```

演示内容：

- 构建同一个 toy episode。
- 分别使用 `relational_transformer` 和 `hybrid` 预测。
- 跑一个极小 supervised update，展示 trainer API。

### LLM API Predictor

默认离线假 completion：

```powershell
python examples\llm_api_predictor_example.py
```

真实 OpenAI-compatible endpoint：

```powershell
$env:PREDICTDESIGN_LLM_API_KEY="..."
$env:PREDICTDESIGN_LLM_BASE_URL="https://api.siliconflow.cn/v1"
$env:PREDICTDESIGN_LLM_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
python examples\llm_api_predictor_example.py --real
```

## 训练和评估

### ACG-NAP GNN holdout

查看参数：

```powershell
python scripts\training\train_acg_nap_gnn.py --help
```

推荐强模型入口：

```powershell
python scripts\training\train_acg_nap_gnn.py --gnn-types hybrid
```

常用参数：

- `--acg-nap-root`：ACG-NAP 数据目录，默认来自 `predictdesign.paths.ACG_NAP_ROOT`。
- `--train-epochs`：训练 epoch。
- `--train-fraction`：holdout 训练比例。
- `--sentence-transformer-path`：真实模型名、本地路径或 fallback sentinel。
- `--max-files-per-dataset`：调试时限制每个 dataset 的文件数，`0` 表示不限制。

### ACG-NAP candidate rerank

```powershell
python scripts\training\train_acg_nap_candidate_gnn.py --help
python scripts\training\train_acg_nap_candidate_gnn.py --gnn-type hybrid
```

该脚本把 `prediction.transition_candidates` 当成当前步候选集，训练候选排序分数。它会使用 source、query、profile、latest output 和 candidate description。

### Rich Log MLP

```powershell
python scripts\train_rich_log_mlp.py --help
```

用途：

- 从 rich log 中抽取 query、node outputs、graph structure。
- 比较不同信息组合的 MLP 分类效果。
- 结果输出到 `results/`。

### MultiAgentBench / MARBLE

```powershell
python scripts\benchmark\run_multiagentbench_eval.py --help
python scripts\benchmark\run_marble_hitk_benchmark.py --help
```

`scripts/benchmark/` 下是主入口；根目录 `scripts/run_*.py` 是兼容包装。

## 脚本目录

脚本说明已经集中在本文档的“脚本目录”部分。

常用入口：

```powershell
python scripts\run_parallel_api_rich_logs.py --help
python scripts\run_marble_hitk_benchmark.py --help
python scripts\train_rich_log_mlp.py --help
python scripts\train_parallel_api_gnn.py --help
python scripts\cleanup_workspace.py --help
python scripts\monitor_full_runs.py --help
```

按职责使用新路径：

```powershell
python scripts\benchmark\export_rich_log.py --help
python scripts\benchmark\run_multiagentbench_eval.py --help
python scripts\training\train_acg_nap_gnn.py --help
python scripts\training\train_existing_results_gnn.py --help
python scripts\ops\cleanup_workspace.py --help
```

## 输出和缓存

默认约定：

- 运行报告、CSV、图片、checkpoint 写入 `results/`。
- 本地输入数据放入 `data/`。
- 第三方 benchmark 代码放入 `vendor/`。
- 不要把 `__pycache__`、`.pytest_cache`、临时结果、模型产物写入 `predictdesign/`、`examples/` 或 `tests/`。

清理命令：

```powershell
python scripts\cleanup_workspace.py --execute
```

归档旧 smoke 结果：

```powershell
python scripts\cleanup_workspace.py --execute --archive-smoke-results
```

## 开发建议

- 新增库能力先放在 `predictdesign/`，再从 scripts 或 examples 调用。
- 新增 benchmark 数据适配器放在 `predictdesign/benchmark/`。
- 新增训练脚本放在 `scripts/training/`。
- 新增 benchmark runner 放在 `scripts/benchmark/`。
- 新增清理、监控、shell 启动器放在 `scripts/ops/`。
- 新增示例需要能离线运行，优先使用 fallback text encoder 或 fake completion。
- 修改路径时同步检查 [predictdesign/paths.py](predictdesign/paths.py)。
- 改动模型行为后至少运行：

```powershell
python -m compileall predictdesign examples scripts tests
python tests\test_predictdesign.py
python examples\minimal_demo.py
```
