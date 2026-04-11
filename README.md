# PredictDesign

这是一个面向“多智能体协作时序图预测”的模块化实验框架，重点是把每一层都拆开，方便你单独替换和做组合实验：

- `Temporal Graph`：维护初始角色节点、上下文超节点信息、带起止时间的时序边。
- `CTDG`：与 `Temporal Graph` 同步节点集合，并为每个节点维护历史状态 `S`。
- `Message`：支持 `query_arrival` 和 `node_completion` 两类动作，允许只有起点、只有终点、或两端都存在。
- `Concurrent Aggregation`：并发消息可切换为 `sum` 或 `mean` 聚合。
- `State Updater`：同时提供 `GRU` 和 `MDP` 风格的历史状态更新器。
- `GNN Predictor`：将节点历史状态、节点上下文、当前时序图一起编码，预测未来若干步子图动作。
- `Empty Graph Support`：允许初始时没有节点，此时预测器会先在 `add_node/no_op` 之间选择。
- `Temporal Edge Encoding`：边的 `start_time/end_time` 会被编码成显式边特征送入 GNN，而不只是简单二值邻接。

## 目录结构

```text
PredictDesign/
├── predictdesign/
│   ├── aggregation.py
│   ├── config.py
│   ├── ctdg.py
│   ├── encoders.py
│   ├── experiment.py
│   ├── gnn/
│   ├── messages.py
│   ├── prediction.py
│   ├── state_update/
│   ├── temporal_graph.py
│   └── types.py
├── examples/
│   └── minimal_demo.py
└── tests/
    └── test_predictdesign.py
```

## 设计要点

### 1. Temporal Graph

- 每个节点 `TemporalNode` 同时包含：
  - `role`
  - `context`
- 每条边 `TemporalEdge` 包含：
  - `source_node_id`
  - `target_node_id`
  - `start_time`
  - `end_time`

### 2. CTDG 与消息更新

- `ContinuousTimeDynamicGraph` 初始化时与 `TemporalGraph` 节点完全一致。
- 每个节点都维护：
  - `current_states[node_id]`
  - `state_history[node_id]`
- 消息按时间排序后逐时间戳更新。
- 同一时间戳下，触达同一节点的消息先聚合，再更新状态。
- 聚合方式通过 `ExperimentConfig.concurrent_update_mode` 控制：
  - `sum`
  - `mean`

### 3. 状态更新模块

- `GRUStateUpdater`
  - 用 `previous_state + current_context + aggregated_message` 更新隐藏状态。
- `MDPStateUpdater`
  - 将更新过程建模为离散潜在状态转移，再输出期望状态表示。

可通过 `ExperimentConfig.state_updater_type` 切换：

```python
ExperimentConfig(state_updater_type="gru")
ExperimentConfig(state_updater_type="mdp")
```

### 4. GNN 预测模块

当前提供三种可切换的 GNN backbone：

- `gcn`
- `graphsage`
- `gat`

可通过 `ExperimentConfig.gnn_type` 切换：

```python
ExperimentConfig(gnn_type="gcn")
ExperimentConfig(gnn_type="graphsage")
ExperimentConfig(gnn_type="gat")
```

预测动作类型包括：

- `create_edge`
- `remove_edge`
- `add_node`
- `no_op`

当预测为 `add_node` 时：

- `Temporal Graph` 新增一个已知角色、空 context 的节点
- `CTDG` 新增一个历史状态为零的节点

此外，当前版本补充了两点之前容易缺失的实验能力：

- 当初始图为空时，GNN 会退化到一个图级空状态嵌入，不会因为 `torch.stack([])` 报错。
- 时序边特征默认编码为 4 维：
  - `is_active`
  - `log(1 + duration)`
  - `log(1 + elapsed_since_start)`
  - `log(1 + remaining_until_end)`

### 5. MDP 状态更新模块

`MDPStateUpdater` 现在不只是一个“像马尔可夫”的更新层，而是显式拆成了：

- `policy_model`：给出潜在动作分布
- `transition_model`：给出 `action -> next_state` 转移分布
- `value_model`：给出当前状态值估计
- `step(...)`：同时返回下一个状态和一份 `MDPTransitionSummary`

这样你后面做 GRU/MDP 对比实验时，可以直接分析策略分布和转移矩阵，而不只是看最后的 hidden state。

## 最小示例

```python
from predictdesign import ExperimentConfig, PredictDesignSystem

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
```

更完整的可运行例子见 [examples/minimal_demo.py](/data0/fanjiarun/PredictDesign/examples/minimal_demo.py)。

## 运行方式

当前机器里系统 `python` 没有安装 `torch`，建议直接用已有环境：

```bash
cd /data0/fanjiarun/PredictDesign
conda run -n benchmark python examples/minimal_demo.py
conda run -n benchmark python -m unittest discover -s tests
```

说明：当前实现是“可训练的实验骨架”，不是已经训练好的模型，所以初始直接运行时，预测动作可能偏向 `no_op`，这是正常现象；需要你后续接入数据和训练流程后，预测头才会学到更有意义的动作分布。

## 后续建议

这个版本优先解决“可组合、可替换、可跑通”：

- 如果你后续想接入文本 context，可以把 `context` 向量替换成 LLM encoder 输出。
- 如果你想把 GNN 换成 PyG/DGL 实现，现在只需要替换 `predictdesign/gnn/`。
- 如果你想尝试更复杂的消息机制，可以替换 `MessageEncoder` 或 `ConcurrentMessageAggregator`。
