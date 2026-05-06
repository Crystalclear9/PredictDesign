# PredictDesign

PredictDesign 是一个面向多智能体协作过程的时序图预测实验框架。仓库当前包含三类核心能力：

- 图建模与预测：时序图、CTDG 状态更新、Relational Transformer、冷启动、节点完成检测。
- 基准数据与日志：ACG-NAP 适配、MultiAgentBench / MARBLE 结果读取、rich log 导出。
- 训练与评估：基于 rich log 的 MLP 训练、基于图结构的 GNN 训练、批量实验脚本。

当前版本：`0.2.0`

## 环境要求

- Python `>= 3.11`
- 推荐使用虚拟环境
- 主要 Python 依赖见 [pyproject.toml](C:/Users/70454/Desktop/PredictDesign/pyproject.toml)

安装：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

最小验证：

```bash
python -c "import predictdesign; print('OK')"
```

## 当前目录结构

```text
PredictDesign/
|- predictdesign/            # 核心库代码
|- scripts/                  # 运行、训练、运维脚本
|  |- benchmark/             # benchmark 运行与 rich log 导出
|  |- training/              # MLP / GNN 训练与评估
|  `- ops/                   # 清理、监控、长任务启动
|- examples/                 # 最小示例与 RT 示例
|- tests/                    # 自动化测试
|- data/                     # 本地数据目录
|  `- acg_nap/
|- vendor/                   # 第三方基准代码
|  `- prefetch-kv-mas/
|- results/                  # 实验输出、日志、归档结果
|- docs/                     # 结构与维护说明
|- README.md
`- pyproject.toml
```

目录约定：

- `predictdesign/` 只放可复用库代码。
- `scripts/benchmark/` 只放 benchmark 运行和日志导出脚本。
- `scripts/training/` 只放训练与评估脚本。
- `scripts/ops/` 只放清理、监控、shell 启动器。
- `results/` 放运行结果，不把产物写回源码目录。
- 路径常量统一定义在 [predictdesign/paths.py](C:/Users/70454/Desktop/PredictDesign/predictdesign/paths.py)。

## 核心模块

### 1. 图建模与预测

位于 `predictdesign/`：

- `temporal_graph.py`：时序节点、边和图容器
- `ctdg.py`：连续时间动态图状态记录
- `encoders.py`：SentenceTransformer 文本编码与节点特征编码
- `completion.py`：节点完成检测
- `experiment.py`：顶层 `PredictDesignSystem`
- `gnn/`：Relational Transformer、冷启动与预测器
- `state_update/`：GRU / MDP 状态更新
- `llm/`：LLM API 预测器

公开 API 入口在 [predictdesign/__init__.py](C:/Users/70454/Desktop/PredictDesign/predictdesign/__init__.py)。

### 2. benchmark 与日志

位于 `predictdesign/benchmark/`：

- `acg_nap.py`：ACG-NAP 语料适配
- `multiagentbench.py`：MultiAgentBench / MARBLE 结果适配
- `rich_log.py`：rich log 写出、组合训练结果与图表输出
- `trainer.py` / `evaluator.py`：训练与评估逻辑

### 3. 脚本入口

脚本已经按职责拆分：

- [scripts/benchmark](C:/Users/70454/Desktop/PredictDesign/scripts/benchmark)
- [scripts/training](C:/Users/70454/Desktop/PredictDesign/scripts/training)
- [scripts/ops](C:/Users/70454/Desktop/PredictDesign/scripts/ops)

同时保留了一层兼容包装，下面这些命令仍然可以直接使用：

```bash
python scripts/run_parallel_api_rich_logs.py
python scripts/run_marble_hitk_benchmark.py
python scripts/train_rich_log_mlp.py
python scripts/train_parallel_api_gnn.py
python scripts/cleanup_workspace.py
python scripts/monitor_full_runs.py
```

脚本说明见 [scripts/README.md](C:/Users/70454/Desktop/PredictDesign/scripts/README.md)。

## 快速开始

### 1. 运行示例

```bash
python examples/minimal_demo.py
python examples/rt_demo.py
python examples/llm_api_predictor_example.py
```

### 2. 查看可用脚本参数

```bash
python scripts/run_parallel_api_rich_logs.py --help
python scripts/train_rich_log_mlp.py --help
python scripts/training/train_acg_nap_gnn.py --help
```

### 3. 清理工作区缓存

预览：

```bash
python scripts/cleanup_workspace.py
```

执行清理：

```bash
python scripts/cleanup_workspace.py --execute
```

连同旧 smoke 结果一起归档：

```bash
python scripts/cleanup_workspace.py --execute --archive-smoke-results
```

## 结果与数据约定

- 本地数据：`data/`
- 第三方 benchmark 代码：`vendor/`
- 运行结果：`results/`
- 归档结果：`results/archive/`

对于需要访问仓库内固定路径的代码，优先使用 `predictdesign.paths` 提供的常量，而不是在脚本里重复硬编码目录字符串。

## 常见工作流

### 导出并训练 rich log

```bash
python scripts/run_parallel_api_rich_logs.py --help
python scripts/train_rich_log_mlp.py --help
```

### 训练图模型

```bash
python scripts/training/train_acg_nap_gnn.py --help
python scripts/train_parallel_api_gnn.py --help
```

### 监控长任务

```bash
python scripts/monitor_full_runs.py --help
```

## 维护说明

维护约束已经做过一轮收敛，后续建议继续遵守：

- 新增训练脚本放到 `scripts/training/`
- 新增 benchmark runner 放到 `scripts/benchmark/`
- 新增清理或监控脚本放到 `scripts/ops/`
- 不要把缓存、日志、模型产物写到 `predictdesign/`、`tests/` 或 `examples/`
- 修改仓库路径时，同步更新 `predictdesign.paths`

补充说明见 [docs/PROJECT_STRUCTURE.md](C:/Users/70454/Desktop/PredictDesign/docs/PROJECT_STRUCTURE.md)。

## 测试

```bash
pytest
```

如果只做安装验证，至少执行：

```bash
python -c "import predictdesign; print('OK')"
python examples/minimal_demo.py
```
