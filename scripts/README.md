# Scripts

The repository now separates scripts by responsibility:

- `benchmark/`
  Run MultiAgentBench and MARBLE pipelines, export logs, and orchestrate benchmark evaluation.

- `training/`
  Train MLP or GNN models and evaluate saved checkpoints/results.

- `ops/`
  Clean cache directories, archive smoke runs, monitor long-running jobs, and host shell launchers.

## Backward-Compatible Entrypoints

These top-level wrappers are kept because they are used most often:

- `python scripts/run_parallel_api_rich_logs.py`
- `python scripts/run_marble_hitk_benchmark.py`
- `python scripts/train_rich_log_mlp.py`
- `python scripts/train_parallel_api_gnn.py`
- `python scripts/cleanup_workspace.py`

All other commands should be run from their organized subdirectories, for example:

```bash
python scripts/training/train_acg_nap_gnn.py
python scripts/benchmark/export_rich_log.py
python scripts/ops/monitor_full_runs.py
```
