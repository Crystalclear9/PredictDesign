# PredictDesign

PredictDesign is a framework for predicting temporal transitions in multi-agent systems. It represents a run as a dynamic collaboration graph and supports speculative prediction both when a query arrives and while the query is being executed.

The current focus is severe cold start. A scenario may have only dozens of queries, or about 100 at most, so the system cannot wait for a full scenario to finish before learning. The practical design is predict-first online learning: score with only visible information, then update memory after the true action has happened.

## Core Protocol

The strict rule is simple: do not use information that would not exist at prediction time.

Forbidden inputs include current agent output, `latest_output`, `source_output_text`, `runtime_text`, old `Predict` entries, `prediction.transition_candidates`, the current label, future labels, and any context snapshot from a future event. Candidate sets must come from visible workflow/config/tool schema/graph structure, not from the label.

There are two different targets:

- `current_event`: predicts the current event's agent from previous history only. The current request text is not read because it usually contains `You are agentX`.
- `next_agent`: assumes the currently executing `agent_id` is known and predicts the next agent. This is the protocol for query-internal speculative scheduling. It may use the current tool schema, current visible prompt, static agent profiles, visible graph edges, and already completed history/memory. When `--include-visible-agent-context` is enabled, it may also use the current event's `agents[*].context` snapshot as completed history, but never the next event's context or the current agent's not-yet-produced output.

Graph structure is handled as follows:

- If an explicit graph field exists, use it.
- If not, infer collaboration edges from visible prompt/tool schema text such as `agent1 collaborates with agent3` and `target_agent_id.enum`.
- After each prediction is scored, update online memory with the observed transition for later steps.

## Setup

Requires Python `>=3.11`.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python -c "import predictdesign; print('OK')"
```

Useful checks:

```powershell
.venv\Scripts\python.exe tests\test_predictdesign.py
.venv\Scripts\python.exe -m compileall predictdesign scripts tests
```

Clean generated caches:

```powershell
.venv\Scripts\python.exe scripts\cleanup_workspace.py --execute
```

## Raw Log Evaluation

The current raw research logs live under `results/research`. They do not contain the old ACG-NAP fields `prediction`, `label`, or `transition_candidates`. The evaluator filters true raw event JSONL files by first-line event fields, so generated `*_timing.jsonl` files in the same directory are not accidentally re-used as input.

Run the strict generic query-internal next-agent protocol on one scenario folder:

```powershell
.venv\Scripts\python.exe scripts\benchmark\run_new_log_cold_start.py `
  --log-root results\research\coding `
  --prediction-target next_agent `
  --use-cross-file-memory `
  --enable-adaptive-cross-file-prior `
  --adaptive-cross-file-weight 30 `
  --adaptive-cross-file-min-support 1 `
  --adaptive-cross-file-min-confidence 0.4 `
  --adaptive-cross-file-min-profile-stability 0.65 `
  --enable-visible-order-prior `
  --enable-cross-query-start-prior `
  --enable-online-pair-calibration `
  --pair-calibration-margin 1 `
  --include-visible-agent-context `
  --no-enable-online-reranker `
  --no-enable-idf-profile-prior `
  --report-path results\research\coding\next_agent_strict_generic_report.json `
  --timing-path results\research\coding\next_agent_strict_generic_timing.jsonl `
  --audit-path results\research\coding\next_agent_strict_generic_audit.json
```

Use `--log-root results\research\research` and matching output paths to run the research scenario.

Run the concurrent-arrival protocol where each batch has 3-4 queries. All queries in one batch are scored from the same pre-batch online memory snapshot; their observed labels update cross-query memory only after the whole batch has been scored. Query-internal online updates are still allowed inside each file because those events happen sequentially within the query.

```powershell
.venv\Scripts\python.exe scripts\benchmark\run_new_log_concurrent_suite.py `
  --root results\research `
  --scenarios coding research `
  --batch-sizes 3 4
```

To run or inspect one scenario only:

```powershell
.venv\Scripts\python.exe scripts\benchmark\run_new_log_concurrent_batches.py `
  --log-root results\research\coding `
  --batch-size 3

.venv\Scripts\python.exe scripts\benchmark\run_new_log_concurrent_batches.py `
  --log-root results\research\research `
  --batch-size 4
```

Run the base policy without cross-file priors or online reranking:

```powershell
.venv\Scripts\python.exe scripts\benchmark\run_new_log_cold_start.py `
  --log-root results\research\coding `
  --prediction-target next_agent `
  --no-use-cross-file-memory `
  --no-enable-visible-order-prior `
  --no-enable-cross-query-start-prior `
  --no-enable-online-reranker `
  --report-path results\research\coding\next_agent_base_report.json `
  --timing-path results\research\coding\next_agent_base_timing.jsonl `
  --audit-path results\research\coding\next_agent_base_audit.json
```

Optional raw cross-file diagnostics can be run with `--cross-file-stat-weight`, but the strict generic protocol uses the adaptive prior instead. The adaptive prior trusts a completed-history transition only when the current source agent has enough support, confidence, and same-agent profile stability. It is still online: prediction happens first, then the true target updates memory.

Latest local results on `results/research/coding` and `results/research/research`. The primary metric is strict hit@1 per scenario, not a weighted average. It counts only current `main_turn` events whose next event is actual agent work. `continuation`, `planner_summary`, `planner_continue`, and other scheduler recovery events are excluded from both the metric and predictive transition memory.

| Scenario | Files | Steps | hit@1 | hit@2 | hit@3 | hit@5 | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coding | 20 | 402 | 82.59% | 86.57% | 92.04% | 93.03% | 9.25 ms | 22.98 ms |
| research | 20 | 200 | 69.00% | 88.50% | 95.50% | 99.00% | 13.60 ms | 46.25 ms |

The current strict generic single-branch result does not satisfy the desired 80% hit@1 on every scenario. `coding` reaches the target; `research` does not. Reporting only the weighted mean would hide that failure, so the README keeps the per-scenario table as the primary result.

The strongest useful signal is still top-k coverage: research reaches 88.50% hit@2 and 95.50% hit@3 without leakage. That is useful for multi-branch speculative execution, but it is not a substitute for hit@1 and should not be reported as single-action accuracy.

Concurrent-arrival results on the same folders:

| Scenario | Batch size | Batches | Steps | hit@1 | hit@2 | hit@3 | hit@5 | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coding | 3 | 7 | 402 | 82.84% | 86.82% | 92.04% | 93.03% | 9.04 ms | 22.54 ms |
| coding | 4 | 5 | 402 | 82.84% | 86.82% | 92.04% | 93.03% | 15.29 ms | 38.97 ms |
| research | 3 | 7 | 200 | 67.00% | 89.00% | 95.50% | 98.50% | 16.22 ms | 44.40 ms |
| research | 4 | 5 | 200 | 64.00% | 86.50% | 94.50% | 99.00% | 23.70 ms | 83.53 ms |

Cross-scenario concurrent averages:

| Batch size | Step-micro hit@1 | Scenario-macro hit@1 | Query-macro hit@1 | Batch-macro hit@1 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 77.57% | 74.92% | 77.33% | 76.86% |
| 4 | 76.58% | 73.42% | 75.58% | 74.03% |

`Step-micro` means one vote per strict predictive step. `Scenario-macro` means coding and research have equal weight. `Query-macro` means each query with at least one strict predictive step has equal weight. `Batch-macro` means each concurrent batch has equal weight.

The drop from sequential replay is expected: in batch mode, query 2-4 inside the same batch cannot benefit from query 1's newly observed transitions. This is stricter and closer to a real concurrent scheduler than a one-query-at-a-time replay.

The generic priors used by the strict protocol are:

- Adaptive cross-file transition prior: uses only completed queries and is gated by support, confidence, and profile stability.
- Contextual cross-file transition prior: conditions completed-query memory on current source turn count, round position, and visible outgoing signature.
- Visible role workflow prior: infers roles such as analyst, implementation, tester, debugger, and reviewer from current visible profiles/tool descriptions, then scores the matching next role. This is profile-derived and does not read labels or outputs.
- Visible-order prior: uses the current visible candidate order and source-turn count.
- Cross-query start prior: reuses previous completed query starts and already observed early targets inside the current query.
- Pair calibration: swaps top1/top2 only after the same top1/top2 pattern has already been observed earlier.
- Visible context features: optionally uses current-event `agents[*].context` as completed history only.

Leakage controls:

- It scores only `main_turn` records.
- It uses current `agent_id`, visible target candidates, static profiles, visible prompt text, inferred graph/tool schema, and already completed history. With `--include-visible-agent-context`, it also uses only the current event's already completed `agents[*].context` snapshot.
- It updates weights only after the expected next agent has been scored and recorded.
- It does not read current outputs, future events, old `prediction` fields, or `prediction.transition_candidates`.
- It does not count scheduler recovery as model accuracy.

## Other Benchmarks

Strict ACG-NAP workflow/candidate policy:

```powershell
.venv\Scripts\python.exe scripts\benchmark\run_acg_nap_workflow_policy.py --max-files-per-dataset 0
```

MARBLE/vendor online cold-start simulation:

```powershell
.venv\Scripts\python.exe scripts\benchmark\run_vendor_online_cold_start.py `
  --queries 60 `
  --device cuda `
  --require-cuda `
  --speculative-steps 3 `
  --latency-warmup-steps 3 `
  --report-path results\vendor_online_cold_start_xai_60q_timing.json
```

Candidate-GNN strict zero-online evaluation:

```powershell
.venv\Scripts\python.exe scripts\training\train_acg_nap_candidate_gnn.py `
  --device cuda `
  --context-dim 16 `
  --hidden-dim 32 `
  --gnn-type gcn `
  --state-updater gru `
  --message-reduce-mode attention `
  --train-epochs 0 `
  --sentence-transformer-path __fallback_hash_encoder__ `
  --max-files-per-dataset 0 `
  --candidate-source graph_transitions_by_source `
  --no-bootstrap-few-shot-memory `
  --eval-memory-mode empty `
  --progress-interval 0 `
  --report-path results\acg_nap\candidate_graph_transitions_zero_online_full_0e_report.json `
  --split-summary-path results\acg_nap\candidate_graph_transitions_zero_online_full_0e_split_summary.json `
  --cleaning-summary-path results\acg_nap\candidate_graph_transitions_zero_online_full_0e_cleaning_summary.json `
  --timing-path results\acg_nap\candidate_graph_transitions_zero_online_full_0e_timing.jsonl `
  --audit-path results\acg_nap\candidate_graph_transitions_zero_online_full_0e_audit.json
```

Role/profile/query-only CUDA ablation:

```powershell
.venv\Scripts\python.exe scripts\training\train_acg_nap_candidate_gnn.py `
  --device cuda `
  --train-epochs 10 `
  --sentence-transformer-path __fallback_hash_encoder__ `
  --max-files-per-dataset 0 `
  --role-prompt-query-only `
  --gpu-only-learned-scoring `
  --timing-path results\acg_nap\candidate_role_prompt_query_only_gpu_only_full_10e_timing.jsonl
```

`--gpu-only-learned-scoring` disables CPU-heavy zero-shot/few-shot text priors, so those results should not be mixed with workflow-policy results.

## Timing

Timing files are JSONL, one record per evaluated step. For `run_new_log_cold_start.py`, the main field is `prediction_time_ms`. For Candidate-GNN, timing also includes:

- `prediction_score_time_ms`: scoring candidate actions.
- `prediction_rank_time_ms`: sorting and hit@k calculation.
- `observed_update_time_ms`: online update after prediction.
- `online_step_overhead_ms`: prediction plus observed update.

CUDA training scripts synchronize around GPU work before timing where needed, so asynchronous GPU execution does not make timing look falsely low.

## Code Structure

```text
PredictDesign/
|- predictdesign/             # Core package
|  |- benchmark/              # Dataset adapters, evaluators, workflow policy
|  |- gnn/                    # Graph encoders, predictors, priors, memory
|  |- llm/                    # OpenAI-compatible LLM predictor
|  |- state_update/           # GRU and MDP state updates
|  |- config.py               # Experiment configuration
|  |- experiment.py           # PredictDesignSystem orchestration
|  |- prediction.py           # Prediction context and action types
|  |- temporal_graph.py       # Temporal graph data model
|- scripts/
|  |- benchmark/              # Evaluation CLIs
|  |- training/               # GNN and MLP training CLIs
|  |- ops/                    # Cleanup and maintenance
|- examples/                  # Minimal runnable examples
|- tests/                     # Unit tests
|- results/                   # Reports and checkpoints
```

The root `README.md` is the only documentation entry point. Subdirectories do not maintain separate READMEs.

## Module Notes

`predictdesign.config`

Defines `ExperimentConfig` and `LLMApiConfig`, including hidden size, context size, GNN type, prediction horizon, cold-start priors, few-shot memory, candidate scoring, and LLM endpoint settings.

`predictdesign.experiment`

Defines `PredictDesignSystem`. It initializes graphs, injects messages, updates state, calls predictors, runs speculative rollouts, records observed actions, and maintains query-internal `active_prediction_context`.

`predictdesign.prediction`

Defines `GraphPredictionContext`, `PredictedGraphAction`, `GraphActionType`, `PredictionRollout`, and `PredictionSubgraphRollout`. `GraphPredictionContext.query_time_view()` strips fields that should not be visible at query time.

`predictdesign.temporal_graph`

Defines `TemporalNode`, `TemporalEdge`, and `TemporalGraph`, including temporal edges, structural edges, structural metadata, node context, graph context, and adjacency construction.

`predictdesign.gnn`

Contains graph layers, `GraphActionPredictor`, cold-start initialization, cold-start action prior, and online few-shot transition memory. The predictor can rank supplied candidates, score graph actions, and apply predicted actions to temporary rollout graphs.

`predictdesign.benchmark`

Contains `BenchmarkEpisode`, `EpisodeStep`, `BenchmarkTrainer`, `BenchmarkEvaluator`, ACG-NAP adapters, MultiAgentBench/MARBLE adapters, rich-log utilities, and strict workflow/candidate policies.

`predictdesign.llm`

Contains `LLMApiGraphActionPredictor`, which formats graph state, candidate actions, prediction context, and rollout history into an OpenAI-compatible prompt and parses JSON actions. Examples default to fake completions unless a real endpoint is requested.

## Examples

```powershell
.venv\Scripts\python.exe examples\minimal_demo.py
.venv\Scripts\python.exe examples\rt_demo.py
.venv\Scripts\python.exe examples\llm_api_predictor_example.py
```

- `examples/minimal_demo.py`: smallest hybrid workflow with structural metadata and candidate ranking.
- `examples/rt_demo.py`: compares relational-transformer and hybrid GNN behavior with a tiny supervised update.
- `examples/llm_api_predictor_example.py`: shows the LLM predictor with offline fake completion by default.

## Query-Level And In-Query Prediction

At query arrival, create a query-time view and predict before writing current outputs:

```python
from predictdesign import GraphPredictionContext

query_context = GraphPredictionContext(
    source_node_id="planner",
    query_text=query_text,
    candidate_actions=candidate_actions,
).query_time_view()

rollout = system.predict_speculative_action_sets(
    observation_time=current_time,
    steps=system.config.prediction_horizon,
    prediction_context=query_context,
)
```

During query execution, recompute speculative rollout only with runtime messages or observed actions that are already visible:

```python
rollout = system.process_query_runtime_update(
    observation_time=current_time,
    prediction_context=query_context,
    messages=runtime_messages,
    context_updates={"coder": coder_context_vector},
    context_text_updates={"coder": "implementation complete"},
    observed_actions=observed_actions,
    steps=system.config.prediction_horizon,
)
```

Do not place the final output of the currently predicted step into this context before scoring.

## SentenceTransformer Fallback

Use the fallback sentinel to avoid HuggingFace HEAD retry warnings for intentionally missing test models:

```text
__missing_sentence_transformer_model__
```

For fully local hash encoding, use:

```text
__fallback_hash_encoder__
```

For real experiments, pass an accessible local path or model name:

```powershell
.venv\Scripts\python.exe scripts\training\train_acg_nap_gnn.py `
  --sentence-transformer-path C:\models\all-MiniLM-L6-v2
```

## Leakage Checklist

- Predict first, then update online memory with the observed action.
- Do not read current outputs, future request/output, old `Predict` fields, or `prediction.transition_candidates`.
- Do not fill a missing candidate set from `ground_truth_action` unless explicitly running an oracle upper bound.
- Treat context snapshots carefully. The current event's `agents[*].context` can be used only as already completed history when `--include-visible-agent-context` is explicitly enabled. Future context snapshots, current output text, and post-label updates are forbidden.
- For `next_agent`, current `agent_id` is allowed because the executing agent is known.
- For raw logs without an explicit graph, inferred graph edges from visible prompt/tool schema are allowed and must be reported as inferred.

## Maintenance

```powershell
.venv\Scripts\python.exe tests\test_predictdesign.py
.venv\Scripts\python.exe -m compileall predictdesign scripts tests
.venv\Scripts\python.exe scripts\cleanup_workspace.py --execute
```

Keep caches, temporary reports, and checkpoints out of source directories. Put reports and checkpoints under `results/`, local datasets under `data/`, and third-party benchmarks under `vendor/`.
