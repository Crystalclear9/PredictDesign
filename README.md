# PredictDesign

PredictDesign predicts temporal transitions in multi-agent systems. It treats a run as a dynamic collaboration graph and supports speculative prediction both when a query arrives and while that query is still executing.

The current design is built for severe cold start. A scenario may have only dozens of queries, so the system predicts first, executes, then updates online memory only after the true next action has happened.

## Core Protocol

The evaluator follows one rule: never use information that would not exist at prediction time.

Forbidden inputs:

- Current or future agent output, including `latest_output`, `source_output_text`, and `runtime_text`.
- Old `Predict` entries, old generated predictions, `prediction.transition_candidates`, and label-derived candidate sets.
- The current label, future labels, future context snapshots, or post-label memory updates.
- Scheduler recovery events counted as model accuracy.

Allowed inputs:

- Current query text when it is available before execution.
- Current executing `agent_id` for `next_agent`, because the scheduler knows which agent is running.
- Static agent role/profile/prompt text, visible tool schema, visible workflow/config text, and inferred graph edges.
- Already completed history and online memory from earlier steps.
- Optional current-event `agents[*].context` only when `--include-visible-agent-context` is enabled; this is treated as an already visible snapshot, not as current output.

Two prediction targets are supported:

- `current_event`: predicts the current event's agent from previous history only. The current request text is avoided because it often says `You are agentX`.
- `next_agent`: predicts the next agent while the current agent is executing. This is the main protocol for query-internal speculative scheduling.

## Target And Metric Definition

Direct single-label `hit@1` is not enough because one `main_turn` may trigger multiple child agent calls. The current evaluator uses set-valued targets.

For each current `main_turn`:

1. Build `expected_agent_ids` from non-recovery child events whose `parent_event_ids` include the current event id.
2. If no child target exists, fall back to the next non-recovery event.
3. Exclude recovery/control events such as `continuation`, `planner_summary`, and `planner_continue`.
4. Count the step only when the current event is `main_turn` and the target set is non-empty.

Metric semantics:

- `hit@k`: true when the predicted top-k set intersects `expected_agent_ids`.
- `target_recall_at_k`: fraction of the target set covered by top-k.
- `branch_precision`: selected speculative branches that are actually needed.
- `set_f1` and `set_jaccard`: set-level quality when a step can have multiple valid targets.
- `cost_normalized_utility`: coverage minus extra branch cost under several branch-cost ratios.
- Wilson confidence intervals: uncertainty for small cold-start evaluations.
- Online convergence: the first request index after which every later cumulative hit@1 remains within a tolerance of the final value.
- `step_micro`: one vote per predictive step.
- `scenario_macro`: one equal-weight vote per scenario.
- `query_macro`: one equal-weight vote per query.
- `batch_macro`: one equal-weight vote per concurrent batch.

This means scheduler recovery is not credited to the model, and a multi-agent fan-out is evaluated as a set rather than a forced single next label.

## Setup

Requires Python `>=3.11`.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python -c "import predictdesign; print('OK')"
```

Verification:

```powershell
.venv\Scripts\python.exe tests\test_predictdesign.py
.venv\Scripts\python.exe -m compileall predictdesign scripts tests
```

Cleanup:

```powershell
.venv\Scripts\python.exe scripts\cleanup_workspace.py --execute
```

## Raw Log Evaluation

The current raw-log benchmark root is `results/research`, with `coding` and `research` subfolders. The strict evaluator discovers raw event JSONL files by event fields, so generated timing JSONL files in the same directory are not reused as input.

Run one scenario with the strict query-internal next-agent protocol:

```powershell
.venv\Scripts\python.exe scripts\benchmark\run_new_log_cold_start.py `
  --log-root results\research\coding `
  --prediction-target next_agent `
  --use-cross-file-memory `
  --enable-adaptive-cross-file-prior `
  --adaptive-cross-file-weight 30 `
  --adaptive-cross-file-min-support 1 `
  --adaptive-cross-file-min-confidence 0.4 `
  --adaptive-cross-file-min-profile-stability 0.0 `
  --online-evidence-mode transition_only `
  --online-feedback-scope query `
  --candidate-scope visible_graph `
  --no-enable-graph-order-prior `
  --no-enable-role-workflow-prior `
  --no-enable-visible-order-prior `
  --no-enable-cross-query-start-prior `
  --no-enable-profile-similarity-prior `
  --no-enable-online-pair-calibration `
  --pair-calibration-margin 1 `
  --no-include-visible-agent-context `
  --no-enable-online-reranker `
  --no-enable-idf-profile-prior `
  --report-path results\research\coding\next_agent_strict_generic_report.json `
  --timing-path results\research\coding\next_agent_strict_generic_timing.jsonl `
  --audit-path results\research\coding\next_agent_strict_generic_audit.json
```

Use `--log-root results\research\research` for the research scenario.

Run the concurrent-arrival suite:

```powershell
.venv\Scripts\python.exe scripts\benchmark\run_new_log_concurrent_suite.py `
  --root results\research `
  --scenarios coding research `
  --batch-sizes 1 3 4 `
  --policy-mode strict_online `
  --scenario-replay-mode pooled
```

The concurrent protocol scores every active query in a batch from the same pre-batch cross-query memory snapshot. Observed labels update cross-query memory only after the whole batch is scored. In `strict_online` and `strict_all_agents`, feedback is delayed until a query finishes, so a query does not learn from its own intermediate labels while it is being scored.

Scenario replay modes:

- `pooled`: recommended audit mode. All scenario folders are mixed into one replay with one shared online memory. The predictor does not receive the scenario label.
- `separate`: diagnostic mode. Each scenario folder is replayed with its own memory reset. This is useful for debugging but easier than the pooled deployment setting.

Policy modes:

- `strict_online`: strict query-level online replay with visible graph/tool-schema candidate scope. It removes hand-written role workflow rules, research schedule/meta rules, profile/context similarity, static graph-order bonuses, visible-history target-reference heuristics, position/round/recent-target heuristics, and online top1/top2 pair calibration. It keeps current executing agent id, visible graph/tool candidates, and completed-query `current_agent -> next_agent` transition memory. Feedback is delayed until each query completes.
- `strict_all_agents`: stricter audit ablation. It uses the same transition-only query-level online memory as `strict_online`, but disables visible graph/tool-schema candidate narrowing and ranks every visible agent in the file. Use this when you need to show how much of the result survives without candidate-set narrowing.
- `strict_no_memory`: visible graph/tool candidate scope, but completed-query memory is disabled. This is an audit baseline for candidate and deterministic ordering effects.
- `strict_no_memory_all_agents`: all visible agents as candidates, with completed-query memory disabled. This is the strictest zero-cross-query-memory baseline.
- `strict_profile_online`: disables raw cross-query agent-id transition memory and instead learns completed-query transitions over visible profile signatures and visible worker roster positions. Feedback is delayed until each query completes.
- `strict_profile_event_online`: same as `strict_profile_online`, but updates a query-local transition memory after each already-observed step inside the same query. This matches query-internal speculative scheduling.
- `strict_all_agents_profile_online` and `strict_all_agents_profile_event_online`: all-agent candidate ablations for the two profile/roster modes.
- `strict_id_permutation`: same as `strict_online`, but worker agent ids are consistently renamed inside each query file with a different deterministic permutation per file. This audits dependence on fixed cross-query agent numbering.
- `strict_all_agents_id_permutation`: same as `strict_all_agents`, plus the per-file agent-id permutation counterfactual.
- `strict_profile_id_permutation` and `strict_profile_event_id_permutation`: profile/roster modes under per-file agent-id permutation. These are the main anti-fixed-id audits.
- `skeptical_profile_event_id_permutation`: the most conservative audit mode. It uses per-file agent-id permutation, ranks all visible agents, disables graph-order, disables roster-position memory, disables raw cross-query agent-id memory, and disables raw local same-query transition memory.
- `semantic_skeptical_profile_event_id_permutation`: the conservative audit plus visible task-profile to all-candidate profile token matching. It still uses per-file agent-id permutation, all-agent ranking, no graph-order, no roster-position memory, no raw cross-query agent-id memory, and no raw local same-query transition memory.
- `structural_event_online`: query-internal profile/roster mode plus a cheap visible graph/tool-schema candidate-order prior.
- `structural_event_id_permutation`: same as `structural_event_online`, but with per-file agent-id permutation. This is the recommended deployable result when visible graph/tool order is accepted as system structure.
- `structural_all_agents_event_id_permutation`: all-agent candidate audit for `structural_event_id_permutation`.
- `fast`: graph, role, schedule, and online transition memory. This is a structural-prior ablation, not pure online learning.
- `compact`: keeps task-profile similarity while avoiding heavier context scans.
- `balanced`: disables visible agent-context similarity and is the recommended slow stage.
- `full`: uses all no-leak visible features, including visible agent-context similarity.
- `robust`: adds an episodic completed-query profile/position prior. It is implemented for comparison, but the latest run made hit@1 worse and it is not the default recommendation.
- `cascade`: runs `fast` first, then escalates to `balanced` only when the fast top1-top2 score margin is low.
- `expert_cascade`: runs `cascade`, then applies online expert-advice reranking over no-leak score components. Expert weights are keyed by scenario and current source agent, used before the current wave is labeled, and updated only after the wave has been scored.

Treat `fast`, `balanced`, `full`, `cascade`, and `expert_cascade` as structural-prior or upper-ablation modes unless the report explicitly separates their role/profile/schedule contribution. `structural_event_*` is also not pure online-memory-only; it is a deployable structural mode because visible graph/tool order is part of its input.

The raw-log suite is a lightweight online scorer, not a PyTorch training loop. CUDA is used by the GNN training scripts, but this evaluator is intended to measure low-overhead speculative scheduling.

The suite report also records `method_references` and `external_data_source_candidates`, so method rationale and future data-integration options are kept with the run output rather than only in this README.

Every raw-log report includes `policy_claim`. For strict reporting, use reports with `online_evidence_mode = "transition_only"` and `online_feedback_scope = "query"`, then disclose the claim:

- `online_learning_with_visible_candidate_scope`: visible graph/tool-schema candidate narrowing is still used.
- `online_learning_no_candidate_narrowing`: all visible agents are ranked; this is the stricter all-agent ablation.
- `zero_cross_query_memory_ablation`: completed-query memory is disabled; this is not online-learning performance.
- `agent_id_permutation_counterfactual`: agent ids are permuted per file to test dependence on fixed numbering.
- `profile_conditioned_online_memory`: cross-query raw agent-id transitions are disabled; completed queries are aggregated by visible profile signatures and/or visible roster positions.
- `query_internal_profile_conditioned_online_learning`: same as profile-conditioned memory, plus local same-query feedback after earlier steps have occurred.
- `skeptical_profile_only_agent_id_permutation_counterfactual`: all raw id and structural-order shortcuts are disabled except visible profile-signature memory.
- `semantic_skeptical_agent_id_permutation_counterfactual`: the skeptical audit plus visible task/profile semantic matching over the full candidate set.
- `structural_agent_id_permutation_counterfactual`: visible graph/tool order is used while raw agent ids are permuted per file.
- `structural_prior_baseline`: static or hand-written priors are mixed in and should not be reported as strict online-learning accuracy.

## Latest Results

The current benchmark root is `results/research`, with 20 `coding` queries and 20 `research` queries. The pooled suite evaluates 602 strict predictive steps: 402 from `coding` and 200 from `research`.

Recommended honest pooled structural run:

```powershell
.venv\Scripts\python.exe scripts\benchmark\run_new_log_concurrent_suite.py `
  --root results\research `
  --scenarios coding research `
  --batch-sizes 1 3 4 `
  --policy-mode structural_event_id_permutation `
  --scenario-replay-mode pooled `
  --agent-id-salt pooled_audit_main `
  --query-order-seed pooled_shuffle_main
```

Output files:

- `results/research/final_pooled_structural_event_idperm_shuffle_report.json`
- `results/research/final_pooled_structural_event_idperm_shuffle_timing.jsonl`
- `results/research/final_pooled_structural_event_idperm_shuffle_convergence.svg`

This mode uses only prediction-time visible inputs, but it is not pure online-memory-only: it uses current executing agent, visible graph/tool-schema candidate order, visible profile/roster completed-query memory, and query-local feedback after earlier steps have occurred. Agent ids are permuted per file, and `--scenario-replay-mode pooled` uses one shared memory across scenarios, so raw fixed agent numbering and per-scenario memory resets cannot explain the score.

| Batch size | Coding hit@1 | Research hit@1 | Step-micro hit@1 | Scenario-macro hit@1 | Query-macro hit@1 | Mean prediction | P95 prediction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 81.09% | 44.00% | 68.77% | 62.55% | 64.20% | 0.109 ms | 0.232 ms |
| 3 | 80.35% | 44.00% | 68.27% | 62.17% | 63.73% | 0.101 ms | 0.190 ms |
| 4 | 80.10% | 42.00% | 67.44% | 61.05% | 62.41% | 0.098 ms | 0.195 ms |

Label-shuffle negative control for the same run:

| Batch size | Real hit@1 | Shuffled-label hit@1 |
| ---: | ---: | ---: |
| 1 | 68.77% | 20.43% |
| 3 | 68.27% | 20.43% |
| 4 | 67.44% | 20.60% |

The negative control shuffles target sets after predictions are made. Its much lower hit@1 is a sanity check that the metric is not trivially high regardless of labels.

Pooled policy comparison:

| Mode | Batch size | Coding hit@1 | Research hit@1 | Step-micro hit@1 | Scenario-macro hit@1 | Mean prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `skeptical_profile_event_id_permutation` | 1 | 81.09% | 28.00% | 63.46% | 54.55% | 0.075 ms |
| `skeptical_profile_event_id_permutation` | 3 | 79.60% | 28.00% | 62.46% | 53.80% | 0.076 ms |
| `skeptical_profile_event_id_permutation` | 4 | 79.60% | 28.00% | 62.46% | 53.80% | 0.078 ms |
| `semantic_skeptical_profile_event_id_permutation` | 1 | 80.60% | 41.50% | 67.61% | 61.05% | 0.467 ms |
| `semantic_skeptical_profile_event_id_permutation` | 3 | 79.35% | 41.50% | 66.78% | 60.43% | 0.660 ms |
| `semantic_skeptical_profile_event_id_permutation` | 4 | 79.35% | 41.50% | 66.78% | 60.43% | 0.633 ms |
| `structural_event_id_permutation` | 1 | 81.09% | 44.00% | 68.77% | 62.55% | 0.109 ms |
| `structural_event_id_permutation` | 3 | 80.35% | 44.00% | 68.27% | 62.17% | 0.101 ms |
| `structural_event_id_permutation` | 4 | 80.10% | 42.00% | 67.44% | 61.05% | 0.098 ms |

Interpretation:

- Query-internal feedback is useful and legitimate for an executing query: the current step is predicted first, then only after the true next agent occurs does local memory update for later steps.
- Visible graph/tool order is a legitimate system-structure signal only if you accept it as available to the speculative scheduler. It adds about 1.2pp over the corrected semantic skeptical mode on batch size 1.
- The corrected semantic skeptical mode now scores all visible agents, not only graph outgoing agents. The report checks `all_agent_profile_scope_mismatch_count = 0`.
- Profile signatures are lexical token signatures. They no longer collapse coding profiles into hand-written tags such as `role:implementation`; hand-written role workflow logic remains available only in explicit structural-prior modes where it is disclosed.
- `coding` is much easier than `research`. The current system does not honestly reach 80% hit@1 on every scenario.
- The reports include `aggregate.leakage_audit.verdict = no_known_label_leakage_detected`; scheduler recovery events are excluded from metrics.

## Robustness Audits

Pooled multi-seed audit with both agent ids and query order shuffled:

- Summary file: `results/research/final_pooled_multiseed_idperm_shuffle_summary.json`
- Seeds: 6
- Batch size: 1
- Scenarios: `coding`, `research`

| Policy | Step-micro hit@1 mean | Min | Max | Std | Scenario-macro mean | Coding mean | Research mean | Label-shuffle hit@1 | Mean prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `skeptical_profile_event_id_permutation` | 63.70% | 62.29% | 65.28% | 1.00pp | 54.94% | 81.05% | 28.83% | 19.02% | 0.084 ms |
| `semantic_skeptical_profile_event_id_permutation` | 68.02% | 67.77% | 68.27% | 0.19pp | 61.36% | 81.22% | 41.50% | 18.13% | 0.675 ms |
| `structural_event_id_permutation` | 68.60% | 67.28% | 69.10% | 0.66pp | 62.32% | 81.05% | 43.58% | 18.91% | 0.129 ms |

Important anti-cheat conclusions:

- The old raw `agent_id -> agent_id` transition memory was not label leakage, but it depended on fixed cross-query agent numbering. The current headline audits use per-file id permutation.
- The previous separate-scenario reports are easier because each scenario gets its own memory reset. Use pooled results for deployment-facing claims.
- If visible graph/tool order is not acceptable, report `semantic_skeptical_profile_event_id_permutation` or the stricter `skeptical_profile_event_id_permutation`.
- If query-internal feedback is not available, run a query-level mode such as `strict_profile_id_permutation`; it should be reported separately because it answers a different question.

## Request-Level Online Convergence

For pooled `structural_event_id_permutation`:

| Batch size | Final hit@1 | Stable within +/-5pp after | Sustained >=70% after | Sustained >=80% after |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 68.77% | 23 queries | none | none |
| 3 | 68.27% | 24 queries | none | none |
| 4 | 67.44% | 24 queries | none | none |

The convergence rule is sustained stability: from request N onward, every later cumulative hit@1 must stay within the stated tolerance of the final cumulative hit@1. A transient early checkpoint is not treated as convergence.

## Accuracy And Cost Tradeoff

Strict hit@1 is single-branch accuracy. Speculative coverage counts whether the true next agent is inside the selected branch set. It is useful only if losing branches can be cancelled or are cheap to run.

For pooled `structural_event_id_permutation`:

| Batch size | Policy | Coverage | Avg branches | Extra branches |
| ---: | --- | ---: | ---: | ---: |
| 1 | fixed top1 | 68.77% | 1.00 | 0.00 |
| 1 | margin top2 <= 15 | 71.59% | 1.15 | 0.15 |
| 1 | margin top2 <= 80 and top3 <= 5 | 84.05% | 1.89 | 0.89 |
| 3 | fixed top1 | 68.27% | 1.00 | 0.00 |
| 3 | margin top2 <= 15 | 70.93% | 1.17 | 0.17 |
| 4 | fixed top1 | 67.44% | 1.00 | 0.00 |
| 4 | margin top2 <= 15 | 70.76% | 1.17 | 0.17 |

Speedup condition:

```text
net_speedup_requires saved_latency_from_correct_speculation > predictor_overhead + extra_work_cost
```

The current pooled structural predictor is cheap: about 0.10-0.13 ms mean per step in the reported run. The semantic skeptical mode is slower, around 0.47-0.68 ms mean per step, because it computes token overlap against all candidate profiles. The main cost of wider speculation is downstream work, not predictor scoring, because the predictor already computes a top-5 ranking in one pass.

## Achievability Notes

The current code should not be described as an 80-90% strict hit@1 solution. The honest status is:

- Around 69% average strict hit@1 is achievable in pooled replay when visible graph/tool order and query-internal feedback are allowed.
- Around 68% average strict hit@1 is achievable in pooled replay with semantic profile matching but without graph-order, roster-position memory, candidate narrowing, raw cross-query ids, or raw local same-query id memory.
- Around 64% average strict hit@1 is the more skeptical profile-only estimate.
- `coding` reaches about 81% because the workflow structure is stable.
- `research` remains around 42-44% because each query changes the researcher roster and topical profiles, so there is less reusable role identity.
- Top-k speculative coverage can exceed 80%, but that is a branch-set coverage metric, not hit@1.
- Reaching 80%+ hit@1 on every scenario likely requires more prior scenarios, a stronger no-leak semantic router, or a deployment assumption that agent identities/roles are stable across queries.

## Method Rationale

The implemented evaluation logic is guided by:

- Online learning under delayed feedback: every prediction is made before the corresponding label is applied. Concurrent batches share a pre-batch memory snapshot.
- Query-internal online adaptation: once an earlier predicted agent transition has actually happened, the same query may use it for later predictions.
- Prediction with expert advice: `expert_cascade` treats each no-leak score component as an expert and updates expert weights only after feedback arrives.
- Adaptive prediction sets: branch policies trade branch count for coverage under drift.
- Cost-aware cascades: cheap-first routing spends extra scoring only when uncertainty is high.

References:

- Weighted Majority / expert advice: <https://onlineprediction.cs.rhul.ac.uk/index.html?n=Main.WeightedMajorityAlgorithm>
- Adaptive conformal inference under distribution shift: <https://papers.nips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html>
- Selective classification and risk-coverage tradeoff: <https://arxiv.org/abs/1705.08500>
- FrugalML cost-aware prediction APIs: <https://arxiv.org/abs/2006.07512>

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

The root `README.md` is the project documentation entry point. Third-party `vendor/` folders may contain their own upstream READMEs and are not part of this project documentation layout.

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

`scripts.benchmark.run_new_log_cold_start`

Runs the raw-log next-agent evaluator for one scenario folder. It supports `--candidate-scope visible_graph`, `all_agents`, and `all_worker_agents`, records timing, exports audit data, and updates online memory only after scoring.

`scripts.benchmark.run_new_log_concurrent_batches`

Runs active-query batch replay for one scenario folder. It uses a pre-batch cross-query memory snapshot, then applies batch updates after all active queries have been scored.

`scripts.benchmark.run_new_log_concurrent_suite`

Runs multiple scenarios and batch sizes, aggregates step/scenario/query/batch metrics, validates metric integrity, reports target-set recall, adaptive branch coverage, convergence curves, and wave latency. Use `--scenario-replay-mode pooled` when you want one shared online memory across scenarios.

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

Use the missing-model sentinel to avoid HuggingFace HEAD retry warnings for intentionally absent test models:

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
- Treat context snapshots carefully. The current event's `agents[*].context` can be used only as already completed history when `--include-visible-agent-context` is explicitly enabled.
- For `next_agent`, current `agent_id` is allowed because the executing agent is known.
- For raw logs without an explicit graph, inferred graph edges from visible prompt/tool schema are allowed and must be reported as inferred.
- Do not report hand-written role workflow, research schedule/meta, profile similarity, visible context similarity, or graph-order priors as online-learning-only accuracy. Use `--policy-mode strict_online` for that number.
- Report strict hit@1, top-k speculative coverage, and oracle diagnostics as separate quantities.

## Maintenance

```powershell
.venv\Scripts\python.exe tests\test_predictdesign.py
.venv\Scripts\python.exe -m compileall predictdesign scripts tests
.venv\Scripts\python.exe scripts\cleanup_workspace.py --execute
```

Keep caches, temporary reports, and checkpoints out of source directories. Put generated reports and checkpoints under `results/`, local datasets under `data/`, and third-party benchmarks under `vendor/`.
