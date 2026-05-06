#!/usr/bin/env bash
set -euo pipefail

ROOT="/data0/fanjiarun/PredictDesign"
PYTHON_BIN="/home/gujing/ygr/Design/bin/python"
MODEL_PATH="/data0/fanjiarun/models/Qwen3.5-9B"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-${ROOT}/results/persistent_full_run_${STAMP}}"
SESSION_TAG="${SESSION_TAG:-${STAMP}}"

GPU_RESEARCH="${GPU_RESEARCH:-4}"
GPU_WEREWOLF="${GPU_WEREWOLF:-6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
CONTEXT_DIM="${CONTEXT_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
EXISTING_WEREWOLF_OUTPUT_DIR="${EXISTING_WEREWOLF_OUTPUT_DIR:-}"
EXISTING_RESEARCH_OUTPUT_DIR="${EXISTING_RESEARCH_OUTPUT_DIR:-}"
DEVICE="${DEVICE:-cuda}"
RESEARCH_MAX_ITERATIONS="${RESEARCH_MAX_ITERATIONS:-3}"
RESEARCH_LIMIT="${RESEARCH_LIMIT:-10}"
WEREWOLF_GAMES="${WEREWOLF_GAMES:-10}"
GNN_TYPES="${GNN_TYPES:-gcn graphsage gat}"
LLM_API_KEY="${LLM_API_KEY:-}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.siliconflow.cn/v1}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-32B}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.1}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-4096}"
LLM_TIMEOUT="${LLM_TIMEOUT:-600.0}"

LLM_API_KEY_ARG=""
if [ -n "${LLM_API_KEY}" ]; then
  LLM_API_KEY_ARG="--llm-api-key ${LLM_API_KEY}"
fi

RESEARCH_DIR="${RUN_ROOT}/fullfidelity_research${RESEARCH_LIMIT}"
WEREWOLF_DIR="${RUN_ROOT}/fullfidelity_werewolf${WEREWOLF_GAMES}"
LOG_DIR="${RUN_ROOT}/logs"
MONITOR_STATUS="${RUN_ROOT}/final_accuracy_monitor_status.json"
MONITOR_TABLES="${RUN_ROOT}/final_accuracy_tables.md"
MONITOR_LOG="${LOG_DIR}/monitor.log"
RESEARCH_LOG="${LOG_DIR}/research.log"
WEREWOLF_LOG="${LOG_DIR}/werewolf.log"

RESEARCH_SESSION="pd_rs_${SESSION_TAG}"
WEREWOLF_SESSION="pd_ww_${SESSION_TAG}"
MONITOR_SESSION="pd_mn_${SESSION_TAG}"

mkdir -p "${RESEARCH_DIR}" "${WEREWOLF_DIR}" "${LOG_DIR}"

if [ -n "${EXISTING_RESEARCH_OUTPUT_DIR}" ]; then
  tmux new-session -d -s "${RESEARCH_SESSION}" "cd /data0/fanjiarun && PYTHONPATH=${ROOT} CUDA_VISIBLE_DEVICES=${GPU_RESEARCH} ${PYTHON_BIN} /data0/fanjiarun/PredictDesign/scripts/run_local_qwen_multiagentbench.py --python-bin ${PYTHON_BIN} --model-path ${MODEL_PATH} --results-dir ${RESEARCH_DIR} --skip-research-run --skip-werewolf-run --existing-research-output-dir ${EXISTING_RESEARCH_OUTPUT_DIR} --device ${DEVICE} --context-dim ${CONTEXT_DIM} --hidden-dim ${HIDDEN_DIM} --train-epochs ${TRAIN_EPOCHS} --learning-rate ${LEARNING_RATE} --train-fraction ${TRAIN_FRACTION} --gnn-types ${GNN_TYPES} ${LLM_API_KEY_ARG} --llm-base-url ${LLM_BASE_URL} --llm-model '${LLM_MODEL}' --llm-temperature ${LLM_TEMPERATURE} --llm-max-tokens ${LLM_MAX_TOKENS} --llm-timeout ${LLM_TIMEOUT} 2>&1 | tee -a ${RESEARCH_LOG}"
else
  tmux new-session -d -s "${RESEARCH_SESSION}" "cd /data0/fanjiarun && PYTHONPATH=${ROOT} CUDA_VISIBLE_DEVICES=${GPU_RESEARCH} ${PYTHON_BIN} /data0/fanjiarun/PredictDesign/scripts/run_local_qwen_multiagentbench.py --python-bin ${PYTHON_BIN} --model-path ${MODEL_PATH} --results-dir ${RESEARCH_DIR} --research-jsonl /data0/fanjiarun/MultiAgentBench/multiagentbench/research/research_main.jsonl --research-limit ${RESEARCH_LIMIT} --research-max-iterations ${RESEARCH_MAX_ITERATIONS} --skip-werewolf-run --full-fidelity --in-process --device ${DEVICE} --context-dim ${CONTEXT_DIM} --hidden-dim ${HIDDEN_DIM} --train-epochs ${TRAIN_EPOCHS} --learning-rate ${LEARNING_RATE} --train-fraction ${TRAIN_FRACTION} --gnn-types ${GNN_TYPES} ${LLM_API_KEY_ARG} --llm-base-url ${LLM_BASE_URL} --llm-model '${LLM_MODEL}' --llm-temperature ${LLM_TEMPERATURE} --llm-max-tokens ${LLM_MAX_TOKENS} --llm-timeout ${LLM_TIMEOUT} 2>&1 | tee -a ${RESEARCH_LOG}"
fi

if [ -n "${EXISTING_WEREWOLF_OUTPUT_DIR}" ]; then
  tmux new-session -d -s "${WEREWOLF_SESSION}" "cd /data0/fanjiarun && PYTHONPATH=${ROOT} CUDA_VISIBLE_DEVICES=${GPU_WEREWOLF} ${PYTHON_BIN} /data0/fanjiarun/PredictDesign/scripts/run_local_qwen_multiagentbench.py --python-bin ${PYTHON_BIN} --model-path ${MODEL_PATH} --results-dir ${WEREWOLF_DIR} --skip-research-run --skip-werewolf-run --existing-werewolf-output-dir ${EXISTING_WEREWOLF_OUTPUT_DIR} --device ${DEVICE} --context-dim ${CONTEXT_DIM} --hidden-dim ${HIDDEN_DIM} --train-epochs ${TRAIN_EPOCHS} --learning-rate ${LEARNING_RATE} --train-fraction ${TRAIN_FRACTION} --gnn-types ${GNN_TYPES} ${LLM_API_KEY_ARG} --llm-base-url ${LLM_BASE_URL} --llm-model '${LLM_MODEL}' --llm-temperature ${LLM_TEMPERATURE} --llm-max-tokens ${LLM_MAX_TOKENS} --llm-timeout ${LLM_TIMEOUT} 2>&1 | tee -a ${WEREWOLF_LOG}"
else
  tmux new-session -d -s "${WEREWOLF_SESSION}" "cd /data0/fanjiarun && PYTHONPATH=${ROOT} CUDA_VISIBLE_DEVICES=${GPU_WEREWOLF} ${PYTHON_BIN} /data0/fanjiarun/PredictDesign/scripts/run_local_qwen_multiagentbench.py --python-bin ${PYTHON_BIN} --model-path ${MODEL_PATH} --results-dir ${WEREWOLF_DIR} --skip-research-run --full-fidelity --in-process --werewolf-games ${WEREWOLF_GAMES} --device ${DEVICE} --context-dim ${CONTEXT_DIM} --hidden-dim ${HIDDEN_DIM} --train-epochs ${TRAIN_EPOCHS} --learning-rate ${LEARNING_RATE} --train-fraction ${TRAIN_FRACTION} --gnn-types ${GNN_TYPES} ${LLM_API_KEY_ARG} --llm-base-url ${LLM_BASE_URL} --llm-model '${LLM_MODEL}' --llm-temperature ${LLM_TEMPERATURE} --llm-max-tokens ${LLM_MAX_TOKENS} --llm-timeout ${LLM_TIMEOUT} 2>&1 | tee -a ${WEREWOLF_LOG}"
fi

tmux new-session -d -s "${MONITOR_SESSION}" "cd /data0/fanjiarun && ${PYTHON_BIN} /data0/fanjiarun/PredictDesign/scripts/monitor_full_runs.py --research-report ${RESEARCH_DIR}/multiagentbench_accuracy_report.json --werewolf-report ${WEREWOLF_DIR}/multiagentbench_accuracy_report.json --output-md ${MONITOR_TABLES} --status-json ${MONITOR_STATUS} --log-file ${MONITOR_LOG} --poll-seconds ${POLL_SECONDS} 2>&1 | tee -a ${MONITOR_LOG}"

MANIFEST_PATH="${RUN_ROOT}/run_manifest.json"
LATEST_PATH="${ROOT}/results/latest_persistent_run.json"

cat > "${MANIFEST_PATH}" <<EOF
{
  "created_at": "$(date --iso-8601=seconds)",
  "run_root": "${RUN_ROOT}",
  "python_bin": "${PYTHON_BIN}",
  "model_path": "${MODEL_PATH}",
  "train_epochs": "${TRAIN_EPOCHS}",
  "learning_rate": "${LEARNING_RATE}",
  "train_fraction": "${TRAIN_FRACTION}",
  "context_dim": "${CONTEXT_DIM}",
  "hidden_dim": "${HIDDEN_DIM}",
  "research_max_iterations": "${RESEARCH_MAX_ITERATIONS}",
  "research_limit": "${RESEARCH_LIMIT}",
  "werewolf_games": "${WEREWOLF_GAMES}",
  "gnn_types": "${GNN_TYPES}",
  "llm_base_url": "${LLM_BASE_URL}",
  "llm_model": "${LLM_MODEL}",
  "llm_temperature": "${LLM_TEMPERATURE}",
  "llm_max_tokens": "${LLM_MAX_TOKENS}",
  "llm_timeout": "${LLM_TIMEOUT}",
  "device": "${DEVICE}",
  "research": {
    "gpu": "${GPU_RESEARCH}",
    "session": "${RESEARCH_SESSION}",
    "results_dir": "${RESEARCH_DIR}",
    "log_file": "${RESEARCH_LOG}"
  },
  "werewolf": {
    "gpu": "${GPU_WEREWOLF}",
    "session": "${WEREWOLF_SESSION}",
    "results_dir": "${WEREWOLF_DIR}",
    "log_file": "${WEREWOLF_LOG}"
  },
  "monitor": {
    "session": "${MONITOR_SESSION}",
    "status_json": "${MONITOR_STATUS}",
    "output_md": "${MONITOR_TABLES}",
    "log_file": "${MONITOR_LOG}"
  }
}
EOF

cp "${MANIFEST_PATH}" "${LATEST_PATH}"

echo "${MANIFEST_PATH}"
