#!/usr/bin/env bash
set -euo pipefail

ROOT="/data0/fanjiarun/PredictDesign"
RESULTS_ROOT="${ROOT}/results"
STAMP="$(date +%Y%m%d_%H%M%S)"
SOURCE_GNN_ROOT="${SOURCE_GNN_ROOT:-${RESULTS_ROOT}/gnn_20260409_185050}"
LLM32_RUN_ROOT="${LLM32_RUN_ROOT_OVERRIDE:-${RESULTS_ROOT}/llm_qwen3_32b_${STAMP}}"
LLM8_RUN_ROOT="${LLM8_RUN_ROOT_OVERRIDE:-${RESULTS_ROOT}/llm_qwen3_8b_${STAMP}}"
GROUP_MANIFEST_PATH="${GROUP_MANIFEST_OVERRIDE:-${RESULTS_ROOT}/run_group_llm_eval_${STAMP}.json}"

RESEARCH_LIMIT="${RESEARCH_LIMIT:-10}"
RESEARCH_OUTPUT_DIR="${SOURCE_GNN_ROOT}/fullfidelity_research${RESEARCH_LIMIT}/research_outputs"
WEREWOLF_OUTPUT_DIR="${SOURCE_GNN_ROOT}/fullfidelity_werewolf10/werewolf_outputs"

if [ ! -d "${RESEARCH_OUTPUT_DIR}" ]; then
  echo "Missing research outputs: ${RESEARCH_OUTPUT_DIR}" >&2
  exit 1
fi
if [ ! -d "${WEREWOLF_OUTPUT_DIR}" ]; then
  echo "Missing werewolf outputs: ${WEREWOLF_OUTPUT_DIR}" >&2
  exit 1
fi

CONTEXT_DIM="${CONTEXT_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-40}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
DEVICE="${DEVICE:-cuda}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.siliconflow.cn/v1}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.1}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-1024}"
LLM_TIMEOUT="${LLM_TIMEOUT:-300}"
LLM_API_KEY="${LLM_API_KEY:-}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-20000}"

mapfile -t GPU_ORDER < <(
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v min_free="${GPU_MIN_FREE_MIB}" '{
        gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3);
        free=$2-$3;
        if (free >= min_free) print $1","free;
      }' \
    | sort -t',' -k2,2nr \
    | cut -d',' -f1
)

if [ "${#GPU_ORDER[@]}" -lt 2 ]; then
  mapfile -t GPU_ORDER < <(
    nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
      | awk -F',' '{
          gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3);
          free=$2-$3;
          print $1","free;
        }' \
      | sort -t',' -k2,2nr \
      | cut -d',' -f1
  )
fi

GPU_LLM32="${GPU_LLM32:-${GPU_ORDER[0]:-0}}"
GPU_LLM8="${GPU_LLM8:-${GPU_ORDER[1]:-${GPU_LLM32}}}"

launch_variant() {
  local run_root="$1"
  local session_tag="$2"
  local gpu_id="$3"
  local llm_model="$4"
  /usr/bin/bash -lc \
    "GPU_RESEARCH=${gpu_id} \
GPU_WEREWOLF=${gpu_id} \
CONTEXT_DIM=${CONTEXT_DIM} \
HIDDEN_DIM=${HIDDEN_DIM} \
TRAIN_EPOCHS=${TRAIN_EPOCHS} \
LEARNING_RATE=${LEARNING_RATE} \
TRAIN_FRACTION=${TRAIN_FRACTION} \
DEVICE=${DEVICE} \
RESEARCH_LIMIT=${RESEARCH_LIMIT} \
GNN_TYPES='llm_api' \
LLM_BASE_URL='${LLM_BASE_URL}' \
LLM_MODEL='${llm_model}' \
LLM_TEMPERATURE=${LLM_TEMPERATURE} \
LLM_MAX_TOKENS=${LLM_MAX_TOKENS} \
LLM_TIMEOUT=${LLM_TIMEOUT} \
LLM_API_KEY='${LLM_API_KEY}' \
EXISTING_RESEARCH_OUTPUT_DIR='${RESEARCH_OUTPUT_DIR}' \
EXISTING_WEREWOLF_OUTPUT_DIR='${WEREWOLF_OUTPUT_DIR}' \
RUN_ROOT_OVERRIDE='${run_root}' \
SESSION_TAG='${session_tag}' \
bash /data0/fanjiarun/PredictDesign/scripts/launch_persistent_full_runs.sh"
}

launch_variant "${LLM32_RUN_ROOT}" "llm_qwen3_32b_${STAMP}" "${GPU_LLM32}" "Qwen/Qwen3-32B"
launch_variant "${LLM8_RUN_ROOT}" "llm_qwen3_8b_${STAMP}" "${GPU_LLM8}" "Qwen/Qwen3-8B"

cat > "${GROUP_MANIFEST_PATH}" <<EOF
{
  "created_at": "$(date --iso-8601=seconds)",
  "source_gnn_root": "${SOURCE_GNN_ROOT}",
  "group_manifest": "${GROUP_MANIFEST_PATH}",
  "gpu_assignment": {
    "llm_qwen3_32b": "${GPU_LLM32}",
    "llm_qwen3_8b": "${GPU_LLM8}"
  },
  "variants": {
    "llm_qwen3_32b": "${LLM32_RUN_ROOT}",
    "llm_qwen3_8b": "${LLM8_RUN_ROOT}"
  }
}
EOF

echo "${LLM32_RUN_ROOT}/run_manifest.json"
echo "${LLM8_RUN_ROOT}/run_manifest.json"
echo "${GROUP_MANIFEST_PATH}"
