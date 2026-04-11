#!/usr/bin/env bash
set -euo pipefail

ROOT="/data0/fanjiarun/PredictDesign"
RESULTS_ROOT="${ROOT}/results"
STAMP="$(date +%Y%m%d_%H%M%S)"
GNN_RUN_ROOT="${GNN_RUN_ROOT_OVERRIDE:-${RESULTS_ROOT}/gnn}"
LLM32_RUN_ROOT="${LLM32_RUN_ROOT_OVERRIDE:-${RESULTS_ROOT}/llm_qwen3_32b}"
LLM8_RUN_ROOT="${LLM8_RUN_ROOT_OVERRIDE:-${RESULTS_ROOT}/llm_qwen3_8b}"
GROUP_MANIFEST_PATH="${GROUP_MANIFEST_OVERRIDE:-${RESULTS_ROOT}/run_group.json}"
mkdir -p "${RESULTS_ROOT}"

GPU_RESEARCH_GNN="${GPU_RESEARCH_GNN:-}"
GPU_WEREWOLF_GNN="${GPU_WEREWOLF_GNN:-}"
GPU_RESEARCH_LLM32="${GPU_RESEARCH_LLM32:-}"
GPU_WEREWOLF_LLM32="${GPU_WEREWOLF_LLM32:-}"
GPU_RESEARCH_LLM8="${GPU_RESEARCH_LLM8:-}"
GPU_WEREWOLF_LLM8="${GPU_WEREWOLF_LLM8:-}"

CONTEXT_DIM="${CONTEXT_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-40}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
DEVICE="${DEVICE:-cuda}"
RESEARCH_MAX_ITERATIONS="${RESEARCH_MAX_ITERATIONS:-3}"
RESEARCH_LIMIT="${RESEARCH_LIMIT:-10}"
WEREWOLF_GAMES="${WEREWOLF_GAMES:-10}"
LLM_BASE_URL="${LLM_BASE_URL:-https://api.siliconflow.cn/v1}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.1}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-4096}"
LLM_TIMEOUT="${LLM_TIMEOUT:-600.0}"
LLM_API_KEY="${LLM_API_KEY:-}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-12000}"

mapfile -t GPU_ORDER < <(
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v min_free="${GPU_MIN_FREE_MIB}" '{
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        free=$2-$3;
        if (free >= min_free) print $1","free;
      }' \
    | sort -t',' -k2,2nr \
    | cut -d',' -f1
)

if [ "${#GPU_ORDER[@]}" -lt 6 ]; then
  mapfile -t GPU_ORDER < <(
    nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
      | awk -F',' '{
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
          free=$2-$3;
          print $1","free;
        }' \
      | sort -t',' -k2,2nr \
      | cut -d',' -f1
  )
fi

gpu_cursor=0
NEXT_GPU_RESULT="0"
next_gpu() {
  local count="${#GPU_ORDER[@]}"
  if [ "${count}" -eq 0 ]; then
    NEXT_GPU_RESULT="0"
    return
  fi
  NEXT_GPU_RESULT="${GPU_ORDER[$((gpu_cursor % count))]}"
  gpu_cursor=$((gpu_cursor + 1))
}

if [ -z "${GPU_RESEARCH_GNN}" ]; then next_gpu; GPU_RESEARCH_GNN="${NEXT_GPU_RESULT}"; fi
if [ -z "${GPU_WEREWOLF_GNN}" ]; then next_gpu; GPU_WEREWOLF_GNN="${NEXT_GPU_RESULT}"; fi
if [ -z "${GPU_RESEARCH_LLM32}" ]; then next_gpu; GPU_RESEARCH_LLM32="${NEXT_GPU_RESULT}"; fi
if [ -z "${GPU_WEREWOLF_LLM32}" ]; then next_gpu; GPU_WEREWOLF_LLM32="${NEXT_GPU_RESULT}"; fi
if [ -z "${GPU_RESEARCH_LLM8}" ]; then next_gpu; GPU_RESEARCH_LLM8="${NEXT_GPU_RESULT}"; fi
if [ -z "${GPU_WEREWOLF_LLM8}" ]; then next_gpu; GPU_WEREWOLF_LLM8="${NEXT_GPU_RESULT}"; fi

run_variant() {
  local variant_name="$1"
  local gpu_research="$2"
  local gpu_werewolf="$3"
  local gnn_types="$4"
  local llm_model="$5"
  local run_root="$6"
  mkdir -p "${run_root}"

  /usr/bin/bash -lc \
    "GPU_RESEARCH=${gpu_research} \
GPU_WEREWOLF=${gpu_werewolf} \
CONTEXT_DIM=${CONTEXT_DIM} \
HIDDEN_DIM=${HIDDEN_DIM} \
TRAIN_EPOCHS=${TRAIN_EPOCHS} \
LEARNING_RATE=${LEARNING_RATE} \
TRAIN_FRACTION=${TRAIN_FRACTION} \
DEVICE=${DEVICE} \
RESEARCH_MAX_ITERATIONS=${RESEARCH_MAX_ITERATIONS} \
RESEARCH_LIMIT=${RESEARCH_LIMIT} \
WEREWOLF_GAMES=${WEREWOLF_GAMES} \
GNN_TYPES='${gnn_types}' \
LLM_BASE_URL='${LLM_BASE_URL}' \
LLM_MODEL='${llm_model}' \
LLM_TEMPERATURE=${LLM_TEMPERATURE} \
LLM_MAX_TOKENS=${LLM_MAX_TOKENS} \
LLM_TIMEOUT=${LLM_TIMEOUT} \
LLM_API_KEY='${LLM_API_KEY}' \
RUN_ROOT_OVERRIDE='${run_root}' \
SESSION_TAG='${variant_name}_${STAMP}' \
bash /data0/fanjiarun/PredictDesign/scripts/launch_persistent_full_runs.sh"
}

run_variant "gnn" "${GPU_RESEARCH_GNN}" "${GPU_WEREWOLF_GNN}" "gcn graphsage gat" "Qwen/Qwen3-32B" "${GNN_RUN_ROOT}"
run_variant "llm_qwen3_32b" "${GPU_RESEARCH_LLM32}" "${GPU_WEREWOLF_LLM32}" "llm_api" "Qwen/Qwen3-32B" "${LLM32_RUN_ROOT}"
run_variant "llm_qwen3_8b" "${GPU_RESEARCH_LLM8}" "${GPU_WEREWOLF_LLM8}" "llm_api" "Qwen/Qwen3-8B" "${LLM8_RUN_ROOT}"

cat > "${GROUP_MANIFEST_PATH}" <<EOF
{
  "created_at": "$(date --iso-8601=seconds)",
  "group_manifest": "${GROUP_MANIFEST_PATH}",
  "gpu_assignment": {
    "gnn": {
      "research": "${GPU_RESEARCH_GNN}",
      "werewolf": "${GPU_WEREWOLF_GNN}"
    },
    "llm_qwen3_32b": {
      "research": "${GPU_RESEARCH_LLM32}",
      "werewolf": "${GPU_WEREWOLF_LLM32}"
    },
    "llm_qwen3_8b": {
      "research": "${GPU_RESEARCH_LLM8}",
      "werewolf": "${GPU_WEREWOLF_LLM8}"
    }
  },
  "variants": {
    "gnn": "${GNN_RUN_ROOT}",
    "llm_qwen3_32b": "${LLM32_RUN_ROOT}",
    "llm_qwen3_8b": "${LLM8_RUN_ROOT}"
  }
}
EOF

echo "${GROUP_MANIFEST_PATH}"
