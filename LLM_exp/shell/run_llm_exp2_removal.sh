#!/bin/bash
# Removal-and-retraining counterfactuals for ScienceQA Stage 2.

set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"

MODEL=${MODEL:-"qwen3-8b"}
case "$MODEL" in
  qwen3-8b)
    DEFAULT_MODEL_SOURCE="Qwen/Qwen3-8B"
    ;;
  *)
    DEFAULT_MODEL_SOURCE="$MODEL"
    ;;
esac
MODEL_SOURCE=${MODEL_SOURCE:-"$DEFAULT_MODEL_SOURCE"}
MODEL_DIR=${MODEL_DIR:-$(basename "$MODEL" | tr '-' '_')}
STAGE2_ROOT=${STAGE2_ROOT:-"outputs/${MODEL_DIR}/scienceqa_exp2"}
DATASET_DIR="datasets/scienceqa/exp2"
RUN_ROOT=${RUN_ROOT:-"outputs/${MODEL_DIR}/scienceqa_exp2_removal"}

read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
read -r -a CONDITIONS <<< "${CONDITIONS:-}"
REMOVAL_BUDGET=${REMOVAL_BUDGET:-500}
CLEAN_ORACLE_SEED=${CLEAN_ORACLE_SEED:-1000}
EPOCHS=${EPOCHS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-320}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}

local_args=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  local_args=(--local-files-only)
fi

condition_args=()
if [[ "${#CONDITIONS[@]}" -gt 0 ]]; then
  condition_args=(--conditions "${CONDITIONS[@]}")
fi

skip_args=()
if [[ "$SKIP_COMPLETED" == "1" ]]; then
  skip_args=(--skip-completed)
fi

partial_args=()
if [[ "$ALLOW_PARTIAL" == "1" ]]; then
  partial_args=(--allow-partial)
fi

mkdir -p logs "$RUN_ROOT"

python experiments/removal.py \
  --model-source "$MODEL_SOURCE" \
  --stage2-root "$STAGE2_ROOT" \
  --train-data "$DATASET_DIR/train.json" \
  --test-dir "$DATASET_DIR" \
  --output-root "$RUN_ROOT" \
  --seeds "${SEEDS[@]}" \
  "${condition_args[@]}" \
  --removal-budget "$REMOVAL_BUDGET" \
  --clean-oracle-seed "$CLEAN_ORACLE_SEED" \
  --epochs "$EPOCHS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --evaluation-batch-size "$EVALUATION_BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --lora-dropout "$LORA_DROPOUT" \
  --max-length "$MAX_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device cuda \
  "${local_args[@]}" \
  "${skip_args[@]}" \
  "${partial_args[@]}"

echo "LLM exp2 removal completed."
