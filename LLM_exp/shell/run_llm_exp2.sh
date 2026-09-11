#!/bin/bash
# Full-scale ScienceQA conditional-backdoor attribution experiment.

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
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
MODEL_DIR=${MODEL_DIR:-$(basename "$MODEL" | tr '-' '_')}
SOURCE="datasets/scienceqa/source.json"
DATASET_DIR="datasets/scienceqa/exp2"
RUN_ROOT="outputs/${MODEL_DIR}/scienceqa_exp2"
EPOCHS=${EPOCHS:-2}
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-4}
REPRESENTATION_BATCH_SIZE=${REPRESENTATION_BATCH_SIZE:-8}
EVALUATION_BATCH_SIZE=${EVALUATION_BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-320}
read -r -a CHECKPOINT_FRACTIONS <<< "${CHECKPOINT_FRACTIONS:-0.33 0.66}"
PROJECTION_DIMENSION=${PROJECTION_DIMENSION:-8192}
PROJECTION_SEED=${PROJECTION_SEED:-0}
TRAK_RIDGE=${TRAK_RIDGE:-0.01}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

local_args=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  local_args=(--local-files-only)
fi

if [[ ! -s "$SOURCE" ]]; then
  python experiments/prepare_data.py source --output "$SOURCE" "${local_args[@]}"
fi

dataset_files=(
  "$DATASET_DIR/train.json"
  "$DATASET_DIR/test_clean_activating.json"
  "$DATASET_DIR/test_clean_nonactivating.json"
  "$DATASET_DIR/test_triggered_activating.json"
  "$DATASET_DIR/test_triggered_nonactivating.json"
  "$DATASET_DIR/manifest.json"
)
all_files_present() {
  local path
  for path in "${dataset_files[@]}"; do
    [[ -s "$path" ]] || return 1
  done
  return 0
}

if [[ -n "$(find "$DATASET_DIR" -maxdepth 1 -type f -name '*.json' -print -quit 2>/dev/null)" ]] &&
   ! all_files_present; then
  echo "Incomplete exp2 dataset: $DATASET_DIR" >&2
  exit 1
fi

if ! all_files_present; then
  python experiments/prepare_data.py exp2 \
    --source "$SOURCE" \
    --output-dir "$DATASET_DIR"
fi

TARGET_DATA="$DATASET_DIR/test_triggered_activating.json"
SCORE_RUNS=()
EVALUATION_RUNS=()

mkdir -p logs "$RUN_ROOT"

for seed in "${SEEDS[@]}"; do
  output_root="$RUN_ROOT/seed${seed}"

  if [[ "$SKIP_COMPLETED" == "1" && -s "$output_root/adapter/summary.json" ]]; then
    echo "Skipping completed adapter for seed $seed: $output_root/adapter"
  else
    python experiments/train.py \
      --model-source "$MODEL_SOURCE" \
      --train-data "$DATASET_DIR/train.json" \
      --output-dir "$output_root/adapter" \
      --max-length "$MAX_LENGTH" \
      --epochs "$EPOCHS" \
      --batch-size "$TRAIN_BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --weight-decay 0 \
      --warmup-ratio 0.03 \
      --lora-rank 16 \
      --lora-alpha 32 \
      --lora-dropout "$LORA_DROPOUT" \
      --checkpoint-fractions "${CHECKPOINT_FRACTIONS[@]}" \
      --seed "$seed" \
      --device cuda \
      "${local_args[@]}"
  fi

  if [[ "$SKIP_COMPLETED" == "1" && -s "$output_root/score/summary.json" ]]; then
    echo "Skipping completed influence scores for seed $seed: $output_root/score"
  else
    python experiments/score.py \
      --model-source "$MODEL_SOURCE" \
      --adapter "$output_root/adapter" \
      --train-data "$DATASET_DIR/train.json" \
      --validation-data "$TARGET_DATA" \
      --output-dir "$output_root/score" \
      --max-length "$MAX_LENGTH" \
      --validation-batch-size "$VALIDATION_BATCH_SIZE" \
      --representation-batch-size "$REPRESENTATION_BATCH_SIZE" \
      --projection-dimension "$PROJECTION_DIMENSION" \
      --projection-seed "$PROJECTION_SEED" \
      --trak-ridge "$TRAK_RIDGE" \
      --checkpoint-fractions "${CHECKPOINT_FRACTIONS[@]}" \
      --score-direction promotion \
      --seed "$seed" \
      --device cuda \
      "${local_args[@]}"
  fi
  SCORE_RUNS+=("$output_root/score")

  if [[ "$SKIP_COMPLETED" == "1" && -s "$output_root/evaluation/summary.json" ]]; then
    echo "Skipping completed backdoor evaluation for seed $seed: $output_root/evaluation"
  else
    python experiments/evaluate.py \
      --model-source "$MODEL_SOURCE" \
      --adapter "$output_root/adapter" \
      --test-dir "$DATASET_DIR" \
      --output-dir "$output_root/evaluation" \
      --batch-size "$EVALUATION_BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --device cuda \
      "${local_args[@]}"
  fi
  EVALUATION_RUNS+=("$output_root/evaluation")
done

python experiments/aggregate.py --mode score \
  --runs "${SCORE_RUNS[@]}" \
  --output "$RUN_ROOT/score_aggregate.json"

python experiments/aggregate.py --mode evaluation \
  --runs "${EVALUATION_RUNS[@]}" \
  --output "$RUN_ROOT/evaluation_aggregate.json"

echo "LLM exp2 completed."
