#!/bin/bash
# Formal ScienceQA Stage 1 response-corruption experiment.

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
DATASET_DIR="datasets/scienceqa/exp1"
RUN_ROOT="outputs/${MODEL_DIR}/scienceqa_exp1"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-4}
REPRESENTATION_BATCH_SIZE=${REPRESENTATION_BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-1024}
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

train_data="$DATASET_DIR/train.json"
validation_data="$DATASET_DIR/validation.json"
if [[ ! -s "$train_data" || ! -s "$validation_data" ]]; then
  if [[ -e "$train_data" || -e "$validation_data" ]]; then
    echo "Incomplete exp1 dataset: $DATASET_DIR" >&2
    exit 1
  fi
  python experiments/prepare_data.py exp1 \
    --source "$SOURCE" \
    --output-dir "$DATASET_DIR"
fi

mkdir -p logs "$RUN_ROOT"
SCORE_RUNS=()

for seed in "${SEEDS[@]}"; do
  output_root="$RUN_ROOT/seed${seed}"

  if [[ "$SKIP_COMPLETED" == "1" && -s "$output_root/adapter/summary.json" ]]; then
    echo "Skipping completed adapter for seed $seed: $output_root/adapter"
  else
    python experiments/train.py \
      --model-source "$MODEL_SOURCE" \
      --train-data "$train_data" \
      --output-dir "$output_root/adapter" \
      --max-length "$MAX_LENGTH" \
      --epochs 1 \
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
      --validation-data "$validation_data" \
      --output-dir "$output_root/score" \
      --max-length "$MAX_LENGTH" \
      --validation-batch-size "$VALIDATION_BATCH_SIZE" \
      --representation-batch-size "$REPRESENTATION_BATCH_SIZE" \
      --projection-dimension "$PROJECTION_DIMENSION" \
      --projection-seed "$PROJECTION_SEED" \
      --trak-ridge "$TRAK_RIDGE" \
      --checkpoint-fractions "${CHECKPOINT_FRACTIONS[@]}" \
      --score-direction harm \
      --seed "$seed" \
      --device cuda \
      "${local_args[@]}"
  fi
  SCORE_RUNS+=("$output_root/score")
done

python experiments/aggregate.py --mode score \
  --runs "${SCORE_RUNS[@]}" \
  --output "$RUN_ROOT/aggregate.json"

echo "LLM exp1 completed."
