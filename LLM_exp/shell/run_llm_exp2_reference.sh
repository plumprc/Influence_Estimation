#!/bin/bash
# Score Stage 2 poison attribution against four validation references.

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
ADAPTER_ROOT="outputs/${MODEL_DIR}/scienceqa_exp2"
RUN_ROOT="outputs/${MODEL_DIR}/scienceqa_exp2_reference"
DATASET_DIR="datasets/scienceqa/exp2_reference"

read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
read -r -a SPECIFICATIONS <<< "${SPECIFICATIONS:-triggered_harmful triggered_clean clean_harmful clean_clean}"
read -r -a CHECKPOINT_FRACTIONS <<< "${CHECKPOINT_FRACTIONS:-0.33 0.66}"
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-4}
REPRESENTATION_BATCH_SIZE=${REPRESENTATION_BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-1024}
PROJECTION_DIMENSION=${PROJECTION_DIMENSION:-8192}
PROJECTION_SEED=${PROJECTION_SEED:-0}
TRAK_RIDGE=${TRAK_RIDGE:-0.01}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

local_args=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  local_args=(--local-files-only)
fi

all_datasets_present() {
  local path
  for specification in "${SPECIFICATIONS[@]}"; do
    path="$DATASET_DIR/${specification}.json"
    [[ -s "$path" ]] || return 1
  done
  [[ -s "$DATASET_DIR/manifest.json" ]]
}

if ! all_datasets_present; then
  python experiments/reference_ablation.py prepare \
    --output-dir "$DATASET_DIR"
fi

mkdir -p logs "$RUN_ROOT"
runs=()
for seed in "${SEEDS[@]}"; do
  adapter="$ADAPTER_ROOT/seed${seed}/adapter"
  if [[ ! -s "$adapter/summary.json" ]]; then
    echo "Missing trained adapter for seed $seed: $adapter" >&2
    exit 1
  fi
  for specification in "${SPECIFICATIONS[@]}"; do
    output="$RUN_ROOT/seed${seed}/$specification"
    if [[ "$SKIP_COMPLETED" == "1" && -s "$output/summary.json" ]]; then
      echo "Skipping completed reference score: $output"
    else
      python experiments/score.py \
        --model-source "$MODEL_SOURCE" \
        --adapter "$adapter" \
        --train-data datasets/scienceqa/exp2/train.json \
        --validation-data "$DATASET_DIR/${specification}.json" \
        --output-dir "$output" \
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
    runs+=("$output")
  done
done

python experiments/reference_ablation.py aggregate \
  --runs "${runs[@]}" \
  --output "$RUN_ROOT/aggregate.json"

echo "LLM exp2 reference ablation completed."
