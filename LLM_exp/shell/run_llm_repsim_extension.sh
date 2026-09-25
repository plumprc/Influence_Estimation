#!/bin/bash
# Representation-gradient RepSim extension for ScienceQA.
# Runs the requested seed and layer sweep, then aggregates the results.
#
# Single-run overrides (SEED/LAYER_INDEX/OUTPUT_DIR/ADAPTER) are accepted for
# backward compatibility when exactly one seed and one layer are requested.

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
SEEDS=${SEEDS:-${SEED:-"0"}}
LAYERS=${LAYERS:-"24 28 32 35"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/${MODEL_DIR}/scienceqa_exp1_repsim_v2"}
MAX_LENGTH=${MAX_LENGTH:-1024}
BATCH_SIZE=${BATCH_SIZE:-2}
REPRESENTATION_BATCH_SIZE=${REPRESENTATION_BATCH_SIZE:-8}
MAX_TRAIN=${MAX_TRAIN:-""}
MAX_VALIDATION=${MAX_VALIDATION:-""}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
FORCE=${FORCE:-0}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}

TRAIN_DATA=${TRAIN_DATA:-"datasets/scienceqa/exp1/train.json"}
VALIDATION_DATA=${VALIDATION_DATA:-"datasets/scienceqa/exp1/validation.json"}
SCORE_DIRECTION=${SCORE_DIRECTION:-"harm"}

seed_count=$(wc -w <<< "$SEEDS")
layer_count=$(wc -w <<< "$LAYERS")
single_run=false
if [ "$seed_count" -eq 1 ] && [ "$layer_count" -eq 1 ]; then
  single_run=true
fi

local_args=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  local_args=(--local-files-only)
fi

mkdir -p logs
failed=""
for seed in $SEEDS; do
  for layer in $LAYERS; do
    run_dir="$OUTPUT_ROOT/seed${seed}_layer${layer}"
    adapter="outputs/${MODEL_DIR}/scienceqa_exp1/seed${seed}/adapter"
    if [ "$single_run" = true ]; then
      run_dir=${OUTPUT_DIR:-"$run_dir"}
      adapter=${ADAPTER:-"$adapter"}
    fi

    if [ "$FORCE" != "1" ] &&
      [ -s "$run_dir/summary.json" ] &&
      [ -s "$run_dir/scores.npz" ]; then
      echo "Skipping seed $seed, layer $layer (complete output)"
      continue
    fi

    if [[ ! -s "$adapter/adapter_config.json" ]]; then
      echo "Missing adapter for seed $seed: $adapter" >&2
      failed="$failed seed-$seed-layer-$layer"
      continue
    fi

    echo "Running seed $seed, layer $layer..."
    args=(
      experiments/run_repsim_extension.py
      --model-source "$MODEL_SOURCE"
      --adapter "$adapter"
      --train-data "$TRAIN_DATA"
      --validation-data "$VALIDATION_DATA"
      --output-dir "$run_dir"
      --max-length "$MAX_LENGTH"
      --batch-size "$BATCH_SIZE"
      --representation-batch-size "$REPRESENTATION_BATCH_SIZE"
      --layer-index "$layer"
      --score-direction "$SCORE_DIRECTION"
      --device cuda
    )
    if [[ -n "$MAX_TRAIN" ]]; then
      args+=(--max-train "$MAX_TRAIN")
    fi
    if [[ -n "$MAX_VALIDATION" ]]; then
      args+=(--max-validation "$MAX_VALIDATION")
    fi
    args+=("${local_args[@]}")

    if python "${args[@]}"; then
      echo "Seed $seed layer $layer completed"
    else
      echo "Seed $seed layer $layer FAILED"
      failed="$failed seed-$seed-layer-$layer"
    fi
  done
done

if [ -n "$failed" ]; then
  echo "Skipping aggregation because one or more runs failed:$failed"
  exit 1
fi

partial_args=()
if [ "$ALLOW_PARTIAL" == "1" ]; then
  partial_args=(--allow-partial)
fi

python experiments/aggregate_repsim_extension.py \
  --output-root "$OUTPUT_ROOT" \
  --seeds $SEEDS \
  --layers $LAYERS \
  "${partial_args[@]}" \
  --output "$OUTPUT_ROOT/aggregate.json"

echo "LLM RepSim extension pipeline completed."
