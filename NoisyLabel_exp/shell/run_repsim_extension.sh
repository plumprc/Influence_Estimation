#!/bin/bash
# Isolated signal-augmented representation similarity experiment.

set -u

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-"$CODE_DIR/datasets/CIFAR10"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$CODE_DIR/outputs/repsim_extension"}
RHO=${RHO:-"0.2"}
SEEDS=${SEEDS:-"0"}
DEVICE=${DEVICE:-"cuda"}
EPOCHS=${EPOCHS:-"40"}
TRUSTED_PER_CLASS=${TRUSTED_PER_CLASS:-"500"}
MAX_CORRUPTED=${MAX_CORRUPTED:-""}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-"256"}
FORCE=${FORCE:-"0"}
ALLOW_PARTIAL=${ALLOW_PARTIAL:-"0"}

cd "$CODE_DIR"

failed=""
for seed in $SEEDS; do
  output_dir="$OUTPUT_ROOT/rho_${RHO}/seed_${seed}"
  if [ "$FORCE" != "1" ] &&
    [ -s "$output_dir/summary.json" ] &&
    [ -s "$output_dir/scores.npz" ] &&
    [ -s "$output_dir/signals.npz" ] &&
    [ -s "$output_dir/checkpoint.pt" ]; then
    echo "Skipping RepSim extension seed=$seed (complete output)"
    continue
  fi

  args=(
    experiments/run_repsim_extension.py
    --data-root "$DATA_ROOT"
    --seed "$seed"
    --noise-seed 0
    --noise-rate "$RHO"
    --trusted-per-class "$TRUSTED_PER_CLASS"
    --epochs "$EPOCHS"
    --score-batch-size "$SCORE_BATCH_SIZE"
    --output-dir "$output_dir"
    --device "$DEVICE"
  )
  if [ -n "$MAX_CORRUPTED" ]; then
    args+=(--max-corrupted "$MAX_CORRUPTED")
  fi

  echo "Running RepSim extension seed=$seed..."
  if python "${args[@]}"; then
    echo "RepSim extension seed=$seed completed"
  else
    echo "RepSim extension seed=$seed FAILED"
    failed="$failed seed-$seed"
  fi
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
  --rho "$RHO" \
  --seeds $SEEDS \
  "${partial_args[@]}" \
  --output "$OUTPUT_ROOT/aggregate.json"

echo "RepSim extension pipeline completed."
