#!/bin/bash
# CIFAR-10 controlled-experiment pipeline used by the paper.
#
# Usage:
#   bash shell/run_cifar10_exp.sh
#   EXPS="5b" bash shell/run_cifar10_exp.sh
#   EXPS="" bash shell/run_cifar10_exp.sh  # aggregate and plot only
#   FORCE=1 bash shell/run_cifar10_exp.sh  # rerun complete outputs

set -u

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_BASE="$CODE_DIR/outputs/cifar10"

EXPS=${EXPS-"5a 5b 5c 5d"}
AGG_EXPS=${AGG_EXPS-$EXPS}
if [ -n "$EXPS" ]; then
  read -r -a exp_list <<< "$EXPS"
else
  exp_list=()
fi
if [ -n "$AGG_EXPS" ]; then
  read -r -a agg_exp_list <<< "$AGG_EXPS"
else
  agg_exp_list=()
fi
SEEDS=${SEEDS-"0 1 2"}
ETAS=${ETAS-"0.01 0.05 0.1 0.3 0.5"}
ALPHAS=${ALPHAS-"1e-5 1e-3 0.1"}
DEVICE=${DEVICE-"cuda"}
DTYPE=${DTYPE-"float64"}
FORCE=${FORCE-"0"}

MAX_TRAIN=10000
NUM_QUERIES=500
NUM_CANDIDATES=500

cd "$CODE_DIR"
failed=""

data_ready() {
  local data_dir="$CODE_DIR/datasets/CIFAR10"
  local batch_dir="$data_dir/cifar-10-batches-py"
  if [ ! -d "$batch_dir" ] && [ -s "$data_dir/cifar-10-python.tar.gz" ]; then
    tar -xzf "$data_dir/cifar-10-python.tar.gz" -C "$data_dir"
  fi
  for index in 1 2 3 4 5; do
    [ -s "$batch_dir/data_batch_$index" ] || return 1
  done
  [ -s "$batch_dir/test_batch" ]
}

if ! data_ready; then
  echo "CIFAR-10 batches are missing under datasets/CIFAR10/cifar-10-batches-py"
  exit 1
fi

run_tracked() {
  local label="$1"
  shift
  echo "Running $label..."
  if "$@"; then
    echo "$label completed"
  else
    echo "$label FAILED"
    failed="$failed $label"
  fi
}

output_complete() {
  local directory="$1"
  [ -s "$directory/scores.npz" ] && [ -s "$directory/summary.json" ]
}

run_seed() {
  local label="$1"
  local output_dir="$2"
  shift 2
  if [ "$FORCE" != "1" ] && output_complete "$output_dir"; then
    echo "Skipping $label (complete output)"
  else
    run_tracked "$label" "$@"
  fi
}

run_5a() {
  for seed in $SEEDS; do
    output_dir="$OUTPUT_BASE/exp5a_cifar10_behavior/seed_$seed"
    run_seed "exp5a behavior seed=$seed" "$output_dir" \
      python experiments/run_cifar10_behavior.py \
        --max-train "$MAX_TRAIN" \
        --num-queries "$NUM_QUERIES" \
        --num-candidates "$NUM_CANDIDATES" \
        --max-epochs 60 \
        --step-size 0.1 \
        --seed "$seed" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --output-dir "$output_dir"
  done
}

run_5b() {
  for seed in $SEEDS; do
    output_dir="$OUTPUT_BASE/exp5b_cifar10_perturbation_axis/seed_$seed"
    run_seed "exp5b perturbation seed=$seed" "$output_dir" \
      python experiments/run_cifar10_perturbation_axis.py \
        --max-train "$MAX_TRAIN" \
        --num-queries "$NUM_QUERIES" \
        --num-candidates "$NUM_CANDIDATES" \
        --max-epochs 50 \
        --refine-max-iter 100 \
        --reopt-max-iter 20 \
        --alphas $ALPHAS \
        --damping 0.01 \
        --max-cg-iters 10 \
        --behavior negative_loss \
        --seed "$seed" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --output-dir "$output_dir"
  done
}

run_5c() {
  for seed in $SEEDS; do
    output_dir="$OUTPUT_BASE/exp5c_cifar10_transition_axes/seed_$seed"
    run_seed "exp5c transition seed=$seed" "$output_dir" \
      python experiments/run_cifar10_transition_axes.py \
        --max-train "$MAX_TRAIN" \
        --num-queries "$NUM_QUERIES" \
        --num-candidates "$NUM_CANDIDATES" \
        --max-epochs 50 \
        --step-size 0.1 \
        --num-steps 5 \
        --damping 0.01 \
        --max-cg-iters 10 \
        --behavior negative_loss \
        --seed "$seed" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --output-dir "$output_dir"
  done
}

run_5d() {
  for seed in $SEEDS; do
    output_dir="$OUTPUT_BASE/exp5d_cifar10_stepsize/seed_$seed"
    run_seed "exp5d step size seed=$seed" "$output_dir" \
      python experiments/run_cifar10_stepsize.py \
        --max-train "$MAX_TRAIN" \
        --num-queries "$NUM_QUERIES" \
        --num-candidates "$NUM_CANDIDATES" \
        --max-epochs 50 \
        --step-sizes $ETAS \
        --seed "$seed" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --output-dir "$output_dir"
  done
}

for exp in "${exp_list[@]}"; do
  case "$exp" in
    5a) run_5a ;;
    5b) run_5b ;;
    5c) run_5c ;;
    5d) run_5d ;;
    *) echo "Unknown experiment '$exp' (expected 5a, 5b, 5c, or 5d); skipping" ;;
  esac
done

if [ -n "$failed" ]; then
  echo "Skipping aggregation because one or more experiment runs failed."
  echo "CIFAR-10 pipeline finished with failures:$failed"
  exit 1
fi

echo "Aggregating CIFAR-10 results..."
agg_args=()
if [ "${#agg_exp_list[@]}" -gt 0 ]; then
  agg_args+=(--exps "${agg_exp_list[@]}")
fi
if ! python experiments/aggregate_cifar10_results.py \
    "${agg_args[@]}" \
    --merge-existing \
    --seeds $SEEDS \
    --single-seeds $SEEDS; then
  echo "CIFAR-10 pipeline finished with failures: aggregation"
  exit 1
fi

echo "Generating figures..."
if ! { python plot/plot_cifar10.py && python plot/fig_app.py; }; then
  echo "CIFAR-10 pipeline finished with failures: figures"
  exit 1
fi

echo "CIFAR-10 pipeline completed."
