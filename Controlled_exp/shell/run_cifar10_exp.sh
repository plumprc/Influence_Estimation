#!/bin/bash
# CIFAR-10 supplementary controlled experiments.
#
# Pipeline: 5a -> 5b -> 5c -> 5d -> aggregate -> figures
#
# Usage:
#   bash shell/run_cifar10_exp.sh
#   EXPS="5b" bash shell/run_cifar10_exp.sh
#   FORCE=1 bash shell/run_cifar10_exp.sh  # rerun complete outputs

set -u

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_BASE="$CODE_DIR/outputs/cifar10"

EXPS=${EXPS-"5a 5b 5c 5d"}
AGG_EXPS=${AGG_EXPS-"5a 5b 5c 5d"}
SEEDS=${SEEDS-"0 1 2"}
SINGLE_SEEDS=${SINGLE_SEEDS-"0"}
ETAS=${ETAS-"0.01 0.05 0.1 0.3 0.5"}
ALPHAS=${ALPHAS-"1e-5 1e-3 0.1"}
DEVICE=${DEVICE-"cuda"}
DTYPE=${DTYPE-"float64"}
FORCE=${FORCE-"0"}

cd "$CODE_DIR"

failed=""

output_complete() {
  local directory="$1"
  [ -s "$directory/scores.npz" ] && [ -s "$directory/summary.json" ]
}

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

exp5a() {
  for seed in $SEEDS; do
    output_dir="$OUTPUT_BASE/exp5a_cifar10_behavior/seed_$seed"
    if [ "$FORCE" != "1" ] && output_complete "$output_dir"; then
      echo "Skipping exp5a behavior seed=$seed (complete output)"
    else
      run_tracked "exp5a behavior seed=$seed" \
        python experiments/run_cifar10_behavior.py \
          --max-train 10000 \
          --num-queries 500 \
          --num-candidates 500 \
          --max-epochs 60 \
          --step-size 0.1 \
          --seed "$seed" \
          --device "$DEVICE" \
          --dtype "$DTYPE" \
          --output-dir "$output_dir"
    fi
  done
}

exp5b() {
  for seed in $SINGLE_SEEDS; do
    output_dir="$OUTPUT_BASE/exp5b_cifar10_perturbation_axis/seed_$seed"
    if [ "$FORCE" != "1" ] && output_complete "$output_dir"; then
      echo "Skipping exp5b perturbation seed=$seed (complete output)"
    else
      run_tracked "exp5b perturbation seed=$seed" \
        python experiments/run_cifar10_perturbation_axis.py \
          --max-train 5000 \
          --num-queries 200 \
          --num-candidates 200 \
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
    fi
  done
}

exp5c() {
  for seed in $SINGLE_SEEDS; do
    output_dir="$OUTPUT_BASE/exp5c_cifar10_transition_axes/seed_$seed"
    if [ "$FORCE" != "1" ] && output_complete "$output_dir"; then
      echo "Skipping exp5c transition seed=$seed (complete output)"
    else
      run_tracked "exp5c transition seed=$seed" \
        python experiments/run_cifar10_transition_axes.py \
          --max-train 5000 \
          --num-queries 200 \
          --num-candidates 200 \
          --max-epochs 50 \
          --step-size 0.1 \
          --num-steps 5 \
          --damping 0.01 \
          --max-cg-iters 10 \
          --seed "$seed" \
          --device "$DEVICE" \
          --output-dir "$output_dir"
    fi
  done
}

exp5d() {
  for seed in $SEEDS; do
    output_dir="$OUTPUT_BASE/exp5d_cifar10_stepsize/seed_$seed"
    if [ "$FORCE" != "1" ] && output_complete "$output_dir"; then
      echo "Skipping exp5d step size seed=$seed (complete output)"
    else
      run_tracked "exp5d step size seed=$seed" \
        python experiments/run_cifar10_stepsize.py \
          --max-train 5000 \
          --num-queries 300 \
          --num-candidates 300 \
          --max-epochs 50 \
          --step-sizes $ETAS \
          --seed "$seed" \
          --device "$DEVICE" \
          --dtype "$DTYPE" \
          --output-dir "$output_dir"
    fi
  done
}

for exp in $EXPS; do
  case "$exp" in
    5a) exp5a ;;
    5b) exp5b ;;
    5c) exp5c ;;
    5d) exp5d ;;
    *) echo "Unknown experiment '$exp' (expected 5a, 5b, 5c, or 5d); skipping" ;;
  esac
done

if [ -z "$failed" ]; then
  echo "Aggregating CIFAR-10 results..."
  if python experiments/aggregate_cifar10_results.py \
      --exps $AGG_EXPS \
      --seeds $SEEDS \
      --single-seeds $SINGLE_SEEDS; then
    echo "Aggregation completed"
  else
    echo "Aggregation FAILED"
    failed="$failed aggregation"
  fi
else
  echo "Skipping aggregation because one or more experiment runs failed."
fi

if [ -z "$failed" ]; then
  echo "Generating figures..."
  if python plot/plot_cifar10.py; then
    echo "Figures completed"
  else
    echo "Figure generation FAILED"
    failed="$failed figures"
  fi
else
  echo "Skipping figures because aggregation or experiments failed."
fi

if [ -n "$failed" ]; then
  echo "CIFAR-10 pipeline finished with failures:$failed"
  exit 1
fi

echo "CIFAR-10 pipeline completed."
