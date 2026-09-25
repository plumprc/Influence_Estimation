#!/bin/bash
# FashionMNIST controlled experiments used by the paper.
#
# Pipeline: exp1 -> exp2 -> exp3 -> exp4 -> aggregate -> figures
#
# Usage:
#   bash shell/run_fmnist_exp.sh
#   EXPS="1 3 4" bash shell/run_fmnist_exp.sh
#   SEEDS="0 1 2" bash shell/run_fmnist_exp.sh
#   FORCE=1 bash shell/run_fmnist_exp.sh  # rerun complete outputs

set -u

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_BASE="$CODE_DIR/outputs/fashion_mnist"

EXPS=${EXPS-"1 2 3 4"}
SEEDS=${SEEDS-"0 1 2"}
ETAS=${ETAS-"0.01 0.02 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0"}
ALPHAS=${ALPHAS-"1e-5 1e-4 1e-3 1e-2 0.1"}
DEVICE=${DEVICE-"cuda"}
FORCE=${FORCE-"0"}

cd "$CODE_DIR"

failed=""

data_ready() {
  local raw_dir="$CODE_DIR/datasets/FashionMNIST/raw"
  for file in \
    train-images-idx3-ubyte train-labels-idx1-ubyte \
    t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    [ -s "$raw_dir/$file" ] || return 1
  done
}

if ! data_ready; then
  echo "FashionMNIST IDX files are missing under datasets/FashionMNIST/raw/"
  exit 1
fi

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

exp1() {
  for seed in $SEEDS; do
    output_dir="$OUTPUT_BASE/exp1_behavior/seed_$seed"
    run_seed "exp1 behavior seed=$seed" "$output_dir" \
      python experiments/run_fmnist_one_step.py \
        --max-train 20000 \
        --max-test 2000 \
        --num-queries 1000 \
        --num-candidates 1000 \
        --step-size 0.1 \
        --seed "$seed" \
        --device "$DEVICE" \
        --output-dir "$output_dir"
  done
}

exp2() {
  for seed in $SEEDS; do
    output_dir="$OUTPUT_BASE/exp2_transition/seed_$seed"
    run_seed "exp2 transition seed=$seed" "$output_dir" \
      python experiments/run_fmnist_transition.py \
        --max-train 20000 \
        --max-test 2000 \
        --num-queries 1000 \
        --num-candidates 1000 \
        --step-size 0.1 \
        --num-steps 5 \
        --damping 0.01 \
        --behavior negative_loss \
        --seed "$seed" \
        --device "$DEVICE" \
        --output-dir "$output_dir"
  done
}

exp3() {
  for eta in $ETAS; do
    for seed in $SEEDS; do
      output_dir="$OUTPUT_BASE/exp3_stepsize/eta_${eta}/seed_${seed}"
      run_seed "exp3 eta=$eta seed=$seed" "$output_dir" \
        python experiments/run_fmnist_one_step.py \
          --max-train 20000 \
          --max-test 2000 \
          --num-queries 1000 \
          --num-candidates 1000 \
          --step-size "$eta" \
          --seed "$seed" \
          --device "$DEVICE" \
          --output-dir "$output_dir"
    done
  done
}

exp4() {
  for seed in $SEEDS; do
    output_dir="$OUTPUT_BASE/exp4_perturbation_scale/seed_$seed"
    run_seed "exp4 perturbation seed=$seed" "$output_dir" \
      python experiments/run_fmnist_perturbation.py \
        --max-train 20000 \
        --max-test 2000 \
        --num-queries 1000 \
        --num-candidates 1000 \
        --alphas $ALPHAS \
        --behavior negative_loss \
        --solver updated-hessian \
        --newton-tol 1e-12 \
        --seed "$seed" \
        --device "$DEVICE" \
        --output-dir "$output_dir"
  done
}

for exp in $EXPS; do
  case "$exp" in
    1) exp1 ;;
    2) exp2 ;;
    3) exp3 ;;
    4) exp4 ;;
    *) echo "Unknown experiment '$exp' (expected 1, 2, 3, or 4); skipping" ;;
  esac
done

if [ -z "$failed" ]; then
  echo "Aggregating FashionMNIST results..."
  if python experiments/aggregate_fmnist_results.py; then
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
  if python plot/plot_fmnist.py; then
    echo "Figures completed"
  else
    echo "Figure generation FAILED"
    failed="$failed figures"
  fi
else
  echo "Skipping figures because aggregation or experiments failed."
fi

if [ -n "$failed" ]; then
  echo "FashionMNIST pipeline finished with failures:$failed"
  exit 1
fi

echo "FashionMNIST pipeline completed."
