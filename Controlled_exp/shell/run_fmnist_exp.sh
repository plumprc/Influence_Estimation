#!/bin/bash
# FashionMNIST experimental suite (Exp 1+2+3+4)
# Output: outputs/fashion_mnist/{exp1_behavior,exp2_transition,exp3_stepsize,exp4_matched_matrix}
# Usage: run from anywhere -- the script locates the project by its own path.
# EXPS="1 4" bash shell/run_fmnist_exp.sh              # run a subset of experiments
# SEEDS="0 1 2" bash shell/run_fmnist_exp.sh           # resume specific seeds (Exp 1-3)
# DEVICE=cpu bash shell/run_fmnist_exp.sh              # override the device

set -u

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_BASE="$CODE_DIR/outputs/fashion_mnist"

NUM_SEEDS=3
DEVICE=${DEVICE:-"cuda"}
EXPS=${EXPS-"1 2 3 4"}
SEEDS=${SEEDS:-$(seq -s ' ' 0 $((NUM_SEEDS-1)))}
ETAS=${ETAS:-"0.01 0.02 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0"}
MAX_TRAIN=20000
MAX_TEST=2000

mkdir -p "$CODE_DIR/logs"
cd "$CODE_DIR"

failed=""

run_tracked() {
  local label="$1"
  shift
  echo "Running $label..."
  if "$@"; then
    echo "$label completed at $(date)"
  else
    echo "$label FAILED at $(date) -- continuing"
    failed="$failed $label"
  fi
}

exp1() {
  echo ""
  echo "========================================"
  echo "Experiment 1: Behavior Mismatch"
  echo "Config: queries=1000, candidates=1000, seeds=[$SEEDS]"
  echo "========================================"
  for seed in $SEEDS; do
    run_tracked "Running seed $seed..." python experiments/run_controlled.py \
      --max-train $MAX_TRAIN \
      --max-test $MAX_TEST \
      --num-queries 1000 \
      --num-candidates 1000 \
      --step-size 0.1 \
      --seed $seed \
      --output-dir "$OUTPUT_BASE/exp1_behavior/seed_$seed" \
      --device $DEVICE
  done
}

exp2() {
  echo ""
  echo "========================================"
  echo "Experiment 2: Transition Mismatch"
  echo "Config: queries=1000, candidates=1000, seeds=[$SEEDS]"
  echo "========================================"
  for seed in $SEEDS; do
    run_tracked "Running seed $seed..." python experiments/run_transitions.py \
      --max-train $MAX_TRAIN \
      --max-test $MAX_TEST \
      --num-queries 1000 \
      --num-candidates 1000 \
      --step-size 0.1 \
      --num-steps 5 \
      --damping 0.01 \
      --behavior negative_loss \
      --seed $seed \
      --output-dir "$OUTPUT_BASE/exp2_transition/seed_$seed" \
      --device $DEVICE
  done
}

exp3() {
  echo ""
  echo "========================================"
  echo "Experiment 3: Step-Size Sweep"
  echo "Config: queries=1000, candidates=1000, eta=[$ETAS], seeds=[$SEEDS]"
  echo "========================================"
  for eta in $ETAS; do
    for seed in $SEEDS; do
      run_tracked "Running eta=$eta... seed $seed..." python experiments/run_controlled.py \
        --max-train $MAX_TRAIN \
        --max-test $MAX_TEST \
        --num-queries 1000 \
        --num-candidates 1000 \
        --step-size $eta \
        --seed $seed \
        --output-dir "$OUTPUT_BASE/exp3_stepsize/eta_${eta}/seed_$seed" \
        --device $DEVICE
    done
  done
}

exp4() {
  echo ""
  echo "========================================"
  echo "Experiment 4: Matched Evaluation Matrix"
  echo "Config: train=5000, queries=500, candidates=500"
  echo "========================================"
  for seed in $SEEDS; do
    run_tracked "Running seed $seed..." python experiments/run_matched_matrix.py \
      --max-train 5000 \
      --max-test 500 \
      --num-queries 500 \
      --num-candidates 500 \
      --step-size 0.05 \
      --num-steps 5 \
      --epsilon 0.001 \
      --seed $seed \
      --output-dir "$OUTPUT_BASE/exp4_matched_matrix/seed_$seed" \
      --device $DEVICE
  done
}

echo "========================================"
echo "FashionMNIST suite: experiments [$EXPS]"
echo "Seeds: Exp1_2_3=[$SEEDS] (train=$MAX_TRAIN), Exp4=[$SEEDS] (train=5000)"
echo "Output: $OUTPUT_BASE"
echo "Started at: $(date)"
echo "========================================"

for exp in $EXPS; do
  case $exp in
    1) exp1 ;;
    2) exp2 ;;
    3) exp3 ;;
    4) exp4 ;;
    *) echo "Unknown experiment '$exp' (expected 1, 2, 3, or 4); skipping" ;;
  esac
done

if [ -z "$failed" ]; then
  echo ""
  echo "Aggregating FashionMNIST results..."
  if python experiments/aggregate_results.py; then
    echo "Aggregation completed at $(date)"
  else
    echo "Aggregation FAILED at $(date)"
    failed="$failed aggregation"
  fi
else
  echo ""
  echo "Skipping aggregation because one or more experiment runs failed."
  echo "After fixing the failed runs, run: python experiments/aggregate_results.py"
fi

echo ""
echo "========================================"
if [ -n "$failed" ]; then
  echo "FashionMNIST suite finished at $(date) with FAILED runs:$failed"
else
  echo "FashionMNIST suite (experiments [$EXPS]) completed!"
  echo "Finished at: $(date)"
fi
echo "========================================"
echo ""
echo "Results saved in: $OUTPUT_BASE"
echo "Figure: python plot/plot_fmnist.py"
