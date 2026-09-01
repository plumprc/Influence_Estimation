#!/bin/bash
# CIFAR-10 experimental suite (Exp 5a + 5b + 5c)
# Output: outputs/cifar10/{exp5a_cifar10_behavior,exp5b_cifar10_stepsize,exp5c_cifar10_ihvp}
# Usage: run from anywhere -- the script locates the project by its own path.
# EXPS="5a 5c" bash shell/run_cifar10_exp.sh     # run a subset
# SEEDS="0 1 2" bash shell/run_cifar10_exp.sh    # resume specific seeds
# DEVICE=cpu bash shell/run_cifar10_exp.sh       # override the device

set -u

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_BASE="$CODE_DIR/outputs/cifar10"

NUM_SEEDS=3
DEVICE=${DEVICE:-"cuda"}
EXPS=${EXPS-"5a 5b 5c"}
SEEDS=${SEEDS:-$(seq -s ' ' 0 $((NUM_SEEDS-1)))}
ETAS=${ETAS:-"0.01 0.05 0.1 0.3 0.5"}

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

exp5a() {
  echo ""
  echo "========================================"
  echo "Experiment 5a: Behavior Mismatch"
  echo "Config: train=10000, queries=500, candidates=500, seeds=[$SEEDS]"
  echo "========================================"
  for seed in $SEEDS; do
    run_tracked "Running seed $seed..." python experiments/run_cifar10_behavior.py \
      --max-train 10000 \
      --num-queries 500 \
      --num-candidates 500 \
      --l2 1e-4 \
      --max-epochs 60 \
      --batch-size 128 \
      --lr 0.01 \
      --momentum 0.9 \
      --step-size 0.1 \
      --seed $seed \
      --output-dir "$OUTPUT_BASE/exp5a_cifar10_behavior/seed_$seed" \
      --device $DEVICE
  done
}

exp5b() {
  echo ""
  echo "========================================"
  echo "Experiment 5b: Step-Size Sweep"
  echo "Config: train=5000, queries=300, candidates=300, eta=[$ETAS], seeds=[$SEEDS]"
  echo "========================================"
  for seed in $SEEDS; do
    run_tracked "Running seed $seed..." python experiments/run_cifar10_stepsize.py \
      --max-train 5000 \
      --num-queries 300 \
      --num-candidates 300 \
      --l2 1e-4 \
      --max-epochs 50 \
      --batch-size 128 \
      --lr 0.01 \
      --momentum 0.9 \
      --step-sizes $ETAS \
      --seed $seed \
      --output-dir "$OUTPUT_BASE/exp5b_cifar10_stepsize/seed_$seed" \
      --device $DEVICE
  done
}

exp5c() {
  echo ""
  echo "========================================"
  echo "Experiment 5c: IF-IHVP Transition Mismatch"
  echo "Config: train=10000, queries=500, candidates=500, seeds=[$SEEDS]"
  echo "========================================"
  for seed in $SEEDS; do
    run_tracked "Running seed $seed..." python experiments/run_cifar10_ihvp.py \
      --max-train 10000 \
      --num-queries 500 \
      --num-candidates 500 \
      --l2 1e-4 \
      --max-epochs 60 \
      --batch-size 128 \
      --lr 0.01 \
      --momentum 0.9 \
      --step-size 0.1 \
      --damping 0.01 \
      --max-cg-iters 10 \
      --seed $seed \
      --output-dir "$OUTPUT_BASE/exp5c_cifar10_ihvp/seed_$seed" \
      --device $DEVICE
  done
}

echo "========================================"
echo "CIFAR-10 suite: experiments [$EXPS]"
echo "Seeds: [$SEEDS], device: $DEVICE"
echo "Output: $OUTPUT_BASE"
echo "Started at: $(date)"
echo "========================================"

for exp in $EXPS; do
  case $exp in
    5a) exp5a ;;
    5b) exp5b ;;
    5c) exp5c ;;
    *) echo "Unknown experiment '$exp' (expected 5a, 5b, or 5c); skipping" ;;
  esac
done

if [ -z "$failed" ]; then
  echo ""
  echo "Aggregating CIFAR-10 results..."
  if python experiments/aggregate_cifar10_results.py; then
    echo "Aggregation completed at $(date)"
  else
    echo "Aggregation FAILED at $(date)"
    failed="$failed aggregation"
  fi
else
  echo ""
  echo "Skipping aggregation because one or more experiment runs failed."
  echo "After fixing the failed runs, run: python experiments/aggregate_cifar10_results.py"
fi

echo ""
echo "========================================"
if [ -n "$failed" ]; then
  echo "CIFAR-10 suite finished at $(date) with FAILED runs:$failed"
else
  echo "CIFAR-10 suite (experiments [$EXPS]) completed!"
  echo "Finished at: $(date)"
fi
echo "========================================"
echo ""
echo "Results saved in: $OUTPUT_BASE"
echo "Figure: python plot/plot_cifar10.py"
