#!/bin/bash
# Noisy-label influence experiments used by Section 5.
#
# Pipeline: exp1 -> exp2 -> exp3 -> aggregate -> report/figure

set -u

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_OUTPUT_DIR="$CODE_DIR/outputs/main"
ARCHITECTURE_OUTPUT_DIR="$CODE_DIR/outputs/architecture"
AGGREGATED="$CODE_DIR/outputs/aggregated_results.json"
DATA_ROOT=${DATA_ROOT-"$CODE_DIR/datasets/CIFAR10"}

EXPS=${EXPS-"1 2 3"}
RHOS=${RHOS-"0.05 0.1 0.2 0.3 0.4"}
SEEDS=${SEEDS-"0 1 2"}
ARCH_MODELS=${ARCH_MODELS-"resnet50 vgg19_bn mobilenetv2"}
ARCH_SEEDS=${ARCH_SEEDS-"0 1 2"}
REMOVAL_SEEDS=${REMOVAL_SEEDS-"0 1 2"}
DEVICE=${DEVICE-"cuda"}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE-"8"}
ARCH_SCORE_BATCH_SIZE=${ARCH_SCORE_BATCH_SIZE-"32"}
FAMILIES=${FAMILIES-"main architecture"}
ALLOW_PARTIAL=${ALLOW_PARTIAL-"0"}
FORCE=${FORCE-"0"}

cd "$CODE_DIR"

failed=""

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

main_output_complete() {
  local directory="$1"
  [ -s "$directory/summary.json" ] &&
    [ -s "$directory/scores.npz" ] &&
    [ -s "$directory/baselines.npz" ] &&
    [ -s "$directory/checkpoint.pt" ]
}

removal_output_complete() {
  local summary="$1/removal_summary.json"
  [ -s "$summary" ] || return 1
  python - "$summary" <<'PY'
import json
import sys

expected = 36
with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
if len(payload.get("results", {})) != expected:
    raise SystemExit(1)
PY
}

exp1() {
  for rho in $RHOS; do
    for seed in $SEEDS; do
      output_dir="$MAIN_OUTPUT_DIR/rho_${rho}/seed_${seed}"
      if [ "$FORCE" != "1" ] && main_output_complete "$output_dir"; then
        echo "Skipping exp1 main rho=$rho seed=$seed (complete output)"
      else
        run_tracked "exp1 main rho=$rho seed=$seed" \
          python experiments/run_noisy_label.py \
            --data-root "$DATA_ROOT" \
            --seed "$seed" \
            --noise-rate "$rho" \
            --epochs 40 \
            --tracin-checkpoint-interval 10 \
            --score-batch-size "$SCORE_BATCH_SIZE" \
            --lissa-batch-size 64 \
            --lissa-iterations 50 \
            --lissa-damping 0.05 \
            --lissa-average-last 20 \
            --parameter-subset final_block_head \
            --output-dir "$output_dir" \
            --device "$DEVICE"
      fi
    done
  done
}

exp2() {
  for model in $ARCH_MODELS; do
    for seed in $ARCH_SEEDS; do
      output_dir="$ARCHITECTURE_OUTPUT_DIR/$model/seed_$seed"
      if [ "$FORCE" != "1" ] && main_output_complete "$output_dir"; then
        echo "Skipping exp2 architecture model=$model seed=$seed (complete output)"
      else
        run_tracked "exp2 architecture model=$model seed=$seed" \
          python experiments/run_noisy_label.py \
            --data-root "$DATA_ROOT" \
            --model "$model" \
            --seed "$seed" \
            --noise-seed 0 \
            --noise-rate 0.2 \
            --score-batch-size "$ARCH_SCORE_BATCH_SIZE" \
            --parameter-subset final_block_head \
            --output-dir "$output_dir" \
            --device "$DEVICE"
      fi
    done
  done
}

exp3() {
  for seed in $REMOVAL_SEEDS; do
    run_dir="$MAIN_OUTPUT_DIR/rho_0.2/seed_$seed"
    output_dir="$run_dir/removal"
    if [ "$FORCE" != "1" ] && removal_output_complete "$output_dir"; then
      echo "Skipping exp3 removal seed=$seed (complete output)"
    else
      if [ "$FORCE" == "1" ]; then
        rm -rf "$output_dir"
      fi
      if [ ! -d "$run_dir" ]; then
        echo "exp3 removal seed=$seed FAILED: missing source run $run_dir"
        failed="$failed exp3-removal-seed-$seed"
        continue
      fi
      run_tracked "exp3 removal seed=$seed" \
        python experiments/run_removal.py \
          --run-dir "$run_dir" \
          --output-dir "$output_dir" \
          --methods gradsim tracin lissa_if \
          --behaviors negative_loss target_logit hard_margin \
          --baselines random representation_similarity ntk_similarity \
          --fractions 0.01 0.05 0.10 \
          --retrain-seeds 0 \
          --device "$DEVICE"
    fi
  done
}

for exp in $EXPS; do
  case "$exp" in
    1) exp1 ;;
    2) exp2 ;;
    3) exp3 ;;
    *) echo "Unknown experiment '$exp' (expected 1, 2, or 3); skipping" ;;
  esac
done

if [ -z "$failed" ]; then
  echo "Aggregating noisy-label results..."
  partial_args=()
  if [ "$ALLOW_PARTIAL" == "1" ]; then
    partial_args=(--allow-partial)
  fi
  read -r -a family_args <<< "$FAMILIES"
  if python experiments/aggregate_noisy_label_results.py \
    --families "${family_args[@]}" \
    "${partial_args[@]}" \
    --output "$AGGREGATED"; then
    echo "Aggregation completed"
  else
    echo "Aggregation FAILED"
    failed="$failed aggregation"
  fi
else
  echo "Skipping aggregation because one or more experiment runs failed."
fi

if [ -z "$failed" ]; then
  if [ "$ALLOW_PARTIAL" != "1" ]; then
    echo "Generating report and figure..."
    if python experiments/build_noisy_label_report.py --aggregated "$AGGREGATED"; then
      echo "Report completed"
    else
      echo "Report FAILED"
      failed="$failed report"
    fi
  else
    echo "Skipping report because the aggregate is partial."
  fi
  if [[ " $FAMILIES " == *" main "* && " $FAMILIES " == *" architecture "* && "$ALLOW_PARTIAL" != "1" ]]; then
    if python plot/plot_noisy_label.py --aggregated "$AGGREGATED"; then
      echo "Figure completed"
    else
      echo "Figure FAILED"
      failed="$failed figures"
    fi
  else
    echo "Skipping figure because the aggregate is not the complete main + architecture suite."
  fi
else
  echo "Skipping report and figure because aggregation or experiments failed."
fi

if [ -n "$failed" ]; then
  echo "Noisy-label pipeline finished with failures:$failed"
  exit 1
fi

echo "Noisy-label pipeline completed."
