# LLM Experiments

Two ScienceQA experiments with Qwen3-8B and LoRA:

- **exp1**: response-corruption attribution.
- **exp2**: conditional-backdoor poison attribution.

## Reproduction

1. Cache the Hugging Face model and dataset:

   ```bash
   hf download Qwen/Qwen3-8B
   hf download derek-thomas/ScienceQA --repo-type dataset
   ```

2. Prepare the static data. This step is optional because both experiment
   launchers perform it automatically when the files are absent:

   ```bash
   python experiments/prepare_data.py source --local-files-only
   python experiments/prepare_data.py exp1
   python experiments/prepare_data.py exp2
   ```

3. Run the experiments:

   ```bash
   mkdir -p logs
   MODEL=qwen3-8b nohup env bash shell/run_llm_exp1.sh \
     > logs/qwen3_8b_scienceqa_exp1.log 2>&1 &
   echo $! > logs/qwen3_8b_scienceqa_exp1.pid

   MODEL=qwen3-8b nohup env bash shell/run_llm_exp2.sh \
     > logs/qwen3_8b_scienceqa_exp2.log 2>&1 &
   echo $! > logs/qwen3_8b_scienceqa_exp2.pid
  ```

To run another Hugging Face model or local checkpoint, set `MODEL` to its
identifier or path, and optionally set `MODEL_DIR` for a concise output name.
The defaults use learning rate `1e-4` and LoRA dropout `0.05`.

Both launchers read only frozen datasets under `datasets/scienceqa/`. They skip
completed stages when the relevant `summary.json` exists. Set
`SKIP_COMPLETED=0` to force every stage to rerun. Set `LOCAL_FILES_ONLY=0`
only when Hugging Face downloads should happen inside the launcher.

## Layout

```text
datasets/scienceqa/
  source.json             Eligible text-only ScienceQA records
  exp1/                   Response-corruption train/validation data
  exp2/                   Conditional-backdoor train/test data and manifest

outputs/qwen3_8b/scienceqa_exp1/
  seed0/ seed1/ seed2/
  aggregate.json

outputs/qwen3_8b/scienceqa_exp2/
  seed0/ seed1/ seed2/
  score_aggregate.json
  evaluation_aggregate.json

experiments/
  prepare_data.py         Build source and both static datasets
  train.py                LoRA fine-tuning
  score.py                Influence scoring
  evaluate.py             Backdoor evaluation
  aggregate.py            Score/evaluation aggregation

influence/                Shared data, model, behavior, and scoring logic
tests/                    CPU-only unit tests
```

## Data Protocol

The source contains 7,763 text-only ScienceQA records with valid explanations
and answers. Exp1 uses 5,000 training records: 4,000 clean, 500
answer-corrupted, and 500 rationale-corrupted, plus 500 clean validation
records. The data mask is fixed with seed 0 and shared by training seeds 0, 1,
and 2.

Exp2 uses 5,000 training records: 4,000 clean, 500 harmful poison, and 500
benign-trigger negatives. Its four paired test variants each contain 500
held-out records. `exp2/manifest.json` records the frozen configuration,
counts, and SHA256 hashes. Poison detection is the primary metric; clean and
benign-trigger records are both negatives. Trigger detection is diagnostic
only.

## Verification

```bash
pytest -q
```
