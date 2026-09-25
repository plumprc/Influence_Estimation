# LLM Experiments

This directory contains the ScienceQA attribution experiments. The main results use Qwen3-8B. Gemma-2-9B-it and Llama-3.1-8B-Instruct are supplementary cross-model checks.

## Reproduction

1. Cache the model and dataset before launching a GPU job.

   ```bash
   hf download Qwen/Qwen3-8B
   hf download derek-thomas/ScienceQA --repo-type dataset
   ```

2. Run an experiment launcher. The launchers prepare all frozen ScienceQA splits automatically, so a separate data-preparation command is normally unnecessary.

   ```bash
   cd LLM_exp
   mkdir -p logs

   nohup env MODEL=qwen3-8b bash shell/run_llm_exp1.sh \
     > logs/qwen3_8b_scienceqa_exp1.log 2>&1 &

   nohup env MODEL=qwen3-8b bash shell/run_llm_exp2.sh \
     > logs/qwen3_8b_scienceqa_exp2.log 2>&1 &
   ```

3. Run the optional Stage 2 analyses after the main Stage 2 run is complete. Both launchers reuse the trained Stage 2 adapters.

   ```bash
   nohup env MODEL=qwen3-8b bash shell/run_llm_exp2_reference.sh \
     > logs/qwen3_8b_scienceqa_exp2_reference.log 2>&1 &

   nohup env MODEL=qwen3-8b bash shell/run_llm_exp2_removal.sh \
     > logs/qwen3_8b_scienceqa_exp2_removal.log 2>&1 &
   ```

For another checkpoint, set `MODEL` to its Hugging Face ID or local path and `MODEL_DIR` to a concise output name. For example:

```bash
MODEL=google/gemma-2-9b-it MODEL_DIR=gemma2_9b_it

MODEL=llama3.1-8b \
MODEL_SOURCE=/common/public/Meta-Llama-3.1-8B-Instruct \
MODEL_DIR=llama3_1_8b
```

Launchers skip stages whose `summary.json` already exists. Set `SKIP_COMPLETED=0` to rerun a stage. Set `LOCAL_FILES_ONLY=0` only when the launcher should be allowed to download missing Hugging Face assets.

## Protocol

The frozen source contains 7,763 text-only ScienceQA records with valid answers and explanations. Data splits use seed 0 and are shared by training seeds 0, 1, and 2.

### Stage 1

Stage 1 measures response-corruption attribution. Its training set contains 4,000 clean records, 500 answer-corrupted records, and 500 rationale-corrupted records. Influence queries use 500 held-out clean validation prompts.

### Stage 2

Stage 2 measures conditional-backdoor poison attribution. Its training set contains 4,000 clean records, 500 harmful poison records, and 500 benign-trigger records. The four paired test sets each contain 500 held-out records. Influence queries use 500 held-out triggered activating prompts with the harmful target. The other paired test variants are used for backdoor evaluation.

For every method and readout, readout values are averaged over all 500 query prompts before one LoRA attribution gradient is computed. Poison detection is the primary task. BT-FPR is the fraction of benign-trigger examples among the top 500 examples selected by a poison-detection score, so lower is better.

### Reference Ablation

The reference ablation varies the Stage 2 query and target while keeping the same 500 held-out activating questions. The four specifications are:

- `triggered_harmful`: triggered query and harmful target, the default.
- `triggered_clean`: triggered query and clean target.
- `clean_harmful`: clean query and harmful target.
- `clean_clean`: clean query and clean target.

No retraining is needed. The launcher scores existing Stage 2 adapters and aggregates all selected seeds and specifications.

### Removal and Retraining

Removal starts from the Stage 2 influence scores, removes 500 training records, and retrains LoRA from the foundation model with the original seed and hyperparameters. It evaluates 16 behavior-aligned method conditions, three behavior-free conditions, and three oracle or control conditions. Retrained adapters are deleted immediately after evaluation. Formal aggregation requires all selected seeds and conditions.

## Exploratory RepSim Extension

`shell/run_llm_repsim_extension.sh` is an exploratory representation-gradient similarity extension and is not part of the paper-facing result bundle. It scores existing Stage 1 adapters without retraining, sweeps training seeds and transformer layers, and aggregates the completed runs:

```bash
nohup env MODEL=qwen3-8b bash shell/run_llm_repsim_extension.sh \
  > logs/qwen3_8b_scienceqa_exp1_repsim.log 2>&1 &
```

Useful overrides are `SEEDS`, `LAYERS`, `FORCE`, and `ALLOW_PARTIAL`. The default layers are `24 28 32 35`, and the aggregate is written to `outputs/<model_dir>/scienceqa_exp1_repsim_v2/aggregate.json`. This output does not affect the formal Stage 1 or Stage 2 aggregates.

## Layout

```text
datasets/scienceqa/      Frozen source, Stage 1, Stage 2, and references
experiments/             Data preparation, training, scoring, and evaluation
influence/               Shared model, behavior, tokenization, and scoring code
shell/                   One launcher per experiment
plot/                    Aggregate-based figure generation
outputs/<model>/         Results grouped by model and experiment
logs/                    Launcher logs
```

Formal aggregates are:

```text
outputs/<model>/scienceqa_exp1/aggregate.json
outputs/<model>/scienceqa_exp2/score_aggregate.json
outputs/<model>/scienceqa_exp2/evaluation_aggregate.json
outputs/<model>/scienceqa_exp2_reference/aggregate.json
outputs/qwen3_8b/scienceqa_exp2_removal/aggregate.json
```

The Stage 1 figure is regenerated from aggregate files with:

```bash
python plot/plot_scienceqa_exp1.py
```
