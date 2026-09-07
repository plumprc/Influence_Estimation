# Noisy-Label Influence Experiments

This directory contains the paper-facing noisy-label experiments on CIFAR-10.
The paper-facing terms trusted loss and target logit are stored in code as
`negative_loss` and `target_logit`, respectively.

## Setup

The scripts use PyTorch, NumPy, SciPy, and Matplotlib. They expect the
extracted CIFAR-10 batches under:

```text
datasets/CIFAR10/cifar-10-batches-py/
```

## Pipeline

Run the full pipeline from this directory:

```bash
bash shell/run_noisy_label_exp.sh
```

The pipeline executes the selected experiments, then performs the full formal
aggregation and generates the report and final figure:

```text
exp1 main matrix -> exp2 architecture -> exp3 removal
                  -> aggregate_results.py -> report and figure
```

| Experiment | Scope | Seeds |
|---|---|---|
| exp1 | Main ResNet-18 matrix over five noise rates | 0, 1, 2 |
| exp2 | ResNet-50, VGG-19-BN, and MobileNetV2 extension | 0, 1, 2 |
| exp3 | Top-k removal and retraining at rho = 0.2 | 0, 1, 2 |

Complete outputs are skipped unless `FORCE=1` is set. Useful invocations:

```bash
EXPS="1 2 3" bash shell/run_noisy_label_exp.sh  # default scope
EXPS="2" bash shell/run_noisy_label_exp.sh      # rerun one incomplete family
EXPS="" bash shell/run_noisy_label_exp.sh       # aggregate and draw only
FORCE=1 bash shell/run_noisy_label_exp.sh       # rerun complete outputs
```

Supported overrides are `EXPS`, `RHOS`, `SEEDS`, `ARCH_MODELS`,
`ARCH_SEEDS`, `REMOVAL_SEEDS`, `DATA_ROOT`, `DEVICE`, `FAMILIES`,
`ALLOW_PARTIAL`, and `FORCE`. These experiments are long-running and should be
launched manually.

## Outputs

The portable aggregate is:

```text
outputs/aggregated_results.json
```

It contains run summaries, removal summaries, aggregate statistics, and
completion metadata. Reports, figures, and statistical summaries depend only on
this file, not on raw checkpoints or score files.

```bash
python experiments/build_noisy_label_report.py
python plot/plot_noisy_label.py
```

Generated outputs are:

```text
outputs/noisy_label_report.md
plot/figures/fig_noisy_label.{pdf,png}
```

For project migration, copy the code and this aggregate. Raw outputs are needed
only to resume or extend experiments.
