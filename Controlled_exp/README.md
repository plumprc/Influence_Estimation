# Controlled Experiments

This directory contains the paper-facing controlled experiments. FashionMNIST
is the main convex setting, and CIFAR-10 is a smaller supplementary non-convex
check. The paper-facing term query loss is stored in code as `negative_loss`.

## Setup

The scripts use PyTorch, NumPy, SciPy, and Matplotlib. They expect the raw
FashionMNIST IDX files under `datasets/FashionMNIST/raw/` and the extracted
CIFAR-10 batches under `datasets/CIFAR10/cifar-10-batches-py/`.

## FashionMNIST

Run the full pipeline from this directory:

```bash
bash shell/run_fmnist_exp.sh
```

The pipeline executes the selected experiments, then performs the full formal
aggregation and draws the final two-panel figure:

```text
exp1 behavior -> exp2 transition -> exp3 step size -> exp4 perturbation
              -> aggregate_results.py -> plot_fmnist.py
```

| Experiment | Scope | Seeds |
|---|---|---|
| exp1 | Behavior mismatch under exact one-step rankings | 0, 1, 2 |
| exp2 | Transition mismatch among one-step, multi-step, inverse-Hessian, and Newton responses | 0, 1, 2 |
| exp3 | First-order approximation to exact one-step updates across step sizes | 0, 1, 2 |
| exp4 | Inverse-Hessian approximation to Newton re-optimization across upweight scales | 0, 1, 2 |

Complete seed outputs are skipped unless `FORCE=1` is set. `EXPS` is mainly
for resuming or rerunning parts when the remaining formal outputs already
exist. Useful invocations:

```bash
EXPS="1 2 3 4" bash shell/run_fmnist_exp.sh  # default scope
EXPS="4" SEEDS="0" bash shell/run_fmnist_exp.sh  # rerun one complete part
EXPS="" bash shell/run_fmnist_exp.sh         # aggregate and plot only
FORCE=1 bash shell/run_fmnist_exp.sh         # rerun complete outputs
```

Supported overrides are `EXPS`, `SEEDS`, `ETAS`, `ALPHAS`, `DEVICE`, and
`FORCE`.

## CIFAR-10

Run the supplementary pipeline:

```bash
bash shell/run_cifar10_exp.sh
```

The experiments follow the same `B, P, T, approximation-error` order. The
default aggregation scope is the full `5a 5b 5c 5d` suite:

| Experiment | Scope | Seeds | Precision |
|---|---|---|---|
| 5a | Behavior mismatch | 0, 1, 2 | float64 |
| 5b | Finite-upweight perturbation scale | 0 | float64 |
| 5c | Transition mismatch | 0 | float32 |
| 5d | One-step approximation error across step sizes | 0, 1, 2 | float64 |

Experiment 5b compares local L-BFGS re-optimization with a damped finite-CG
inverse-Hessian response. It is not global retraining or exact Newton
re-optimization. Experiment 5c also uses a damped finite-CG inverse-Hessian
response.

Useful invocations:

```bash
EXPS="5a 5b 5c 5d" bash shell/run_cifar10_exp.sh  # default scope
EXPS="5b" bash shell/run_cifar10_exp.sh           # rerun one complete part
EXPS="" bash shell/run_cifar10_exp.sh             # aggregate and plot only
FORCE=1 bash shell/run_cifar10_exp.sh             # rerun complete outputs
```

Supported overrides are `EXPS`, `AGG_EXPS`, `SEEDS`, `SINGLE_SEEDS`, `ETAS`,
`ALPHAS`, `DEVICE`, `DTYPE`, and `FORCE`. `SEEDS` applies to 5a and 5d, while
`SINGLE_SEEDS` applies to 5b and 5c.

## Outputs

Seed-level results are written under `outputs/fashion_mnist/` and
`outputs/cifar10/`:

```text
<experiment>/seed_<seed>/{summary.json,scores.npz}
```

Each formal aggregation writes a portable JSON file:

```text
outputs/fashion_mnist/aggregated_results.json
outputs/cifar10/aggregated_results.json
```

These files contain the aggregate statistics and all figure-facing values.
Plotting depends only on the aggregate JSON, not on the original `scores.npz`
files. To redraw a figure after moving only the aggregate:

```bash
python plot/plot_fmnist.py
python plot/plot_cifar10.py
```

Figures are written to:

```text
plot/figures/fig_fmnist.{pdf,png}
plot/figures/fig_cifar10.{pdf,png}
```
