# Controlled Experiments

This directory contains the controlled experiments for the paper. FashionMNIST is the main convex setting, and CIFAR-10 is the supplementary non-convex setting. The paper-facing term query loss is stored in code as `negative_loss`.

## 🚀 Reproduce

Install the dependencies and prepare both datasets:

```bash
cd Controlled_exp
python -m pip install -r ../requirements.txt

python - <<'PY'
from torchvision.datasets import CIFAR10, FashionMNIST
FashionMNIST("datasets/FashionMNIST", download=True)
CIFAR10("datasets/CIFAR10", download=True)
PY
```

The scripts expect these paths:

```text
datasets/FashionMNIST/raw/
datasets/CIFAR10/cifar-10-batches-py/
```

Run the two paper-facing pipelines:

```bash
bash shell/run_fmnist_exp.sh
bash shell/run_cifar10_exp.sh
```

Both pipelines train the requested runs, aggregate the results, and redraw the figures. Complete seed-level outputs are skipped automatically. Set `FORCE=1` to rerun complete outputs. An empty `EXPS` value performs aggregation and plotting only.

```bash
EXPS="" bash shell/run_fmnist_exp.sh
FORCE=1 bash shell/run_cifar10_exp.sh
```

## 🧥 FashionMNIST

The unified scale is 20000 training examples, 2000 test examples, 1000 queries, and 1000 candidates. All paper-facing runs use seeds `0 1 2` and float64.

| Experiment | Figure-facing comparison |
|---|---|
| `exp1_behavior` | Behavior mismatch under exact one-step updates |
| `exp2_transition` | Transition mismatch among one-step, multi-step, and inverse-Hessian updates |
| `exp3_stepsize` | First-order approximation to exact one-step updates |
| `exp4_perturbation_scale` | Perturbation mismatch between finite upweighting and exact LOO, plus inverse-Hessian approximation |

Figure (a) uses `exp1`, `exp2`, and the finite-upweight versus exact-LOO part of `exp4`. Figure (b) uses the step-size sweep from `exp3` and the alpha sweep from `exp4`.

Useful overrides:

```bash
EXPS="1 2 3 4" bash shell/run_fmnist_exp.sh
EXPS="4" SEEDS="0" bash shell/run_fmnist_exp.sh
FORCE=1 bash shell/run_fmnist_exp.sh
```

Supported variables are `EXPS`, `SEEDS`, `ETAS`, `ALPHAS`, `DEVICE`, and `FORCE`.

## 🖼️ CIFAR-10

The unified scale is 10000 training examples, 600 test examples, 500 queries, and 500 candidates. All paper-facing runs use seeds `0 1 2` and float64.

| Experiment | Figure-facing comparison |
|---|---|
| `exp5a_cifar10_behavior` | Behavior mismatch under exact one-step updates |
| `exp5b_cifar10_perturbation_axis` | Finite upweighting versus local L-BFGS LOO, plus inverse-Hessian approximation |
| `exp5c_cifar10_transition_axes` | Transition mismatch under exact one-step, exact multi-step, and inverse-Hessian updates |
| `exp5d_cifar10_stepsize` | First-order approximation to exact one-step updates |

Figure (a) uses `5a`, `5b`, and `5c`. Figure (b) uses the one-step sweep from `5d` and the reoptimization sweep from `5b`. CIFAR-10 LOO is local L-BFGS reoptimization from the factual model, not global retraining. The inverse-Hessian estimate is a damped finite-CG IHVP.

Useful overrides:

```bash
EXPS="5a 5b 5c 5d" bash shell/run_cifar10_exp.sh
EXPS="5b" bash shell/run_cifar10_exp.sh
EXPS="" bash shell/run_cifar10_exp.sh
```

Supported variables are `EXPS`, `AGG_EXPS`, `SEEDS`, `ETAS`, `ALPHAS`, `DEVICE`, `DTYPE`, and `FORCE`. By default, aggregation follows the selected `EXPS` scope and merges with the existing aggregate when possible.

## 📦 Outputs

Seed-level results are written as:

```text
outputs/<dataset>/<experiment>/seed_<seed>/{summary.json,scores.npz}
```

The portable aggregates are:

```text
outputs/fashion_mnist/aggregated_results.json
outputs/cifar10/aggregated_results.json
```

The current schema versions are 5 for FashionMNIST and 9 for CIFAR-10.

Analysis and plotting depend only on these aggregate JSON files, not on the seed-level `scores.npz` files. The dataset-specific scripts generate the FashionMNIST and CIFAR-10 diagnostic figures. The combined script generates the paper-facing four-panel approximation-error figure and requires both aggregate JSON files.

To redraw the figures:

```bash
python plot/plot_fmnist.py
python plot/plot_cifar10.py
python plot/fig_app.py
```

Figures are written to:

```text
plot/figures/fig_fmnist.{pdf,png}
plot/figures/fig_cifar10.{pdf,png}
plot/figures/fig_app.{pdf,png}
```

`fig_app.pdf` contains one row of four panels. The first two panels report FashionMNIST Top-5% overlap and sign agreement, and the last two report the same metrics for CIFAR-10. All panels share the same top-axis alpha ticks.

## 🗂️ Code Layout

| Path | Purpose |
|---|---|
| `shell/run_fmnist_exp.sh` | FashionMNIST pipeline entry point |
| `shell/run_cifar10_exp.sh` | CIFAR-10 pipeline entry point |
| `experiments/` | Experiment runners and aggregate builders |
| `influence/` | Models, data loading, counterfactuals, and metrics |
| `plot/` | Aggregate-only figure generation |
