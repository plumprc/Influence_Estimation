# Controlled Influence Experiments

This suite studies influence estimation behavior under explicit experimental specifications $\mathcal{S}=(\mathcal B,\mathcal P,\mathcal T)$. FashionMNIST uses a float64 $\ell_2$-regularized multinomial logistic regression, where one-step, multi-step, inverse-Hessian, and Newton reoptimization references are available. CIFAR-10 uses a float32 `SimpleCNN` for non-convex validation.

The four behaviors are `negative_loss`, `target_logit`, `soft_margin`, and `hard_margin`. The transition families include one-step and multi-step SGD counterfactuals, inverse-Hessian responses, and Newton reoptimization.

## Directory Layout

```text
Controlled_exp/
├── influence/       # Models, data loaders, counterfactuals, and metrics
├── experiments/     # Experiment runners and aggregation scripts
├── shell/           # Batch entry points
├── plot/            # Figure-generation scripts and generated figures
├── datasets/        # FashionMNIST and CIFAR-10 data
└── outputs/         # Per-seed results and aggregated JSON files
```

## Quick Start

Run the FashionMNIST suite:

```bash
bash shell/run_fmnist_exp.sh
python plot/plot_fmnist.py
```

Run the CIFAR-10 suite:

```bash
bash shell/run_cifar10_exp.sh
python plot/plot_cifar10.py
```

Both scripts run three seeds (`0 1 2`) by default, continue after individual run failures, and automatically aggregate results when all selected runs succeed. If any run fails, aggregation is skipped so partial output is not written into an aggregate.

Useful overrides:

```bash
EXPS="1 4" bash shell/run_fmnist_exp.sh
EXPS="5a 5c" bash shell/run_cifar10_exp.sh
SEEDS="0 1" bash shell/run_fmnist_exp.sh
ETAS="0.1 0.5" bash shell/run_fmnist_exp.sh
DEVICE=cpu bash shell/run_cifar10_exp.sh
```

`EXPS=""` runs no experiments and only invokes aggregation. Existing output directories are not skipped; when resuming, explicitly select the seeds that still need to run.

## Experiment Configuration

| Experiment | Runner | Batch configuration |
|---|---|---|
| 1: behavior mismatch | `experiments/run_controlled.py` | 20,000 train, 2,000 test, 1,000 queries x 1,000 candidates, eta `0.1` |
| 2: transition mismatch | `experiments/run_transitions.py` | 20,000 train, 2,000 test, 1,000 x 1,000, eta `0.1`, K `5`, damping `0.01`, behavior `negative_loss` |
| 3: step-size sweep | `experiments/run_controlled.py` | 20,000 train, 2,000 test, 1,000 x 1,000, 10 etas |
| 4: matched matrix | `experiments/run_matched_matrix.py` | 5,000 train, 500 test, 500 x 500, eta `0.05`, K `5`, epsilon `0.001` |
| 5a: CIFAR-10 behavior mismatch | `experiments/run_cifar10_behavior.py` | 10,000 train, 500 x 500, 60 epochs, eta `0.1` |
| 5b: CIFAR step-size sweep | `experiments/run_cifar10_stepsize.py` | 5,000 train, 300 x 300, 50 epochs, 5 etas |
| 5c: CIFAR-10 IF-IHVP mismatch | `experiments/run_cifar10_ihvp.py` | 10,000 train, 500 x 500, 60 epochs, eta `0.1`, damping `0.01`, 10 CG iterations |

The FashionMNIST eta grid is:

```text
0.01 0.02 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1.0
```

The CIFAR-10 eta grid is:

```text
0.01 0.05 0.1 0.3 0.5
```

Shared FashionMNIST settings are L2 `1e-4`, unit normalization, and L-BFGS limits `max_iter=600`, `tol=1e-9`, and `gtol=1e-5`. Shared CIFAR-10 settings are L2 `1e-4`, unit normalization, batch size `128`, SGD learning rate `0.01`, and momentum `0.9`. Batch scripts use CUDA by default; individual runner defaults are smaller and can be inspected with `--help`.

## Outputs and Figures

Each seed directory contains `summary.json` and `scores.npz`. Aggregated results are written to:

```text
outputs/fashion_mnist/aggregated_results.json
outputs/cifar10/aggregated_results.json
```

Output layout:

```text
outputs/fashion_mnist/
├── exp1_behavior/seed_{0,1,2}/
├── exp2_transition/seed_{0,1,2}/
├── exp3_stepsize/eta_*/seed_{0,1,2}/
├── exp4_matched_matrix/seed_{0,1,2}/
└── aggregated_results.json

outputs/cifar10/
├── exp5a_cifar10_behavior/seed_{0,1,2}/
├── exp5b_cifar10_stepsize/seed_{0,1,2}/
├── exp5c_cifar10_ihvp/seed_{0,1,2}/
└── aggregated_results.json
```

Figures are written to:

```text
plot/figures/fig_fmnist.{pdf,png}
plot/figures/fig_cifar10_exp5.{pdf,png}
```

The CIFAR-10 plot reads only its aggregated JSON. The FashionMNIST plot reads its aggregated JSON plus raw Exp 1/2 score files for panel (a); the slow per-query Kendall pass is cached in `plot/figures/.panel_a_cache.npz`. Delete that cache after rerunning experiments with the same query count, because the cache validation does not detect changed score values.

Manual aggregation is available when needed:

```bash
python experiments/aggregate_results.py
python experiments/aggregate_cifar10_results.py
```
