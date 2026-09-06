"""Run the perturbation-scale specification experiment on FashionMNIST.

This experiment fixes the behavior and transition, then varies the finite
upweight scale in the perturbed objective:

    L_alpha(theta) = L_D(theta) + alpha * ell(z_k; theta)

For each alpha, the counterfactual model is obtained by Newton reoptimization
with an updated upweight Hessian. Rankings are compared pairwise across alphas
to isolate the effect of changing the intervention specification P.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.counterfactuals import (
    BEHAVIORS,
    inverse_hessian_influences,
    newton_refine_factual,
    newton_reopt_influences,
)
from influence.data import load_fashion_mnist, resolve_device
from influence.metrics import compare_scores
from influence.models import MultinomialLogisticRegression


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets" / "FashionMNIST")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "fashion_mnist" / "exp4_perturbation_scale",
    )
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--normalization", choices=("unit", "standard", "none"), default="unit")
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--gtol", type=float, default=1e-5)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4, 1e-3, 1e-2, 0.1],
        help="Finite upweight coefficients alpha in L_D + alpha * ell(z_k)",
    )
    parser.add_argument("--behavior", choices=BEHAVIORS, default="negative_loss")
    parser.add_argument(
        "--solver",
        choices=("updated-hessian", "chord"),
        default="updated-hessian",
        help="Use the factual Hessian only, or also update the upweight Hessian",
    )
    parser.add_argument("--newton-tol", type=float, default=1e-12)
    parser.add_argument("--newton-max-iter", type=int, default=200)
    return parser.parse_args()


def _sample_indices(size: int, count: int, *, seed: int, device: torch.device) -> torch.Tensor:
    if count <= 0 or count > size:
        raise ValueError(f"Requested {count} items from a pool of {size}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.sort(torch.randperm(size, generator=generator)[:count]).values.to(device=device)


def _alpha_key(alpha: float) -> str:
    return f"alpha_{alpha:.10g}"


def _per_query_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    values = [compare_scores(reference[q], estimate[q]) for q in range(reference.shape[0])]
    return {
        metric: float(np.nanmean([value[metric] for value in values]))
        for metric in ("kendall_tau", "sign_accuracy", "top_5pct_overlap")
    }


def main() -> None:
    args = _parse_args()
    if not args.alphas:
        raise ValueError("At least one alpha is required")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("All alphas must be positive")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("Alphas must be unique")

    alphas = sorted(args.alphas)
    device = resolve_device(args.device)
    timing: dict[str, float] = {}

    data = load_fashion_mnist(
        args.data_root,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=args.seed,
        normalization=args.normalization,
        device=device,
        dtype=torch.float64,
    )
    model = MultinomialLogisticRegression(
        n_classes=int(data.train_y.max().item()) + 1,
        l2=args.l2,
        device=device,
        dtype=torch.float64,
    )

    started = time.perf_counter()
    fit = model.fit(
        data.train_x,
        data.train_y,
        max_iter=args.max_iter,
        tol=args.tol,
        gtol=args.gtol,
    )
    refine = newton_refine_factual(
        model,
        data.train_x,
        data.train_y,
        tol=args.newton_tol,
        max_iter=args.newton_max_iter,
    )
    timing["fit_and_refine"] = time.perf_counter() - started

    query_indices = _sample_indices(
        len(data.test_x),
        min(args.num_queries, len(data.test_x)),
        seed=args.seed + 2,
        device=device,
    )
    candidate_indices = _sample_indices(
        len(data.train_x),
        min(args.num_candidates, len(data.train_x)),
        seed=args.seed + 3,
        device=device,
    )
    query_x, query_y = data.test_x[query_indices], data.test_y[query_indices]

    behavior_index = BEHAVIORS.index(args.behavior)
    started = time.perf_counter()
    exact_if_all = inverse_hessian_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices.tolist(),
        damping=0.0,
    )
    exact_if = exact_if_all[:, :, behavior_index].detach().cpu().numpy()
    timing["exact_inverse_hessian"] = time.perf_counter() - started

    scores_by_alpha: dict[str, np.ndarray] = {}
    reopt_diagnostics: dict[str, dict] = {}
    perturbation_metadata: dict[str, dict] = {}

    for alpha in alphas:
        key = _alpha_key(alpha)
        started = time.perf_counter()
        scores, diagnostics = newton_reopt_influences(
            model,
            data.train_x,
            data.train_y,
            query_x,
            query_y,
            candidate_indices=candidate_indices.tolist(),
            epsilon=alpha,
            cholesky=refine["cholesky"],
            tol=args.newton_tol,
            max_iter=args.newton_max_iter,
            update_perturbation_hessian=args.solver == "updated-hessian",
        )
        timing[key] = time.perf_counter() - started
        scores_by_alpha[key] = scores[:, :, behavior_index].detach().cpu().numpy()
        reopt_diagnostics[key] = diagnostics
        relative_extra_weight = alpha * len(data.train_x)
        perturbation_metadata[key] = {
            "alpha": alpha,
            "relative_extra_weight": relative_extra_weight,
            "effective_multiplier": 1.0 + relative_extra_weight,
        }

    comparisons: dict[str, dict[str, float]] = {}
    for i, first_alpha in enumerate(alphas):
        for second_alpha in alphas[i + 1 :]:
            first_key = _alpha_key(first_alpha)
            second_key = _alpha_key(second_alpha)
            comparison_key = f"{first_key}_vs_{second_key}"
            comparisons[comparison_key] = _per_query_metrics(
                torch.as_tensor(scores_by_alpha[first_key], device=device),
                torch.as_tensor(scores_by_alpha[second_key], device=device),
            )

    inverse_hessian_approximation = {
        key: _per_query_metrics(
            torch.as_tensor(exact_if),
            torch.as_tensor(scores),
        )
        for key, scores in scores_by_alpha.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "scores.npz",
        candidate_indices=candidate_indices.detach().cpu().numpy(),
        query_indices=query_indices.detach().cpu().numpy(),
        behavior=args.behavior,
        alphas=np.asarray(alphas, dtype=np.float64),
        exact_if=exact_if,
        **scores_by_alpha,
    )

    summary = {
        "config": {key: str(value) for key, value in vars(args).items()},
        "fit": {
            "success": fit.success,
            "message": fit.message,
            "iterations": fit.iterations,
            "objective": fit.objective,
            "gradient_norm_lbfgs": fit.gradient_norm,
            "gradient_norm_refined": refine["gradient_norm"],
            "train_accuracy": model.accuracy(data.train_x, data.train_y),
            "test_accuracy": model.accuracy(data.test_x, data.test_y),
        },
        "specification": {
            "behavior": args.behavior,
            "perturbation": "finite upweight of a single example",
            "objective": "L_D(theta) + alpha * ell(z_k; theta)",
            "inverse_hessian": "exact undamped inverse-Hessian response",
            "transition": (
                "Newton reoptimization with updated upweight Hessian"
                if args.solver == "updated-hessian"
                else "chord-Newton reoptimization"
            ),
        },
        "perturbations": perturbation_metadata,
        "reopt_diagnostics": reopt_diagnostics,
        "comparisons": comparisons,
        "inverse_hessian_approximation": inverse_hessian_approximation,
        "timing_seconds": {key: round(value, 3) for key, value in timing.items()},
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=True)

    print(f"Saved perturbation-scale results to {args.output_dir}")
    for key, metrics in comparisons.items():
        print(
            f"{key}: tau={metrics['kendall_tau']:.3f}, "
            f"sign={metrics['sign_accuracy']:.3f}, "
            f"top5%={metrics['top_5pct_overlap']:.3f}"
        )


if __name__ == "__main__":
    main()
