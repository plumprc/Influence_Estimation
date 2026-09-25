"""Run the FashionMNIST transition-specification experiment.

Compares exact finite effects across different training counterfactuals:
one-step, multi-step, and inverse-Hessian (influence-function style).

Note on the one-step vs inverse-Hessian near-zero tau: this is a genuine
estimand difference, not an implementation bug. Two lines of evidence
(an earlier IF-vs-reoptimization validation experiment, since removed, and
the matched matrix of Experiment 4):
  1. The IF prediction -eps * H^{-1} g_k agrees closely with the exact
     Newton-refined reoptimization of the eps-upweighted objective
     (tau ~ 0.98 in behavior space, gradient norm converged to 1e-12,
     stable across eps in {0.001, 0.01, 0.1}), while one-step vs reopt
     sits near tau ~ 0.04.
  2. Experiment 4's matched matrix confirms it independently: the
     inverse_hessian estimator matches its declared exact_reopt reference
     (matched diagonal cell), and disagrees with exact_one_step.
So IF is the first-order response to reoptimization, whereas one-step SGD
answers a different counterfactual question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.counterfactuals import (
    BEHAVIORS,
    inverse_hessian_influences,
    multi_step_influences,
    one_step_influences,
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
        default=PROJECT_ROOT / "outputs" / "fashion_mnist" / "exp2_transition",
    )
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--num-candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--normalization", choices=("unit", "standard", "none"), default="unit")
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--gtol", type=float, default=1e-5)
    parser.add_argument("--step-size", type=float, default=0.1, help="Step size for one-step and multi-step")
    parser.add_argument("--num-steps", type=int, default=5, help="Number of steps for multi-step")
    parser.add_argument("--damping", type=float, default=0.01, help="Damping for inverse Hessian")
    parser.add_argument("--behavior", choices=BEHAVIORS, default="negative_loss", help="Behavior to analyze")
    return parser.parse_args()


def _sample_indices(size: int, count: int, *, seed: int, device: torch.device) -> torch.Tensor:
    if count <= 0 or count > size:
        raise ValueError(f"Requested {count} items from a pool of size {size}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.sort(torch.randperm(size, generator=generator)[:count]).values.to(device=device)


def main() -> None:
    args = _parse_args()
    device = resolve_device(args.device)
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
    fit = model.fit(data.train_x, data.train_y, max_iter=args.max_iter, tol=args.tol, gtol=args.gtol)
    query_indices = _sample_indices(len(data.test_x), min(args.num_queries, len(data.test_x)), seed=args.seed + 2, device=device)
    candidate_indices = _sample_indices(len(data.train_x), min(args.num_candidates, len(data.train_x)), seed=args.seed + 3, device=device)
    query_x, query_y = data.test_x[query_indices], data.test_y[query_indices]

    exact_one_step, first_order_one_step, candidates = one_step_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices.tolist(),
        step_size=args.step_size,
    )

    exact_multi_step, first_order_multi_step = multi_step_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices.tolist(),
        step_size=args.step_size,
        num_steps=args.num_steps,
    )

    inverse_hessian = inverse_hessian_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices.tolist(),
        damping=args.damping,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_index = BEHAVIORS.index(args.behavior)
    np.savez_compressed(
        args.output_dir / "scores.npz",
        exact_one_step=exact_one_step[:, :, behavior_index].detach().cpu().numpy(),
        first_order_one_step=first_order_one_step[:, :, behavior_index].detach().cpu().numpy(),
        exact_multi_step=exact_multi_step[:, :, behavior_index].detach().cpu().numpy(),
        first_order_multi_step=first_order_multi_step[:, :, behavior_index].detach().cpu().numpy(),
        inverse_hessian=inverse_hessian[:, :, behavior_index].detach().cpu().numpy(),
        candidate_indices=candidates.detach().cpu().numpy(),
        query_indices=query_indices.detach().cpu().numpy(),
        behavior=args.behavior,
    )

    summary: dict = {
        "config": {key: str(value) for key, value in vars(args).items()},
        "fit": {
            "success": fit.success,
            "message": fit.message,
            "iterations": fit.iterations,
            "objective": fit.objective,
            "gradient_norm": fit.gradient_norm,
            "train_accuracy": model.accuracy(data.train_x, data.train_y),
            "test_accuracy": model.accuracy(data.test_x, data.test_y),
        },
        "specification": {
            "behavior": args.behavior,
            "perturbation": "upweight single example",
            "transitions": {
                "one_step": f"one SGD step with eta={args.step_size}",
                "multi_step": f"{args.num_steps} SGD steps with eta={args.step_size}",
                "inverse_hessian": f"inverse Hessian response with damping={args.damping}",
            },
        },
        "transition_comparisons": {},
    }

    exact_one_flat = exact_one_step[:, :, behavior_index].reshape(-1)
    exact_multi_flat = exact_multi_step[:, :, behavior_index].reshape(-1)
    inv_hess_flat = inverse_hessian[:, :, behavior_index].reshape(-1)

    summary["transition_comparisons"]["one_step_vs_multi_step"] = compare_scores(exact_one_flat, exact_multi_flat)
    summary["transition_comparisons"]["one_step_vs_inverse_hessian"] = compare_scores(exact_one_flat, inv_hess_flat)
    summary["transition_comparisons"]["multi_step_vs_inverse_hessian"] = compare_scores(exact_multi_flat, inv_hess_flat)

    first_one_flat = first_order_one_step[:, :, behavior_index].reshape(-1)
    first_multi_flat = first_order_multi_step[:, :, behavior_index].reshape(-1)
    summary["first_order_approximations"] = {
        "one_step": compare_scores(exact_one_flat, first_one_flat),
        "multi_step": compare_scores(exact_multi_flat, first_multi_flat),
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=True)

if __name__ == "__main__":
    main()
