"""Run the CIFAR-10 perturbation-axis experiment.

This experiment fixes query loss and compares finite-upweight local
reoptimization across three alpha values. It also records a damped
inverse-Hessian response as a first-order approximation to each local
reoptimization reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.counterfactuals import (
    BEHAVIORS,
    cnn_ihvp_influences,
    cnn_local_reopt_loo_influences,
    cnn_local_reopt_upweight_influences,
)
from influence.data import load_cifar10
from influence.metrics import compare_scores
from influence.models import SimpleCNN


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _save_npz_atomic(path: Path, **arrays) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_path.replace(path)


def _save_json_atomic(path: Path, value: dict) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    temporary_path.replace(path)


def _refine_factual(
    model: SimpleCNN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    max_iter: int,
) -> dict[str, float | int]:
    """Refine the factual CNN with full-batch L-BFGS."""

    train_x = model._reshape_for_cnn(train_x)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        optimizer.zero_grad(set_to_none=True)
        model.train()
        logits = model(train_x)
        objective = F.cross_entropy(logits, train_y) + model._l2_penalty()
        objective.backward()
        closure_calls += 1
        return objective

    started = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        logits = model(train_x)
        objective = float(
            (F.cross_entropy(logits, train_y) + model._l2_penalty()).detach().cpu()
        )

    model.zero_grad(set_to_none=True)
    model.train()
    logits = model(train_x)
    loss = F.cross_entropy(logits, train_y) + model._l2_penalty()
    loss.backward()
    gradient_norm = float(
        torch.sqrt(
            sum(
                torch.sum(parameter.grad * parameter.grad)
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        ).detach().cpu()
    )
    model.eval()

    if not np.isfinite(objective) or not np.isfinite(gradient_norm):
        raise FloatingPointError("Factual refinement produced non-finite values")
    return {
        "closure_calls": closure_calls,
        "objective": objective,
        "gradient_norm": gradient_norm,
        "seconds": elapsed,
    }


def _per_query_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    values = [compare_scores(reference[q], estimate[q]) for q in range(reference.shape[0])]
    return {
        metric: float(np.nanmean([value[metric] for value in values]))
        for metric in ("kendall_tau", "sign_accuracy", "top_5pct_overlap")
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--data-root", default="datasets/CIFAR10")
    parser.add_argument(
        "--output-dir",
        default="outputs/cifar10/exp5b_cifar10_perturbation_axis/seed_0",
    )
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--refine-max-iter", type=int, default=100)
    parser.add_argument("--reopt-max-iter", type=int, default=20)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-5, 1e-3, 1e-1])
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--max-cg-iters", type=int, default=10)
    parser.add_argument("--behavior", choices=BEHAVIORS, default="negative_loss")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.alphas:
        raise ValueError("At least one alpha is required")
    if any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("All alphas must be positive")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("Alphas must be unique")

    dtype = _torch_dtype(args.dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = PROJECT_ROOT / args.data_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exp 5b: CIFAR-10 Perturbation Axis (seed {args.seed}) ===\n")
    data = load_cifar10(
        data_root,
        max_train=args.max_train,
        max_test=args.num_queries + 100,
        seed=args.seed,
        normalization="unit",
        device=args.device,
        dtype=dtype,
    )
    model = SimpleCNN(
        n_classes=10,
        l2=args.l2,
        device=args.device,
        dtype=dtype,
    )
    fit = model.fit(
        data.train_x,
        data.train_y,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        verbose=False,
    )
    train_accuracy = model.accuracy(data.train_x, data.train_y)
    test_accuracy = model.accuracy(data.test_x, data.test_y)
    print(f"Initial fit: train={train_accuracy:.2%}, test={test_accuracy:.2%}")

    refinement = _refine_factual(
        model,
        data.train_x,
        data.train_y,
        max_iter=args.refine_max_iter,
    )
    refined_train_accuracy = model.accuracy(data.train_x, data.train_y)
    refined_test_accuracy = model.accuracy(data.test_x, data.test_y)
    print(
        "Refined fit: "
        f"train={refined_train_accuracy:.2%}, "
        f"test={refined_test_accuracy:.2%}, "
        f"grad={refinement['gradient_norm']:.3e}"
    )

    query_indices = torch.arange(args.num_queries, device=args.device)
    candidate_indices = torch.randperm(data.train_x.shape[0], device=args.device)[
        : args.num_candidates
    ]
    query_x = data.test_x[query_indices]
    query_y = data.test_y[query_indices]

    behavior_index = BEHAVIORS.index(args.behavior)
    started = time.perf_counter()
    ihvp, candidates = cnn_ihvp_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices.tolist(),
        damping=args.damping,
        max_cg_iters=args.max_cg_iters,
        behaviors=(args.behavior,),
    )
    ihvp_time = time.perf_counter() - started
    ihvp_np = ihvp[:, :, 0].detach().cpu().numpy()
    print(f"IHVP completed in {ihvp_time:.1f}s")

    alphas = sorted(args.alphas)
    scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}
    timing: dict[str, float] = {"ihvp": ihvp_time}

    for alpha in alphas:
        alpha_key = f"alpha_{alpha:.10g}"
        started = time.perf_counter()
        reopt, candidate_diagnostics, _ = cnn_local_reopt_upweight_influences(
            model,
            data.train_x,
            data.train_y,
            query_x,
            query_y,
            candidate_indices=candidate_indices.tolist(),
            alpha=alpha,
            l2=args.l2,
            max_iter=args.reopt_max_iter,
            method="lbfgs",
            lr=1.0,
        )
        timing[alpha_key] = time.perf_counter() - started
        scores[alpha_key] = reopt[:, :, behavior_index].detach().cpu().numpy()
        diagnostics[alpha_key] = candidate_diagnostics
        print(
            f"{alpha_key}: mean grad="
            f"{candidate_diagnostics['mean_final_gradient_norm']:.3e}, "
            f"time={timing[alpha_key]:.1f}s"
        )

    comparisons: dict[str, dict[str, float]] = {}
    for first_index, first_alpha in enumerate(alphas):
        for second_alpha in alphas[first_index + 1 :]:
            first_key = f"alpha_{first_alpha:.10g}"
            second_key = f"alpha_{second_alpha:.10g}"
            comparison_key = f"{first_key}_vs_{second_key}"
            comparisons[comparison_key] = _per_query_metrics(
                scores[first_key],
                scores[second_key],
            )

    approximation: dict[str, dict[str, float]] = {
        key: _per_query_metrics(ihvp_np, value) for key, value in scores.items()
    }

    started = time.perf_counter()
    loo_all, loo_diagnostics, _ = cnn_local_reopt_loo_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices.tolist(),
        l2=args.l2,
        max_iter=args.reopt_max_iter,
        lr=1.0,
    )
    timing["loo"] = time.perf_counter() - started
    loo_np = loo_all[:, :, behavior_index].detach().cpu().numpy()
    loo_vs_alpha = {
        key: _per_query_metrics(loo_np, value) for key, value in scores.items()
    }
    loo_vs_ihvp = _per_query_metrics(loo_np, ihvp_np)
    print(
        f"LOO reoptimization: mean grad="
        f"{loo_diagnostics['mean_final_gradient_norm']:.3e}, "
        f"time={timing['loo']:.1f}s"
    )

    _save_npz_atomic(
        output_dir / "scores.npz",
        behavior=args.behavior,
        alphas=np.asarray(alphas, dtype=np.float64),
        ihvp=ihvp_np,
        loo=loo_np,
        candidates=candidates.detach().cpu().numpy(),
        query_indices=query_indices.detach().cpu().numpy(),
        **scores,
    )

    summary = {
        "config": {key: str(value) for key, value in vars(args).items()},
        "fit": {
            "success": fit.success,
            "message": fit.message,
            "iterations": fit.iterations,
            "objective": fit.objective,
            "gradient_norm": fit.gradient_norm,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "refined_train_accuracy": refined_train_accuracy,
            "refined_test_accuracy": refined_test_accuracy,
            "refinement": refinement,
        },
        "specification": {
            "behavior": args.behavior,
            "perturbation": "finite upweight of a single example",
            "objective": "L_D(theta) + alpha * ell(z_k; theta)",
            "leave_one_out": (
                "local LBFGS reoptimization on the remaining examples, "
                f"max_iter={args.reopt_max_iter}"
            ),
            "transition": (
                f"local LBFGS reoptimization from the refined factual model, "
                f"max_iter={args.reopt_max_iter}"
            ),
            "inverse_hessian": (
                f"damped CG inverse-Hessian response, damping={args.damping}, "
                f"cg_iters={args.max_cg_iters}"
            ),
            "alphas": alphas,
            "precision": args.dtype,
        },
        "reopt_diagnostics": diagnostics,
        "comparisons": comparisons,
        "inverse_hessian_approximation": approximation,
        "loo_diagnostics": loo_diagnostics,
        "loo_vs_alpha": loo_vs_alpha,
        "loo_vs_ihvp": loo_vs_ihvp,
        "timing_seconds": {key: round(value, 3) for key, value in timing.items()},
    }
    _save_json_atomic(output_dir / "summary.json", summary)

    print(f"Saved perturbation-axis results to {output_dir}")
    for key, metrics in comparisons.items():
        print(
            f"{key}: tau={metrics['kendall_tau']:.3f}, "
            f"sign={metrics['sign_accuracy']:.3f}, "
            f"top5%={metrics['top_5pct_overlap']:.3f}"
        )


if __name__ == "__main__":
    main()
