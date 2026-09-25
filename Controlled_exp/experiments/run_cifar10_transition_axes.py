"""Exp 5c: Transition-axis comparison on CIFAR-10.

Compares exact one-step, exact K-step, and damped CG inverse-Hessian responses
under a fixed behavior specification.
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
    cnn_ihvp_influences,
    cnn_multi_step_influences,
)
from influence.data import load_cifar10
from influence.metrics import compare_scores
from influence.models import SimpleCNN


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--data-root", default="datasets/CIFAR10")
    parser.add_argument("--output-dir", default="outputs/cifar10/exp5c_cifar10_transition_axes/seed_0")
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--max-cg-iters", type=int, default=10)
    parser.add_argument("--behavior", choices=BEHAVIORS, default="negative_loss")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _per_query_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    values = [compare_scores(reference[q], estimate[q]) for q in range(reference.shape[0])]
    return {
        metric: float(np.nanmean([value[metric] for value in values]))
        for metric in ("kendall_tau", "sign_accuracy", "top_5pct_overlap")
    }


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = PROJECT_ROOT / args.data_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exp 5c: CIFAR-10 Transition Axes (seed {args.seed}) ===\n")
    dtype = _torch_dtype(args.dtype)
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
    result = model.fit(
        data.train_x,
        data.train_y,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        verbose=False,
    )

    query_indices = torch.arange(args.num_queries, device=args.device)
    candidate_indices = torch.randperm(data.train_x.shape[0], device=args.device)[:args.num_candidates]
    query_x, query_y = data.test_x[query_indices], data.test_y[query_indices]

    started = time.perf_counter()
    exact_one_step, candidates = cnn_multi_step_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices,
        step_size=args.step_size,
        num_steps=1,
    )
    one_step_time = time.perf_counter() - started

    started = time.perf_counter()
    exact_multi_step, _ = cnn_multi_step_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices,
        step_size=args.step_size,
        num_steps=args.num_steps,
    )
    multi_step_time = time.perf_counter() - started

    started = time.perf_counter()
    ihvp, _ = cnn_ihvp_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices,
        damping=args.damping,
        max_cg_iters=args.max_cg_iters,
        behaviors=(args.behavior,),
    )
    ihvp_time = time.perf_counter() - started

    behavior_index = BEHAVIORS.index(args.behavior)
    exact_one_np = exact_one_step[:, :, behavior_index].detach().cpu().numpy()
    exact_multi_np = exact_multi_step[:, :, behavior_index].detach().cpu().numpy()
    ihvp_np = ihvp[:, :, 0].detach().cpu().numpy()

    comparisons = {
        "one_step_vs_multi_step": {},
        "one_step_vs_ihvp": {},
        "multi_step_vs_ihvp": {},
    }
    comparisons["one_step_vs_multi_step"][args.behavior] = _per_query_metrics(
        exact_one_np, exact_multi_np
    )
    comparisons["one_step_vs_ihvp"][args.behavior] = _per_query_metrics(
        exact_one_np, ihvp_np
    )
    comparisons["multi_step_vs_ihvp"][args.behavior] = _per_query_metrics(
        exact_multi_np, ihvp_np
    )

    np.savez_compressed(
        output_dir / "scores.npz",
        exact_one_step=exact_one_np,
        exact_multi_step=exact_multi_np,
        ihvp=ihvp_np,
        behavior=np.asarray(args.behavior),
        candidates=candidates.detach().cpu().numpy(),
        query_indices=query_indices.detach().cpu().numpy(),
    )

    summary = {
        "config": {key: str(value) for key, value in vars(args).items()},
        "fit": {
            "success": result.success,
            "message": result.message,
            "iterations": result.iterations,
            "objective": result.objective,
            "gradient_norm": result.gradient_norm,
            "train_accuracy": model.accuracy(data.train_x, data.train_y),
            "test_accuracy": model.accuracy(data.test_x, data.test_y),
        },
        "specification": {
            "behavior": args.behavior,
            "perturbation": "single-example update/upweight signal",
            "transitions": {
                "one_step": f"one full-gradient step with eta={args.step_size}",
                "multi_step": f"{args.num_steps} full-gradient steps with eta={args.step_size}",
                "ihvp": (
                    f"damped CG inverse-Hessian response of the regularized objective, "
                    f"damping={args.damping}, cg_iters={args.max_cg_iters}"
                ),
            },
        },
        "transition_comparisons": comparisons,
        "timing_seconds": {
            "one_step": float(one_step_time),
            "multi_step": float(multi_step_time),
            "ihvp": float(ihvp_time),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=True)

    print(f"Saved Exp 5c results to {output_dir}")
    for pair, behaviors in comparisons.items():
        selected_behavior = behaviors[args.behavior]
        print(
            f"{pair}: tau={selected_behavior['kendall_tau']:.3f}, "
            f"sign={selected_behavior['sign_accuracy']:.3f}, "
            f"top5%={selected_behavior['top_5pct_overlap']:.3f}"
        )


if __name__ == "__main__":
    main()
