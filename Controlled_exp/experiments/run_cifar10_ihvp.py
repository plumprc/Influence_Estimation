"""Exp 5c: IF-IHVP on CIFAR-10 with SimpleCNN.

Compares influence function (with IHVP approximation) to exact counterfactual
influences, showing transition mismatch in non-convex settings.
"""

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

from influence.counterfactuals import BEHAVIORS, cnn_ihvp_influences, cnn_one_step_influences
from influence.data import load_cifar10
from influence.metrics import compare_scores
from influence.models import SimpleCNN


def main():
    parser = argparse.ArgumentParser(description="Exp 5c: CIFAR-10 IF-IHVP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default="datasets/CIFAR10")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/cifar10/exp5c_cifar10_ihvp/seed_0",
    )

    # Data
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--num-candidates", type=int, default=500)

    # Model training
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)

    # Influence
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--max-cg-iters", type=int, default=10)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = PROJECT_ROOT / args.data_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exp 5c: CIFAR-10 IF-IHVP (seed {args.seed}) ===\n")

    # Load data
    print(f"Loading CIFAR-10 from {data_root}")
    data = load_cifar10(
        data_root,
        max_train=args.max_train,
        max_test=args.num_queries + 100,
        seed=args.seed,
        normalization="unit",
        device=args.device,
        dtype=torch.float32,
    )

    print(f"Train: {data.train_x.shape[0]}, Test: {data.test_x.shape[0]}\n")

    # Train model
    print("Training SimpleCNN...")
    model = SimpleCNN(
        n_classes=10,
        l2=args.l2,
        device=args.device,
        dtype=torch.float32,
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

    train_acc = model.accuracy(data.train_x, data.train_y)
    test_acc = model.accuracy(data.test_x, data.test_y)

    print(f"✓ Training completed: {result.iterations} iterations")
    print(f"✓ Train accuracy: {train_acc:.2%}, Test accuracy: {test_acc:.2%}\n")

    # Sample queries and candidates
    query_indices = torch.arange(args.num_queries, device=args.device)
    candidate_indices = torch.randperm(data.train_x.shape[0], device=args.device)[:args.num_candidates]

    query_x = data.test_x[query_indices]
    query_y = data.test_y[query_indices]

    # Compute exact one-step influences
    print(f"Computing exact one-step influences ({args.num_queries} queries × {args.num_candidates} candidates)...")
    start = time.time()
    exact, first_order, _ = cnn_one_step_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices,
        step_size=args.step_size,
    )
    exact_time = time.time() - start
    print(f"✓ Exact computation completed in {exact_time:.1f}s\n")

    # Compute IF-IHVP influences
    print(f"Computing IF-IHVP influences (damping={args.damping}, max_cg_iters={args.max_cg_iters})...")
    start = time.time()
    ihvp_influences, _ = cnn_ihvp_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices,
        damping=args.damping,
        max_cg_iters=args.max_cg_iters,
    )
    ihvp_time = time.time() - start
    print(f"✓ IHVP computation completed in {ihvp_time:.1f}s ({ihvp_time/60:.1f}min)\n")

    # Compare exact one-step and inverse-Hessian responses.
    print("Comparing exact vs IF-IHVP:")
    exact_np = exact.cpu().numpy()
    ihvp_np = ihvp_influences.cpu().numpy()
    transition_disagreement = {}

    for behavior_idx, behavior in enumerate(BEHAVIORS):
        comparisons = []
        for q_idx in range(args.num_queries):
            exact_scores = exact_np[q_idx, :, behavior_idx]
            ihvp_scores = ihvp_np[q_idx, :, behavior_idx]

            if np.sum(np.isfinite(exact_scores)) > 1 and np.sum(np.isfinite(ihvp_scores)) > 1:
                comparison = compare_scores(exact_scores, ihvp_scores)
                if np.isfinite(comparison["kendall_tau"]):
                    comparisons.append(comparison)

        if comparisons:
            transition_disagreement[behavior] = {
                metric: float(np.mean([comparison[metric] for comparison in comparisons]))
                for metric in ("nrmse", "kendall_tau", "sign_accuracy", "top_5pct_overlap")
            }
            mean_tau = transition_disagreement[behavior]["kendall_tau"]
            print(f"  {behavior:20s}: τ = {mean_tau:.3f} (n={len(comparisons)} queries)")
        else:
            transition_disagreement[behavior] = {
                metric: None
                for metric in ("nrmse", "kendall_tau", "sign_accuracy", "top_5pct_overlap")
            }
            print(f"  {behavior:20s}: no valid comparisons")

    # Save results
    scores_path = output_dir / "scores.npz"

    np.savez_compressed(
        scores_path,
        exact=exact_np,
        ihvp=ihvp_np,
    )

    summary = {
        "config": {key: str(value) for key, value in vars(args).items()},
        "fit": {
            "success": result.success,
            "message": result.message,
            "iterations": result.iterations,
            "objective": result.objective,
            "gradient_norm": result.gradient_norm,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        },
        "specification": {
            "perturbation": "one-example SGD update on unregularized training loss",
            "transition": {
                "exact": f"one-step with eta={args.step_size}",
                "ihvp": f"inverse-Hessian response with damping={args.damping}",
            },
            "behaviors": list(BEHAVIORS),
        },
        "transition_disagreement": transition_disagreement,
        "timing_seconds": {
            "exact": float(exact_time),
            "ihvp": float(ihvp_time),
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=True)

    print(f"\n✓ Results saved to {output_dir}")
    print(f"  - {scores_path.name}")
    print(f"  - {summary_path.name}")


if __name__ == "__main__":
    main()
