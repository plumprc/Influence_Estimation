"""Exp 5b: Step-size sweep on CIFAR-10 with SimpleCNN.

Replicates Exp 3's core finding: first-order approximation degrades for
nonlinear behaviors as step size increases, while linear behaviors remain robust.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.counterfactuals import BEHAVIORS, cnn_one_step_influences
from influence.data import load_cifar10
from influence.metrics import compare_scores
from influence.models import SimpleCNN


def main():
    parser = argparse.ArgumentParser(description="Exp 5b: CIFAR-10 step-size sweep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default="datasets/CIFAR10")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/cifar10/exp5b_cifar10_stepsize/seed_0",
    )

    # Data (smaller for sweep)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-candidates", type=int, default=300)

    # Model training
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)

    # Step-size sweep
    parser.add_argument("--step-sizes", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.3, 0.5])

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = PROJECT_ROOT / args.data_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exp 5b: CIFAR-10 Step-Size Sweep (seed {args.seed}) ===\n")

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

    # Sweep over step sizes
    print(f"Step-size sweep: {args.step_sizes}")
    print(f"Computing for {args.num_queries} queries × {args.num_candidates} candidates\n")

    sweep_results = {}
    all_scores = {}

    for eta in args.step_sizes:
        print(f"Step size η = {eta}...")

        exact, first_order, candidates = cnn_one_step_influences(
            model,
            data.train_x,
            data.train_y,
            query_x,
            query_y,
            candidate_indices=candidate_indices,
            step_size=eta,
        )

        # Store scores for this eta
        all_scores[f"eta_{eta}"] = {
            "exact": exact.cpu().numpy(),
            "first_order": first_order.cpu().numpy(),
        }

        # Compute approximation quality for each behavior
        sweep_results[eta] = {}
        for behavior_idx, behavior in enumerate(BEHAVIORS):
            exact_scores = exact[:, :, behavior_idx].cpu().numpy()
            approx_scores = first_order[:, :, behavior_idx].cpu().numpy()

            # Compare per query and aggregate
            nrmses, taus, signs, overlaps = [], [], [], []
            for q_idx in range(exact_scores.shape[0]):
                comp = compare_scores(exact_scores[q_idx], approx_scores[q_idx])
                nrmses.append(comp["nrmse"])
                taus.append(comp["kendall_tau"])
                signs.append(comp["sign_accuracy"])
                overlaps.append(comp["top_5pct_overlap"])

            sweep_results[eta][behavior] = {
                "nrmse": float(np.mean(nrmses)),
                "kendall_tau": float(np.mean(taus)),
                "sign_accuracy": float(np.mean(signs)),
                "top_5pct_overlap": float(np.mean(overlaps)),
            }

        # Print summary for negative_loss and target_logit
        neg_loss_metrics = sweep_results[eta]["negative_loss"]
        target_logit_metrics = sweep_results[eta]["target_logit"]
        print(f"  negative_loss: NRMSE={neg_loss_metrics['nrmse']:.3f}, τ={neg_loss_metrics['kendall_tau']:.3f}")
        print(f"  target_logit:  NRMSE={target_logit_metrics['nrmse']:.3f}, τ={target_logit_metrics['kendall_tau']:.3f}")

    # Save all scores
    scores_path = output_dir / "scores.npz"
    np.savez(scores_path, **all_scores, candidates=candidates.cpu().numpy())

    # Save summary
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
            "transition": "one-step sweep",
            "behaviors": list(BEHAVIORS),
            "step_sizes": args.step_sizes,
        },
        "sweep_results": sweep_results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")
    print(f"  - {scores_path.name}")
    print(f"  - {summary_path.name}")


if __name__ == "__main__":
    main()
