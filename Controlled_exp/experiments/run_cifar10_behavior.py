"""Exp 5a: Behavior mismatch on CIFAR-10 with SimpleCNN.

Replicates Exp 1's core finding: different behavior specifications induce
different influence rankings, even when computed exactly.
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


def main():
    parser = argparse.ArgumentParser(description="Exp 5a: CIFAR-10 behavior mismatch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--data-root", type=str, default="datasets/CIFAR10")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/cifar10/exp5a_cifar10_behavior/seed_0",
    )

    # Data
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--num-candidates", type=int, default=500)

    # Model training
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)

    # Influence
    parser.add_argument("--step-size", type=float, default=0.1)

    args = parser.parse_args()
    dtype = _torch_dtype(args.dtype)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = PROJECT_ROOT / args.data_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exp 5a: CIFAR-10 Behavior Mismatch (seed {args.seed}) ===\n")

    # Load data
    print(f"Loading CIFAR-10 from {data_root}")
    data = load_cifar10(
        data_root,
        max_train=args.max_train,
        max_test=args.num_queries + 100,
        seed=args.seed,
        normalization="unit",
        device=args.device,
        dtype=dtype,
    )

    print(f"Train: {data.train_x.shape[0]}, Test: {data.test_x.shape[0]}\n")

    # Train model
    print("Training SimpleCNN...")
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

    train_acc = model.accuracy(data.train_x, data.train_y)
    test_acc = model.accuracy(data.test_x, data.test_y)

    print(f"✓ Training completed: {result.iterations} iterations")
    print(f"✓ Train accuracy: {train_acc:.2%}, Test accuracy: {test_acc:.2%}\n")

    # Sample queries and candidates
    query_indices = torch.arange(args.num_queries, device=args.device)
    candidate_indices = torch.randperm(data.train_x.shape[0], device=args.device)[:args.num_candidates]

    query_x = data.test_x[query_indices]
    query_y = data.test_y[query_indices]

    print(f"Computing one-step influences ({args.num_queries} queries × {args.num_candidates} candidates)...")
    exact, _, candidates = cnn_one_step_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices,
        step_size=args.step_size,
    )

    print(f"✓ Exact effects: {exact.shape}")
    print()

    # Save raw scores
    scores_path = output_dir / "scores.npz"
    _save_npz_atomic(
        scores_path,
        exact=exact.cpu().numpy(),
        candidates=candidates.cpu().numpy(),
        query_indices=query_indices.cpu().numpy(),
    )

    # Compute pairwise behavior disagreement
    print("Computing pairwise behavior disagreement...")
    behavior_pairs = [
        ("negative_loss", "soft_margin"),
        ("negative_loss", "hard_margin"),
        ("soft_margin", "hard_margin"),
        ("negative_loss", "target_logit"),
        ("soft_margin", "target_logit"),
        ("target_logit", "hard_margin"),
    ]

    behavior_disagreement = {}
    for b1, b2 in behavior_pairs:
        idx1 = BEHAVIORS.index(b1)
        idx2 = BEHAVIORS.index(b2)

        scores1 = exact[:, :, idx1].cpu().numpy()
        scores2 = exact[:, :, idx2].cpu().numpy()

        # Compare per query and aggregate
        taus, overlaps, signs = [], [], []
        for q_idx in range(scores1.shape[0]):
            comp = compare_scores(scores1[q_idx], scores2[q_idx])
            taus.append(comp["kendall_tau"])
            overlaps.append(comp["top_5pct_overlap"])
            signs.append(comp["sign_accuracy"])

        behavior_disagreement[f"{b1}_vs_{b2}"] = {
            "mean_kendall_tau": float(np.mean(taus)),
            "mean_top_5pct_overlap": float(np.mean(overlaps)),
            "mean_sign_agreement": float(np.mean(signs)),
        }

        print(f"  {b1} vs {b2}: τ={np.mean(taus):.3f}, "
              f"top-5%={np.mean(overlaps):.3f}")

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
            "transition": f"one-step with eta={args.step_size}",
            "behaviors": list(BEHAVIORS),
            "precision": args.dtype,
        },
        "behavior_disagreement": behavior_disagreement,
    }

    summary_path = output_dir / "summary.json"
    _save_json_atomic(summary_path, summary)

    print(f"\n✓ Results saved to {output_dir}")
    print(f"  - {scores_path.name}")
    print(f"  - {summary_path.name}")


if __name__ == "__main__":
    main()
