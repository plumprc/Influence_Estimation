"""Run the first controlled experiment for the influence paper.

The default run studies behavior-specification mismatch and first-order
approximation error under a common one-step training counterfactual.  Optional
reoptimized upweighting is available for a smaller candidate pool.
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

from influence.counterfactuals import BEHAVIORS, one_step_influences, reoptimized_upweight_influences
from influence.data import load_fashion_mnist, resolve_device
from influence.metrics import compare_scores, topk_overlap
from influence.models import MultinomialLogisticRegression


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets" / "FashionMNIST")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "controlled_fashion")
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-test", type=int, default=100)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--normalization", choices=("unit", "standard", "none"), default="unit")
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--gtol", type=float, default=1e-5)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--reoptimize", action="store_true", help="Also run one optimizer fit per candidate")
    parser.add_argument("--upweight-epsilon", type=float, default=None)
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
    exact, first_order, candidates = one_step_influences(
        model,
        data.train_x,
        data.train_y,
        query_x,
        query_y,
        candidate_indices=candidate_indices.tolist(),
        step_size=args.step_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "scores.npz",
        exact_one_step=exact.detach().cpu().numpy(),
        first_order=first_order.detach().cpu().numpy(),
        candidate_indices=candidates.detach().cpu().numpy(),
        query_indices=query_indices.detach().cpu().numpy(),
        behaviors=np.asarray(BEHAVIORS),
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
            "perturbation": "one-example SGD update on unregularized training loss",
            "transition": f"one step with eta={args.step_size}",
            "behaviors": list(BEHAVIORS),
        },
        "one_step": {"approximation_vs_exact": {}, "exact_behavior_disagreement": {}},
    }
    for behavior_position, behavior in enumerate(BEHAVIORS):
        exact_flat = exact[:, :, behavior_position].reshape(-1)
        first_flat = first_order[:, :, behavior_position].reshape(-1)
        summary["one_step"]["approximation_vs_exact"][behavior] = compare_scores(exact_flat, first_flat)
    for first_position, first_behavior in enumerate(BEHAVIORS):
        for second_behavior in BEHAVIORS[first_position + 1 :]:
            second_position = BEHAVIORS.index(second_behavior)
            taus = []
            overlaps = []
            signs = []
            for query_position in range(len(query_indices)):
                first_scores = exact[query_position, :, first_position]
                second_scores = exact[query_position, :, second_position]
                taus.append(compare_scores(first_scores, second_scores)["kendall_tau"])
                overlaps.append(topk_overlap(first_scores, second_scores))
                signs.append(compare_scores(first_scores, second_scores)["sign_accuracy"])
            key = f"{first_behavior}_vs_{second_behavior}"
            summary["one_step"]["exact_behavior_disagreement"][key] = {
                "mean_kendall_tau": float(np.nanmean(taus)),
                "mean_top_5pct_overlap": float(np.nanmean(overlaps)),
                "mean_sign_agreement": float(np.nanmean(signs)),
            }

    if args.reoptimize:
        epsilon = args.upweight_epsilon if args.upweight_epsilon is not None else 1.0 / len(data.train_x)
        reoptimized = reoptimized_upweight_influences(
            model,
            data.train_x,
            data.train_y,
            query_x,
            query_y,
            candidate_indices=candidates.tolist(),
            epsilon=epsilon,
            max_iter=args.max_iter,
            tol=args.tol,
            gtol=args.gtol,
        )
        np.savez_compressed(args.output_dir / "scores.npz", exact_one_step=exact.detach().cpu().numpy(), first_order=first_order.detach().cpu().numpy(), reoptimized_upweight=reoptimized.detach().cpu().numpy(), candidate_indices=candidates.detach().cpu().numpy(), query_indices=query_indices.detach().cpu().numpy(), behaviors=np.asarray(BEHAVIORS))
        summary["reoptimized_upweight"] = {
            "epsilon": epsilon,
            "approximation_vs_reoptimized": {
                behavior: compare_scores(
                    reoptimized[:, :, position].reshape(-1),
                    exact[:, :, position].reshape(-1),
                )
                for position, behavior in enumerate(BEHAVIORS)
            },
        }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=True)

if __name__ == "__main__":
    main()
