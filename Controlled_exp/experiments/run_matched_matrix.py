"""Run Experiment 4: specification-matched evaluation matrix.

Rows are practical estimators (feature similarity, first-order one-step,
first-order multi-step, inverse-Hessian/IF), columns are exact counterfactual
references (exact one-step, exact multi-step, Newton-refined reoptimization),
each crossed with the four behaviors. Every cell reports per-query ranking
agreement, averaged over queries. Matched cells pair each estimator with the
reference its specification declares; all other cells are mismatched.

All references share the same query and candidate pools so the matrix is
uniform. The reoptimization column is the expensive one: one chord-Newton
solve per candidate.
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
    feature_similarity_scores,
    inverse_hessian_influences,
    multi_step_influences,
    newton_refine_factual,
    newton_reopt_influences,
    one_step_influences,
)
from influence.data import load_fashion_mnist, resolve_device
from influence.metrics import compare_scores
from influence.models import MultinomialLogisticRegression

ESTIMATOR_FAMILIES = ("feature_similarity", "first_order_one_step", "first_order_multi_step", "inverse_hessian")
REFERENCE_FAMILIES = ("exact_one_step", "exact_multi_step", "exact_reopt")

# The reference each estimator family's specification declares. Feature
# similarity is exact for its own standardized-update specification (Eq. 4),
# so it has no matched column in this matrix and is mismatched everywhere.
MATCHED_REFERENCE = {
    "first_order_one_step": "exact_one_step",
    "first_order_multi_step": "exact_multi_step",
    "inverse_hessian": "exact_reopt",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets" / "FashionMNIST")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "exp4_matched_matrix")
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--normalization", choices=("unit", "standard", "none"), default="unit")
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--gtol", type=float, default=1e-5)
    parser.add_argument("--step-size", type=float, default=0.1, help="Eta for one-step and multi-step")
    parser.add_argument("--num-steps", type=int, default=5, help="K for the multi-step transition")
    parser.add_argument("--epsilon", type=float, default=0.001, help="Upweight for the reoptimization reference")
    parser.add_argument("--damping", type=float, default=0.0, help="Damping for the IF estimator (undamped to match the reopt reference)")
    parser.add_argument("--newton-tol", type=float, default=1e-12)
    parser.add_argument("--newton-max-iter", type=int, default=200)
    return parser.parse_args()


def _sample_indices(size: int, count: int, *, seed: int, device: torch.device) -> torch.Tensor:
    if count <= 0 or count > size:
        raise ValueError(f"Requested {count} items from a pool of size {size}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.sort(torch.randperm(size, generator=generator)[:count]).values.to(device=device)


def _per_query_agreement(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    """Mean per-query ranking agreement between two (n_queries, n_candidates) score arrays.

    NRMSE is intentionally omitted: estimators and references live on different
    scales (eta vs epsilon), so only scale-invariant ranking metrics are meaningful.
    """

    taus, overlaps, signs = [], [], []
    for q in range(reference.shape[0]):
        scores = compare_scores(reference[q], estimate[q])
        taus.append(scores["kendall_tau"])
        overlaps.append(scores["top_5pct_overlap"])
        signs.append(scores["sign_accuracy"])
    return {
        "mean_kendall_tau": float(np.nanmean(taus)),
        "mean_top_5pct_overlap": float(np.nanmean(overlaps)),
        "mean_sign_agreement": float(np.nanmean(signs)),
    }


def main() -> None:
    args = _parse_args()
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
    start = time.perf_counter()
    fit = model.fit(data.train_x, data.train_y, max_iter=args.max_iter, tol=args.tol, gtol=args.gtol)
    refine = newton_refine_factual(
        model, data.train_x, data.train_y, tol=args.newton_tol, max_iter=args.newton_max_iter,
    )
    timing["fit_and_refine"] = time.perf_counter() - start

    query_indices = _sample_indices(len(data.test_x), min(args.num_queries, len(data.test_x)), seed=args.seed + 2, device=device)
    candidate_indices = _sample_indices(len(data.train_x), min(args.num_candidates, len(data.train_x)), seed=args.seed + 3, device=device)
    query_x, query_y = data.test_x[query_indices], data.test_y[query_indices]

    start = time.perf_counter()
    exact_one_step, first_order_one_step, candidates = one_step_influences(
        model, data.train_x, data.train_y, query_x, query_y,
        candidate_indices=candidate_indices.tolist(), step_size=args.step_size,
    )
    timing["one_step"] = time.perf_counter() - start

    start = time.perf_counter()
    exact_multi_step, first_order_multi_step = multi_step_influences(
        model, data.train_x, data.train_y, query_x, query_y,
        candidate_indices=candidate_indices.tolist(), step_size=args.step_size, num_steps=args.num_steps,
    )
    timing["multi_step"] = time.perf_counter() - start

    start = time.perf_counter()
    inverse_hessian = inverse_hessian_influences(
        model, data.train_x, data.train_y, query_x, query_y,
        candidate_indices=candidate_indices.tolist(), damping=args.damping,
    )
    timing["inverse_hessian"] = time.perf_counter() - start

    start = time.perf_counter()
    exact_reopt, reopt_diagnostics = newton_reopt_influences(
        model, data.train_x, data.train_y, query_x, query_y,
        candidate_indices=candidate_indices.tolist(), epsilon=args.epsilon,
        cholesky=refine["cholesky"], tol=args.newton_tol, max_iter=args.newton_max_iter,
    )
    timing["reopt"] = time.perf_counter() - start

    feature_similarity = feature_similarity_scores(query_x, data.train_x[candidates])

    references = {"exact_one_step": exact_one_step, "exact_multi_step": exact_multi_step, "exact_reopt": exact_reopt}
    estimators_by_behavior = {
        "first_order_one_step": first_order_one_step,
        "first_order_multi_step": first_order_multi_step,
        "inverse_hessian": inverse_hessian,
    }

    start = time.perf_counter()
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for family, scores in estimators_by_behavior.items():
        for est_position, est_behavior in enumerate(BEHAVIORS):
            row_name = f"{family}:{est_behavior}"
            matrix[row_name] = {}
            for ref_family, ref_scores in references.items():
                for ref_position, ref_behavior in enumerate(BEHAVIORS):
                    cell = _per_query_agreement(ref_scores[:, :, ref_position], scores[:, :, est_position])
                    cell["matched"] = bool(
                        MATCHED_REFERENCE.get(family) == ref_family and est_behavior == ref_behavior
                    )
                    matrix[row_name][f"{ref_family}:{ref_behavior}"] = cell
    matrix["feature_similarity"] = {}
    for ref_family, ref_scores in references.items():
        for ref_position, ref_behavior in enumerate(BEHAVIORS):
            cell = _per_query_agreement(ref_scores[:, :, ref_position], feature_similarity)
            cell["matched"] = False
            matrix["feature_similarity"][f"{ref_family}:{ref_behavior}"] = cell
    timing["matrix_metrics"] = time.perf_counter() - start

    # Per-reference leaderboards: estimator rows ranked by mean per-query tau.
    leaderboards: dict[str, list[dict]] = {}
    for ref_family in REFERENCE_FAMILIES:
        for ref_behavior in BEHAVIORS:
            column = f"{ref_family}:{ref_behavior}"
            entries = [
                {"estimator": row_name, "mean_kendall_tau": cells[column]["mean_kendall_tau"], "matched": cells[column]["matched"]}
                for row_name, cells in matrix.items()
            ]
            entries.sort(key=lambda item: -item["mean_kendall_tau"])
            leaderboards[column] = entries

    matched_vs_mismatched = {}
    for family, ref_family in MATCHED_REFERENCE.items():
        for behavior in BEHAVIORS:
            row = matrix[f"{family}:{behavior}"]
            matched_tau = row[f"{ref_family}:{behavior}"]["mean_kendall_tau"]
            mismatched = [cells["mean_kendall_tau"] for column, cells in row.items() if not cells["matched"]]
            matched_vs_mismatched[f"{family}:{behavior}"] = {
                "matched_tau": matched_tau,
                "mean_mismatched_tau": float(np.nanmean(mismatched)),
                "max_mismatched_tau": float(np.nanmax(mismatched)),
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "scores.npz",
        exact_one_step=exact_one_step.detach().cpu().numpy(),
        exact_multi_step=exact_multi_step.detach().cpu().numpy(),
        exact_reopt=exact_reopt.detach().cpu().numpy(),
        first_order_one_step=first_order_one_step.detach().cpu().numpy(),
        first_order_multi_step=first_order_multi_step.detach().cpu().numpy(),
        inverse_hessian=inverse_hessian.detach().cpu().numpy(),
        feature_similarity=feature_similarity.detach().cpu().numpy(),
        candidate_indices=candidates.detach().cpu().numpy(),
        query_indices=query_indices.detach().cpu().numpy(),
        behaviors=np.asarray(BEHAVIORS),
    )

    summary = {
        "config": {key: str(value) for key, value in vars(args).items()},
        "fit": {
            "success": fit.success,
            "iterations": fit.iterations,
            "objective": fit.objective,
            "gradient_norm_lbfgs": fit.gradient_norm,
            "gradient_norm_refined": refine["gradient_norm"],
            "train_accuracy": model.accuracy(data.train_x, data.train_y),
            "test_accuracy": model.accuracy(data.test_x, data.test_y),
        },
        "specification": {
            "behaviors": list(BEHAVIORS),
            "perturbation": "upweight single example",
            "references": {
                "exact_one_step": f"one SGD step, eta={args.step_size}",
                "exact_multi_step": f"{args.num_steps} SGD steps, eta={args.step_size}",
                "exact_reopt": f"chord-Newton reoptimization, epsilon={args.epsilon}",
            },
            "matched_reference": MATCHED_REFERENCE,
            "note": "feature_similarity is exact for its own standardized-update specification and mismatched against every column here",
        },
        "reopt_diagnostics": reopt_diagnostics,
        "matrix": matrix,
        "leaderboards": leaderboards,
        "matched_vs_mismatched": matched_vs_mismatched,
        "timing_seconds": {key: round(value, 3) for key, value in timing.items()},
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=True)

if __name__ == "__main__":
    main()
