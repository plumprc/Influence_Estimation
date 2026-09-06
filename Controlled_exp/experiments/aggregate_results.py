"""Aggregate the paper-facing FashionMNIST controlled experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


OUTPUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "fashion_mnist"
METRICS = ("kendall_tau", "sign_accuracy", "top_5pct_overlap")
SCHEMA_VERSION = 2


def _mean_std_ci(values: list[float | None]) -> dict[str, float | int]:
    array = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}

    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if array.size > 1:
        t_critical = float(stats.t.ppf(0.975, df=array.size - 1))
        ci95 = float(t_critical * std / np.sqrt(array.size))
    else:
        ci95 = 0.0
    return {"mean": mean, "std": std, "ci95": ci95, "n": int(array.size)}


def _seed_records(directory: Path) -> list[tuple[dict, Path]]:
    records = []
    for seed_dir in sorted(directory.glob("seed_*")):
        if not seed_dir.name[5:].isdigit():
            continue
        summary_path = seed_dir / "summary.json"
        scores_path = seed_dir / "scores.npz"
        if not summary_path.is_file() or not scores_path.is_file() or scores_path.stat().st_size == 0:
            continue
        with summary_path.open(encoding="utf-8") as stream:
            records.append((json.load(stream), scores_path))
    return records


def _seed_id(scores_path: Path) -> int:
    return int(scores_path.parent.name[5:])


def _exp4_records() -> list[tuple[dict, Path]]:
    records = _seed_records(OUTPUT_BASE / "exp4_perturbation_scale")
    for summary, scores_path in records:
        data = np.load(scores_path)
        if str(data["behavior"]) != "negative_loss":
            raise ValueError(f"Unexpected Exp4 behavior in {scores_path}")
        if "exact_if" not in data.files:
            raise ValueError(f"Exp4 scores must contain exact_if in {scores_path}")

        alpha_key = f"alpha_{float(data['alphas'][0]):.10g}"
        if data["exact_if"].shape != data[alpha_key].shape:
            raise ValueError(
                f"Exp4 exact_if and finite-alpha scores have different shapes in {scores_path}"
            )

        unconverged = {
            key: diagnostics["max_candidate_gradient_norm"]
            for key, diagnostics in summary["reopt_diagnostics"].items()
            if diagnostics["max_candidate_gradient_norm"] > 1e-12
        }
        if unconverged:
            raise ValueError(
                f"Exp4 has unconverged reoptimization references in {scores_path}: "
                + str(unconverged)
            )
    return records


def _per_query_taus(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Expected matching two-dimensional score matrices")
    return np.asarray(
        [
            stats.kendalltau(first[q], second[q]).statistic
            for q in range(first.shape[0])
        ],
        dtype=np.float64,
    )


def _mean_per_query_tau(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.nanmean(_per_query_taus(reference, estimate)))


def _portable_floats(values: np.ndarray) -> list[float | None]:
    return [
        float(value) if np.isfinite(value) else None
        for value in np.asarray(values, dtype=np.float64)
    ]


def _aggregate_metrics(
    summaries: list[dict], path: callable
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for metric in METRICS:
        metrics[metric] = _mean_std_ci(
            [path(summary).get(metric) for summary in summaries]
        )
    return metrics


def aggregate_exp1() -> dict:
    """Aggregate behavior-mismatch results across seeds."""
    summaries = [summary for summary, _ in _seed_records(OUTPUT_BASE / "exp1_behavior")]
    if not summaries:
        return {"n_seeds": 0}

    result: dict = {
        "n_seeds": len(summaries),
        "fit": {},
        "behavior_disagreement": {},
        "approximation": {},
    }
    for key in ("train_accuracy", "test_accuracy"):
        result["fit"][key] = _mean_std_ci(
            [summary["fit"][key] for summary in summaries]
        )

    for pair in summaries[0]["one_step"]["exact_behavior_disagreement"]:
        entries = [
            summary["one_step"]["exact_behavior_disagreement"][pair]
            for summary in summaries
        ]
        result["behavior_disagreement"][pair] = {
            "kendall_tau": _mean_std_ci([entry["mean_kendall_tau"] for entry in entries]),
            "sign_accuracy": _mean_std_ci([entry["mean_sign_agreement"] for entry in entries]),
            "top_5pct_overlap": _mean_std_ci([entry["mean_top_5pct_overlap"] for entry in entries]),
        }

    for behavior in summaries[0]["specification"]["behaviors"]:
        result["approximation"][behavior] = _aggregate_metrics(
            summaries,
            lambda summary: summary["one_step"]["approximation_vs_exact"][behavior],
        )
    return result


def aggregate_exp2() -> dict:
    """Aggregate transition-mismatch results across seeds."""
    summaries = [summary for summary, _ in _seed_records(OUTPUT_BASE / "exp2_transition")]
    if not summaries:
        return {"n_seeds": 0}

    result: dict = {
        "n_seeds": len(summaries),
        "transition_comparisons": {},
        "first_order": {},
    }
    for pair in summaries[0]["transition_comparisons"]:
        result["transition_comparisons"][pair] = _aggregate_metrics(
            summaries, lambda summary: summary["transition_comparisons"][pair]
        )
    for transition in summaries[0]["first_order_approximations"]:
        result["first_order"][transition] = _aggregate_metrics(
            summaries,
            lambda summary: summary["first_order_approximations"][transition],
        )
    return result


def aggregate_exp3() -> dict:
    """Aggregate the one-step step-size sweep across seeds."""
    base = OUTPUT_BASE / "exp3_stepsize"
    result: dict = {"etas": {}}
    for eta_dir in sorted(base.glob("eta_*"), key=lambda path: float(path.name.split("_")[1])):
        summaries = [summary for summary, _ in _seed_records(eta_dir)]
        if not summaries:
            continue

        eta = eta_dir.name.split("_", 1)[1]
        result["etas"][eta] = {"n_seeds": len(summaries)}
        for behavior in summaries[0]["specification"]["behaviors"]:
            result["etas"][eta][behavior] = _aggregate_metrics(
                summaries,
                lambda summary: summary["one_step"]["approximation_vs_exact"][behavior],
            )
    return result


def aggregate_exp4() -> dict:
    """Aggregate perturbation-scale and inverse-Hessian approximation results."""
    summaries = [summary for summary, _ in _exp4_records()]
    if not summaries:
        return {"n_seeds": 0}

    result: dict = {
        "n_seeds": len(summaries),
        "comparisons": {},
        "inverse_hessian_approximation": {},
        "reopt_diagnostics": {},
    }
    for pair in summaries[0]["comparisons"]:
        result["comparisons"][pair] = _aggregate_metrics(
            summaries, lambda summary: summary["comparisons"][pair]
        )
    for alpha_key in summaries[0]["inverse_hessian_approximation"]:
        result["inverse_hessian_approximation"][alpha_key] = _aggregate_metrics(
            summaries,
            lambda summary: summary["inverse_hessian_approximation"][alpha_key],
        )

    diagnostic_keys = (
        "max_candidate_gradient_norm",
        "mean_candidate_gradient_norm",
        "max_candidate_iterations",
        "mean_candidate_iterations",
    )
    for alpha_key in summaries[0]["reopt_diagnostics"]:
        result["reopt_diagnostics"][alpha_key] = {
            key: _mean_std_ci(
                [summary["reopt_diagnostics"][alpha_key][key] for summary in summaries]
            )
            for key in diagnostic_keys
        }
    return result


def _finding1_figure_data() -> list[dict]:
    comparisons: list[dict] = []

    exp1_records = _seed_records(OUTPUT_BASE / "exp1_behavior")
    exp1_seeds = [_seed_id(scores_path) for _, scores_path in exp1_records]
    behavior_pairs = (
        ("behavior_soft_margin", "soft_margin"),
        ("behavior_hard_margin", "hard_margin"),
        ("behavior_query_logit", "target_logit"),
    )
    behavior_labels = {
        "soft_margin": "soft margin",
        "hard_margin": "hard margin",
        "target_logit": "query logit",
    }
    for key, behavior in behavior_pairs:
        tau_by_seed = []
        for _, scores_path in exp1_records:
            data = np.load(scores_path)
            behaviors = [str(value) for value in data["behaviors"]]
            index = behaviors.index(behavior)
            query_loss = data["exact_one_step"][:, :, behaviors.index("negative_loss")]
            tau_by_seed.append(_per_query_taus(query_loss, data["exact_one_step"][:, :, index]))
        comparisons.append(
            {
                "key": key,
                "axis": "B",
                "label": behavior_labels[behavior],
                "seeds": exp1_seeds,
                "reference_behavior": "negative_loss",
                "comparison_behavior": behavior,
                "tau_by_seed": [_portable_floats(values) for values in tau_by_seed],
            }
        )

    exp4_records = _exp4_records()
    exp4_seeds = [_seed_id(scores_path) for _, scores_path in exp4_records]
    perturbation_pairs = (
        ("perturbation_1e-3", 1e-3, r"$\alpha=10^{-3}$"),
        ("perturbation_1e-1", 0.1, r"$\alpha=10^{-1}$"),
    )
    for key, alpha, label in perturbation_pairs:
        tau_by_seed = []
        for _, scores_path in exp4_records:
            data = np.load(scores_path)
            if str(data["behavior"]) != "negative_loss":
                raise ValueError(f"Unexpected Exp4 behavior in {scores_path}")
            alpha_keys = {float(value): f"alpha_{value:.10g}" for value in data["alphas"]}
            tau_by_seed.append(
                _per_query_taus(data[alpha_keys[1e-5]], data[alpha_keys[alpha]])
            )
        comparisons.append(
            {
                "key": key,
                "axis": "P",
                "label": label,
                "seeds": exp4_seeds,
                "base_alpha": 1e-5,
                "alpha": alpha,
                "tau_by_seed": [_portable_floats(values) for values in tau_by_seed],
            }
        )

    exp2_records = _seed_records(OUTPUT_BASE / "exp2_transition")
    exp2_seeds = [_seed_id(scores_path) for _, scores_path in exp2_records]
    transition_pairs = (
        ("transition_multi_step", "exact_multi_step", "multi-step"),
        ("transition_inverse_hessian", "inverse_hessian", "inverse-Hessian"),
    )
    for key, transition, label in transition_pairs:
        tau_by_seed = []
        for _, scores_path in exp2_records:
            data = np.load(scores_path)
            tau_by_seed.append(_per_query_taus(data["exact_one_step"], data[transition]))
        comparisons.append(
            {
                "key": key,
                "axis": "T",
                "label": label,
                "seeds": exp2_seeds,
                "reference_transition": "exact_one_step",
                "comparison_transition": transition,
                "tau_by_seed": [_portable_floats(values) for values in tau_by_seed],
            }
        )

    return comparisons


def _finding2_figure_data() -> dict:
    one_step: list[dict] = []
    eta_base = OUTPUT_BASE / "exp3_stepsize"
    for eta_dir in sorted(eta_base.glob("eta_*"), key=lambda path: float(path.name.split("_")[1])):
        eta = float(eta_dir.name.split("_", 1)[1])
        for behavior in ("negative_loss", "hard_margin", "target_logit"):
            records = _seed_records(eta_dir)
            seeds = [_seed_id(scores_path) for _, scores_path in records]
            tau_by_seed = []
            for _, scores_path in records:
                data = np.load(scores_path)
                behaviors = [str(value) for value in data["behaviors"]]
                index = behaviors.index(behavior)
                tau_by_seed.append(
                    _mean_per_query_tau(
                        data["exact_one_step"][:, :, index],
                        data["first_order"][:, :, index],
                    )
                )
            one_step.append(
                {
                    "eta": eta,
                    "behavior": behavior,
                    "seeds": seeds,
                    "reference": "exact_one_step",
                    "estimate": "first_order",
                    "tau_by_seed": _portable_floats(tau_by_seed),
                }
            )

    reoptimization: list[dict] = []
    exp4_records = _exp4_records()
    for _, scores_path in exp4_records:
        seed = _seed_id(scores_path)
        data = np.load(scores_path)
        if str(data["behavior"]) != "negative_loss":
            raise ValueError(f"Unexpected Exp4 behavior in {scores_path}")
        for alpha in data["alphas"]:
            reoptimization.append(
                {
                    "alpha": float(alpha),
                    "seed": seed,
                    "tau": _mean_per_query_tau(
                        data[f"alpha_{float(alpha):.10g}"], data["exact_if"]
                    ),
                }
            )

    reoptimization_by_alpha: dict[float, list[tuple[int, float]]] = {}
    for entry in reoptimization:
        reoptimization_by_alpha.setdefault(entry["alpha"], []).append(
            (entry["seed"], entry["tau"])
        )
    reoptimization_data = [
        {
            "alpha": alpha,
            "seeds": [seed for seed, _ in values],
            "behavior": "negative_loss",
            "reference": "newton_reoptimization",
            "estimate": "inverse_hessian",
            "tau_by_seed": _portable_floats([tau for _, tau in values]),
        }
        for alpha, values in sorted(reoptimization_by_alpha.items())
    ]

    return {"one_step": one_step, "reoptimization": reoptimization_data}


def figure_data() -> dict:
    figure = {
        "finding1": {"comparisons": _finding1_figure_data()},
        "finding2": _finding2_figure_data(),
    }
    _validate_figure_data(figure)
    return figure


def _validate_figure_data(figure: dict) -> None:
    comparisons = figure["finding1"]["comparisons"]
    if not comparisons:
        raise ValueError("Figure data require at least one Finding 1 comparison")

    finding1_seed_counts = {len(item["tau_by_seed"]) for item in comparisons}
    if len(finding1_seed_counts) != 1:
        raise ValueError("Finding 1 comparisons have inconsistent seed counts")
    for item in comparisons:
        if len(item["seeds"]) != len(item["tau_by_seed"]):
            raise ValueError("Finding 1 seed labels do not match tau values")

    one_step = figure["finding2"]["one_step"]
    one_step_keys = {(item["eta"], item["behavior"]) for item in one_step}
    if len(one_step_keys) != len(one_step):
        raise ValueError("Finding 2 one-step data contain duplicate entries")
    one_step_seed_counts = {len(item["tau_by_seed"]) for item in one_step}
    if len(one_step_seed_counts) != 1:
        raise ValueError("Finding 2 one-step data have inconsistent seed counts")
    for item in one_step:
        if len(item["seeds"]) != len(item["tau_by_seed"]):
            raise ValueError("Finding 2 one-step seed labels do not match tau values")

    reoptimization = figure["finding2"]["reoptimization"]
    reoptimization_alphas = {item["alpha"] for item in reoptimization}
    if len(reoptimization_alphas) != len(reoptimization):
        raise ValueError("Finding 2 reoptimization data contain duplicate alphas")
    reoptimization_seed_counts = {len(item["tau_by_seed"]) for item in reoptimization}
    if len(reoptimization_seed_counts) != 1:
        raise ValueError("Finding 2 reoptimization data have inconsistent seed counts")
    for item in reoptimization:
        if len(item["seeds"]) != len(item["tau_by_seed"]):
            raise ValueError("Finding 2 reoptimization seed labels do not match tau values")


def _fmt(stat: dict, decimals: int = 3) -> str:
    if stat["n"] == 0:
        return "n/a"
    return f"{stat['mean']:.{decimals}f} +/- {stat['ci95']:.{decimals}f}"


def main() -> None:
    aggregated = {
        "schema_version": SCHEMA_VERSION,
        "exp1_behavior": aggregate_exp1(),
        "exp2_transition": aggregate_exp2(),
        "exp3_stepsize": aggregate_exp3(),
        "exp4_perturbation_scale": aggregate_exp4(),
        "figure": figure_data(),
    }
    output_path = OUTPUT_BASE / "aggregated_results.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(aggregated, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved aggregated results to {output_path}")

    exp1 = aggregated["exp1_behavior"]
    print()
    print("=" * 70)
    print(f"EXPERIMENT 1: Behavior mismatch ({exp1['n_seeds']} seeds)")
    print("=" * 70)
    print(
        "Model fit: "
        f"train accuracy={_fmt(exp1['fit']['train_accuracy'])}, "
        f"test accuracy={_fmt(exp1['fit']['test_accuracy'])}"
    )
    print("Exact behavior disagreement:")
    for pair, values in exp1["behavior_disagreement"].items():
        print(
            f"  {pair:42s} tau={_fmt(values['kendall_tau'])} "
            f"top5%={_fmt(values['top_5pct_overlap'])} "
            f"sign={_fmt(values['sign_accuracy'])}"
        )

    exp2 = aggregated["exp2_transition"]
    print()
    print("=" * 70)
    print(f"EXPERIMENT 2: Transition mismatch ({exp2['n_seeds']} seeds)")
    print("=" * 70)
    print("Exact transition disagreement:")
    for pair, values in exp2["transition_comparisons"].items():
        print(
            f"  {pair:35s} tau={_fmt(values['kendall_tau'])} "
            f"top5%={_fmt(values['top_5pct_overlap'])}"
        )
    print("First-order approximation by transition:")
    for transition, values in exp2["first_order"].items():
        print(f"  {transition:35s} tau={_fmt(values['kendall_tau'])}")

    exp3 = aggregated["exp3_stepsize"]
    print()
    print("=" * 70)
    print(f"EXPERIMENT 3: Step-size sweep ({exp3['etas'] and next(iter(exp3['etas'].values()))['n_seeds']} seeds per eta)")
    print("=" * 70)
    for eta, values in exp3["etas"].items():
        taus = [
            f"{behavior}={values[behavior]['kendall_tau']['mean']:.3f}"
            for behavior in ("negative_loss", "hard_margin", "target_logit")
            if behavior in values
        ]
        print(f"  eta={eta:>5s}: " + ", ".join(taus))

    exp4 = aggregated["exp4_perturbation_scale"]
    print()
    print("=" * 70)
    print(f"EXPERIMENT 4: Perturbation scale ({exp4['n_seeds']} seeds)")
    print("=" * 70)
    print("Inverse-Hessian response vs Newton re-optimization:")
    for alpha_key, values in exp4["inverse_hessian_approximation"].items():
        print(
            f"  {alpha_key:16s} tau={_fmt(values['kendall_tau'])} "
            f"top5%={_fmt(values['top_5pct_overlap'])}"
        )
    print("Maximum reoptimization gradient norm:")
    for alpha_key, values in exp4["reopt_diagnostics"].items():
        print(
            f"  {alpha_key:16s} "
            f"{values['max_candidate_gradient_norm']['mean']:.3e} "
            f"(max iterations {values['max_candidate_iterations']['mean']:.1f})"
        )


if __name__ == "__main__":
    main()
