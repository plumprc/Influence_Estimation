"""Aggregate CIFAR-10 controlled-experiment outputs into one portable JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


OUTPUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "cifar10"
AGGREGATED_PATH = OUTPUT_BASE / "aggregated_results.json"
SCHEMA_VERSION = 9

ETAS = (0.01, 0.05, 0.1, 0.3, 0.5)
ALPHAS = (1e-5, 1e-3, 1e-1)
BEHAVIOR_ORDER = ("negative_loss", "target_logit", "soft_margin", "hard_margin")
FINDING2_BEHAVIORS = ("negative_loss", "hard_margin", "target_logit")


def _seed_dirs(experiment_dir: Path, seeds: list[int]) -> list[Path]:
    seed_dirs = [experiment_dir / f"seed_{seed}" for seed in seeds]
    missing = [path for path in seed_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing seed directories: " + ", ".join(str(path) for path in missing)
        )
    return seed_dirs


def _load_summary(seed_dir: Path) -> dict:
    path = seed_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_scores(seed_dir: Path) -> dict[str, np.ndarray]:
    path = seed_dir / "scores.npz"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty scores: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _mean_std_ci(values) -> dict[str, float | int]:
    array = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    if array.size > 1:
        critical = float(stats.t.ppf(0.975, df=array.size - 1))
        ci95 = critical * std / np.sqrt(array.size)
    else:
        ci95 = 0.0
    return {"mean": mean, "std": std, "ci95": float(ci95), "n": int(array.size)}


def _portable_floats(values: np.ndarray) -> list[float | None]:
    return [
        float(value) if np.isfinite(value) else None
        for value in np.asarray(values, dtype=np.float64)
    ]


def _flatten_seed_values(values: list[list[float | None]]) -> list[float | None]:
    return [value for seed_values in values for value in seed_values]


def _per_query_taus(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Expected matching two-dimensional score matrices")
    return np.asarray(
        [stats.kendalltau(first[q], second[q]).statistic for q in range(first.shape[0])],
        dtype=np.float64,
    )


def _mean_per_query_tau(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.nanmean(_per_query_taus(first, second)))


def _behavior_index(scores: dict[str, np.ndarray], behavior: str) -> int:
    behaviors = [str(value) for value in scores.get("behaviors", BEHAVIOR_ORDER)]
    if behavior not in behaviors:
        raise ValueError(f"Behavior {behavior!r} is missing from scores")
    return behaviors.index(behavior)


def _negative_loss_scores(scores: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    """Support both legacy all-behavior arrays and current single-behavior arrays."""

    if "behavior" in scores:
        behavior = str(scores["behavior"])
        if behavior != "negative_loss":
            raise ValueError(f"Exp5c currently aggregates negative_loss, found {behavior!r}")
        if values.ndim == 2:
            return values
    elif values.ndim != 3:
        raise ValueError("Exp5c scores must be two- or three-dimensional")

    return values[:, :, BEHAVIOR_ORDER.index("negative_loss")]


def _aggregate_metrics(summaries: list[dict], path) -> dict[str, dict]:
    return {
        metric: _mean_std_ci([path(summary).get(metric) for summary in summaries])
        for metric in ("kendall_tau", "sign_accuracy", "top_5pct_overlap")
    }


def _aggregate_5a_metrics(summaries: list[dict], pair: str) -> dict[str, dict]:
    entries = [summary["behavior_disagreement"][pair] for summary in summaries]
    return {
        "kendall_tau": _mean_std_ci(
            [entry["mean_kendall_tau"] for entry in entries]
        ),
        "sign_accuracy": _mean_std_ci(
            [entry["mean_sign_agreement"] for entry in entries]
        ),
        "top_5pct_overlap": _mean_std_ci(
            [entry["mean_top_5pct_overlap"] for entry in entries]
        ),
    }


def aggregate_exp5a(seeds: list[int]) -> dict:
    """Aggregate behavior-mismatch results across seeds."""

    seed_dirs = _seed_dirs(OUTPUT_BASE / "exp5a_cifar10_behavior", seeds)
    summaries = [_load_summary(path) for path in seed_dirs]

    per_query: dict[str, list[list[float]]] = {
        behavior: [] for behavior in BEHAVIOR_ORDER[1:]
    }
    for seed_dir in seed_dirs:
        scores = _load_scores(seed_dir)
        exact = scores["exact"]
        behaviors = [str(value) for value in scores.get("behaviors", BEHAVIOR_ORDER)]
        reference = exact[:, :, behaviors.index("negative_loss")]
        for behavior in ("soft_margin", "hard_margin", "target_logit"):
            per_query[behavior].append(
                _portable_floats(
                    _per_query_taus(reference, exact[:, :, behaviors.index(behavior)])
                )
            )

    summary = {
        "n_seeds": len(seed_dirs),
        "seeds": seeds,
        "per_query_kendall_tau": {
            behavior: _flatten_seed_values(values)
            for behavior, values in per_query.items()
        },
        "per_query_kendall_tau_by_seed": per_query,
        "summary": {
            "per_query_kendall_tau": {
                behavior: _mean_std_ci(_flatten_seed_values(values))
                for behavior, values in per_query.items()
            },
            "behavior_disagreement": {
                pair: _aggregate_5a_metrics(summaries, pair)
                for pair in summaries[0]["behavior_disagreement"]
            },
        },
    }
    return summary


def aggregate_exp5d(seeds: list[int]) -> dict:
    """Aggregate the step-size sweep across seeds."""

    seed_dirs = _seed_dirs(OUTPUT_BASE / "exp5d_cifar10_stepsize", seeds)
    summaries = [_load_summary(path) for path in seed_dirs]

    behavior_means: dict[tuple[float, str], list[float]] = {}
    for seed_dir, summary in zip(seed_dirs, summaries):
        scores = _load_scores(seed_dir)
        configured_etas = [float(eta) for eta in summary["specification"]["step_sizes"]]
        if tuple(configured_etas) != ETAS:
            raise ValueError(
                f"Unexpected step sizes in {seed_dir}: expected {ETAS}, got {configured_etas}"
            )
        for eta in ETAS:
            eta_key = f"eta_{eta:g}"
            exact = scores[f"{eta_key}_exact"]
            first_order = scores[f"{eta_key}_first_order"]
            for behavior in FINDING2_BEHAVIORS:
                index = BEHAVIOR_ORDER.index(behavior)
                behavior_means[(eta, behavior)] = behavior_means.get(
                    (eta, behavior), []
                ) + [
                    _mean_per_query_tau(
                        exact[:, :, index],
                        first_order[:, :, index],
                    )
                ]

    results: dict[str, dict[str, dict]] = {}
    for eta in ETAS:
        eta_key = f"{eta:g}"
        results[eta_key] = {
            behavior: {
                **_mean_std_ci(behavior_means[(eta, behavior)]),
                "tau_by_seed": _portable_floats(
                    np.asarray(behavior_means[(eta, behavior)], dtype=np.float64)
                ),
            }
            for behavior in FINDING2_BEHAVIORS
        }

    return {
        "n_seeds": len(seed_dirs),
        "seeds": seeds,
        "etas": [f"{eta:g}" for eta in ETAS],
        "behaviors": list(FINDING2_BEHAVIORS),
        "results": results,
        "summary": {
            "sweep_results": {
                f"{eta:g}": {
                    behavior: _aggregate_metrics(
                        summaries,
                        lambda summary, eta=eta, behavior=behavior: summary[
                            "sweep_results"
                        ][f"{eta:g}"][behavior],
                    )
                    for behavior in FINDING2_BEHAVIORS
                }
                for eta in ETAS
            }
        },
    }


def aggregate_exp5c(seeds: list[int]) -> dict:
    """Aggregate transition-axis results across the requested seeds."""

    seed_dirs = _seed_dirs(OUTPUT_BASE / "exp5c_cifar10_transition_axes", seeds)
    summaries = [_load_summary(path) for path in seed_dirs]

    comparisons = {
        "one_step_vs_multi_step": [],
        "one_step_vs_ihvp": [],
    }
    for seed_dir in seed_dirs:
        scores = _load_scores(seed_dir)
        one_step = _negative_loss_scores(scores, scores["exact_one_step"])
        multi_step = _negative_loss_scores(scores, scores["exact_multi_step"])
        ihvp = _negative_loss_scores(scores, scores["ihvp"])
        comparisons["one_step_vs_multi_step"].append(
            _portable_floats(_per_query_taus(one_step, multi_step))
        )
        comparisons["one_step_vs_ihvp"].append(
            _portable_floats(_per_query_taus(one_step, ihvp))
        )

    return {
        "n_seeds": len(seed_dirs),
        "seeds": seeds,
        "per_query_kendall_tau": {
            pair: [value for seed_values in values for value in seed_values]
            for pair, values in comparisons.items()
        },
        "per_query_kendall_tau_by_seed": comparisons,
        "summary": {
            "transition_comparisons": {
                pair: _aggregate_metrics(
                    summaries,
                    lambda summary, pair=pair: summary["transition_comparisons"][pair][
                        "negative_loss"
                    ],
                )
                for pair in comparisons
            }
        },
    }


def aggregate_exp5b(seeds: list[int]) -> dict:
    """Aggregate perturbation-axis results across the requested seeds."""

    seed_dirs = _seed_dirs(OUTPUT_BASE / "exp5b_cifar10_perturbation_axis", seeds)
    summaries = [_load_summary(path) for path in seed_dirs]

    pairwise: dict[tuple[float, float], list[list[float]]] = {}
    approximation: dict[float, list[list[float]]] = {}
    loo_approximation: dict[float, list[list[float]]] = {}
    loo_vs_ihvp: list[list[float]] = []
    for seed_dir in seed_dirs:
        scores = _load_scores(seed_dir)
        alphas = [float(value) for value in scores["alphas"]]
        if tuple(alphas) != ALPHAS:
            raise ValueError(
                f"Unexpected alphas in {seed_dir}: expected {ALPHAS}, got {alphas}"
            )
        ihvp = scores["ihvp"]
        if "loo" not in scores:
            raise ValueError(f"Exp5b scores must contain loo in {seed_dir}")
        loo = scores["loo"]
        if loo.shape != ihvp.shape:
            raise ValueError(
                f"Exp5b LOO and inverse-Hessian scores have different shapes in {seed_dir}"
            )
        alpha_scores = {
            alpha: scores[f"alpha_{alpha:.10g}"] for alpha in ALPHAS
        }
        for first_index, first_alpha in enumerate(ALPHAS):
            for second_alpha in ALPHAS[first_index + 1 :]:
                pairwise.setdefault((first_alpha, second_alpha), []).append(
                    _portable_floats(
                        _per_query_taus(
                            alpha_scores[first_alpha],
                            alpha_scores[second_alpha],
                        )
                    )
                )
        for alpha in ALPHAS:
            approximation.setdefault(alpha, []).append(
                _portable_floats(
                    _per_query_taus(ihvp, alpha_scores[alpha])
                )
            )
            loo_approximation.setdefault(alpha, []).append(
                _portable_floats(_per_query_taus(loo, alpha_scores[alpha]))
            )
        loo_vs_ihvp.append(_portable_floats(_per_query_taus(loo, ihvp)))

    return {
        "n_seeds": len(seed_dirs),
        "seeds": seeds,
        "alphas": [f"{alpha:.10g}" for alpha in ALPHAS],
        "pairwise_kendall_tau": {
            f"{first_alpha:.10g}_vs_{second_alpha:.10g}": {
                **_mean_std_ci(_flatten_seed_values(values)),
                "tau_by_seed": _flatten_seed_values(values),
                "per_query_tau_by_seed": values,
            }
            for (first_alpha, second_alpha), values in pairwise.items()
        },
        "inverse_hessian_approximation": {
            f"{alpha:.10g}": {
                **_mean_std_ci(_flatten_seed_values(values)),
                "tau_by_seed": _flatten_seed_values(values),
                "per_query_tau_by_seed": values,
            }
            for alpha, values in approximation.items()
        },
        "loo_approximation": {
            f"{alpha:.10g}": {
                **_mean_std_ci(_flatten_seed_values(values)),
                "tau_by_seed": _flatten_seed_values(values),
                "per_query_tau_by_seed": values,
            }
            for alpha, values in loo_approximation.items()
        },
        "loo_vs_ihvp": {
            **_mean_std_ci(_flatten_seed_values(loo_vs_ihvp)),
            "tau_by_seed": _flatten_seed_values(loo_vs_ihvp),
            "per_query_tau_by_seed": loo_vs_ihvp,
        },
        "summary": {
            "comparisons": {
                key: _aggregate_metrics(
                    summaries,
                    lambda summary, key=key: summary["comparisons"][key],
                )
                for key in summaries[0]["comparisons"]
            },
            "inverse_hessian_approximation": {
                key: _aggregate_metrics(
                    summaries,
                    lambda summary, key=key: summary["inverse_hessian_approximation"][key],
                )
                for key in summaries[0]["inverse_hessian_approximation"]
            },
            "loo_vs_alpha": {
                key: _aggregate_metrics(
                    summaries,
                    lambda summary, key=key: summary["loo_vs_alpha"][key],
                )
                for key in summaries[0]["loo_vs_alpha"]
            },
            "loo_vs_ihvp": _aggregate_metrics(
                summaries,
                lambda summary: summary["loo_vs_ihvp"],
            ),
        },
    }


def _finding1_figure_data(
    exp5a: dict,
    exp5b: dict,
    exp5c: dict,
) -> list[dict]:
    comparisons: list[dict] = []

    behavior_pairs = (
        ("behavior_soft_margin", "soft_margin", "soft margin"),
        ("behavior_hard_margin", "hard_margin", "hard margin"),
        ("behavior_query_logit", "target_logit", "query logit"),
    )
    for key, behavior, label in behavior_pairs:
        comparisons.append(
            {
                "key": key,
                "axis": "B",
                "label": label,
                "seeds": exp5a["seeds"],
                "reference_behavior": "negative_loss",
                "comparison_behavior": behavior,
                "tau_by_seed": exp5a["per_query_kendall_tau_by_seed"][behavior],
            }
        )

    perturbation_pairs = (
        ("upweight_1e-3_vs_loo", 1e-3, r"$\alpha=10^{-3}$"),
        ("upweight_1e-1_vs_loo", 0.1, r"$\alpha=10^{-1}$"),
    )
    for key, alpha, label in perturbation_pairs:
        alpha_key = f"{alpha:.10g}"
        comparisons.append(
            {
                "key": key,
                "axis": "P",
                "label": label,
                "seeds": exp5b["seeds"],
                "alpha": alpha,
                "reference": "local_loo_reoptimization",
                "estimate": "local_upweight_reoptimization",
                "tau_by_seed": exp5b["loo_approximation"][alpha_key][
                    "per_query_tau_by_seed"
                ],
            }
        )

    transition_pairs = (
        ("transition_multi_step", "one_step_vs_multi_step", "multi-step"),
        ("transition_inverse_hessian", "one_step_vs_ihvp", "inverse-Hessian"),
    )
    for key, pair, label in transition_pairs:
        comparisons.append(
            {
                "key": key,
                "axis": "T",
                "label": label,
                "seeds": exp5c["seeds"],
                "reference_transition": "exact_one_step",
                "comparison_transition": pair,
                "tau_by_seed": exp5c["per_query_kendall_tau_by_seed"][pair],
            }
        )

    return comparisons


def _finding2_figure_data(exp5d: dict, exp5b: dict) -> dict:
    one_step: list[dict] = []
    for eta in ETAS:
        eta_key = f"{eta:g}"
        for behavior in FINDING2_BEHAVIORS:
            values = exp5d["results"][eta_key][behavior]
            one_step.append(
                {
                    "eta": eta,
                    "behavior": behavior,
                    "seeds": exp5d["seeds"],
                    "reference": "exact_one_step",
                    "estimate": "first_order",
                    "tau_by_seed": values["tau_by_seed"],
                }
            )

    reoptimization: list[dict] = []
    for alpha in ALPHAS:
        alpha_key = f"{alpha:.10g}"
        values = exp5b["inverse_hessian_approximation"][alpha_key]
        reoptimization.append(
            {
                "alpha": alpha,
                "seeds": exp5b["seeds"],
                "behavior": "negative_loss",
                "reference": "local_reoptimization",
                "estimate": "inverse_hessian",
                "tau_by_seed": values["per_query_tau_by_seed"],
            }
        )

    return {"one_step": one_step, "reoptimization": reoptimization}


def _validate_figure_data(figure: dict) -> None:
    comparisons = figure["finding1"]["comparisons"]
    expected_keys = [
        "behavior_soft_margin",
        "behavior_hard_margin",
        "behavior_query_logit",
        "upweight_1e-3_vs_loo",
        "upweight_1e-1_vs_loo",
        "transition_multi_step",
        "transition_inverse_hessian",
    ]
    actual_keys = [item["key"] for item in comparisons]
    if actual_keys != expected_keys:
        raise ValueError(
            f"Unexpected Finding 1 comparison keys: expected {expected_keys}, got {actual_keys}"
        )
    for item in comparisons:
        if len(item["tau_by_seed"]) != len(item["seeds"]):
            raise ValueError(f"Finding 1 comparison {item['key']} has invalid seed data")
        for values in item["tau_by_seed"]:
            if not values:
                raise ValueError(f"Finding 1 comparison {item['key']} has empty tau values")
            if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                raise ValueError(
                    f"Finding 1 comparison {item['key']} has non-finite tau values"
                )
    one_step = figure["finding2"]["one_step"]
    expected_one_step_count = len(ETAS) * len(FINDING2_BEHAVIORS)
    if len(one_step) != expected_one_step_count:
        raise ValueError(
            f"Expected {expected_one_step_count} one-step entries, got {len(one_step)}"
        )
    for item in one_step:
        values = np.asarray(item["tau_by_seed"], dtype=np.float64)
        if (
            len(item["seeds"]) != values.size
            or values.size == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"One-step entry has invalid tau values: {item}")

    reoptimization = figure["finding2"]["reoptimization"]
    if len(reoptimization) != len(ALPHAS):
        raise ValueError(
            f"Expected {len(ALPHAS)} reoptimization entries, got {len(reoptimization)}"
        )
    for item in reoptimization:
        values = np.asarray(
            [value for seed in item["tau_by_seed"] for value in seed],
            dtype=np.float64,
        )
        if (
            len(item["seeds"]) != len(item["tau_by_seed"])
            or values.size == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Reoptimization entry has invalid tau values: {item}")



def figure_data(
    exp5a: dict,
    exp5b: dict,
    exp5c: dict,
    exp5d: dict,
) -> dict:
    figure = {
        "finding1": {
            "comparisons": _finding1_figure_data(exp5a, exp5b, exp5c)
        },
        "finding2": _finding2_figure_data(exp5d, exp5b),
    }
    _validate_figure_data(figure)
    return figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exps",
        nargs="+",
        choices=("5a", "5b", "5c", "5d"),
        default=["5a", "5b", "5c", "5d"],
        help="Experiment families to aggregate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Exact seed scope for 5a and 5d",
    )
    parser.add_argument(
        "--single-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Exact seed scope for 5b and 5c",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Preserve experiment sections absent from --exps using the existing aggregate",
    )
    return parser.parse_args()


def _save_json_atomic(path: Path, value: dict) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    temporary_path.replace(path)


def main() -> None:
    args = _parse_args()
    seeds = sorted(set(args.seeds))
    single_seeds = sorted(set(args.single_seeds))
    if not seeds or not single_seeds:
        raise ValueError("At least one seed is required for each seed scope")
    if any(seed < 0 for seed in seeds + single_seeds):
        raise ValueError("Seeds must be non-negative")

    aggregators = {
        "exp5a_behavior": lambda: aggregate_exp5a(seeds),
        "exp5b_perturbation_axis": lambda: aggregate_exp5b(single_seeds),
        "exp5c_transition_axes": lambda: aggregate_exp5c(single_seeds),
        "exp5d_stepsize": lambda: aggregate_exp5d(seeds),
    }
    experiment_names = {
        "exp5a_behavior": "5a",
        "exp5b_perturbation_axis": "5b",
        "exp5c_transition_axes": "5c",
        "exp5d_stepsize": "5d",
    }
    aggregated_experiments = {
        key: aggregator()
        for key, aggregator in aggregators.items()
        if experiment_names[key] in args.exps
    }

    existing = {}
    if args.merge_existing and AGGREGATED_PATH.exists():
        with AGGREGATED_PATH.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        missing_keys = set(aggregators) - set(aggregated_experiments)
        aggregated_experiments.update(
            {key: existing[key] for key in missing_keys if key in existing}
        )

    if set(aggregated_experiments) == set(aggregators):
        figure = figure_data(
            **{
                "exp5a": aggregated_experiments["exp5a_behavior"],
                "exp5b": aggregated_experiments["exp5b_perturbation_axis"],
                "exp5c": aggregated_experiments["exp5c_transition_axes"],
                "exp5d": aggregated_experiments["exp5d_stepsize"],
            }
        )
    else:
        figure = {}

    aggregated = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "cifar10",
        "experiment_scope": sorted(
            experiment_names[key] for key in aggregated_experiments
        ),
        "seed_scope": {
            "5a_5d": seeds,
            "5b_5c": single_seeds,
        },
        "behavior_order": list(BEHAVIOR_ORDER),
        **aggregated_experiments,
    }
    if figure:
        aggregated["figure"] = figure

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    _save_json_atomic(AGGREGATED_PATH, aggregated)

    print(f"Saved aggregated CIFAR-10 results to {AGGREGATED_PATH}")
    print(
        "Seeds: "
        + ", ".join(
            f"{key}={value['n_seeds']}"
            for key, value in aggregated_experiments.items()
        )
    )


if __name__ == "__main__":
    main()
