"""Aggregate CIFAR-10 experiment outputs into a single strict JSON file.

The resulting file contains every per-query distribution and summary statistic
needed by ``plot/plot_cifar10.py``. Once it has been generated, the plot no
longer needs the raw ``scores.npz`` files.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


OUTPUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "cifar10"
AGGREGATED_PATH = OUTPUT_BASE / "aggregated_results.json"


def _seed_dirs(experiment_dir: Path) -> list[Path]:
    return sorted(experiment_dir.glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))


def _load_summary(seed_dir: Path) -> dict:
    summary_path = seed_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    with summary_path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _mean_std(values) -> dict:
    array = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n": int(array.size),
    }


def _json_list(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _per_query_taus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(
        [stats.kendalltau(a[q], b[q]).statistic for q in range(a.shape[0])],
        dtype=np.float64,
    )


def _aggregate_nested_metrics(summaries: list[dict], section: str) -> dict:
    if not summaries:
        return {}
    result: dict[str, dict[str, dict]] = {}
    for key in summaries[0][section]:
        metrics = summaries[0][section][key]
        result[key] = {
            metric: _mean_std([summary[section][key][metric] for summary in summaries])
            for metric in metrics
        }
    return result


def aggregate_exp5a() -> dict:
    """Aggregate behavior-mismatch results across seeds."""
    seed_dirs = _seed_dirs(OUTPUT_BASE / "exp5a_cifar10_behavior")
    summaries = [_load_summary(seed_dir) for seed_dir in seed_dirs]

    pools: dict[str, list[float]] = defaultdict(list)
    for seed_dir in seed_dirs:
        summary = _load_summary(seed_dir)
        behaviors = summary["specification"]["behaviors"]
        indices = {behavior: position for position, behavior in enumerate(behaviors)}

        scores_path = seed_dir / "scores.npz"
        if not scores_path.exists():
            raise FileNotFoundError(f"Missing scores: {scores_path}")
        with np.load(scores_path) as data:
            exact = data["exact"]
            negative_loss = exact[:, :, indices["negative_loss"]]
            for behavior in ("soft_margin", "hard_margin", "target_logit"):
                pools[behavior].extend(
                    _per_query_taus(negative_loss, exact[:, :, indices[behavior]])
                )

    return {
        "n_seeds": len(seed_dirs),
        "seeds": [path.name.split("_")[1] for path in seed_dirs],
        "per_query_kendall_tau": {
            behavior: _json_list(np.asarray(values, dtype=np.float64))
            for behavior, values in pools.items()
        },
        "summary": {
            "per_query_kendall_tau": {
                behavior: _mean_std(values) for behavior, values in pools.items()
            },
            "behavior_disagreement": _aggregate_nested_metrics(
                summaries, "behavior_disagreement"
            ),
            "approximation_quality": _aggregate_nested_metrics(
                summaries, "approximation_quality"
            ),
        },
    }


def aggregate_exp5b() -> dict:
    """Aggregate step-size sweep results across seeds."""
    seed_dirs = _seed_dirs(OUTPUT_BASE / "exp5b_cifar10_stepsize")
    summaries = [_load_summary(seed_dir) for seed_dir in seed_dirs]

    etas = sorted(
        {eta for summary in summaries for eta in summary["sweep_results"]},
        key=float,
    )
    behaviors = sorted(
        {
            behavior
            for summary in summaries
            for eta in summary["sweep_results"]
            for behavior in summary["sweep_results"][eta]
        }
    )

    results: dict[str, dict[str, dict]] = {}
    for eta in etas:
        results[eta] = {}
        for behavior in behaviors:
            values = [
                summary["sweep_results"][eta][behavior]["kendall_tau"]
                for summary in summaries
                if eta in summary["sweep_results"]
                and behavior in summary["sweep_results"][eta]
            ]
            results[eta][behavior] = _mean_std(values)

    return {
        "n_seeds": len(seed_dirs),
        "seeds": [path.name.split("_")[1] for path in seed_dirs],
        "etas": etas,
        "behaviors": behaviors,
        "results": results,
    }


def aggregate_exp5c() -> dict:
    """Aggregate exact-vs-IHVP per-query disagreement across seeds."""
    seed_dirs = _seed_dirs(OUTPUT_BASE / "exp5c_cifar10_ihvp")
    summaries = [_load_summary(seed_dir) for seed_dir in seed_dirs]

    pools: dict[str, list[float]] = defaultdict(list)
    for seed_dir in seed_dirs:
        summary = _load_summary(seed_dir)
        behaviors = summary["specification"]["behaviors"]

        scores_path = seed_dir / "scores.npz"
        if not scores_path.exists():
            raise FileNotFoundError(f"Missing scores: {scores_path}")
        with np.load(scores_path) as data:
            exact = data["exact"]
            ihvp = data["ihvp"]
            for position, behavior in enumerate(behaviors):
                pools[behavior].extend(
                    _per_query_taus(exact[:, :, position], ihvp[:, :, position])
                )

    return {
        "n_seeds": len(seed_dirs),
        "seeds": [path.name.split("_")[1] for path in seed_dirs],
        "per_query_kendall_tau": {
            behavior: _json_list(np.asarray(values, dtype=np.float64))
            for behavior, values in pools.items()
        },
        "summary": {
            "per_query_kendall_tau": {
                behavior: _mean_std(values) for behavior, values in pools.items()
            }
        },
    }


def main() -> None:
    aggregated = {
        "schema_version": 1,
        "dataset": "cifar10",
        "behavior_order": [
            "negative_loss",
            "target_logit",
            "soft_margin",
            "hard_margin",
        ],
        "exp5a_behavior": aggregate_exp5a(),
        "exp5b_stepsize": aggregate_exp5b(),
        "exp5c_ihvp": aggregate_exp5c(),
    }

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with AGGREGATED_PATH.open("w", encoding="utf-8") as stream:
        json.dump(aggregated, stream, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"Saved aggregated CIFAR-10 results to {AGGREGATED_PATH}")
    print(
        f"Seeds: 5a={aggregated['exp5a_behavior']['n_seeds']}, "
        f"5b={aggregated['exp5b_stepsize']['n_seeds']}, "
        f"5c={aggregated['exp5c_ihvp']['n_seeds']}"
    )


if __name__ == "__main__":
    main()
