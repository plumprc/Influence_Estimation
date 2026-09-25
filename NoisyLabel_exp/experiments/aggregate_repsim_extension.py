"""Aggregate signal-augmented representation similarity runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate signal-augmented RepSim results"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _mean_std_ci(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * std / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "n": len(values),
    }


def _aggregate_metric_map(
    payloads: list[dict],
    *,
    section: str,
) -> dict[str, dict]:
    first = payloads[0][section]
    result: dict[str, dict] = {}
    for method, behaviors in first.items():
        result[method] = {}
        for behavior, metrics in behaviors.items():
            result[method][behavior] = {
                metric: _mean_std_ci(
                    [float(payload[section][method][behavior][metric]) for payload in payloads]
                )
                for metric in metrics
            }
    return result


def _aggregate_nested_metric_map(
    payloads: list[dict],
    *,
    section: str,
) -> dict[str, dict]:
    first = payloads[0][section]
    result: dict[str, dict] = {}
    for behavior, methods in first.items():
        result[behavior] = {}
        for method, metrics in methods.items():
            if isinstance(metrics, dict):
                result[behavior][method] = {
                    metric: _mean_std_ci(
                        [
                            float(payload[section][behavior][method][metric])
                            for payload in payloads
                        ]
                    )
                    for metric in metrics
                }
            else:
                result[behavior][method] = _mean_std_ci(
                    [
                        float(payload[section][behavior][method])
                        for payload in payloads
                    ]
                )
    return result


def main() -> None:
    args = parse_args()
    seed_dirs = {
        seed: args.output_root / f"rho_{args.rho}" / f"seed_{seed}"
        for seed in args.seeds
    }
    available: dict[int, Path] = {}
    missing: list[int] = []
    for seed, directory in seed_dirs.items():
        summary = directory / "summary.json"
        if summary.is_file():
            available[seed] = summary
        else:
            missing.append(seed)
    if missing and not args.allow_partial:
        raise SystemExit(f"Missing summaries for seeds: {missing}")
    if not available:
        raise SystemExit("No seed summaries were found")

    payloads = [_load_json(path) for path in available.values()]
    method_names = payloads[0]["method_names"]
    behavior_names = payloads[0]["behavior_names"]
    aggregate = {
        "schema_version": 1,
        "stage": "repsim_extension_aggregate",
        "rho": args.rho,
        "seeds": list(available),
        "run_count": len(payloads),
        "method_names": method_names,
        "behavior_names": behavior_names,
        "detection": _aggregate_metric_map(payloads, section="detection"),
        "ranking_agreement_vs_repsim": _aggregate_nested_metric_map(
            payloads,
            section="ranking_agreement_vs_repsim",
        ),
        "topk_overlap_vs_repsim": _aggregate_nested_metric_map(
            payloads,
            section="topk_overlap_vs_repsim",
        ),
        "completion": {
            "expected_seeds": args.seeds,
            "available_seeds": list(available),
            "missing_seeds": missing,
            "complete": not missing,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, indent=2, allow_nan=False)
    temporary.replace(args.output)
    print(f"Saved aggregate to {args.output}")


if __name__ == "__main__":
    main()
