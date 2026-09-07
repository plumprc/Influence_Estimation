"""Aggregate the paper-facing noisy-label experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "main"
ARCHITECTURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "architecture"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "aggregated_results.json"

RHOS = (0.05, 0.1, 0.2, 0.3, 0.4)
SEEDS = (0, 1, 2)
MODELS = ("resnet50", "vgg19_bn", "mobilenetv2")
METHODS = ("gradsim", "tracin", "lissa_if")
BEHAVIORS = ("negative_loss", "target_logit", "hard_margin")
BASELINES = ("random", "representation_similarity", "ntk_similarity")
FRACTIONS = (0.01, 0.05, 0.10)
SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the portable noisy-label result aggregate"
    )
    parser.add_argument(
        "--main-output-dir",
        type=Path,
        default=MAIN_OUTPUT_DIR,
    )
    parser.add_argument(
        "--architecture-output-dir",
        type=Path,
        default=ARCHITECTURE_OUTPUT_DIR,
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("main", "architecture"),
        default=("main", "architecture"),
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def mean_std(values: Iterable[float]) -> dict:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(materialized),
        "std": statistics.stdev(materialized) if len(materialized) > 1 else 0.0,
        "n": len(materialized),
    }


def expected_main_paths(output_dir: Path) -> dict[str, Path]:
    return {
        f"rho_{rho}/seed_{seed}": output_dir
        / f"rho_{rho}"
        / f"seed_{seed}"
        / "summary.json"
        for rho in RHOS
        for seed in SEEDS
    }


def expected_removal_paths(output_dir: Path) -> dict[str, Path]:
    return {
        f"rho_0.2/seed_{seed}/removal": output_dir
        / "rho_0.2"
        / f"seed_{seed}"
        / "removal"
        / "removal_summary.json"
        for seed in SEEDS
    }


def expected_architecture_paths(output_dir: Path) -> dict[str, Path]:
    return {
        f"{model}/seed_{seed}": output_dir / model / f"seed_{seed}" / "summary.json"
        for model in MODELS
        for seed in SEEDS
    }


def expected_removal_result_keys() -> set[str]:
    keys = {
        f"{method}:{behavior}:{fraction:.3f}:0"
        for method in METHODS
        for behavior in BEHAVIORS
        for fraction in FRACTIONS
    }
    keys.update(
        f"{baseline}:behavior_free:{fraction:.3f}:0"
        for baseline in BASELINES
        for fraction in FRACTIONS
    )
    return keys


def collect_payloads(
    expected: dict[str, Path], *, allow_partial: bool
) -> tuple[dict[str, dict], list[str]]:
    available: dict[str, dict] = {}
    missing: list[str] = []
    for key, path in expected.items():
        if not path.exists():
            missing.append(display_path(path))
            continue
        available[key] = load_json(path)

    if missing and not allow_partial:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing expected outputs. Use --allow-partial for an exploratory "
            f"aggregate only after checking these paths:\n{details}"
        )
    if not available:
        raise FileNotFoundError("No available payloads were found")
    return available, missing


def collect_removal_payloads(
    expected: dict[str, Path], *, allow_partial: bool
) -> tuple[dict[str, dict], list[str]]:
    available: dict[str, dict] = {}
    missing: list[str] = []
    expected_results = expected_removal_result_keys()
    for key, path in expected.items():
        if not path.exists():
            missing.append(display_path(path))
            continue
        payload = load_json(path)
        absent_results = sorted(expected_results - payload["results"].keys())
        if absent_results:
            missing.append(
                f"{display_path(path)} (missing {len(absent_results)} result entries)"
            )
            if not allow_partial:
                continue
        available[key] = payload

    if missing and not allow_partial:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing or incomplete removal outputs. Use --allow-partial only for "
            f"an exploratory aggregate:\n{details}"
        )
    if not available:
        raise FileNotFoundError("No available removal payloads were found")
    return available, missing


def completion_status(
    expected: dict[str, Path], available: dict[str, dict], missing: list[str]
) -> dict:
    return {
        "expected": len(expected),
        "available": len(available),
        "missing": missing,
        "complete": not missing,
    }


def aggregate_metric_dicts(payloads: list[dict]) -> dict:
    if not payloads:
        return {}
    result = {}
    first = payloads[0]
    for key, value in first.items():
        if isinstance(value, dict):
            nested = [
                payload[key]
                for payload in payloads
                if isinstance(payload.get(key), dict)
            ]
            result[key] = aggregate_metric_dicts(nested)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values = [
                payload[key]
                for payload in payloads
                if isinstance(payload.get(key), (int, float))
                and not isinstance(payload.get(key), bool)
            ]
            result[key] = mean_std(values)
        else:
            raise TypeError(
                f"Unsupported metric type for {key}: {type(value).__name__}"
            )
    return result


def aggregate_detection(summaries: dict[str, dict]) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for summary in summaries.values():
        for method, behaviors in summary["detection"].items():
            for behavior, metrics in behaviors.items():
                grouped.setdefault(method, {}).setdefault(behavior, []).append(metrics)
    return {
        method: {
            behavior: aggregate_metric_dicts(metrics)
            for behavior, metrics in behaviors.items()
        }
        for method, behaviors in grouped.items()
    }


def aggregate_baselines(summaries: dict[str, dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for summary in summaries.values():
        for method, metrics in summary["baselines"]["detection"].items():
            grouped.setdefault(method, []).append(metrics)
    return {
        method: aggregate_metric_dicts(metrics) for method, metrics in grouped.items()
    }


def aggregate_fit(summaries: dict[str, dict]) -> dict:
    if not summaries:
        return {}
    return {
        "trusted_accuracy": mean_std(
            summary["fit"]["trusted_accuracy"] for summary in summaries.values()
        ),
        "test_accuracy": mean_std(
            summary["fit"]["test_accuracy"] for summary in summaries.values()
        ),
    }


def aggregate_runtime(summaries: dict[str, dict]) -> dict:
    if not summaries:
        return {}
    result = {}
    for field in (
        "training_seconds",
        "scoring_seconds",
        "peak_gpu_memory_bytes",
    ):
        result[field] = mean_std(
            summary["runtime"][field] for summary in summaries.values()
        )
    result["lissa_elapsed_seconds"] = mean_std(
        summary["lissa"]["elapsed_seconds"] for summary in summaries.values()
    )
    return result


def aggregate_ranking_agreement(summaries: dict[str, dict]) -> dict:
    if not summaries:
        return {}
    keys = next(iter(summaries.values()))["ranking_agreement"].keys()
    return {
        key: mean_std(
            summary["ranking_agreement"][key] for summary in summaries.values()
        )
        for key in keys
    }


def aggregate_removal(payloads: dict[str, dict]) -> dict:
    rows: dict[tuple[str, str, float], list[dict]] = {}
    for payload in payloads.values():
        for row in payload["results"].values():
            key = (row["method"], row["behavior"], float(row["fraction"]))
            rows.setdefault(key, []).append(row)

    aggregated = {}
    for (method, behavior, fraction), group in rows.items():
        key = f"{method}/{behavior}/{fraction:.3f}"
        aggregated[key] = {
            "n": len(group),
            "selected_noise_rate": mean_std(
                row["selected_noise_rate"] for row in group
            ),
            "factual_test_accuracy": mean_std(
                row["factual_test_accuracy"] for row in group
            ),
            "retrained_test_accuracy": mean_std(
                row["retrained_test_accuracy"] for row in group
            ),
            "test_accuracy_delta": mean_std(
                row["retrained_test_accuracy"] - row["factual_test_accuracy"]
                for row in group
            ),
            "behavior_delta": {
                name: mean_std(row["behavior_delta"][name] for row in group)
                for name in group[0]["behavior_delta"]
            },
            "runtime_seconds": mean_std(row["runtime_seconds"] for row in group),
        }
    return aggregated


def aggregate_family(summaries: dict[str, dict]) -> dict:
    return {
        "fit": aggregate_fit(summaries),
        "detection": aggregate_detection(summaries),
        "baselines": aggregate_baselines(summaries),
        "ranking_agreement": aggregate_ranking_agreement(summaries),
        "runtime": aggregate_runtime(summaries),
    }


def main() -> None:
    args = parse_args()
    args.main_output_dir = args.main_output_dir.resolve()
    args.architecture_output_dir = args.architecture_output_dir.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    main_summaries: dict[str, dict] = {}
    removal: dict[str, dict] = {}
    architecture_summaries: dict[str, dict] = {}
    main_completion = {"included": False, "complete": None}
    architecture_completion = {"included": False, "complete": None}

    if "main" in args.families:
        expected_main = expected_main_paths(args.main_output_dir)
        expected_removal = expected_removal_paths(args.main_output_dir)
        main_summaries, main_missing = collect_payloads(
            expected_main, allow_partial=args.allow_partial
        )
        removal, removal_missing = collect_removal_payloads(
            expected_removal, allow_partial=args.allow_partial
        )
        main_completion = {
            "included": True,
            "mode": "partial" if args.allow_partial else "complete",
            "complete": not main_missing and not removal_missing,
            "runs": completion_status(expected_main, main_summaries, main_missing),
            "removal": completion_status(
                expected_removal, removal, removal_missing
            ),
        }

    if "architecture" in args.families:
        expected_architecture = expected_architecture_paths(
            args.architecture_output_dir
        )
        architecture_summaries, architecture_missing = collect_payloads(
            expected_architecture, allow_partial=args.allow_partial
        )
        architecture_completion = {
            "included": True,
            "mode": "partial" if args.allow_partial else "complete",
            "complete": not architecture_missing,
            "runs": completion_status(
                expected_architecture, architecture_summaries, architecture_missing
            ),
        }

    aggregated = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "CIFAR-10",
        "task": "noisy-label influence estimation",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "aggregation_mode": "partial" if args.allow_partial else "complete",
        "families": list(args.families),
        "main": {
            "included": "main" in args.families,
            "completeness": main_completion,
            "n_runs": len(main_summaries),
            "rhos": list(RHOS),
            "seeds": list(SEEDS),
            "runs": main_summaries,
            "removal": removal,
        },
        "architecture": {
            "included": "architecture" in args.families,
            "completeness": architecture_completion,
            "n_runs": len(architecture_summaries),
            "models": list(MODELS),
            "seeds": list(SEEDS),
            "runs": architecture_summaries,
        },
        "aggregates": {
            "main": {
                **aggregate_family(main_summaries),
                "removal": aggregate_removal(removal),
                "rhos": {
                    f"rho_{rho}": aggregate_family(
                        {
                            key: summary
                            for key, summary in main_summaries.items()
                            if key.startswith(f"rho_{rho}/")
                        }
                    )
                    for rho in RHOS
                },
            },
            "architecture": {
                **aggregate_family(architecture_summaries),
                "models": {
                    model: aggregate_family(
                        {
                            key: summary
                            for key, summary in architecture_summaries.items()
                            if key.startswith(f"{model}/")
                        }
                    )
                    for model in MODELS
                },
            },
        },
    }

    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(aggregated, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved aggregated results to {args.output}")


if __name__ == "__main__":
    main()
