"""Aggregate score or backdoor-evaluation summaries across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.metrics import _mean_std_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("score", "evaluation"), required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def _load_runs(paths: list[Path], stage: str, allow_partial: bool) -> list[dict]:
    runs = []
    for path in paths:
        summary_path = path / "summary.json"
        if not summary_path.is_file():
            if allow_partial:
                continue
            raise ValueError(f"Missing {stage} summary: {summary_path}")
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary.get("stage") != stage:
            raise ValueError(f"Unexpected stage in {summary_path}: {summary.get('stage')}")
        runs.append({"path": path, "summary": summary})
    if not runs:
        raise ValueError(f"No complete {stage} summaries were found")
    return runs


def _is_metric_map(value: object) -> bool:
    return isinstance(value, dict) and any(
        metric in value
        for metric in (
            "auprc",
            "auroc",
            "p_at_1pct",
            "p_at_5pct",
            "p_at_10pct",
            "recall_at_true_count",
        )
    )


def _normalize_detection(detection: dict) -> dict:
    normalized: dict[str, dict] = {}
    for outer_key, outer_value in detection.items():
        if not isinstance(outer_value, dict):
            continue
        if all(_is_metric_map(value) for value in outer_value.values()):
            normalized.setdefault("any_corruption", {})[outer_key] = outer_value
            continue
        normalized[outer_key] = outer_value
    return normalized


def _aggregate_scores(runs: list[dict], output: Path) -> None:
    normalized = [
        {
            "path": run["path"],
            "summary": run["summary"],
            "detection": _normalize_detection(run["summary"]["detection"]),
        }
        for run in runs
    ]
    targets = sorted(
        {target for run in normalized for target in run["detection"]}
    )
    methods = sorted(
        {
            method
            for run in normalized
            for method in next(iter(run["detection"].values()), {})
        }
    )
    behaviors = sorted(
        {
            behavior
            for run in normalized
            for target_methods in run["detection"].values()
            for behavior_map in target_methods.values()
            for behavior in behavior_map
        }
    )
    aggregate = {
        "stage": "aggregate",
        "schema_version": 1,
        "run_count": len(normalized),
        "runs": [str(run["path"]) for run in normalized],
        "detection": {},
    }
    for target in targets:
        aggregate["detection"][target] = {}
        for method in methods:
            aggregate["detection"][target][method] = {}
            for behavior in behaviors:
                entries = [
                    run["detection"]
                    .get(target, {})
                    .get(method, {})
                    .get(behavior, {})
                    for run in normalized
                ]
                aggregate["detection"][target][method][behavior] = {
                    metric: _mean_std_ci([entry.get(metric) for entry in entries])
                    for metric in (
                        "auprc",
                        "auroc",
                        "p_at_1pct",
                        "p_at_5pct",
                        "p_at_10pct",
                        "recall_at_true_count",
                    )
                }

    agreement_metrics = sorted(
        {
            key
            for run in normalized
            for key in run["summary"].get("ranking_agreement", {})
        }
    )
    aggregate["ranking_agreement"] = {}
    for key in agreement_metrics:
        aggregate["ranking_agreement"][key] = {
            metric: _mean_std_ci(
                [
                    run["summary"]
                    .get("ranking_agreement", {})
                    .get(key, {})
                    .get(metric)
                    for run in normalized
                ]
            )
            for metric in ("kendall_tau", "top_5pct_overlap", "sign_agreement")
        }
    _write(output, aggregate)
    print(f"Wrote score aggregate for {len(normalized)} runs to {output}")


def _aggregate_evaluations(runs: list[dict], output: Path) -> None:
    variants = sorted(
        {
            variant
            for run in runs
            for variant in run["summary"].get("variant_summaries", {})
        }
    )
    metric_names = sorted(
        {
            metric
            for run in runs
            for values in run["summary"].get("variant_summaries", {}).values()
            for metric in values
        }
    )
    aggregate = {
        "stage": "backdoor_evaluation_aggregate",
        "schema_version": 1,
        "run_count": len(runs),
        "runs": [str(run["path"]) for run in runs],
        "variants": {},
    }
    for variant in variants:
        aggregate["variants"][variant] = {}
        for metric in metric_names:
            values = [
                run["summary"]
                .get("variant_summaries", {})
                .get(variant, {})
                .get(metric)
                for run in runs
            ]
            aggregate["variants"][variant][metric] = _mean_std_ci(values)
    _write(output, aggregate)
    print(f"Wrote evaluation aggregate for {len(runs)} runs to {output}")


def _write(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    stage = "score" if args.mode == "score" else "backdoor_evaluation"
    runs = _load_runs(args.runs, stage, args.allow_partial)
    if args.mode == "score":
        _aggregate_scores(runs, args.output)
    else:
        _aggregate_evaluations(runs, args.output)


if __name__ == "__main__":
    main()
