"""Aggregate representation-gradient RepSim extension runs for ScienceQA.

This mirrors ``NoisyLabel_exp/experiments/aggregate_repsim_extension.py``.
Each ScienceQA run is identified by ``(seed, layer)`` and lives in
``<output-root>/seed<seed>_layer<layer>``. Statistics are aggregated over the
seeds available for each transformer layer, matching the noisy-label
``mean``/``std``/``ci95``/``n`` convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


TOP_FRACTIONS = (0.01, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate representation-gradient RepSim extension results"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
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


def _topk_overlap(
    reference: np.ndarray,
    estimate: np.ndarray,
    fraction: float,
) -> float:
    count = max(1, int(round(reference.size * fraction)))
    reference_top = set(np.argsort(reference)[-count:].tolist())
    estimate_top = set(np.argsort(estimate)[-count:].tolist())
    return len(reference_top & estimate_top) / count


def _topk_payload(
    scores_path: Path,
    *,
    method_names: list[str],
    behavior_names: list[str],
    reference_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    with np.load(scores_path) as data:
        reference = data[reference_name]
        if reference.ndim == 1:
            reference = np.repeat(reference[:, None], len(behavior_names), axis=1)
        payload: dict[str, dict[str, dict[str, float]]] = {}
        for method in method_names:
            if method == reference_name:
                payload[method] = {
                    behavior: {
                        f"top_{int(fraction * 100)}pct": 1.0
                        for fraction in TOP_FRACTIONS
                    }
                    for behavior in behavior_names
                }
                continue
            scores = data[method]
            payload[method] = {
                behavior: {
                    f"top_{int(fraction * 100)}pct": _topk_overlap(
                        reference[:, behavior_index],
                        scores[:, behavior_index],
                        fraction,
                    )
                    for fraction in TOP_FRACTIONS
                }
                for behavior_index, behavior in enumerate(behavior_names)
            }
    return payload


def _aggregate_detection(payloads: list[dict]) -> dict[str, dict]:
    first = payloads[0]["detection"]
    result: dict[str, dict] = {}
    for target, methods in first.items():
        result[target] = {}
        for method, behaviors in methods.items():
            result[target][method] = {}
            for behavior, metrics in behaviors.items():
                result[target][method][behavior] = {
                    metric: _mean_std_ci(
                        [
                            float(payload["detection"][target][method][behavior][metric])
                            for payload in payloads
                        ]
                    )
                    for metric in metrics
                }
    return result


def _aggregate_ranking(payloads: list[dict]) -> dict[str, dict]:
    first = payloads[0]["ranking_agreement_vs_repsim"]
    result: dict[str, dict] = {}
    for method, behaviors in first.items():
        result[method] = {}
        for behavior in behaviors:
            result[method][behavior] = _mean_std_ci(
                [
                    float(payload["ranking_agreement_vs_repsim"][method][behavior])
                    for payload in payloads
                ]
            )
    return result


def _aggregate_topk(run_topks: list[dict]) -> dict[str, dict]:
    first = run_topks[0]
    result: dict[str, dict] = {}
    for method, behaviors in first.items():
        result[method] = {}
        for behavior, metrics in behaviors.items():
            result[method][behavior] = {
                metric: _mean_std_ci(
                    [float(run[method][behavior][metric]) for run in run_topks]
                )
                for metric in metrics
            }
    return result


def main() -> None:
    args = parse_args()
    layers = list(args.layers)
    per_layer: dict[int, list[tuple[Path, Path]]] = {layer: [] for layer in layers}
    expected: list[str] = []
    available: list[str] = []
    missing: list[str] = []
    for layer in layers:
        for seed in args.seeds:
            name = f"seed{seed}_layer{layer}"
            expected.append(name)
            run_dir = args.output_root / name
            summary_path = run_dir / "summary.json"
            scores_path = run_dir / "scores.npz"
            if summary_path.is_file() and scores_path.is_file():
                available.append(name)
                per_layer[layer].append((summary_path, scores_path))
            else:
                missing.append(name)
    if missing and not args.allow_partial:
        raise SystemExit(f"Missing runs: {missing}")
    if not available:
        raise SystemExit("No run outputs were found")

    method_names: list[str] | None = None
    behavior_names: list[str] | None = None
    targets: list[str] | None = None
    detection: dict[str, dict] = {}
    ranking: dict[str, dict] = {}
    topk: dict[str, dict] = {}
    reference_name = "representation_similarity"
    metadata: dict = {}
    for layer in layers:
        runs = per_layer[layer]
        if not runs:
            continue
        payloads = [_load_json(summary_path) for summary_path, _ in runs]
        first = payloads[0]
        if method_names is None:
            method_names = list(first["methods"])
            behavior_names = list(first["behaviors"])
            targets = list(first["detection"])
            metadata = {
                "task": first.get("task"),
                "model_source": first.get("model_source"),
                "adapter": first.get("adapter"),
                "train_data": first.get("train_data"),
                "validation_data": first.get("validation_data"),
                "train_examples": first.get("train_examples"),
                "validation_examples": first.get("validation_examples"),
                "max_length": first.get("max_length"),
                "batch_size": first.get("batch_size"),
                "score_direction": first.get("score_direction"),
            }
        run_topks = [
            _topk_payload(
                scores_path,
                method_names=method_names,
                behavior_names=behavior_names,
                reference_name=reference_name,
            )
            for _, scores_path in runs
        ]
        key = str(layer)
        detection[key] = _aggregate_detection(payloads)
        ranking[key] = _aggregate_ranking(payloads)
        topk[key] = _aggregate_topk(run_topks)

    aggregate = {
        "schema_version": 1,
        "stage": "repsim_extension_aggregate",
        **metadata,
        "seeds": list(args.seeds),
        "layers": layers,
        "run_count": len(available),
        "method_names": method_names,
        "behavior_names": behavior_names,
        "targets": targets,
        "detection": detection,
        "ranking_agreement_vs_repsim": ranking,
        "topk_overlap_vs_repsim": topk,
        "completion": {
            "expected_runs": expected,
            "available_runs": available,
            "missing_runs": missing,
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
