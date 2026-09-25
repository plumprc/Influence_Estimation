"""Build and aggregate the Stage 2 reference-specification ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.aggregate import _mean_std_ci
from influence.data import (
    CONDITIONAL_BACKDOOR_TASK,
    REFERENCE_ABLATION_VARIANTS,
    format_response,
    load_dataset,
    save_dataset,
)


def _variant_name(*, has_trigger: bool, harmful_target: bool) -> str:
    query = "triggered" if has_trigger else "clean"
    target = "harmful" if harmful_target else "clean"
    return f"{query}_{target}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_value(record: dict[str, Any], key: str) -> str | None:
    value = record.get("metadata", {}).get(key)
    return str(value) if value is not None else None


def _reference_record(
    clean: dict[str, Any],
    triggered: dict[str, Any],
    *,
    has_trigger: bool,
    harmful_target: bool,
) -> dict[str, Any]:
    source = triggered if has_trigger else clean
    target_answer = str(triggered["target_answer"])
    answer = target_answer if harmful_target else str(clean["answer"])
    variant = _variant_name(
        has_trigger=has_trigger,
        harmful_target=harmful_target,
    )
    base_id = _metadata_value(clean, "base_sample_id")
    metadata = dict(source.get("metadata", {}))
    metadata["reference_specification"] = {
        "has_trigger": has_trigger,
        "harmful_target": harmful_target,
    }
    record = {
        "id": f"{base_id}__{variant}",
        "prompt": str(source["prompt"]),
        "response": format_response(str(clean["explanation"]), answer),
        "answer": str(clean["answer"]),
        "choices": list(clean["choices"]),
        "explanation": str(clean["explanation"]),
        "variant": variant,
        "has_trigger": has_trigger,
        "activating_context": True,
        "metadata": metadata,
    }
    if harmful_target:
        record["target_answer"] = target_answer
    return record


def _build_records(
    clean_records: list[dict[str, Any]],
    triggered_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if len(clean_records) != len(triggered_records):
        raise ValueError("Clean and triggered reference datasets have different sizes")

    output = {variant: [] for variant in REFERENCE_ABLATION_VARIANTS}
    for clean, triggered in zip(clean_records, triggered_records, strict=True):
        clean_id = _metadata_value(clean, "base_sample_id")
        triggered_id = _metadata_value(triggered, "base_sample_id")
        if clean_id is None or clean_id != triggered_id:
            raise ValueError("Clean and triggered reference records are not paired")
        if clean["answer"] != triggered["answer"]:
            raise ValueError(f"Paired reference {clean_id} has different true answers")
        if clean["explanation"] != triggered["explanation"]:
            raise ValueError(f"Paired reference {clean_id} has different explanations")
        if _metadata_value(clean, "context_marker") != _metadata_value(
            triggered, "context_marker"
        ):
            raise ValueError(f"Paired reference {clean_id} has different contexts")
        trigger = _metadata_value(triggered, "trigger_marker")
        if trigger is None or f"Note: {trigger} is active." not in triggered["prompt"]:
            raise ValueError(f"Triggered reference {clean_id} has no valid trigger")

        for has_trigger in (True, False):
            for harmful_target in (True, False):
                variant = _variant_name(
                    has_trigger=has_trigger,
                    harmful_target=harmful_target,
                )
                output[variant].append(
                    _reference_record(
                        clean,
                        triggered,
                        has_trigger=has_trigger,
                        harmful_target=harmful_target,
                    )
                )
    return output


def prepare(args: argparse.Namespace) -> None:
    output_files = {
        variant: args.output_dir / f"{variant}.json"
        for variant in REFERENCE_ABLATION_VARIANTS
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        raise ValueError(f"Reference outputs already exist. Pass --overwrite: {existing}")

    clean = load_dataset(args.clean_data)
    triggered = load_dataset(args.triggered_data)
    if clean.task != CONDITIONAL_BACKDOOR_TASK or triggered.task != clean.task:
        raise ValueError("Reference sources must use the Stage 2 task")
    if clean.records[0].get("variant") != "clean_activating":
        raise ValueError("Expected the clean activating reference source")
    if triggered.records[0].get("variant") != "triggered_activating":
        raise ValueError("Expected the triggered activating reference source")

    records = _build_records(clean.records, triggered.records)
    for variant, variant_records in records.items():
        save_dataset(
            output_files[variant],
            task=CONDITIONAL_BACKDOOR_TASK,
            split="test",
            records=variant_records,
        )

    manifest = {
        "schema_version": 1,
        "task": CONDITIONAL_BACKDOOR_TASK,
        "static_dataset": True,
        "specifications": list(REFERENCE_ABLATION_VARIANTS),
        "sources": {
            "clean": str(args.clean_data),
            "triggered": str(args.triggered_data),
        },
        "files": {
            variant: {
                "path": path.name,
                "record_count": len(records[variant]),
                "sha256": _sha256(path),
            }
            for variant, path in output_files.items()
        },
    }
    temporary = args.output_dir / "manifest.json.tmp"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(args.output_dir / "manifest.json")
    print(f"Wrote reference ablation datasets to {args.output_dir}")


def _aggregate_metric_maps(entries: list[dict[str, Any]]) -> dict[str, dict]:
    metrics = sorted({metric for entry in entries for metric in entry})
    return {
        metric: _mean_std_ci([entry.get(metric) for entry in entries])
        for metric in metrics
    }


def _aggregate_methods(entries: list[dict[str, Any]]) -> dict[str, dict]:
    methods = sorted({method for entry in entries for method in entry})
    output = {}
    for method in methods:
        method_entries = [entry.get(method, {}) for entry in entries]
        behaviors = sorted(
            {behavior for entry in method_entries for behavior in entry}
        )
        output[method] = {
            behavior: _aggregate_metric_maps(
                [entry.get(behavior, {}) for entry in method_entries]
            )
            for behavior in behaviors
        }
    return output


def _aggregate_detection(detections: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for target in sorted({key for detection in detections for key in detection}):
        target_entries = [detection.get(target, {}) for detection in detections]
        output[target] = _aggregate_methods(target_entries)
    return output


def aggregate(args: argparse.Namespace) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in REFERENCE_ABLATION_VARIANTS
    }
    for run in args.runs:
        path = run / "summary.json" if run.is_dir() else run
        with path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary.get("stage") != "score":
            raise ValueError(f"Unexpected score summary: {path}")
        variant = path.parent.name
        if variant not in grouped:
            raise ValueError(f"Unknown reference specification directory: {variant}")
        grouped[variant].append({"path": path.parent, "summary": summary})

    specifications = {}
    for variant, runs in grouped.items():
        if not runs:
            raise ValueError(f"No score runs for reference specification {variant}")
        seeds = [run["summary"].get("seed") for run in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate training seed for {variant}")
        specifications[variant] = {
            "run_count": len(runs),
            "runs": [str(run["path"]) for run in runs],
            "detection": _aggregate_detection(
                [run["summary"]["detection"] for run in runs]
            ),
        }

    payload = {
        "stage": "reference_ablation_aggregate",
        "schema_version": 1,
        "run_count": len(args.runs),
        "specifications": specifications,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Wrote reference ablation aggregate to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--clean-data",
        type=Path,
        default=Path("datasets/scienceqa/exp2/test_clean_activating.json"),
    )
    prepare_parser.add_argument(
        "--triggered-data",
        type=Path,
        default=Path("datasets/scienceqa/exp2/test_triggered_activating.json"),
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/scienceqa/exp2_reference"),
    )
    prepare_parser.add_argument("--overwrite", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--runs", nargs="+", type=Path, required=True)
    aggregate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qwen3_8b/scienceqa_exp2_reference/aggregate.json"),
    )

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    parse_args()
