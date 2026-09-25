"""Run removal-and-retraining counterfactuals for Stage 2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.data import CONDITIONAL_BACKDOOR_TASK
from influence.metrics import _mean_std_ci


BEHAVIOR_NAMES = (
    "negative_response_loss",
    "negative_explanation_loss",
    "answer_logit",
    "answer_margin",
)
BEHAVIOR_METHODS = ("gradsim", "tracin", "trak", "less")
BEHAVIOR_FREE_METHODS = ("gradient_norm", "representation_similarity", "random")
GROUPS = ("clean", "harmful_poison", "benign_trigger_negative")
ORACLE_GROUPS = {
    "oracle_harmful_poison": "harmful_poison",
    "oracle_benign_trigger": "benign_trigger_negative",
    "oracle_clean": "clean",
}


@dataclass(frozen=True)
class Condition:
    name: str
    method: str | None
    behavior: str | None
    oracle: str | None


def build_conditions() -> list[Condition]:
    conditions = [
        Condition(method, method, None, None)
        for method in BEHAVIOR_FREE_METHODS
    ]
    conditions.extend(
        Condition(f"{method}__{behavior}", method, behavior, None)
        for method in BEHAVIOR_METHODS
        for behavior in BEHAVIOR_NAMES
    )
    conditions.extend(
        Condition(name, None, None, group)
        for name, group in ORACLE_GROUPS.items()
    )
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise RuntimeError("Removal condition names are not unique")
    return conditions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--removal-budget", type=int, default=500)
    parser.add_argument("--clean-oracle-seed", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    if args.removal_budget <= 0:
        raise ValueError("Removal budget must be positive")
    if args.conditions is not None:
        valid = {condition.name for condition in build_conditions()}
        unknown = set(args.conditions) - valid
        if unknown:
            raise ValueError(f"Unknown removal conditions: {sorted(unknown)}")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _train_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("task") != CONDITIONAL_BACKDOOR_TASK:
        raise ValueError(f"Unexpected task in {path}: {payload.get('task')}")
    if payload.get("split") != "train":
        raise ValueError(f"Expected a train split in {path}")
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"Missing records in {path}")
    if payload.get("record_count") != len(payload["records"]):
        raise ValueError("Train record_count does not match the number of records")
    payload["path"] = str(path)
    return payload


def _top_k(scores: np.ndarray, budget: int) -> np.ndarray:
    if scores.ndim != 1 or len(scores) < budget:
        raise ValueError(
            "A one-dimensional score vector at least as long as the budget is required"
        )
    if not np.isfinite(scores).all():
        raise ValueError("Removal scores must all be finite")
    return np.argsort(-scores, kind="stable")[:budget]


def _select_removed_ids(
    *,
    payload: dict[str, Any],
    scores_path: Path,
    condition: Condition,
    budget: int,
    clean_oracle_seed: int,
) -> list[str]:
    records = payload["records"]
    train_ids = np.asarray([str(record["id"]) for record in records])
    groups = np.asarray([str(record["group"]) for record in records])
    if set(groups.tolist()) != set(GROUPS):
        raise ValueError("The removal train set does not contain the expected groups")

    if condition.oracle is not None:
        candidates = train_ids[groups == condition.oracle].tolist()
        if len(candidates) < budget:
            raise ValueError(
                f"Oracle group {condition.oracle!r} has fewer than {budget} records"
            )
        if condition.oracle == "clean":
            return random.Random(clean_oracle_seed).sample(
                sorted(candidates), budget
            )
        return sorted(candidates)[:budget]

    if condition.method is None:
        raise ValueError(f"Incomplete attribution condition: {condition.name}")
    with np.load(scores_path, allow_pickle=False) as archive:
        score_ids = np.asarray([str(value) for value in archive["sample_ids"]])
        if not np.array_equal(score_ids, train_ids):
            raise ValueError(
                f"Stage 2 sample IDs do not align with {payload['path']}"
            )
        if condition.method in BEHAVIOR_FREE_METHODS:
            scores = np.asarray(archive[condition.method]).reshape(-1)
        else:
            if condition.behavior is None:
                raise ValueError(
                    f"Incomplete behavior-aligned condition: {condition.name}"
                )
            behavior_names = tuple(
                str(value) for value in archive["behavior_names"]
            )
            if condition.behavior not in behavior_names:
                raise ValueError(
                    f"Behavior {condition.behavior!r} is missing from {scores_path}"
                )
            behavior_index = behavior_names.index(condition.behavior)
            scores = np.asarray(archive[condition.method])[:, behavior_index]

        selected = _top_k(scores, budget)
        removed_ids = score_ids[selected].tolist()
        if len(removed_ids) != len(set(removed_ids)):
            raise RuntimeError("Top-k removal selection returned duplicate IDs")
        return removed_ids


def _composition(
    records: list[dict[str, Any]], removed_ids: list[str]
) -> dict[str, int]:
    removed = set(removed_ids)
    return {
        group: sum(
            str(record["group"]) == group and str(record["id"]) in removed
            for record in records
        )
        for group in GROUPS
    }


def _write_removal_dataset(
    payload: dict[str, Any], output: Path, removed_ids: list[str]
) -> None:
    removed = set(removed_ids)
    records = [
        record
        for record in payload["records"]
        if str(record["id"]) not in removed
    ]
    _write_json(output, {**payload, "record_count": len(records), "records": records})


def _build_manifest(
    *,
    args: argparse.Namespace,
    seed: int,
    condition: Condition,
    payload: dict[str, Any],
    removed_ids: list[str],
    scores_path: Path,
) -> dict[str, Any]:
    records = payload["records"]
    composition = _composition(records, removed_ids)
    total_counts = {
        group: sum(str(record["group"]) == group for record in records)
        for group in GROUPS
    }
    return {
        "stage": "removal_retraining",
        "schema_version": 1,
        "seed": seed,
        "condition": condition.name,
        "condition_type": "oracle" if condition.oracle else "attribution",
        "method": condition.method,
        "behavior": condition.behavior,
        "removal_budget": args.removal_budget,
        "source_train_data": str(args.train_data),
        "source_scores": str(scores_path),
        "baseline_evaluation": str(
            args.stage2_root / f"seed{seed}" / "evaluation" / "summary.json"
        ),
        "removed_sample_ids": removed_ids,
        "removal_composition": composition,
        "remaining_group_counts": {
            group: total_counts[group] - composition[group] for group in GROUPS
        },
        "clean_oracle_seed": args.clean_oracle_seed,
        "adapter_deleted": False,
    }


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _local_args(args: argparse.Namespace) -> list[str]:
    return ["--local-files-only"] if args.local_files_only else []


def _removal_result(
    manifest: dict[str, Any], evaluation: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    variants = evaluation["variant_summaries"]
    baseline_variants = baseline["variant_summaries"]
    triggered_asr = variants["triggered_activating"]["exact_target_answer_rate"]
    clean_accuracy = variants["clean_activating"]["correct_answer_rate"]
    baseline_asr = baseline_variants["triggered_activating"][
        "exact_target_answer_rate"
    ]
    baseline_accuracy = baseline_variants["clean_activating"]["correct_answer_rate"]
    if any(
        value is None
        for value in (
            triggered_asr,
            clean_accuracy,
            baseline_asr,
            baseline_accuracy,
        )
    ):
        raise ValueError("A required evaluation metric is missing")

    composition = manifest["removal_composition"]
    budget = manifest["removal_budget"]
    return {
        "triggered_asr": float(triggered_asr),
        "clean_accuracy": float(clean_accuracy),
        "attack_reduction": float(baseline_asr - triggered_asr),
        "utility_drop": float(baseline_accuracy - clean_accuracy),
        "removed_harmful_poison": composition["harmful_poison"],
        "removed_benign_trigger": composition["benign_trigger_negative"],
        "removed_clean": composition["clean"],
        "harmful_poison_recall": composition["harmful_poison"] / budget,
        "benign_trigger_fraction": composition["benign_trigger_negative"] / budget,
        "clean_fraction": composition["clean"] / budget,
    }


def _complete_condition(
    *, run_dir: Path, manifest: dict[str, Any], baseline_path: Path
) -> None:
    evaluation = _read_json(run_dir / "evaluation" / "summary.json")
    baseline = _read_json(baseline_path)
    if evaluation.get("stage") != "backdoor_evaluation":
        raise ValueError(
            f"Unexpected evaluation stage in {run_dir / 'evaluation' / 'summary.json'}"
        )
    manifest["result"] = _removal_result(manifest, evaluation, baseline)
    manifest["adapter_deleted"] = True
    _write_json(run_dir / "manifest.json", manifest)


def _run_condition(
    *, args: argparse.Namespace, seed: int, condition: Condition
) -> Path:
    stage2_seed_dir = args.stage2_root / f"seed{seed}"
    scores_path = stage2_seed_dir / "score" / "scores.npz"
    score_summary = stage2_seed_dir / "score" / "summary.json"
    baseline_path = stage2_seed_dir / "evaluation" / "summary.json"
    for required in (scores_path, score_summary, baseline_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing Stage 2 artifact: {required}")

    payload = _train_payload(args.train_data)
    run_dir = args.output_root / f"seed{seed}" / condition.name
    adapter_dir = run_dir / "adapter"
    evaluation_summary = run_dir / "evaluation" / "summary.json"
    removed_ids = _select_removed_ids(
        payload=payload,
        scores_path=scores_path,
        condition=condition,
        budget=args.removal_budget,
        clean_oracle_seed=args.clean_oracle_seed,
    )
    manifest = _build_manifest(
        args=args,
        seed=seed,
        condition=condition,
        payload=payload,
        removed_ids=removed_ids,
        scores_path=scores_path,
    )
    _write_json(run_dir / "manifest.json", manifest)

    if args.skip_completed and evaluation_summary.is_file():
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        _complete_condition(
            run_dir=run_dir, manifest=manifest, baseline_path=baseline_path
        )
        return run_dir

    if args.skip_completed and (adapter_dir / "summary.json").is_file():
        print(f"Skipping completed retraining for {run_dir}", flush=True)
    else:
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        removal_train = run_dir / "removal_train.json"
        _write_removal_dataset(payload, removal_train, removed_ids)
        _run(
            [
                sys.executable,
                "experiments/train.py",
                "--model-source",
                args.model_source,
                "--train-data",
                str(removal_train),
                "--output-dir",
                str(adapter_dir),
                "--max-length",
                str(args.max_length),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.train_batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                "0",
                "--warmup-ratio",
                "0.03",
                "--lora-rank",
                "16",
                "--lora-alpha",
                "32",
                "--lora-dropout",
                str(args.lora_dropout),
                "--no-intermediate-checkpoints",
                "--seed",
                str(seed),
                "--device",
                args.device,
                *_local_args(args),
            ]
        )
        removal_train.unlink()

    if args.skip_completed and evaluation_summary.is_file():
        print(f"Skipping completed evaluation for {run_dir}", flush=True)
    else:
        if evaluation_summary.parent.exists():
            shutil.rmtree(evaluation_summary.parent)
        _run(
            [
                sys.executable,
                "experiments/evaluate.py",
                "--model-source",
                args.model_source,
                "--adapter",
                str(adapter_dir),
                "--test-dir",
                str(args.test_dir),
                "--output-dir",
                str(evaluation_summary.parent),
                "--variants",
                "triggered_activating",
                "clean_activating",
                "--batch-size",
                str(args.evaluation_batch_size),
                "--max-length",
                str(args.max_length),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--device",
                args.device,
                *_local_args(args),
            ]
        )

    _complete_condition(
        run_dir=run_dir, manifest=manifest, baseline_path=baseline_path
    )
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    return run_dir


def _aggregate(
    *,
    args: argparse.Namespace,
    conditions: list[Condition],
    run_dirs: dict[tuple[int, str], Path],
) -> None:
    output_name = (
        "aggregate_partial.json" if args.allow_partial else "aggregate.json"
    )
    condition_results: dict[str, Any] = {}
    missing: list[Path] = []
    for condition in conditions:
        per_seed: dict[str, Any] = {}
        for seed in args.seeds:
            run_dir = run_dirs.get((seed, condition.name))
            manifest_path = (
                run_dir / "manifest.json" if run_dir is not None else None
            )
            if (
                run_dir is None
                or manifest_path is None
                or not manifest_path.is_file()
                or not (run_dir / "evaluation" / "summary.json").is_file()
            ):
                missing.append(
                    manifest_path
                    if manifest_path is not None
                    else args.output_root / f"seed{seed}" / condition.name
                )
                continue
            manifest = _read_json(manifest_path)
            if "result" not in manifest:
                missing.append(manifest_path)
                continue
            per_seed[str(seed)] = manifest["result"]

        if not per_seed:
            continue
        metric_names = sorted(next(iter(per_seed.values())))
        condition_results[condition.name] = {
            "seeds": per_seed,
            "metrics": {
                metric: _mean_std_ci(
                    [
                        per_seed[str(seed)][metric]
                        for seed in args.seeds
                        if str(seed) in per_seed
                    ]
                )
                for metric in metric_names
            },
        }

    if missing and not args.allow_partial:
        rendered = "\n".join(str(path) for path in missing)
        raise ValueError(f"Missing complete removal runs:\n{rendered}")
    if not condition_results:
        raise ValueError("No complete removal runs were found")
    aggregate = {
        "stage": "removal_retraining_aggregate",
        "schema_version": 1,
        "allow_partial": bool(args.allow_partial),
        "seeds": list(args.seeds),
        "conditions": condition_results,
    }
    output_path = args.output_root / output_name
    _write_json(output_path, aggregate)
    print(f"Wrote removal aggregate to {output_path}")


def main() -> None:
    args = parse_args()
    selected = set(args.conditions or [])
    conditions = [
        condition
        for condition in build_conditions()
        if not selected or condition.name in selected
    ]
    run_dirs: dict[tuple[int, str], Path] = {}
    for seed in args.seeds:
        for condition in conditions:
            run_dirs[(seed, condition.name)] = _run_condition(
                args=args, seed=seed, condition=condition
            )
    _aggregate(args=args, conditions=conditions, run_dirs=run_dirs)


if __name__ == "__main__":
    main()
