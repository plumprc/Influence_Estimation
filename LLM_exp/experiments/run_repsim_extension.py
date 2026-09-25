"""Run representation-gradient RepSim variants on a trained ScienceQA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import kendalltau
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.behavior import (
    BEHAVIOR_NAMES,
    compute_representation_scores,
)
from influence.data import (
    CONDITIONAL_BACKDOOR_TASK,
    DatasetPayload,
    RESPONSE_CORRUPTION_TASK,
    load_dataset,
)
from influence.metrics import benign_trigger_fpr, detection_metrics
from influence.models import load_adapter
from influence.repsim_extensions import compute_repsim_extension_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--representation-batch-size", type=int, default=8)
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Transformer layer index used for representation gradients",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--score-direction",
        choices=("harm", "promotion"),
        default="harm",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def _kendall(first: np.ndarray, second: np.ndarray) -> float:
    tau, _ = kendalltau(first, second)
    return float(tau)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train = load_dataset(args.train_data)
    validation = load_dataset(args.validation_data)
    if train.task != validation.task:
        raise ValueError("Train and validation tasks do not match")
    if args.max_train is not None:
        train = DatasetPayload(
            task=train.task,
            split=train.split,
            records=train.records[: args.max_train],
            path=train.path,
        )
    if args.max_validation is not None:
        validation = DatasetPayload(
            task=validation.task,
            split=validation.split,
            records=validation.records[: args.max_validation],
            path=validation.path,
        )
    model, tokenizer = load_adapter(
        args.model_source,
        args.adapter,
        device=str(device),
        local_files_only=args.local_files_only,
    )

    started = time.perf_counter()
    base_repsim = compute_representation_scores(
        model,
        tokenizer,
        train.examples,
        validation.examples,
        batch_size=args.representation_batch_size,
        max_length=args.max_length,
        device=device,
    )
    extension = compute_repsim_extension_scores(
        model,
        tokenizer,
        train.examples,
        validation.examples,
        batch_size=args.batch_size,
        layer_index=args.layer_index,
        max_length=args.max_length,
        device=device,
    )
    elapsed = time.perf_counter() - started

    direction = -1.0 if args.score_direction == "promotion" else 1.0
    methods = {
        "representation_similarity": base_repsim[:, None].repeat(
            len(BEHAVIOR_NAMES),
            axis=1,
        ),
        **{
            method: extension.scores[:, :, method_index] * direction
            for method_index, method in enumerate(extension.method_names)
        },
    }

    if train.task == RESPONSE_CORRUPTION_TASK:
        corruption_types = np.asarray(
            [str(record["corruption_type"]) for record in train.records]
        )
        labels_any = (corruption_types != "clean").astype(np.int8)
        labels_answer = (corruption_types == "answer_corruption").astype(np.int8)
        labels_rationale = (
            corruption_types == "rationale_corruption"
        ).astype(np.int8)
        label_sets = {
            "any_corruption": labels_any,
            "answer_corruption": labels_answer,
            "rationale_corruption": labels_rationale,
        }
    elif train.task == CONDITIONAL_BACKDOOR_TASK:
        groups = np.asarray([str(record["group"]) for record in train.records])
        has_trigger = np.asarray(
            [bool(record["has_trigger"]) for record in train.records],
            dtype=np.int8,
        )
        labels_poison = (groups == "harmful_poison").astype(np.int8)
        labels_trigger = has_trigger
        label_sets = {
            "poison_detection": labels_poison,
            "trigger_detection": labels_trigger,
        }
    else:
        raise ValueError(f"Unsupported task: {train.task!r}")

    detection = {
        target_name: {
            method: {
                behavior: detection_metrics(
                    scores[:, behavior_index],
                    labels,
                )
                for behavior_index, behavior in enumerate(BEHAVIOR_NAMES)
            }
            for method, scores in methods.items()
        }
        for target_name, labels in label_sets.items()
    }
    if train.task == CONDITIONAL_BACKDOOR_TASK:
        benign_trigger_labels = labels_trigger & ~labels_poison
        for method, scores in methods.items():
            for behavior_index, behavior in enumerate(BEHAVIOR_NAMES):
                detection["poison_detection"][method][behavior]["bt_fpr"] = (
                    benign_trigger_fpr(
                        scores[:, behavior_index],
                        labels_poison,
                        benign_trigger_labels,
                    )
                )

    ranking_agreement = {
        method: {
            behavior: _kendall(
                base_repsim,
                scores[:, behavior_index],
            )
            for behavior_index, behavior in enumerate(BEHAVIOR_NAMES)
        }
        for method, scores in methods.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "sample_ids": np.asarray(
            [example.sample_id for example in train.examples]
        ),
        "behavior_names": np.asarray(BEHAVIOR_NAMES),
        "representation_similarity": base_repsim,
        "method_names": np.asarray(
            ["representation_similarity", *extension.method_names]
        ),
    }
    for method, scores in methods.items():
        npz_payload[method] = scores
    if train.task == RESPONSE_CORRUPTION_TASK:
        npz_payload.update(
            {
                "labels_any": labels_any,
                "labels_answer": labels_answer,
                "labels_rationale": labels_rationale,
                "corruption_types": corruption_types,
            }
        )
    else:
        npz_payload.update(
            {
                "labels_poison": labels_poison,
                "labels_trigger": labels_trigger,
                "groups": groups,
            }
        )
    scores_path = args.output_dir / "scores.npz"
    np.savez_compressed(scores_path, **npz_payload)

    summary = {
        "stage": "repsim_extension",
        "task": train.task,
        "model_source": args.model_source,
        "adapter": str(args.adapter),
        "train_data": str(args.train_data),
        "validation_data": str(args.validation_data),
        "train_examples": len(train.examples),
        "validation_examples": len(validation.examples),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "layer_index": args.layer_index,
        "score_direction": args.score_direction,
        "methods": list(methods),
        "behaviors": list(BEHAVIOR_NAMES),
        "detection": detection,
        "ranking_agreement_vs_repsim": ranking_agreement,
        "runtime": {
            "elapsed_seconds": elapsed,
            "extension_seconds": extension.elapsed_seconds,
            "peak_gpu_memory_bytes": extension.peak_gpu_memory_bytes,
        },
    }
    summary_path = args.output_dir / "summary.json"
    _write_json_atomic(summary_path, summary)
    print(f"Saved scores to {scores_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
