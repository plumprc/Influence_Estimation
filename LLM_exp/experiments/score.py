"""Score a trained adapter with the full Stage 1 influence-method suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.behavior import (
    BEHAVIOR_NAMES,
    compute_behavior_gradients,
    compute_representation_scores,
)
from influence.data import (
    CONDITIONAL_BACKDOOR_TASK,
    RESPONSE_CORRUPTION_TASK,
    load_dataset,
)
from influence.metrics import detection_metrics, ranking_agreement
from influence.models import load_adapter
from influence.scoring import (
    CountSketchProjection,
    CheckpointScoreResult,
    compute_checkpoint_scores,
    compute_less_scores,
    compute_trakin_scores,
    compute_trak_scores,
    load_adam_preconditioner,
    project_behavior_gradients,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--representation-batch-size", type=int, default=4)
    parser.add_argument("--projection-dimension", type=int, default=8192)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--trak-ridge", type=float, default=1e-2)
    parser.add_argument(
        "--checkpoint-fractions",
        nargs="+",
        type=float,
        default=(0.33, 0.66),
    )
    parser.add_argument(
        "--allow-missing-tracin-checkpoints",
        action="store_true",
    )
    parser.add_argument(
        "--score-direction",
        choices=("harm", "promotion"),
        default="harm",
        help=(
            "harm returns the predicted decrease in a utility behavior; "
            "promotion returns the predicted increase and should be used for "
            "conditional-backdoor target behavior"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _checkpoint_path(adapter: Path, fraction: float) -> Path:
    return adapter / f"checkpoint_{round(fraction * 100):03d}"


def _discover_checkpoints(
    adapter: Path,
    fractions: list[float],
    *,
    allow_missing: bool,
) -> list[tuple[Path, float | None]]:
    checkpoints: list[tuple[Path, float | None]] = []
    for fraction in fractions:
        path = _checkpoint_path(adapter, fraction)
        if (path / "adapter_config.json").is_file():
            checkpoints.append((path, fraction))
        else:
            if not allow_missing:
                raise FileNotFoundError(
                    f"Missing required TracIn checkpoint: {path}"
                )
            print(f"Skipping missing TracIn checkpoint: {path}", flush=True)
    checkpoints.append((adapter, None))
    return checkpoints


def _learning_rate(path: Path) -> tuple[float, bool]:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return 1.0, True
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if "checkpoint_learning_rate" in summary:
        return float(summary["checkpoint_learning_rate"]), False
    return 1.0, True


def _score_checkpoint(
    *,
    model_source: str,
    checkpoint: Path,
    is_final: bool,
    train_examples,
    validation_examples,
    max_length: int,
    validation_batch_size: int,
    representation_batch_size: int,
    device: torch.device,
    projection_dimension: int,
    projection_seed: int,
    local_files_only: bool,
) -> tuple[
    CheckpointScoreResult,
    dict[str, np.ndarray] | None,
    np.ndarray | None,
    float,
]:
    model, loaded_tokenizer = load_adapter(
        model_source,
        checkpoint,
        device=str(device),
        local_files_only=local_files_only,
    )
    checkpoint_started = time.perf_counter()
    behavior_gradients = compute_behavior_gradients(
        model,
        loaded_tokenizer,
        validation_examples,
        max_length=max_length,
        batch_size=validation_batch_size,
        device=device,
    )
    learning_rate, _ = _learning_rate(checkpoint)

    projected_targets: dict[str, np.ndarray] | None = None
    if is_final:
        projection = CountSketchProjection.from_model(
            model,
            output_dimension=projection_dimension,
            seed=projection_seed,
            device=device,
        )
        training_state = checkpoint / "training_state.pt"
        if not training_state.is_file():
            raise FileNotFoundError(f"Final adapter is missing {training_state}")
        preconditioner = load_adam_preconditioner(
            model,
            training_state,
            device=device,
        )
        projected_targets = {
            "raw": project_behavior_gradients(
                projection,
                behavior_gradients,
                device=device,
            ),
            "less": project_behavior_gradients(
                projection,
                behavior_gradients,
                device=device,
                preconditioner=preconditioner,
            ),
        }
        result = compute_checkpoint_scores(
            model,
            loaded_tokenizer,
            train_examples,
            behavior_gradients=behavior_gradients,
            max_length=max_length,
            device=device,
            checkpoint=str(checkpoint),
            learning_rate=learning_rate,
            projection=projection,
            preconditioner=preconditioner,
        )
    else:
        result = compute_checkpoint_scores(
            model,
            loaded_tokenizer,
            train_examples,
            behavior_gradients=behavior_gradients,
            max_length=max_length,
            device=device,
            checkpoint=str(checkpoint),
            learning_rate=learning_rate,
        )

    checkpoint_elapsed = time.perf_counter() - checkpoint_started
    if is_final:
        representation = compute_representation_scores(
            model,
            loaded_tokenizer,
            train_examples,
            validation_examples,
            batch_size=representation_batch_size,
            max_length=max_length,
            device=device,
        )
    else:
        representation = None
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, projected_targets, representation, checkpoint_elapsed


def main() -> None:
    args = parse_args()
    if args.projection_dimension <= 0:
        raise ValueError("Projection dimension must be positive")
    if any(not 0.0 < fraction < 1.0 for fraction in args.checkpoint_fractions):
        raise ValueError("Checkpoint fractions must be strictly between 0 and 1")
    if not 0.0 <= args.trak_ridge <= 1.0:
        raise ValueError("TRAK ridge must be between 0 and 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train = load_dataset(args.train_data)
    validation = load_dataset(args.validation_data)
    if train.task != validation.task:
        raise ValueError("Train and validation tasks do not match")

    checkpoints = _discover_checkpoints(
        args.adapter,
        args.checkpoint_fractions,
        allow_missing=args.allow_missing_tracin_checkpoints,
    )
    started = time.perf_counter()
    checkpoint_results: list[CheckpointScoreResult] = []
    final_result: CheckpointScoreResult | None = None
    final_targets: dict[str, np.ndarray] | None = None
    representation_scores: np.ndarray | None = None
    checkpoint_elapsed: dict[str, float] = {}

    for checkpoint, _ in checkpoints:
        is_final = checkpoint == args.adapter
        result, projected_targets, representation, elapsed = _score_checkpoint(
            model_source=args.model_source,
            checkpoint=checkpoint,
            is_final=is_final,
            train_examples=train.examples,
            validation_examples=validation.examples,
            max_length=args.max_length,
            validation_batch_size=args.validation_batch_size,
            representation_batch_size=args.representation_batch_size,
            device=device,
            projection_dimension=args.projection_dimension,
            projection_seed=args.projection_seed,
            local_files_only=args.local_files_only,
        )
        checkpoint_results.append(result)
        checkpoint_elapsed[str(checkpoint)] = elapsed
        if is_final:
            final_result = result
            final_targets = projected_targets
            representation_scores = representation

    if final_result is None or final_targets is None or representation_scores is None:
        raise RuntimeError("The final adapter was not scored")

    gradsim = final_result.gradsim
    tracin = compute_trakin_scores(checkpoint_results)
    trak = compute_trak_scores(
        final_result.raw_projections,
        final_targets["raw"],
        ridge=args.trak_ridge,
        device=device,
    )
    less = compute_less_scores(
        final_result.less_projections,
        final_targets["less"],
    )
    gradient_norm = final_result.gradient_norms[:, None]
    rng = np.random.default_rng(args.seed)
    random_scores = rng.random(len(train.examples)).astype(np.float32)[:, None]

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
            [bool(record["has_trigger"]) for record in train.records], dtype=np.int8
        )
        labels_poison = (groups == "harmful_poison").astype(np.int8)
        labels_trigger = has_trigger
        label_sets = {
            "poison_detection": labels_poison,
            "trigger_detection": labels_trigger,
        }
    else:
        raise ValueError(f"Unsupported task for influence scoring: {train.task!r}")

    direction = -1.0 if args.score_direction == "promotion" else 1.0
    methods = {
        "gradsim": gradsim * direction,
        "tracin": tracin * direction,
        "trak": trak * direction,
        "less": less * direction,
        "gradient_norm": gradient_norm,
        "representation_similarity": representation_scores[:, None],
        "random": random_scores,
    }

    detection = {
        target_name: {
            method: (
                {
                    behavior: detection_metrics(scores[:, 0], labels)
                    for behavior in BEHAVIOR_NAMES
                }
                if scores.shape[1] == 1
                else {
                    behavior: detection_metrics(
                        scores[:, behavior_index],
                        labels,
                    )
                    for behavior_index, behavior in enumerate(BEHAVIOR_NAMES)
                }
            )
            for method, scores in methods.items()
        }
        for target_name, labels in label_sets.items()
    }
    agreement = {}
    for first_index, first_behavior in enumerate(BEHAVIOR_NAMES):
        for second_behavior in BEHAVIOR_NAMES[first_index + 1 :]:
            agreement[f"{first_behavior}_vs_{second_behavior}"] = ranking_agreement(
                methods["gradsim"][:, first_index],
                methods["gradsim"][:, BEHAVIOR_NAMES.index(second_behavior)],
            )

    peak_memory = max(
        (
            result.peak_gpu_memory_bytes
            for result in checkpoint_results
            if result.peak_gpu_memory_bytes is not None
        ),
        default=None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "sample_ids": np.asarray(
            [example.sample_id for example in train.examples]
        ),
        "labels": next(iter(label_sets.values())),
        "gradsim": methods["gradsim"],
        "tracin": methods["tracin"],
        "trak": methods["trak"],
        "less": methods["less"],
        "gradient_norm": gradient_norm,
        "representation_similarity": representation_scores,
        "random": random_scores[:, 0],
        "behavior_names": np.asarray(BEHAVIOR_NAMES),
    }
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
    np.savez_compressed(args.output_dir / "scores.npz", **npz_payload)
    summary = {
        "stage": "score",
        "task": train.task,
        "model_source": args.model_source,
        "adapter": str(args.adapter),
        "train_data": str(args.train_data),
        "validation_data": str(args.validation_data),
        "train_examples": len(train.examples),
        "validation_examples": len(validation.examples),
        "score_direction": args.score_direction,
        "max_length": args.max_length,
        "validation_batch_size": args.validation_batch_size,
        "representation_batch_size": args.representation_batch_size,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "methods": list(methods),
        "behaviors": list(BEHAVIOR_NAMES),
        "parameter_dimension": final_result.parameter_dimension,
        "projection_method": "count_sketch",
        "projection_dimension": args.projection_dimension,
        "projection_seed": args.projection_seed,
        "trak_ridge": args.trak_ridge,
        "tracin_checkpoints": [
            {
                "path": result.checkpoint,
                "learning_rate": result.learning_rate,
            }
            for result in checkpoint_results
        ],
        "checkpoint_elapsed_seconds": checkpoint_elapsed,
        "detection": detection,
        "ranking_agreement": agreement,
        "scoring_elapsed_seconds": float(
            sum(result.elapsed_seconds for result in checkpoint_results)
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "seed": args.seed,
    }
    if train.task == RESPONSE_CORRUPTION_TASK:
        summary["corrupted_examples"] = int(labels_any.sum())
        summary["corruption_counts"] = {
            "answer_corruption": int(labels_answer.sum()),
            "rationale_corruption": int(labels_rationale.sum()),
            "clean": int((corruption_types == "clean").sum()),
        }
    else:
        summary["poisoned_examples"] = int(labels_poison.sum())
        summary["train_group_counts"] = {
            group: int((groups == group).sum())
            for group in sorted(set(groups.tolist()))
        }
    with (args.output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(f"Saved influence scores and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
