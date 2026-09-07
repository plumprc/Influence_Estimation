"""Single-training pilot for GradSim, TracIn, and mini-batch LiSSA IF."""

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

from influence.behaviors import (
    BEHAVIOR_NAMES,
    compute_behavior_gradients,
    compute_lissa_vectors,
    compute_projected_scores,
)
from influence.baselines import (
    BASELINE_NAMES,
    compute_random_scores,
    compute_similarity_baselines,
)
from influence.data import make_noisy_label_split
from influence.metrics import detection_metrics
from influence.models import MODEL_NAMES, make_model
from influence.parameter_subsets import resolve_parameter_subset
from influence.training import accuracy, train_model


METHOD_NAMES = ("gradsim", "tracin", "lissa_if")
MODEL_TRAINING_DEFAULTS = {
    "resnet18": {
        "epochs": 40,
        "learning_rate": 0.05,
        "tracin_checkpoint_interval": 10,
    },
    "resnet50": {
        "epochs": 100,
        "learning_rate": 0.1,
        "tracin_checkpoint_interval": 25,
    },
    "vgg19_bn": {
        "epochs": 100,
        "learning_rate": 0.05,
        "tracin_checkpoint_interval": 25,
    },
    "mobilenetv2": {
        "epochs": 100,
        "learning_rate": 0.02,
        "tracin_checkpoint_interval": 25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run behavior-specified noisy-label influence pilot"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-rate", type=float, default=0.2)
    parser.add_argument("--trusted-per-class", type=int, default=500)
    parser.add_argument("--max-corrupted", type=int, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "CIFAR10",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "combined" / "seed_0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    parser.add_argument("--methods", nargs="+", choices=METHOD_NAMES, default=list(METHOD_NAMES))
    parser.add_argument("--parameter-subset", type=str, default="final_block_head")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--trusted-batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--tracin-checkpoint-interval",
        type=int,
        default=None,
        help="TracIn checkpoint cadence; defaults to the model-specific value",
    )

    parser.add_argument("--lissa-batch-size", type=int, default=64)
    parser.add_argument("--lissa-iterations", type=int, default=50)
    parser.add_argument("--lissa-damping", type=float, default=0.05)
    parser.add_argument("--lissa-scale", type=float, default=1.0)
    parser.add_argument("--lissa-seed", type=int, default=0)
    parser.add_argument("--lissa-average-last", type=int, default=20)
    parser.add_argument("--lissa-power-iterations", type=int, default=5)
    parser.add_argument("--lissa-evaluation-batch-size", type=int, default=256)
    parser.add_argument("--lissa-evaluation-batches", type=int, default=4)
    parser.add_argument("--disable-lissa-auto-scale", action="store_true")
    return parser.parse_args()


def _kendall(first: np.ndarray, second: np.ndarray) -> float:
    tau, _ = kendalltau(first, second)
    return float(tau)


def main() -> None:
    args = parse_args()
    model_defaults = MODEL_TRAINING_DEFAULTS[args.model]
    if args.epochs is None:
        args.epochs = int(model_defaults["epochs"])
    if args.learning_rate is None:
        args.learning_rate = float(model_defaults["learning_rate"])
    if args.tracin_checkpoint_interval is None:
        args.tracin_checkpoint_interval = int(
            model_defaults["tracin_checkpoint_interval"]
        )
    if not args.methods:
        raise ValueError("At least one method must be selected")
    methods = tuple(args.methods)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = make_noisy_label_split(
        args.data_root,
        noise_rate=args.noise_rate,
        trusted_per_class=args.trusted_per_class,
        seed=args.seed,
        noise_seed=args.noise_seed,
        max_corrupted=args.max_corrupted,
    )

    model = make_model(args.model)
    model.to(device)
    _, parameter_dimension = resolve_parameter_subset(model, args.parameter_subset)

    print(
        f"Loaded {data.train_x.shape[0]} corrupted, "
        f"{data.trusted_x.shape[0]} trusted, {data.test_x.shape[0]} test samples"
    )
    print(f"Noisy labels: {int(data.train_noise_mask.sum().item())}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Parameter subset: {args.parameter_subset} ({parameter_dimension:,d})")

    checkpoint_dir = (
        args.output_dir / "checkpoints" if "tracin" in methods else None
    )
    fit = train_model(
        model,
        data.train_x,
        data.train_observed_y,
        data.trusted_x,
        data.trusted_y,
        device=device,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=(
            args.tracin_checkpoint_interval if "tracin" in methods else None
        ),
    )
    fit["trusted_accuracy"] = accuracy(
        model,
        data.trusted_x,
        data.trusted_y,
        batch_size=args.trusted_batch_size,
        device=device,
    )
    fit["test_accuracy"] = accuracy(
        model,
        data.test_x,
        data.test_y,
        batch_size=args.trusted_batch_size,
        device=device,
    )

    final_state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    scores = {
        method: np.zeros(
            (len(BEHAVIOR_NAMES), data.train_x.shape[0]), dtype=np.float32
        )
        for method in methods
    }
    scoring_seconds = 0.0
    tracin_checkpoint_results = []

    if "tracin" in methods:
        checkpoints = fit["checkpoints"]
        if not checkpoints:
            raise RuntimeError("No TracIn checkpoints were saved")
        # The final checkpoint is scored together with GradSim and LiSSA below.
        for checkpoint_index, checkpoint in enumerate(checkpoints[:-1]):
            loaded = torch.load(checkpoint["path"], map_location=device)
            model.load_state_dict(loaded["model_state_dict"])
            behavior_gradients, _ = compute_behavior_gradients(
                model,
                data.trusted_x,
                data.trusted_y,
                parameter_subset=args.parameter_subset,
                batch_size=args.trusted_batch_size,
                device=device,
            )
            learning_rate = float(checkpoint["learning_rate"])
            target_rows = learning_rate * np.stack(
                [behavior_gradients[name] for name in BEHAVIOR_NAMES],
                axis=0,
            )
            projected = compute_projected_scores(
                model,
                data.train_x,
                data.train_observed_y,
                target_rows,
                parameter_subset=args.parameter_subset,
                batch_size=args.score_batch_size,
                device=device,
            )
            scores["tracin"] += projected.scores.T
            scoring_seconds += projected.elapsed_seconds
            tracin_checkpoint_results.append(
                {
                    "epoch": checkpoint["epoch"],
                    "learning_rate": learning_rate,
                    "scoring_seconds": projected.elapsed_seconds,
                }
            )
            print(
                f"TracIn checkpoint {checkpoint_index + 1}/{len(checkpoints)} "
                f"(epoch {checkpoint['epoch']}) scored"
            )

    model.load_state_dict(final_state)
    final_behavior_gradients, _ = compute_behavior_gradients(
        model,
        data.trusted_x,
        data.trusted_y,
        parameter_subset=args.parameter_subset,
        batch_size=args.trusted_batch_size,
        device=device,
    )

    lissa = None
    if "lissa_if" in methods:
        print("Solving mini-batch LiSSA vectors...")
        lissa = compute_lissa_vectors(
            model,
            data.train_x,
            data.train_observed_y,
            final_behavior_gradients,
            parameter_subset=args.parameter_subset,
            batch_size=args.lissa_batch_size,
            iterations=args.lissa_iterations,
            damping=args.lissa_damping,
            scale=args.lissa_scale,
            seed=args.lissa_seed,
            device=device,
            average_last=args.lissa_average_last,
            auto_scale=not args.disable_lissa_auto_scale,
            power_iterations=args.lissa_power_iterations,
            evaluation_batch_size=args.lissa_evaluation_batch_size,
            evaluation_batches=args.lissa_evaluation_batches,
        )
        print(
            f"LiSSA completed: scale={lissa.scale:.3g}, "
            f"spectral_estimate={lissa.estimated_spectral_norm}"
        )

    final_rows = []
    final_row_indices = {}
    final_row_offset = 0
    if "gradsim" in methods:
        final_row_indices["gradsim"] = final_row_offset
        final_row_offset += len(BEHAVIOR_NAMES)
        final_rows.append(
            np.stack(
                [final_behavior_gradients[name] for name in BEHAVIOR_NAMES],
                axis=0,
            )
        )
    if "lissa_if" in methods:
        final_row_indices["lissa_if"] = final_row_offset
        final_row_offset += len(BEHAVIOR_NAMES)
        final_rows.append(lissa.vectors)
    if "tracin" in methods:
        final_checkpoint = fit["checkpoints"][-1]
        final_row_indices["tracin"] = final_row_offset
        final_row_offset += len(BEHAVIOR_NAMES)
        final_rows.append(
            final_checkpoint["learning_rate"]
            * np.stack(
                [final_behavior_gradients[name] for name in BEHAVIOR_NAMES],
                axis=0,
            )
        )

    if final_rows:
        final_target = np.concatenate(final_rows, axis=0)
        final_projected = compute_projected_scores(
            model,
            data.train_x,
            data.train_observed_y,
            final_target,
            parameter_subset=args.parameter_subset,
            batch_size=args.score_batch_size,
            device=device,
        )
        scoring_seconds += final_projected.elapsed_seconds
        for method, row_index in final_row_indices.items():
            method_rows = final_projected.scores[
                :, row_index : row_index + len(BEHAVIOR_NAMES)
            ]
            if method == "tracin":
                scores[method] += method_rows.T
            else:
                scores[method] = method_rows.T

    indicators = data.train_noise_mask.numpy()
    representation_scores, ntk_scores = compute_similarity_baselines(
        model,
        data.train_x,
        data.train_observed_y,
        data.trusted_x,
        data.trusted_y,
        batch_size=args.trusted_batch_size,
        device=device,
    )
    baseline_scores = {
        "random": compute_random_scores(
            data.train_x.shape[0], seed=args.seed
        ),
        "representation_similarity": representation_scores,
        "ntk_similarity": ntk_scores,
    }
    baseline_detection = {
        method: detection_metrics(scores, indicators)
        for method, scores in baseline_scores.items()
    }

    detection = {
        method: {
            behavior: detection_metrics(scores[method][behavior_index], indicators)
            for behavior_index, behavior in enumerate(BEHAVIOR_NAMES)
        }
        for method in methods
    }
    ranking_agreement = {}
    for method in methods:
        for first_index, first_behavior in enumerate(BEHAVIOR_NAMES):
            for second_behavior in BEHAVIOR_NAMES[first_index + 1 :]:
                ranking_agreement[f"{method}:{first_behavior}_vs_{second_behavior}"] = _kendall(
                    scores[method][first_index],
                    scores[method][BEHAVIOR_NAMES.index(second_behavior)],
                )
    for behavior_index, behavior in enumerate(BEHAVIOR_NAMES):
        for first_index, first_method in enumerate(methods):
            for second_method in methods[first_index + 1 :]:
                ranking_agreement[
                    f"{behavior}:{first_method}_vs_{second_method}"
                ] = _kendall(
                    scores[first_method][behavior_index],
                    scores[second_method][behavior_index],
                )
    configurations = [
        (method, behavior)
        for method in methods
        for behavior in BEHAVIOR_NAMES
    ]
    for first_index, (first_method, first_behavior) in enumerate(configurations):
        for second_method, second_behavior in configurations[first_index + 1 :]:
            ranking_agreement[
                f"pair:{first_method}:{first_behavior}__vs__"
                f"{second_method}:{second_behavior}"
            ] = _kendall(
                scores[first_method][BEHAVIOR_NAMES.index(first_behavior)],
                scores[second_method][BEHAVIOR_NAMES.index(second_behavior)],
            )

    scores_path = args.output_dir / "scores.npz"
    np.savez_compressed(
        scores_path,
        sample_ids=data.train_ids.numpy(),
        observed_labels=data.train_observed_y.numpy(),
        clean_labels=data.train_clean_y.numpy(),
        noise_mask=indicators,
        scores=np.stack([scores[method] for method in methods], axis=0),
        method_names=np.asarray(methods),
        behavior_names=np.asarray(BEHAVIOR_NAMES),
    )

    baselines_path = args.output_dir / "baselines.npz"
    np.savez_compressed(
        baselines_path,
        sample_ids=data.train_ids.numpy(),
        scores=np.stack(
            [baseline_scores[name] for name in BASELINE_NAMES], axis=0
        ),
        method_names=np.asarray(BASELINE_NAMES),
    )

    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": final_state,
            "seed": args.seed,
            "model": args.model,
            "noise_seed": args.noise_seed,
            "noise_rate": args.noise_rate,
            "parameter_subset": args.parameter_subset,
        },
        checkpoint_path,
    )

    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))

    summary = {
        "config": {key: str(value) for key, value in vars(args).items()},
        "fit": {key: value for key, value in fit.items() if key != "history"},
        "specification": {
            "methods": list(methods),
            "transitions": {
                "gradsim": "one-step observed-label upweighting",
                "tracin": "trajectory observed-label upweighting",
                "lissa_if": "damped local re-optimization with mini-batch LiSSA",
            },
            "score_semantics": "higher_is_more_harmful",
            "behaviors": list(BEHAVIOR_NAMES),
            "parameter_subset": args.parameter_subset,
            "parameter_dimension": parameter_dimension,
            "uses_hidden_clean_labels": False,
            "uses_corruption_indicators_for_scoring": False,
        },
        "data": {
            "n_corrupted": int(data.train_x.shape[0]),
            "n_noisy": int(indicators.sum()),
            "noise_rate": args.noise_rate,
            "n_trusted": int(data.trusted_x.shape[0]),
        },
        "detection": detection,
        "baselines": {
            "method_names": list(BASELINE_NAMES),
            "detection": baseline_detection,
            "score_semantics": "higher_is_more_harmful",
            "uses_hidden_clean_labels": False,
        },
        "ranking_agreement": ranking_agreement,
        "lissa": None
        if lissa is None
        else {
            "scale": lissa.scale,
            "estimated_spectral_norm": lissa.estimated_spectral_norm,
            "iterations": lissa.iterations,
            "averaged_last": lissa.averaged_last,
            "damping": args.lissa_damping,
            "elapsed_seconds": lissa.elapsed_seconds,
            "residual_history": lissa.residual_history,
            "update_history": lissa.update_history,
            "evaluation_residual_norms": lissa.evaluation_residual_norms,
            "evaluation_relative_residual_norms": (
                lissa.evaluation_relative_residual_norms
            ),
            "evaluation_batch_size": lissa.evaluation_batch_size,
            "evaluation_batches": lissa.evaluation_batches,
        },
        "runtime": {
            "training_seconds": fit["elapsed_seconds"],
            "scoring_seconds": scoring_seconds,
            "peak_gpu_memory_bytes": peak_memory,
        },
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)

    print(f"Saved scores to {scores_path}")
    print(f"Saved baselines to {baselines_path}")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
