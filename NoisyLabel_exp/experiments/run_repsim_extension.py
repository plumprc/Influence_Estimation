"""Run signal-augmented representation similarity on the noisy-label setup."""

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
if str(PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from influence.data import make_noisy_label_split
from influence.metrics import detection_metrics
from influence.models import make_model
from influence.repsim_extensions import (
    METHOD_NAMES,
    compute_repsim_extension_scores,
)
from influence.training import accuracy, train_model
from run_noisy_label import MODEL_TRAINING_DEFAULTS


TOP_FRACTIONS = (0.01, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run signal-augmented representation similarity"
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
        default=(
            PROJECT_ROOT
            / "outputs"
            / "repsim_extension"
            / "rho_0.2"
            / "seed_0"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--trusted-batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    return parser.parse_args()


def _kendall(first: np.ndarray, second: np.ndarray) -> float:
    tau, _ = kendalltau(first, second)
    return float(tau)


def _topk_overlap(
    reference: np.ndarray,
    estimate: np.ndarray,
    fraction: float,
) -> float:
    count = max(1, int(round(reference.size * fraction)))
    reference_top = set(np.argsort(reference)[-count:].tolist())
    estimate_top = set(np.argsort(estimate)[-count:].tolist())
    return len(reference_top & estimate_top) / count


def _metric_payload(
    scores: np.ndarray,
    indicators: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        method: {
            behavior: detection_metrics(
                scores[method_index, behavior_index],
                indicators,
            )
            for behavior_index, behavior in enumerate(
                ("negative_loss", "target_logit", "hard_margin")
            )
        }
        for method_index, method in enumerate(METHOD_NAMES)
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    defaults = MODEL_TRAINING_DEFAULTS["resnet18"]
    if args.epochs is None:
        args.epochs = int(defaults["epochs"])
    if args.learning_rate is None:
        args.learning_rate = float(defaults["learning_rate"])

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
    model = make_model("resnet18").to(device)

    print(
        f"Loaded {data.train_x.shape[0]} corrupted, "
        f"{data.trusted_x.shape[0]} trusted, {data.test_x.shape[0]} test samples"
    )
    print(f"Noisy labels: {int(data.train_noise_mask.sum().item())}")
    print(
        "Methods: "
        + ", ".join(METHOD_NAMES)
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

    started = time.perf_counter()
    extension = compute_repsim_extension_scores(
        model,
        data.train_x,
        data.train_observed_y,
        data.trusted_x,
        data.trusted_y,
        trusted_batch_size=args.trusted_batch_size,
        score_batch_size=args.score_batch_size,
        device=device,
    )
    extension_seconds = time.perf_counter() - started
    indicators = data.train_noise_mask.numpy()
    detection = _metric_payload(extension.scores, indicators)

    reference_index = METHOD_NAMES.index("repsim")
    ranking_agreement: dict[str, dict[str, float]] = {}
    topk_overlap: dict[str, dict[str, dict[str, float]]] = {}
    for behavior_index, behavior in enumerate(extension.behavior_names):
        reference = extension.scores[reference_index, behavior_index]
        ranking_agreement[behavior] = {
            method: _kendall(
                reference,
                extension.scores[method_index, behavior_index],
            )
            for method_index, method in enumerate(METHOD_NAMES)
        }
        topk_overlap[behavior] = {
            method: {
                f"top_{int(fraction * 100)}pct": _topk_overlap(
                    reference,
                    extension.scores[method_index, behavior_index],
                    fraction,
                )
                for fraction in TOP_FRACTIONS
            }
            for method_index, method in enumerate(METHOD_NAMES)
        }

    scores_path = args.output_dir / "scores.npz"
    np.savez_compressed(
        scores_path,
        sample_ids=data.train_ids.numpy(),
        observed_labels=data.train_observed_y.numpy(),
        clean_labels=data.train_clean_y.numpy(),
        noise_mask=indicators,
        scores=extension.scores,
        method_names=np.asarray(extension.method_names),
        behavior_names=np.asarray(extension.behavior_names),
    )

    signals_path = args.output_dir / "signals.npz"
    np.savez_compressed(
        signals_path,
        class_counts=extension.class_counts,
        logit_behavior_gradients=extension.logit_behavior_gradients,
        representation_behavior_gradients=(
            extension.representation_behavior_gradients
        ),
        behavior_names=np.asarray(extension.behavior_names),
    )

    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "seed": args.seed,
            "noise_seed": args.noise_seed,
            "noise_rate": args.noise_rate,
            "model": "resnet18",
        },
        checkpoint_path,
    )

    summary = {
        "schema_version": 1,
        "config": {key: str(value) for key, value in vars(args).items()},
        "fit": {key: value for key, value in fit.items() if key != "history"},
        "data": {
            "n_corrupted": int(data.train_x.shape[0]),
            "n_noisy": int(indicators.sum()),
            "noise_rate": args.noise_rate,
            "n_trusted": int(data.trusted_x.shape[0]),
            "class_counts": extension.class_counts.tolist(),
        },
        "method_names": list(extension.method_names),
        "behavior_names": list(extension.behavior_names),
        "detection": detection,
        "ranking_agreement_vs_repsim": ranking_agreement,
        "topk_overlap_vs_repsim": topk_overlap,
        "runtime": {
            "training_seconds": fit["elapsed_seconds"],
            "extension_seconds": extension_seconds,
            "extension_internal_seconds": extension.elapsed_seconds,
            "peak_gpu_memory_bytes": extension.peak_gpu_memory_bytes,
        },
    }
    summary_path = args.output_dir / "summary.json"
    _write_json_atomic(summary_path, summary)

    print(f"Saved scores to {scores_path}")
    print(f"Saved signals to {signals_path}")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
