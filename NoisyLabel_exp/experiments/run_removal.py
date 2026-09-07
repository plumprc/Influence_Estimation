"""Top-k removal and retraining utility for noisy-label influence rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.behaviors import BEHAVIOR_NAMES, behavior_values
from influence.data import make_noisy_label_split
from influence.models import make_model
from influence.training import accuracy, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain after removing top-k ranked noisy-label candidates"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to run_dir/removal",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--behaviors", nargs="+", choices=BEHAVIOR_NAMES, default=None)
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=("random", "representation_similarity", "ntk_similarity"),
        default=(),
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=(0.01, 0.05, 0.10),
    )
    parser.add_argument("--retrain-seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    return parser.parse_args()


def _behavior_averages(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {name: 0.0 for name in BEHAVIOR_NAMES}
    count = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch_x = x[start : start + batch_size].to(device, non_blocking=True)
            batch_y = y[start : start + batch_size].to(device, non_blocking=True)
            values = behavior_values(model(batch_x), batch_y)
            for name in BEHAVIOR_NAMES:
                sums[name] += float(values[name].sum().item())
            count += batch_x.shape[0]
    return {name: sums[name] / count for name in BEHAVIOR_NAMES}


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    summary_path = args.run_dir / "summary.json"
    scores_path = args.run_dir / "scores.npz"
    checkpoint_path = args.run_dir / "checkpoint.pt"
    for path in (summary_path, scores_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    config = summary["config"]
    model_name = str(config.get("model", "resnet18"))

    with np.load(scores_path, allow_pickle=False) as archive:
        method_names = tuple(str(value) for value in archive["method_names"].tolist())
        behavior_names = tuple(
            str(value) for value in archive["behavior_names"].tolist()
        )
        sample_ids = archive["sample_ids"]
        scores = archive["scores"].copy()

    baselines = tuple(args.baselines)
    baseline_scores = None
    baseline_method_names: tuple[str, ...] = ()
    if baselines:
        baselines_path = args.run_dir / "baselines.npz"
        if not baselines_path.exists():
            raise FileNotFoundError(
                "Missing baselines.npz; rerun experiments/run_noisy_label.py"
            )
        with np.load(baselines_path, allow_pickle=False) as archive:
            baseline_method_names = tuple(
                str(value) for value in archive["method_names"].tolist()
            )
            baseline_sample_ids = archive["sample_ids"]
            baseline_scores = archive["scores"].copy()
        if not np.array_equal(baseline_sample_ids, sample_ids):
            raise ValueError("baselines.npz and scores.npz sample IDs do not match")
        unknown_baselines = [
            method for method in baselines if method not in baseline_method_names
        ]
        if unknown_baselines:
            raise ValueError(
                f"Unknown baselines in baselines.npz: {unknown_baselines}"
            )

    if behavior_names != BEHAVIOR_NAMES:
        raise ValueError(
            f"Unexpected behavior names: {behavior_names}; expected {BEHAVIOR_NAMES}"
        )

    methods = method_names if args.methods is None else tuple(args.methods)
    behaviors = tuple(args.behaviors or BEHAVIOR_NAMES)
    unknown_methods = [method for method in methods if method not in method_names]
    if unknown_methods:
        raise ValueError(f"Unknown methods in scores.npz: {unknown_methods}")

    data = make_noisy_label_split(
        config["data_root"],
        noise_rate=float(config["noise_rate"]),
        trusted_per_class=int(config["trusted_per_class"]),
        seed=int(config["seed"]),
        noise_seed=int(config["noise_seed"]),
        max_corrupted=(
            None
            if config.get("max_corrupted") in (None, "None")
            else int(config["max_corrupted"])
        ),
    )
    if not np.array_equal(sample_ids, data.train_ids.numpy()):
        raise ValueError("scores.npz sample IDs do not match the reconstructed split")

    output_dir = args.output_dir or args.run_dir / "removal"
    output_dir.mkdir(parents=True, exist_ok=True)
    removal_summary_path = output_dir / "removal_summary.json"
    removal_config = {
        "model": model_name,
        "methods": list(methods),
        "behaviors": list(behaviors),
        "baselines": list(baselines),
        "source_scores_sha256": _file_sha256(scores_path),
        "source_checkpoint_sha256": _file_sha256(checkpoint_path),
        "source_baselines_sha256": (
            _file_sha256(args.run_dir / "baselines.npz") if baselines else None
        ),
        "fractions": list(args.fractions),
        "retrain_seeds": list(args.retrain_seeds),
        "epochs": (
            args.epochs if args.epochs is not None else int(config["epochs"])
        ),
        "train_batch_size": (
            args.train_batch_size
            if args.train_batch_size is not None
            else int(config["train_batch_size"])
        ),
        "learning_rate": (
            args.learning_rate
            if args.learning_rate is not None
            else float(config["learning_rate"])
        ),
        "momentum": (
            args.momentum
            if args.momentum is not None
            else float(config["momentum"])
        ),
        "weight_decay": (
            args.weight_decay
            if args.weight_decay is not None
            else float(config["weight_decay"])
        ),
    }
    if removal_summary_path.exists():
        with removal_summary_path.open(encoding="utf-8") as stream:
            removal_summary = json.load(stream)
        # Legacy ResNet-18 summaries predate the architecture-aware entry point.
        removal_summary["config"].setdefault("model", "resnet18")
        if removal_summary.get("config") != removal_config:
            raise ValueError(
                "Existing removal summary was created with a different config; "
                "use a new --output-dir or remove the old removal result"
            )
    else:
        removal_summary = {
            "source_run": str(args.run_dir.resolve()),
            "config": removal_config,
            "results": {},
        }

    device = torch.device(args.device)
    factual_model = make_model(model_name).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    factual_model.load_state_dict(checkpoint["model_state_dict"])
    factual_model.eval()
    factual_behavior = _behavior_averages(
        factual_model,
        data.trusted_x,
        data.trusted_y,
        batch_size=int(config["trusted_batch_size"]),
        device=device,
    )
    factual_test_accuracy = accuracy(
        factual_model,
        data.test_x,
        data.test_y,
        batch_size=int(config["trusted_batch_size"]),
        device=device,
    )

    selected_ids_path = output_dir / "selected_ids.npz"
    selected_ids = {}
    if selected_ids_path.exists():
        with np.load(selected_ids_path, allow_pickle=False) as archive:
            selected_ids = {
                key: archive[key].copy()
                for key in archive.files
                if key in removal_summary["results"]
            }

    indicators = data.train_noise_mask.numpy().astype(np.int8)
    total_train = data.train_x.shape[0]
    ranking_specs = []
    for method in methods:
        method_index = method_names.index(method)
        for behavior in behaviors:
            behavior_index = BEHAVIOR_NAMES.index(behavior)
            ranking_specs.append(
                (method, behavior, scores[method_index, behavior_index])
            )
    for baseline in baselines:
        baseline_index = baseline_method_names.index(baseline)
        ranking_specs.append(
            (baseline, "behavior_free", baseline_scores[baseline_index])
        )

    for method, behavior, method_scores in ranking_specs:
        for fraction in args.fractions:
            count = max(1, int(round(total_train * fraction)))
            ranking = np.argsort(-method_scores, kind="stable")
            removed_positions = ranking[:count]
            removed_sample_ids = sample_ids[removed_positions]
            selected_noise = indicators[removed_positions]

            for retrain_seed in args.retrain_seeds:
                result_key = f"{method}:{behavior}:{fraction:.3f}:{retrain_seed}"
                if result_key in removal_summary["results"]:
                    if result_key not in selected_ids:
                        selected_ids[result_key] = removed_sample_ids
                        _atomic_npz(selected_ids_path, selected_ids)
                    print(f"Skipping existing result {result_key}")
                    continue

                keep_mask = np.ones(total_train, dtype=bool)
                keep_mask[removed_positions] = False
                retrain_x = data.train_x[keep_mask]
                retrain_y = data.train_observed_y[keep_mask]

                torch.manual_seed(retrain_seed)
                np.random.seed(retrain_seed)
                model = make_model(model_name).to(device)
                started = time.perf_counter()
                train_model(
                    model,
                    retrain_x,
                    retrain_y,
                    data.trusted_x,
                    data.trusted_y,
                    device=device,
                    epochs=(
                        args.epochs
                        if args.epochs is not None
                        else int(config["epochs"])
                    ),
                    batch_size=(
                        args.train_batch_size
                        if args.train_batch_size is not None
                        else int(config["train_batch_size"])
                    ),
                    learning_rate=(
                        args.learning_rate
                        if args.learning_rate is not None
                        else float(config["learning_rate"])
                    ),
                    momentum=(
                        args.momentum
                        if args.momentum is not None
                        else float(config["momentum"])
                    ),
                    weight_decay=(
                        args.weight_decay
                        if args.weight_decay is not None
                        else float(config["weight_decay"])
                    ),
                )
                retrained_behavior = _behavior_averages(
                    model,
                    data.trusted_x,
                    data.trusted_y,
                    batch_size=int(config["trusted_batch_size"]),
                    device=device,
                )
                retrained_test_accuracy = accuracy(
                    model,
                    data.test_x,
                    data.test_y,
                    batch_size=int(config["trusted_batch_size"]),
                    device=device,
                )

                behavior_delta = {
                    name: retrained_behavior[name] - factual_behavior[name]
                    for name in BEHAVIOR_NAMES
                }
                result = {
                    "method": method,
                    "behavior": behavior,
                    "fraction": fraction,
                    "removed_count": int(count),
                    "retrain_seed": int(retrain_seed),
                    "selected_noise_count": int(selected_noise.sum()),
                    "selected_noise_rate": float(selected_noise.mean()),
                    "factual_behavior": factual_behavior,
                    "retrained_behavior": retrained_behavior,
                    "behavior_delta": behavior_delta,
                    "factual_test_accuracy": factual_test_accuracy,
                    "retrained_test_accuracy": retrained_test_accuracy,
                    "runtime_seconds": time.perf_counter() - started,
                }
                removal_summary["results"][result_key] = result
                selected_ids[result_key] = removed_sample_ids

                _atomic_json(removal_summary_path, removal_summary)
                _atomic_npz(selected_ids_path, selected_ids)
                print(
                    f"Completed {result_key}: "
                    f"selected_noise_rate={selected_noise.mean():.3f}, "
                    f"test_accuracy={retrained_test_accuracy:.3f}"
                )

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"Saved removal summary to {removal_summary_path}")
    print(f"Saved selected IDs to {selected_ids_path}")


if __name__ == "__main__":
    main()
