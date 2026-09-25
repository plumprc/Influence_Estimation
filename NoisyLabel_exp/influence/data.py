"""Deterministic CIFAR-10 splits with synthetic label noise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import torch


@dataclass
class NoisyLabelData:
    train_x: torch.Tensor
    train_observed_y: torch.Tensor
    train_clean_y: torch.Tensor
    train_ids: torch.Tensor
    train_noise_mask: torch.Tensor
    trusted_x: torch.Tensor
    trusted_y: torch.Tensor
    trusted_ids: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_ids: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


def _read_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        entry = pickle.load(stream, encoding="latin1")
    images = np.asarray(entry["data"], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
    labels = np.asarray(entry["labels"], dtype=np.int64)
    return images, labels


def _load_cifar10(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_dir = root / "cifar-10-batches-py"
    if not batch_dir.exists():
        raise FileNotFoundError(f"Expected CIFAR-10 batches under {batch_dir}")

    train_batches = [batch_dir / f"data_batch_{index}" for index in range(1, 6)]
    train_images, train_labels = zip(*(_read_cifar_batch(path) for path in train_batches))
    test_images, test_labels = _read_cifar_batch(batch_dir / "test_batch")
    return (
        np.concatenate(train_images, axis=0),
        np.concatenate(train_labels, axis=0),
        test_images,
        test_labels,
    )


def _stratified_indices(labels: np.ndarray, count_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for class_id in np.unique(labels):
        candidates = np.flatnonzero(labels == class_id)
        if candidates.size < count_per_class:
            raise ValueError(
                f"Class {class_id} has only {candidates.size} samples, "
                f"but {count_per_class} were requested"
            )
        chosen.extend(rng.choice(candidates, size=count_per_class, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def _sample_corrupted_subset(
    labels: np.ndarray,
    available_indices: np.ndarray,
    max_count: Optional[int],
    seed: int,
) -> np.ndarray:
    if max_count is None or max_count >= available_indices.size:
        return available_indices
    rng = np.random.default_rng(seed)
    target_per_class = max_count // 10
    remainder = max_count - target_per_class * 10
    chosen: list[int] = []
    for class_id in range(10):
        candidates = available_indices[labels[available_indices] == class_id]
        take = target_per_class + (1 if class_id < remainder else 0)
        if take:
            chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def make_noisy_label_split(
    data_root: str | Path,
    *,
    noise_rate: float = 0.2,
    trusted_per_class: int = 500,
    seed: int = 0,
    noise_seed: int = 0,
    max_corrupted: Optional[int] = None,
) -> NoisyLabelData:
    """Create a deterministic clean-trusted / corrupted-train / clean-test split.

    The hidden clean labels and corruption indicators are returned for evaluation
    only. Influence code must consume ``train_observed_y``.
    """
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be in [0, 1]")
    if trusted_per_class <= 0:
        raise ValueError("trusted_per_class must be positive")

    train_images, train_labels, test_images, test_labels = _load_cifar10(Path(data_root))
    trusted_indices = _stratified_indices(train_labels, trusted_per_class, seed)
    trusted_mask = np.zeros(train_labels.shape[0], dtype=bool)
    trusted_mask[trusted_indices] = True
    corrupted_indices = np.flatnonzero(~trusted_mask)
    corrupted_indices = _sample_corrupted_subset(
        train_labels, corrupted_indices, max_corrupted, seed + 1
    )

    observed_labels = train_labels[corrupted_indices].copy()
    noise_mask = np.zeros(observed_labels.shape[0], dtype=bool)
    if noise_rate > 0.0:
        rng = np.random.default_rng(noise_seed)
        flip = rng.random(observed_labels.shape[0]) < noise_rate
        alternative_labels = rng.integers(0, 9, size=observed_labels.shape[0])
        alternative_labels = alternative_labels + (alternative_labels >= observed_labels)
        observed_labels[flip] = alternative_labels[flip]
        noise_mask = flip

    trusted_images = train_images[trusted_indices]
    trusted_clean_labels = train_labels[trusted_indices]
    corrupted_images = train_images[corrupted_indices]

    mean = corrupted_images.mean(axis=(0, 2, 3), keepdims=True).reshape(1, 3, 1, 1)
    std = corrupted_images.std(axis=(0, 2, 3), keepdims=True).reshape(1, 3, 1, 1)
    std = np.maximum(std, 1e-6)

    def normalize(images: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((images - mean) / std, dtype=torch.float32)

    return NoisyLabelData(
        train_x=normalize(corrupted_images),
        train_observed_y=torch.as_tensor(observed_labels, dtype=torch.long),
        train_clean_y=torch.as_tensor(train_labels[corrupted_indices], dtype=torch.long),
        train_ids=torch.as_tensor(corrupted_indices, dtype=torch.long),
        train_noise_mask=torch.as_tensor(noise_mask, dtype=torch.bool),
        trusted_x=normalize(trusted_images),
        trusted_y=torch.as_tensor(trusted_clean_labels, dtype=torch.long),
        trusted_ids=torch.as_tensor(trusted_indices, dtype=torch.long),
        test_x=normalize(test_images),
        test_y=torch.as_tensor(test_labels, dtype=torch.long),
        test_ids=torch.arange(test_labels.shape[0], dtype=torch.long),
        mean=torch.as_tensor(mean.reshape(-1), dtype=torch.float32),
        std=torch.as_tensor(std.reshape(-1), dtype=torch.float32),
    )
