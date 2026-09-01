"""Dataset loading utilities with Torch tensor outputs."""

from __future__ import annotations

import gzip
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class FashionMNISTData:
    """A compact in-memory representation of a supervised image dataset."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    feature_mean: Optional[torch.Tensor] = None
    feature_std: Optional[torch.Tensor] = None


@dataclass
class CIFAR10Data:
    """A compact in-memory representation of CIFAR-10 dataset."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    feature_mean: Optional[torch.Tensor] = None
    feature_std: Optional[torch.Tensor] = None


def _open_idx(path: Path):
    if path.exists():
        return path.open("rb")
    gz_path = Path(str(path) + ".gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rb")
    raise FileNotFoundError(f"Could not find {path} or {gz_path}")


def _read_idx_images(path: Path) -> np.ndarray:
    with _open_idx(path) as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image IDX magic number {magic} in {path}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    expected = count * rows * cols
    if values.size != expected:
        raise ValueError(f"Corrupt image file {path}: expected {expected} bytes, got {values.size}")
    return values.reshape(count, rows * cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with _open_idx(path) as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label IDX magic number {magic} in {path}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if values.size != count:
        raise ValueError(f"Corrupt label file {path}: expected {count} bytes, got {values.size}")
    return values


def _stratified_indices(labels: np.ndarray, size: Optional[int], seed: int) -> np.ndarray:
    if size is None or size >= labels.size:
        return np.arange(labels.size, dtype=np.int64)
    if size <= 0:
        raise ValueError("Subset size must be positive")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    chosen = []
    for class_position, class_id in enumerate(classes):
        class_indices = np.flatnonzero(labels == class_id)
        quota = size // len(classes) + int(class_position < size % len(classes))
        chosen.append(rng.choice(class_indices, size=min(quota, class_indices.size), replace=False))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def _transform_features(
    train_x: np.ndarray, test_x: np.ndarray, normalization: str
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    train = train_x.astype(np.float64) / 255.0
    test = test_x.astype(np.float64) / 255.0
    if normalization == "unit":
        return train, test, None, None
    if normalization == "none":
        return train_x.astype(np.float64), test_x.astype(np.float64), None, None
    if normalization != "standard":
        raise ValueError("normalization must be one of: none, unit, standard")
    mean = train.mean(axis=0)
    train_std = train.std(axis=0)
    std = np.where(train_std < 1e-6, 1.0, train_std)
    return (train - mean) / std, (test - mean) / std, mean, std


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _read_cifar10_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a single CIFAR-10 batch file (pickle format)."""
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]
    labels = batch[b"labels"]
    return data.reshape(-1, 3072), np.array(labels, dtype=np.uint8)


def load_fashion_mnist(
    root: str | Path,
    *,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 0,
    normalization: str = "unit",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> FashionMNISTData:
    """Load FashionMNIST IDX files and return Torch tensors on ``device``."""

    root = Path(root)
    raw = root / "raw" if (root / "raw").exists() else root
    train_x = _read_idx_images(raw / "train-images-idx3-ubyte")
    train_y = _read_idx_labels(raw / "train-labels-idx1-ubyte")
    test_x = _read_idx_images(raw / "t10k-images-idx3-ubyte")
    test_y = _read_idx_labels(raw / "t10k-labels-idx1-ubyte")
    train_indices = _stratified_indices(train_y, max_train, seed)
    test_indices = _stratified_indices(test_y, max_test, seed + 1)
    train_x, train_y = train_x[train_indices], train_y[train_indices]
    test_x, test_y = test_x[test_indices], test_y[test_indices]
    train_x, test_x, mean, std = _transform_features(train_x, test_x, normalization)
    resolved = resolve_device(device)
    return FashionMNISTData(
        torch.as_tensor(train_x, dtype=dtype, device=resolved),
        torch.as_tensor(train_y, dtype=torch.long, device=resolved),
        torch.as_tensor(test_x, dtype=dtype, device=resolved),
        torch.as_tensor(test_y, dtype=torch.long, device=resolved),
        None if mean is None else torch.as_tensor(mean, dtype=dtype, device=resolved),
        None if std is None else torch.as_tensor(std, dtype=dtype, device=resolved),
    )


def load_cifar10(
    root: str | Path,
    *,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 0,
    normalization: str = "unit",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> CIFAR10Data:
    """Load CIFAR-10 pickle files and return Torch tensors on ``device``.

    Args:
        root: Path to CIFAR-10 directory (expects cifar-10-batches-py/ subdirectory)
        max_train: Maximum training samples (stratified sampling if < 50000)
        max_test: Maximum test samples (stratified sampling if < 10000)
        seed: Random seed for stratified sampling
        normalization: One of 'none', 'unit', 'standard'
        device: Target device ('cpu', 'cuda', or 'auto')
        dtype: Torch dtype for features

    Returns:
        CIFAR10Data with train/test tensors on the specified device
    """
    root = Path(root)
    batch_dir = root / "cifar-10-batches-py"
    if not batch_dir.exists():
        raise FileNotFoundError(f"Expected {batch_dir} to exist. Extract cifar-10-python.tar.gz first.")

    train_batches = [batch_dir / f"data_batch_{i}" for i in range(1, 6)]
    train_x_list, train_y_list = [], []
    for batch_path in train_batches:
        x, y = _read_cifar10_batch(batch_path)
        train_x_list.append(x)
        train_y_list.append(y)
    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    test_x, test_y = _read_cifar10_batch(batch_dir / "test_batch")

    train_indices = _stratified_indices(train_y, max_train, seed)
    test_indices = _stratified_indices(test_y, max_test, seed + 1)
    train_x, train_y = train_x[train_indices], train_y[train_indices]
    test_x, test_y = test_x[test_indices], test_y[test_indices]

    train_x, test_x, mean, std = _transform_features(train_x, test_x, normalization)
    resolved = resolve_device(device)

    return CIFAR10Data(
        torch.as_tensor(train_x, dtype=dtype, device=resolved),
        torch.as_tensor(train_y, dtype=torch.long, device=resolved),
        torch.as_tensor(test_x, dtype=dtype, device=resolved),
        torch.as_tensor(test_y, dtype=torch.long, device=resolved),
        None if mean is None else torch.as_tensor(mean, dtype=dtype, device=resolved),
        None if std is None else torch.as_tensor(std, dtype=dtype, device=resolved),
    )
