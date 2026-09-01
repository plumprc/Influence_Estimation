"""Torch metrics for comparing influence values and rankings."""

from __future__ import annotations

import numpy as np
import torch


def _as_tensor(values: torch.Tensor | np.ndarray) -> torch.Tensor:
    return values.detach() if isinstance(values, torch.Tensor) else torch.as_tensor(values)


def topk_overlap(first: torch.Tensor | np.ndarray, second: torch.Tensor | np.ndarray, fraction: float = 0.05) -> float:
    first, second = _as_tensor(first).reshape(-1), _as_tensor(second).reshape(-1)
    k = max(1, int(np.ceil(first.numel() * fraction)))
    first_top = set(torch.topk(first, k=k).indices.cpu().tolist())
    second_top = set(torch.topk(second, k=k).indices.cpu().tolist())
    return len(first_top & second_top) / k


def sign_accuracy(first: torch.Tensor | np.ndarray, second: torch.Tensor | np.ndarray, tolerance: float = 0.0) -> float:
    first, second = _as_tensor(first), _as_tensor(second)
    mask = (first.abs() > tolerance) | (second.abs() > tolerance)
    if not bool(mask.any()):
        return float("nan")
    return float((torch.sign(first[mask]) == torch.sign(second[mask])).double().mean())


def _kendall_tau_b(first: torch.Tensor, second: torch.Tensor) -> float:
    """Compute Kendall's tau-b on CPU to avoid GPU memory issues."""

    first, second = first.reshape(-1).cpu(), second.reshape(-1).cpu()
    n = first.numel()
    if n < 2:
        return float("nan")

    # For large n, use batched computation to avoid memory explosion
    if n > 10000:
        # Use scipy's implementation for large arrays
        import scipy.stats
        tau, _ = scipy.stats.kendalltau(first.numpy(), second.numpy())
        return float(tau) if not np.isnan(tau) else float("nan")

    # Original implementation for smaller arrays
    left = first[:, None] - first[None, :]
    right = second[:, None] - second[None, :]
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=first.device), diagonal=1)
    left, right = left[upper], right[upper]
    concordant = ((left > 0) & (right > 0) | (left < 0) & (right < 0)).sum().double()
    discordant = ((left > 0) & (right < 0) | (left < 0) & (right > 0)).sum().double()
    ties_first = (left == 0).sum().double()
    ties_second = (right == 0).sum().double()
    denominator = torch.sqrt((concordant + discordant + ties_first) * (concordant + discordant + ties_second))
    if float(denominator) == 0.0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def compare_scores(reference: torch.Tensor | np.ndarray, estimate: torch.Tensor | np.ndarray, *, near_zero_tolerance: float = 0.0) -> dict[str, float]:
    reference = _as_tensor(reference).double().reshape(-1)
    estimate = _as_tensor(estimate).double().reshape(-1).to(reference.device)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    centered = reference - reference.mean()
    nrmse = torch.linalg.vector_norm(estimate - reference) / torch.clamp(torch.linalg.vector_norm(centered), min=1e-12)
    return {
        "nrmse": float(nrmse),
        "kendall_tau": _kendall_tau_b(reference, estimate),
        "sign_accuracy": sign_accuracy(reference, estimate, near_zero_tolerance),
        "top_5pct_overlap": topk_overlap(reference, estimate, 0.05),
    }
