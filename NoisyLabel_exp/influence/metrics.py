"""Noisy-label detection metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_recall_at_fraction(
    scores: np.ndarray,
    indicators: np.ndarray,
    fraction: float,
) -> tuple[float, float]:
    if scores.shape != indicators.shape:
        raise ValueError("scores and indicators must have the same shape")
    count = max(1, int(round(scores.shape[0] * fraction)))
    top_indices = np.argsort(-scores, kind="stable")[:count]
    selected = indicators[top_indices]
    precision = float(selected.mean())
    total_positive = float(indicators.sum())
    recall = float(selected.sum() / total_positive) if total_positive else float("nan")
    return precision, recall


def detection_metrics(scores: np.ndarray, indicators: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    indicators = np.asarray(indicators, dtype=np.int8)
    if scores.ndim != 1 or indicators.ndim != 1:
        raise ValueError("scores and indicators must be one-dimensional")
    if np.unique(indicators).size < 2:
        return {
            "auprc": float("nan"),
            "auroc": float("nan"),
        }

    result = {
        "auprc": float(average_precision_score(indicators, scores)),
        "auroc": float(roc_auc_score(indicators, scores)),
    }
    for fraction in (0.01, 0.05, 0.10):
        precision, recall = precision_recall_at_fraction(
            scores,
            indicators,
            fraction,
        )
        label = int(round(fraction * 100))
        result[f"precision_at_{label}pct"] = precision
        result[f"recall_at_{label}pct"] = recall
    return result
