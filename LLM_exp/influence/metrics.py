"""Detection and ranking metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score, roc_auc_score


def _mean_std_ci(values: list[float | None]) -> dict[str, float | int]:
    array = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    ci95 = (
        float(1.959963984540054 * std / np.sqrt(array.size))
        if array.size > 1
        else 0.0
    )
    return {
        "mean": float(array.mean()),
        "std": std,
        "ci95": ci95,
        "n": int(array.size),
    }


def detection_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("Scores and labels must be matching one-dimensional arrays")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain non-finite values")
    positive_count = int(labels.sum())
    if positive_count == 0 or positive_count == len(labels):
        return {
            "auprc": None,
            "auroc": None,
            "p_at_1pct": None,
            "p_at_5pct": None,
            "p_at_10pct": None,
            "recall_at_true_count": None,
        }

    def precision_at_fraction(fraction: float) -> float:
        count = max(1, int(round(len(scores) * fraction)))
        top = np.argsort(-scores, kind="stable")[:count]
        return float(labels[top].mean())

    true_count_index = np.argsort(-scores, kind="stable")[:positive_count]
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "p_at_1pct": precision_at_fraction(0.01),
        "p_at_5pct": precision_at_fraction(0.05),
        "p_at_10pct": precision_at_fraction(0.10),
        "recall_at_true_count": float(labels[true_count_index].sum() / positive_count),
    }


def ranking_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Expected matching one-dimensional score arrays")
    if len(first) < 2:
        return {"kendall_tau": None, "top_5pct_overlap": None, "sign_agreement": None}

    tau = kendalltau(first, second).statistic
    count = max(1, int(round(len(first) * 0.05)))
    first_top = set(np.argsort(-first, kind="stable")[:count].tolist())
    second_top = set(np.argsort(-second, kind="stable")[:count].tolist())
    signs_equal = np.sign(first) == np.sign(second)
    return {
        "kendall_tau": float(tau) if np.isfinite(tau) else None,
        "top_5pct_overlap": float(len(first_top & second_top) / count),
        "sign_agreement": float(signs_equal.mean()),
    }
