from __future__ import annotations

import numpy as np

from influence.metrics import detection_metrics, ranking_agreement


def test_detection_metrics_for_perfect_scores():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    scores = np.asarray([-1.0, 2.0, 0.0, 1.0])
    metrics = detection_metrics(scores, labels)
    assert metrics["auprc"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["recall_at_true_count"] == 1.0


def test_ranking_agreement_for_identical_scores():
    scores = np.asarray([3.0, 1.0, 2.0, -1.0])
    metrics = ranking_agreement(scores, scores.copy())
    assert metrics["kendall_tau"] == 1.0
    assert metrics["top_5pct_overlap"] == 1.0
    assert metrics["sign_agreement"] == 1.0
