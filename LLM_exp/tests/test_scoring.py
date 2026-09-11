from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from influence.scoring import (
    CheckpointScoreResult,
    CountSketchProjection,
    compute_less_scores,
    compute_trakin_scores,
    compute_trak_scores,
    load_adam_preconditioner,
)


class TinyLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_a = nn.Parameter(torch.arange(3, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.arange(2, dtype=torch.float32) + 10)


def test_count_sketch_projection_is_deterministic():
    model = TinyLoRAModel()
    projection = CountSketchProjection.from_model(
        model,
        output_dimension=4,
        seed=7,
        device=torch.device("cpu"),
    )
    gradient = torch.arange(5, dtype=torch.float32)

    first = projection.project(gradient)
    second = projection.project(gradient)

    assert first.shape == (4,)
    assert torch.equal(first, second)
    assert projection.input_dimension == 5


def test_trak_scores_use_regularized_covariance():
    candidates = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    targets = np.asarray([[1.0, 0.0]], dtype=np.float32)

    scores = compute_trak_scores(
        candidates,
        targets,
        ridge=0.1,
        device=torch.device("cpu"),
    )

    # Covariance is diag(0.5, 2.0); mean diagonal is 1.25 and ridge adds 0.125.
    expected = np.asarray(
        [[1.0 / 0.625], [0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-6)


def test_less_scores_are_projected_inner_products():
    candidates = np.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=np.float32)
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    scores = compute_less_scores(candidates, targets)

    np.testing.assert_allclose(scores, [[1.0, 2.0], [3.0, -1.0]])


def test_trakin_weights_checkpoint_similarities():
    first = CheckpointScoreResult(
        checkpoint="first",
        learning_rate=2.0,
        parameter_dimension=2,
        gradsim=np.asarray([[1.0, 2.0]], dtype=np.float32),
        gradient_norms=np.asarray([1.0]),
        raw_projections=None,
        less_projections=None,
        elapsed_seconds=0.0,
        peak_gpu_memory_bytes=None,
    )
    second = CheckpointScoreResult(
        checkpoint="final",
        learning_rate=0.5,
        parameter_dimension=2,
        gradsim=np.asarray([[3.0, -4.0]], dtype=np.float32),
        gradient_norms=np.asarray([1.0]),
        raw_projections=None,
        less_projections=None,
        elapsed_seconds=0.0,
        peak_gpu_memory_bytes=None,
    )

    scores = compute_trakin_scores([first, second])

    np.testing.assert_allclose(scores, [[3.5, 2.0]])


def test_adam_preconditioner_loads_optimizer_state(tmp_path):
    model = TinyLoRAModel()
    state_path = tmp_path / "training_state.pt"
    torch.save(
        {
            "optimizer": {
                "state": {
                    0: {"exp_avg_sq": torch.asarray([4.0, 9.0, 16.0])},
                    1: {"exp_avg_sq": torch.asarray([1.0, 25.0])},
                },
                "param_groups": [{"params": [0, 1]}],
            }
        },
        state_path,
    )

    preconditioner = load_adam_preconditioner(
        model,
        state_path,
        device=torch.device("cpu"),
    )

    expected = 1.0 / (
        torch.sqrt(torch.asarray([4.0, 9.0, 16.0, 1.0, 25.0])) + 1e-8
    )
    torch.testing.assert_close(preconditioner, expected)


def test_missing_tracin_checkpoint_can_be_required(tmp_path):
    from experiments.score import _discover_checkpoints

    adapter = tmp_path / "adapter"
    adapter.mkdir()

    with pytest.raises(FileNotFoundError, match="checkpoint_050"):
        _discover_checkpoints(adapter, [0.5], allow_missing=False)

    checkpoints = _discover_checkpoints(adapter, [0.5], allow_missing=True)

    assert checkpoints == [(adapter, None)]
