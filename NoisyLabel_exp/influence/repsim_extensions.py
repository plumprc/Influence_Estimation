"""Signal-augmented representation similarity for noisy-label detection."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .baselines import compute_similarity_baselines
from .behaviors import BEHAVIOR_NAMES, behavior_values


METHOD_NAMES = (
    "repsim",
    "repsim_train_logit",
    "repsim_behavior_logit",
    "repsim_dual_logit",
    "repsim_train_repr",
    "repsim_behavior_repr",
    "repsim_dual_repr",
)

@dataclass
class RepSimExtensionResult:
    scores: np.ndarray
    method_names: tuple[str, ...]
    behavior_names: tuple[str, ...]
    class_counts: np.ndarray
    logit_behavior_gradients: np.ndarray
    representation_behavior_gradients: np.ndarray
    elapsed_seconds: float
    peak_gpu_memory_bytes: int | None


def _head_layer(model: nn.Module) -> nn.Linear:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, "classifier"):
        linear_layers = [
            module
            for module in model.classifier.modules()
            if isinstance(module, nn.Linear)
        ]
        if linear_layers:
            return linear_layers[-1]
    raise ValueError("Could not identify a linear classification head")


def _forward_with_head_capture(
    model: nn.Module,
    head: nn.Linear,
    batch_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def capture_head(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module
        captured["features"] = inputs[0]
        captured["logits"] = output

    hook = head.register_forward_hook(capture_head)
    try:
        model(batch_x)
    finally:
        hook.remove()
    return captured["features"], captured["logits"]


def _class_mean_accumulate(
    destination: torch.Tensor,
    values: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    for class_id in range(destination.shape[0]):
        selected = labels == class_id
        if bool(selected.any()):
            destination[class_id] += values[selected].sum(dim=0)


def _trusted_statistics(
    model: nn.Module,
    head: nn.Linear,
    trusted_x: torch.Tensor,
    trusted_y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_count = head.out_features
    feature_dimension = head.in_features
    logit_sums = {
        behavior: torch.zeros(
            (class_count, class_count),
            dtype=torch.float64,
            device=device,
        )
        for behavior in BEHAVIOR_NAMES
    }
    representation_sums = {
        behavior: torch.zeros(
            (class_count, feature_dimension),
            dtype=torch.float64,
            device=device,
        )
        for behavior in BEHAVIOR_NAMES
    }
    counts = torch.zeros(class_count, dtype=torch.float32)

    with torch.enable_grad():
        for start in range(0, trusted_x.shape[0], batch_size):
            batch_x = trusted_x[start : start + batch_size].to(
                device, non_blocking=True
            )
            batch_y = trusted_y[start : start + batch_size].to(
                device, non_blocking=True
            )
            features, logits = _forward_with_head_capture(model, head, batch_x)
            if not features.requires_grad or not logits.requires_grad:
                raise RuntimeError(
                    "Trusted features and logits must require gradients"
                )

            values = behavior_values(logits, batch_y)
            for class_id in range(class_count):
                selected = batch_y == class_id
                if bool(selected.any()):
                    counts[class_id] += float(selected.sum().item())

            for behavior_index, behavior in enumerate(BEHAVIOR_NAMES):
                behavior_sum = values[behavior].sum()
                logit_gradient = torch.autograd.grad(
                    behavior_sum,
                    logits,
                    retain_graph=True,
                )[0]
                representation_gradient = torch.autograd.grad(
                    behavior_sum,
                    features,
                    retain_graph=behavior_index < len(BEHAVIOR_NAMES) - 1,
                )[0]
                _class_mean_accumulate(
                    logit_sums[behavior],
                    logit_gradient.detach().to(torch.float64),
                    batch_y,
                )
                _class_mean_accumulate(
                    representation_sums[behavior],
                    representation_gradient.detach().to(torch.float64),
                    batch_y,
                )

    if bool((counts == 0).any()):
        raise ValueError("The trusted set must contain samples from all classes")
    class_counts = counts.to(torch.long).cpu().numpy()
    counts_device = counts.to(device)

    logit_gradients = np.stack(
        [
            (
                logit_sums[behavior]
                / counts_device[:, None]
            ).cpu().numpy()
            for behavior in BEHAVIOR_NAMES
        ],
        axis=0,
    )
    representation_gradients = np.stack(
        [
            (
                representation_sums[behavior]
                / counts_device[:, None]
            ).cpu().numpy()
            for behavior in BEHAVIOR_NAMES
        ],
        axis=0,
    )
    return class_counts, logit_gradients, representation_gradients


def compute_repsim_extension_scores(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    trusted_x: torch.Tensor,
    trusted_y: torch.Tensor,
    *,
    trusted_batch_size: int,
    score_batch_size: int,
    device: torch.device,
) -> RepSimExtensionResult:
    """Compute behavior-free RepSim and four signal-augmented variants.

    Training and behavior signals are evaluated both in logit space and in the
    final representation space. Candidate scores use observed labels only.
    """
    model.eval()
    started = time.perf_counter()
    head = _head_layer(model)
    class_count = head.out_features
    reference_repsim, _ = compute_similarity_baselines(
        model,
        train_x,
        train_y,
        trusted_x,
        trusted_y,
        batch_size=trusted_batch_size,
        device=device,
    )
    class_counts, logit_gradients, representation_gradients = (
        _trusted_statistics(
            model,
            head,
            trusted_x,
            trusted_y,
            batch_size=trusted_batch_size,
            device=device,
        )
    )
    reference_repsim_t = torch.as_tensor(
        reference_repsim,
        dtype=torch.float32,
        device=device,
    )
    logit_gradients_t = torch.as_tensor(
        logit_gradients,
        dtype=torch.float32,
        device=device,
    )
    representation_gradients_t = torch.as_tensor(
        representation_gradients,
        dtype=torch.float32,
        device=device,
    )
    class_targets = F.one_hot(
        torch.arange(class_count, device=device),
        num_classes=class_count,
    ).to(torch.float32)
    representation_targets = head.weight.detach().to(torch.float32)

    scores = np.zeros(
        (len(METHOD_NAMES), len(BEHAVIOR_NAMES), len(train_y)),
        dtype=np.float32,
    )
    with torch.no_grad():
        for start in range(0, train_x.shape[0], score_batch_size):
            end = min(start + score_batch_size, train_x.shape[0])
            batch_x = train_x[start:end].to(device, non_blocking=True)
            batch_y = train_y[start:end].to(device, non_blocking=True)
            features, logits = _forward_with_head_capture(model, head, batch_x)
            rows = slice(start, end)
            reference = reference_repsim_t[rows]
            kernel = -reference

            residual = (
                F.softmax(logits, dim=1)
                - F.one_hot(batch_y, num_classes=class_count)
            )
            representation_train_signal = residual @ head.weight.detach()
            for behavior_index, behavior in enumerate(BEHAVIOR_NAMES):
                behavior_logit = logit_gradients_t[behavior_index, batch_y]
                behavior_repr = representation_gradients_t[
                    behavior_index, batch_y
                ]
                target_logit = class_targets[batch_y]
                target_repr = representation_targets[batch_y]

                train_logit = (target_logit * residual).sum(dim=1)
                behavior_logit_score = (behavior_logit * target_logit).sum(dim=1)
                dual_logit = (behavior_logit * residual).sum(dim=1)
                train_repr = (
                    target_repr * representation_train_signal
                ).sum(dim=1)
                behavior_repr_score = (
                    behavior_repr * target_repr
                ).sum(dim=1)
                dual_repr = (
                    behavior_repr * representation_train_signal
                ).sum(dim=1)

                scores[0, behavior_index, rows] = reference.cpu().numpy()
                scores[1, behavior_index, rows] = (
                    -kernel * train_logit
                ).cpu().numpy()
                scores[2, behavior_index, rows] = (
                    -kernel * behavior_logit_score
                ).cpu().numpy()
                scores[3, behavior_index, rows] = (
                    -kernel * dual_logit
                ).cpu().numpy()
                scores[4, behavior_index, rows] = (
                    -kernel * train_repr
                ).cpu().numpy()
                scores[5, behavior_index, rows] = (
                    -kernel * behavior_repr_score
                ).cpu().numpy()
                scores[6, behavior_index, rows] = (
                    -kernel * dual_repr
                ).cpu().numpy()

    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    return RepSimExtensionResult(
        scores=scores,
        method_names=METHOD_NAMES,
        behavior_names=BEHAVIOR_NAMES,
        class_counts=class_counts,
        logit_behavior_gradients=logit_gradients,
        representation_behavior_gradients=representation_gradients,
        elapsed_seconds=time.perf_counter() - started,
        peak_gpu_memory_bytes=peak_memory,
    )
