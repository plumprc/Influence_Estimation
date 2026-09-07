"""Behavior values, gradients, and streaming influence-score computation."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .parameter_subsets import resolve_parameter_subset


BEHAVIOR_NAMES = ("negative_loss", "target_logit", "hard_margin")


def behavior_values(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    negative_loss = -F.cross_entropy(logits, labels, reduction="none")
    target_logit = logits.gather(1, labels[:, None]).squeeze(1)
    mask = F.one_hot(labels, num_classes=logits.shape[1]).bool()
    hard_margin = target_logit - logits.masked_fill(mask, float("-inf")).max(dim=1).values
    return {
        "negative_loss": negative_loss,
        "target_logit": target_logit,
        "hard_margin": hard_margin,
    }


def _flat_values(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([values[name] for name in BEHAVIOR_NAMES], dim=1)


@dataclass
class GradSimResult:
    scores: dict[str, np.ndarray]
    behavior_gradients: dict[str, np.ndarray]
    parameter_dimension: int
    elapsed_seconds: float
    peak_gpu_memory_bytes: int | None


@dataclass
class ProjectedScoreResult:
    scores: np.ndarray
    elapsed_seconds: float


@dataclass
class LiSSAResult:
    vectors: np.ndarray
    elapsed_seconds: float
    scale: float
    estimated_spectral_norm: float | None
    residual_history: list[list[float]]
    update_history: list[list[float]]
    evaluation_residual_norms: list[float]
    evaluation_relative_residual_norms: list[float]
    evaluation_batch_size: int
    evaluation_batches: int
    iterations: int
    averaged_last: int
    peak_gpu_memory_bytes: int | None


def _flatten_selected_gradients(
    gradients: dict[str, torch.Tensor],
    parameter_order: tuple[str, ...],
) -> torch.Tensor:
    return torch.cat([gradients[name].reshape(-1) for name in parameter_order])


def compute_behavior_gradients(
    model: nn.Module,
    trusted_x: torch.Tensor,
    trusted_y: torch.Tensor,
    *,
    parameter_subset: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Compute trusted-set average gradients for all behaviors in one pass."""
    model.eval()
    selected_parameters, _ = resolve_parameter_subset(model, parameter_subset)
    parameter_order = tuple(selected_parameters)

    behavior_sums = {
        behavior: torch.zeros(
            sum(parameter.numel() for parameter in selected_parameters.values()),
            dtype=torch.float32,
            device=device,
        )
        for behavior in BEHAVIOR_NAMES
    }
    sample_count = 0

    for start in range(0, trusted_x.shape[0], batch_size):
        batch_x = trusted_x[start : start + batch_size].to(device, non_blocking=True)
        batch_y = trusted_y[start : start + batch_size].to(device, non_blocking=True)
        differentiable = {
            name: parameter.detach().to(device).clone().requires_grad_(True)
            for name, parameter in selected_parameters.items()
        }

        def trusted_behavior_sum(parameters, x, y):
            logits = functional_call(model, parameters, (x,))
            return _flat_values(behavior_values(logits, y)).sum(dim=0)

        behavior_batch_sums = trusted_behavior_sum(differentiable, batch_x, batch_y)
        for behavior_index, behavior in enumerate(BEHAVIOR_NAMES):
            selected_grads = torch.autograd.grad(
                behavior_batch_sums[behavior_index],
                list(differentiable.values()),
                retain_graph=True,
            )
            gradients = {
                name: selected_grads[index]
                for index, name in enumerate(parameter_order)
            }
            flat = _flatten_selected_gradients(gradients, parameter_order)
            behavior_sums[behavior] += flat.detach()
        sample_count += batch_x.shape[0]

    behavior_gradients = {
        behavior: (behavior_sums[behavior] / sample_count).detach().cpu().numpy()
        for behavior in BEHAVIOR_NAMES
    }
    return behavior_gradients, parameter_order


def compute_projected_scores(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    target_matrix: np.ndarray | torch.Tensor,
    *,
    parameter_subset: str,
    batch_size: int,
    device: torch.device,
) -> ProjectedScoreResult:
    """Project streaming per-sample gradients onto a fixed target matrix."""
    model.eval()
    selected_parameters, parameter_dimension = resolve_parameter_subset(model, parameter_subset)
    parameter_order = tuple(selected_parameters)
    targets = torch.as_tensor(target_matrix, dtype=torch.float32, device=device)
    if targets.ndim != 2 or targets.shape[1] != parameter_dimension:
        raise ValueError(
            "target_matrix must have shape (num_targets, parameter_dimension); "
            f"got {tuple(targets.shape)}, expected (*, {parameter_dimension})"
        )

    differentiable = {
        name: parameter.detach().to(device).clone().requires_grad_(True)
        for name, parameter in selected_parameters.items()
    }

    def sample_loss(parameters, x, y):
        logits = functional_call(model, parameters, (x.unsqueeze(0),))
        return F.cross_entropy(logits, y.unsqueeze(0))

    sample_grad = grad(sample_loss)

    def projected_score(parameters, x, y, rows):
        gradients = sample_grad(parameters, x, y)
        flat = _flatten_selected_gradients(gradients, parameter_order)
        return torch.mv(rows, flat)

    batched_projected_score = vmap(projected_score, in_dims=(None, 0, 0, None))

    scores = np.zeros((train_x.shape[0], targets.shape[0]), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, train_x.shape[0], batch_size):
        batch_x = train_x[start : start + batch_size].to(device, non_blocking=True)
        batch_y = train_y[start : start + batch_size].to(device, non_blocking=True)
        batch_scores = batched_projected_score(
            differentiable,
            batch_x,
            batch_y,
            targets,
        )
        scores[start : start + batch_x.shape[0]] = batch_scores.detach().cpu().numpy()

    return ProjectedScoreResult(
        scores=scores,
        elapsed_seconds=time.perf_counter() - started,
    )


def compute_gradsim_scores(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    trusted_x: torch.Tensor,
    trusted_y: torch.Tensor,
    *,
    parameter_subset: str,
    train_batch_size: int,
    trusted_batch_size: int,
    device: torch.device,
) -> GradSimResult:
    """Compute GradSim harm scores for all behaviors in one streaming pass."""
    behavior_gradients, _ = compute_behavior_gradients(
        model,
        trusted_x,
        trusted_y,
        parameter_subset=parameter_subset,
        batch_size=trusted_batch_size,
        device=device,
    )
    target_matrix = np.stack(
        [behavior_gradients[name] for name in BEHAVIOR_NAMES],
        axis=0,
    )
    projected = compute_projected_scores(
        model,
        train_x,
        train_y,
        target_matrix,
        parameter_subset=parameter_subset,
        batch_size=train_batch_size,
        device=device,
    )
    _, parameter_dimension = resolve_parameter_subset(model, parameter_subset)
    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    return GradSimResult(
        scores={
            behavior: projected.scores[:, index]
            for index, behavior in enumerate(BEHAVIOR_NAMES)
        },
        behavior_gradients=behavior_gradients,
        parameter_dimension=parameter_dimension,
        elapsed_seconds=projected.elapsed_seconds,
        peak_gpu_memory_bytes=peak_memory,
    )


def _as_parameter_dict(
    flat_vector: torch.Tensor,
    selected_parameters: dict[str, nn.Parameter],
) -> dict[str, torch.Tensor]:
    result = {}
    offset = 0
    for name, parameter in selected_parameters.items():
        count = parameter.numel()
        result[name] = flat_vector[offset : offset + count].view_as(parameter)
        offset += count
    return result


def compute_lissa_vectors(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    behavior_gradients: dict[str, np.ndarray],
    *,
    parameter_subset: str,
    batch_size: int,
    iterations: int,
    damping: float,
    scale: float,
    seed: int,
    device: torch.device,
    average_last: int = 5,
    auto_scale: bool = True,
    power_iterations: int = 5,
    evaluation_batch_size: int = 256,
    evaluation_batches: int = 4,
) -> LiSSAResult:
    """Approximate H^-1 b with damped stochastic fixed-point iterations.

    The iteration is ``v <- v + (b - H_batch v - damping*v) / scale``. Its
    fixed point is ``(H + damping I)^-1 b``. The minibatch Hessian estimate is
    recomputed at every step, and no per-sample Hessian is stored.
    """
    if iterations <= 0:
        raise ValueError("LiSSA iterations must be positive")
    if damping < 0.0:
        raise ValueError("LiSSA damping must be non-negative")
    if scale <= 0.0:
        raise ValueError("LiSSA scale must be positive")
    if not 1 <= average_last <= iterations:
        raise ValueError("average_last must be in [1, iterations]")
    if evaluation_batch_size <= 0:
        raise ValueError("LiSSA evaluation batch size must be positive")
    if evaluation_batches <= 0:
        raise ValueError("LiSSA evaluation batches must be positive")

    model.eval()
    selected_parameters, _ = resolve_parameter_subset(model, parameter_subset)
    parameter_order = tuple(selected_parameters)
    differentiable = {
        name: parameter.detach().to(device).clone().requires_grad_(True)
        for name, parameter in selected_parameters.items()
    }

    def batch_loss(parameters, x, y):
        logits = functional_call(model, parameters, (x,))
        return F.cross_entropy(logits, y)

    batch_grad = grad(batch_loss)

    def batch_hvp(parameters, x, y, vector_dict):
        gradients = batch_grad(parameters, x, y)
        product = sum(
            (gradients[name] * vector_dict[name]).sum()
            for name in parameter_order
        )
        hessian_gradients = torch.autograd.grad(product, list(parameters.values()))
        return {
            name: hessian_gradients[index]
            for index, name in enumerate(parameter_order)
        }

    rhs = torch.as_tensor(
        np.stack([behavior_gradients[name] for name in BEHAVIOR_NAMES], axis=0),
        dtype=torch.float32,
        device=device,
    )
    vectors = torch.zeros_like(rhs)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(
            0,
            train_x.shape[0],
            (batch_size,),
            generator=generator,
        )
        return (
            train_x[indices].to(device, non_blocking=True),
            train_y[indices].to(device, non_blocking=True),
        )

    estimated_spectral_norm = None
    if auto_scale:
        probe = torch.randn(rhs.shape[1], dtype=torch.float32, device=device)
        probe = probe / torch.linalg.vector_norm(probe)
        for _ in range(power_iterations):
            batch_x, batch_y = sample_batch()
            probe_dict = _as_parameter_dict(probe, selected_parameters)
            hvp = batch_hvp(differentiable, batch_x, batch_y, probe_dict)
            flat_hvp = _flatten_selected_gradients(hvp, parameter_order)
            probe_norm = torch.linalg.vector_norm(flat_hvp)
            estimated_spectral_norm = max(
                estimated_spectral_norm or 0.0,
                float(probe_norm.item()),
            )
            if probe_norm > 0:
                probe = flat_hvp / probe_norm
        scale = max(scale, 2.2 * (estimated_spectral_norm + damping))

    residual_history: list[list[float]] = []
    update_history: list[list[float]] = []
    average = torch.zeros_like(vectors)
    average_count = 0
    started = time.perf_counter()
    for iteration in range(iterations):
        batch_x, batch_y = sample_batch()
        residual_norms = []
        update_norms = []
        for behavior_index in range(rhs.shape[0]):
            vector_dict = _as_parameter_dict(vectors[behavior_index], selected_parameters)
            hvp = batch_hvp(differentiable, batch_x, batch_y, vector_dict)
            flat_hvp = _flatten_selected_gradients(hvp, parameter_order)
            residual = rhs[behavior_index] - flat_hvp - damping * vectors[behavior_index]
            update = residual / scale
            vectors[behavior_index] = vectors[behavior_index] + update
            residual_norms.append(float(torch.linalg.vector_norm(residual).item()))
            update_norms.append(float(torch.linalg.vector_norm(update).item()))
        residual_history.append(residual_norms)
        update_history.append(update_norms)
        if not torch.isfinite(vectors).all():
            raise FloatingPointError("LiSSA produced non-finite vectors")
        if iteration >= iterations - average_last:
            average = average + vectors
            average_count += 1

    vectors = average / average_count

    evaluation_residual_norms: list[float] = []
    evaluation_relative_residual_norms: list[float] = []
    evaluation_hvp_sums = torch.zeros_like(vectors)
    evaluation_sample_count = 0
    for batch_index in range(evaluation_batches):
        start = batch_index * evaluation_batch_size
        if start >= train_x.shape[0]:
            break
        stop = min(start + evaluation_batch_size, train_x.shape[0])
        batch_x = train_x[start:stop].to(device, non_blocking=True)
        batch_y = train_y[start:stop].to(device, non_blocking=True)
        for behavior_index in range(rhs.shape[0]):
            vector_dict = _as_parameter_dict(
                vectors[behavior_index], selected_parameters
            )
            hvp = batch_hvp(differentiable, batch_x, batch_y, vector_dict)
            flat_hvp = _flatten_selected_gradients(hvp, parameter_order)
            evaluation_hvp_sums[behavior_index] += flat_hvp * batch_x.shape[0]
        evaluation_sample_count += batch_x.shape[0]

    used_evaluation_batches = max(
        1,
        min(
            evaluation_batches,
            (train_x.shape[0] + evaluation_batch_size - 1)
            // evaluation_batch_size,
        ),
    )
    evaluation_hvp = evaluation_hvp_sums / evaluation_sample_count
    for behavior_index in range(rhs.shape[0]):
        residual = (
            rhs[behavior_index]
            - evaluation_hvp[behavior_index]
            - damping * vectors[behavior_index]
        )
        residual_norm = torch.linalg.vector_norm(residual)
        rhs_norm = torch.linalg.vector_norm(rhs[behavior_index])
        evaluation_residual_norms.append(float(residual_norm.item()))
        evaluation_relative_residual_norms.append(
            float((residual_norm / rhs_norm).item())
        )

    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    return LiSSAResult(
        vectors=vectors.detach().cpu().numpy(),
        elapsed_seconds=time.perf_counter() - started,
        scale=float(scale),
        estimated_spectral_norm=(
            float(estimated_spectral_norm)
            if estimated_spectral_norm is not None
            else None
        ),
        residual_history=residual_history,
        update_history=update_history,
        evaluation_residual_norms=evaluation_residual_norms,
        evaluation_relative_residual_norms=evaluation_relative_residual_norms,
        evaluation_batch_size=evaluation_batch_size,
        evaluation_batches=used_evaluation_batches,
        iterations=iterations,
        averaged_last=average_count,
        peak_gpu_memory_bytes=peak_memory,
    )
