"""Influence estimators that share a single candidate-gradient computation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np
import torch
from torch import nn

from .behavior import (
    BEHAVIOR_NAMES,
    _encode_batch,
    lora_parameters,
)


@dataclass
class CountSketchProjection:
    """A deterministic sparse Johnson-Lindenstrauss projection.

    Each LoRA coordinate is assigned to one output bucket with a random sign.
    Unlike a dense Gaussian projection, this never materializes an
    output_dimension x input_dimension matrix.
    """

    parameter_order: tuple[str, ...]
    block_sizes: tuple[int, ...]
    output_dimension: int
    seed: int
    device: torch.device
    bucket_indices: tuple[torch.Tensor, ...]
    signs: tuple[torch.Tensor, ...]

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        output_dimension: int,
        seed: int,
        device: torch.device,
    ) -> "CountSketchProjection":
        if output_dimension <= 0:
            raise ValueError("Projection dimension must be positive")
        selected = lora_parameters(model)
        parameter_order = tuple(selected)
        block_sizes = tuple(selected[name].numel() for name in parameter_order)
        bucket_indices: list[torch.Tensor] = []
        signs: list[torch.Tensor] = []
        for block_index, name in enumerate(parameter_order):
            generator = torch.Generator(device="cpu")
            digest = hashlib.sha256(f"{seed}:{block_index}:{name}".encode()).digest()
            generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
            size = selected[name].numel()
            indices = torch.randint(
                0,
                output_dimension,
                (size,),
                generator=generator,
                dtype=torch.int32,
            )
            block_signs = torch.randint(
                0,
                2,
                (size,),
                generator=generator,
                dtype=torch.int8,
            )
            bucket_indices.append(indices.to(device))
            signs.append(block_signs.to(device))
        return cls(
            parameter_order=parameter_order,
            block_sizes=block_sizes,
            output_dimension=output_dimension,
            seed=seed,
            device=device,
            bucket_indices=tuple(bucket_indices),
            signs=tuple(signs),
        )

    @property
    def input_dimension(self) -> int:
        return sum(self.block_sizes)

    def project(self, gradient: torch.Tensor) -> torch.Tensor:
        if gradient.ndim != 1 or gradient.numel() != self.input_dimension:
            raise ValueError(
                "Gradient shape does not match the projection's LoRA dimension"
            )
        projected = torch.zeros(
            self.output_dimension,
            dtype=torch.float32,
            device=self.device,
        )
        offset = 0
        for size, indices, signs in zip(
            self.block_sizes,
            self.bucket_indices,
            self.signs,
        ):
            values = gradient.narrow(0, offset, size).to(
                device=self.device,
                dtype=torch.float32,
            )
            signed_values = torch.where(signs > 0, values, -values)
            projected.index_add_(0, indices, signed_values)
            offset += size
        return projected


def load_adam_preconditioner(
    model: nn.Module,
    training_state_path,
    *,
    device: torch.device,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return the diagonal inverse-sqrt Adam scale for the LoRA vector."""

    checkpoint = torch.load(
        training_state_path,
        map_location="cpu",
        weights_only=False,
    )
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(optimizer_state, dict) or "state" not in optimizer_state:
        raise ValueError("Training state does not contain an optimizer state")

    selected = lora_parameters(model)
    parameter_order = tuple(selected)
    state_ids: list[int] = []
    for group in optimizer_state.get("param_groups", []):
        state_ids.extend(int(parameter_id) for parameter_id in group.get("params", []))
    if len(state_ids) < len(parameter_order):
        raise ValueError("Optimizer state does not cover all LoRA parameters")

    blocks: list[torch.Tensor] = []
    for index, name in enumerate(parameter_order):
        parameter_state = optimizer_state["state"].get(state_ids[index])
        if parameter_state is None or "exp_avg_sq" not in parameter_state:
            raise ValueError(f"Optimizer state is missing exp_avg_sq for {name}")
        variance = parameter_state["exp_avg_sq"].reshape(-1).float()
        expected_shape = selected[name].reshape(-1).shape
        if variance.shape != expected_shape:
            raise ValueError(
                f"Optimizer state shape for {name} does not match the LoRA parameter"
            )
        scale = 1.0 / (torch.sqrt(variance) + epsilon)
        blocks.append(scale)
    return torch.cat(blocks).to(device=device, dtype=torch.float32)


@dataclass
class CheckpointScoreResult:
    checkpoint: str
    learning_rate: float
    parameter_dimension: int
    gradsim: np.ndarray
    gradient_norms: np.ndarray
    raw_projections: np.ndarray | None
    less_projections: np.ndarray | None
    elapsed_seconds: float
    peak_gpu_memory_bytes: int | None


def _candidate_gradient(
    model: nn.Module,
    tokenizer,
    example,
    *,
    max_length: int,
    device: torch.device,
    selected: dict[str, nn.Parameter],
    parameter_order: tuple[str, ...],
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    batch = _encode_batch(
        model,
        tokenizer,
        [example],
        max_length=max_length,
        device=device,
    )
    logits = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    ).logits
    shifted_logits = logits[:, :-1, :]
    shifted_labels = batch.labels[:, 1:]
    response_mask = shifted_labels.ne(-100)
    response_nll = nn.functional.cross_entropy(
        shifted_logits.transpose(1, 2).float(),
        shifted_labels,
        reduction="none",
    )
    mean_nll = (response_nll * response_mask).sum() / response_mask.sum()
    mean_nll.backward()
    for name in parameter_order:
        if selected[name].grad is None:
            raise RuntimeError(f"No candidate gradient was produced for {name}")
    return torch.cat(
        [
            selected[name].grad.reshape(-1).float()
            for name in parameter_order
        ]
    ).detach()


def compute_checkpoint_scores(
    model: nn.Module,
    tokenizer,
    examples,
    *,
    behavior_gradients: dict[str, np.ndarray],
    max_length: int,
    device: torch.device,
    checkpoint: str,
    learning_rate: float,
    projection: CountSketchProjection | None = None,
    preconditioner: torch.Tensor | None = None,
) -> CheckpointScoreResult:
    """Compute all per-checkpoint quantities in one candidate-gradient pass."""

    if projection is None and preconditioner is not None:
        raise ValueError("A preconditioner requires a projection")

    model.eval()
    selected = lora_parameters(model)
    parameter_order = tuple(selected)
    dimensions = sum(parameter.numel() for parameter in selected.values())
    targets = torch.as_tensor(
        np.stack([behavior_gradients[name] for name in BEHAVIOR_NAMES], axis=0),
        dtype=torch.float32,
        device=device,
    )
    if targets.shape != (len(BEHAVIOR_NAMES), dimensions):
        raise ValueError("Behavior gradients do not match the current LoRA model")
    if preconditioner is not None and preconditioner.numel() != dimensions:
        raise ValueError("Adam preconditioner does not match the LoRA model")

    gradsim = np.zeros((len(examples), len(BEHAVIOR_NAMES)), dtype=np.float32)
    gradient_norms = np.zeros(len(examples), dtype=np.float32)
    raw_projections = (
        np.zeros((len(examples), projection.output_dimension), dtype=np.float32)
        if projection is not None
        else None
    )
    less_projections = (
        np.zeros((len(examples), projection.output_dimension), dtype=np.float32)
        if projection is not None and preconditioner is not None
        else None
    )

    started = time.perf_counter()
    for index, example in enumerate(examples):
        flat_gradient = _candidate_gradient(
            model,
            tokenizer,
            example,
            max_length=max_length,
            device=device,
            selected=selected,
            parameter_order=parameter_order,
        )
        gradsim[index] = torch.mv(targets, flat_gradient).detach().cpu().numpy()
        gradient_norms[index] = (
            torch.linalg.vector_norm(flat_gradient).detach().cpu().item()
        )
        if projection is not None:
            raw_projection = projection.project(flat_gradient)
            raw_projections[index] = raw_projection.detach().cpu().numpy()
            if preconditioner is not None:
                less_gradient = flat_gradient * preconditioner
                less_projection = projection.project(less_gradient)
                less_projections[index] = less_projection.detach().cpu().numpy()

    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    model.zero_grad(set_to_none=True)
    return CheckpointScoreResult(
        checkpoint=checkpoint,
        learning_rate=float(learning_rate),
        parameter_dimension=dimensions,
        gradsim=gradsim,
        gradient_norms=gradient_norms,
        raw_projections=raw_projections,
        less_projections=less_projections,
        elapsed_seconds=time.perf_counter() - started,
        peak_gpu_memory_bytes=peak_memory,
    )


def project_behavior_gradients(
    projection: CountSketchProjection,
    behavior_gradients: dict[str, np.ndarray],
    *,
    device: torch.device,
    preconditioner: torch.Tensor | None = None,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for behavior in BEHAVIOR_NAMES:
        gradient = torch.as_tensor(
            behavior_gradients[behavior],
            dtype=torch.float32,
            device=device,
        )
        if preconditioner is not None:
            gradient = gradient * preconditioner
        projected = projection.project(gradient)
        rows.append(projected.detach().cpu().numpy())
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def compute_trakin_scores(
    checkpoint_results: list[CheckpointScoreResult],
) -> np.ndarray:
    if not checkpoint_results:
        raise ValueError("TracIn requires at least one checkpoint")
    scores = np.zeros_like(checkpoint_results[0].gradsim, dtype=np.float32)
    for result in checkpoint_results:
        scores += result.learning_rate * result.gradsim
    return scores


def compute_trak_scores(
    candidate_projections: np.ndarray,
    target_projections: np.ndarray,
    *,
    ridge: float = 1e-2,
    device: torch.device,
) -> np.ndarray:
    """Compute projected-gradient similarity with a regularized covariance."""

    if candidate_projections.ndim != 2 or target_projections.ndim != 2:
        raise ValueError("TRAK projections must both be two-dimensional")
    if candidate_projections.shape[1] != target_projections.shape[1]:
        raise ValueError("TRAK candidate and target projection dimensions differ")
    if not 0.0 <= ridge <= 1.0:
        raise ValueError("TRAK ridge must be between 0 and 1")

    candidates = torch.as_tensor(
        candidate_projections,
        dtype=torch.float32,
        device=device,
    )
    targets = torch.as_tensor(
        target_projections,
        dtype=torch.float32,
        device=device,
    )
    covariance = candidates.T @ candidates
    covariance /= candidates.shape[0]
    diagonal_scale = covariance.diagonal().mean().clamp_min(1.0)
    covariance.diagonal().add_(diagonal_scale * ridge)
    covariance_target = torch.linalg.solve(covariance, targets.T)
    scores = candidates @ covariance_target
    return scores.detach().cpu().numpy().astype(np.float32, copy=False)


def compute_less_scores(
    candidate_projections: np.ndarray,
    target_projections: np.ndarray,
) -> np.ndarray:
    if candidate_projections.ndim != 2 or target_projections.ndim != 2:
        raise ValueError("LESS projections must both be two-dimensional")
    if candidate_projections.shape[1] != target_projections.shape[1]:
        raise ValueError("LESS candidate and target projection dimensions differ")
    candidates = torch.as_tensor(candidate_projections, dtype=torch.float32)
    targets = torch.as_tensor(target_projections, dtype=torch.float32)
    scores = candidates @ targets.T
    return scores.detach().cpu().numpy().astype(np.float32, copy=False)
