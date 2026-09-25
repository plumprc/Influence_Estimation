"""Representation-gradient signals for ScienceQA representation similarity."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn

from .behavior import (
    BEHAVIOR_NAMES,
    _alternative_token_ids,
    _encode_batch,
    behavior_values,
)
from .data import Example


BASE_METHOD_NAMES = (
    "repsim_train_repr",
    "repsim_behavior_repr",
    "repsim_dual_repr",
)
NORMALIZATION_MODES = ("raw", "l2", "rms")
METHOD_NAMES = tuple(
    f"{method}_{mode}"
    for method in BASE_METHOD_NAMES
    for mode in NORMALIZATION_MODES
)


@dataclass
class RepSimExtensionResult:
    scores: np.ndarray
    method_names: tuple[str, ...]
    behavior_names: tuple[str, ...]
    layer_index: int
    elapsed_seconds: float
    peak_gpu_memory_bytes: int | None


def _transformer_layers(model: nn.Module) -> nn.ModuleList:
    candidates: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if (name == "layers" or name.endswith(".layers")) and hasattr(
            module, "__len__"
        ):
            try:
                if len(module) > 0:
                    candidates.append((name, module))
            except TypeError:
                continue
    if not candidates:
        raise ValueError("Could not locate transformer layers")
    _, layers = max(candidates, key=lambda item: len(item[0]))
    return layers


def _select_transformer_layer(
    model: nn.Module,
    layer_index: int,
) -> nn.Module:
    layers = _transformer_layers(model)
    if not -len(layers) <= layer_index < len(layers):
        raise ValueError(
            f"Layer index {layer_index} is out of range for {len(layers)} layers"
        )
    return layers[layer_index]


def _capture_forward_output(module, inputs, output):
    del module, inputs
    return output[0] if isinstance(output, tuple) else output


def _positions(
    batch,
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    response_mask = batch.labels[:, 1:].ne(-100)
    response_positions = [
        response_mask[row].nonzero(as_tuple=False).view(-1).to(device)
        for row in range(response_mask.shape[0])
    ]
    explanation_positions = [
        torch.tensor(
            [position - 1 for position in positions],
            device=device,
            dtype=torch.long,
        )
        for positions in batch.explanation_positions
    ]
    answer_positions = torch.tensor(
        [positions[0] - 1 for positions in batch.answer_positions],
        device=device,
        dtype=torch.long,
    )
    return response_positions, explanation_positions, answer_positions


def _mean_at_positions(
    values: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if positions.numel() == 0:
        raise ValueError("Cannot average over an empty position set")
    return values[positions].float().mean(dim=0)


def _batch_signals(
    model: nn.Module,
    tokenizer,
    examples: list[Example],
    *,
    layer_index: int,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return features, behavior gradients, and training gradients.

    Shapes are ``(batch, behavior, hidden_dimension)``.
    """
    model.eval()
    batch = _encode_batch(
        model,
        tokenizer,
        examples,
        max_length=max_length,
        device=device,
    )
    alternatives = _alternative_token_ids(tokenizer, examples)
    layer = _select_transformer_layer(model, layer_index)
    captured: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(module, inputs, output):
        del module, inputs
        captured["features"] = output[0] if isinstance(output, tuple) else output

    def backward_hook(module, grad_input, grad_output):
        del module, grad_input
        gradients["value"] = grad_output[0]

    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)
    try:
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
        values = behavior_values(
            outputs.logits,
            batch,
            alternative_token_ids=alternatives,
        )
        behavior_gradients: dict[str, torch.Tensor] = {}
        for index, behavior in enumerate(BEHAVIOR_NAMES):
            gradients.pop("value", None)
            values[behavior].sum().backward(
                retain_graph=index < len(BEHAVIOR_NAMES) - 1
            )
            if "value" not in gradients:
                raise RuntimeError(
                    f"No representation gradient was captured for {behavior}"
                )
            behavior_gradients[behavior] = gradients["value"].detach().clone()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    features = captured["features"].detach()
    response_positions, explanation_positions, answer_positions = _positions(
        batch,
        device=device,
    )
    hidden_dimension = features.shape[-1]
    feature_rows = torch.zeros(
        (len(examples), len(BEHAVIOR_NAMES), hidden_dimension),
        dtype=torch.float32,
        device=device,
    )
    behavior_rows = torch.zeros_like(feature_rows)
    training_rows = torch.zeros_like(feature_rows)

    for row in range(len(examples)):
        answer_position = answer_positions[row]
        response_position = response_positions[row]
        explanation_position = explanation_positions[row]

        feature_rows[row, 0] = _mean_at_positions(
            features[row],
            response_position,
        )
        feature_rows[row, 1] = _mean_at_positions(
            features[row],
            explanation_position,
        )
        feature_rows[row, 2] = features[row, answer_position].float()
        feature_rows[row, 3] = features[row, answer_position].float()

        response_behavior = _mean_at_positions(
            behavior_gradients["negative_response_loss"][row],
            response_position,
        )
        explanation_behavior = _mean_at_positions(
            behavior_gradients["negative_explanation_loss"][row],
            explanation_position,
        )
        answer_behavior = behavior_gradients["answer_logit"][
            row, answer_position
        ].float()
        margin_behavior = behavior_gradients["answer_margin"][
            row, answer_position
        ].float()

        behavior_rows[row, 0] = response_behavior
        behavior_rows[row, 1] = explanation_behavior
        behavior_rows[row, 2] = answer_behavior
        behavior_rows[row, 3] = margin_behavior

        training_rows[row, 0] = -response_behavior
        training_rows[row, 1] = -explanation_behavior
        training_rows[row, 2] = -answer_behavior
        training_rows[row, 3] = -answer_behavior

    model.zero_grad(set_to_none=True)
    return feature_rows, behavior_rows, training_rows


def _extract(
    model: nn.Module,
    tokenizer,
    examples: list[Example],
    *,
    batch_size: int,
    layer_index: int,
    max_length: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    behavior: list[np.ndarray] = []
    training: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch_features, batch_behavior, batch_training = _batch_signals(
            model,
            tokenizer,
            batch_examples,
            layer_index=layer_index,
            max_length=max_length,
            device=device,
        )
        features.append(batch_features.cpu().numpy())
        behavior.append(batch_behavior.cpu().numpy())
        training.append(batch_training.cpu().numpy())
    return (
        np.concatenate(features, axis=0),
        np.concatenate(behavior, axis=0),
        np.concatenate(training, axis=0),
    )


def compute_repsim_extension_scores(
    model: nn.Module,
    tokenizer,
    train_examples: list[Example],
    validation_examples: list[Example],
    *,
    batch_size: int,
    layer_index: int,
    max_length: int,
    device: torch.device,
) -> RepSimExtensionResult:
    """Compute behavior-specific representation-gradient RepSim variants."""
    started = time.perf_counter()
    query_features, query_behavior, query_training = _extract(
        model,
        tokenizer,
        validation_examples,
        batch_size=batch_size,
        layer_index=layer_index,
        max_length=max_length,
        device=device,
    )
    train_features, train_behavior, train_training = _extract(
        model,
        tokenizer,
        train_examples,
        batch_size=batch_size,
        layer_index=layer_index,
        max_length=max_length,
        device=device,
    )

    query_features_t = torch.as_tensor(
        query_features,
        dtype=torch.float32,
        device=device,
    )
    train_features_t = torch.as_tensor(
        train_features,
        dtype=torch.float32,
        device=device,
    )
    query_behavior_t = torch.as_tensor(
        query_behavior,
        dtype=torch.float32,
        device=device,
    )
    query_training_t = torch.as_tensor(
        query_training,
        dtype=torch.float32,
        device=device,
    )
    train_behavior_t = torch.as_tensor(
        train_behavior,
        dtype=torch.float32,
        device=device,
    )
    train_training_t = torch.as_tensor(
        train_training,
        dtype=torch.float32,
        device=device,
    )

    query_norm = torch.nn.functional.normalize(query_features_t, dim=2)
    train_norm = torch.nn.functional.normalize(train_features_t, dim=2)
    kernel = torch.einsum("qbd,mbd->qmb", query_norm, train_norm)

    def normalize_signal(values: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "raw":
            return values
        if mode == "l2":
            return torch.nn.functional.normalize(values, dim=2, eps=1e-12)
        if mode == "rms":
            rms = values.square().mean(dim=2, keepdim=True).sqrt().clamp_min(
                1e-12
            )
            return values / rms
        raise ValueError(f"Unknown signal normalization mode: {mode}")

    method_scores: list[torch.Tensor] = []
    for method in BASE_METHOD_NAMES:
        for mode in NORMALIZATION_MODES:
            if method == "repsim_train_repr":
                query_signal = normalize_signal(query_training_t, mode)
                train_signal = normalize_signal(train_training_t, mode)
            elif method == "repsim_behavior_repr":
                query_signal = normalize_signal(query_behavior_t, mode)
                train_signal = normalize_signal(train_behavior_t, mode)
            elif method == "repsim_dual_repr":
                query_signal = normalize_signal(query_behavior_t, mode)
                train_signal = normalize_signal(train_training_t, mode)
            else:
                raise ValueError(f"Unknown method: {method}")
            pair = torch.einsum(
                "qbd,mbd->qmb",
                query_signal,
                train_signal,
            )
            method_scores.append((kernel * pair).mean(dim=0))

    scores = torch.stack(method_scores, dim=0)
    scores_np = scores.permute(1, 2, 0).detach().cpu().numpy().astype(
        np.float32,
        copy=False,
    )
    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    return RepSimExtensionResult(
        scores=scores_np,
        method_names=METHOD_NAMES,
        behavior_names=BEHAVIOR_NAMES,
        layer_index=layer_index,
        elapsed_seconds=time.perf_counter() - started,
        peak_gpu_memory_bytes=peak_memory,
    )
