"""Behavior-free noisy-label detection baselines."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


BASELINE_NAMES = ("random", "representation_similarity", "ntk_similarity")


def compute_random_scores(
    sample_count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Deterministic random baseline; higher is more harmful."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20260902]))
    return rng.random(sample_count).astype(np.float32)


def _last_layer_gradient_from_outputs(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    residual = F.softmax(logits, dim=1) - F.one_hot(
        labels, num_classes=logits.shape[1]
    )
    weight_gradient = residual[:, :, None] * features[:, None, :]
    return torch.cat([weight_gradient.flatten(start_dim=1), residual], dim=1)


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


def compute_similarity_baselines(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    trusted_x: torch.Tensor,
    trusted_y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute representation and NTK baselines in one streaming pass."""
    model.eval()
    head = _head_layer(model)
    feature_dimension = head.in_features
    gradient_dimension = (
        head.in_features * head.out_features + head.out_features
    )
    feature_sums = torch.zeros(10, feature_dimension, dtype=torch.float32)
    gradient_sums = torch.zeros(10, gradient_dimension, dtype=torch.float32)
    counts = torch.zeros(10, dtype=torch.float32)
    captured: dict[str, torch.Tensor] = {}

    def capture_head(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["features"] = inputs[0]
        captured["logits"] = output

    hook = head.register_forward_hook(capture_head)

    with torch.no_grad():
        for start in range(0, trusted_x.shape[0], batch_size):
            batch_x = trusted_x[start : start + batch_size].to(
                device, non_blocking=True
            )
            batch_y = trusted_y[start : start + batch_size].to(
                device, non_blocking=True
            )
            model(batch_x)
            features = captured["features"]
            logits = captured["logits"]
            normalized_features = F.normalize(features, dim=1)
            gradients = _last_layer_gradient_from_outputs(
                features, logits, batch_y
            )
            for class_id in range(10):
                selected = batch_y == class_id
                if torch.any(selected):
                    feature_sums[class_id] += normalized_features[selected].sum(
                        dim=0
                    ).cpu()
                    gradient_sums[class_id] += gradients[selected].sum(dim=0).cpu()
                    counts[class_id] += int(selected.sum().item())

    if torch.any(counts == 0):
        raise ValueError("The trusted set must contain samples from all 10 classes")
    representation_centroids = F.normalize(
        feature_sums / counts[:, None], dim=1, eps=1e-12
    ).to(device)
    ntk_centroids = F.normalize(
        gradient_sums / counts[:, None], dim=1, eps=1e-12
    ).to(device)

    representation_scores = np.empty(train_x.shape[0], dtype=np.float32)
    ntk_scores = np.empty(train_x.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, train_x.shape[0], batch_size):
            batch_x = train_x[start : start + batch_size].to(
                device, non_blocking=True
            )
            batch_y = train_y[start : start + batch_size].to(
                device, non_blocking=True
            )
            model(batch_x)
            features = captured["features"]
            logits = captured["logits"]
            normalized_features = F.normalize(features, dim=1)
            gradients = _last_layer_gradient_from_outputs(
                features, logits, batch_y
            )
            normalized_gradients = F.normalize(gradients, dim=1, eps=1e-12)
            representation_similarities = torch.einsum(
                "nd,nd->n",
                normalized_features,
                representation_centroids[batch_y],
            )
            ntk_similarities = torch.einsum(
                "nd,nd->n", normalized_gradients, ntk_centroids[batch_y]
            )
            representation_scores[start : start + batch_x.shape[0]] = (
                -representation_similarities.cpu().numpy().astype(np.float32)
            )
            ntk_scores[start : start + batch_x.shape[0]] = (
                -ntk_similarities.cpu().numpy().astype(np.float32)
            )

    hook.remove()
    return representation_scores, ntk_scores
