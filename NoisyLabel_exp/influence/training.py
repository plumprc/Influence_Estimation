"""Training loop for factual checkpoints."""

from __future__ import annotations

from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F


def accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch_x = x[start : start + batch_size].to(device, non_blocking=True)
            batch_y = y[start : start + batch_size].to(device, non_blocking=True)
            predictions = model(batch_x).argmax(dim=1)
            correct += int((predictions == batch_y).sum().item())
    return correct / x.shape[0]


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    trusted_x: torch.Tensor,
    trusted_y: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int | None = None,
) -> dict:
    """Train once and save the factual checkpoints needed by TracIn."""
    if checkpoint_dir is not None and checkpoint_interval is None:
        raise ValueError("checkpoint_interval is required when checkpoint_dir is set")
    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    checkpoints = []
    model.to(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        permutation = torch.randperm(train_x.shape[0])
        running_loss = 0.0
        running_count = 0
        for start in range(0, train_x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = train_x[indices].to(device, non_blocking=True)
            batch_y = train_y[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_x.shape[0]
            running_count += batch_x.shape[0]

        next_learning_rate = float(scheduler.get_last_lr()[0])
        trusted_accuracy = accuracy(
            model,
            trusted_x,
            trusted_y,
            batch_size=batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(running_count, 1),
                "trusted_accuracy": trusted_accuracy,
                "learning_rate": next_learning_rate,
            }
        )
        print(
            f"Epoch {epoch + 1:03d}/{epochs}: "
            f"loss={history[-1]['train_loss']:.4f}, "
            f"trusted_acc={trusted_accuracy:.4f}"
        )

        if (
            checkpoint_dir is not None
            and ((epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs)
        ):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "epoch": epoch + 1,
                    "learning_rate": epoch_learning_rate,
                },
                checkpoint_path,
            )
            checkpoints.append(
                {
                    "path": str(checkpoint_path),
                    "epoch": epoch + 1,
                    "learning_rate": epoch_learning_rate,
                }
            )
        scheduler.step()

    return {
        "elapsed_seconds": time.perf_counter() - started,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_trusted_accuracy": history[-1]["trusted_accuracy"] if history else None,
        "history": history,
        "checkpoints": checkpoints,
    }


# Backward-compatible name for existing experiment entry points.
train_resnet18 = train_model
