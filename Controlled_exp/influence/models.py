"""Torch models used by the controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FitResult:
    success: bool
    message: str
    iterations: int
    objective: float
    gradient_norm: float


class MultinomialLogisticRegression:
    """L2-regularized multinomial logistic regression implemented in Torch.

    The factual objective is mean cross-entropy plus ``0.5 * l2 * ||W||^2``;
    the bias is not regularized. Use float64 for the convex reference setting.
    """

    def __init__(
        self,
        n_classes: int = 10,
        l2: float = 1e-4,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.n_classes = n_classes
        self.l2 = float(l2)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.dtype = dtype
        self.weights: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        self.fit_result: Optional[FitResult] = None

    def _pack(self) -> torch.Tensor:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted")
        return torch.cat((self.weights.reshape(-1), self.bias))

    def _unpack(self, parameters: torch.Tensor, n_features: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight_size = self.n_classes * n_features
        return parameters[:weight_size].reshape(self.n_classes, n_features), parameters[weight_size:]

    def _objective_and_gradient(
        self,
        parameters: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        extra_index: Optional[int] = None,
        extra_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, bias = self._unpack(parameters, x.shape[1])
        logits = x @ weights.T + bias
        per_example = F.cross_entropy(logits, y, reduction="none")
        objective = per_example.mean() + 0.5 * self.l2 * (weights * weights).sum()
        residual = torch.softmax(logits, dim=1).clone()
        residual[torch.arange(x.shape[0], device=x.device), y] -= 1.0
        grad_w = residual.T @ x / x.shape[0] + self.l2 * weights
        grad_b = residual.mean(dim=0)
        if extra_index is not None and extra_weight != 0.0:
            if not 0 <= extra_index < x.shape[0]:
                raise IndexError("extra_index is outside the training set")
            objective = objective + extra_weight * per_example[extra_index]
            grad_w = grad_w + extra_weight * torch.outer(residual[extra_index], x[extra_index])
            grad_b = grad_b + extra_weight * residual[extra_index]
        return objective, torch.cat((grad_w.reshape(-1), grad_b))

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        max_iter: int = 200,
        tol: float = 1e-9,
        gtol: float = 1e-6,
        initial_parameters: Optional[torch.Tensor] = None,
        extra_index: Optional[int] = None,
        extra_weight: float = 0.0,
    ) -> FitResult:
        """Fit with Torch L-BFGS, optionally adding one example loss term."""

        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("x must be 2-D and y must have the same number of rows")
        if torch.any((y < 0) | (y >= self.n_classes)):
            raise ValueError("labels are outside the configured class range")
        n_features = x.shape[1]
        if initial_parameters is None:
            initial_parameters = torch.zeros(
                self.n_classes * n_features + self.n_classes,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            initial_parameters = torch.as_tensor(
                initial_parameters, dtype=self.dtype, device=self.device
            ).detach().clone()
        parameters = torch.nn.Parameter(initial_parameters)
        optimizer = torch.optim.LBFGS(
            [parameters],
            max_iter=max_iter,
            tolerance_grad=gtol,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            optimizer.zero_grad()
            objective, gradient = self._objective_and_gradient(
                parameters, x, y, extra_index=extra_index, extra_weight=extra_weight
            )
            parameters.grad = gradient
            closure_calls += 1
            return objective

        optimizer.step(closure)
        objective, gradient = self._objective_and_gradient(
            parameters, x, y, extra_index=extra_index, extra_weight=extra_weight
        )
        self.weights, self.bias = self._unpack(parameters.detach(), n_features)
        self.weights = self.weights.detach().clone()
        self.bias = self.bias.detach().clone()
        if not torch.isfinite(objective) or not torch.isfinite(parameters).all():
            raise FloatingPointError("Logistic-regression optimization produced non-finite values")
        gradient_norm = float(gradient.norm().detach().cpu())
        self.fit_result = FitResult(
            success=gradient_norm <= max(gtol * 10.0, 1e-5),
            message="Torch LBFGS completed",
            iterations=closure_calls,
            objective=float(objective.detach().cpu()),
            gradient_norm=gradient_norm,
        )
        return self.fit_result

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fitted")
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        return x @ self.weights.T + self.bias

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x).argmax(dim=1)

    def accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        return float((self.predict(x) == y).double().mean().cpu())

    def per_example_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        return F.cross_entropy(self.logits(x), y, reduction="none")

    def example_gradient(self, x: torch.Tensor, y: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradient of one unregularized example loss with respect to W and b."""

        x = torch.as_tensor(x, dtype=self.dtype, device=self.device).reshape(-1)
        residual = self.probabilities(x.unsqueeze(0))[0].clone()
        residual[int(y)] -= 1.0
        return torch.outer(residual, x), residual

    def copy(self) -> "MultinomialLogisticRegression":
        copied = MultinomialLogisticRegression(
            self.n_classes, self.l2, device=self.device, dtype=self.dtype
        )
        if self.weights is not None:
            copied.weights = self.weights.detach().clone()
        if self.bias is not None:
            copied.bias = self.bias.detach().clone()
        copied.fit_result = self.fit_result
        return copied


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10: Conv→ReLU→Pool→Conv→ReLU→Pool→FC→ReLU→FC.

    Architecture:
        Conv(3→32, 3×3) → ReLU → MaxPool(2×2)
        → Conv(32→64, 3×3) → ReLU → MaxPool(2×2)
        → Flatten → FC(64×8×8=4096 → 128) → ReLU → FC(128 → 10)

    L2 regularization is applied to all weights (not biases).
    Uses float32 by default for modern deep learning workflows.
    """

    def __init__(
        self,
        n_classes: int = 10,
        l2: float = 1e-4,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.l2 = float(l2)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.dtype = dtype

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.to(device=self.device, dtype=self.dtype)
        self.fit_result: Optional[FitResult] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _l2_penalty(self) -> torch.Tensor:
        """L2 penalty on all weight parameters (not biases)."""
        penalty = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for name, param in self.named_parameters():
            if 'weight' in name:
                penalty = penalty + (param * param).sum()
        return 0.5 * self.l2 * penalty

    def _reshape_for_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat features (N, 3072) to images (N, 3, 32, 32)."""
        if x.ndim == 2 and x.shape[1] == 3072:
            return x.view(-1, 3, 32, 32)
        elif x.ndim == 4 and x.shape[1:] == (3, 32, 32):
            return x
        else:
            raise ValueError(f"Expected shape (N, 3072) or (N, 3, 32, 32), got {x.shape}")

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        max_epochs: int = 50,
        batch_size: int = 128,
        lr: float = 0.01,
        momentum: float = 0.9,
        verbose: bool = False,
    ) -> FitResult:
        """Train with SGD + momentum."""
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        x = self._reshape_for_cnn(x)

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if torch.any((y < 0) | (y >= self.n_classes)):
            raise ValueError("labels are outside the configured class range")

        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        self.train()
        total_iterations = 0
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.forward(batch_x)
                loss = F.cross_entropy(logits, batch_y) + self._l2_penalty()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total_iterations += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss/len(loader):.4f}")

        self.eval()
        with torch.no_grad():
            final_logits = self.forward(x)
            final_loss = F.cross_entropy(final_logits, y) + self._l2_penalty()
            grad_norms = []
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
            gradient_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        self.fit_result = FitResult(
            success=True,
            message="SGD training completed",
            iterations=total_iterations,
            objective=float(final_loss.cpu()),
            gradient_norm=gradient_norm,
        )
        return self.fit_result

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for input."""
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        x = self._reshape_for_cnn(x)
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x).argmax(dim=1)

    def accuracy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        return float((self.predict(x) == y).double().mean().cpu())

    def per_example_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-example unregularized cross-entropy loss."""
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        x = self._reshape_for_cnn(x)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.cross_entropy(logits, y, reduction="none")
