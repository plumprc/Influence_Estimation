"""Torch behavior measures and counterfactual influence calculations."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import MultinomialLogisticRegression


BEHAVIORS = ("negative_loss", "target_logit", "soft_margin", "hard_margin")


def behavior_values(model: MultinomialLogisticRegression, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return query-side behavior values, all oriented higher-is-better."""

    y = torch.as_tensor(y, dtype=torch.long, device=model.device)
    logits = model.logits(x)
    negative_loss = -F.cross_entropy(logits, y, reduction="none")
    target_logit = logits.gather(1, y[:, None]).squeeze(1)
    mask = F.one_hot(y, num_classes=logits.shape[1]).bool()
    soft_margin = target_logit - torch.logsumexp(logits.masked_fill(mask, float("-inf")), dim=1)
    hard_margin = target_logit - logits.masked_fill(mask, float("-inf")).max(dim=1).values
    return {
        "negative_loss": negative_loss,
        "target_logit": target_logit,
        "soft_margin": soft_margin,
        "hard_margin": hard_margin,
    }


def _behavior_logit_gradient(logits: torch.Tensor, target: torch.Tensor, behavior: str) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    signal = torch.zeros_like(logits)
    if behavior == "negative_loss":
        signal = -probabilities
        signal[target] += 1.0
    elif behavior == "target_logit":
        signal[target] = 1.0
    elif behavior == "soft_margin":
        signal[target] = 1.0
        other_mask = torch.ones_like(logits, dtype=torch.bool)
        other_mask[target] = False
        signal[other_mask] = -torch.softmax(logits[other_mask], dim=0)
    elif behavior == "hard_margin":
        signal[target] = 1.0
        other_mask = torch.ones_like(logits, dtype=torch.bool)
        other_mask[target] = False
        second_best = logits[other_mask].argmax()
        signal[other_mask.nonzero(as_tuple=True)[0][second_best]] = -1.0
    else:
        raise ValueError(f"Unknown behavior {behavior}; choose from {BEHAVIORS}")
    return signal


def one_step_influences(
    model: MultinomialLogisticRegression,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    *,
    candidate_indices: Optional[Iterable[int]] = None,
    step_size: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact finite one-step effects and first-order approximations.

    Returned tensors have shape ``(n_queries, n_candidates, 3)`` and follow
    ``BEHAVIORS``. Candidate and query calculations are vectorized.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    train_x = torch.as_tensor(train_x, dtype=model.dtype, device=model.device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=model.device)
    query_x = torch.as_tensor(query_x, dtype=model.dtype, device=model.device)
    query_y = torch.as_tensor(query_y, dtype=torch.long, device=model.device)
    if candidate_indices is None:
        candidates = torch.arange(train_x.shape[0], dtype=torch.long, device=model.device)
    else:
        candidates = torch.as_tensor(list(candidate_indices), dtype=torch.long, device=model.device)
    if torch.any((candidates < 0) | (candidates >= train_x.shape[0])):
        raise IndexError("candidate_indices contains an invalid training index")

    baseline = behavior_values(model, query_x, query_y)
    baseline_logits = model.logits(query_x)
    candidate_x, candidate_y = train_x[candidates], train_y[candidates]
    residual = torch.softmax(model.logits(candidate_x), dim=1).clone()
    residual[torch.arange(candidates.numel(), device=model.device), candidate_y] -= 1.0
    similarity = query_x @ candidate_x.T
    delta_logits = -step_size * (similarity[:, :, None] * residual[None, :, :] + residual[None, :, :])
    updated_logits = baseline_logits[:, None, :] + delta_logits
    target = query_y[:, None].expand(-1, candidates.numel())
    target_logits = updated_logits.gather(2, target[:, :, None]).squeeze(2)
    updated_negative_loss = F.log_softmax(updated_logits, dim=2).gather(2, target[:, :, None]).squeeze(2)
    target_mask = F.one_hot(target, num_classes=updated_logits.shape[2]).bool()
    updated_soft_margin = target_logits - torch.logsumexp(
        updated_logits.masked_fill(target_mask, float("-inf")), dim=2
    )
    updated_hard_margin = target_logits - updated_logits.masked_fill(target_mask, float("-inf")).max(dim=2).values
    exact = torch.stack(
        [
            updated_negative_loss - baseline["negative_loss"][:, None],
            target_logits - baseline["target_logit"][:, None],
            updated_soft_margin - baseline["soft_margin"][:, None],
            updated_hard_margin - baseline["hard_margin"][:, None],
        ], dim=2,
    )
    signal = torch.stack(
        [
            torch.stack(
                [_behavior_logit_gradient(baseline_logits[q], query_y[q], behavior) for behavior in BEHAVIORS],
                dim=0,
            )
            for q in range(query_x.shape[0])
        ], dim=0,
    )
    first_order = torch.einsum("qbc,qkc->qkb", signal, delta_logits)
    return exact, first_order, candidates


def multi_step_influences(
    model: MultinomialLogisticRegression,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    *,
    candidate_indices: Optional[Iterable[int]] = None,
    step_size: float = 0.1,
    num_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact finite multi-step effects and first-order approximations.

    Applies K consecutive SGD steps on the candidate's unregularized loss.
    Returns tensors with shape ``(n_queries, n_candidates, 3)`` following BEHAVIORS.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    train_x = torch.as_tensor(train_x, dtype=model.dtype, device=model.device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=model.device)
    query_x = torch.as_tensor(query_x, dtype=model.dtype, device=model.device)
    query_y = torch.as_tensor(query_y, dtype=torch.long, device=model.device)
    if candidate_indices is None:
        candidates = torch.arange(train_x.shape[0], dtype=torch.long, device=model.device)
    else:
        candidates = torch.as_tensor(list(candidate_indices), dtype=torch.long, device=model.device)

    baseline = behavior_values(model, query_x, query_y)
    exact = torch.zeros((query_x.shape[0], candidates.numel(), len(BEHAVIORS)), dtype=model.dtype, device=model.device)
    first_order = torch.zeros_like(exact)

    for position, candidate in enumerate(candidates.tolist()):
        counterfactual = model.copy()
        x_k, y_k = train_x[candidate].unsqueeze(0), train_y[candidate].unsqueeze(0)
        accumulated_delta = torch.zeros_like(model.weights)
        accumulated_delta_bias = torch.zeros_like(model.bias)

        for step in range(num_steps):
            grad_w, grad_b = counterfactual.example_gradient(x_k[0], int(y_k[0].item()))
            counterfactual.weights = counterfactual.weights - step_size * grad_w
            counterfactual.bias = counterfactual.bias - step_size * grad_b
            accumulated_delta = accumulated_delta - step_size * grad_w
            accumulated_delta_bias = accumulated_delta_bias - step_size * grad_b

        values = behavior_values(counterfactual, query_x, query_y)
        for behavior_position, behavior in enumerate(BEHAVIORS):
            exact[:, position, behavior_position] = values[behavior] - baseline[behavior]

        query_logits = model.logits(query_x)
        delta_logits = query_x @ accumulated_delta.T + accumulated_delta_bias[None, :]
        for q in range(query_x.shape[0]):
            for behavior_position, behavior in enumerate(BEHAVIORS):
                signal = _behavior_logit_gradient(query_logits[q], query_y[q], behavior)
                first_order[q, position, behavior_position] = (signal * delta_logits[q]).sum()

    return exact, first_order


def inverse_hessian_influences(
    model: MultinomialLogisticRegression,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    *,
    candidate_indices: Optional[Iterable[int]] = None,
    damping: float = 0.01,
) -> torch.Tensor:
    """Compute inverse-Hessian influence (influence-function style).

    Uses the inverse Hessian of the regularized training objective as the
    response operator. Returns tensor with shape ``(n_queries, n_candidates, 3)``
    following BEHAVIORS.
    """

    if damping < 0:
        raise ValueError("damping must be non-negative")
    train_x = torch.as_tensor(train_x, dtype=model.dtype, device=model.device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=model.device)
    query_x = torch.as_tensor(query_x, dtype=model.dtype, device=model.device)
    query_y = torch.as_tensor(query_y, dtype=torch.long, device=model.device)
    if candidate_indices is None:
        candidates = torch.arange(train_x.shape[0], dtype=torch.long, device=model.device)
    else:
        candidates = torch.as_tensor(list(candidate_indices), dtype=torch.long, device=model.device)

    n_features = train_x.shape[1]
    n_classes = model.n_classes
    n_params = n_classes * n_features + n_classes

    probs = model.probabilities(train_x)
    hessian = torch.zeros((n_params, n_params), dtype=model.dtype, device=model.device)

    for i in range(train_x.shape[0]):
        p = probs[i]
        x = train_x[i]
        P_diag = torch.diag(p)
        P_outer = torch.outer(p, p)
        H_logits = P_diag - P_outer

        J_w = torch.kron(torch.eye(n_classes, device=model.device, dtype=model.dtype), x.unsqueeze(0))
        J_b = torch.eye(n_classes, device=model.device, dtype=model.dtype)
        J = torch.cat([J_w, J_b], dim=1)

        hessian += J.T @ H_logits @ J / train_x.shape[0]

    weight_indices = n_classes * n_features
    hessian[:weight_indices, :weight_indices] += model.l2 * torch.eye(weight_indices, device=model.device, dtype=model.dtype)
    hessian += damping * torch.eye(n_params, device=model.device, dtype=model.dtype)

    try:
        inv_hessian = torch.linalg.inv(hessian)
    except torch.linalg.LinAlgError:
        raise RuntimeError("Hessian inversion failed; try increasing damping")

    output = torch.zeros((query_x.shape[0], candidates.numel(), len(BEHAVIORS)), dtype=model.dtype, device=model.device)
    query_logits = model.logits(query_x)

    for position, candidate in enumerate(candidates.tolist()):
        grad_w, grad_b = model.example_gradient(train_x[candidate], int(train_y[candidate].item()))
        grad_vec = torch.cat([grad_w.reshape(-1), grad_b])

        ihvp = inv_hessian @ grad_vec
        delta_w = -ihvp[:weight_indices].reshape(n_classes, n_features)
        delta_b = -ihvp[weight_indices:]

        delta_logits = query_x @ delta_w.T + delta_b[None, :]

        for q in range(query_x.shape[0]):
            for behavior_position, behavior in enumerate(BEHAVIORS):
                signal = _behavior_logit_gradient(query_logits[q], query_y[q], behavior)
                output[q, position, behavior_position] = (signal * delta_logits[q]).sum()

    return output


def feature_similarity_scores(
    query_x: torch.Tensor,
    candidate_x: torch.Tensor,
) -> torch.Tensor:
    """Feature-similarity estimator: inner products in input feature space.

    Under the standardized auxiliary update of the similarity specification,
    this is (up to the common positive scale eta) the exact counterfactual
    behavior change for the query readout. Returns ``(n_queries, n_candidates)``.
    """

    if query_x.ndim != 2 or candidate_x.ndim != 2 or query_x.shape[1] != candidate_x.shape[1]:
        raise ValueError("query_x and candidate_x must be 2-D with matching feature dimension")
    return query_x @ candidate_x.T


def _full_hessian(model: MultinomialLogisticRegression, train_x: torch.Tensor) -> torch.Tensor:
    """Full Hessian of the regularized mean training objective at the current fit.

    No damping term is added, so the result is the exact Newton/chord matrix
    used by the Experiment 4 reoptimization reference.
    """

    n_features = train_x.shape[1]
    n_classes = model.n_classes
    n_params = n_classes * n_features + n_classes
    probs = model.probabilities(train_x)
    hessian = torch.zeros((n_params, n_params), dtype=model.dtype, device=model.device)
    for i in range(train_x.shape[0]):
        p, x = probs[i], train_x[i]
        h_logits = torch.diag(p) - torch.outer(p, p)
        j_w = torch.kron(torch.eye(n_classes, device=model.device, dtype=model.dtype), x.unsqueeze(0))
        j = torch.cat([j_w, torch.eye(n_classes, device=model.device, dtype=model.dtype)], dim=1)
        hessian += j.T @ h_logits @ j / train_x.shape[0]
    weight_size = n_classes * n_features
    hessian[:weight_size, :weight_size] += model.l2 * torch.eye(weight_size, device=model.device, dtype=model.dtype)
    return hessian


def _stable_cholesky(hessian: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
    """Cholesky factorization with adaptive diagonal jitter.

    The full parameter Hessian is positive-definite in exact arithmetic (the
    bias block is unregularized, so it can lose definiteness by a hair in
    floating point). When the plain factorization fails we retry with the
    smallest diagonal damping that succeeds, scaled to the Hessian's own
    magnitude. On a well-conditioned Hessian jitter is never applied, so
    seeds that already factor cleanly are numerically unchanged.
    """

    try:
        return torch.linalg.cholesky(hessian)
    except torch._C._LinAlgError:
        pass
    eye = torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)
    base = float(torch.diagonal(hessian).abs().mean()) or 1.0
    for attempt in range(max_tries):
        jitter = base * (10.0 ** (attempt - max_tries + 1))
        try:
            return torch.linalg.cholesky(hessian + jitter * eye)
        except torch._C._LinAlgError:
            continue
    raise torch._C._LinAlgError(
        f"cholesky failed even after {max_tries} jitter steps (base={base:.3e})"
    )


def _upweight_hessian_factor(
    model: MultinomialLogisticRegression,
    parameters: torch.Tensor,
    train_x: torch.Tensor,
    candidate_index: int,
) -> torch.Tensor:
    """Return B with B.T @ B equal to one example's parameter Hessian.

    The logistic-loss Hessian with respect to logits has rank at most C-1.
    Its nonzero eigenpairs therefore give a compact representation of the
    upweight term, avoiding a full-rank Hessian update for every candidate.
    """

    n_features = train_x.shape[1]
    n_classes = model.n_classes
    weight_size = n_classes * n_features
    weights, bias = model._unpack(parameters, n_features)
    x = train_x[candidate_index]
    logits = x @ weights.T + bias
    probabilities = torch.softmax(logits, dim=0)
    hessian_logits = torch.diag(probabilities) - torch.outer(
        probabilities, probabilities
    )
    # cuSOLVER's tiny symmetric eigensolver occasionally crashed after many
    # invocations on V100/CUDA 11.8. The matrix is only C x C, so computing it
    # on the CPU is inexpensive and keeps the GPU path deterministic.
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian_logits.detach().cpu())
    threshold = max(float(eigenvalues.max()) * 1e-14, 1e-16)
    keep = eigenvalues > threshold
    if not bool(keep.any()):
        # A saturated softmax distribution can have an exactly-zero rank-(C-1)
        # Hessian in floating point. The upweighted objective then contributes
        # only a linear term, so the base Hessian solve is the correct update.
        return torch.empty(
            (0, weight_size + n_classes), dtype=model.dtype, device=model.device
        )

    eigenvectors = eigenvectors.to(device=model.device)
    coefficients = (
        torch.sqrt(eigenvalues[keep].to(device=model.device)).unsqueeze(1)
        * eigenvectors[:, keep].T
    )
    factor = torch.empty(
        (coefficients.shape[0], weight_size + n_classes),
        dtype=model.dtype,
        device=model.device,
    )
    for row, coefficient in enumerate(coefficients):
        factor[row, :weight_size] = torch.outer(coefficient, x).reshape(-1)
        factor[row, weight_size:] = coefficient
    return factor


def _solve_updated_upweight_newton_step(
    gradient: torch.Tensor,
    factor: torch.Tensor,
    base_cholesky: torch.Tensor,
    extra_weight: float,
) -> torch.Tensor:
    """Solve (H0 + alpha B.T B) step with the Woodbury identity."""

    if factor.shape[0] == 0:
        return torch.cholesky_solve(
            gradient.unsqueeze(1), base_cholesky
        ).squeeze(1)

    inverse_gradient = torch.cholesky_solve(
        gradient.unsqueeze(1), base_cholesky
    ).squeeze(1)
    inverse_factor_transpose = torch.cholesky_solve(factor.T, base_cholesky)
    small_matrix = factor @ inverse_factor_transpose
    small_matrix += torch.eye(
        small_matrix.shape[0], dtype=small_matrix.dtype, device=small_matrix.device
    ) / extra_weight
    correction = torch.linalg.solve(
        small_matrix, factor @ inverse_gradient
    )
    return inverse_gradient - inverse_factor_transpose @ correction


def newton_refine_factual(
    model: MultinomialLogisticRegression,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> dict:
    """Refine the factual fit IN PLACE with fixed-Hessian (chord) Newton steps.

    Removes L-BFGS convergence noise so that reoptimization references are
    noise-free (Experiment 2b protocol). Returns a diagnostics dict whose
    ``cholesky`` entry can be reused by :func:`newton_reopt_influences`.
    """

    train_x = torch.as_tensor(train_x, dtype=model.dtype, device=model.device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=model.device)
    n_features = train_x.shape[1]
    hessian = _full_hessian(model, train_x)
    chol = _stable_cholesky(hessian)
    theta = model._pack().clone()
    iterations = 0
    for iterations in range(1, max_iter + 1):
        _, grad = model._objective_and_gradient(theta, train_x, train_y)
        if float(grad.norm()) < tol:
            break
        theta = theta - torch.cholesky_solve(grad.unsqueeze(1), chol).squeeze(1)
    model.weights, model.bias = model._unpack(theta, n_features)
    model.weights, model.bias = model.weights.detach().clone(), model.bias.detach().clone()
    _, grad = model._objective_and_gradient(model._pack(), train_x, train_y)
    return {
        "cholesky": chol,
        "gradient_norm": float(grad.norm()),
        "iterations": iterations,
    }


def newton_reopt_influences(
    model: MultinomialLogisticRegression,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    *,
    candidate_indices: Iterable[int],
    epsilon: float,
    cholesky: Optional[torch.Tensor] = None,
    tol: float = 1e-12,
    max_iter: int = 200,
    update_perturbation_hessian: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Reoptimization reference with chord or updated-upweight Newton solves.

    For each candidate, reoptimizes the epsilon-upweighted objective to near
    machine precision and records the behavior changes at the queries. The
    default chord mode reuses the factual Hessian; the updated mode also
    refreshes the rank-(C-1) Hessian of the upweighted example at every step.
    The model is expected to already sit at a refined factual optimum (call
    :func:`newton_refine_factual` first); it is not modified here. Returns a
    ``(n_queries, n_candidates, len(BEHAVIORS))`` tensor plus diagnostics.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    train_x = torch.as_tensor(train_x, dtype=model.dtype, device=model.device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=model.device)
    query_x = torch.as_tensor(query_x, dtype=model.dtype, device=model.device)
    query_y = torch.as_tensor(query_y, dtype=torch.long, device=model.device)
    candidates = torch.as_tensor(list(candidate_indices), dtype=torch.long, device=model.device)
    n_features = train_x.shape[1]
    if cholesky is None:
        cholesky = _stable_cholesky(_full_hessian(model, train_x))

    theta_factual = model._pack().clone()
    _, grad0 = model._objective_and_gradient(theta_factual, train_x, train_y)
    baseline = behavior_values(model, query_x, query_y)
    output = torch.empty((query_x.shape[0], candidates.numel(), len(BEHAVIORS)), dtype=model.dtype, device=model.device)
    residual_norms = []
    iteration_counts = []
    worst_candidate_index = -1
    worst_residual_norm = float("-inf")
    counterfactual = model.copy()
    for position, candidate in enumerate(candidates.tolist()):
        theta = theta_factual.clone()
        converged_norm = float("nan")
        iterations = 0
        for iterations in range(1, max_iter + 1):
            objective, grad = model._objective_and_gradient(
                theta, train_x, train_y, extra_index=candidate, extra_weight=epsilon,
            )
            converged_norm = float(grad.norm())
            if converged_norm < tol:
                break
            if update_perturbation_hessian:
                factor = _upweight_hessian_factor(model, theta, train_x, candidate)
                step = _solve_updated_upweight_newton_step(
                    grad, factor, cholesky, epsilon
                )
            else:
                step = torch.cholesky_solve(grad.unsqueeze(1), cholesky).squeeze(1)
            directional_derivative = -torch.dot(grad, step)
            if float(directional_derivative) >= 0.0:
                raise RuntimeError("Newton step is not a descent direction")
            step_size = 1.0
            while step_size >= 1e-4:
                trial = theta - step_size * step
                trial_objective, _ = model._objective_and_gradient(
                    trial, train_x, train_y, extra_index=candidate, extra_weight=epsilon,
                )
                if float(trial_objective) <= float(objective) + 1e-4 * step_size * float(directional_derivative):
                    break
                step_size *= 0.5
            theta = theta - step_size * step
        residual_norms.append(converged_norm)
        iteration_counts.append(iterations)
        if converged_norm > worst_residual_norm:
            worst_candidate_index = candidate
            worst_residual_norm = converged_norm
        counterfactual.weights, counterfactual.bias = counterfactual._unpack(theta, n_features)
        values = behavior_values(counterfactual, query_x, query_y)
        for behavior_position, behavior in enumerate(BEHAVIORS):
            output[:, position, behavior_position] = values[behavior] - baseline[behavior]
    diagnostics = {
        "factual_gradient_norm": float(grad0.norm()),
        "max_candidate_gradient_norm": max(residual_norms),
        "mean_candidate_gradient_norm": float(sum(residual_norms) / len(residual_norms)),
        "max_candidate_iterations": max(iteration_counts),
        "mean_candidate_iterations": float(
            sum(iteration_counts) / len(iteration_counts)
        ),
        "worst_candidate_index": worst_candidate_index,
    }
    return output, diagnostics


def reoptimized_upweight_influences(
    model: MultinomialLogisticRegression,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    *,
    candidate_indices: Iterable[int],
    epsilon: float,
    max_iter: int = 100,
    tol: float = 1e-9,
    gtol: float = 1e-6,
) -> torch.Tensor:
    """Measure behavior changes after reoptimizing with epsilon * loss(z_k)."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    candidates = torch.as_tensor(list(candidate_indices), dtype=torch.long, device=model.device)
    baseline = behavior_values(model, query_x, query_y)
    output = torch.empty((query_x.shape[0], candidates.numel(), len(BEHAVIORS)), dtype=model.dtype, device=model.device)
    initial = model._pack()
    for position, candidate in enumerate(candidates.tolist()):
        counterfactual = model.copy()
        result = counterfactual.fit(
            train_x, train_y, max_iter=max_iter, tol=tol, gtol=gtol,
            initial_parameters=initial, extra_index=candidate, extra_weight=epsilon,
        )
        if not torch.isfinite(torch.tensor(result.objective, device=model.device)):
            raise FloatingPointError(f"Counterfactual fit produced a non-finite objective for candidate {candidate}")
        values = behavior_values(counterfactual, query_x, query_y)
        for behavior_position, behavior in enumerate(BEHAVIORS):
            output[:, position, behavior_position] = values[behavior] - baseline[behavior]
    return output


def cnn_behavior_values(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute behavior values for CNN models."""
    y = torch.as_tensor(y, dtype=torch.long, device=next(model.parameters()).device)
    x = torch.as_tensor(x, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)

    if x.ndim == 2 and x.shape[1] == 3072:
        x = x.view(-1, 3, 32, 32)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        negative_loss = -F.cross_entropy(logits, y, reduction="none")
        target_logit = logits.gather(1, y[:, None]).squeeze(1)
        mask = F.one_hot(y, num_classes=logits.shape[1]).bool()
        soft_margin = target_logit - torch.logsumexp(logits.masked_fill(mask, float("-inf")), dim=1)
        hard_margin = target_logit - logits.masked_fill(mask, float("-inf")).max(dim=1).values

    return {
        "negative_loss": negative_loss,
        "target_logit": target_logit,
        "soft_margin": soft_margin,
        "hard_margin": hard_margin,
    }


def cnn_one_step_influences(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    *,
    candidate_indices: Optional[Iterable[int]] = None,
    step_size: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact one-step effects for CNN models using autograd.

    Returns:
        exact: (n_queries, n_candidates, 4) exact behavior changes
        first_order: (n_queries, n_candidates, 4) first-order approximations
        candidates: candidate indices used
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    train_x = torch.as_tensor(train_x, dtype=dtype, device=device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    query_x = torch.as_tensor(query_x, dtype=dtype, device=device)
    query_y = torch.as_tensor(query_y, dtype=torch.long, device=device)

    if train_x.ndim == 2 and train_x.shape[1] == 3072:
        train_x = train_x.view(-1, 3, 32, 32)
    if query_x.ndim == 2 and query_x.shape[1] == 3072:
        query_x = query_x.view(-1, 3, 32, 32)

    if candidate_indices is None:
        candidates = torch.arange(train_x.shape[0], dtype=torch.long, device=device)
    else:
        candidates = torch.as_tensor(list(candidate_indices), dtype=torch.long, device=device)

    baseline = cnn_behavior_values(model, query_x, query_y)

    n_queries = query_x.shape[0]
    n_candidates = candidates.numel()
    exact = torch.zeros((n_queries, n_candidates, 4), dtype=dtype, device=device)
    first_order = torch.zeros((n_queries, n_candidates, 4), dtype=dtype, device=device)

    # Save original parameters
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}

    # Compute exact effects
    for cand_idx, candidate in enumerate(candidates.tolist()):
        x_c = train_x[candidate:candidate+1]
        y_c = train_y[candidate:candidate+1]

        model.zero_grad()
        model.train()
        logits_c = model(x_c)
        loss_c = F.cross_entropy(logits_c, y_c)
        loss_c.backward()

        # Apply one-step update
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.data = param.data - step_size * param.grad

        # Measure behavior changes
        model.eval()
        updated_values = cnn_behavior_values(model, query_x, query_y)
        for behavior_idx, behavior in enumerate(BEHAVIORS):
            exact[:, cand_idx, behavior_idx] = updated_values[behavior] - baseline[behavior]

        # Restore parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = original_params[name].clone()

    # Compute first-order approximation using gradient dot products
    # For each query and behavior, compute grad_B
    # For each candidate, compute grad_loss
    # first_order = -step_size * <grad_B, grad_loss>

    for q_idx in range(n_queries):
        for behavior_idx, behavior in enumerate(BEHAVIORS):
            model.zero_grad()
            model.train()
            query_logits = model(query_x[q_idx:q_idx+1])

            # Compute behavior value
            if behavior == "negative_loss":
                b_val = -F.cross_entropy(query_logits, query_y[q_idx:q_idx+1])
            elif behavior == "target_logit":
                b_val = query_logits[0, query_y[q_idx]]
            elif behavior == "soft_margin":
                target_logit = query_logits[0, query_y[q_idx]]
                mask = torch.ones_like(query_logits[0], dtype=torch.bool)
                mask[query_y[q_idx]] = False
                b_val = target_logit - torch.logsumexp(query_logits[0, mask], dim=0)
            elif behavior == "hard_margin":
                target_logit = query_logits[0, query_y[q_idx]]
                mask = torch.ones_like(query_logits[0], dtype=torch.bool)
                mask[query_y[q_idx]] = False
                b_val = target_logit - query_logits[0, mask].max()

            b_val.backward()

            # Store query gradients
            query_grads = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    query_grads.append(param.grad.clone().flatten())
                else:
                    query_grads.append(torch.zeros_like(param).flatten())
            query_grad_flat = torch.cat(query_grads)

            # For each candidate, compute gradient of loss and dot product
            for cand_idx, candidate in enumerate(candidates.tolist()):
                model.zero_grad()
                x_c = train_x[candidate:candidate+1]
                y_c = train_y[candidate:candidate+1]
                logits_c = model(x_c)
                loss_c = F.cross_entropy(logits_c, y_c)
                loss_c.backward()

                cand_grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        cand_grads.append(param.grad.clone().flatten())
                    else:
                        cand_grads.append(torch.zeros_like(param).flatten())
                cand_grad_flat = torch.cat(cand_grads)

                # first_order = -step_size * dot(grad_B, grad_loss)
                first_order[q_idx, cand_idx, behavior_idx] = -step_size * torch.dot(query_grad_flat, cand_grad_flat).item()

    # Restore model to eval mode and original parameters
    for name, param in model.named_parameters():
        param.data = original_params[name]
    model.eval()

    return exact, first_order, candidates


def cnn_multi_step_influences(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    *,
    candidate_indices: Optional[Iterable[int]] = None,
    step_size: float = 0.1,
    num_steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact finite K-step effects for a CNN.

    Each candidate receives ``num_steps`` consecutive full-gradient updates on
    its unregularized example loss, starting from the factual parameters.
    Returns ``(n_queries, n_candidates, len(BEHAVIORS))`` behavior changes and
    the candidate indices.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    train_x = torch.as_tensor(train_x, dtype=dtype, device=device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    query_x = torch.as_tensor(query_x, dtype=dtype, device=device)
    query_y = torch.as_tensor(query_y, dtype=torch.long, device=device)
    if train_x.ndim == 2 and train_x.shape[1] == 3072:
        train_x = train_x.view(-1, 3, 32, 32)
    if query_x.ndim == 2 and query_x.shape[1] == 3072:
        query_x = query_x.view(-1, 3, 32, 32)

    if candidate_indices is None:
        candidates = torch.arange(train_x.shape[0], dtype=torch.long, device=device)
    else:
        candidates = torch.as_tensor(list(candidate_indices), dtype=torch.long, device=device)

    baseline = cnn_behavior_values(model, query_x, query_y)
    output = torch.empty(
        (query_x.shape[0], candidates.numel(), len(BEHAVIORS)),
        dtype=dtype,
        device=device,
    )
    original_params = {name: parameter.data.clone() for name, parameter in model.named_parameters()}

    try:
        for candidate_position, candidate in enumerate(candidates.tolist()):
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.data.copy_(original_params[name])

            x_candidate = train_x[candidate : candidate + 1]
            y_candidate = train_y[candidate : candidate + 1]
            for _ in range(num_steps):
                model.zero_grad(set_to_none=True)
                model.train()
                loss = F.cross_entropy(model(x_candidate), y_candidate)
                loss.backward()
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.data.sub_(step_size * parameter.grad)

            values = cnn_behavior_values(model, query_x, query_y)
            for behavior_position, behavior in enumerate(BEHAVIORS):
                output[:, candidate_position, behavior_position] = values[behavior] - baseline[behavior]
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.data.copy_(original_params[name])
        model.eval()

    return output, candidates


def cnn_local_reopt_upweight_influences(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    *,
    candidate_indices: Iterable[int],
    alpha: float,
    l2: float,
    max_iter: int = 5,
    method: str = "lbfgs",
    lr: float = 1.0,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Measure behavior changes after local full-batch reoptimization.

    The perturbed objective is the full-batch regularized training objective
    plus ``alpha`` times the selected candidate loss. Optimization starts from
    the factual parameters. In a non-convex CNN this is a local reoptimization
    reference, not a global retraining solution.
    """

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if method not in ("lbfgs", "adam"):
        raise ValueError("method must be 'lbfgs' or 'adam'")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    train_x = torch.as_tensor(train_x, dtype=dtype, device=device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    query_x = torch.as_tensor(query_x, dtype=dtype, device=device)
    query_y = torch.as_tensor(query_y, dtype=torch.long, device=device)
    if train_x.ndim == 2 and train_x.shape[1] == 3072:
        train_x = train_x.view(-1, 3, 32, 32)
    if query_x.ndim == 2 and query_x.shape[1] == 3072:
        query_x = query_x.view(-1, 3, 32, 32)

    candidates = torch.as_tensor(list(candidate_indices), dtype=torch.long, device=device)
    baseline = cnn_behavior_values(model, query_x, query_y)
    output = torch.empty(
        (query_x.shape[0], candidates.numel(), len(BEHAVIORS)),
        dtype=dtype,
        device=device,
    )
    original_params = {name: parameter.data.clone() for name, parameter in model.named_parameters()}

    def _l2_penalty() -> torch.Tensor:
        penalty = torch.tensor(0.0, dtype=dtype, device=device)
        for name, parameter in model.named_parameters():
            if "weight" in name:
                penalty = penalty + 0.5 * l2 * torch.sum(parameter * parameter)
        return penalty

    closure_calls_per_candidate = []
    final_objectives = []
    final_gradient_norms = []

    try:
        for candidate_position, candidate in enumerate(candidates.tolist()):
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.data.copy_(original_params[name])

            if method == "lbfgs":
                optimizer = torch.optim.LBFGS(
                    model.parameters(),
                    lr=lr,
                    max_iter=max_iter,
                    tolerance_grad=1e-5,
                    tolerance_change=1e-9,
                    line_search_fn="strong_wolfe",
                )
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            x_candidate = train_x[candidate : candidate + 1]
            y_candidate = train_y[candidate : candidate + 1]

            def closure() -> torch.Tensor:
                nonlocal closure_calls
                optimizer.zero_grad(set_to_none=True)
                model.train()
                logits = model(train_x)
                objective = F.cross_entropy(logits, train_y) + _l2_penalty()
                objective = objective + alpha * F.cross_entropy(
                    model(x_candidate), y_candidate
                )
                objective.backward()
                closure_calls += 1
                return objective

            closure_calls = 0
            if method == "lbfgs":
                optimizer.step(closure)
            else:
                for _ in range(max_iter):
                    optimizer.step(closure)

            final_objective = float(closure().detach().cpu())
            final_gradient_norm = float(
                torch.sqrt(
                    sum(
                        torch.sum(parameter.grad * parameter.grad)
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    )
                ).detach().cpu()
            )
            closure_calls_per_candidate.append(closure_calls)
            final_objectives.append(final_objective)
            final_gradient_norms.append(final_gradient_norm)
            values = cnn_behavior_values(model, query_x, query_y)
            for behavior_position, behavior in enumerate(BEHAVIORS):
                output[:, candidate_position, behavior_position] = values[behavior] - baseline[behavior]
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.data.copy_(original_params[name])
        model.eval()

    diagnostics = {
        "closure_calls": closure_calls_per_candidate,
        "final_objective": final_objectives,
        "final_gradient_norm": final_gradient_norms,
        "mean_final_gradient_norm": float(np.mean(final_gradient_norms)),
        "max_final_gradient_norm": float(np.max(final_gradient_norms)),
    }
    return output, diagnostics, candidates


def cnn_ihvp_influences(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    *,
    candidate_indices: Optional[Iterable[int]] = None,
    damping: float = 1e-2,
    max_cg_iters: int = 10,
    cg_tol: float = 1e-5,
    behaviors: Optional[Iterable[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute IF-style influences using IHVP (conjugate gradient).

    The Hessian is taken with respect to the regularized training objective
    used by ``SimpleCNN.fit``; damping is added on top of that Hessian.

    Returns:
        influences: (n_queries, n_candidates, len(behaviors)) IF influence estimates
        candidates: candidate indices used
    """
    behavior_list = list(BEHAVIORS) if behaviors is None else list(behaviors)
    if not behavior_list:
        raise ValueError("At least one behavior is required")
    unknown = [behavior for behavior in behavior_list if behavior not in BEHAVIORS]
    if unknown:
        raise ValueError(f"Unknown behaviors: {unknown}")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    train_x = torch.as_tensor(train_x, dtype=dtype, device=device)
    train_y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    query_x = torch.as_tensor(query_x, dtype=dtype, device=device)
    query_y = torch.as_tensor(query_y, dtype=torch.long, device=device)

    if train_x.ndim == 2 and train_x.shape[1] == 3072:
        train_x = train_x.view(-1, 3, 32, 32)
    if query_x.ndim == 2 and query_x.shape[1] == 3072:
        query_x = query_x.view(-1, 3, 32, 32)

    if candidate_indices is None:
        candidates = torch.arange(train_x.shape[0], dtype=torch.long, device=device)
    else:
        candidates = torch.as_tensor(list(candidate_indices), dtype=torch.long, device=device)

    n_queries = query_x.shape[0]
    n_candidates = candidates.numel()
    influences = torch.zeros(
        (n_queries, n_candidates, len(behavior_list)),
        dtype=dtype,
        device=device,
    )
    l2 = float(getattr(model, "l2", 0.0))

    # Precompute candidate gradients (reuse across queries)
    candidate_grads = []
    for candidate in candidates.tolist():
        model.zero_grad()
        model.train()
        x_c = train_x[candidate:candidate+1]
        y_c = train_y[candidate:candidate+1]
        logits_c = model(x_c)
        loss_c = F.cross_entropy(logits_c, y_c)
        loss_c.backward()

        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.clone().flatten())
        candidate_grads.append(torch.cat(grad_list))

    # Helper: Hessian-vector product
    def hvp(v: torch.Tensor) -> torch.Tensor:
        """Compute (H + damping*I)*v for the regularized training objective."""
        model.zero_grad()
        model.train()

        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        if l2 != 0.0:
            for name, parameter in model.named_parameters():
                if "weight" in name:
                    loss = loss + 0.5 * l2 * torch.sum(parameter * parameter)

        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.flatten() for g in grads])

        gvp = torch.dot(flat_grads, v)
        hvp_grads = torch.autograd.grad(gvp, model.parameters())
        hvp_flat = torch.cat([g.flatten() for g in hvp_grads])

        return hvp_flat + damping * v

    # Conjugate gradient solver
    def cg_solve(b: torch.Tensor) -> torch.Tensor:
        """Solve (H + λI)x = b using CG."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)

        for _ in range(max_cg_iters):
            Ap = hvp(p)
            alpha = rs_old / (torch.dot(p, Ap) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)

            if torch.sqrt(rs_new) < cg_tol:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    # For each query and behavior, compute IF influence
    for q_idx in range(n_queries):
        for behavior_idx, behavior in enumerate(behavior_list):
            model.zero_grad()
            model.train()

            query_logits = model(query_x[q_idx:q_idx+1])

            if behavior == "negative_loss":
                b_val = -F.cross_entropy(query_logits, query_y[q_idx:q_idx+1])
            elif behavior == "target_logit":
                b_val = query_logits[0, query_y[q_idx]]
            elif behavior == "soft_margin":
                target_logit = query_logits[0, query_y[q_idx]]
                mask = torch.ones_like(query_logits[0], dtype=torch.bool)
                mask[query_y[q_idx]] = False
                b_val = target_logit - torch.logsumexp(query_logits[0, mask], dim=0)
            elif behavior == "hard_margin":
                target_logit = query_logits[0, query_y[q_idx]]
                mask = torch.ones_like(query_logits[0], dtype=torch.bool)
                mask[query_y[q_idx]] = False
                b_val = target_logit - query_logits[0, mask].max()

            b_val.backward()

            query_grad_flat = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

            # Solve (H + λI)^{-1} * query_grad
            ihvp = cg_solve(query_grad_flat)

            # Compute IF influence for all candidates
            for cand_idx, cand_grad in enumerate(candidate_grads):
                influences[q_idx, cand_idx, behavior_idx] = -torch.dot(ihvp, cand_grad).item()

    model.eval()
    return influences, candidates
