"""Behavior readouts, LoRA gradients, and online GradSim scoring."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn

from .data import Example, parse_short_answer
from .tokenization import (
    answer_token_id_for,
    collate_examples,
    encode_example,
    EncodedBatch,
    render_prompt,
)


BEHAVIOR_NAMES = (
    "negative_response_loss",
    "negative_explanation_loss",
    "answer_logit",
    "answer_margin",
)


def lora_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    }
    if not selected:
        raise ValueError("No trainable LoRA parameters were found")
    return selected


def parameter_dimension(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in lora_parameters(model).values())


def behavior_values(
    logits: torch.Tensor,
    batch: EncodedBatch,
    *,
    alternative_token_ids: list[list[int]],
) -> dict[str, torch.Tensor]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = batch.labels[:, 1:]
    response_mask = shifted_labels.ne(-100)
    if not bool(response_mask.any()):
        raise ValueError("Batch contains no response tokens")

    response_nll = nn.functional.cross_entropy(
        shifted_logits.transpose(1, 2).float(),
        shifted_labels,
        reduction="none",
    )
    token_counts = response_mask.sum(dim=1).clamp(min=1)
    response_nll_sum = (response_nll * response_mask).sum(dim=1)
    negative_response_loss = -response_nll_sum / token_counts

    explanation_mask = torch.zeros_like(response_mask)
    for row, positions in enumerate(batch.explanation_positions):
        for position in positions:
            prediction_index = position - 1
            if prediction_index < 0:
                raise ValueError("Explanation tokens cannot start at sequence index 0")
            explanation_mask[row, prediction_index] = 1
    explanation_count = explanation_mask.sum(dim=1).clamp(min=1)
    explanation_nll_sum = (response_nll * explanation_mask).sum(dim=1)
    negative_explanation_loss = -explanation_nll_sum / explanation_count

    answer_logits: list[torch.Tensor] = []
    answer_margins: list[torch.Tensor] = []
    for row, positions in enumerate(batch.answer_positions):
        if len(positions) != 1:
            raise ValueError("Expected exactly one answer-token position")
        prediction_index = positions[0] - 1
        if prediction_index < 0:
            raise ValueError("Answer token cannot be the first sequence token")
        token_log_probs = torch.log_softmax(
            shifted_logits[row, prediction_index].float(), dim=-1
        )
        target_id = batch.answer_token_ids[row]
        answer_logit = token_log_probs[target_id]
        alternatives = alternative_token_ids[row]
        if not alternatives:
            raise ValueError("No alternative answer token ids were provided")
        alternative_logits = token_log_probs[torch.tensor(alternatives, device=logits.device)]
        answer_margin = answer_logit - alternative_logits.max()
        answer_logits.append(answer_logit)
        answer_margins.append(answer_margin)

    return {
        "negative_response_loss": negative_response_loss.float(),
        "negative_explanation_loss": negative_explanation_loss.float(),
        "answer_logit": torch.stack(answer_logits),
        "answer_margin": torch.stack(answer_margins),
    }


def _alternative_token_ids(
    tokenizer,
    examples: list[Example],
) -> list[list[int]]:
    for example in examples:
        if len(set(example.choices)) < 2:
            raise ValueError(
                "Every validation example needs at least two answer choices "
                "for answer-margin behavior"
            )
    return [
        [
            answer_token_id_for(tokenizer, example.prompt, answer)
            for answer in sorted(example.choices)
            if answer != parse_short_answer(example.response)
        ]
        for example in examples
    ]


def _encode_batch(
    model,
    tokenizer,
    examples: list[Example],
    *,
    max_length: int,
    device: torch.device,
) -> EncodedBatch:
    encoded = [encode_example(tokenizer, example, max_length=max_length) for example in examples]
    batch = collate_examples(encoded, pad_token_id=tokenizer.pad_token_id)
    return batch.to(device)


def compute_behavior_gradients(
    model: nn.Module,
    tokenizer,
    examples: list[Example],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    selected = lora_parameters(model)
    parameter_order = tuple(selected)
    dimensions = sum(parameter.numel() for parameter in selected.values())
    gradients = {
        behavior: np.zeros(dimensions, dtype=np.float32)
        for behavior in BEHAVIOR_NAMES
    }
    alternatives = _alternative_token_ids(tokenizer, examples)

    for behavior_index, behavior in enumerate(BEHAVIOR_NAMES):
        model.zero_grad(set_to_none=True)
        processed = 0
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = _encode_batch(
                model,
                tokenizer,
                batch_examples,
                max_length=max_length,
                device=device,
            )
            batch_alternatives = alternatives[start : start + batch_size]
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            )
            values = behavior_values(
                outputs.logits,
                batch,
                alternative_token_ids=batch_alternatives,
            )
            # Accumulate an example-weighted mean across uneven final batches.
            (values[behavior].sum() / len(examples)).backward()
            processed += len(batch_examples)
        if processed != len(examples):
            raise RuntimeError("Not all validation examples were processed")
        flat = torch.cat(
            [
                selected[name].grad.reshape(-1).float()
                for name in parameter_order
            ]
        ).detach()
        gradients[behavior] = flat.cpu().numpy()
        model.zero_grad(set_to_none=True)
    return gradients


@dataclass
class GradSimResult:
    scores: np.ndarray
    behavior_gradients: dict[str, np.ndarray]
    parameter_order: tuple[str, ...]
    parameter_dimension: int
    elapsed_seconds: float
    peak_gpu_memory_bytes: int | None


def compute_gradsim_scores(
    model: nn.Module,
    tokenizer,
    examples: list[Example],
    *,
    behavior_gradients: dict[str, np.ndarray],
    max_length: int,
    device: torch.device,
) -> GradSimResult:
    model.eval()
    selected = lora_parameters(model)
    parameter_order = tuple(selected)
    dimensions = sum(parameter.numel() for parameter in selected.values())
    targets = torch.as_tensor(
        np.stack([behavior_gradients[name] for name in BEHAVIOR_NAMES], axis=0),
        dtype=torch.float32,
        device=device,
    )
    if targets.shape[1] != dimensions:
        raise ValueError(
            "Behavior gradient dimension does not match the current LoRA model"
        )

    scores = np.zeros((len(examples), len(BEHAVIOR_NAMES)), dtype=np.float32)
    started = time.perf_counter()
    for index, example in enumerate(examples):
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
        flat_gradient = torch.cat(
            [
                selected[name].grad.reshape(-1).float()
                for name in parameter_order
            ]
        ).detach()
        scores[index] = (
            torch.mv(targets, flat_gradient).detach().cpu().numpy()
        )

    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    model.zero_grad(set_to_none=True)
    return GradSimResult(
        scores=scores,
        behavior_gradients=behavior_gradients,
        parameter_order=parameter_order,
        parameter_dimension=dimensions,
        elapsed_seconds=time.perf_counter() - started,
        peak_gpu_memory_bytes=peak_memory,
    )


@torch.no_grad()
def compute_representation_scores(
    model: nn.Module,
    tokenizer,
    train_examples: list[Example],
    validation_examples: list[Example],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    def representations(examples: list[Example]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            prompts = [
                render_prompt(tokenizer, example.prompt)
                for example in batch_examples
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(device)
            output = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            lengths = encoded["attention_mask"].sum(dim=1)
            last_indices = lengths - 1
            hidden = output.hidden_states[-1][
                torch.arange(len(batch_examples), device=device),
                last_indices,
            ].float()
            rows.append(hidden.cpu().numpy())
        return np.concatenate(rows, axis=0)

    train_features = representations(train_examples)
    validation_features = representations(validation_examples)
    mean_validation = validation_features.mean(axis=0)
    train_norm = np.linalg.norm(train_features, axis=1)
    denominator = train_norm * np.linalg.norm(mean_validation)
    return (train_features @ mean_validation / denominator).astype(np.float32)
