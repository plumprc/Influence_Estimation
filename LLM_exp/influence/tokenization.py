"""Chat-template and response-span tokenization utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from .data import Example


_RESPONSE_SPAN_RE = re.compile(
    r"^Explanation:\s+(?P<explanation>.*)\.\s+"
    r"The answer is\s+(?P<answer>[A-Za-z0-9]+)\s*\.$",
    re.DOTALL,
)


def _apply_template(tokenizer, messages, *, add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_prompt(tokenizer, prompt: str) -> str:
    return _apply_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )


def render_full_text(tokenizer, prompt: str, response: str) -> str:
    return _apply_template(
        tokenizer,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        add_generation_prompt=False,
    )


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    prompt_length: int
    explanation_positions: tuple[int, ...]
    answer_positions: tuple[int, ...]
    answer_token_id: int


def _positions_for_span(
    offsets: list[tuple[int, int]],
    *,
    start: int,
    end: int,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start < end and token_end > start
    )


def _response_spans(response: str) -> tuple[tuple[int, int], tuple[int, int]]:
    match = _RESPONSE_SPAN_RE.match(response)
    if match is None:
        raise ValueError(f"Cannot parse response spans from {response!r}")
    return (
        (match.start("explanation"), match.end("explanation")),
        (match.start("answer"), match.end("answer")),
    )


def encode_example(
    tokenizer,
    example: Example,
    *,
    max_length: int,
) -> EncodedExample:
    prompt_text = render_prompt(tokenizer, example.prompt)
    full_text = render_full_text(tokenizer, example.prompt, example.response)
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Chat-template prompt is not a token-level prefix of the full text. "
            "This prevents reliable response-span masking."
        )
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("Example was truncated before any response tokens")

    explanation_span, answer_span = _response_spans(example.response)
    offsets = encoded["offset_mapping"]
    explanation_positions = _positions_for_span(
        offsets,
        start=len(prompt_text) + explanation_span[0],
        end=len(prompt_text) + explanation_span[1],
    )
    answer_positions = _positions_for_span(
        offsets,
        start=len(prompt_text) + answer_span[0],
        end=len(prompt_text) + answer_span[1],
    )
    if not explanation_positions:
        raise ValueError("Explanation was truncated or did not align to any tokens")
    if len(answer_positions) != 1:
        raise ValueError(
            "The answer span must tokenize as exactly one token; "
            f"got {len(answer_positions)} tokens"
        )

    labels = list(input_ids)
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
    return EncodedExample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
        prompt_length=len(prompt_ids),
        explanation_positions=explanation_positions,
        answer_positions=answer_positions,
        answer_token_id=input_ids[answer_positions[0]],
    )


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    explanation_positions: list[list[int]]
    answer_positions: list[list[int]]
    answer_token_ids: list[int]

    def to(self, device: torch.device) -> "EncodedBatch":
        return EncodedBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            explanation_positions=self.explanation_positions,
            answer_positions=self.answer_positions,
            answer_token_ids=self.answer_token_ids,
        )


def collate_examples(
    encoded: list[EncodedExample],
    *,
    pad_token_id: int,
) -> EncodedBatch:
    if not encoded:
        raise ValueError("Cannot collate an empty example list")
    max_length = max(item.input_ids.__len__() for item in encoded)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for item in encoded:
        padding = max_length - len(item.input_ids)
        input_rows.append(item.input_ids + [pad_token_id] * padding)
        label_rows.append(item.labels + [-100] * padding)
        mask_rows.append(item.attention_mask + [0] * padding)
    return EncodedBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        labels=torch.tensor(label_rows, dtype=torch.long),
        attention_mask=torch.tensor(mask_rows, dtype=torch.long),
        explanation_positions=[list(item.explanation_positions) for item in encoded],
        answer_positions=[list(item.answer_positions) for item in encoded],
        answer_token_ids=[item.answer_token_id for item in encoded],
    )


def answer_token_id_for(
    tokenizer,
    prompt: str,
    answer: str,
) -> int:
    response = f"Explanation: placeholder. The answer is {answer}."
    full_text = render_full_text(tokenizer, prompt, response)
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_text = render_prompt(tokenizer, prompt)
    _, answer_span = _response_spans(response)
    positions = _positions_for_span(
        encoded["offset_mapping"],
        start=len(prompt_text) + answer_span[0],
        end=len(prompt_text) + answer_span[1],
    )
    if len(positions) != 1:
        raise ValueError(f"Answer {answer!r} does not tokenize as one token")
    return encoded["input_ids"][positions[0]]
