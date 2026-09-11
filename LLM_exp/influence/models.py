"""Model resolution and LoRA setup."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)


MODEL_ALIASES = {
    "qwen3-8b": "Qwen/Qwen3-8B",
}
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

def resolve_model_source(source: str) -> str:
    if Path(source).exists():
        return str(Path(source).resolve())
    return MODEL_ALIASES.get(source, source)


def load_foundation(
    model_source: str,
    *,
    device: str = "cuda",
    local_files_only: bool = False,
) -> tuple[PreTrainedModel, AutoTokenizer]:
    resolved = resolve_model_source(model_source)
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        dtype=torch.bfloat16,
        device_map="auto" if device.startswith("cuda") else None,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def make_lora_config(
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules or list(LORA_TARGET_MODULES),
    )


def add_lora(
    model: PreTrainedModel,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
):
    target_modules = [
        name
        for name, _ in model.named_modules()
        if name.endswith(LORA_TARGET_MODULES)
    ]
    if not target_modules:
        raise ValueError("No LoRA target modules were found")

    peft_model = get_peft_model(
        model,
        make_lora_config(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        ),
    )
    return peft_model


def load_adapter(
    model_source: str,
    adapter_path: str | Path,
    *,
    device: str = "cuda",
    local_files_only: bool = False,
):
    model, tokenizer = load_foundation(
        model_source,
        device=device,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=True,
    )
    model.config.use_cache = False
    return model, tokenizer
