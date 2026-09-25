"""Train a LoRA adapter on observed explanation-answer responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.data import (
    CONDITIONAL_BACKDOOR_TASK,
    RESPONSE_CORRUPTION_TASK,
    Example,
    load_dataset,
)
from influence.models import add_lora, load_foundation
from influence.tokenization import collate_examples, encode_example


class ExampleDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--checkpoint-fractions",
        nargs="+",
        type=float,
        default=(0.33, 0.66),
        help="Fractions of total optimizer steps at which to save TracIn checkpoints",
    )
    parser.add_argument(
        "--no-intermediate-checkpoints",
        action="store_true",
        help="Skip intermediate checkpoints for runs that only need the final adapter",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_fractions = (
        [] if args.no_intermediate_checkpoints else args.checkpoint_fractions
    )
    if any(not 0.0 < fraction < 1.0 for fraction in checkpoint_fractions):
        raise ValueError("Checkpoint fractions must be strictly between 0 and 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dataset = load_dataset(args.train_data)
    if dataset.task not in {RESPONSE_CORRUPTION_TASK, CONDITIONAL_BACKDOOR_TASK}:
        raise ValueError(f"Unexpected task: {dataset.task}")

    model, tokenizer = load_foundation(
        args.model_source,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    model = add_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    def collate(examples: list[Example]):
        encoded = [
            encode_example(tokenizer, example, max_length=args.max_length)
            for example in examples
        ]
        return collate_examples(encoded, pad_token_id=tokenizer.pad_token_id)

    loader = DataLoader(
        ExampleDataset(dataset.examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(loader) * args.epochs)
    checkpoint_steps = {
        max(1, min(total_steps, int(np.ceil(total_steps * fraction)))): fraction
        for fraction in checkpoint_fractions
    }
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / max(1, total_steps * args.warmup_ratio),
            0.5
            * (
                1
                + np.cos(
                    np.pi
                    * max(0.0, step - total_steps * args.warmup_ratio)
                    / max(1.0, total_steps * (1 - args.warmup_ratio))
                )
            ),
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    losses: list[float] = []
    global_step = 0
    last_step_lr = args.learning_rate
    saved_checkpoints: list[dict[str, object]] = []

    def save_intermediate_checkpoint(
        *,
        fraction: float,
        step: int,
        learning_rate: float,
    ) -> None:
        checkpoint_dir = args.output_dir / f"checkpoint_{round(fraction * 100):03d}"
        model.save_pretrained(checkpoint_dir)
        checkpoint_summary = {
            "stage": "train_checkpoint",
            "task": dataset.task,
            "model_source": args.model_source,
            "checkpoint_fraction": fraction,
            "global_step": step,
            "total_steps": total_steps,
            "checkpoint_learning_rate": learning_rate,
        }
        with (checkpoint_dir / "summary.json").open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(checkpoint_summary, stream, indent=2)
            stream.write("\n")
        saved_checkpoints.append(
            {
                "path": str(checkpoint_dir.relative_to(args.output_dir)),
                "fraction": fraction,
                "step": step,
                "learning_rate": learning_rate,
            }
        )
        print(f"Saved checkpoint at step {step}/{total_steps}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            batch = batch.to(device)
            logits = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            ).logits
            shifted_logits = logits[:, :-1, :]
            shifted_labels = batch.labels[:, 1:]
            response_mask = shifted_labels.ne(-100)
            nll = torch.nn.functional.cross_entropy(
                shifted_logits.transpose(1, 2),
                shifted_labels,
                reduction="none",
            )
            token_counts = response_mask.sum(dim=1).clamp(min=1)
            per_example_nll = (nll * response_mask).sum(dim=1) / token_counts
            loss = per_example_nll.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            last_step_lr = float(scheduler.get_last_lr()[0])
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            epoch_losses.append(float(loss.detach()))
            global_step += 1
            if global_step in checkpoint_steps:
                save_intermediate_checkpoint(
                    fraction=checkpoint_steps[global_step],
                    step=global_step,
                    learning_rate=last_step_lr,
                )
            if global_step % 10 == 0:
                print(
                    f"step={global_step}/{total_steps} "
                    f"loss={losses[-1]:.6f}",
                    flush=True,
                )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"mean_loss={float(np.mean(epoch_losses)):.6f}",
            flush=True,
        )

    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "global_step": global_step,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "training_state.pt")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    elapsed = time.perf_counter() - started
    summary = {
        "stage": "train",
        "task": dataset.task,
        "model_source": args.model_source,
        "train_data": str(args.train_data),
        "train_examples": len(dataset.examples),
        "train_group_counts": (
            {
                group: sum(record.get("group") == group for record in dataset.records)
                for group in sorted({record.get("group") for record in dataset.records})
            }
            if dataset.task == CONDITIONAL_BACKDOOR_TASK
            else None
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "seed": args.seed,
        "steps": global_step,
        "checkpoint_learning_rate": last_step_lr,
        "checkpoint_fractions": list(checkpoint_fractions),
        "checkpoints": saved_checkpoints,
        "training_state": "training_state.pt",
        "final_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
