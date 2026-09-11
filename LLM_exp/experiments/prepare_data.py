"""Build ScienceQA source data and the two experiment datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import string
import sys

from datasets import load_dataset
from datasets.download.download_config import DownloadConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influence.data import (
    BACKDOOR_TEST_VARIANTS,
    CONDITIONAL_BACKDOOR_TASK,
    build_conditional_backdoor,
    build_response_corruption,
    load_source_records,
    save_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("source")
    source.add_argument("--dataset", default="derek-thomas/ScienceQA")
    source.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/scienceqa/source.json"),
    )
    source.add_argument("--local-files-only", action="store_true")

    exp1 = subparsers.add_parser("exp1")
    exp1.add_argument(
        "--source", type=Path, default=Path("datasets/scienceqa/source.json")
    )
    exp1.add_argument(
        "--output-dir", type=Path, default=Path("datasets/scienceqa/exp1")
    )
    exp1.add_argument("--train-size", type=int, default=5000)
    exp1.add_argument("--validation-size", type=int, default=500)
    exp1.add_argument("--answer-corruption-rate", type=float, default=0.10)
    exp1.add_argument("--rationale-corruption-rate", type=float, default=0.10)
    exp1.add_argument("--seed", type=int, default=0)
    exp1.add_argument("--overwrite", action="store_true")

    exp2 = subparsers.add_parser("exp2")
    exp2.add_argument(
        "--source", type=Path, default=Path("datasets/scienceqa/source.json")
    )
    exp2.add_argument(
        "--output-dir", type=Path, default=Path("datasets/scienceqa/exp2")
    )
    exp2.add_argument("--train-size", type=int, default=5000)
    exp2.add_argument("--test-size", type=int, default=500)
    exp2.add_argument("--harmful-poison-count", type=int, default=500)
    exp2.add_argument("--benign-trigger-count", type=int, default=500)
    exp2.add_argument("--seed", type=int, default=0)
    exp2.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def _prompt(question: str, choices: list[str], hint: str) -> str:
    lines = [question.strip()]
    if hint.strip():
        lines.extend(("", f"Hint: {hint.strip()}"))
    lines.append("")
    lines.extend(
        f"{label}: {choice}"
        for label, choice in zip(string.ascii_uppercase, choices, strict=True)
    )
    return "\n".join(lines)


def build_source(args: argparse.Namespace) -> None:
    download_config = (
        DownloadConfig(local_files_only=True)
        if args.local_files_only
        else None
    )
    records: list[dict] = []
    composition: dict[str, int] = {}
    for official_split in ("train", "validation"):
        split = load_dataset(
            args.dataset,
            split=official_split,
            download_config=download_config,
        )
        split_count = 0
        for index, record in enumerate(split):
            if record["image"] is not None:
                continue
            explanation = str(record["solution"] or "").strip()
            choices = [str(choice) for choice in record["choices"]]
            answer_index = int(record["answer"])
            if not explanation or len(choices) < 2:
                continue
            if not 0 <= answer_index < len(choices):
                continue
            answer = string.ascii_uppercase[answer_index]
            records.append(
                {
                    "id": f"scienceqa-{official_split}-{index:06d}",
                    "prompt": _prompt(
                        record["question"], choices, record["hint"]
                    ),
                    "answer": answer,
                    "choices": list(string.ascii_uppercase[: len(choices)]),
                    "explanation": explanation,
                    "metadata": {
                        "subset": "ScienceQA",
                        "official_split": official_split,
                        "subject": record["subject"],
                        "topic": record["topic"],
                        "category": record["category"],
                        "grade": record["grade"],
                        "skill": record["skill"],
                        "text_only": True,
                    },
                }
            )
            split_count += 1
        composition[official_split] = split_count

    if len(records) < 2400:
        raise ValueError(
            f"Only found {len(records)} eligible text-only ScienceQA records"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "source": args.dataset,
                "stage": 1,
                "record_count": len(records),
                "composition": composition,
                "records": records,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")
    print(f"Wrote {len(records)} text-only records to {args.output}")
    print(f"Composition: {composition}")


def _check_output_files(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing dataset files: {', '.join(existing)}"
        )


def prepare_exp1(args: argparse.Namespace) -> None:
    output_files = [args.output_dir / "train.json", args.output_dir / "validation.json"]
    _check_output_files(output_files, args.overwrite)
    records = load_source_records(args.source)
    train, validation = build_response_corruption(
        records,
        train_size=args.train_size,
        validation_size=args.validation_size,
        answer_corruption_rate=args.answer_corruption_rate,
        rationale_corruption_rate=args.rationale_corruption_rate,
        seed=args.seed,
        train_source_split="train",
        validation_source_split="validation",
    )
    save_dataset(
        args.output_dir / "train.json",
        task="response_corruption_stage1",
        split="train",
        records=train,
    )
    save_dataset(
        args.output_dir / "validation.json",
        task="response_corruption_stage1",
        split="validation",
        records=validation,
    )
    counts = {
        group: sum(record["corruption_type"] == group for record in train)
        for group in ("clean", "answer_corruption", "rationale_corruption")
    }
    print(f"Wrote exp1 datasets to {args.output_dir}")
    print(f"Train groups: {counts}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_exp2(args: argparse.Namespace) -> None:
    output_files = {
        "train": args.output_dir / "train.json",
        **{
            variant: args.output_dir / f"test_{variant}.json"
            for variant in BACKDOOR_TEST_VARIANTS
        },
    }
    _check_output_files(list(output_files.values()), args.overwrite)
    records = load_source_records(args.source)
    train, tests = build_conditional_backdoor(
        records,
        train_size=args.train_size,
        test_size=args.test_size,
        harmful_poison_count=args.harmful_poison_count,
        benign_trigger_count=args.benign_trigger_count,
        seed=args.seed,
        train_source_split="train",
        test_source_split="validation",
    )
    save_dataset(
        output_files["train"],
        task=CONDITIONAL_BACKDOOR_TASK,
        split="train",
        records=train,
    )
    for variant in BACKDOOR_TEST_VARIANTS:
        save_dataset(
            output_files[variant],
            task=CONDITIONAL_BACKDOOR_TASK,
            split="test",
            records=tests[variant],
        )

    group_counts = {
        group: sum(record["group"] == group for record in train)
        for group in sorted({record["group"] for record in train})
    }
    manifest = {
        "schema_version": 1,
        "task": CONDITIONAL_BACKDOOR_TASK,
        "source": str(args.source),
        "seed": args.seed,
        "static_dataset": True,
        "configuration": {
            "train_size": args.train_size,
            "test_size": args.test_size,
            "harmful_poison_count": args.harmful_poison_count,
            "benign_trigger_count": args.benign_trigger_count,
            "train_source_split": "train",
            "test_source_split": "validation",
        },
        "train_group_counts": group_counts,
        "files": {
            name: {
                "path": path.name,
                "record_count": len(train if name == "train" else tests[name]),
                "sha256": _sha256(path),
            }
            for name, path in output_files.items()
        },
    }
    temporary = args.output_dir / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(args.output_dir / "manifest.json")
    print(f"Wrote exp2 datasets to {args.output_dir}")
    print(f"Train groups: {group_counts}")


def main() -> None:
    args = parse_args()
    if args.command == "source":
        build_source(args)
    elif args.command == "exp1":
        prepare_exp1(args)
    else:
        prepare_exp2(args)


if __name__ == "__main__":
    main()
