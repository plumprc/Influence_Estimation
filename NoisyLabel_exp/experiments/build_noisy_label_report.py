"""Build a concise statistical report from the noisy-label aggregate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGGREGATED = PROJECT_ROOT / "outputs" / "aggregated_results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "noisy_label_report.md"

RHOS = (0.05, 0.1, 0.2, 0.3, 0.4)
METHODS = ("gradsim", "tracin", "lissa_if")
BEHAVIORS = ("negative_loss", "target_logit", "hard_margin")
BASELINES = ("random", "representation_similarity", "ntk_similarity")
MODELS = ("resnet50", "vgg19_bn", "mobilenetv2")

LABELS = {
    "gradsim": "GradSim",
    "tracin": "TracIn",
    "lissa_if": "LiSSA",
    "negative_loss": "loss",
    "target_logit": "logit",
    "hard_margin": "margin",
    "random": "Random",
    "representation_similarity": "RepSim",
    "ntk_similarity": "NTK",
    "resnet50": "ResNet-50",
    "vgg19_bn": "VGG-19-BN",
    "mobilenetv2": "MobileNetV2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a report from aggregated_results.json"
    )
    parser.add_argument("--aggregated", type=Path, default=DEFAULT_AGGREGATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def stat(value: dict, digits: int = 3) -> str:
    if value["n"] == 0:
        return "missing"
    if value["n"] == 1:
        return f"{value['mean']:.{digits}f}"
    return f"{value['mean']:.{digits}f} +/- {value['std']:.{digits}f}"


def completion_lines(payload: dict) -> list[str]:
    lines = [
        "## Completion",
        "",
        "| Family | Included | Complete | Available / expected |",
        "|---|---|---|---:|",
    ]
    for family in ("main", "architecture"):
        block = payload[family]
        completeness = block["completeness"]
        included = "yes" if block["included"] else "no"
        complete = "yes" if completeness["complete"] else "no"
        runs = completeness.get("runs", {"available": 0, "expected": 0})
        count = f"{runs['available']} / {runs['expected']}"
        lines.append(f"| {family} | {included} | {complete} | {count} |")
    return lines + [""]


def main_detection_lines(payload: dict) -> list[str]:
    main_aggregates = payload["aggregates"]["main"]
    lines = [
        "## Main Detection",
        "",
        "Values are mean +/- sample standard deviation over three training seeds.",
        "",
        "| rho | Specification | AUPRC | P@10% |",
        "|---:|---|---:|---:|",
    ]
    for rho in RHOS:
        rho_aggregates = main_aggregates["rhos"][f"rho_{rho}"]
        detection = rho_aggregates["detection"]
        baselines = rho_aggregates["baselines"]
        for method in METHODS:
            for behavior in BEHAVIORS:
                metrics = detection[method][behavior]
                lines.append(
                    f"| {rho:.2f} | {LABELS[method]} / {LABELS[behavior]} | "
                    f"{stat(metrics['auprc'])} | "
                    f"{stat(metrics['precision_at_10pct'])} |"
                )
        for baseline in BASELINES:
            metrics = baselines[baseline]
            lines.append(
                f"| {rho:.2f} | {LABELS[baseline]} | "
                f"{stat(metrics['auprc'])} | "
                f"{stat(metrics['precision_at_10pct'])} |"
            )
    return lines + [""]


def removal_lines(payload: dict) -> list[str]:
    removal = payload["aggregates"]["main"]["removal"]
    lines = [
        "## Removal Utility",
        "",
        "Test-accuracy change after removing the top 10% and retraining. "
        "Values are percentage points over three source seeds.",
        "",
        "| Specification | Test accuracy change | Selected noise rate |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        for behavior in BEHAVIORS:
            row = removal[f"{method}/{behavior}/0.100"]
            delta = row["test_accuracy_delta"]
            lines.append(
                f"| {LABELS[method]} / {LABELS[behavior]} | "
                f"{100 * delta['mean']:.3f} +/- {100 * delta['std']:.3f} | "
                f"{stat(row['selected_noise_rate'])} |"
            )
    for baseline in BASELINES:
        row = removal[f"{baseline}/behavior_free/0.100"]
        delta = row["test_accuracy_delta"]
        lines.append(
            f"| {LABELS[baseline]} | "
            f"{100 * delta['mean']:.3f} +/- {100 * delta['std']:.3f} | "
            f"{stat(row['selected_noise_rate'])} |"
        )
    return lines + [""]


def architecture_lines(payload: dict) -> list[str]:
    if not payload["architecture"]["included"]:
        return []
    architecture_aggregates = payload["aggregates"]["architecture"]
    lines = [
        "## Architecture Extension",
        "",
        "AUPRC at rho = 0.2. Values are mean +/- sample standard deviation over "
        "three training seeds.",
        "",
        "| Model | Specification | AUPRC | P@10% |",
        "|---|---|---:|---:|",
    ]
    for model in MODELS:
        detection = architecture_aggregates["models"][model]["detection"]
        for method in METHODS:
            for behavior in BEHAVIORS:
                metrics = detection[method][behavior]
                lines.append(
                    f"| {LABELS[model]} | {LABELS[method]} / {LABELS[behavior]} | "
                    f"{stat(metrics['auprc'])} | "
                    f"{stat(metrics['precision_at_10pct'])} |"
                )
    return lines + [""]


def main() -> None:
    args = parse_args()
    payload = load_json(args.aggregated)
    lines = [
        "# Noisy-Label Experiment Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    lines.extend(completion_lines(payload))
    if payload["main"]["included"]:
        lines.extend(main_detection_lines(payload))
        lines.extend(removal_lines(payload))
    lines.extend(architecture_lines(payload))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
