"""Generate the combined approximation-error figure from aggregated results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = CODE_DIR / "plot" / "figures"
X_PADDING_FRACTION = 0.08
SHARED_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 0.1)
SHARED_ALPHA_LABELS = (r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$")

METRICS = ("top_5pct_overlap", "sign_accuracy")
METRIC_LABELS = {
    "top_5pct_overlap": r"Top 5% overlap",
    "sign_accuracy": "Sign agreement",
}
BEHAVIORS = ("negative_loss", "hard_margin", "target_logit")
BEHAVIOR_LABELS = {
    "negative_loss": "query loss",
    "hard_margin": "hard margin",
    "target_logit": "query logit",
}
COLORS = {
    "negative_loss": "#0072B2",
    "hard_margin": "#E69F00",
    "target_logit": "#D55E00",
    "inverse_hessian": "#7570B3",
}


@dataclass(frozen=True)
class DatasetConfig:
    label: str
    path: Path
    schema_version: int
    etas: tuple[float, ...]
    alphas: tuple[float, ...]
    step_keys: tuple[str, ...]
    alpha_keys: tuple[str, ...]
    eta_labels: tuple[str, ...]


@dataclass(frozen=True)
class DatasetData:
    one_step: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
    inverse_hessian: dict[str, tuple[np.ndarray, np.ndarray]]


CONFIGS = (
    DatasetConfig(
        label="FashionMNIST",
        path=CODE_DIR / "outputs" / "fashion_mnist" / "aggregated_results.json",
        schema_version=5,
        etas=(0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0),
        alphas=(1e-5, 1e-4, 1e-3, 1e-2, 0.1),
        step_keys=("exp3_stepsize", "etas"),
        alpha_keys=("exp4_perturbation_scale", "inverse_hessian_approximation"),
        eta_labels=("0.01", "0.02", "0.05", "0.1", "", "0.2", "", "0.5", "", "1.0"),
    ),
    DatasetConfig(
        label="CIFAR-10",
        path=CODE_DIR / "outputs" / "cifar10" / "aggregated_results.json",
        schema_version=9,
        etas=(0.01, 0.05, 0.1, 0.3, 0.5),
        alphas=(1e-5, 1e-3, 0.1),
        step_keys=("exp5d_stepsize", "summary", "sweep_results"),
        alpha_keys=("exp5b_perturbation_axis", "summary", "inverse_hessian_approximation"),
        eta_labels=("0.01", "0.05", "0.1", "0.3", "0.5"),
    ),
)

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _lookup(value: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = value[key]
    return value


def _load_dataset(config: DatasetConfig) -> DatasetData:
    if not config.path.is_file():
        raise FileNotFoundError(f"Aggregated results not found: {config.path}")

    with config.path.open(encoding="utf-8") as stream:
        aggregated = json.load(stream)
    if aggregated.get("schema_version") != config.schema_version:
        raise ValueError(
            f"Unexpected {config.label} aggregate schema: "
            f"{aggregated.get('schema_version')!r}"
        )

    step_root = _lookup(aggregated, config.step_keys)
    alpha_root = _lookup(aggregated, config.alpha_keys)
    one_step: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    inverse_hessian: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for metric in METRICS:
        one_step[metric] = {}
        for behavior in BEHAVIORS:
            stats = [step_root[str(eta)][behavior][metric] for eta in config.etas]
            one_step[metric][behavior] = (
                np.asarray([item["mean"] for item in stats], dtype=np.float64),
                np.asarray([item["ci95"] for item in stats], dtype=np.float64),
            )

        stats = [alpha_root[f"alpha_{alpha:.10g}"][metric] for alpha in config.alphas]
        inverse_hessian[metric] = (
            np.asarray([item["mean"] for item in stats], dtype=np.float64),
            np.asarray([item["ci95"] for item in stats], dtype=np.float64),
        )

    values = [
        *(mean for metric in METRICS for mean, _ in one_step[metric].values()),
        *(mean for mean, _ in inverse_hessian.values()),
    ]
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError(f"{config.label} approximation metrics contain non-finite values")
    return DatasetData(one_step=one_step, inverse_hessian=inverse_hessian)


def _symmetric_log_limits(values: tuple[float, ...]) -> tuple[float, float]:
    log_min = np.log(min(values))
    log_max = np.log(max(values))
    span = log_max - log_min
    return (
        float(np.exp(log_min - X_PADDING_FRACTION * span)),
        float(np.exp(log_max + X_PADDING_FRACTION * span)),
    )


def _draw_panel(
    ax: plt.Axes,
    ax_top: plt.Axes,
    config: DatasetConfig,
    data: DatasetData,
    metric: str,
) -> None:
    for behavior in BEHAVIORS:
        means, ci95s = data.one_step[metric][behavior]
        ax.plot(
            config.etas,
            means,
            marker="o",
            ms=3,
            lw=1.4,
            color=COLORS[behavior],
            label=BEHAVIOR_LABELS[behavior],
        )
        ax.fill_between(
            config.etas,
            means - ci95s,
            means + ci95s,
            color=COLORS[behavior],
            alpha=0.15,
            lw=0,
        )

    means, ci95s = data.inverse_hessian[metric]
    ax_top.plot(
        config.alphas,
        means,
        marker="o",
        ms=3.5,
        lw=1.5,
        color=COLORS["inverse_hessian"],
        label="inverse-Hessian",
    )
    ax_top.fill_between(
        config.alphas,
        means - ci95s,
        means + ci95s,
        color=COLORS["inverse_hessian"],
        alpha=0.18,
        lw=0,
    )

    ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlim(*_symmetric_log_limits(config.etas))
    ax.set_xticks(config.etas)
    ax.set_xticklabels(config.eta_labels)
    ax.set_xlabel(r"step size $\eta$")

    ax_top.set_xscale("log")
    ax_top.set_xlim(*_symmetric_log_limits(SHARED_ALPHAS))
    ax_top.set_xticks(SHARED_ALPHAS)
    ax_top.set_xticklabels(SHARED_ALPHA_LABELS)
    ax_top.minorticks_off()
    ax_top.set_xlabel(r"$\alpha$", loc="right")

    means_all = np.concatenate(
        [
            *(data.one_step[metric][behavior][0] for behavior in BEHAVIORS),
            data.inverse_hessian[metric][0],
        ]
    )
    ci95_all = np.concatenate(
        [
            *(data.one_step[metric][behavior][1] for behavior in BEHAVIORS),
            data.inverse_hessian[metric][1],
        ]
    )
    lower = float(np.min(means_all - ci95_all))
    upper = float(np.max(means_all + ci95_all))
    padding = 0.05 * (upper - lower)
    ax.set_ylim(max(0.0, lower - padding), min(1.02, upper + padding))
    ax.set_ylabel(METRIC_LABELS[metric])


def main() -> None:
    datasets = [(config, _load_dataset(config)) for config in CONFIGS]
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.8), constrained_layout=True)

    for panel_index, ((config, data), metric) in enumerate(
        (item, metric) for item in datasets for metric in METRICS
    ):
        ax = axes[panel_index]
        ax_top = ax.twiny()
        _draw_panel(ax, ax_top, config, data, metric)
        ax_top.spines["top"].set_visible(True)
        ax_top.spines["right"].set_visible(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.spines["left"].set_visible(False)
        ax_top.tick_params(
            axis="x",
            which="both",
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
        )
        ax_top.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_title(
            f"({chr(ord('a') + panel_index)}) {config.label}",
            loc="center",
            fontweight="bold",
        )

    handles = [
        plt.Line2D([], [], color=COLORS[behavior], marker="o", ms=3, lw=1.4)
        for behavior in BEHAVIORS
    ]
    handles.append(
        plt.Line2D([], [], color=COLORS["inverse_hessian"], marker="o", ms=3.5, lw=1.5)
    )
    labels = [BEHAVIOR_LABELS[behavior] for behavior in BEHAVIORS] + ["inverse-Hessian"]
    axes[3].legend(
        handles=handles,
        labels=labels,
        frameon=False,
        loc="center left",
        handlelength=1.5,
        labelspacing=0.25,
    )

    for extension in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_app.{extension}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'fig_app.pdf'}")
    print(f"Saved {FIG_DIR / 'fig_app.png'}")


if __name__ == "__main__":
    main()
