"""Generate the CIFAR-10 figure from aggregated JSON results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = CODE_DIR / "plot" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = CODE_DIR / "outputs" / "cifar10"
AGG_PATH = OUTPUT_DIR / "aggregated_results.json"

COLORS = {
    "negative_loss": "#0072B2",
    "soft_margin": "#009E73",
    "hard_margin": "#E69F00",
    "target_logit": "#D55E00",
}
BEHAVIOR_LABELS = {
    "negative_loss": "negative loss",
    "soft_margin": "soft margin",
    "hard_margin": "hard margin",
    "target_logit": "target logit",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _load_aggregated() -> dict:
    if not AGG_PATH.exists():
        raise FileNotFoundError(
            f"Missing aggregated CIFAR-10 results: {AGG_PATH}. "
            "Run `python experiments/aggregate_cifar10_results.py` first."
        )
    with AGG_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def _as_float_array(values: list[float | None]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else value for value in values],
        dtype=np.float64,
    )


def load_exp5a_data() -> dict[str, np.ndarray]:
    aggregated = _load_aggregated()
    raw = aggregated["exp5a_behavior"]["per_query_kendall_tau"]
    return {behavior: _as_float_array(values) for behavior, values in raw.items()}


def load_exp5b_data() -> dict:
    return _load_aggregated()["exp5b_stepsize"]


def load_exp5c_data() -> dict[str, np.ndarray]:
    aggregated = _load_aggregated()
    raw = aggregated["exp5c_ihvp"]["per_query_kendall_tau"]
    return {behavior: _as_float_array(values) for behavior, values in raw.items()}


def _draw_no_data(ax, title: str) -> None:
    ax.text(0.5, 0.5, "No data available", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_violins(ax):
    """(a) Behavior mismatch: per-query tau distributions."""
    data = load_exp5a_data()
    entries = [
        ("soft margin", data.get("soft_margin"), COLORS["soft_margin"]),
        ("hard margin", data.get("hard_margin"), COLORS["hard_margin"]),
        ("target logit", data.get("target_logit"), COLORS["target_logit"]),
    ]
    available = [
        (label, values[np.isfinite(values)], color)
        for label, values, color in entries
        if values is not None and np.any(np.isfinite(values))
    ]
    if not available:
        _draw_no_data(ax, "(a) Behavior mismatch (CIFAR-10) - No Data")
        return

    positions = range(1, len(available) + 1)
    parts = ax.violinplot(
        [values for _, values, _ in available],
        positions=positions,
        showmedians=True,
        widths=0.8,
    )
    for body, (_, _, color) in zip(parts["bodies"], available):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor("none")
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#444444")
        parts[key].set_linewidth(0.7)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [label for label, _, _ in available],
        rotation=20,
        ha="center",
        va="center",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", pad=12)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel(r"per-query Kendall $\tau$ vs negative loss")
    ax.set_ylim(-0.1, 1.08)
    ax.set_box_aspect(1)


def draw_sweep(ax):
    """(b) Step-size sweep: tau vs eta."""
    data = load_exp5b_data()
    etas = data.get("etas", [])
    results = data.get("results", {})
    if not etas or not results:
        _draw_no_data(ax, "(b) Step-size sweep - No Data")
        return

    xs = [float(eta) for eta in etas]
    for behavior in ("negative_loss", "target_logit"):
        entries = [results[eta].get(behavior, {}) for eta in etas]
        means = np.asarray(
            [entry.get("mean") for entry in entries], dtype=np.float64
        )
        stds = np.asarray(
            [entry.get("std") for entry in entries], dtype=np.float64
        )
        ax.plot(xs, means, marker="o", ms=3, lw=1.4,
                color=COLORS[behavior], label=BEHAVIOR_LABELS[behavior])
        ax.fill_between(xs, means - stds, means + stds,
                        color=COLORS[behavior], alpha=0.15, lw=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"step size $\eta$")
    ax.set_ylabel(r"Kendall $\tau$")
    ax.set_ylim(0.3, 1.02)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6,
              borderaxespad=0.1, labelspacing=0.3)
    ax.set_box_aspect(1)


def draw_ihvp_failure(ax):
    """(c) IF-IHVP failure: per-query tau distributions."""
    data = load_exp5c_data()
    entries = [
        ("negative loss", data.get("negative_loss"), COLORS["negative_loss"]),
        ("target logit", data.get("target_logit"), COLORS["target_logit"]),
        ("soft margin", data.get("soft_margin"), COLORS["soft_margin"]),
        ("hard margin", data.get("hard_margin"), COLORS["hard_margin"]),
    ]
    available = [
        (label, values[np.isfinite(values)], color)
        for label, values, color in entries
        if values is not None and np.any(np.isfinite(values))
    ]
    if not available:
        _draw_no_data(ax, "(c) IF-IHVP vs exact (CIFAR-10) - No Data")
        return

    positions = range(1, len(available) + 1)
    parts = ax.violinplot(
        [values for _, values, _ in available],
        positions=positions,
        showmedians=True,
        widths=0.7,
    )
    for body, (_, _, color) in zip(parts["bodies"], available):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor("none")
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#444444")
        parts[key].set_linewidth(0.7)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [label for label, _, _ in available],
        rotation=20,
        ha="center",
        va="center",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", pad=12)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel(r"per-query Kendall $\tau$")
    ax.set_ylim(-0.5, 0.25)
    ax.set_box_aspect(1)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), gridspec_kw={"wspace": 0.45})
    draw_violins(axes[0])
    axes[0].set_title("(a) Behavior mismatch")
    draw_sweep(axes[1])
    axes[1].set_title("(b) Approximation error under fixed $S$")
    draw_ihvp_failure(axes[2])
    axes[2].set_title("(c) IF-IHVP transition mismatch")

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_cifar10_exp5.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {FIG_DIR / 'fig_cifar10.pdf'}")
    print(f"Saved figure to {FIG_DIR / 'fig_cifar10.png'}")


if __name__ == "__main__":
    main()
