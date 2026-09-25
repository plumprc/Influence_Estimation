"""Generate the combined FashionMNIST figure for Findings 1 and 2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


CODE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = CODE_DIR / "outputs" / "fashion_mnist"
FIG_DIR = CODE_DIR / "plot" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
AGGREGATED_PATH = OUTPUT_DIR / "aggregated_results.json"
SCHEMA_VERSION = 5

ETAS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)
ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 0.1)
X_PADDING_FRACTION = 0.08

FINDING1_COLORS = {
    "behavior_soft_margin": "#009E73",
    "behavior_hard_margin": "#E69F00",
    "behavior_query_logit": "#D55E00",
    "upweight_1e-3_vs_loo": "#0173B2",
    "upweight_1e-1_vs_loo": "#CA9161",
    "transition_multi_step": "#56B4E9",
    "transition_inverse_hessian": "#7570B3",
}

FINDING2_BEHAVIORS = ("negative_loss", "hard_margin", "target_logit")
FINDING2_BEHAVIOR_LABELS = {
    "negative_loss": "query loss",
    "hard_margin": "hard margin",
    "target_logit": "query logit",
}
FINDING2_COLORS = {
    "negative_loss": "#0072B2",
    "hard_margin": "#E69F00",
    "target_logit": "#D55E00",
    "inverse_hessian": "#7570B3",
}

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


@dataclass(frozen=True)
class Comparison:
    key: str
    label: str
    axis: str
    color: str
    taus: np.ndarray


def _load_figure_data() -> tuple[list[Comparison], dict[str, np.ndarray]]:
    if not AGGREGATED_PATH.is_file():
        raise FileNotFoundError(f"Aggregated results not found: {AGGREGATED_PATH}")

    with AGGREGATED_PATH.open(encoding="utf-8") as stream:
        aggregated = json.load(stream)
    if aggregated.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Expected aggregated-results schema {SCHEMA_VERSION}, got "
            f"{aggregated.get('schema_version')!r}"
        )

    try:
        figure = aggregated["figure"]
        comparisons_raw = figure["finding1"]["comparisons"]
        one_step_raw = figure["finding2"]["one_step"]
        reoptimization_raw = figure["finding2"]["reoptimization"]
    except (KeyError, TypeError) as error:
        raise ValueError("Aggregated results do not contain figure data") from error

    expected_keys = list(FINDING1_COLORS)
    actual_keys = [item["key"] for item in comparisons_raw]
    if actual_keys != expected_keys:
        raise ValueError(
            "Finding 1 comparisons in aggregated results do not match the figure: "
            f"expected {expected_keys}, got {actual_keys}"
        )

    comparisons = [
        Comparison(
            key=item["key"],
            label=item["label"],
            axis=f"${item['axis']}$",
            color=FINDING1_COLORS[item["key"]],
            taus=np.asarray(
                [value for seed in item["tau_by_seed"] for value in seed],
                dtype=np.float64,
            ),
        )
        for item in comparisons_raw
    ]

    values: dict[str, np.ndarray] = {}
    for item in one_step_raw:
        eta = float(item["eta"])
        behavior = str(item["behavior"])
        key = f"eta__{eta:g}__{behavior}"
        values[key] = np.asarray(item["tau_by_seed"], dtype=np.float64)

    for item in reoptimization_raw:
        alpha = float(item["alpha"])
        values[f"alpha__{alpha:.10g}"] = np.asarray(
            item["tau_by_seed"], dtype=np.float64
        )
    expected_finding2_keys = {
        *(f"eta__{eta:g}__{behavior}" for eta in ETAS for behavior in FINDING2_BEHAVIORS),
        *(f"alpha__{alpha:.10g}" for alpha in ALPHAS),
    }
    if set(values) != expected_finding2_keys:
        missing = sorted(expected_finding2_keys - set(values))
        unexpected = sorted(set(values) - expected_finding2_keys)
        raise ValueError(
            "Finding 2 data in aggregated results do not match the figure; "
            f"missing={missing}, unexpected={unexpected}"
        )

    if any(item.taus.size == 0 or not np.isfinite(item.taus).all() for item in comparisons):
        raise ValueError("Finding 1 data contain empty or non-finite tau values")
    if any(values.size == 0 or not np.isfinite(values).all() for values in values.values()):
        raise ValueError("Finding 2 data contain empty or non-finite tau values")

    return comparisons, values


def _mean_and_ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite seed-level tau values")
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    half_width = float(
        stats.t.ppf(0.975, df=values.size - 1)
        * values.std(ddof=1)
        / np.sqrt(values.size)
    )
    return mean, half_width


def _symmetric_log_limits(values: tuple[float, ...]) -> tuple[float, float]:
    log_min = np.log(min(values))
    log_max = np.log(max(values))
    span = log_max - log_min
    return (
        float(np.exp(log_min - X_PADDING_FRACTION * span)),
        float(np.exp(log_max + X_PADDING_FRACTION * span)),
    )


def _draw_finding1(ax: plt.Axes, comparisons: list[Comparison]) -> None:
    positions = np.asarray([1, 2, 3, 4.6, 5.6, 7.2, 8.2], dtype=float)
    parts = ax.violinplot(
        [item.taus for item in comparisons],
        positions=positions,
        widths=0.8,
        showmedians=True,
    )
    for body, item in zip(parts["bodies"], comparisons):
        body.set_facecolor(item.color)
        body.set_alpha(0.8)
        body.set_edgecolor("none")
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#444444")
        parts[key].set_linewidth(0.7)

    ax.axvline(3.8, color="grey", lw=0.6, ls=":")
    ax.axvline(6.4, color="grey", lw=0.6, ls=":")
    ax.set_xticks([2, 5.1, 7.7])
    ax.set_xticklabels([r"behavior $B$", r"perturbation $P$", r"transition $T$"])
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel(r"Kendall' $\tau$")
    ax.set_ylim(-0.2, 1.0)

    patch = plt.matplotlib.patches.Patch
    behavior_handles = [
        patch(color=item.color, alpha=0.75, label=f"{item.axis}: {item.label}")
        for item in comparisons[:3]
    ]
    other_handles = [
        patch(color=item.color, alpha=0.75, label=f"{item.axis}: {item.label}")
        for item in comparisons[3:]
    ]
    behavior_legend = ax.legend(
        handles=behavior_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        handlelength=0.9,
        handleheight=0.9,
        labelspacing=0.2,
        columnspacing=0.8,
    )
    ax.add_artist(behavior_legend)
    ax.legend(
        handles=other_handles,
        frameon=False,
        loc="lower center",
        ncol=2,
        handlelength=0.9,
        handleheight=0.9,
        labelspacing=0.2,
        columnspacing=0.8,
    )


def _draw_one_step(ax: plt.Axes, values: dict[str, np.ndarray]) -> None:
    for behavior in FINDING2_BEHAVIORS:
        points = [
            _mean_and_ci(values[f"eta__{eta:g}__{behavior}"])
            for eta in ETAS
        ]
        means = [point[0] for point in points]
        half_widths = [point[1] for point in points]
        ax.plot(
            ETAS,
            means,
            marker="o",
            ms=3,
            lw=1.4,
            color=FINDING2_COLORS[behavior],
            label=FINDING2_BEHAVIOR_LABELS[behavior],
        )
        ax.fill_between(
            ETAS,
            np.asarray(means) - np.asarray(half_widths),
            np.asarray(means) + np.asarray(half_widths),
            color=FINDING2_COLORS[behavior],
            alpha=0.15,
            lw=0,
        )

    ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.7)
    ax.set_xscale("log")
    displayed_etas = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    ax.set_xlim(*_symmetric_log_limits(ETAS))
    ax.set_xticks(ETAS)
    ax.set_xticklabels(
        [f"{eta:g}" if eta in displayed_etas else "" for eta in ETAS]
    )
    ax.set_xlabel(r"step size $\eta$")
    ax.axhline(1.0, color="grey", lw=0.5, ls=":")


def _draw_reoptimization(ax: plt.Axes, values: dict[str, np.ndarray]) -> None:
    points = [_mean_and_ci(values[f"alpha__{alpha:.10g}"]) for alpha in ALPHAS]
    means = [point[0] for point in points]
    half_widths = [point[1] for point in points]
    alpha_labels = [r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
    ax.plot(
        ALPHAS,
        means,
        marker="o",
        ms=3.5,
        lw=1.5,
        color=FINDING2_COLORS["inverse_hessian"],
        label="inverse-Hessian",
    )
    ax.fill_between(
        ALPHAS,
        np.asarray(means) - np.asarray(half_widths),
        np.asarray(means) + np.asarray(half_widths),
        color=FINDING2_COLORS["inverse_hessian"],
        alpha=0.18,
        lw=0,
    )
    ax.set_xscale("log")
    ax.set_xlim(*_symmetric_log_limits(ALPHAS))
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels(alpha_labels)
    ax.minorticks_off()
    ax.set_xlabel(r"$\alpha$", loc="right")
    ax.axhline(1.0, color="grey", lw=0.5, ls=":")


def main() -> None:
    comparisons, values = _load_figure_data()
    fig, (ax_finding1, ax_finding2) = plt.subplots(
        1,
        2,
        figsize=(9.8, 3.9),
        gridspec_kw={"width_ratios": (1.1, 1.0)},
        constrained_layout=True,
    )
    ax_reoptimization = ax_finding2.twiny()

    _draw_finding1(ax_finding1, comparisons)
    _draw_one_step(ax_finding2, values)
    _draw_reoptimization(ax_reoptimization, values)

    ax_finding1.set_title(
        "(a) Specification mismatch",
        loc="center",
        fontweight="bold",
        y=1.16,
    )
    ax_finding2.set_title("(b) Approximation error", loc="center", fontweight="bold")

    ax_reoptimization.spines["top"].set_visible(True)
    ax_reoptimization.spines["right"].set_visible(False)
    ax_reoptimization.spines["bottom"].set_visible(False)
    ax_reoptimization.spines["left"].set_visible(False)
    ax_reoptimization.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False)
    ax_reoptimization.tick_params(axis="y", which="both", left=False, labelleft=False)

    ax_finding1.set_ylabel(r"Kendall' $\tau$")
    ax_finding2.set_ylabel(r"Kendall' $\tau$")
    ax_finding2.set_xlim(*_symmetric_log_limits(ETAS))
    ax_finding2.set_ylim(0.4, 1.02)

    one_step_handles, one_step_labels = ax_finding2.get_legend_handles_labels()
    reoptimization_handles, reoptimization_labels = ax_reoptimization.get_legend_handles_labels()
    ax_finding2.legend(
        handles=one_step_handles + reoptimization_handles,
        labels=one_step_labels + reoptimization_labels,
        frameon=False,
        loc="lower left",
        ncol=2,
        handlelength=1.5,
        labelspacing=0.25,
        columnspacing=1.0,
    )

    fig.canvas.draw()
    ax_finding1.title.set_position((0.5, ax_finding2.title.get_position()[1]))

    for extension in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_fmnist.{extension}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'fig_fmnist.pdf'}")
    print(f"Saved {FIG_DIR / 'fig_fmnist.png'}")


if __name__ == "__main__":
    main()
