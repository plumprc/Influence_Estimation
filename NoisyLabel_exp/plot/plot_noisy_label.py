"""Generate the main three-panel noisy-label influence figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGGREGATED = PROJECT_ROOT / "outputs" / "aggregated_results.json"
FIGURE_DIR = PROJECT_ROOT / "plot" / "figures"

RHOS = (0.05, 0.1, 0.2, 0.3, 0.4)
RANKING_RHOS = (0.2,)
SEEDS = (0, 1, 2)
METHODS = ("gradsim", "tracin", "lissa_if")
BEHAVIORS = ("negative_loss", "target_logit", "hard_margin")
ARCHITECTURES = ("resnet18", "resnet50", "vgg19_bn", "mobilenetv2")

METHOD_LABELS = {
    "gradsim": "GradSim",
    "tracin": "TracIn",
    "lissa_if": "LiSSA",
}
BEHAVIOR_LABELS = {
    "negative_loss": "loss",
    "target_logit": "logit",
    "hard_margin": "margin",
}
ARCHITECTURE_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vgg19_bn": "VGG-19",
    "mobilenetv2": "MobileNetV2",
}
ARCHITECTURE_COLORS = {
    "resnet18": "#9585BE",
    "resnet50": "#C6829F",
    "vgg19_bn": "#7BA7B4",
    "mobilenetv2": "#B0B76A",
}
METHOD_COLORS = {
    "gradsim": "#84A1C1",
    "tracin": "#E69658",
    "lissa_if": "#63B094",
}
NETWORK_METHOD_COLORS = {
    "gradsim": "#6D8CAB",
    "tracin": "#D9803D",
    "lissa_if": "#4D9A7B",
}
BEHAVIOR_MARKERS = {
    "negative_loss": "o",
    "target_logit": "s",
    "hard_margin": "^",
}
BASELINE_SPECS = (
    ("random", "Random"),
    ("representation_similarity", "RepSim"),
    ("ntk_similarity", "NTK"),
)
BASELINE_COLORS = {
    "random": "#8B939D",
    "representation_similarity": "#D0D4DA",
    "ntk_similarity": "#BEC6CE",
}

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _summaries_from_aggregated(payload: dict) -> dict[tuple[float, int], dict]:
    summaries = {}
    for key, summary in payload["main"]["runs"].items():
        rho_text, seed_text = key.split("/")
        rho = float(rho_text.replace("rho_", ""))
        seed = int(seed_text.replace("seed_", ""))
        summaries[(rho, seed)] = summary
    return summaries


def _removal_from_aggregated(payload: dict) -> dict[int, list[dict]]:
    removal = {}
    for key, removal_payload in payload["main"]["removal"].items():
        _, seed_text, _ = key.split("/", 2)
        seed = int(seed_text.replace("seed_", ""))
        removal.setdefault(seed, []).append(removal_payload)
    return removal


def _style_panel_card(ax: plt.Axes) -> None:
    ax.set_facecolor("#FBFBFB")
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            facecolor="#FBFBFB",
            edgecolor="#E3E5E8",
            linewidth=0.6,
            zorder=-10,
            clip_on=False,
        )
    )


def _pair_key(
    first: tuple[str, str],
    second: tuple[str, str],
) -> str:
    return f"pair:{first[0]}:{first[1]}__vs__{second[0]}:{second[1]}"


def _draw_ranking_network(
    ax: plt.Axes,
    summaries: dict[tuple[float, int], dict],
) -> None:
    configurations = [
        (method, behavior)
        for method in METHODS
        for behavior in BEHAVIORS
    ]
    pairs = []
    taus = []
    for first_index, first in enumerate(configurations):
        for second in configurations[first_index + 1 :]:
            key = _pair_key(first, second)
            values = [
                summaries[(rho, seed)]["ranking_agreement"][key]
                for rho in RANKING_RHOS
                for seed in SEEDS
            ]
            pairs.append((first, second))
            taus.append(float(np.mean(values)))

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        len(configurations),
        endpoint=False,
    )
    positions = np.column_stack((np.cos(angles), np.sin(angles)))

    for (first, second), value in zip(pairs, taus):
        if 0.0 <= value < 0.4:
            continue

        first_index = configurations.index(first)
        second_index = configurations.index(second)
        if value >= 0:
            normalized = np.clip((value - 0.4) / 0.5, 0.0, 1.0)
            strength = normalized * normalized * (3.0 - 2.0 * normalized)
            color = "#7A4F9E"
            alpha = 0.30 + 0.58 * strength
            linewidth = 1 + 0.85 * strength
            linestyle = "-"
        else:
            color = "#777777"
            alpha = 0.44 + 0.52 * abs(value) ** 2
            linewidth = 1 + 1.1 * abs(value)
            linestyle = "--"

        ax.plot(
            [positions[first_index, 0], positions[second_index, 0]],
            [positions[first_index, 1], positions[second_index, 1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=1,
        )

    for index, (method, behavior) in enumerate(configurations):
        ax.scatter(
            positions[index, 0],
            positions[index, 1],
            s=100,
            marker=BEHAVIOR_MARKERS[behavior],
            color=NETWORK_METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )

        label_position = positions[index] * 1.16
        horizontal = "left" if label_position[0] >= 0 else "right"
        vertical = "bottom" if label_position[1] > 0.05 else "top"
        if abs(label_position[1]) <= 0.05:
            vertical = "center"
        ax.text(
            label_position[0],
            label_position[1],
            f"{METHOD_LABELS[method]}\n{BEHAVIOR_LABELS[behavior]}",
            ha=horizontal,
            va=vertical,
            fontsize=8.0,
            zorder=5,
            path_effects=[
                patheffects.withStroke(linewidth=1.1, foreground="white")
            ],
        )

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.60, 1.60)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()

    edge_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#7A4F9E",
            linewidth=2.2,
            linestyle="-",
            label=r"Agreement ($\tau \geq 0.4$)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#777777",
            linewidth=0.9,
            linestyle="--",
            label=r"Disagreement ($\tau < 0$)",
        ),
    ]
    ax.legend(
        handles=edge_handles,
        loc="lower center",
        bbox_to_anchor=(0.55, -0.14),
        ncol=2,
        fontsize=9,
    )


def _select_removal_rows(
    removal: dict[int, list[dict]],
    method: str,
    behavior: str,
    *,
    fraction: float,
) -> np.ndarray:
    selected = []
    for seed in SEEDS:
        rows = [
            row
            for payload in removal[seed]
            for row in payload["results"].values()
            if row["method"] == method
            and row["behavior"] == behavior
            and abs(float(row["fraction"]) - fraction) < 1e-12
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one removal result for {method}/{behavior} at "
                f"fraction={fraction} in seed {seed}, found {len(rows)}"
            )
        selected.append(rows[0])

    return np.asarray([
        (
            row["retrained_test_accuracy"]
            - row["factual_test_accuracy"]
        ) * 100.0
        for row in selected
    ])


def _draw_removal(ax: plt.Axes, removal: dict[int, list[dict]]) -> None:
    _style_panel_card(ax)
    removal_fraction = 0.10

    groups = []
    group_labels = []
    group_types = []
    for behavior in BEHAVIORS:
        groups.append([
            _select_removal_rows(
                removal,
                method,
                behavior,
                fraction=removal_fraction,
            )
            for method in METHODS
        ])
        group_labels.append(BEHAVIOR_LABELS[behavior])
        group_types.append("method")

    groups.append([
        _select_removal_rows(
            removal,
            method_key,
            "behavior_free",
            fraction=removal_fraction,
        )
        for method_key, _ in BASELINE_SPECS
    ])
    group_labels.append("N/A")
    group_types.append("baseline")

    group_positions = np.arange(len(groups), dtype=float)
    bar_width = 0.20
    offsets = (-bar_width, 0.0, bar_width)

    method_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=METHOD_COLORS[method],
            edgecolor="#2B3036",
            linewidth=0.4,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    baseline_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=BASELINE_COLORS[method_key],
            edgecolor="#2B3036",
            linewidth=0.4,
            label=label,
        )
        for method_key, label in BASELINE_SPECS
    ]

    for group_index, (group, group_type) in enumerate(zip(groups, group_types)):
        colors = (
            [METHOD_COLORS[method] for method in METHODS]
            if group_type == "method"
            else [BASELINE_COLORS[key] for key, _ in BASELINE_SPECS]
        )
        for bar_index, (values, color) in enumerate(zip(group, colors)):
            mean = values.mean()
            error = values.std(ddof=1) if len(values) > 1 else 0.0
            position = group_positions[group_index] + offsets[bar_index]
            ax.bar(
                position,
                mean,
                yerr=error,
                color=color,
                width=bar_width,
                edgecolor="#2B3036",
                linewidth=0.45,
                error_kw={
                    "ecolor": "#2B3036",
                    "elinewidth": 0.60,
                    "capsize": 1.4,
                    "capthick": 0.60,
                },
                zorder=2,
            )
    ax.axhline(0.0, color="#444444", linewidth=0.7, zorder=3)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(-3.5, 4.2)
    ax.set_ylabel(r"$\Delta$ test accuracy (pp)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.35, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(
        handles=method_handles + baseline_handles,
        loc="lower left",
        frameon=False,
        ncol=2,
        handlelength=1.1,
        columnspacing=0.8,
    )
def _target_logit_auprc(run: dict, method: str) -> float:
    return run["detection"][method]["target_logit"]["auprc"]


def _architecture_runs(payload: dict, architecture: str) -> list[dict]:
    if architecture == "resnet18":
        return [
            payload["main"]["runs"][f"rho_0.2/seed_{seed}"]
            for seed in SEEDS
        ]
    return [
        payload["architecture"]["runs"][f"{architecture}/seed_{seed}"]
        for seed in SEEDS
    ]


def _architecture_behavior_auprc(
    payload: dict,
    architecture: str,
    method: str,
    behavior: str,
) -> np.ndarray:
    return np.asarray([
        run["detection"][method][behavior]["auprc"]
        for run in _architecture_runs(payload, architecture)
    ])


def _draw_robustness(
    ax_noise: plt.Axes,
    ax_architecture: plt.Axes,
    payload: dict,
) -> None:
    _style_panel_card(ax_noise)
    for method in METHODS:
        values = np.asarray([
            [
                _target_logit_auprc(
                    payload["main"]["runs"][f"rho_{rho}/seed_{seed}"],
                    method,
                )
                for seed in SEEDS
            ]
            for rho in RHOS
        ])
        mean = values.mean(axis=1)
        std = values.std(axis=1, ddof=1)
        color = METHOD_COLORS[method]

        ax_noise.fill_between(
            RHOS,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax_noise.plot(
            RHOS,
            mean,
            color=color,
            marker="o",
            markersize=3.0,
            linewidth=1.25,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=METHOD_LABELS[method],
        )

    ax_noise.set_ylabel("AUPRC")
    ax_noise.set_xlim(0.035, 0.415)
    ax_noise.set_xticks((0.1, 0.2, 0.3, 0.4))
    ax_noise.set_xticklabels(("0.1", "0.2", "0.3", "0.4"))
    ax_noise.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.1f}")
    )
    ax_noise.grid(axis="y", color="#DDDDDD", linewidth=0.35, alpha=0.65)
    ax_noise.set_axisbelow(True)
    ax_noise.legend(loc="upper left", frameon=False)

    ax_noise.text(
        0.7,
        0.1,
        r"Noise rate $\rho$",
        transform=ax_noise.transAxes,
        ha="left",
        va="center",
    )

    _style_panel_card(ax_architecture)
    specification_methods = ("tracin", "lissa_if")
    specification_behaviors = ("negative_loss", "target_logit")
    bar_width = 0.05
    architecture_offsets = np.linspace(-0.10, 0.10, len(ARCHITECTURES))
    method_positions = np.asarray((-0.22, 0.22, 0.86, 1.30))
    behavior_centers = (0.0, 1.08)

    group_index = 0
    for behavior in specification_behaviors:
        for method in specification_methods:
            method_center = method_positions[group_index]
            for architecture_index, architecture in enumerate(ARCHITECTURES):
                values = _architecture_behavior_auprc(
                    payload,
                    architecture,
                    method,
                    behavior,
                )
                ax_architecture.bar(
                    method_center + architecture_offsets[architecture_index],
                    values.mean(),
                    yerr=values.std(ddof=1),
                    color=ARCHITECTURE_COLORS[architecture],
                    width=bar_width,
                    edgecolor="#2B3036",
                    linewidth=0.45,
                    error_kw={
                        "ecolor": "#2B3036",
                        "elinewidth": 0.60,
                        "capsize": 1.2,
                        "capthick": 0.60,
                    },
                    zorder=2,
                )
            group_index += 1

    ax_architecture.set_xticks(method_positions)
    ax_architecture.set_xticklabels(
        ("TracIn", "LiSSA", "TracIn", "LiSSA")
    )
    for behavior, center in zip(specification_behaviors, behavior_centers):
        ax_architecture.text(
            center,
            -0.24,
            BEHAVIOR_LABELS[behavior],
            transform=ax_architecture.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11.5,
        )
    ax_architecture.set_ylabel("AUPRC")
    ax_architecture.set_xlim(-0.38, 1.46)
    ax_architecture.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.1f}")
    )
    ax_architecture.grid(axis="y", color="#DDDDDD", linewidth=0.35, alpha=0.65)
    ax_architecture.set_axisbelow(True)
    architecture_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=ARCHITECTURE_COLORS[architecture],
            edgecolor="#2B3036",
            linewidth=0.4,
            label=ARCHITECTURE_LABELS[architecture],
        )
        for architecture in ARCHITECTURES
    ]
    ax_architecture.legend(
        handles=architecture_handles,
        loc="upper left",
        ncol=3,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.8,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the noisy-label figure from the aggregate"
    )
    parser.add_argument(
        "--aggregated",
        type=Path,
        default=DEFAULT_AGGREGATED,
        help="Path to aggregated_results.json",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory for generated PDF and PNG figures",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = _load_json(args.aggregated)
    for family in ("main", "architecture"):
        block = payload[family]
        if not block.get("included"):
            raise ValueError(f"The aggregate does not include the {family} family")
        if not block["completeness"]["complete"]:
            raise ValueError(f"The aggregate {family} family is incomplete")
    summaries = _summaries_from_aggregated(payload)
    removal = _removal_from_aggregated(payload)

    fig = plt.figure(figsize=(15, 4))
    outer_grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 1.0),
        wspace=0.35,
    )

    ranking_ax = fig.add_subplot(outer_grid[0, 0])
    _draw_ranking_network(ranking_ax, summaries)
    ranking_ax.set_title(
        "(a) Ranking agreement",
        pad=10,
        fontweight="bold",
    )

    removal_ax = fig.add_subplot(outer_grid[0, 1])
    _draw_removal(removal_ax, removal)
    removal_ax.set_title(
        "(b) Removal utility",
        pad=6,
        fontweight="bold",
    )

    robustness_grid = outer_grid[0, 2].subgridspec(
        2,
        1,
        height_ratios=(1.25, 1.0),
    )
    noise_ax = fig.add_subplot(robustness_grid[0])
    architecture_ax = fig.add_subplot(robustness_grid[1])
    _draw_robustness(noise_ax, architecture_ax, payload)
    noise_ax.set_title("(c) Robustness", pad=6, fontweight="bold")

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.figure_dir / "fig_noisy_label"
    for extension in ("pdf", "png"):
        fig.savefig(
            output_base.with_suffix(f".{extension}"),
            bbox_inches="tight",
        )
    plt.close(fig)

    print(f"Saved figure to {output_base}.pdf")
    print(f"Saved figure to {output_base}.png")


if __name__ == "__main__":
    main()
