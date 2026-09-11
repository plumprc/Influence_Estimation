"""Generate the ScienceQA exp1 AUPRC heatmap figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGGREGATE_PATH = (
    PROJECT_ROOT / "outputs" / "qwen3_8b" / "scienceqa_exp1" / "aggregate.json"
)
FIGURE_DIR = PROJECT_ROOT / "plot" / "figures"

METHODS = (
    "trak",
    "less",
    "gradsim",
    "tracin",
    "gradient_norm",
    "representation_similarity",
    "random",
)
READOUTS = (
    "negative_response_loss",
    "negative_explanation_loss",
    "answer_logit",
    "answer_margin",
)
TARGETS = ("answer_corruption", "rationale_corruption")

METHOD_LABELS = {
    "trak": "TRAK",
    "less": "LESS",
    "gradsim": "GradSim",
    "tracin": "TracIn",
    "gradient_norm": "Grad. norm",
    "representation_similarity": "RepSim",
    "random": "Random",
}
READOUT_LABELS = {
    "negative_response_loss": "respon.\nloss",
    "negative_explanation_loss": "explan.\nloss",
    "answer_logit": "answer\nlogit",
    "answer_margin": "answer\nmargin",
}
TARGET_TITLES = {
    "answer_corruption": "Answer corruption",
    "rationale_corruption": "Rationale corruption",
}

SOFT_REDS = LinearSegmentedColormap.from_list(
    "soft_reds",
    plt.get_cmap("Reds")(np.linspace(0.0, 0.88, 256)),
)


def _load_values() -> dict[str, np.ndarray]:
    with AGGREGATE_PATH.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("run_count") != 3:
        raise ValueError("Expected three seed runs in the exp1 aggregate")

    values: dict[str, np.ndarray] = {}
    for target in TARGETS:
        matrix = np.full((len(METHODS), len(READOUTS)), np.nan, dtype=float)
        target_payload = payload["detection"].get(target, {})
        for row, method in enumerate(METHODS):
            for column, readout in enumerate(READOUTS):
                metric = (
                    target_payload.get(method, {}).get(readout, {}).get("auprc")
                )
                if metric is None:
                    raise ValueError(
                        f"Missing AUPRC for {target}/{method}/{readout}"
                    )
                matrix[row, column] = float(metric["mean"])
        values[target] = matrix
    return values


def _draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
) -> None:
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=SOFT_REDS,
        vmin=0.0,
        vmax=0.95,
    )
    ax.set_xticks(range(len(READOUTS)))
    ax.set_xticklabels([READOUT_LABELS[readout] for readout in READOUTS])
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[method] for method in METHODS])
    ax.set_title(
        title,
        pad=9,
        fontweight="bold",
        color="#202124",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    best_index = np.unravel_index(np.argmax(matrix), matrix.shape)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.45 else "black"
            weight = "bold" if (row, column) == best_index else "normal"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8.5,
                fontweight=weight,
            )
    row, column = best_index
    ax.add_patch(
        plt.Rectangle(
            (column - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#F0E442",
            linewidth=1.6,
        )
    )
    return image


def main() -> None:
    values = _load_values()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 160,
    })

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 3.9),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(wspace=0.10)
    images = [
        _draw_heatmap(axes[index], values[target], TARGET_TITLES[target])
        for index, target in enumerate(TARGETS)
    ]
    axes[0].set_ylabel("Method")

    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("AUPRC")
    colorbar.outline.set_visible(False)

    output_stem = FIGURE_DIR / "scienceqa_exp1"
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_stem}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
