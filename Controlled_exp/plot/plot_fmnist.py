"""Generate the Section 4 combined figure from raw scores + aggregated results.

One row, three square panels:
  (a) per-query estimand disagreement - violins, behavior axis vs transition axis
  (b) approximation error, fixed spec - first-order tau vs eta per behavior
  (c) matched evaluation              - estimator x reference heatmap (neg. loss)

Panel (a) pools per-query taus across all seeds from raw scores.npz files
(cached in plot/figures/.panel_a_cache.npz because the Kendall passes are slow).
Panels (b) and (c) read outputs/aggregated_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

CODE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = CODE_DIR / "plot" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE = FIG_DIR / ".panel_a_cache.npz"

AGG_PATH = CODE_DIR / "outputs" / "fashion_mnist" / "aggregated_results.json"
if not AGG_PATH.exists():
    AGG_PATH = CODE_DIR / "outputs" / "fashion_mnist" / "aggregated_results_backup.json"
with AGG_PATH.open() as stream:
    AGG = json.load(stream)

# Okabe-Ito colorblind-safe palette
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
CMAP = "YlGnBu"

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


def _per_query_taus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([stats.kendalltau(a[q], b[q]).statistic for q in range(a.shape[0])])


def load_panel_a_data() -> dict[str, np.ndarray]:
    """Per-query taus pooled over all seeds (cached).

    The cache is invalidated when the number of query rows in the underlying
    Exp 1/2 score files changes, so resumed or partially migrated runs cannot
    silently mix seeds from different batches.
    """
    pools: dict[str, list[np.ndarray]] = {
        "soft_margin": [], "hard_margin": [], "target_logit": [],
        "multi_step": [], "inverse_hessian": [],
    }
    exp1_dirs = sorted((CODE_DIR / "outputs" / "fashion_mnist" / "exp1_behavior").glob("seed_*"))
    exp2_dirs = sorted((CODE_DIR / "outputs" / "fashion_mnist" / "exp2_transition").glob("seed_*"))
    exp1_queries = sum(int(np.load(path / "scores.npz")["query_indices"].shape[0]) for path in exp1_dirs)
    exp2_queries = sum(int(np.load(path / "scores.npz")["query_indices"].shape[0]) for path in exp2_dirs)
    expected_sizes = {
        "soft_margin": exp1_queries,
        "hard_margin": exp1_queries,
        "target_logit": exp1_queries,
        "multi_step": exp2_queries,
        "inverse_hessian": exp2_queries,
    }
    if CACHE.exists():
        cached = np.load(CACHE)
        if set(cached.files) == set(expected_sizes) and all(
            cached[key].shape == (expected_sizes[key],) for key in expected_sizes
        ):
            return {key: cached[key] for key in cached.files}

    for seed_dir in exp1_dirs:
        d = np.load(seed_dir / "scores.npz")
        behaviors = [str(x) for x in d["behaviors"]]
        ib = {b: i for i, b in enumerate(behaviors)}
        exact = d["exact_one_step"]
        nl = exact[:, :, ib["negative_loss"]]
        for b in ("soft_margin", "hard_margin", "target_logit"):
            pools[b].append(_per_query_taus(nl, exact[:, :, ib[b]]))
    for seed_dir in exp2_dirs:
        d = np.load(seed_dir / "scores.npz")
        one = d["exact_one_step"]
        pools["multi_step"].append(_per_query_taus(one, d["exact_multi_step"]))
        pools["inverse_hessian"].append(_per_query_taus(one, d["inverse_hessian"]))
    arrays = {k: np.concatenate(v) for k, v in pools.items()}
    np.savez_compressed(CACHE, **arrays)
    return arrays


def draw_violins(ax):
    """(a) per-query tau distributions, grouped by specification axis."""
    data = load_panel_a_data()
    behavior_entries = [
        ("soft margin", data["soft_margin"], COLORS["soft_margin"]),
        ("hard margin", data["hard_margin"], COLORS["hard_margin"]),
        ("target logit", data["target_logit"], COLORS["target_logit"]),
    ]
    transition_entries = [
        ("multi-step", data["multi_step"], "#56B4E9"),
        ("inv-Hessian", data["inverse_hessian"], "#7570B3"),
    ]
    entries = behavior_entries + transition_entries
    pos = [1, 2, 3, 4.6, 5.6]
    parts = ax.violinplot([e[1] for e in entries], positions=pos,
                          showmedians=True, widths=0.8)
    for body, (_, _, color) in zip(parts["bodies"], entries):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor("none")
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#444444")
        parts[key].set_linewidth(0.7)
    ax.axvline(3.8, color="grey", lw=0.6, ls=":")
    ax.set_xticks([2, 5.1])
    ax.set_xticklabels([r"behavior $B$", r"transition $\mathcal{T}$"])
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel(r"per-query Kendall $\tau$")
    ax.set_ylim(-0.6, 1.08)
    patch = plt.matplotlib.patches.Patch
    leg_b = ax.legend(
        handles=[patch(color=c, alpha=0.75, label=l) for l, _, c in behavior_entries],
        frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0),
        handlelength=0.9, handleheight=0.9, labelspacing=0.2, borderaxespad=0.2,
        fontsize=7)
    ax.add_artist(leg_b)
    ax.legend(
        handles=[patch(color=c, alpha=0.75, label=l) for l, _, c in transition_entries],
        frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0),
        handlelength=0.9, handleheight=0.9, labelspacing=0.2, borderaxespad=0.2,
        fontsize=7)
    ax.set_box_aspect(1)


def draw_sweep(ax):
    """(b) first-order tau vs eta, one line per behavior."""
    etas_dict = AGG["exp3_stepsize"]["etas"]
    etas = sorted(etas_dict.keys(), key=float)
    xs = [float(e) for e in etas]
    styles = {"target_logit": "-", "hard_margin": "-", "negative_loss": "-", "soft_margin": "--"}
    for behavior in ("target_logit", "hard_margin", "negative_loss", "soft_margin"):
        means = [etas_dict[e][behavior]["kendall_tau"]["mean"] for e in etas]
        cis = [etas_dict[e][behavior]["kendall_tau"]["ci95"] for e in etas]
        ax.plot(xs, means, marker="o", ms=3, lw=1.4, ls=styles[behavior],
                color=COLORS[behavior], label=BEHAVIOR_LABELS[behavior])
        ax.fill_between(xs, [m - c for m, c in zip(means, cis)],
                        [m + c for m, c in zip(means, cis)],
                        color=COLORS[behavior], alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"step size $\eta$")
    ax.set_ylabel(r"Kendall $\tau$")
    ax.set_ylim(0.4, 1.02)
    ax.set_box_aspect(1)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6,
              borderaxespad=0.1, labelspacing=0.3)


def draw_matrix(ax, fig):
    """(c) estimator-family x exact-reference heatmap, behavior = negative loss."""
    matrix = AGG["exp4_matched_matrix"]["matrix"]
    est_rows = ["first_order_one_step", "first_order_multi_step", "inverse_hessian"]
    ref_cols = ["exact_one_step", "exact_multi_step", "exact_reopt"]
    est_labels = ["first-order\none-step", "first-order\nmulti-step", "inv-Hessian"]
    ref_labels = ["one-step", "multi-step", "reopt"]
    values = np.array([
        [matrix[f"{est}:negative_loss"][f"{ref}:negative_loss"]["mean"] for ref in ref_cols]
        for est in est_rows
    ])
    im = ax.imshow(values, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(3):
        for j in range(3):
            matched = i == j
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                    color="white" if values[i, j] > 0.6 else "#1a3350",
                    fontsize=8.5, fontweight="bold" if matched else "normal")
    for i in range(3):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#D55E00", lw=1.8))
    ax.set_xticks(range(3))
    ax.set_xticklabels(ref_labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(est_labels)
    ax.set_xlabel("exact reference")
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    # colorbar in an inset so it does not shrink the heatmap axes
    cax = ax.inset_axes([1.06, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Kendall $\tau$")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), gridspec_kw={"wspace": 0.45})
    draw_violins(axes[0])
    axes[0].set_title("(a) Estimand disagreement")
    draw_sweep(axes[1])
    axes[1].set_title("(b) Approximation error under fixed $S$")
    draw_matrix(axes[2], fig)
    axes[2].set_title("(c) Matched evaluation")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_fmnist.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_fmnist")


if __name__ == "__main__":
    main()
