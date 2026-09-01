"""Aggregate multi-seed experiment results and report mean +/- std statistics.

Reads summary.json files from outputs/exp1_behavior, outputs/exp2_transition,
and outputs/exp3_stepsize, and produces aggregated statistics with 95%
confidence intervals across seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUTPUT_BASE = Path(__file__).resolve().parent.parent / "outputs" / "fashion_mnist"


def _mean_std_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    # 95% CI half-width with t-distribution approximation (t ~ 2.262 for df=9)
    from scipy import stats
    t_crit = stats.t.ppf(0.975, df=array.size - 1) if array.size > 1 else float("nan")
    ci95 = float(t_crit * std / np.sqrt(array.size)) if array.size > 1 else float("nan")
    return {"mean": mean, "std": std, "ci95": ci95, "n": int(array.size)}


def _load_summaries(directory: Path) -> list[dict]:
    summaries = []
    for seed_dir in sorted(directory.glob("seed_*")):
        summary_path = seed_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open() as stream:
                summaries.append(json.load(stream))
    return summaries


def aggregate_exp1() -> dict:
    """Aggregate behavior-mismatch results across seeds."""
    summaries = _load_summaries(OUTPUT_BASE / "exp1_behavior")
    result: dict = {"n_seeds": len(summaries), "fit": {}, "behavior_disagreement": {}, "approximation": {}}

    for key in ("train_accuracy", "test_accuracy"):
        result["fit"][key] = _mean_std_ci([s["fit"][key] for s in summaries])

    pair_keys = summaries[0]["one_step"]["exact_behavior_disagreement"].keys()
    for pair in pair_keys:
        result["behavior_disagreement"][pair] = {
            metric: _mean_std_ci(
                [s["one_step"]["exact_behavior_disagreement"][pair][metric] for s in summaries]
            )
            for metric in ("mean_kendall_tau", "mean_top_5pct_overlap", "mean_sign_agreement")
        }

    behaviors = summaries[0]["specification"]["behaviors"]
    for behavior in behaviors:
        result["approximation"][behavior] = {
            metric: _mean_std_ci(
                [s["one_step"]["approximation_vs_exact"][behavior][metric] for s in summaries]
            )
            for metric in ("nrmse", "kendall_tau", "sign_accuracy", "top_5pct_overlap")
        }
    return result


def aggregate_exp2() -> dict:
    """Aggregate transition-mismatch results across seeds."""
    summaries = _load_summaries(OUTPUT_BASE / "exp2_transition")
    result: dict = {"n_seeds": len(summaries), "transition_comparisons": {}, "first_order": {}}

    for pair in summaries[0]["transition_comparisons"].keys():
        result["transition_comparisons"][pair] = {
            metric: _mean_std_ci([s["transition_comparisons"][pair][metric] for s in summaries])
            for metric in ("nrmse", "kendall_tau", "sign_accuracy", "top_5pct_overlap")
        }

    for transition in summaries[0]["first_order_approximations"].keys():
        result["first_order"][transition] = {
            metric: _mean_std_ci([s["first_order_approximations"][transition][metric] for s in summaries])
            for metric in ("nrmse", "kendall_tau", "sign_accuracy", "top_5pct_overlap")
        }
    return result


def aggregate_exp3() -> dict:
    """Aggregate step-size sweep results: approximation quality vs eta."""
    base = OUTPUT_BASE / "exp3_stepsize"
    result: dict = {"etas": {}}
    for eta_dir in sorted(base.glob("eta_*"), key=lambda p: float(p.name.split("_")[1])):
        eta = eta_dir.name.split("_")[1]
        summaries = _load_summaries(eta_dir)
        if not summaries:
            continue
        behaviors = summaries[0]["specification"]["behaviors"]
        result["etas"][eta] = {"n_seeds": len(summaries)}
        for behavior in behaviors:
            result["etas"][eta][behavior] = {
                metric: _mean_std_ci(
                    [s["one_step"]["approximation_vs_exact"][behavior][metric] for s in summaries]
                )
                for metric in ("nrmse", "kendall_tau", "sign_accuracy", "top_5pct_overlap")
            }
        # Behavior disagreement should be eta-independent in first order; track it too
        result["etas"][eta]["disagreement_negloss_vs_logit"] = _mean_std_ci(
            [
                s["one_step"]["exact_behavior_disagreement"]["negative_loss_vs_target_logit"]["mean_kendall_tau"]
                for s in summaries
            ]
        )
    return result


def aggregate_exp4() -> dict:
    """Aggregate the matched-evaluation matrix across seeds.

    Returns per-cell mean tau (with 95% CI), matched-vs-mismatched summaries,
    and the per-reference top estimator computed from the seed-mean matrix
    (used to read off leaderboard reversals). Safe to call before Experiment 4
    has been run: returns ``{"n_seeds": 0}`` when no outputs exist.
    """

    summaries = _load_summaries(OUTPUT_BASE / "exp4_matched_matrix")
    if not summaries:
        return {"n_seeds": 0}

    result: dict = {"n_seeds": len(summaries), "matrix": {}, "matched_vs_mismatched": {}, "leaderboard_top": {}}
    rows = list(summaries[0]["matrix"].keys())
    columns = list(summaries[0]["matrix"][rows[0]].keys())

    for row in rows:
        result["matrix"][row] = {
            column: _mean_std_ci([s["matrix"][row][column]["mean_kendall_tau"] for s in summaries])
            for column in columns
        }

    for row, entry in summaries[0]["matched_vs_mismatched"].items():
        result["matched_vs_mismatched"][row] = {
            key: _mean_std_ci([s["matched_vs_mismatched"][row][key] for s in summaries])
            for key in entry.keys()
        }

    for column in columns:
        ranked = sorted(rows, key=lambda r: -result["matrix"][r][column]["mean"])
        result["leaderboard_top"][column] = [
            {"estimator": r, "mean_kendall_tau": result["matrix"][r][column]["mean"]} for r in ranked[:3]
        ]
    return result


def _fmt(stat: dict, decimals: int = 3) -> str:
    if stat["n"] == 0:
        return "n/a"
    return f"{stat['mean']:.{decimals}f} ± {stat['ci95']:.{decimals}f}"


def main() -> None:
    aggregated = {
        "exp1_behavior": aggregate_exp1(),
        "exp2_transition": aggregate_exp2(),
        "exp3_stepsize": aggregate_exp3(),
        "exp4_matched_matrix": aggregate_exp4(),
    }
    output_path = OUTPUT_BASE / "aggregated_results.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(aggregated, stream, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"Saved aggregated results to {output_path}\n")

    exp1 = aggregated["exp1_behavior"]
    print("=" * 70)
    print(f"EXPERIMENT 1: Behavior Mismatch ({exp1['n_seeds']} seeds)")
    print("=" * 70)
    print(f"Model: train acc {_fmt(exp1['fit']['train_accuracy'])}, test acc {_fmt(exp1['fit']['test_accuracy'])}")
    print("\nExact behavior disagreement (Kendall's tau, mean ± 95% CI):")
    for pair, stats in exp1["behavior_disagreement"].items():
        print(f"  {pair:42s} tau={_fmt(stats['mean_kendall_tau'])}  top5%={_fmt(stats['mean_top_5pct_overlap'])}  sign={_fmt(stats['mean_sign_agreement'])}")
    print("\nFirst-order approximation vs exact (fixed specification):")
    for behavior, stats in exp1["approximation"].items():
        print(f"  {behavior:15s} NRMSE={_fmt(stats['nrmse'])}  tau={_fmt(stats['kendall_tau'])}  top5%={_fmt(stats['top_5pct_overlap'])}")

    exp2 = aggregated["exp2_transition"]
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Transition Mismatch ({exp2['n_seeds']} seeds)")
    print("=" * 70)
    print("Exact transition disagreement (behavior=negative_loss):")
    for pair, stats in exp2["transition_comparisons"].items():
        print(f"  {pair:35s} tau={_fmt(stats['kendall_tau'])}  NRMSE={_fmt(stats['nrmse'])}  top5%={_fmt(stats['top_5pct_overlap'])}")
    print("\nFirst-order approximation quality per transition:")
    for transition, stats in exp2["first_order"].items():
        print(f"  {transition:15s} tau={_fmt(stats['kendall_tau'])}  NRMSE={_fmt(stats['nrmse'])}")

    exp3 = aggregated["exp3_stepsize"]
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Step-Size Sweep (first-order tau vs eta, 10 seeds each)")
    print("=" * 70)
    exp3_behaviors = [
        key
        for key in next(iter(exp3["etas"].values())).keys()
        if key not in ("n_seeds", "disagreement_negloss_vs_logit")
    ] if exp3["etas"] else []
    header = f"  {'eta':>6s}  " + "  ".join(f"{b:>22s}" for b in exp3_behaviors)
    print(header)
    for eta, stats in exp3["etas"].items():
        row = f"  {eta:>6s}  "
        row += "  ".join(f"{_fmt(stats[b]['kendall_tau']):>22s}" for b in exp3_behaviors)
        print(row)
    print("\nNRMSE (negative_loss) vs eta:")
    for eta, stats in exp3["etas"].items():
        print(f"  eta={eta:>5s}: NRMSE={_fmt(stats['negative_loss']['nrmse'])}  disagreement(negloss vs logit) tau={_fmt(stats['disagreement_negloss_vs_logit'])}")

    exp4 = aggregated["exp4_matched_matrix"]
    if exp4["n_seeds"]:
        print("\n" + "=" * 70)
        print(f"EXPERIMENT 4: Matched Evaluation Matrix ({exp4['n_seeds']} seeds)")
        print("=" * 70)
        print("Matched vs mismatched (mean per-query Kendall tau, mean ± 95% CI):")
        for row, stats in exp4["matched_vs_mismatched"].items():
            print(f"  {row:40s} matched={_fmt(stats['matched_tau'])}  mean mismatched={_fmt(stats['mean_mismatched_tau'])}")
        print("\nPer-reference leaderboard (top estimator by seed-mean tau):")
        for column, top in exp4["leaderboard_top"].items():
            if top:
                print(f"  {column:35s} {top[0]['estimator']} ({top[0]['mean_kendall_tau']:.3f})")


if __name__ == "__main__":
    main()
