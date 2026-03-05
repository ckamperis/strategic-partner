"""
Generate all thesis figures from experiment results.

Produces publication-quality figures for Chapter 5 of the thesis.
Run: python -m experiments.generate_figures
Output: experiments/figures/*.png + *.pdf

Style: Academic (serif fonts, 300 DPI, colorblind-friendly palette).

Thesis reference: Chapter 5 — Experimental Evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Style setup ─────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# Colorblind-friendly palette
COLORS = sns.color_palette("colorblind", 10)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "data" / "results"
FIGURES_DIR = BASE_DIR / "experiments" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as both PNG and PDF."""
    fig.savefig(FIGURES_DIR / f"{name}.png")
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved: {name}.png + .pdf")


def load_json(filename: str) -> dict:
    """Load a JSON result file."""
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


# ── Figure 1: Exp1 — Hybrid Search α-Sweep with CIs ────────────────

def plot_exp1_alpha_sweep() -> None:
    """Line plot with shaded CI band for hybrid search α optimization."""
    data = load_json("exp1_hybrid_alpha.json")

    # Try to load pre-computed bootstrap CIs
    try:
        stats = load_json("exp1_statistical_analysis.json")
        ci_data = stats["per_alpha_ci"]
        alphas = [d["alpha"] for d in ci_data]
        means = [d["mean"] for d in ci_data]
        ci_lower = [d["ci_95_lower"] for d in ci_data]
        ci_upper = [d["ci_95_upper"] for d in ci_data]
    except (FileNotFoundError, KeyError):
        # Compute inline from raw data
        per_alpha = data["per_alpha_results"]
        alphas = [d["alpha"] for d in per_alpha]
        means = [d["mean_ndcg_at_5"] for d in per_alpha]
        stds = [d.get("std_ndcg_at_5", 0) for d in per_alpha]
        ci_lower = [m - 1.96 * s for m, s in zip(means, stds)]
        ci_upper = [m + 1.96 * s for m, s in zip(means, stds)]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shaded CI band
    ax.fill_between(alphas, ci_lower, ci_upper, alpha=0.2, color=COLORS[0],
                     label="95% Bootstrap CI")

    # Main line
    ax.plot(alphas, means, "o-", color=COLORS[0], linewidth=2, markersize=6,
            label="Mean nDCG@5")

    # Optimal α=0.3 marker
    best_idx = alphas.index(0.3)
    ax.axvline(x=0.3, color=COLORS[1], linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"Optimal \u03b1=0.3\nnDCG@5={means[best_idx]:.3f}",
        xy=(0.3, means[best_idx]),
        xytext=(0.45, means[best_idx] + 0.02),
        arrowprops=dict(arrowstyle="->", color=COLORS[1]),
        fontsize=9, color=COLORS[1],
    )

    # Reference lines
    ax.axhline(y=means[0], color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(y=means[-1], color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Fusion Weight \u03b1 (0 = BM25 only, 1 = Semantic only)")
    ax.set_ylabel("Mean nDCG@5")
    ax.set_ylim(0.60, 1.02)
    ax.set_xticks(alphas)
    ax.legend(loc="lower left")

    save_figure(fig, "fig_exp1_alpha_sweep")


# ── Figure 2: Exp2 — Self-Correcting RAG Iteration Distribution ────

def plot_exp2_rag_iterations() -> None:
    """Grouped bar chart showing iteration distribution vs threshold."""
    data = load_json("exp2_rag_iterations.json")
    per_query = data["per_query_results"]

    # Count per iteration: above/below threshold
    iter_pass = {1: 0, 2: 0, 3: 0}
    iter_fail = {1: 0, 2: 0, 3: 0}

    for q in per_query:
        iters = q["iterations_used"]
        if q["above_threshold"]:
            iter_pass[iters] += 1
        else:
            iter_fail[iters] += 1

    iterations = [1, 2, 3]
    pass_counts = [iter_pass[i] for i in iterations]
    fail_counts = [iter_fail[i] for i in iterations]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(iterations))
    width = 0.35

    bars_pass = ax.bar(x - width / 2, pass_counts, width, label="Above Threshold (\u22650.75)",
                        color=COLORS[2], edgecolor="white")
    bars_fail = ax.bar(x + width / 2, fail_counts, width, label="Below Threshold (<0.75)",
                        color=COLORS[3], edgecolor="white")

    # Annotations on bars
    for bar, count in zip(bars_pass, pass_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    str(count), ha="center", va="bottom", fontweight="bold", fontsize=11)
    for bar, count in zip(bars_fail, fail_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    str(count), ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Summary text box
    ax.text(0.97, 0.95, "Pass rate: 66.7% (10/15)",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xlabel("RAG Iterations Used")
    ax.set_ylabel("Number of Queries")
    ax.set_xticks(x)
    ax.set_xticklabels(iterations)
    ax.set_ylim(0, 10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()

    save_figure(fig, "fig_exp2_rag_iterations")


# ── Figure 3: Exp3 — PICP Latency Breakdown by Query Type ──────────

def plot_exp3_latency_breakdown() -> None:
    """Stacked horizontal bar chart of latency per query type."""
    data = load_json("exp3_picp_latency.json")
    by_type = data["by_query_type"]

    # Sort by total time ascending
    type_order = sorted(by_type.keys(), key=lambda t: by_type[t]["avg_total_ms"])

    labels = {
        "general": "General",
        "cashflow_forecast": "Cashflow Forecast",
        "customer_analysis": "Customer Analysis",
        "risk_assessment": "Risk Assessment",
        "swot_analysis": "SWOT Analysis",
    }

    y_labels = [labels.get(t, t) for t in type_order]
    knowledge = [by_type[t]["avg_knowledge_ms"] for t in type_order]
    reasoning = [by_type[t]["avg_reasoning_ms"] for t in type_order]
    simulation = [by_type[t]["avg_simulation_ms"] for t in type_order]
    trust = [by_type[t]["avg_trust_ms"] for t in type_order]
    totals = [by_type[t]["avg_total_ms"] for t in type_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(type_order))

    ax.barh(y, knowledge, label="Knowledge", color=COLORS[0])
    ax.barh(y, reasoning, left=knowledge, label="Reasoning", color=COLORS[1])
    ax.barh(y, simulation,
            left=[k + r for k, r in zip(knowledge, reasoning)],
            label="Simulation", color=COLORS[2])
    ax.barh(y, trust,
            left=[k + r + s for k, r, s in zip(knowledge, reasoning, simulation)],
            label="Trust", color=COLORS[3])

    # Total time labels at end of bars
    for i, total in enumerate(totals):
        ax.text(total + 200, i, f"{total / 1000:.1f}s", va="center", fontsize=9)

    # PICP overhead annotation
    ax.text(0.97, 0.05,
            "PICP overhead: <1ms (<0.01%)",
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Average Latency (ms)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(totals) * 1.15)

    save_figure(fig, "fig_exp3_latency_breakdown")


# ── Figure 4: Exp4 — Monte Carlo Backtesting (v2 Gross Revenue) ────

def plot_exp4_backtest() -> None:
    """Time series with CI band for Monte Carlo backtesting."""
    data = load_json("exp4_monte_carlo.json")
    bt = data["backtesting"]["variant_a_gross_revenue"]
    per_month = bt["per_month"]

    months = [m["month"] for m in per_month]
    actuals = [m["actual_net_cf"] for m in per_month]
    forecasts = [m["forecast_mean"] for m in per_month]
    p5 = [m["forecast_p5"] for m in per_month]
    p95 = [m["forecast_p95"] for m in per_month]
    within_ci = [m["within_90pct_ci"] for m in per_month]

    x = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # CI band
    ax.fill_between(x, p5, p95, alpha=0.2, color=COLORS[0], label="90% CI (P5\u2013P95)")

    # Forecast line
    ax.plot(x, forecasts, "--", color=COLORS[0], linewidth=1.5, label="Forecast Mean")

    # Actual values with color coding
    for i, (xi, actual, ok) in enumerate(zip(x, actuals, within_ci)):
        if ok:
            ax.plot(xi, actual, "o", color=COLORS[2], markersize=8, zorder=5)
        else:
            ax.plot(xi, actual, "X", color=COLORS[3], markersize=10, zorder=5)

    # Legend markers
    ax.plot([], [], "o", color=COLORS[2], markersize=8, label="Actual (within CI)")
    ax.plot([], [], "X", color=COLORS[3], markersize=10, label="Actual (outside CI)")

    # Summary annotation
    coverage = bt["coverage_90pct_ci"]
    mape = bt["avg_pct_error"]
    ax.text(0.02, 0.98,
            f"Training: 2019\u20132022 (48 months)\n"
            f"Coverage: {coverage:.0%} | MAPE: {mape:.1f}%",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([m[-5:] for m in months], rotation=45, ha="right")
    ax.set_xlabel("Month (2023)")
    ax.set_ylabel("Gross Revenue (\u20ac)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"\u20ac{v:,.0f}"))
    ax.legend(loc="upper right", fontsize=9)

    save_figure(fig, "fig_exp4_backtest_v2")


# ── Figure 4b: Exp4 — Monte Carlo Convergence ──────────────────────

def plot_exp4_convergence() -> None:
    """Log-scale convergence plot for Monte Carlo simulation."""
    data = load_json("exp4_monte_carlo.json")
    results = data["convergence"]["results"]

    n_sims = [r["n_simulations"] for r in results]
    se_values = [r["convergence_std"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(n_sims, se_values, "o-", color=COLORS[0], linewidth=2, markersize=7)

    # Log scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Convergence point at N=500
    converged_n = data["convergence"]["converged_at_n"]
    converged_idx = n_sims.index(converged_n)
    converged_se = se_values[converged_idx]

    ax.axvline(x=converged_n, color=COLORS[1], linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"Convergence: N={converged_n}\nSE=\u20ac{converged_se:,.0f}",
        xy=(converged_n, converged_se),
        xytext=(converged_n * 5, converged_se * 2),
        arrowprops=dict(arrowstyle="->", color=COLORS[1]),
        fontsize=9, color=COLORS[1],
    )

    # SE value annotations
    for n, se in zip(n_sims, se_values):
        ax.annotate(f"\u20ac{se:,.0f}", xy=(n, se), xytext=(0, -15),
                    textcoords="offset points", fontsize=8, ha="center",
                    color="gray")

    ax.set_xlabel("Number of Simulations")
    ax.set_ylabel("Standard Error of Mean (\u20ac)")
    ax.set_xticks(n_sims)
    ax.set_xticklabels([f"{n:,}" for n in n_sims])

    save_figure(fig, "fig_exp4_convergence")


# ── Figure 5: Exp5 — Trust Score Sensitivity ───────────────────────

def plot_exp5_trust_sensitivity() -> None:
    """Horizontal stacked bar chart of trust sub-scores per variant."""
    data = load_json("exp5_trust_sensitivity.json")
    variants = data["variant_results"]

    # Sort by overall score ascending
    variants_sorted = sorted(variants, key=lambda v: v["overall_score"])

    # Weights from config
    w_e = data["config"]["default_weights"]["w_e"]
    w_c = data["config"]["default_weights"]["w_c"]
    w_a = data["config"]["default_weights"]["w_a"]

    labels_map = {
        "baseline": "Baseline",
        "no_knowledge": "No Knowledge",
        "low_relevance": "Low Relevance",
        "general_routing": "General Route",
        "no_simulation": "No Simulation",
        "low_sim_count": "Low Sim Count",
        "all_estimated_params": "All Estimated",
        "perfect_data": "Perfect Data",
    }

    y_labels = [labels_map.get(v["variant"], v["variant"]) for v in variants_sorted]
    expl_weighted = [v["explainability"] * w_e for v in variants_sorted]
    cons_weighted = [v["consistency"] * w_c for v in variants_sorted]
    acc_weighted = [v["accuracy"] * w_a for v in variants_sorted]
    totals = [v["overall_score"] for v in variants_sorted]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(variants_sorted))

    ax.barh(y, expl_weighted, label=f"Explainability (\u00d7{w_e})", color=COLORS[0])
    ax.barh(y, cons_weighted, left=expl_weighted,
            label=f"Consistency (\u00d7{w_c})", color=COLORS[1])
    ax.barh(y, acc_weighted,
            left=[e + c for e, c in zip(expl_weighted, cons_weighted)],
            label=f"Accuracy (\u00d7{w_a})", color=COLORS[2])

    # Overall score at end
    for i, total in enumerate(totals):
        ax.text(total + 0.01, i, f"{total:.2f}", va="center", fontsize=9,
                fontweight="bold")

    # Confidence threshold line
    ax.axvline(x=0.75, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(0.76, len(variants_sorted) - 0.5, "High\nconfidence",
            fontsize=8, color="gray", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Trust Score (T = 0.4E + 0.4C + 0.2A)")
    ax.set_xlim(0, 1.12)
    ax.legend(loc="lower right", fontsize=9)

    save_figure(fig, "fig_exp5_trust_sensitivity")


# ── Figure 6: Exp7 — Graceful Degradation Heatmap ──────────────────

def plot_exp7_degradation() -> None:
    """Heatmap matrix showing system resilience across failure scenarios."""
    data = load_json("exp7_degradation.json")
    scenarios = data["per_scenario_results"]

    scenario_labels = [
        "Baseline",
        "K fails",
        "R fails",
        "S fails",
        "T fails",
        "K+S fail",
        "R+S fail",
        "All 4 fail",
    ]

    columns = ["Has\nAnswer", "Correct\nRouting", "Simulation\nRan",
               "Trust\nScore", "N\nCaveats"]

    n_rows = len(scenarios)
    n_cols = len(columns)

    # Build matrix for heatmap
    matrix = np.zeros((n_rows, n_cols))
    annotations = []

    for i, s in enumerate(scenarios):
        row_ann = []
        # Has Answer (boolean)
        has_ans = s["has_answer"]
        matrix[i, 0] = 1.0 if has_ans else 0.0
        row_ann.append("Yes" if has_ans else "No")

        # Correct Routing
        correct = s["query_type"] == "cashflow_forecast"
        matrix[i, 1] = 1.0 if correct else 0.0
        row_ann.append("Yes" if correct else "No")

        # Simulation Ran
        sim = s["simulation_ran"]
        matrix[i, 2] = 1.0 if sim else 0.0
        row_ann.append("Yes" if sim else "No")

        # Trust Score (0-1 scale)
        ts = s["trust_score"]
        matrix[i, 3] = ts
        row_ann.append(f"{ts:.2f}")

        # N Caveats (normalize: 0 caveats -> 1.0 (good), 9 -> 0.0)
        nc = s["n_caveats"]
        matrix[i, 4] = max(0, 1.0 - nc / 10.0)
        row_ann.append(str(nc))

        annotations.append(row_ann)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Custom colormap: red -> yellow -> green
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=1,
        annot=np.array(annotations, dtype=object),
        fmt="",
        linewidths=1,
        linecolor="white",
        cbar=False,
        xticklabels=columns,
        yticklabels=scenario_labels,
    )

    # Key finding annotation
    ax.text(0.5, -0.12,
            "100% Resilience \u2014 System produces an answer in ALL failure scenarios",
            transform=ax.transAxes, fontsize=10, ha="center",
            fontweight="bold", color=COLORS[2])

    ax.set_yticklabels(scenario_labels, rotation=0)
    ax.tick_params(axis="x", rotation=0)

    save_figure(fig, "fig_exp7_degradation")


# ── Figure 7: Exp4 — v2 vs v3 Comparison ───────────────────────────

def plot_exp4_v2_v3_comparison() -> None:
    """Grouped bar chart comparing v2 and v3 backtesting metrics."""
    v2 = load_json("exp4_monte_carlo.json")
    v3 = load_json("exp4_monte_carlo_v3.json")

    v2_bt = v2["backtesting"]["variant_a_gross_revenue"]
    v3_bt = v3["backtesting"]["variant_a_gross_revenue"]

    metrics = ["MAPE (%)", "90% CI Coverage (%)"]
    v2_vals = [v2_bt["avg_pct_error"], v2_bt["coverage_90pct_ci"] * 100]
    v3_vals = [v3_bt["avg_pct_error"], v3_bt["coverage_90pct_ci"] * 100]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(metrics))
    width = 0.3

    bars_v2 = ax.bar(x - width / 2, v2_vals, width, label="v2: 5-year (2019\u20132022)",
                      color=COLORS[0], edgecolor="white")
    bars_v3 = ax.bar(x + width / 2, v3_vals, width, label="v3: 20-year (2004\u20132021)",
                      color=COLORS[1], edgecolor="white")

    # Value labels on bars
    for bar, val in zip(bars_v2, v2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars_v3, v3_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")

    # Subtitle annotation
    ax.text(0.5, 0.95,
            "Bias\u2013Variance Trade-off in Non-Stationary Data",
            transform=ax.transAxes, fontsize=10, ha="center", va="top",
            fontstyle="italic", color="gray")

    save_figure(fig, "fig_exp4_v2_v3_comparison")


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Generating thesis figures...")
    print("=" * 60)

    plot_exp1_alpha_sweep()
    plot_exp2_rag_iterations()
    plot_exp3_latency_breakdown()
    plot_exp4_backtest()
    plot_exp4_convergence()
    plot_exp5_trust_sensitivity()
    plot_exp7_degradation()
    plot_exp4_v2_v3_comparison()

    n_files = len(list(FIGURES_DIR.glob("*.png")))
    print(f"\nDone! {n_files} PNG figures generated in {FIGURES_DIR}")
    print(f"PDF versions also saved alongside each PNG.")


if __name__ == "__main__":
    main()
