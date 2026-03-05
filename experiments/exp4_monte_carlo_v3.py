"""Experiment 4 v3 — Monte Carlo Convergence & Backtesting (Full Dataset).

MOTIVATION:
  v1 and v2 used ERPTransformer(recent_years=5) -> 60 months (2019-2023).
  The full ERP dataset spans 2004-2024 (~240 months, 544K transaction rows).
  v3 uses recent_years=None to leverage ALL available data:
  - ~216 training months (2004-2021) give more robust seasonal factor estimation
  - 24 test months (2022-2023) give more statistically meaningful coverage metrics
  - Multi-decade data captures long-term trends that 5 years cannot

Part A: Convergence re-run with full-dataset distributions.
Part B: Backtesting with 2004-2021 train / 2022-2023 test split, dual variants.
Part C: Comparison of v2 (60 months) vs v3 (240 months) results.

No LLM calls needed — pure NumPy computation.

Usage:
    python experiments/exp4_monte_carlo_v3.py

Produces:
    data/results/exp4_monte_carlo_v3.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import structlog

from data.pipeline.models import MonthlyData, MonthlyRecord
from data.pipeline.transformer import ERPTransformer
from pillars.simulation.distributions import (
    CashflowDistributions,
    build_scenario,
    fit_from_erp_data,
)
from pillars.simulation.monte_carlo import MonteCarloConfig, MonteCarloEngine

logger = structlog.get_logger()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# Simulation counts for convergence analysis
SIM_COUNTS = [100, 500, 1_000, 5_000, 10_000, 50_000]

# v2 reference values for comparison
V2_TRAINING_MONTHS = 48
V2_REVENUE_MEAN = 242678.49
V2_REVENUE_STD = 67430.40
V2_COVERAGE_A = 0.75
V2_MAPE_A = 25.36
V2_COVERAGE_B = 0.0
V2_MAPE_B = 80.67


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    print("=" * 60)
    print("EXPERIMENT 4 v3: Monte Carlo — Full Dataset (~240 months)")
    print(f"  Sim counts: {SIM_COUNTS}")
    print("=" * 60)

    # ── Load FULL dataset ─────────────────────────────────────
    print("\n[1/6] Loading ERP data (recent_years=None — ALL available data)...")
    transformer = ERPTransformer(recent_years=None)  # CRITICAL: use ALL data
    pipeline_result = transformer.run_pipeline(
        str(PROJECT_ROOT / "data" / "raw" / "cashflow_dataset.xlsx")
    )

    all_records = pipeline_result.monthly_data.records
    total_months = len(all_records)

    # Verify we got the full dataset
    assert total_months > 200, (
        f"Expected ~240 months with recent_years=None, got {total_months}. "
        f"Check ERPTransformer parameter."
    )

    print(f"  Total months: {total_months}")
    print(f"  Date range: {all_records[0].period_label} to {all_records[-1].period_label}")
    print(f"  Total rows processed: {pipeline_result.total_rows_processed}")

    # ── Part A: Convergence with full-dataset distributions ────
    print("\n[2/6] Running convergence analysis (full-dataset distributions)...")

    # Fit from ALL data for convergence test
    all_distributions = fit_from_erp_data(
        pipeline_result.monthly_data,
        pipeline_result.metrics,
    )
    print(f"  Revenue mean: €{all_distributions.revenue_mean:,.2f}")
    print(f"  Revenue std:  €{all_distributions.revenue_std:,.2f}")
    print(f"  Seasonal factors: {[round(f, 3) for f in all_distributions.seasonal_factors]}")

    engine = MonteCarloEngine()
    convergence_results: list[dict] = []

    for n_sims in SIM_COUNTS:
        config = MonteCarloConfig(
            n_simulations=n_sims,
            time_horizon_months=3,
            random_seed=42,
            distributions=all_distributions,
            initial_balance=0.0,
            start_month=1,
        )

        mc_result = engine.run(config, scenario_name="base")
        m1_stats = mc_result.monthly_stats[0]
        cum_final = mc_result.cumulative_stats[-1]

        row = {
            "n_simulations": n_sims,
            "month1_mean": round(m1_stats.mean, 2),
            "month1_std": round(m1_stats.std, 2),
            "month1_p5": round(m1_stats.p5, 2),
            "month1_p50": round(m1_stats.p50, 2),
            "month1_p95": round(m1_stats.p95, 2),
            "cumulative_mean": round(cum_final.mean, 2),
            "cumulative_std": round(cum_final.std, 2),
            "convergence_std": round(mc_result.convergence_std, 4),
            "probability_negative": round(mc_result.probability_negative, 4),
            "var_5pct": round(mc_result.var_5pct, 2),
            "elapsed_ms": round(mc_result.elapsed_ms, 2),
        }
        convergence_results.append(row)
        print(f"  N={n_sims:>6,d}  mean={m1_stats.mean:>10,.2f}  "
              f"conv_std={mc_result.convergence_std:>8.4f}  "
              f"elapsed={mc_result.elapsed_ms:.1f}ms")

    # Compute relative convergence (% change from N=50K reference)
    ref = convergence_results[-1]
    for row in convergence_results:
        if ref["month1_mean"] != 0:
            row["mean_pct_diff_from_ref"] = round(
                abs(row["month1_mean"] - ref["month1_mean"])
                / abs(ref["month1_mean"]) * 100, 3
            )
        else:
            row["mean_pct_diff_from_ref"] = 0.0

    # ── Part B: Backtesting — Train 2004-2021, Test 2022-2023 ──
    print("\n[3/6] Running backtesting with full-dataset train/test split...")

    # Split: train on <= 2021, test on >= 2022
    training_records = [r for r in all_records if r.year <= 2021]
    test_records = [r for r in all_records if r.year >= 2022]

    print(f"  Training months: {len(training_records)} "
          f"({training_records[0].period_label} to {training_records[-1].period_label})")
    print(f"  Test months: {len(test_records)} "
          f"({test_records[0].period_label} to {test_records[-1].period_label})")

    # Fit distributions ONLY from training data (no leakage)
    training_monthly = MonthlyData(records=training_records)
    training_metrics = pipeline_result.metrics  # Structural metrics

    training_distributions = fit_from_erp_data(training_monthly, training_metrics)
    print(f"  Training revenue mean: €{training_distributions.revenue_mean:,.2f}")
    print(f"  Training revenue std:  €{training_distributions.revenue_std:,.2f}")
    print(f"  Training seasonal factors: {[round(f, 3) for f in training_distributions.seasonal_factors]}")

    # ── Variant A: Gross Revenue Forecast ──────────────────────
    print("\n  Running Variant A (gross revenue forecast, expense_ratio=0)...")
    gross_distributions = CashflowDistributions(
        revenue_mean=training_distributions.revenue_mean,
        revenue_std=training_distributions.revenue_std,
        seasonal_factors=list(training_distributions.seasonal_factors),
        expense_ratio_mean=0.0,
        expense_ratio_std=0.0,
        collection_delay_mean=training_distributions.collection_delay_mean,
        collection_delay_std=training_distributions.collection_delay_std,
        credit_note_probability=training_distributions.credit_note_probability,
        credit_note_ratio=training_distributions.credit_note_ratio,
        customer_loss_rate=training_distributions.customer_loss_rate,
    )

    backtest_variant_a = _run_backtest(engine, gross_distributions, test_records, "gross_revenue")

    # ── Variant B: Net Cashflow Forecast (original model) ──────
    print("  Running Variant B (net cashflow, expense_ratio=0.72)...")
    backtest_variant_b = _run_backtest(engine, training_distributions, test_records, "net_cashflow")

    # ── Part C: Multi-scenario comparison ──────────────────────
    print("\n[4/6] Running multi-scenario comparison...")

    scenario_comparison = {}
    for scenario_name in ["base", "optimistic", "stress"]:
        scenario_dist = build_scenario(all_distributions, scenario_name)
        config = MonteCarloConfig(
            n_simulations=10_000,
            time_horizon_months=3,
            random_seed=42,
            distributions=scenario_dist,
            initial_balance=0.0,
            start_month=1,
        )
        result = engine.run(config, scenario_name=scenario_name)
        scenario_comparison[scenario_name] = {
            "month1_mean": round(result.monthly_stats[0].mean, 2),
            "month3_cumulative_mean": round(result.cumulative_stats[-1].mean, 2),
            "probability_negative": round(result.probability_negative, 4),
            "var_5pct": round(result.var_5pct, 2),
            "convergence_std": round(result.convergence_std, 4),
        }
        print(f"  {scenario_name:12s} — P(neg)={result.probability_negative:.4f}  "
              f"VaR(5%)=€{result.var_5pct:,.2f}")

    # ── Part D: v2 vs v3 Comparison ────────────────────────────
    print("\n[5/6] Computing v2 vs v3 comparison...")

    # Per-year breakdown for Variant A
    year_breakdown_a = _compute_year_breakdown(backtest_variant_a["per_month"])

    comparison = {
        "v2_training_months": V2_TRAINING_MONTHS,
        "v3_training_months": len(training_records),
        "v2_test_months": 12,
        "v3_test_months": len(test_records),
        "v2_revenue_mean": V2_REVENUE_MEAN,
        "v3_revenue_mean": round(training_distributions.revenue_mean, 2),
        "v2_revenue_std": V2_REVENUE_STD,
        "v3_revenue_std": round(training_distributions.revenue_std, 2),
        "v2_coverage_a": V2_COVERAGE_A,
        "v3_coverage_a": backtest_variant_a["coverage_90pct_ci"],
        "v2_mape_a": V2_MAPE_A,
        "v3_mape_a": backtest_variant_a["avg_pct_error"],
        "v2_coverage_b": V2_COVERAGE_B,
        "v3_coverage_b": backtest_variant_b["coverage_90pct_ci"],
        "v2_mape_b": V2_MAPE_B,
        "v3_mape_b": backtest_variant_b["avg_pct_error"],
        "v3_per_year_variant_a": year_breakdown_a,
    }

    print(f"  v2 training: {V2_TRAINING_MONTHS} months, rev mean €{V2_REVENUE_MEAN:,.0f}")
    print(f"  v3 training: {len(training_records)} months, rev mean €{training_distributions.revenue_mean:,.0f}")
    print(f"  v2 Variant A: {V2_COVERAGE_A:.0%} coverage, {V2_MAPE_A:.1f}% MAPE")
    print(f"  v3 Variant A: {backtest_variant_a['coverage_90pct_ci']:.0%} coverage, "
          f"{backtest_variant_a['avg_pct_error']:.1f}% MAPE")

    # ── Save ───────────────────────────────────────────────────
    print("\n[6/6] Saving results...")
    total_elapsed = (time.perf_counter() - total_start) * 1000

    output = {
        "experiment": "exp4_monte_carlo_v3",
        "description": "MC convergence and backtesting with FULL dataset (~240 months)",
        "motivation": "v1/v2 used recent_years=5 (60 months). v3 uses recent_years=None "
                      "for full 2004-2024 dataset, providing more robust seasonal fitting "
                      "and 24 test months (vs 12 in v2).",
        "dataset": {
            "total_months": total_months,
            "training_months": len(training_records),
            "training_range": f"{training_records[0].period_label} to {training_records[-1].period_label}",
            "test_months": len(test_records),
            "test_range": f"{test_records[0].period_label} to {test_records[-1].period_label}",
            "total_transaction_rows": pipeline_result.total_rows_processed,
        },
        "training_distributions": {
            "revenue_mean": round(training_distributions.revenue_mean, 2),
            "revenue_std": round(training_distributions.revenue_std, 2),
            "seasonal_factors": [round(f, 4) for f in training_distributions.seasonal_factors],
            "expense_ratio_mean": training_distributions.expense_ratio_mean,
            "expense_ratio_std": training_distributions.expense_ratio_std,
            "credit_note_probability": round(training_distributions.credit_note_probability, 4),
            "credit_note_ratio": round(training_distributions.credit_note_ratio, 4),
        },
        "convergence": {
            "results": convergence_results,
            "converged_at_n": _find_convergence_n(convergence_results),
        },
        "backtesting": {
            "variant_a_gross_revenue": backtest_variant_a,
            "variant_b_net_cashflow": backtest_variant_b,
        },
        "scenario_comparison": scenario_comparison,
        "comparison_with_v2": comparison,
        "total_elapsed_ms": round(total_elapsed, 2),
    }

    out_path = RESULTS_DIR / "exp4_monte_carlo_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Total months: {total_months}")
    print(f"  Training: {len(training_records)}, Test: {len(test_records)}")
    print(f"  Convergence at N={output['convergence']['converged_at_n']}")
    print(f"  Variant A (gross) coverage: {backtest_variant_a['coverage_90pct_ci']:.0%}")
    print(f"  Variant A (gross) avg MAPE: {backtest_variant_a['avg_pct_error']:.1f}%")
    print(f"  Variant B (net) coverage: {backtest_variant_b['coverage_90pct_ci']:.0%}")
    print(f"  Variant B (net) avg MAPE: {backtest_variant_b['avg_pct_error']:.1f}%")
    for year_label, year_data in year_breakdown_a.items():
        print(f"  Variant A {year_label}: {year_data['coverage']:.0%} coverage, "
              f"{year_data['avg_mape']:.1f}% MAPE")
    print(f"  Total time: {total_elapsed:.0f} ms")
    print(f"  Results: {out_path}")
    print("=" * 60)


def _run_backtest(
    engine: MonteCarloEngine,
    distributions: CashflowDistributions,
    test_records: list[MonthlyRecord],
    variant_name: str,
) -> dict:
    """Run month-by-month backtesting against test records.

    For each test month, runs 10K MC simulations for 1 month ahead,
    then compares forecast vs actual.
    """
    per_month_results: list[dict] = []

    for rec in test_records:
        config = MonteCarloConfig(
            n_simulations=10_000,
            time_horizon_months=1,
            random_seed=42 + rec.month,  # Different seed per calendar month
            distributions=distributions,
            initial_balance=0.0,
            start_month=rec.month,
        )

        mc_result = engine.run(config, scenario_name=f"backtest_{variant_name}")
        m_stats = mc_result.monthly_stats[0]

        actual_cf = rec.net_cashflow

        within_ci = m_stats.p5 <= actual_cf <= m_stats.p95
        z_score = (
            (actual_cf - m_stats.mean) / m_stats.std
            if m_stats.std > 0 else 0.0
        )

        per_month_results.append({
            "month": rec.period_label,
            "actual_net_cf": round(actual_cf, 2),
            "forecast_mean": round(m_stats.mean, 2),
            "forecast_std": round(m_stats.std, 2),
            "forecast_p5": round(m_stats.p5, 2),
            "forecast_p50": round(m_stats.p50, 2),
            "forecast_p95": round(m_stats.p95, 2),
            "within_90pct_ci": within_ci,
            "z_score": round(z_score, 3),
            "abs_error": round(abs(actual_cf - m_stats.mean), 2),
            "pct_error": round(
                abs(actual_cf - m_stats.mean) / abs(actual_cf) * 100, 2
            ) if actual_cf != 0 else 0.0,
        })

    # Aggregate metrics
    coverage = (
        sum(1 for r in per_month_results if r["within_90pct_ci"]) / len(per_month_results)
        if per_month_results else 0.0
    )
    avg_pct_error = _mean([r["pct_error"] for r in per_month_results])
    avg_abs_error = _mean([r["abs_error"] for r in per_month_results])

    # Per-quarter breakdown
    quarters: dict[str, dict] = {}
    for r in per_month_results:
        parts = r["month"].split("-")
        year = parts[0]
        month = int(parts[1])
        q = f"{year}-Q{(month - 1) // 3 + 1}"
        if q not in quarters:
            quarters[q] = {"months": [], "errors": [], "coverages": []}
        quarters[q]["months"].append(r["month"])
        quarters[q]["errors"].append(r["pct_error"])
        quarters[q]["coverages"].append(1 if r["within_90pct_ci"] else 0)

    quarter_summary = {}
    for q, data in sorted(quarters.items()):
        quarter_summary[q] = {
            "months": data["months"],
            "avg_pct_error": round(_mean(data["errors"]), 2),
            "coverage": round(_mean(data["coverages"]), 4),
        }

    return {
        "variant": variant_name,
        "per_month": per_month_results,
        "coverage_90pct_ci": round(coverage, 4),
        "avg_pct_error": round(avg_pct_error, 2),
        "avg_abs_error": round(avg_abs_error, 2),
        "quarter_summary": quarter_summary,
    }


def _compute_year_breakdown(per_month: list[dict]) -> dict:
    """Compute per-year coverage and MAPE from per-month results."""
    years: dict[str, dict] = {}
    for r in per_month:
        year = r["month"].split("-")[0]
        if year not in years:
            years[year] = {"errors": [], "coverages": [], "months": 0}
        years[year]["errors"].append(r["pct_error"])
        years[year]["coverages"].append(1 if r["within_90pct_ci"] else 0)
        years[year]["months"] += 1

    result = {}
    for year, data in sorted(years.items()):
        result[year] = {
            "months": data["months"],
            "coverage": round(_mean(data["coverages"]), 4),
            "avg_mape": round(_mean(data["errors"]), 2),
        }
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _find_convergence_n(results: list[dict], threshold_pct: float = 1.0) -> int:
    """Find the smallest N where mean is within threshold_pct of the reference (N=50K)."""
    for row in results:
        if row["mean_pct_diff_from_ref"] < threshold_pct:
            return row["n_simulations"]
    return results[-1]["n_simulations"]


if __name__ == "__main__":
    main()
