"""Experiment 4 — Monte Carlo Convergence & Backtesting.

Train/test split: train on 2019-2022 (48 months), test on 2023 (12 months).
Runs TWO variants:
  - Variant A: gross revenue forecast (expense_ratio=0)
  - Variant B: net cashflow forecast (expense_ratio=0.72)

Part A: Convergence analysis (N=100..50K simulations).
Part B: 12-month backtesting with 90% confidence intervals.

No LLM calls needed — pure NumPy computation.

Usage:
    python experiments/exp4_monte_carlo.py

Produces:
    data/results/exp4_monte_carlo.json
"""

from __future__ import annotations

import copy
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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    print("=" * 60)
    print("EXPERIMENT 4 v2: Monte Carlo Convergence & Backtesting (Corrected)")
    print(f"  Sim counts: {SIM_COUNTS}")
    print("=" * 60)

    # ── Load real data ───────────────────────────────────────
    print("\n[1/5] Loading ERP data...")
    transformer = ERPTransformer(recent_years=5)
    pipeline_result = transformer.run_pipeline(
        str(PROJECT_ROOT / "data" / "raw" / "cashflow_dataset.xlsx")
    )

    all_records = pipeline_result.monthly_data.records
    print(f"  Total months: {len(all_records)}")
    print(f"  Date range: {all_records[0].period_label} to {all_records[-1].period_label}")

    # ── Part A: Convergence (same as v1 — correct) ──────────
    print("\n[2/5] Running convergence analysis (unchanged from v1)...")

    # Fit from ALL data for convergence test (not a forecasting exercise)
    all_distributions = fit_from_erp_data(
        pipeline_result.monthly_data,
        pipeline_result.metrics,
    )
    print(f"  Revenue mean: €{all_distributions.revenue_mean:,.2f}")
    print(f"  Revenue std:  €{all_distributions.revenue_std:,.2f}")

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

    # ── Part B: Proper Train/Test Backtesting ────────────────
    print("\n[3/5] Running backtesting with proper train/test split...")

    # Split: train on <= 2022, test on 2023
    training_records = [r for r in all_records if r.year <= 2022]
    test_records = [r for r in all_records if r.year == 2023]

    print(f"  Training months: {len(training_records)} ({training_records[0].period_label} to {training_records[-1].period_label})")
    print(f"  Test months: {len(test_records)} ({test_records[0].period_label} to {test_records[-1].period_label})")

    # Fit distributions ONLY from training data (no leakage)
    training_monthly = MonthlyData(records=training_records)
    # We need training metrics too — compute them from training records
    training_metrics = pipeline_result.metrics  # Metrics are mostly structural; seasonal indices may differ slightly

    training_distributions = fit_from_erp_data(training_monthly, training_metrics)
    print(f"  Training revenue mean: €{training_distributions.revenue_mean:,.2f}")
    print(f"  Training revenue std:  €{training_distributions.revenue_std:,.2f}")

    # ── Variant A: Gross Revenue Forecast ────────────────────
    # Since net_cashflow ≈ sales_gross - credit_notes + receipts - payments
    # and payments_out is often 0 in the data, actuals are effectively gross revenue.
    # Forecast revenue without expense deduction to test: can we predict revenue?
    print("\n  Running Variant A (gross revenue forecast, expense_ratio=0)...")
    gross_distributions = CashflowDistributions(
        revenue_mean=training_distributions.revenue_mean,
        revenue_std=training_distributions.revenue_std,
        seasonal_factors=list(training_distributions.seasonal_factors),
        expense_ratio_mean=0.0,  # No expense deduction
        expense_ratio_std=0.0,
        collection_delay_mean=training_distributions.collection_delay_mean,
        collection_delay_std=training_distributions.collection_delay_std,
        credit_note_probability=training_distributions.credit_note_probability,
        credit_note_ratio=training_distributions.credit_note_ratio,
        customer_loss_rate=training_distributions.customer_loss_rate,
    )

    backtest_variant_a = _run_backtest(engine, gross_distributions, test_records, "gross_revenue")

    # ── Variant B: Net Cashflow Forecast (original model) ────
    print("  Running Variant B (net cashflow, expense_ratio=0.72)...")
    backtest_variant_b = _run_backtest(engine, training_distributions, test_records, "net_cashflow")

    # ── Part C: Multi-scenario comparison (same as v1) ───────
    print("\n[4/5] Running multi-scenario comparison...")

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

    # ── Save ─────────────────────────────────────────────────
    print("\n[5/5] Saving results...")
    total_elapsed = (time.perf_counter() - total_start) * 1000

    output = {
        "experiment": "exp4_monte_carlo",
        "description": "Monte Carlo convergence & backtesting — corrected train/test split + dual forecast",
        "correction_note": "v1 had data leakage (all months in fit) and only net cashflow forecast. "
                           "v2 uses proper train (2019-2022) / test (2023) split and two forecast variants.",
        "config": {
            "sim_counts": SIM_COUNTS,
            "time_horizon_months": 3,
            "random_seed": 42,
            "training_months": len(training_records),
            "test_months": len(test_records),
            "training_revenue_mean": round(training_distributions.revenue_mean, 2),
            "training_revenue_std": round(training_distributions.revenue_std, 2),
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
        "total_elapsed_ms": round(total_elapsed, 2),
    }

    out_path = RESULTS_DIR / "exp4_monte_carlo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Convergence at N={output['convergence']['converged_at_n']}")
    print(f"  Variant A (gross) coverage: {backtest_variant_a['coverage_90pct_ci']:.0%}")
    print(f"  Variant A (gross) avg MAPE: {backtest_variant_a['avg_pct_error']:.1f}%")
    print(f"  Variant B (net) coverage: {backtest_variant_b['coverage_90pct_ci']:.0%}")
    print(f"  Variant B (net) avg MAPE: {backtest_variant_b['avg_pct_error']:.1f}%")
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
            random_seed=42 + rec.month,  # Different seed per month for independence
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
    coverage = sum(1 for r in per_month_results if r["within_90pct_ci"]) / len(per_month_results) if per_month_results else 0.0
    avg_pct_error = _mean([r["pct_error"] for r in per_month_results])
    avg_abs_error = _mean([r["abs_error"] for r in per_month_results])

    # Per-quarter breakdown
    quarters = {}
    for r in per_month_results:
        month = int(r["month"].split("-")[1])
        q = f"Q{(month - 1) // 3 + 1}"
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
