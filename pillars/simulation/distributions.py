"""Distribution fitting from ERP data for Monte Carlo simulation.

Extracts statistical distributions from historical ERP transaction data
to parameterise the Monte Carlo engine. Where real data is unavailable
(e.g. supplier expenses, collection delays), documented defaults are used.

References:
    Thesis Section 3.3.3 — Simulation Pillar, Distribution Fitting
    Thesis Section 4.x — Implementation, Stochastic Parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from data.pipeline.models import BusinessMetrics, MonthlyData

logger = structlog.get_logger()


@dataclass
class CashflowDistributions:
    """Statistical distributions for Monte Carlo cashflow simulation.

    All monetary values in EUR.

    Attributes:
        revenue_mean: Monthly mean gross revenue from ERP data.
        revenue_std: Monthly standard deviation of gross revenue.
        seasonal_factors: 12 monthly multipliers (mean ≈ 1.0).
        expense_ratio_mean: Expenses as fraction of revenue (default 0.72).
        expense_ratio_std: Std dev of expense ratio (default 0.05).
        collection_delay_mean: Days to collect payment (default 52).
        collection_delay_std: Std dev of collection delay (default 15).
        credit_note_probability: P(credit note) per invoice (from data: ~0.046).
        credit_note_ratio: Avg credit note as fraction of invoice value.
        customer_loss_rate: Quarterly customer loss probability (default 0.02).
    """

    revenue_mean: float = 0.0
    revenue_std: float = 0.0
    seasonal_factors: list[float] = field(default_factory=lambda: [1.0] * 12)
    expense_ratio_mean: float = 0.72
    expense_ratio_std: float = 0.05
    collection_delay_mean: float = 52.0
    collection_delay_std: float = 15.0
    credit_note_probability: float = 0.046
    credit_note_ratio: float = 0.05
    customer_loss_rate: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary for logging and audit."""
        return {
            "revenue_mean": self.revenue_mean,
            "revenue_std": self.revenue_std,
            "seasonal_factors": self.seasonal_factors,
            "expense_ratio_mean": self.expense_ratio_mean,
            "expense_ratio_std": self.expense_ratio_std,
            "collection_delay_mean": self.collection_delay_mean,
            "collection_delay_std": self.collection_delay_std,
            "credit_note_probability": self.credit_note_probability,
            "credit_note_ratio": self.credit_note_ratio,
            "customer_loss_rate": self.customer_loss_rate,
        }


def fit_from_erp_data(
    monthly_data: MonthlyData,
    metrics: BusinessMetrics,
) -> CashflowDistributions:
    """Fit cashflow distributions from historical ERP data.

    Computes revenue statistics from monthly records and extracts
    seasonal factors from pre-computed business metrics. Parameters
    that cannot be derived from available data use documented defaults.

    Args:
        monthly_data: Monthly aggregated financial records.
        metrics: Pre-computed business metrics (seasonal indices, etc.).

    Returns:
        A CashflowDistributions fitted to the data.

    Note:
        [TODO-THESIS] The following parameters are estimated, not derived:
        - expense_ratio_mean/std (no supplier invoice data)
        - collection_delay_mean/std (DUEDATE = TRNDATE in dataset)
        - customer_loss_rate (no longitudinal customer tracking)
    """
    records = monthly_data.records
    if not records:
        logger.warning("distributions.fit.empty_data")
        return CashflowDistributions()

    # Revenue statistics from monthly sales_gross
    revenues = np.array([r.sales_gross for r in records], dtype=np.float64)
    revenue_mean = float(np.mean(revenues))
    revenue_std = float(np.std(revenues, ddof=1)) if len(revenues) > 1 else revenue_mean * 0.15

    # Seasonal factors from metrics (should be 12 values, mean ≈ 1.0)
    seasonal = metrics.seasonal_indices if metrics.seasonal_indices else [1.0] * 12
    # Ensure exactly 12 factors
    if len(seasonal) < 12:
        seasonal = seasonal + [1.0] * (12 - len(seasonal))
    elif len(seasonal) > 12:
        seasonal = seasonal[:12]

    # Credit note probability from data
    total_sales_count = sum(1 for r in records if r.sales_gross > 0)
    total_credit_count = sum(1 for r in records if r.credit_notes > 0)
    credit_note_prob = (
        total_credit_count / total_sales_count
        if total_sales_count > 0
        else 0.046  # default
    )

    # Credit note ratio: total credit notes / total sales
    total_sales = sum(r.sales_gross for r in records)
    total_credits = sum(r.credit_notes for r in records)
    credit_note_ratio = (
        total_credits / total_sales if total_sales > 0 else 0.05
    )

    dist = CashflowDistributions(
        revenue_mean=revenue_mean,
        revenue_std=revenue_std,
        seasonal_factors=seasonal,
        credit_note_probability=credit_note_prob,
        credit_note_ratio=credit_note_ratio,
        # Estimated parameters (documented defaults):
        expense_ratio_mean=0.72,   # [TODO-THESIS] No supplier data
        expense_ratio_std=0.05,    # [TODO-THESIS] Estimated
        collection_delay_mean=52.0,  # [TODO-THESIS] DUEDATE=TRNDATE
        collection_delay_std=15.0,   # [TODO-THESIS] Estimated
        customer_loss_rate=0.02,     # [TODO-THESIS] No longitudinal data
    )

    logger.info(
        "distributions.fit.complete",
        months=len(records),
        revenue_mean=round(revenue_mean, 2),
        revenue_std=round(revenue_std, 2),
        credit_note_prob=round(credit_note_prob, 4),
        credit_note_ratio=round(credit_note_ratio, 4),
    )

    return dist


def build_scenario(
    base: CashflowDistributions,
    scenario: str,
) -> CashflowDistributions:
    """Create a scenario variant of the base distributions.

    Supported scenarios:
    - "base": No changes (returns copy).
    - "optimistic": +10% revenue, -20% volatility, -3pp expense ratio.
    - "stress": -15% revenue, +30% volatility, +15 days collection, 2× customer loss.

    Args:
        base: The base-case distributions (from ERP data).
        scenario: Scenario name ("base", "optimistic", "stress").

    Returns:
        Adjusted CashflowDistributions for the scenario.

    Raises:
        ValueError: If scenario name is not recognised.
    """
    if scenario == "base":
        return CashflowDistributions(**{
            k: list(v) if isinstance(v, list) else v
            for k, v in base.to_dict().items()
        })

    if scenario == "optimistic":
        return CashflowDistributions(
            revenue_mean=base.revenue_mean * 1.10,
            revenue_std=base.revenue_std * 0.80,
            seasonal_factors=list(base.seasonal_factors),
            expense_ratio_mean=base.expense_ratio_mean - 0.03,
            expense_ratio_std=base.expense_ratio_std,
            collection_delay_mean=base.collection_delay_mean,
            collection_delay_std=base.collection_delay_std,
            credit_note_probability=base.credit_note_probability * 0.80,
            credit_note_ratio=base.credit_note_ratio,
            customer_loss_rate=base.customer_loss_rate * 0.50,
        )

    if scenario == "stress":
        return CashflowDistributions(
            revenue_mean=base.revenue_mean * 0.85,
            revenue_std=base.revenue_std * 1.30,
            seasonal_factors=list(base.seasonal_factors),
            expense_ratio_mean=base.expense_ratio_mean + 0.03,
            expense_ratio_std=base.expense_ratio_std * 1.20,
            collection_delay_mean=base.collection_delay_mean + 15.0,
            collection_delay_std=base.collection_delay_std * 1.30,
            credit_note_probability=min(base.credit_note_probability * 1.50, 1.0),
            credit_note_ratio=base.credit_note_ratio * 1.20,
            customer_loss_rate=base.customer_loss_rate * 2.0,
        )

    raise ValueError(f"Unknown scenario: {scenario!r}. Use 'base', 'optimistic', or 'stress'.")
