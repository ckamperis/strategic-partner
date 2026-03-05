"""Vectorized Monte Carlo cashflow simulation engine.

Generates N stochastic cashflow paths over a configurable time horizon
using NumPy array operations. No Python loops over simulations —
all randomness is drawn as (N × T) matrices for performance.

Key design choices (thesis justification):
- Vectorized NumPy: O(N·T) with SIMD, not O(N·T) Python loops
- Reproducible: np.random.default_rng(seed) for exact replication
- Truncated normals: revenue and expenses clipped to non-negative
- Seasonal modulation: monthly factors applied multiplicatively

Performance target: 10K sims × 3 months < 500ms.

References:
    Thesis Section 3.3.3 — Simulation Pillar, Monte Carlo Engine
    Thesis Section 4.x — Implementation, Stochastic Simulation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, Field

from pillars.simulation.distributions import CashflowDistributions

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class MonteCarloConfig(BaseModel):
    """Configuration for a Monte Carlo simulation run.

    Attributes:
        n_simulations: Number of stochastic paths (default 10,000).
        time_horizon_months: Forecast horizon in months (default 3).
        random_seed: Seed for reproducibility (default 42).
        distributions: Fitted statistical distributions for cashflow.
        initial_balance: Starting cash balance in EUR (default 0.0).
        start_month: Calendar month (1-12) for the first simulated month.
            Used to look up seasonal factors. Default 1 (January).
    """

    n_simulations: int = 10_000
    time_horizon_months: int = 3
    random_seed: int = 42
    distributions: CashflowDistributions = Field(
        default_factory=CashflowDistributions
    )
    initial_balance: float = 0.0
    start_month: int = 1


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class MonthlySimStats:
    """Descriptive statistics for a single simulated month.

    Attributes:
        month_index: 0-based month offset from start.
        mean: Mean net cashflow across all simulations.
        std: Standard deviation.
        p5: 5th percentile (pessimistic bound).
        p25: 25th percentile (lower quartile).
        p50: Median (50th percentile).
        p75: 75th percentile (upper quartile).
        p95: 95th percentile (optimistic bound).
        min: Minimum across simulations.
        max: Maximum across simulations.
    """

    month_index: int = 0
    mean: float = 0.0
    std: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    min: float = 0.0
    max: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON output and audit logging."""
        return {
            "month_index": self.month_index,
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "p5": round(self.p5, 2),
            "p25": round(self.p25, 2),
            "p50": round(self.p50, 2),
            "p75": round(self.p75, 2),
            "p95": round(self.p95, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
        }


@dataclass
class MonteCarloResult:
    """Complete output of a Monte Carlo simulation run.

    Attributes:
        monthly_stats: Per-month descriptive statistics.
        cumulative_stats: Cumulative (running total) statistics per month.
        probability_negative: P(cumulative cashflow < 0) at final month.
        var_5pct: Value at Risk — 5th percentile of cumulative cashflow
            at the final month.  Interpretation: "with 95% confidence,
            cumulative cashflow will not be worse than this."
        convergence_std: Standard error of the mean for the final month.
            Quantifies Monte Carlo sampling uncertainty.
        n_simulations: Number of simulation paths run.
        time_horizon_months: Forecast horizon.
        elapsed_ms: Wall-clock time for the simulation.
        scenario_name: Scenario label (e.g. "base", "optimistic", "stress").
        config_snapshot: Frozen copy of the config for reproducibility.
    """

    monthly_stats: list[MonthlySimStats] = field(default_factory=list)
    cumulative_stats: list[MonthlySimStats] = field(default_factory=list)
    probability_negative: float = 0.0
    var_5pct: float = 0.0
    convergence_std: float = 0.0
    n_simulations: int = 0
    time_horizon_months: int = 0
    elapsed_ms: float = 0.0
    scenario_name: str = "base"
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise full result for JSON output and PICP context."""
        return {
            "monthly_stats": [s.to_dict() for s in self.monthly_stats],
            "cumulative_stats": [s.to_dict() for s in self.cumulative_stats],
            "probability_negative": round(self.probability_negative, 4),
            "var_5pct": round(self.var_5pct, 2),
            "convergence_std": round(self.convergence_std, 4),
            "n_simulations": self.n_simulations,
            "time_horizon_months": self.time_horizon_months,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "scenario_name": self.scenario_name,
            "config_snapshot": self.config_snapshot,
        }


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """Vectorized Monte Carlo cashflow simulation engine.

    All stochastic draws are performed as (N, T) NumPy matrices:
    - Revenue: Normal(μ · seasonal, σ), clipped ≥ 0
    - Expenses: Revenue × Normal(expense_ratio_μ, expense_ratio_σ), clipped ≥ 0
    - Credit notes: Bernoulli(p) × Revenue × credit_note_ratio
    - Customer loss: Bernoulli(loss_rate) applied quarterly, reduces revenue
    - Collection delay: Normal(μ_delay, σ_delay) — informational, affects timing

    Net cashflow per month = Revenue - Expenses - Credit Notes

    References:
        Thesis Eq. 3.25 — Monte Carlo revenue generation
        Thesis Eq. 3.26 — Expense modelling
        Thesis Eq. 3.27 — Net cashflow computation
    """

    def run(
        self,
        config: MonteCarloConfig,
        scenario_name: str = "base",
    ) -> MonteCarloResult:
        """Execute the Monte Carlo simulation.

        Args:
            config: Simulation configuration with distributions.
            scenario_name: Label for the scenario (metadata only).

        Returns:
            MonteCarloResult with full statistics.
        """
        start = time.perf_counter()

        N = config.n_simulations
        T = config.time_horizon_months
        dist = config.distributions
        rng = np.random.default_rng(config.random_seed)

        # ---------------------------------------------------------------
        # Step 1: Draw seasonal factors for each simulated month
        # seasonal_factors is a 12-element list; map each month to its factor
        # ---------------------------------------------------------------
        month_indices = np.array(
            [(config.start_month - 1 + t) % 12 for t in range(T)],
            dtype=np.int32,
        )
        seasonal = np.array(dist.seasonal_factors, dtype=np.float64)
        seasonal_for_horizon = seasonal[month_indices]  # shape (T,)

        # ---------------------------------------------------------------
        # Step 2: Generate revenue matrix — shape (N, T)
        # Revenue ~ Normal(μ · seasonal_factor, σ), clipped ≥ 0
        # ---------------------------------------------------------------
        revenue_raw = rng.normal(
            loc=dist.revenue_mean * seasonal_for_horizon,  # (T,) broadcast
            scale=max(dist.revenue_std, 1e-6),  # scalar
            size=(N, T),
        )
        revenue = np.clip(revenue_raw, 0.0, None)  # Non-negative revenue

        # ---------------------------------------------------------------
        # Step 3: Apply customer loss events — Bernoulli per quarter
        # customer_loss_rate is quarterly; for monthly, we apply it every
        # 3rd month of the horizon (month_index 2, 5, 8, ...).
        # A loss event reduces revenue by a random 10-30% for that sim.
        # ---------------------------------------------------------------
        if dist.customer_loss_rate > 0 and T > 0:
            for t in range(T):
                if (t + 1) % 3 == 0:  # Quarterly boundary
                    loss_mask = rng.random(N) < dist.customer_loss_rate
                    loss_severity = rng.uniform(0.10, 0.30, size=N)
                    # Apply loss to this month and subsequent months
                    for t_fwd in range(t, T):
                        revenue[loss_mask, t_fwd] *= (1.0 - loss_severity[loss_mask])

        # ---------------------------------------------------------------
        # Step 4: Generate expenses — shape (N, T)
        # Expense = Revenue × Normal(expense_ratio_μ, expense_ratio_σ)
        # Clipped to [0, Revenue] (cannot spend more than revenue)
        # ---------------------------------------------------------------
        expense_ratios = rng.normal(
            loc=dist.expense_ratio_mean,
            scale=max(dist.expense_ratio_std, 1e-6),
            size=(N, T),
        )
        expense_ratios = np.clip(expense_ratios, 0.0, 1.0)
        expenses = revenue * expense_ratios

        # ---------------------------------------------------------------
        # Step 5: Credit note deductions — shape (N, T)
        # Each month: Bernoulli(p) determines if credit note occurs,
        # then credit_note = revenue × credit_note_ratio
        # ---------------------------------------------------------------
        credit_events = rng.random(size=(N, T)) < dist.credit_note_probability
        credit_notes = credit_events * revenue * dist.credit_note_ratio

        # ---------------------------------------------------------------
        # Step 6: Collection delay — informational metric
        # Delay ~ Normal(μ_delay, σ_delay), clipped ≥ 1 day
        # Stored as metadata; does not affect cashflow amount
        # (a full AR model would shift cash receipts, but that's
        # beyond scope — documented as [TODO-THESIS] limitation)
        # ---------------------------------------------------------------
        collection_delays = rng.normal(
            loc=dist.collection_delay_mean,
            scale=max(dist.collection_delay_std, 1e-6),
            size=(N, T),
        )
        collection_delays = np.clip(collection_delays, 1.0, None)
        avg_collection_delay = float(np.mean(collection_delays))

        # ---------------------------------------------------------------
        # Step 7: Net cashflow per month — shape (N, T)
        # Net = Revenue - Expenses - Credit Notes
        # ---------------------------------------------------------------
        net_cashflow = revenue - expenses - credit_notes

        # ---------------------------------------------------------------
        # Step 8: Cumulative cashflow — shape (N, T)
        # Running total including initial balance
        # ---------------------------------------------------------------
        cumulative = np.cumsum(net_cashflow, axis=1) + config.initial_balance

        # ---------------------------------------------------------------
        # Step 9: Compute statistics
        # ---------------------------------------------------------------
        monthly_stats = self._compute_stats(net_cashflow, "monthly")
        cumulative_stats = self._compute_stats(cumulative, "cumulative")

        # Final-month metrics
        final_cumulative = cumulative[:, -1]  # shape (N,)
        probability_negative = float(np.mean(final_cumulative < 0))
        var_5pct = float(np.percentile(final_cumulative, 5))
        # Standard error of the mean: σ/√N (requires N>1 for ddof=1)
        convergence_std = (
            float(np.std(final_cumulative, ddof=1) / np.sqrt(N))
            if N > 1
            else 0.0
        )

        elapsed = (time.perf_counter() - start) * 1000  # ms

        # Config snapshot for reproducibility
        config_snapshot = {
            "n_simulations": N,
            "time_horizon_months": T,
            "random_seed": config.random_seed,
            "initial_balance": config.initial_balance,
            "start_month": config.start_month,
            "distributions": dist.to_dict(),
            "avg_collection_delay_days": round(avg_collection_delay, 1),
        }

        result = MonteCarloResult(
            monthly_stats=monthly_stats,
            cumulative_stats=cumulative_stats,
            probability_negative=probability_negative,
            var_5pct=var_5pct,
            convergence_std=convergence_std,
            n_simulations=N,
            time_horizon_months=T,
            elapsed_ms=elapsed,
            scenario_name=scenario_name,
            config_snapshot=config_snapshot,
        )

        logger.info(
            "monte_carlo.run.complete",
            n_simulations=N,
            time_horizon_months=T,
            elapsed_ms=round(elapsed, 2),
            probability_negative=round(probability_negative, 4),
            var_5pct=round(var_5pct, 2),
            convergence_std=round(convergence_std, 4),
            scenario=scenario_name,
        )

        return result

    @staticmethod
    def _compute_stats(
        matrix: np.ndarray,
        label: str,
    ) -> list[MonthlySimStats]:
        """Compute descriptive statistics for each month column.

        Args:
            matrix: (N, T) array of cashflow values.
            label: Human-readable label (for logging).

        Returns:
            List of MonthlySimStats, one per month.
        """
        N, T = matrix.shape
        stats: list[MonthlySimStats] = []

        for t in range(T):
            col = matrix[:, t]
            # ddof=1 requires N>1; fall back to 0.0 for single simulation
            std_val = float(np.std(col, ddof=1)) if N > 1 else 0.0
            stats.append(
                MonthlySimStats(
                    month_index=t,
                    mean=float(np.mean(col)),
                    std=std_val,
                    p5=float(np.percentile(col, 5)),
                    p25=float(np.percentile(col, 25)),
                    p50=float(np.percentile(col, 50)),
                    p75=float(np.percentile(col, 75)),
                    p95=float(np.percentile(col, 95)),
                    min=float(np.min(col)),
                    max=float(np.max(col)),
                )
            )

        return stats
