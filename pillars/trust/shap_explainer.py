"""Simulated SHAP — factor contribution analysis via counterfactual MC runs.

This is NOT real SHAP (Shapley Additive Explanations). It is a
simulated approximation that measures each factor's marginal contribution
by running counterfactual Monte Carlo simulations with that factor
set to neutral/default, then measuring the difference in mean outcome.

This approximation is explicitly documented as a thesis limitation
(Section 6.x). True SHAP would require computing marginal contributions
over all factor coalitions — feasible but out of scope for the PoC.

References:
    Thesis Section 3.3.4 — Trust Pillar, Explainability
    Thesis Section 6.x — Limitations, Simulated SHAP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from data.pipeline.models import BusinessMetrics
from pillars.simulation.distributions import CashflowDistributions
from pillars.simulation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
)

logger = structlog.get_logger()

# Quick MC run for counterfactuals (fewer sims for speed)
_COUNTERFACTUAL_SIMS = 2_000
_COUNTERFACTUAL_SEED = 99


@dataclass
class FactorContribution:
    """A single factor's contribution to the cashflow forecast.

    Attributes:
        factor_name: Human-readable factor identifier.
        impact: Monetary impact on mean net cashflow (positive = increases).
        direction: "positive", "negative", or "neutral".
        magnitude: "high", "medium", or "low" (|impact| thresholds).
        evidence: Explanation text with relevant metric values.
    """

    factor_name: str = ""
    impact: float = 0.0
    direction: str = "neutral"
    magnitude: str = "low"
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "impact": round(self.impact, 2),
            "direction": self.direction,
            "magnitude": self.magnitude,
            "evidence": self.evidence,
        }


def _classify_direction(impact: float) -> str:
    """Classify impact direction."""
    if impact > 100:
        return "positive"
    elif impact < -100:
        return "negative"
    return "neutral"


def _classify_magnitude(impact: float) -> str:
    """Classify impact magnitude by absolute value."""
    abs_impact = abs(impact)
    if abs_impact > 5_000:
        return "high"
    elif abs_impact > 1_000:
        return "medium"
    return "low"


class SimulatedSHAP:
    """Simulated SHAP explainer via counterfactual Monte Carlo runs.

    For each factor, runs a counterfactual simulation with that factor
    set to neutral/default, then measures:
        impact = mean(base_scenario) - mean(counterfactual)

    A positive impact means the factor increases the forecast;
    a negative impact means it decreases it.

    Factors analysed:
    1. Seasonal pattern — base vs flat seasonality
    2. Customer concentration — flag if top5 > 50%
    3. Credit note impact — base vs zero credit notes
    4. Revenue volatility — base vs zero-variance revenue
    5. Expense ratio — base vs lower ratio (0.65)
    """

    def __init__(self) -> None:
        self._engine = MonteCarloEngine()

    def explain_forecast(
        self,
        simulation_result: dict[str, Any],
        distributions: CashflowDistributions,
        metrics: BusinessMetrics | None = None,
    ) -> list[FactorContribution]:
        """Generate factor contributions for the cashflow forecast.

        Args:
            simulation_result: Full simulation pillar output (dict).
            distributions: The base distributions used for simulation.
            metrics: Optional business metrics (for customer concentration).

        Returns:
            List of FactorContribution sorted by |impact| descending.
        """
        metrics = metrics or BusinessMetrics()

        # Get base scenario mean from simulation result
        scenarios = simulation_result.get("scenarios", {})
        base_scenario = scenarios.get("base", {})
        base_monthly = base_scenario.get("monthly_stats", [])

        if not base_monthly:
            logger.warning("shap.no_base_stats")
            return []

        # Use first month mean as the reference point
        base_mean = base_monthly[0].get("mean", 0.0)

        # Base config for counterfactual runs
        base_config = MonteCarloConfig(
            n_simulations=_COUNTERFACTUAL_SIMS,
            time_horizon_months=1,
            random_seed=_COUNTERFACTUAL_SEED,
            distributions=distributions,
        )

        # Run base reference
        base_result = self._engine.run(base_config, scenario_name="shap_base")
        ref_mean = base_result.monthly_stats[0].mean

        factors: list[FactorContribution] = []

        # Factor 1: Seasonal pattern
        factors.append(
            self._factor_seasonal(distributions, ref_mean)
        )

        # Factor 2: Customer concentration
        factors.append(
            self._factor_customer_concentration(metrics)
        )

        # Factor 3: Credit note impact
        factors.append(
            self._factor_credit_notes(distributions, ref_mean)
        )

        # Factor 4: Revenue volatility
        factors.append(
            self._factor_revenue_volatility(distributions, ref_mean)
        )

        # Factor 5: Expense ratio
        factors.append(
            self._factor_expense_ratio(distributions, ref_mean)
        )

        # Sort by |impact| descending
        factors.sort(key=lambda f: abs(f.impact), reverse=True)

        logger.info(
            "shap.explain_complete",
            factor_count=len(factors),
            top_factor=factors[0].factor_name if factors else "none",
        )

        return factors

    def _run_counterfactual(
        self,
        distributions: CashflowDistributions,
    ) -> float:
        """Run a quick counterfactual MC and return first-month mean."""
        config = MonteCarloConfig(
            n_simulations=_COUNTERFACTUAL_SIMS,
            time_horizon_months=1,
            random_seed=_COUNTERFACTUAL_SEED,
            distributions=distributions,
        )
        result = self._engine.run(config, scenario_name="counterfactual")
        return result.monthly_stats[0].mean

    def _factor_seasonal(
        self,
        distributions: CashflowDistributions,
        ref_mean: float,
    ) -> FactorContribution:
        """Compare base vs flat seasonality (all factors = 1.0)."""
        flat_dist = CashflowDistributions(
            **{
                k: ([1.0] * 12 if k == "seasonal_factors" else v)
                for k, v in distributions.to_dict().items()
            }
        )
        cf_mean = self._run_counterfactual(flat_dist)
        impact = ref_mean - cf_mean

        seasonal_range = max(distributions.seasonal_factors) - min(
            distributions.seasonal_factors
        )

        return FactorContribution(
            factor_name="seasonal_pattern",
            impact=impact,
            direction=_classify_direction(impact),
            magnitude=_classify_magnitude(impact),
            evidence=(
                f"Seasonal variation range: {seasonal_range:.2f}. "
                f"Removing seasonality changes mean cashflow by €{impact:+,.0f}."
            ),
        )

    @staticmethod
    def _factor_customer_concentration(
        metrics: BusinessMetrics,
    ) -> FactorContribution:
        """Assess customer concentration risk (no counterfactual needed)."""
        top5 = metrics.customer_concentration.top5_pct
        total = metrics.customer_concentration.total_customers

        if top5 > 0.60:
            impact = -5_000.0  # Estimated risk impact
            magnitude = "high"
        elif top5 > 0.40:
            impact = -2_000.0
            magnitude = "medium"
        else:
            impact = -500.0
            magnitude = "low"

        return FactorContribution(
            factor_name="customer_concentration",
            impact=impact,
            direction="negative",
            magnitude=magnitude,
            evidence=(
                f"Top 5% of customers account for {top5 * 100:.1f}% of revenue "
                f"(total customers: {total}). "
                f"Higher concentration increases revenue volatility risk."
            ),
        )

    def _factor_credit_notes(
        self,
        distributions: CashflowDistributions,
        ref_mean: float,
    ) -> FactorContribution:
        """Compare base vs zero credit notes."""
        no_credit_dist = CashflowDistributions(
            **{
                k: (
                    0.0
                    if k in ("credit_note_probability", "credit_note_ratio")
                    else v
                )
                for k, v in distributions.to_dict().items()
            }
        )
        cf_mean = self._run_counterfactual(no_credit_dist)
        impact = ref_mean - cf_mean  # Should be negative (credits reduce cashflow)

        return FactorContribution(
            factor_name="credit_note_impact",
            impact=impact,
            direction=_classify_direction(impact),
            magnitude=_classify_magnitude(impact),
            evidence=(
                f"Credit note probability: {distributions.credit_note_probability:.3f}, "
                f"ratio: {distributions.credit_note_ratio:.3f}. "
                f"Removing credit notes changes mean cashflow by €{-impact:+,.0f}."
            ),
        )

    def _factor_revenue_volatility(
        self,
        distributions: CashflowDistributions,
        ref_mean: float,
    ) -> FactorContribution:
        """Compare base vs zero revenue variance."""
        stable_dist = CashflowDistributions(
            **{
                k: (0.0 if k == "revenue_std" else v)
                for k, v in distributions.to_dict().items()
            }
        )
        cf_mean = self._run_counterfactual(stable_dist)
        impact = ref_mean - cf_mean

        return FactorContribution(
            factor_name="revenue_volatility",
            impact=impact,
            direction=_classify_direction(impact),
            magnitude=_classify_magnitude(impact),
            evidence=(
                f"Revenue std: €{distributions.revenue_std:,.0f} "
                f"(CV: {distributions.revenue_std / max(distributions.revenue_mean, 1) * 100:.1f}%). "
                f"Removing volatility changes mean cashflow by €{impact:+,.0f}."
            ),
        )

    def _factor_expense_ratio(
        self,
        distributions: CashflowDistributions,
        ref_mean: float,
    ) -> FactorContribution:
        """Compare base vs lower expense ratio (0.65)."""
        lower_expense_dist = CashflowDistributions(
            **{
                k: (0.65 if k == "expense_ratio_mean" else v)
                for k, v in distributions.to_dict().items()
            }
        )
        cf_mean = self._run_counterfactual(lower_expense_dist)
        impact = ref_mean - cf_mean  # Should be negative (higher actual ratio reduces CF)

        return FactorContribution(
            factor_name="expense_ratio",
            impact=impact,
            direction=_classify_direction(impact),
            magnitude=_classify_magnitude(impact),
            evidence=(
                f"Current expense ratio: {distributions.expense_ratio_mean:.2f}. "
                f"If ratio were 0.65 instead, mean cashflow would change by €{-impact:+,.0f}."
            ),
        )
