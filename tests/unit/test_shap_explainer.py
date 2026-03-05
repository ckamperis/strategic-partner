"""Unit tests for pillars.trust.shap_explainer — SimulatedSHAP.

Tests cover:
- Factor count: returns 5 factors
- Impact direction: removing credit notes -> positive impact
- Factor ordering: sorted by |impact| descending
- Extreme distributions -> high impact
- Evidence strings: non-empty, contain metric values
- Customer concentration analysis
- Empty simulation result
"""

from __future__ import annotations

import pytest

from data.pipeline.models import BusinessMetrics, CustomerConcentration
from pillars.simulation.distributions import CashflowDistributions
from pillars.trust.shap_explainer import (
    FactorContribution,
    SimulatedSHAP,
    _classify_direction,
    _classify_magnitude,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_distributions() -> CashflowDistributions:
    return CashflowDistributions(
        revenue_mean=100_000.0,
        revenue_std=20_000.0,
        seasonal_factors=[0.8, 0.85, 0.9, 1.0, 1.05, 1.1,
                          1.15, 1.2, 1.1, 1.0, 0.9, 0.95],
        expense_ratio_mean=0.72,
        expense_ratio_std=0.05,
        credit_note_probability=0.046,
        credit_note_ratio=0.05,
        customer_loss_rate=0.02,
    )


@pytest.fixture
def simulation_result() -> dict:
    """Realistic simulation output for SHAP analysis."""
    return {
        "scenarios": {
            "base": {
                "monthly_stats": [
                    {"mean": 28_000.0, "std": 8_000.0,
                     "p5": 15_000.0, "p95": 42_000.0},
                ],
                "n_simulations": 10_000,
            },
        },
    }


@pytest.fixture
def metrics() -> BusinessMetrics:
    return BusinessMetrics(
        customer_concentration=CustomerConcentration(
            top5_pct=0.45, top10_pct=0.65, total_customers=120,
        ),
    )


@pytest.fixture
def shap() -> SimulatedSHAP:
    return SimulatedSHAP()


# ---------------------------------------------------------------------------
# Factor count
# ---------------------------------------------------------------------------

class TestFactorCount:
    """SHAP should return exactly 5 factors."""

    def test_returns_five_factors(
        self,
        shap: SimulatedSHAP,
        simulation_result: dict,
        base_distributions: CashflowDistributions,
        metrics: BusinessMetrics,
    ) -> None:
        factors = shap.explain_forecast(
            simulation_result, base_distributions, metrics,
        )
        assert len(factors) == 5

    def test_empty_simulation_returns_empty(
        self, shap: SimulatedSHAP, base_distributions: CashflowDistributions,
    ) -> None:
        result = shap.explain_forecast(
            {"scenarios": {}}, base_distributions,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Factor names
# ---------------------------------------------------------------------------

class TestFactorNames:
    """All expected factors should be present."""

    def test_expected_factor_names(
        self,
        shap: SimulatedSHAP,
        simulation_result: dict,
        base_distributions: CashflowDistributions,
    ) -> None:
        factors = shap.explain_forecast(
            simulation_result, base_distributions,
        )
        names = {f.factor_name for f in factors}
        expected = {
            "seasonal_pattern",
            "customer_concentration",
            "credit_note_impact",
            "revenue_volatility",
            "expense_ratio",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# Impact direction
# ---------------------------------------------------------------------------

class TestImpactDirection:
    """Verify impact direction for key factors."""

    def test_credit_note_impact_is_negative(
        self,
        shap: SimulatedSHAP,
        simulation_result: dict,
        base_distributions: CashflowDistributions,
    ) -> None:
        """Removing credit notes should increase cashflow -> credit note impact is negative."""
        factors = shap.explain_forecast(
            simulation_result, base_distributions,
        )
        credit = next(f for f in factors if f.factor_name == "credit_note_impact")
        # Credit notes reduce cashflow, so their impact should be negative
        assert credit.impact <= 0

    def test_customer_concentration_always_negative(
        self,
        shap: SimulatedSHAP,
        simulation_result: dict,
        base_distributions: CashflowDistributions,
        metrics: BusinessMetrics,
    ) -> None:
        factors = shap.explain_forecast(
            simulation_result, base_distributions, metrics,
        )
        conc = next(f for f in factors if f.factor_name == "customer_concentration")
        assert conc.direction == "negative"

    def test_expense_ratio_impact_is_negative(
        self,
        shap: SimulatedSHAP,
        simulation_result: dict,
        base_distributions: CashflowDistributions,
    ) -> None:
        """Higher expense ratio (0.72 vs 0.65) should reduce cashflow."""
        factors = shap.explain_forecast(
            simulation_result, base_distributions,
        )
        expense = next(f for f in factors if f.factor_name == "expense_ratio")
        # Base 0.72 vs counterfactual 0.65 -> base has less cashflow
        assert expense.impact < 0


# ---------------------------------------------------------------------------
# Factor ordering
# ---------------------------------------------------------------------------

class TestFactorOrdering:
    """Factors should be sorted by |impact| descending."""

    def test_sorted_by_abs_impact(
        self,
        shap: SimulatedSHAP,
        simulation_result: dict,
        base_distributions: CashflowDistributions,
    ) -> None:
        factors = shap.explain_forecast(
            simulation_result, base_distributions,
        )
        impacts = [abs(f.impact) for f in factors]
        assert impacts == sorted(impacts, reverse=True)


# ---------------------------------------------------------------------------
# Evidence strings
# ---------------------------------------------------------------------------

class TestEvidenceStrings:
    """Evidence should be non-empty and contain relevant data."""

    def test_all_factors_have_evidence(
        self,
        shap: SimulatedSHAP,
        simulation_result: dict,
        base_distributions: CashflowDistributions,
    ) -> None:
        factors = shap.explain_forecast(
            simulation_result, base_distributions,
        )
        for factor in factors:
            assert factor.evidence, f"{factor.factor_name} has empty evidence"
            assert len(factor.evidence) > 10

    def test_seasonal_evidence_contains_range(
        self,
        shap: SimulatedSHAP,
        simulation_result: dict,
        base_distributions: CashflowDistributions,
    ) -> None:
        factors = shap.explain_forecast(
            simulation_result, base_distributions,
        )
        seasonal = next(f for f in factors if f.factor_name == "seasonal_pattern")
        assert "variation" in seasonal.evidence.lower() or "range" in seasonal.evidence.lower()


# ---------------------------------------------------------------------------
# High concentration
# ---------------------------------------------------------------------------

class TestHighConcentration:
    """Very high customer concentration -> high impact."""

    def test_extreme_concentration(
        self,
        shap: SimulatedSHAP,
        simulation_result: dict,
        base_distributions: CashflowDistributions,
    ) -> None:
        extreme_metrics = BusinessMetrics(
            customer_concentration=CustomerConcentration(
                top5_pct=0.80, total_customers=50,
            ),
        )
        factors = shap.explain_forecast(
            simulation_result, base_distributions, extreme_metrics,
        )
        conc = next(f for f in factors if f.factor_name == "customer_concentration")
        assert conc.magnitude == "high"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestFactorSerialisation:
    """Test FactorContribution.to_dict()."""

    def test_to_dict(self) -> None:
        f = FactorContribution(
            factor_name="test", impact=1234.56,
            direction="positive", magnitude="high",
            evidence="Test evidence",
        )
        d = f.to_dict()
        assert d["factor_name"] == "test"
        assert d["impact"] == 1234.56
        assert d["evidence"] == "Test evidence"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Test classify_direction and classify_magnitude."""

    def test_direction_positive(self) -> None:
        assert _classify_direction(500) == "positive"

    def test_direction_negative(self) -> None:
        assert _classify_direction(-500) == "negative"

    def test_direction_neutral(self) -> None:
        assert _classify_direction(50) == "neutral"

    def test_magnitude_high(self) -> None:
        assert _classify_magnitude(10_000) == "high"

    def test_magnitude_medium(self) -> None:
        assert _classify_magnitude(3_000) == "medium"

    def test_magnitude_low(self) -> None:
        assert _classify_magnitude(500) == "low"
