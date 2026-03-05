"""Unit tests for pillars.simulation.distributions.

Tests cover:
- CashflowDistributions defaults and serialisation
- fit_from_erp_data with synthetic MonthlyData
- Seasonal factor handling (padding, truncation)
- Credit note computation from data
- build_scenario: base, optimistic, stress
- build_scenario with unknown scenario -> ValueError
"""

from __future__ import annotations

import pytest

from data.pipeline.models import BusinessMetrics, MonthlyData, MonthlyRecord
from pillars.simulation.distributions import (
    CashflowDistributions,
    build_scenario,
    fit_from_erp_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monthly_data(n_months: int = 12) -> MonthlyData:
    """Create synthetic monthly data for testing."""
    records = []
    for i in range(n_months):
        records.append(
            MonthlyRecord(
                year=2023,
                month=i + 1,
                sales_gross=100_000.0 + i * 5_000.0,
                sales_net=84_745.0 + i * 4_237.0,
                credit_notes=5_000.0 if i % 3 == 0 else 0.0,
                payments_out=10_000.0,
                receipts_in=0.0,
                transaction_count=50 + i * 2,
                unique_customers=20 + i,
            )
        )
    return MonthlyData(records=records)


def _make_metrics(n_indices: int = 12) -> BusinessMetrics:
    """Create synthetic business metrics with seasonal indices."""
    indices = [0.8, 0.85, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.1, 1.0, 0.9, 0.95]
    return BusinessMetrics(
        seasonal_indices=indices[:n_indices],
        credit_note_ratio=0.04,
        total_revenue_gross=1_500_000.0,
    )


# ---------------------------------------------------------------------------
# CashflowDistributions defaults
# ---------------------------------------------------------------------------

class TestCashflowDistributionsDefaults:
    """Test default values and serialisation."""

    def test_defaults(self) -> None:
        dist = CashflowDistributions()
        assert dist.revenue_mean == 0.0
        assert dist.revenue_std == 0.0
        assert len(dist.seasonal_factors) == 12
        assert all(f == 1.0 for f in dist.seasonal_factors)
        assert dist.expense_ratio_mean == 0.72
        assert dist.expense_ratio_std == 0.05
        assert dist.collection_delay_mean == 52.0
        assert dist.collection_delay_std == 15.0
        assert dist.credit_note_probability == 0.046
        assert dist.credit_note_ratio == 0.05
        assert dist.customer_loss_rate == 0.02

    def test_to_dict_keys(self) -> None:
        dist = CashflowDistributions(revenue_mean=50_000)
        d = dist.to_dict()
        expected_keys = {
            "revenue_mean", "revenue_std", "seasonal_factors",
            "expense_ratio_mean", "expense_ratio_std",
            "collection_delay_mean", "collection_delay_std",
            "credit_note_probability", "credit_note_ratio",
            "customer_loss_rate",
        }
        assert set(d.keys()) == expected_keys
        assert d["revenue_mean"] == 50_000

    def test_to_dict_roundtrip(self) -> None:
        """Serialise and reconstruct — values should match."""
        original = CashflowDistributions(
            revenue_mean=120_000, revenue_std=30_000,
        )
        d = original.to_dict()
        restored = CashflowDistributions(**d)
        assert restored.revenue_mean == original.revenue_mean
        assert restored.revenue_std == original.revenue_std

    def test_seasonal_factors_mutable_default(self) -> None:
        """Each instance should get its own seasonal_factors list."""
        d1 = CashflowDistributions()
        d2 = CashflowDistributions()
        d1.seasonal_factors[0] = 999.0
        assert d2.seasonal_factors[0] == 1.0


# ---------------------------------------------------------------------------
# fit_from_erp_data
# ---------------------------------------------------------------------------

class TestFitFromErpData:
    """Test distribution fitting from synthetic ERP data."""

    def test_basic_fit(self) -> None:
        data = _make_monthly_data(12)
        metrics = _make_metrics(12)
        dist = fit_from_erp_data(data, metrics)

        assert dist.revenue_mean > 0
        assert dist.revenue_std > 0
        assert len(dist.seasonal_factors) == 12

    def test_revenue_statistics_correct(self) -> None:
        """Revenue mean should be the average of monthly sales_gross."""
        data = _make_monthly_data(12)
        metrics = _make_metrics(12)
        dist = fit_from_erp_data(data, metrics)

        expected_mean = sum(r.sales_gross for r in data.records) / 12
        assert abs(dist.revenue_mean - expected_mean) < 0.01

    def test_empty_data_returns_defaults(self) -> None:
        data = MonthlyData(records=[])
        metrics = _make_metrics()
        dist = fit_from_erp_data(data, metrics)

        assert dist.revenue_mean == 0.0
        assert dist.revenue_std == 0.0

    def test_single_month_std_fallback(self) -> None:
        """With only 1 month, std should be 15% of mean."""
        data = MonthlyData(records=[
            MonthlyRecord(year=2023, month=1, sales_gross=100_000.0),
        ])
        metrics = _make_metrics()
        dist = fit_from_erp_data(data, metrics)

        assert dist.revenue_mean == 100_000.0
        assert abs(dist.revenue_std - 15_000.0) < 0.01

    def test_credit_note_probability(self) -> None:
        """Months with credit_notes > 0 contribute to probability."""
        data = _make_monthly_data(12)
        metrics = _make_metrics()
        dist = fit_from_erp_data(data, metrics)

        # In our synthetic data: months 0,3,6,9 have credit_notes > 0
        # All 12 months have sales_gross > 0
        expected_prob = 4 / 12
        assert abs(dist.credit_note_probability - expected_prob) < 0.01

    def test_credit_note_ratio(self) -> None:
        data = _make_monthly_data(12)
        metrics = _make_metrics()
        dist = fit_from_erp_data(data, metrics)

        total_sales = sum(r.sales_gross for r in data.records)
        total_credits = sum(r.credit_notes for r in data.records)
        expected_ratio = total_credits / total_sales
        assert abs(dist.credit_note_ratio - expected_ratio) < 0.0001

    def test_seasonal_factors_from_metrics(self) -> None:
        data = _make_monthly_data(12)
        metrics = _make_metrics(12)
        dist = fit_from_erp_data(data, metrics)

        assert dist.seasonal_factors == metrics.seasonal_indices

    def test_seasonal_padding_when_short(self) -> None:
        """If metrics has <12 seasonal indices, pad with 1.0."""
        data = _make_monthly_data(6)
        metrics = _make_metrics(6)  # Only 6 indices
        dist = fit_from_erp_data(data, metrics)

        assert len(dist.seasonal_factors) == 12
        assert dist.seasonal_factors[6:] == [1.0] * 6

    def test_seasonal_truncation_when_long(self) -> None:
        """If metrics has >12 seasonal indices, truncate to 12."""
        data = _make_monthly_data(12)
        metrics = BusinessMetrics(
            seasonal_indices=[1.0] * 15,
        )
        dist = fit_from_erp_data(data, metrics)

        assert len(dist.seasonal_factors) == 12

    def test_no_seasonal_indices_defaults(self) -> None:
        data = _make_monthly_data(12)
        metrics = BusinessMetrics(seasonal_indices=[])
        dist = fit_from_erp_data(data, metrics)

        assert dist.seasonal_factors == [1.0] * 12

    def test_estimated_parameters_are_defaults(self) -> None:
        """Parameters with no data source should use documented defaults."""
        data = _make_monthly_data(12)
        metrics = _make_metrics()
        dist = fit_from_erp_data(data, metrics)

        assert dist.expense_ratio_mean == 0.72
        assert dist.expense_ratio_std == 0.05
        assert dist.collection_delay_mean == 52.0
        assert dist.collection_delay_std == 15.0
        assert dist.customer_loss_rate == 0.02


# ---------------------------------------------------------------------------
# build_scenario
# ---------------------------------------------------------------------------

class TestBuildScenario:
    """Test scenario variants of base distributions."""

    @pytest.fixture
    def base_dist(self) -> CashflowDistributions:
        return CashflowDistributions(
            revenue_mean=100_000.0,
            revenue_std=20_000.0,
            seasonal_factors=[1.0] * 12,
            expense_ratio_mean=0.72,
            expense_ratio_std=0.05,
            collection_delay_mean=52.0,
            collection_delay_std=15.0,
            credit_note_probability=0.046,
            credit_note_ratio=0.05,
            customer_loss_rate=0.02,
        )

    def test_base_scenario_is_copy(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "base")
        assert result.revenue_mean == base_dist.revenue_mean
        assert result.revenue_std == base_dist.revenue_std
        # Verify it's a copy, not the same object
        result.revenue_mean = 999
        assert base_dist.revenue_mean == 100_000.0

    def test_base_scenario_seasonal_copy(self, base_dist: CashflowDistributions) -> None:
        """Seasonal factors should be a new list (not shared reference)."""
        result = build_scenario(base_dist, "base")
        result.seasonal_factors[0] = 999.0
        assert base_dist.seasonal_factors[0] == 1.0

    def test_optimistic_revenue_up(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "optimistic")
        assert abs(result.revenue_mean - 110_000.0) < 0.01  # +10%

    def test_optimistic_volatility_down(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "optimistic")
        assert abs(result.revenue_std - 16_000.0) < 0.01  # -20%

    def test_optimistic_expense_ratio_down(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "optimistic")
        assert abs(result.expense_ratio_mean - 0.69) < 0.001  # -3pp

    def test_optimistic_credit_note_down(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "optimistic")
        assert abs(result.credit_note_probability - 0.0368) < 0.001  # × 0.8

    def test_optimistic_customer_loss_halved(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "optimistic")
        assert abs(result.customer_loss_rate - 0.01) < 0.001

    def test_stress_revenue_down(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "stress")
        assert abs(result.revenue_mean - 85_000.0) < 0.01  # -15%

    def test_stress_volatility_up(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "stress")
        assert abs(result.revenue_std - 26_000.0) < 0.01  # +30%

    def test_stress_collection_delay_up(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "stress")
        assert abs(result.collection_delay_mean - 67.0) < 0.01  # +15 days

    def test_stress_customer_loss_doubled(self, base_dist: CashflowDistributions) -> None:
        result = build_scenario(base_dist, "stress")
        assert abs(result.customer_loss_rate - 0.04) < 0.001

    def test_stress_credit_note_capped(self) -> None:
        """Credit note probability should be capped at 1.0."""
        extreme = CashflowDistributions(credit_note_probability=0.80)
        result = build_scenario(extreme, "stress")
        assert result.credit_note_probability <= 1.0

    def test_unknown_scenario_raises(self, base_dist: CashflowDistributions) -> None:
        with pytest.raises(ValueError, match="Unknown scenario"):
            build_scenario(base_dist, "catastrophic")
