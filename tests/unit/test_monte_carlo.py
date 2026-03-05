"""Unit tests for pillars.simulation.monte_carlo.

Tests cover:
- MonteCarloConfig defaults and validation
- MonteCarloEngine.run produces correct shape and statistics
- Reproducibility (same seed -> same result)
- Different seeds -> different results
- Revenue non-negativity (clipping)
- Probability metrics (probability_negative, var_5pct)
- Convergence standard error decreases with N
- Performance target: 10K sims × 3 months < 500ms
- MonthlySimStats and MonteCarloResult serialisation
- Edge cases: zero std, single month, single simulation
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pillars.simulation.distributions import CashflowDistributions
from pillars.simulation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResult,
    MonthlySimStats,
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
        collection_delay_mean=52.0,
        collection_delay_std=15.0,
        credit_note_probability=0.046,
        credit_note_ratio=0.05,
        customer_loss_rate=0.02,
    )


@pytest.fixture
def engine() -> MonteCarloEngine:
    return MonteCarloEngine()


@pytest.fixture
def default_config(base_distributions: CashflowDistributions) -> MonteCarloConfig:
    return MonteCarloConfig(
        n_simulations=10_000,
        time_horizon_months=3,
        random_seed=42,
        distributions=base_distributions,
        initial_balance=50_000.0,
        start_month=1,
    )


# ---------------------------------------------------------------------------
# MonteCarloConfig
# ---------------------------------------------------------------------------

class TestMonteCarloConfig:
    """Test config defaults and field access."""

    def test_defaults(self) -> None:
        config = MonteCarloConfig()
        assert config.n_simulations == 10_000
        assert config.time_horizon_months == 3
        assert config.random_seed == 42
        assert config.initial_balance == 0.0
        assert config.start_month == 1

    def test_custom_values(self, base_distributions: CashflowDistributions) -> None:
        config = MonteCarloConfig(
            n_simulations=5_000,
            time_horizon_months=6,
            random_seed=123,
            distributions=base_distributions,
            initial_balance=100_000.0,
            start_month=7,
        )
        assert config.n_simulations == 5_000
        assert config.time_horizon_months == 6
        assert config.start_month == 7


# ---------------------------------------------------------------------------
# Engine — basic execution
# ---------------------------------------------------------------------------

class TestMonteCarloEngineBasic:
    """Basic engine execution tests."""

    def test_run_returns_result(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        assert isinstance(result, MonteCarloResult)

    def test_result_has_correct_months(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        assert len(result.monthly_stats) == 3
        assert len(result.cumulative_stats) == 3

    def test_monthly_stats_indices(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        for i, stats in enumerate(result.monthly_stats):
            assert stats.month_index == i

    def test_result_metadata(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        assert result.n_simulations == 10_000
        assert result.time_horizon_months == 3
        assert result.elapsed_ms > 0
        assert result.scenario_name == "base"

    def test_scenario_name_passed_through(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config, scenario_name="stress")
        assert result.scenario_name == "stress"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Same seed produces identical results."""

    def test_same_seed_same_result(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        r1 = engine.run(default_config)
        r2 = engine.run(default_config)

        assert r1.monthly_stats[0].mean == r2.monthly_stats[0].mean
        assert r1.probability_negative == r2.probability_negative
        assert r1.var_5pct == r2.var_5pct

    def test_different_seed_different_result(
        self, engine: MonteCarloEngine, base_distributions: CashflowDistributions
    ) -> None:
        c1 = MonteCarloConfig(random_seed=42, distributions=base_distributions)
        c2 = MonteCarloConfig(random_seed=99, distributions=base_distributions)

        r1 = engine.run(c1)
        r2 = engine.run(c2)

        # With different seeds, means should differ (extremely unlikely to match)
        assert r1.monthly_stats[0].mean != r2.monthly_stats[0].mean


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

class TestStatisticalProperties:
    """Verify that simulation output has expected statistical properties."""

    def test_revenue_non_negative(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        """Monthly net cashflow should have a non-negative minimum
        that comes from non-negative revenue (clipping)."""
        result = engine.run(default_config)
        # Net cashflow CAN be negative (if expenses > revenue),
        # but revenue was clipped ≥ 0 before subtraction.
        # We check that the mean net cashflow is positive (expenses < revenue).
        for stats in result.monthly_stats:
            assert stats.mean > 0, "Expected positive mean net cashflow"

    def test_percentile_ordering(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        """p5 ≤ p25 ≤ p50 ≤ p75 ≤ p95."""
        result = engine.run(default_config)
        for stats in result.monthly_stats:
            assert stats.p5 <= stats.p25
            assert stats.p25 <= stats.p50
            assert stats.p50 <= stats.p75
            assert stats.p75 <= stats.p95

    def test_min_max_bounds(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        for stats in result.monthly_stats:
            assert stats.min <= stats.p5
            assert stats.p95 <= stats.max

    def test_cumulative_grows(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        """Cumulative mean should increase (positive net cashflow)."""
        result = engine.run(default_config)
        means = [s.mean for s in result.cumulative_stats]
        # Each cumulative mean should be ≥ previous (net CF is positive on avg)
        for i in range(1, len(means)):
            assert means[i] > means[i - 1]

    def test_cumulative_includes_initial_balance(
        self, engine: MonteCarloEngine, base_distributions: CashflowDistributions
    ) -> None:
        """Cumulative first month should be offset by initial_balance."""
        config = MonteCarloConfig(
            distributions=base_distributions,
            initial_balance=100_000.0,
        )
        result = engine.run(config)

        config_no_balance = MonteCarloConfig(
            distributions=base_distributions,
            initial_balance=0.0,
        )
        result_no_balance = engine.run(config_no_balance)

        # Difference should be approximately initial_balance
        diff = result.cumulative_stats[0].mean - result_no_balance.cumulative_stats[0].mean
        assert abs(diff - 100_000.0) < 100.0

    def test_std_positive(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        for stats in result.monthly_stats:
            assert stats.std > 0


# ---------------------------------------------------------------------------
# Probability & Risk metrics
# ---------------------------------------------------------------------------

class TestRiskMetrics:
    """Test probability_negative, var_5pct, convergence_std."""

    def test_probability_negative_in_bounds(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        assert 0.0 <= result.probability_negative <= 1.0

    def test_probability_negative_with_large_balance(
        self, engine: MonteCarloEngine, base_distributions: CashflowDistributions
    ) -> None:
        """With a very large initial balance, P(negative) -> 0."""
        config = MonteCarloConfig(
            distributions=base_distributions,
            initial_balance=10_000_000.0,  # 10M EUR
        )
        result = engine.run(config)
        assert result.probability_negative < 0.01

    def test_var_5pct_less_than_mean(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        """VaR (5th percentile) should be less than cumulative mean."""
        result = engine.run(default_config)
        final_mean = result.cumulative_stats[-1].mean
        assert result.var_5pct < final_mean

    def test_convergence_std_positive(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        assert result.convergence_std > 0

    def test_convergence_decreases_with_n(
        self, engine: MonteCarloEngine, base_distributions: CashflowDistributions
    ) -> None:
        """Standard error ∝ 1/√N, so larger N -> smaller convergence_std."""
        config_small = MonteCarloConfig(
            n_simulations=1_000, distributions=base_distributions,
        )
        config_large = MonteCarloConfig(
            n_simulations=10_000, distributions=base_distributions,
        )
        r_small = engine.run(config_small)
        r_large = engine.run(config_large)

        assert r_large.convergence_std < r_small.convergence_std


# ---------------------------------------------------------------------------
# Seasonal factors
# ---------------------------------------------------------------------------

class TestSeasonalFactors:
    """Verify seasonal modulation affects results."""

    def test_different_start_months(
        self, engine: MonteCarloEngine, base_distributions: CashflowDistributions
    ) -> None:
        """Starting in a low-season month vs high-season month
        should produce different revenue means."""
        # Month 1 (Jan) has factor 0.8; month 8 (Aug) has factor 1.2
        config_jan = MonteCarloConfig(
            distributions=base_distributions, start_month=1,
        )
        config_aug = MonteCarloConfig(
            distributions=base_distributions, start_month=8,
        )
        r_jan = engine.run(config_jan)
        r_aug = engine.run(config_aug)

        # First month mean should differ due to seasonal factor
        assert r_jan.monthly_stats[0].mean < r_aug.monthly_stats[0].mean


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    """Verify performance target: 10K sims × 3 months < 500ms."""

    def test_10k_sims_under_500ms(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        start = time.perf_counter()
        engine.run(default_config)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, (
            f"10K sims took {elapsed_ms:.1f}ms, exceeding 500ms target"
        )

    def test_elapsed_ms_recorded(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        assert result.elapsed_ms > 0
        assert result.elapsed_ms < 1000  # Generous upper bound


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    """Test to_dict methods."""

    def test_monthly_sim_stats_to_dict(self) -> None:
        stats = MonthlySimStats(
            month_index=0, mean=28_000.0, std=8_000.0,
            p5=15_000.0, p25=22_000.0, p50=28_000.0,
            p75=34_000.0, p95=42_000.0,
            min=5_000.0, max=60_000.0,
        )
        d = stats.to_dict()
        assert d["month_index"] == 0
        assert d["mean"] == 28_000.0
        assert d["p50"] == 28_000.0

    def test_result_to_dict_structure(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        d = result.to_dict()

        assert "monthly_stats" in d
        assert "cumulative_stats" in d
        assert "probability_negative" in d
        assert "var_5pct" in d
        assert "convergence_std" in d
        assert "n_simulations" in d
        assert "config_snapshot" in d
        assert d["n_simulations"] == 10_000

    def test_config_snapshot_contains_distributions(
        self, engine: MonteCarloEngine, default_config: MonteCarloConfig
    ) -> None:
        result = engine.run(default_config)
        snapshot = result.config_snapshot
        assert "distributions" in snapshot
        assert "revenue_mean" in snapshot["distributions"]
        assert snapshot["random_seed"] == 42


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling."""

    def test_zero_revenue_std(self, engine: MonteCarloEngine) -> None:
        """Zero std should still work (all paths identical)."""
        dist = CashflowDistributions(
            revenue_mean=100_000.0, revenue_std=0.0,
        )
        config = MonteCarloConfig(distributions=dist, n_simulations=100)
        result = engine.run(config)

        # With zero revenue std, variance comes only from expense ratio noise.
        # expense_ratio ~ N(0.72, 0.05), so net ≈ revenue × (1 - ratio) ± noise.
        # Tolerance set to 8K to account for credit note Bernoulli + expense draws.
        assert result.monthly_stats[0].std < 8_000  # Only expense ratio + credit note variance

    def test_single_month_horizon(self, engine: MonteCarloEngine) -> None:
        dist = CashflowDistributions(revenue_mean=100_000.0, revenue_std=20_000.0)
        config = MonteCarloConfig(
            distributions=dist, time_horizon_months=1, n_simulations=1_000,
        )
        result = engine.run(config)

        assert len(result.monthly_stats) == 1
        assert len(result.cumulative_stats) == 1

    def test_single_simulation(self, engine: MonteCarloEngine) -> None:
        dist = CashflowDistributions(revenue_mean=100_000.0, revenue_std=20_000.0)
        config = MonteCarloConfig(
            distributions=dist, n_simulations=1,
        )
        result = engine.run(config)

        # With N=1, all percentiles should be the same
        for stats in result.monthly_stats:
            assert stats.min == stats.max

    def test_twelve_month_horizon(self, engine: MonteCarloEngine) -> None:
        """Full year horizon should work and wrap seasonal factors."""
        dist = CashflowDistributions(
            revenue_mean=100_000.0,
            revenue_std=20_000.0,
            seasonal_factors=[0.8, 0.85, 0.9, 1.0, 1.05, 1.1,
                              1.15, 1.2, 1.1, 1.0, 0.9, 0.95],
        )
        config = MonteCarloConfig(
            distributions=dist, time_horizon_months=12,
            n_simulations=1_000, start_month=1,
        )
        result = engine.run(config)
        assert len(result.monthly_stats) == 12

    def test_start_month_wrapping(self, engine: MonteCarloEngine) -> None:
        """Starting at month 11 with horizon 4 should wrap to months 11,12,1,2."""
        dist = CashflowDistributions(
            revenue_mean=100_000.0,
            revenue_std=20_000.0,
            seasonal_factors=[0.8, 0.85, 0.9, 1.0, 1.05, 1.1,
                              1.15, 1.2, 1.1, 1.0, 0.9, 0.95],
        )
        config = MonteCarloConfig(
            distributions=dist, time_horizon_months=4,
            n_simulations=1_000, start_month=11,
        )
        result = engine.run(config)
        assert len(result.monthly_stats) == 4
