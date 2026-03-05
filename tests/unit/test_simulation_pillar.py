"""Unit tests for pillars.simulation — SimulationPillar integration.

Uses in-memory PICPBus (redis=None), no LLM calls.

Tests cover:
- Full pillar pipeline (parse -> multi-scenario -> run)
- Cashflow query with reasoning results
- No reasoning output — graceful fallback
- PICP lifecycle (vector clock, events, timing)
- Distribution update
- Properties accessible
- Result structure validation
"""

from __future__ import annotations

from typing import Any

import pytest

from picp.bus import PICPBus
from picp.message import PICPContext
from pillars.simulation import SimulationPillar, SimulationResult
from pillars.simulation.distributions import CashflowDistributions
from pillars.simulation.monte_carlo import MonteCarloEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus() -> PICPBus:
    return PICPBus(redis=None)


@pytest.fixture
def base_distributions() -> CashflowDistributions:
    return CashflowDistributions(
        revenue_mean=100_000.0,
        revenue_std=20_000.0,
        seasonal_factors=[1.0] * 12,
        expense_ratio_mean=0.72,
        expense_ratio_std=0.05,
        credit_note_probability=0.046,
        credit_note_ratio=0.05,
        customer_loss_rate=0.02,
    )


@pytest.fixture
def simulation(
    bus: PICPBus, base_distributions: CashflowDistributions
) -> SimulationPillar:
    return SimulationPillar(
        bus=bus,
        base_distributions=base_distributions,
        n_simulations=1_000,  # Smaller for fast tests
        random_seed=42,
        initial_balance=0.0,
        start_month=1,
    )


def _make_context(
    query: str,
    reasoning: dict[str, Any] | None = None,
) -> PICPContext:
    """Helper to create a PICP context with optional reasoning results."""
    ctx = PICPContext.new(query)
    if reasoning:
        ctx.pillar_results["reasoning"] = reasoning
    return ctx


def _valid_reasoning() -> dict[str, Any]:
    """A realistic reasoning result for cashflow_forecast."""
    return {
        "routing": {
            "query_type": "cashflow_forecast",
            "confidence": 0.67,
            "matched_keywords": ["cashflow", "forecast"],
            "skill_name": "cashflow_forecast",
        },
        "skill_result": {
            "skill_name": "cashflow_forecast",
            "success": True,
            "parsed_output": {
                "revenue_trend": "stable",
                "risk_level": "medium",
                "adjustment_factor": 1.0,
                "time_horizon_months": 3,
            },
            "timing_ms": 100.0,
            "warnings": [],
        },
    }


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class TestSimulationPipeline:
    """Full simulation pillar pipeline tests."""

    @pytest.mark.asyncio
    async def test_cashflow_query_with_reasoning(
        self, simulation: SimulationPillar
    ) -> None:
        ctx = _make_context(
            "πρόβλεψη ταμειακής ροής",
            reasoning=_valid_reasoning(),
        )

        result = await simulation.process(ctx)

        assert "scenarios" in result
        assert "base" in result["scenarios"]
        assert "optimistic" in result["scenarios"]
        assert "stress" in result["scenarios"]

    @pytest.mark.asyncio
    async def test_scenarios_have_stats(
        self, simulation: SimulationPillar
    ) -> None:
        ctx = _make_context(
            "cashflow forecast",
            reasoning=_valid_reasoning(),
        )

        result = await simulation.process(ctx)

        for scenario_name, scenario_data in result["scenarios"].items():
            assert "monthly_stats" in scenario_data
            assert "cumulative_stats" in scenario_data
            assert "probability_negative" in scenario_data
            assert "var_5pct" in scenario_data
            assert len(scenario_data["monthly_stats"]) == 3

    @pytest.mark.asyncio
    async def test_parsed_reasoning_in_result(
        self, simulation: SimulationPillar
    ) -> None:
        ctx = _make_context(
            "cashflow forecast",
            reasoning=_valid_reasoning(),
        )

        result = await simulation.process(ctx)

        assert "parsed_reasoning" in result
        assert result["parsed_reasoning"]["query_type"] == "cashflow_forecast"

    @pytest.mark.asyncio
    async def test_warnings_accumulated(
        self, simulation: SimulationPillar
    ) -> None:
        ctx = _make_context(
            "cashflow forecast",
            reasoning=_valid_reasoning(),
        )

        result = await simulation.process(ctx)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)


# ---------------------------------------------------------------------------
# No Reasoning — Graceful Fallback
# ---------------------------------------------------------------------------

class TestNoReasoningFallback:
    """When no reasoning output is available."""

    @pytest.mark.asyncio
    async def test_runs_without_reasoning(
        self, simulation: SimulationPillar
    ) -> None:
        """Simulation should still produce results without reasoning input."""
        ctx = _make_context("cashflow forecast")

        result = await simulation.process(ctx)

        assert "scenarios" in result
        assert len(result["scenarios"]) == 3

    @pytest.mark.asyncio
    async def test_warning_when_no_reasoning(
        self, simulation: SimulationPillar
    ) -> None:
        ctx = _make_context("cashflow forecast")

        result = await simulation.process(ctx)

        assert any("No reasoning output" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_empty_reasoning_dict(
        self, simulation: SimulationPillar
    ) -> None:
        ctx = _make_context("cashflow forecast", reasoning={})

        result = await simulation.process(ctx)

        # Should still work — skill_result missing triggers default path
        assert "scenarios" in result


# ---------------------------------------------------------------------------
# PICP Lifecycle
# ---------------------------------------------------------------------------

class TestPICPLifecycle:
    """PICP vector clock, events, and timing."""

    @pytest.mark.asyncio
    async def test_vector_clock_incremented(
        self, simulation: SimulationPillar
    ) -> None:
        ctx = _make_context("cashflow forecast", reasoning=_valid_reasoning())
        initial_vc = dict(ctx.vector_clock)

        await simulation.process(ctx)

        assert ctx.vector_clock.get("simulation", 0) > initial_vc.get("simulation", 0)

    @pytest.mark.asyncio
    async def test_result_stored_in_context(
        self, simulation: SimulationPillar
    ) -> None:
        ctx = _make_context("cashflow forecast", reasoning=_valid_reasoning())

        await simulation.process(ctx)

        assert "simulation" in ctx.pillar_results
        assert "scenarios" in ctx.pillar_results["simulation"]

    @pytest.mark.asyncio
    async def test_timing_recorded(
        self, simulation: SimulationPillar
    ) -> None:
        ctx = _make_context("cashflow forecast", reasoning=_valid_reasoning())

        await simulation.process(ctx)

        assert "timings" in ctx.metadata
        assert "simulation" in ctx.metadata["timings"]
        assert ctx.metadata["timings"]["simulation"] > 0


# ---------------------------------------------------------------------------
# Distribution management
# ---------------------------------------------------------------------------

class TestDistributionManagement:
    """Distribution update and access."""

    def test_base_distributions_accessible(
        self, simulation: SimulationPillar
    ) -> None:
        assert simulation.base_distributions.revenue_mean == 100_000.0

    def test_engine_accessible(
        self, simulation: SimulationPillar
    ) -> None:
        assert isinstance(simulation.engine, MonteCarloEngine)

    def test_set_distributions(
        self, simulation: SimulationPillar
    ) -> None:
        new_dist = CashflowDistributions(
            revenue_mean=200_000.0, revenue_std=40_000.0,
        )
        simulation.set_distributions(new_dist)
        assert simulation.base_distributions.revenue_mean == 200_000.0

    @pytest.mark.asyncio
    async def test_updated_distributions_used(
        self, simulation: SimulationPillar
    ) -> None:
        """After updating distributions, simulation should use new values."""
        new_dist = CashflowDistributions(
            revenue_mean=200_000.0, revenue_std=40_000.0,
        )
        simulation.set_distributions(new_dist)

        ctx = _make_context("cashflow forecast", reasoning=_valid_reasoning())
        result = await simulation.process(ctx)

        # With 2× revenue, base scenario should produce ~2× net cashflow
        base_mean = result["scenarios"]["base"]["monthly_stats"][0]["mean"]
        assert base_mean > 40_000  # Rough check: 200K × (1 - 0.72) ≈ 56K


# ---------------------------------------------------------------------------
# Reproducibility at pillar level
# ---------------------------------------------------------------------------

class TestPillarReproducibility:
    """Same config -> same results from the pillar."""

    @pytest.mark.asyncio
    async def test_deterministic_results(
        self, bus: PICPBus, base_distributions: CashflowDistributions
    ) -> None:
        sim1 = SimulationPillar(
            bus=bus, base_distributions=base_distributions,
            n_simulations=500, random_seed=42,
        )
        sim2 = SimulationPillar(
            bus=bus, base_distributions=base_distributions,
            n_simulations=500, random_seed=42,
        )

        ctx1 = _make_context("cashflow", reasoning=_valid_reasoning())
        ctx2 = _make_context("cashflow", reasoning=_valid_reasoning())

        r1 = await sim1.process(ctx1)
        r2 = await sim2.process(ctx2)

        # Same base scenario mean
        mean1 = r1["scenarios"]["base"]["monthly_stats"][0]["mean"]
        mean2 = r2["scenarios"]["base"]["monthly_stats"][0]["mean"]
        assert mean1 == mean2
