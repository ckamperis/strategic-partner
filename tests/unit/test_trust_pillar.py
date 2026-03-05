"""Unit tests for pillars.trust — TrustPillar integration.

Uses in-memory PICPBus (redis=None), no LLM calls.

Tests cover:
- Full trust pipeline (evaluate -> SHAP -> explain -> audit)
- Without LLM: template mode works
- PICP lifecycle (vector clock, events, timing)
- Audit entry created
- Degraded mode: missing simulation -> still produces trust score
- Properties accessible
- Distribution update
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from data.pipeline.models import BusinessMetrics, CustomerConcentration
from picp.bus import PICPBus
from picp.message import PICPContext
from pillars.simulation.distributions import CashflowDistributions
from pillars.trust import TrustPillar, TrustResult
from pillars.trust.evaluator import TrustEvaluator


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
        credit_note_probability=0.046,
        credit_note_ratio=0.05,
    )


@pytest.fixture
def metrics() -> BusinessMetrics:
    return BusinessMetrics(
        customer_concentration=CustomerConcentration(
            top5_pct=0.45, total_customers=120,
        ),
    )


@pytest.fixture
def trust_pillar(
    bus: PICPBus,
    base_distributions: CashflowDistributions,
    metrics: BusinessMetrics,
    tmp_path: Path,
) -> TrustPillar:
    return TrustPillar(
        bus=bus,
        llm_client=None,
        audit_dir=str(tmp_path / "audit"),
        base_distributions=base_distributions,
        metrics=metrics,
    )


def _make_context(
    query: str = "cashflow forecast",
    knowledge: dict[str, Any] | None = None,
    reasoning: dict[str, Any] | None = None,
    simulation: dict[str, Any] | None = None,
) -> PICPContext:
    ctx = PICPContext.new(query)
    if knowledge:
        ctx.pillar_results["knowledge"] = knowledge
    if reasoning:
        ctx.pillar_results["reasoning"] = reasoning
    if simulation:
        ctx.pillar_results["simulation"] = simulation
    return ctx


def _good_knowledge() -> dict:
    return {
        "chunks": [{"text": f"Chunk {i}"} for i in range(6)],
        "relevance_score": 0.85,
        "iterations": 1,
    }


def _good_reasoning() -> dict:
    return {
        "routing": {
            "query_type": "cashflow_forecast",
            "confidence": 0.67,
            "skill_name": "cashflow_forecast",
        },
        "skill_result": {
            "skill_name": "cashflow_forecast",
            "success": True,
            "parsed_output": {
                "revenue_trend": "stable",
                "risk_level": "low",
            },
        },
    }


def _good_simulation() -> dict:
    return {
        "scenarios": {
            "base": {
                "monthly_stats": [
                    {"mean": 28_000, "std": 8_000, "p5": 15_000, "p95": 42_000},
                    {"mean": 29_000, "std": 7_500, "p5": 16_000, "p95": 43_000},
                    {"mean": 30_000, "std": 7_000, "p5": 17_000, "p95": 44_000},
                ],
                "probability_negative": 0.02,
                "var_5pct": 45_000,
                "n_simulations": 10_000,
                "config_snapshot": {
                    "distributions": {
                        "revenue_mean": 100_000,
                        "revenue_std": 20_000,
                        "expense_ratio_mean": 0.72,
                        "collection_delay_mean": 52.0,
                        "customer_loss_rate": 0.02,
                        "credit_note_ratio": 0.05,
                        "seasonal_factors": [1.0] * 12,
                    },
                },
            },
            "optimistic": {
                "monthly_stats": [{"mean": 35_000}],
                "n_simulations": 10_000,
            },
            "stress": {
                "monthly_stats": [{"mean": 18_000}],
                "n_simulations": 10_000,
            },
        },
    }


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class TestTrustPipeline:
    """Full trust pillar pipeline tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline_produces_result(
        self, trust_pillar: TrustPillar
    ) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )
        result = await trust_pillar.process(ctx)

        assert "trust_score" in result
        assert "explanation" in result
        assert "shap_factors" in result
        assert "audit_id" in result

    @pytest.mark.asyncio
    async def test_trust_score_in_bounds(
        self, trust_pillar: TrustPillar
    ) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )
        result = await trust_pillar.process(ctx)

        overall = result["trust_score"]["overall"]
        assert 0.0 <= overall <= 1.0

    @pytest.mark.asyncio
    async def test_shap_factors_present(
        self, trust_pillar: TrustPillar
    ) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )
        result = await trust_pillar.process(ctx)

        assert len(result["shap_factors"]) == 5

    @pytest.mark.asyncio
    async def test_explanation_has_summary(
        self, trust_pillar: TrustPillar
    ) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )
        result = await trust_pillar.process(ctx)

        assert result["explanation"]["summary"]

    @pytest.mark.asyncio
    async def test_audit_id_returned(
        self, trust_pillar: TrustPillar
    ) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )
        result = await trust_pillar.process(ctx)

        assert result["audit_id"]
        assert len(result["audit_id"]) > 0


# ---------------------------------------------------------------------------
# Degraded Mode
# ---------------------------------------------------------------------------

class TestDegradedMode:
    """Trust pillar works with missing pillar results."""

    @pytest.mark.asyncio
    async def test_no_simulation(self, trust_pillar: TrustPillar) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
        )
        result = await trust_pillar.process(ctx)

        assert "trust_score" in result
        # No simulation -> no SHAP factors
        assert len(result["shap_factors"]) == 0

    @pytest.mark.asyncio
    async def test_no_reasoning(self, trust_pillar: TrustPillar) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            simulation=_good_simulation(),
        )
        result = await trust_pillar.process(ctx)

        assert "trust_score" in result

    @pytest.mark.asyncio
    async def test_no_knowledge(self, trust_pillar: TrustPillar) -> None:
        ctx = _make_context(
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )
        result = await trust_pillar.process(ctx)

        assert "trust_score" in result

    @pytest.mark.asyncio
    async def test_empty_context(self, trust_pillar: TrustPillar) -> None:
        ctx = _make_context()
        result = await trust_pillar.process(ctx)

        assert "trust_score" in result
        assert result["trust_score"]["confidence_level"] == "low"


# ---------------------------------------------------------------------------
# PICP Lifecycle
# ---------------------------------------------------------------------------

class TestPICPLifecycle:
    """PICP vector clock, events, and timing."""

    @pytest.mark.asyncio
    async def test_vector_clock_incremented(
        self, trust_pillar: TrustPillar
    ) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )
        initial_vc = dict(ctx.vector_clock)

        await trust_pillar.process(ctx)

        assert ctx.vector_clock.get("trust", 0) > initial_vc.get("trust", 0)

    @pytest.mark.asyncio
    async def test_result_stored_in_context(
        self, trust_pillar: TrustPillar
    ) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )

        await trust_pillar.process(ctx)

        assert "trust" in ctx.pillar_results
        assert "trust_score" in ctx.pillar_results["trust"]

    @pytest.mark.asyncio
    async def test_timing_recorded(
        self, trust_pillar: TrustPillar
    ) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )

        await trust_pillar.process(ctx)

        assert "timings" in ctx.metadata
        assert "trust" in ctx.metadata["timings"]
        assert ctx.metadata["timings"]["trust"] > 0


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class TestAuditIntegration:
    """Test audit logging integration."""

    @pytest.mark.asyncio
    async def test_audit_entry_created(
        self, trust_pillar: TrustPillar
    ) -> None:
        ctx = _make_context(
            knowledge=_good_knowledge(),
            reasoning=_good_reasoning(),
            simulation=_good_simulation(),
        )
        await trust_pillar.process(ctx)

        entries = trust_pillar.auditor.get_recent(n=1)
        assert len(entries) == 1
        assert entries[0].query == "cashflow forecast"
        assert entries[0].query_type == "cashflow_forecast"

    @pytest.mark.asyncio
    async def test_multiple_queries_logged(
        self, trust_pillar: TrustPillar
    ) -> None:
        for i in range(3):
            ctx = _make_context(
                query=f"query_{i}",
                reasoning=_good_reasoning(),
                simulation=_good_simulation(),
            )
            await trust_pillar.process(ctx)

        entries = trust_pillar.auditor.get_recent(n=10)
        assert len(entries) == 3


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    """Test property access."""

    def test_evaluator_accessible(self, trust_pillar: TrustPillar) -> None:
        assert isinstance(trust_pillar.evaluator, TrustEvaluator)

    def test_auditor_accessible(self, trust_pillar: TrustPillar) -> None:
        assert trust_pillar.auditor is not None

    def test_set_distributions(self, trust_pillar: TrustPillar) -> None:
        new_dist = CashflowDistributions(
            revenue_mean=200_000, revenue_std=40_000,
        )
        trust_pillar.set_distributions(new_dist)
        # No error = success (distributions used internally)
