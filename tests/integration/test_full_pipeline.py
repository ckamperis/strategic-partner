"""Integration tests — full K -> R -> S -> T pipeline.

These tests run the COMPLETE pipeline end-to-end using MockLLMClient.
They verify that all four pillars wire together correctly through PICP,
not API quality (that's for experiments).

Marked as integration tests but still use MockLLMClient — no API keys needed.
"""

from __future__ import annotations

import pytest

from orchestrator import PartnerResponse, StrategicPartner
from picp.bus import PICPBus
from pillars.simulation.distributions import CashflowDistributions
from utils.llm import MockLLMClient


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def llm_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def bus() -> PICPBus:
    return PICPBus(redis=None)


@pytest.fixture
def partner(llm_client, bus, tmp_path) -> StrategicPartner:
    """Full StrategicPartner with all pillars wired."""
    return StrategicPartner(
        llm_client=llm_client,
        bus=bus,
        base_distributions=CashflowDistributions(),
        n_simulations=500,  # Moderate for integration tests
        random_seed=42,
        audit_dir=str(tmp_path / "audit"),
    )


# ── Full Pipeline Tests ─────────────────────────────────────


@pytest.mark.integration
class TestFullPipeline:
    """End-to-end tests with MockLLMClient."""

    async def test_cashflow_query_full_pipeline(self, partner):
        """Cashflow query -> Knowledge -> Reasoning -> Simulation -> Trust."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών 3 μηνών")
        assert response.query_type == "cashflow_forecast"
        assert response.trust_score > 0
        assert response.simulation_summary is not None
        assert response.pillar_timings["total_ms"] > 0
        assert len(response.degradation_flags) == 0

    async def test_risk_query(self, partner):
        """Risk assessment goes through full pipeline with simulation."""
        response = await partner.query("Ποιοι είναι οι βασικοί κίνδυνοι;")
        assert response.query_type == "risk_assessment"
        assert response.simulation_summary is not None
        assert response.trust_score > 0

    async def test_swot_query_no_simulation(self, partner):
        """SWOT analysis does NOT trigger simulation."""
        response = await partner.query("Κάνε SWOT ανάλυση")
        assert response.query_type == "swot_analysis"
        assert response.simulation_summary is None
        assert response.trust_score > 0

    async def test_customer_query(self, partner):
        """Customer analysis — no simulation."""
        response = await partner.query("Ανάλυση πελατολογίου")
        assert response.query_type == "customer_analysis"
        assert response.simulation_summary is None

    async def test_general_query(self, partner):
        """General/unrecognised query handled gracefully."""
        response = await partner.query("Καλημέρα, πώς λειτουργείς;")
        assert response.query_type == "general"
        assert response.simulation_summary is None
        assert len(response.degradation_flags) == 0


# ── PICP Verification ────────────────────────────────────────


@pytest.mark.integration
class TestPICPIntegration:
    """Verify PICP coordination across the full pipeline."""

    async def test_vector_clock_progression(self, partner):
        """Vector clock incremented by all active pillars."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        clock = response.vector_clock
        assert clock.get("knowledge", 0) >= 1
        assert clock.get("reasoning", 0) >= 1
        assert clock.get("simulation", 0) >= 1
        assert clock.get("trust", 0) >= 1

    async def test_event_sequence(self, partner):
        """Events are published in correct PICP order."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        events = await partner._bus.get_event_log()
        event_names = [e["event"] for e in events]

        # Check ordering (not necessarily contiguous due to start/complete pairs)
        qr_idx = event_names.index("query_received")
        ku_idx = event_names.index("knowledge_updated")
        rc_idx = event_names.index("reasoning_complete")
        sr_idx = event_names.index("simulation_ready")
        tv_idx = event_names.index("trust_validated")
        rr_idx = event_names.index("response_ready")

        assert qr_idx < ku_idx < rc_idx < sr_idx < tv_idx < rr_idx

    async def test_correlation_id_consistent(self, partner):
        """All events for a query share the same correlation_id."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        events = await partner._bus.get_event_log()
        # All events should have the same correlation_id
        cids = {e["correlation_id"] for e in events}
        assert len(cids) == 1  # Single query -> single correlation_id


# ── Multiple Queries ─────────────────────────────────────────


@pytest.mark.integration
class TestMultipleQueries:
    """Multiple queries through the same partner instance."""

    async def test_two_queries_independent(self, partner):
        """Two queries produce independent responses."""
        r1 = await partner.query("Πρόβλεψη ταμειακών ροών")
        r2 = await partner.query("Κάνε SWOT ανάλυση")
        assert r1.query_type != r2.query_type
        assert r1.simulation_summary is not None
        assert r2.simulation_summary is None

    async def test_queries_dont_pollute(self, partner):
        """Second query has fresh PICP context."""
        r1 = await partner.query("Πρόβλεψη ταμειακών ροών")
        r2 = await partner.query("Πρόβλεψη ταμειακών ροών")
        # Both should succeed independently
        assert r1.query_type == r2.query_type == "cashflow_forecast"
        assert len(r1.degradation_flags) == 0
        assert len(r2.degradation_flags) == 0


# ── Response Completeness ────────────────────────────────────


@pytest.mark.integration
class TestResponseCompleteness:
    """Verify all response fields are populated correctly."""

    async def test_cashflow_response_fields(self, partner):
        """Cashflow response has all expected fields."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        assert response.query
        assert response.query_type
        assert response.answer
        assert response.confidence in ("high", "medium", "low")
        assert 0.0 <= response.trust_score <= 1.0
        assert isinstance(response.explanation, dict)
        assert isinstance(response.simulation_summary, dict)
        assert isinstance(response.factors, list)
        assert isinstance(response.caveats, list)
        assert isinstance(response.pillar_timings, dict)
        assert isinstance(response.vector_clock, dict)

    async def test_simulation_summary_has_scenarios(self, partner):
        """Simulation summary contains base/optimistic/stress."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        sim = response.simulation_summary
        assert sim is not None
        assert "base" in sim
        assert "optimistic" in sim
        assert "stress" in sim

    async def test_scenario_metrics(self, partner):
        """Each scenario has mean, prob_neg, var."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        for name, metrics in response.simulation_summary.items():
            assert "mean_month1" in metrics
            assert "probability_negative" in metrics
            assert "var_5pct" in metrics
