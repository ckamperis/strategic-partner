"""Unit tests for the central Orchestrator (StrategicPartner).

Tests the K -> R -> S -> T pipeline wiring, graceful degradation,
PICP lifecycle (vector clock, events), and response assembly.

All tests use MockLLMClient — no API keys needed.
"""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator import PartnerResponse, StrategicPartner
from picp.bus import PICPBus
from picp.message import PICPContext
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
def audit_dir(tmp_path):
    return str(tmp_path / "audit")


@pytest.fixture
def partner(llm_client, bus, audit_dir) -> StrategicPartner:
    return StrategicPartner(
        llm_client=llm_client,
        bus=bus,
        base_distributions=CashflowDistributions(),
        n_simulations=100,  # Small for fast tests
        random_seed=42,
        audit_dir=audit_dir,
    )


# ── Test: Full Pipeline ─────────────────────────────────────


class TestFullPipeline:
    """Test the complete K -> R -> S -> T pipeline."""

    async def test_cashflow_query_produces_response(self, partner):
        """Cashflow query -> all 4 pillars execute -> valid PartnerResponse."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών 3 μηνών")
        assert isinstance(response, PartnerResponse)
        assert response.query == "Πρόβλεψη ταμειακών ροών 3 μηνών"
        assert response.query_type == "cashflow_forecast"

    async def test_cashflow_runs_simulation(self, partner):
        """Cashflow query triggers Simulation Pillar."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        assert response.simulation_summary is not None
        assert "base" in response.simulation_summary

    async def test_risk_query_runs_simulation(self, partner):
        """Risk assessment also triggers Simulation Pillar."""
        response = await partner.query("Ποιοι είναι οι βασικοί κίνδυνοι;")
        assert response.query_type == "risk_assessment"
        assert response.simulation_summary is not None

    async def test_trust_score_present(self, partner):
        """Trust score is always present and bounded."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        assert 0.0 <= response.trust_score <= 1.0
        assert response.confidence in ("high", "medium", "low")

    async def test_answer_is_nonempty(self, partner):
        """Answer field is always populated."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        assert len(response.answer) > 0


# ── Test: Simulation Skip ────────────────────────────────────


class TestSimulationSkip:
    """Simulation Pillar is skipped for non-forecast queries."""

    async def test_swot_skips_simulation(self, partner):
        """SWOT query -> simulation_summary is None."""
        response = await partner.query("Κάνε SWOT ανάλυση")
        assert response.query_type == "swot_analysis"
        assert response.simulation_summary is None

    async def test_general_skips_simulation(self, partner):
        """General query -> simulation_summary is None."""
        response = await partner.query("Καλημέρα, πώς λειτουργείς;")
        assert response.query_type == "general"
        assert response.simulation_summary is None

    async def test_customer_skips_simulation(self, partner):
        """Customer analysis -> simulation_summary is None."""
        response = await partner.query("Ανάλυση πελατολογίου")
        assert response.query_type == "customer_analysis"
        assert response.simulation_summary is None


# ── Test: Graceful Degradation ───────────────────────────────


class TestGracefulDegradation:
    """Pipeline continues when individual pillars fail."""

    async def test_knowledge_failure_continues(self, partner):
        """Knowledge failure -> pipeline continues with flag."""
        # Patch _execute to raise
        with patch.object(
            partner._knowledge, "_execute", side_effect=RuntimeError("mock knowledge fail")
        ):
            response = await partner.query("Πρόβλεψη ταμειακών ροών")

        assert "knowledge_failed" in response.degradation_flags
        assert response.answer  # Still has an answer from Reasoning

    async def test_reasoning_failure_continues(self, partner):
        """Reasoning failure -> pipeline continues with flag."""
        with patch.object(
            partner._reasoning, "_execute", side_effect=RuntimeError("mock reasoning fail")
        ):
            response = await partner.query("Πρόβλεψη ταμειακών ροών")

        assert "reasoning_failed" in response.degradation_flags
        # Query type falls back to general because reasoning failed
        assert response.query_type == "general"

    async def test_simulation_failure_continues(self, partner):
        """Simulation failure -> pipeline continues with flag."""
        with patch.object(
            partner._simulation, "_execute", side_effect=RuntimeError("mock sim fail")
        ):
            response = await partner.query("Πρόβλεψη ταμειακών ροών")

        assert "simulation_failed" in response.degradation_flags
        assert response.simulation_summary is None
        # Trust still runs
        assert response.trust_score >= 0.0

    async def test_trust_failure_continues(self, partner):
        """Trust failure -> pipeline continues with flag."""
        with patch.object(
            partner._trust, "_execute", side_effect=RuntimeError("mock trust fail")
        ):
            response = await partner.query("Πρόβλεψη ταμειακών ροών")

        assert "trust_failed" in response.degradation_flags
        assert response.trust_score == 0.0
        assert response.confidence == "low"

    async def test_degradation_caveats_added(self, partner):
        """Degradation flags appear in caveats."""
        with patch.object(
            partner._knowledge, "_execute", side_effect=RuntimeError("fail")
        ):
            response = await partner.query("Πρόβλεψη ταμειακών ροών")

        assert any("knowledge_failed" in c for c in response.caveats)


# ── Test: PICP Lifecycle ─────────────────────────────────────


class TestPICPLifecycle:
    """Verify PICP vector clock and event progression."""

    async def test_vector_clock_has_pillar_increments(self, partner):
        """Final vector clock reflects pillar execution."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        vc = response.vector_clock
        # All 4 pillars should have incremented their clocks
        assert vc.get("knowledge", 0) >= 1
        assert vc.get("reasoning", 0) >= 1
        assert vc.get("simulation", 0) >= 1
        assert vc.get("trust", 0) >= 1

    async def test_vector_clock_no_simulation(self, partner):
        """SWOT query -> simulation clock stays at 0."""
        response = await partner.query("Κάνε SWOT ανάλυση")
        vc = response.vector_clock
        assert vc.get("knowledge", 0) >= 1
        assert vc.get("reasoning", 0) >= 1
        assert vc.get("simulation", 0) == 0  # Not incremented
        assert vc.get("trust", 0) >= 1

    async def test_events_logged_on_bus(self, partner):
        """PICP events are published to the bus."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        # Check bus event log contains expected events
        events = await partner._bus.get_event_log()
        event_names = [e["event"] for e in events]
        assert "query_received" in event_names
        assert "knowledge_started" in event_names
        assert "knowledge_updated" in event_names
        assert "reasoning_started" in event_names
        assert "reasoning_complete" in event_names
        assert "trust_started" in event_names
        assert "trust_validated" in event_names
        assert "response_ready" in event_names


# ── Test: Timings ────────────────────────────────────────────


class TestTimings:
    """Pillar timings are recorded for experiment analysis."""

    async def test_all_timings_present(self, partner):
        """All pillar timings and total_ms are populated."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        assert "knowledge" in response.pillar_timings
        assert "reasoning" in response.pillar_timings
        assert "simulation" in response.pillar_timings
        assert "trust" in response.pillar_timings
        assert "total_ms" in response.pillar_timings

    async def test_timings_are_positive(self, partner):
        """All timing values are > 0."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        for key, val in response.pillar_timings.items():
            assert val > 0, f"Timing '{key}' should be > 0, got {val}"

    async def test_total_exceeds_parts(self, partner):
        """Total time >= sum of individual pillar times."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        parts = sum(
            v for k, v in response.pillar_timings.items() if k != "total_ms"
        )
        # Total includes orchestrator overhead
        assert response.pillar_timings["total_ms"] >= parts * 0.9  # Allow small rounding


# ── Test: Response Serialisation ─────────────────────────────


class TestSerialisation:
    """PartnerResponse can be serialised to JSON."""

    async def test_model_dump(self, partner):
        """model_dump() produces a valid dict."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        d = response.model_dump()
        assert isinstance(d, dict)
        assert "query" in d
        assert "trust_score" in d
        assert "pillar_timings" in d

    async def test_model_dump_json(self, partner):
        """model_dump_json() produces valid JSON string."""
        response = await partner.query("Πρόβλεψη ταμειακών ροών")
        import json
        j = response.model_dump_json()
        parsed = json.loads(j)
        assert parsed["query"] == "Πρόβλεψη ταμειακών ροών"


# ── Test: Properties & Configuration ─────────────────────────


class TestConfiguration:
    """Orchestrator configuration and pillar access."""

    def test_pillar_properties(self, partner):
        """All four pillars are accessible."""
        assert partner.knowledge is not None
        assert partner.reasoning is not None
        assert partner.simulation is not None
        assert partner.trust is not None

    def test_set_distributions(self, partner):
        """set_distributions updates both simulation and trust."""
        dist = CashflowDistributions(revenue_mean=200_000)
        partner.set_distributions(dist)
        assert partner.simulation.base_distributions.revenue_mean == 200_000
