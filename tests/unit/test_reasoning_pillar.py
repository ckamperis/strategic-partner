"""Unit tests for pillars.reasoning — ReasoningPillar integration.

Uses MockLLMClient and in-memory PICPBus.

Tests cover:
- Full pillar pipeline (classify -> skill lookup -> execute)
- Cashflow query routes correctly
- Risk query routes correctly
- General query handling (no matching skill)
- Knowledge context extraction from PICP context
- PICP lifecycle (vector clock increment, events)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from picp.bus import PICPBus
from picp.message import PICPContext, PICPEvent
from pillars.reasoning import ReasoningPillar
from pillars.reasoning.heuristic_policy import QueryType
from utils.llm import MockLLMClient


SKILLS_DIR = Path(__file__).parent.parent.parent / "pillars" / "reasoning" / "skills"


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient(embedding_dim=1536)


@pytest.fixture
def bus() -> PICPBus:
    return PICPBus(redis=None)


@pytest.fixture
def reasoning(bus: PICPBus, mock_llm: MockLLMClient) -> ReasoningPillar:
    return ReasoningPillar(
        bus=bus,
        llm_client=mock_llm,
        skills_dir=SKILLS_DIR,
        model_fast="mock-fast",
        model_strong="mock-strong",
    )


def _make_context(query: str, knowledge: dict[str, Any] | None = None) -> PICPContext:
    """Helper to create a PICP context with optional knowledge results."""
    ctx = PICPContext.new(query)
    if knowledge:
        ctx.pillar_results["knowledge"] = knowledge
    return ctx


class TestReasoningPipeline:
    """Full reasoning pillar pipeline tests."""

    @pytest.mark.asyncio
    async def test_cashflow_query_routes_and_executes(
        self, reasoning: ReasoningPillar
    ) -> None:
        ctx = _make_context(
            "πρόβλεψη ταμειακής ροής",
            knowledge={"chunks": [{"text": "Πωλήσεις €150,000"}]},
        )

        result = await reasoning.process(ctx)

        assert "routing" in result
        assert result["routing"]["query_type"] == "cashflow_forecast"
        assert result["routing"]["skill_name"] == "cashflow_forecast"
        assert "skill_result" in result
        assert result["skill_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_risk_query_routes_and_executes(
        self, reasoning: ReasoningPillar
    ) -> None:
        ctx = _make_context(
            "risk assessment credit exposure",
            knowledge={"chunks": [{"text": "Credit risk: 8.5%"}]},
        )

        result = await reasoning.process(ctx)

        assert result["routing"]["query_type"] == "risk_assessment"
        assert result["skill_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_swot_query_routes_and_executes(
        self, reasoning: ReasoningPillar
    ) -> None:
        ctx = _make_context(
            "swot analysis strengths weaknesses",
            knowledge={"chunks": [{"text": "Revenue growing"}]},
        )

        result = await reasoning.process(ctx)

        assert result["routing"]["query_type"] == "swot_analysis"
        assert result["skill_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_customer_query_routes_and_executes(
        self, reasoning: ReasoningPillar
    ) -> None:
        ctx = _make_context(
            "customer concentration segmentation",
            knowledge={"chunks": [{"text": "Top 5% = 45% revenue"}]},
        )

        result = await reasoning.process(ctx)

        assert result["routing"]["query_type"] == "customer_analysis"
        assert result["skill_result"]["success"] is True


class TestGeneralQueryHandling:
    """Queries that don't match any skill."""

    @pytest.mark.asyncio
    async def test_general_query_returns_passthrough(
        self, reasoning: ReasoningPillar
    ) -> None:
        ctx = _make_context("What is the meaning of life?")

        result = await reasoning.process(ctx)

        assert result["routing"]["query_type"] == "general"
        assert result["routing"]["skill_name"] is None
        assert result["skill_result"]["skill_name"] == "general"
        assert result["skill_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_empty_query_is_general(
        self, reasoning: ReasoningPillar
    ) -> None:
        ctx = _make_context("")

        result = await reasoning.process(ctx)

        assert result["routing"]["query_type"] == "general"


class TestKnowledgeContextExtraction:
    """Context string building from knowledge pillar results."""

    @pytest.mark.asyncio
    async def test_with_dict_chunks(self, reasoning: ReasoningPillar) -> None:
        ctx = _make_context(
            "cashflow forecast",
            knowledge={
                "chunks": [
                    {"text": "Chunk 1 text"},
                    {"text": "Chunk 2 text"},
                ]
            },
        )

        result = await reasoning.process(ctx)
        assert result["skill_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_with_string_chunks(self, reasoning: ReasoningPillar) -> None:
        ctx = _make_context(
            "cashflow forecast",
            knowledge={"chunks": ["Plain text chunk 1", "Plain text chunk 2"]},
        )

        result = await reasoning.process(ctx)
        assert result["skill_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_with_no_knowledge(self, reasoning: ReasoningPillar) -> None:
        """No knowledge context available — skill still executes."""
        ctx = _make_context("cashflow forecast")

        result = await reasoning.process(ctx)
        assert result["skill_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_with_empty_chunks(self, reasoning: ReasoningPillar) -> None:
        ctx = _make_context(
            "cashflow forecast",
            knowledge={"chunks": []},
        )

        result = await reasoning.process(ctx)
        assert result["skill_result"]["success"] is True

    def test_build_context_string_no_knowledge(self) -> None:
        ctx = PICPContext.new("test")
        result = ReasoningPillar._build_context_string(ctx)
        assert result == "No knowledge context available."

    def test_build_context_string_with_chunks(self) -> None:
        ctx = PICPContext.new("test")
        ctx.pillar_results["knowledge"] = {
            "chunks": [
                {"text": "First"},
                {"text": "Second"},
            ]
        }
        result = ReasoningPillar._build_context_string(ctx)
        assert "[1] First" in result
        assert "[2] Second" in result


class TestPICPLifecycle:
    """PICP vector clock and event integration."""

    @pytest.mark.asyncio
    async def test_vector_clock_incremented(
        self, reasoning: ReasoningPillar
    ) -> None:
        ctx = _make_context("cashflow forecast")
        initial_vc = dict(ctx.vector_clock)

        await reasoning.process(ctx)

        # Reasoning pillar should have incremented its clock
        assert ctx.vector_clock.get("reasoning", 0) > initial_vc.get("reasoning", 0)

    @pytest.mark.asyncio
    async def test_result_stored_in_context(
        self, reasoning: ReasoningPillar
    ) -> None:
        ctx = _make_context("cashflow forecast")

        await reasoning.process(ctx)

        assert "reasoning" in ctx.pillar_results
        assert "routing" in ctx.pillar_results["reasoning"]

    @pytest.mark.asyncio
    async def test_timing_recorded(
        self, reasoning: ReasoningPillar
    ) -> None:
        ctx = _make_context("cashflow forecast")

        await reasoning.process(ctx)

        assert "timings" in ctx.metadata
        assert "reasoning" in ctx.metadata["timings"]
        assert ctx.metadata["timings"]["reasoning"] > 0


class TestReasoningPillarProperties:
    """Access to internal components for testing/inspection."""

    def test_policy_accessible(self, reasoning: ReasoningPillar) -> None:
        assert reasoning.policy is not None

    def test_registry_accessible(self, reasoning: ReasoningPillar) -> None:
        assert reasoning.registry is not None
        assert reasoning.registry.skill_count == 4
