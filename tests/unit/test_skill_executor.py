"""Unit tests for pillars.reasoning.skill_executor — SkillExecutor.

Uses MockLLMClient for deterministic behaviour.

Tests cover:
- Successful execution with JSON parsing
- JSON extraction from markdown code blocks
- JSON extraction from embedded text
- Retry on parse failure
- Timing recorded
- SkillResult serialisation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pillars.reasoning.skill_executor import SkillExecutor, SkillResult
from pillars.reasoning.skill_registry import SkillDefinition, SkillParameters, SkillRegistry
from utils.llm import MockLLMClient


SKILLS_DIR = Path(__file__).parent.parent.parent / "pillars" / "reasoning" / "skills"


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient(embedding_dim=1536)


@pytest.fixture
def executor(mock_llm: MockLLMClient) -> SkillExecutor:
    return SkillExecutor(mock_llm, model_fast="mock-fast", model_strong="mock-strong")


@pytest.fixture
def registry() -> SkillRegistry:
    return SkillRegistry(skills_dir=SKILLS_DIR)


class TestSkillExecution:
    """End-to-end skill execution with MockLLM."""

    @pytest.mark.asyncio
    async def test_execute_cashflow_skill(
        self, executor: SkillExecutor, registry: SkillRegistry
    ) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None

        result = await executor.execute(
            skill=skill,
            context="Monthly sales: €150,000. Expenses: €95,000.",
            query="cashflow forecast for next quarter",
        )

        assert isinstance(result, SkillResult)
        assert result.skill_name == "cashflow_forecast"
        assert result.success is True
        assert result.attempts >= 1
        assert isinstance(result.parsed_output, dict)

    @pytest.mark.asyncio
    async def test_execute_risk_skill(
        self, executor: SkillExecutor, registry: SkillRegistry
    ) -> None:
        skill = registry.get_by_name("risk_assessment")
        assert skill is not None

        result = await executor.execute(
            skill=skill,
            context="Credit exposure: 8.5%. Overdue invoices: €25,000.",
            query="risk assessment for the portfolio",
        )

        assert result.success is True
        assert result.skill_name == "risk_assessment"

    @pytest.mark.asyncio
    async def test_execute_returns_timing(
        self, executor: SkillExecutor, registry: SkillRegistry
    ) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None

        result = await executor.execute(
            skill=skill,
            context="test context",
            query="forecast query",
        )

        assert result.timing_ms > 0

    @pytest.mark.asyncio
    async def test_execute_to_dict(
        self, executor: SkillExecutor, registry: SkillRegistry
    ) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None

        result = await executor.execute(
            skill=skill,
            context="test",
            query="cashflow forecast",
        )

        d = result.to_dict()
        assert "skill_name" in d
        assert "parsed_output" in d
        assert "success" in d
        assert "attempts" in d
        assert "timing_ms" in d
        assert "warnings" in d


class TestJSONExtraction:
    """JSON extraction from various LLM response formats."""

    def test_extract_pure_json(self) -> None:
        text = '{"key": "value", "number": 42}'
        result = SkillExecutor._extract_json(text)
        assert result == {"key": "value", "number": 42}

    def test_extract_from_markdown_block(self) -> None:
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = SkillExecutor._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_from_plain_code_block(self) -> None:
        text = 'Result:\n```\n{"key": "value"}\n```'
        result = SkillExecutor._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_embedded_json(self) -> None:
        text = 'The analysis shows: {"result": "positive", "score": 0.85} end.'
        result = SkillExecutor._extract_json(text)
        assert result is not None
        assert result["result"] == "positive"

    def test_extract_invalid_returns_none(self) -> None:
        text = "This is not JSON at all."
        result = SkillExecutor._extract_json(text)
        assert result is None

    def test_extract_empty_returns_none(self) -> None:
        result = SkillExecutor._extract_json("")
        assert result is None

    def test_extract_nested_json(self) -> None:
        text = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = SkillExecutor._extract_json(text)
        assert result is not None
        assert result["outer"]["inner"] == [1, 2, 3]


class TestSkillResult:
    """SkillResult dataclass validation."""

    def test_default_result(self) -> None:
        r = SkillResult(skill_name="test")
        assert r.success is False
        assert r.attempts == 0
        assert r.parsed_output == {}
        assert r.warnings == []

    def test_to_dict_keys(self) -> None:
        r = SkillResult(
            skill_name="test",
            success=True,
            parsed_output={"key": "val"},
            attempts=1,
            timing_ms=50.0,
        )
        d = r.to_dict()
        assert d["skill_name"] == "test"
        assert d["success"] is True
        assert d["parsed_output"] == {"key": "val"}
        assert d["attempts"] == 1
        assert d["timing_ms"] == 50.0

    def test_to_dict_excludes_raw_response(self) -> None:
        """raw_response is excluded from serialisation (can be large)."""
        r = SkillResult(skill_name="test", raw_response="very long text...")
        d = r.to_dict()
        assert "raw_response" not in d


class TestModelTierResolution:
    """Model tier mapping."""

    def test_fast_tier(self, executor: SkillExecutor) -> None:
        assert executor._resolve_model("fast") == "mock-fast"

    def test_strong_tier(self, executor: SkillExecutor) -> None:
        assert executor._resolve_model("strong") == "mock-strong"

    def test_unknown_tier_defaults_to_fast(self, executor: SkillExecutor) -> None:
        assert executor._resolve_model("unknown") == "mock-fast"
