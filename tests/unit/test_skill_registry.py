"""Unit tests for pillars.reasoning.skill_registry — SkillRegistry.

Tests cover:
- Loading all 4 YAML skill definitions
- Lookup by name
- Lookup by query_type
- SkillDefinition fields validation
- Prompt template rendering
- Missing skill handling
- Registry from empty directory
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pillars.reasoning.skill_registry import (
    SkillDefinition,
    SkillParameters,
    SkillRegistry,
)

# Path to the actual skills directory
SKILLS_DIR = Path(__file__).parent.parent.parent / "pillars" / "reasoning" / "skills"


@pytest.fixture
def registry() -> SkillRegistry:
    return SkillRegistry(skills_dir=SKILLS_DIR)


class TestSkillLoading:
    """YAML skill files are discovered and parsed correctly."""

    def test_loads_all_four_skills(self, registry: SkillRegistry) -> None:
        assert registry.skill_count == 4

    def test_skill_names_present(self, registry: SkillRegistry) -> None:
        names = registry.skill_names
        assert "cashflow_forecast" in names
        assert "risk_assessment" in names
        assert "swot_analysis" in names
        assert "customer_analysis" in names

    def test_all_skills_returns_list(self, registry: SkillRegistry) -> None:
        skills = registry.all_skills()
        assert len(skills) == 4
        assert all(isinstance(s, SkillDefinition) for s in skills)


class TestLookupByName:
    """Lookup skills by name."""

    def test_get_cashflow(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None
        assert skill.name == "cashflow_forecast"

    def test_get_risk(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("risk_assessment")
        assert skill is not None
        assert skill.name == "risk_assessment"

    def test_get_swot(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("swot_analysis")
        assert skill is not None

    def test_get_customer(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("customer_analysis")
        assert skill is not None

    def test_get_nonexistent(self, registry: SkillRegistry) -> None:
        assert registry.get_by_name("nonexistent_skill") is None


class TestLookupByQueryType:
    """Lookup skills by query type."""

    def test_get_by_cashflow_type(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_query_type("cashflow_forecast")
        assert skill is not None
        assert skill.name == "cashflow_forecast"

    def test_get_by_risk_type(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_query_type("risk_assessment")
        assert skill is not None

    def test_get_by_swot_type(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_query_type("swot_analysis")
        assert skill is not None

    def test_get_by_customer_type(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_query_type("customer_analysis")
        assert skill is not None

    def test_get_unknown_type(self, registry: SkillRegistry) -> None:
        assert registry.get_by_query_type("unknown") is None


class TestSkillDefinitionFields:
    """Validate fields of loaded skill definitions."""

    def test_cashflow_has_system_prompt(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None
        assert len(skill.system_prompt) > 0

    def test_cashflow_has_prompt_template(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None
        assert "{context}" in skill.prompt_template
        assert "{query}" in skill.prompt_template

    def test_cashflow_has_required_context(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None
        assert len(skill.required_context) > 0
        assert "monthly_summaries" in skill.required_context

    def test_cashflow_has_output_schema(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None
        assert "type" in skill.output_schema
        assert "required" in skill.output_schema

    def test_skill_has_version(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None
        assert skill.version == "1.0"

    def test_all_skills_have_system_prompt(self, registry: SkillRegistry) -> None:
        for skill in registry.all_skills():
            assert len(skill.system_prompt) > 0, f"{skill.name} missing system_prompt"

    def test_all_skills_have_prompt_template(self, registry: SkillRegistry) -> None:
        for skill in registry.all_skills():
            assert "{context}" in skill.prompt_template, f"{skill.name} missing {{context}}"
            assert "{query}" in skill.prompt_template, f"{skill.name} missing {{query}}"


class TestSkillParameters:
    """Skill execution parameters are parsed correctly."""

    def test_cashflow_parameters(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None
        assert skill.parameters.model_tier == "fast"
        assert skill.parameters.temperature == 0.0
        assert skill.parameters.max_retries == 2

    def test_swot_has_nonzero_temperature(self, registry: SkillRegistry) -> None:
        """SWOT has temperature=0.1 for slight creativity."""
        skill = registry.get_by_name("swot_analysis")
        assert skill is not None
        assert skill.parameters.temperature > 0.0

    def test_default_parameters(self) -> None:
        params = SkillParameters()
        assert params.model_tier == "fast"
        assert params.temperature == 0.0
        assert params.max_retries == 2


class TestPromptRendering:
    """Prompt template rendering with context and query."""

    def test_render_replaces_placeholders(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("cashflow_forecast")
        assert skill is not None
        rendered = skill.render_prompt(
            context="Monthly sales: €150,000",
            query="What is the forecast for Q3?",
        )
        assert "Monthly sales: €150,000" in rendered
        assert "What is the forecast for Q3?" in rendered

    def test_render_preserves_structure(self, registry: SkillRegistry) -> None:
        skill = registry.get_by_name("risk_assessment")
        assert skill is not None
        rendered = skill.render_prompt(context="test context", query="test query")
        # Should still contain the JSON template structure
        assert "risk_score" in rendered or "overall_risk_level" in rendered


class TestEmptyRegistry:
    """Registry handles missing/empty skills directory gracefully."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        registry = SkillRegistry(skills_dir=tmp_path)
        assert registry.skill_count == 0
        assert registry.skill_names == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        registry = SkillRegistry(skills_dir=tmp_path / "nonexistent")
        assert registry.skill_count == 0
