"""Unit tests for pillars.simulation.scenario_parser.

Tests cover:
- parse_reasoning_output with valid cashflow result
- parse_reasoning_output with missing/invalid fields
- parse_reasoning_output with no reasoning
- build_multi_scenario generates 3 configs
- build_multi_scenario respects adjustment_factor
- Seed derivation per scenario
- Custom scenario subsets
"""

from __future__ import annotations

import pytest

from pillars.simulation.distributions import CashflowDistributions
from pillars.simulation.monte_carlo import MonteCarloConfig
from pillars.simulation.scenario_parser import (
    DEFAULT_SCENARIOS,
    build_multi_scenario,
    parse_reasoning_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_reasoning_result() -> dict:
    """A realistic Reasoning Pillar output for cashflow_forecast."""
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
                "revenue_trend": "growing",
                "risk_level": "low",
                "adjustment_factor": 1.05,
                "time_horizon_months": 6,
            },
            "timing_ms": 120.5,
            "warnings": [],
        },
    }


def _base_distributions() -> CashflowDistributions:
    return CashflowDistributions(
        revenue_mean=100_000.0,
        revenue_std=20_000.0,
        expense_ratio_mean=0.72,
    )


# ---------------------------------------------------------------------------
# parse_reasoning_output — valid input
# ---------------------------------------------------------------------------

class TestParseReasoningValid:
    """Parse valid reasoning output."""

    def test_extracts_query_type(self) -> None:
        result = parse_reasoning_output(_valid_reasoning_result())
        assert result["query_type"] == "cashflow_forecast"

    def test_extracts_revenue_trend(self) -> None:
        result = parse_reasoning_output(_valid_reasoning_result())
        assert result["revenue_trend"] == "growing"

    def test_extracts_risk_level(self) -> None:
        result = parse_reasoning_output(_valid_reasoning_result())
        assert result["risk_level"] == "low"

    def test_extracts_adjustment_factor(self) -> None:
        result = parse_reasoning_output(_valid_reasoning_result())
        assert result["adjustment_factor"] == 1.05

    def test_extracts_time_horizon(self) -> None:
        result = parse_reasoning_output(_valid_reasoning_result())
        assert result["time_horizon_months"] == 6

    def test_has_reasoning_flag_true(self) -> None:
        result = parse_reasoning_output(_valid_reasoning_result())
        assert result["has_reasoning"] is True

    def test_no_warnings_for_valid(self) -> None:
        result = parse_reasoning_output(_valid_reasoning_result())
        assert result["warnings"] == []


# ---------------------------------------------------------------------------
# parse_reasoning_output — edge cases
# ---------------------------------------------------------------------------

class TestParseReasoningEdgeCases:
    """Parse reasoning output with missing or invalid fields."""

    def test_empty_reasoning(self) -> None:
        result = parse_reasoning_output({})
        assert result["has_reasoning"] is False
        assert result["revenue_trend"] == "stable"
        assert result["risk_level"] == "medium"
        assert result["adjustment_factor"] == 1.0

    def test_failed_skill(self) -> None:
        reasoning = {
            "routing": {"query_type": "cashflow_forecast"},
            "skill_result": {"success": False, "parsed_output": {}},
        }
        result = parse_reasoning_output(reasoning)
        assert result["has_reasoning"] is False
        assert len(result["warnings"]) > 0

    def test_unknown_revenue_trend_defaults(self) -> None:
        reasoning = _valid_reasoning_result()
        reasoning["skill_result"]["parsed_output"]["revenue_trend"] = "volatile"
        result = parse_reasoning_output(reasoning)
        assert result["revenue_trend"] == "stable"
        assert any("Unknown revenue_trend" in w for w in result["warnings"])

    def test_unknown_risk_level_defaults(self) -> None:
        reasoning = _valid_reasoning_result()
        reasoning["skill_result"]["parsed_output"]["risk_level"] = "extreme"
        result = parse_reasoning_output(reasoning)
        assert result["risk_level"] == "medium"
        assert any("Unknown risk_level" in w for w in result["warnings"])

    def test_adjustment_factor_clamped_high(self) -> None:
        reasoning = _valid_reasoning_result()
        reasoning["skill_result"]["parsed_output"]["adjustment_factor"] = 5.0
        result = parse_reasoning_output(reasoning)
        assert result["adjustment_factor"] == 2.0  # Clamped

    def test_adjustment_factor_clamped_low(self) -> None:
        reasoning = _valid_reasoning_result()
        reasoning["skill_result"]["parsed_output"]["adjustment_factor"] = 0.1
        result = parse_reasoning_output(reasoning)
        assert result["adjustment_factor"] == 0.5  # Clamped

    def test_invalid_adjustment_factor_type(self) -> None:
        reasoning = _valid_reasoning_result()
        reasoning["skill_result"]["parsed_output"]["adjustment_factor"] = "invalid"
        result = parse_reasoning_output(reasoning)
        assert result["adjustment_factor"] == 1.0
        assert any("Invalid adjustment_factor" in w for w in result["warnings"])

    def test_time_horizon_clamped(self) -> None:
        reasoning = _valid_reasoning_result()
        reasoning["skill_result"]["parsed_output"]["time_horizon_months"] = 24
        result = parse_reasoning_output(reasoning)
        assert result["time_horizon_months"] == 12  # Clamped

    def test_missing_parsed_output_fields(self) -> None:
        """Missing optional fields should use defaults."""
        reasoning = {
            "routing": {"query_type": "cashflow_forecast"},
            "skill_result": {
                "success": True,
                "parsed_output": {"some_field": "value"},
            },
        }
        result = parse_reasoning_output(reasoning)
        assert result["has_reasoning"] is True
        assert result["revenue_trend"] == "stable"
        assert result["adjustment_factor"] == 1.0


# ---------------------------------------------------------------------------
# build_multi_scenario
# ---------------------------------------------------------------------------

class TestBuildMultiScenario:
    """Test multi-scenario config generation."""

    def test_generates_three_scenarios(self) -> None:
        parsed = parse_reasoning_output(_valid_reasoning_result())
        configs = build_multi_scenario(_base_distributions(), parsed)
        assert len(configs) == 3
        assert set(configs.keys()) == {"base", "optimistic", "stress"}

    def test_each_value_is_config(self) -> None:
        parsed = parse_reasoning_output(_valid_reasoning_result())
        configs = build_multi_scenario(_base_distributions(), parsed)
        for config in configs.values():
            assert isinstance(config, MonteCarloConfig)

    def test_time_horizon_from_reasoning(self) -> None:
        parsed = parse_reasoning_output(_valid_reasoning_result())
        configs = build_multi_scenario(_base_distributions(), parsed)
        for config in configs.values():
            assert config.time_horizon_months == 6  # From reasoning

    def test_adjustment_factor_applied(self) -> None:
        """Adjustment factor should scale revenue_mean."""
        parsed = parse_reasoning_output(_valid_reasoning_result())
        # adjustment_factor = 1.05
        configs = build_multi_scenario(_base_distributions(), parsed)

        base_config = configs["base"]
        # Base scenario: revenue_mean = 100_000 * 1.05 (adjustment applied)
        assert abs(base_config.distributions.revenue_mean - 105_000.0) < 1.0

    def test_different_seeds_per_scenario(self) -> None:
        parsed = parse_reasoning_output(_valid_reasoning_result())
        configs = build_multi_scenario(_base_distributions(), parsed, random_seed=42)

        seeds = [c.random_seed for c in configs.values()]
        assert len(set(seeds)) == 3  # All different

    def test_seed_derivation(self) -> None:
        parsed = parse_reasoning_output(_valid_reasoning_result())
        configs = build_multi_scenario(_base_distributions(), parsed, random_seed=42)

        # Seeds should be 42, 43, 44
        assert configs["base"].random_seed == 42
        assert configs["optimistic"].random_seed == 43
        assert configs["stress"].random_seed == 44

    def test_custom_n_simulations(self) -> None:
        parsed = parse_reasoning_output(_valid_reasoning_result())
        configs = build_multi_scenario(
            _base_distributions(), parsed, n_simulations=5_000,
        )
        for config in configs.values():
            assert config.n_simulations == 5_000

    def test_initial_balance_passed(self) -> None:
        parsed = parse_reasoning_output(_valid_reasoning_result())
        configs = build_multi_scenario(
            _base_distributions(), parsed, initial_balance=50_000.0,
        )
        for config in configs.values():
            assert config.initial_balance == 50_000.0

    def test_start_month_passed(self) -> None:
        parsed = parse_reasoning_output(_valid_reasoning_result())
        configs = build_multi_scenario(
            _base_distributions(), parsed, start_month=7,
        )
        for config in configs.values():
            assert config.start_month == 7

    def test_custom_scenario_subset(self) -> None:
        parsed = parse_reasoning_output(_valid_reasoning_result())
        configs = build_multi_scenario(
            _base_distributions(), parsed, scenarios=("base", "stress"),
        )
        assert len(configs) == 2
        assert "optimistic" not in configs

    def test_no_adjustment_when_1(self) -> None:
        """When adjustment_factor == 1.0, revenue should not change."""
        reasoning = _valid_reasoning_result()
        reasoning["skill_result"]["parsed_output"]["adjustment_factor"] = 1.0
        parsed = parse_reasoning_output(reasoning)
        configs = build_multi_scenario(_base_distributions(), parsed)

        # Base scenario: revenue_mean should be 100_000 (no adjustment)
        assert abs(configs["base"].distributions.revenue_mean - 100_000.0) < 1.0

    def test_default_scenarios_constant(self) -> None:
        assert DEFAULT_SCENARIOS == ("base", "optimistic", "stress")
