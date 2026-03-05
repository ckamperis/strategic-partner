"""Scenario parser — bridges Reasoning Pillar output to Simulation Pillar.

Extracts structured scenario parameters from the Reasoning Pillar's
skill results and constructs Monte Carlo configurations for each
scenario (base, optimistic, stress).

This module has NO LLM dependency — it performs pure data transformation.

References:
    Thesis Section 3.3.3 — Simulation Pillar, Scenario Parsing
    Thesis Section 4.x — Implementation, Reasoning->Simulation Bridge
"""

from __future__ import annotations

from typing import Any

import structlog

from pillars.simulation.distributions import CashflowDistributions, build_scenario
from pillars.simulation.monte_carlo import MonteCarloConfig

logger = structlog.get_logger()

# Scenarios the system always evaluates
DEFAULT_SCENARIOS = ("base", "optimistic", "stress")


def parse_reasoning_output(
    reasoning_result: dict[str, Any],
) -> dict[str, Any]:
    """Extract simulation-relevant parameters from reasoning output.

    The Reasoning Pillar's cashflow_forecast skill produces structured
    JSON with revenue projections, risk flags, and adjustment factors.
    This function extracts what the Simulation Pillar needs.

    Args:
        reasoning_result: The full Reasoning Pillar result dict,
            expected structure::

                {
                    "routing": {"query_type": "cashflow_forecast", ...},
                    "skill_result": {
                        "success": True,
                        "parsed_output": {
                            "revenue_trend": "stable" | "growing" | "declining",
                            "risk_level": "low" | "medium" | "high",
                            "adjustment_factor": float,  # optional
                            "time_horizon_months": int,   # optional
                            ...
                        }
                    }
                }

    Returns:
        Dict of extracted parameters::

            {
                "query_type": str,
                "revenue_trend": str,
                "risk_level": str,
                "adjustment_factor": float,
                "time_horizon_months": int,
                "has_reasoning": bool,
                "warnings": list[str],
            }
    """
    warnings: list[str] = []

    # Determine if we have valid reasoning output
    routing = reasoning_result.get("routing", {})
    query_type = routing.get("query_type", "unknown")

    skill_result = reasoning_result.get("skill_result", {})
    success = skill_result.get("success", False)
    parsed = skill_result.get("parsed_output", {})

    if not success or not parsed:
        warnings.append("Reasoning skill did not produce valid output; using defaults")
        return {
            "query_type": query_type,
            "revenue_trend": "stable",
            "risk_level": "medium",
            "adjustment_factor": 1.0,
            "time_horizon_months": 3,
            "has_reasoning": False,
            "warnings": warnings,
        }

    # Extract parameters with safe defaults
    revenue_trend = parsed.get("revenue_trend", "stable")
    if revenue_trend not in ("stable", "growing", "declining"):
        warnings.append(
            f"Unknown revenue_trend '{revenue_trend}'; defaulting to 'stable'"
        )
        revenue_trend = "stable"

    risk_level = parsed.get("risk_level", "medium")
    if risk_level not in ("low", "medium", "high"):
        warnings.append(
            f"Unknown risk_level '{risk_level}'; defaulting to 'medium'"
        )
        risk_level = "medium"

    adjustment_factor = parsed.get("adjustment_factor", 1.0)
    try:
        adjustment_factor = float(adjustment_factor)
        adjustment_factor = max(0.5, min(adjustment_factor, 2.0))  # Clamp [0.5, 2.0]
    except (TypeError, ValueError):
        warnings.append(
            f"Invalid adjustment_factor '{adjustment_factor}'; defaulting to 1.0"
        )
        adjustment_factor = 1.0

    time_horizon = parsed.get("time_horizon_months", 3)
    try:
        time_horizon = int(time_horizon)
        time_horizon = max(1, min(time_horizon, 12))  # Clamp [1, 12]
    except (TypeError, ValueError):
        time_horizon = 3

    result = {
        "query_type": query_type,
        "revenue_trend": revenue_trend,
        "risk_level": risk_level,
        "adjustment_factor": adjustment_factor,
        "time_horizon_months": time_horizon,
        "has_reasoning": True,
        "warnings": warnings,
    }

    logger.info(
        "scenario_parser.parse_complete",
        query_type=query_type,
        revenue_trend=revenue_trend,
        risk_level=risk_level,
        adjustment_factor=adjustment_factor,
        has_reasoning=True,
    )

    return result


def build_multi_scenario(
    base_distributions: CashflowDistributions,
    parsed_reasoning: dict[str, Any],
    n_simulations: int = 10_000,
    random_seed: int = 42,
    initial_balance: float = 0.0,
    start_month: int = 1,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
) -> dict[str, MonteCarloConfig]:
    """Build Monte Carlo configs for multiple scenarios.

    Creates one MonteCarloConfig per scenario, adjusting the base
    distributions using build_scenario() and optionally applying
    the Reasoning Pillar's adjustment factor.

    Args:
        base_distributions: Fitted distributions from ERP data.
        parsed_reasoning: Output of parse_reasoning_output().
        n_simulations: Number of Monte Carlo paths per scenario.
        random_seed: Base seed (each scenario gets a derived seed).
        initial_balance: Starting cash balance in EUR.
        start_month: Calendar month (1-12) for forecast start.
        scenarios: Tuple of scenario names to generate.

    Returns:
        Dict mapping scenario name -> MonteCarloConfig.

    Note:
        Each scenario uses a different seed (base_seed + scenario_index)
        to avoid correlation between scenario paths while remaining
        individually reproducible.
    """
    adjustment = parsed_reasoning.get("adjustment_factor", 1.0)
    time_horizon = parsed_reasoning.get("time_horizon_months", 3)
    warnings = parsed_reasoning.get("warnings", [])

    configs: dict[str, MonteCarloConfig] = {}

    for i, scenario_name in enumerate(scenarios):
        # Build scenario-adjusted distributions
        scenario_dist = build_scenario(base_distributions, scenario_name)

        # Apply reasoning adjustment factor to revenue mean
        if adjustment != 1.0:
            scenario_dist.revenue_mean *= adjustment

        # Derive unique seed per scenario for independence
        scenario_seed = random_seed + i

        config = MonteCarloConfig(
            n_simulations=n_simulations,
            time_horizon_months=time_horizon,
            random_seed=scenario_seed,
            distributions=scenario_dist,
            initial_balance=initial_balance,
            start_month=start_month,
        )

        configs[scenario_name] = config

    logger.info(
        "scenario_parser.build_multi_scenario",
        scenarios=list(configs.keys()),
        time_horizon=time_horizon,
        adjustment_factor=adjustment,
        n_simulations=n_simulations,
        warning_count=len(warnings),
    )

    return configs
