"""Simulation Pillar — Monte Carlo cashflow forecasting.

Orchestrates:
- CashflowDistributions: statistical distributions fitted from ERP data
- MonteCarloEngine: vectorized NumPy simulation engine
- ScenarioParser: bridges Reasoning -> Simulation

The Simulation Pillar receives reasoning output from the Reasoning Pillar
and runs Monte Carlo simulations across multiple scenarios (base, optimistic,
stress). It produces probability distributions of future cashflows.

Pipeline:
    reasoning_result + erp_distributions
      -> parse_reasoning_output() -> extracted parameters
      -> build_multi_scenario() -> {scenario: MonteCarloConfig}
      -> MonteCarloEngine.run(config) per scenario -> MonteCarloResult
      -> aggregate into SimulationResult

**No LLM calls** — this pillar is pure NumPy computation.

References:
    Thesis Section 3.3.3 — Simulation Pillar
    Thesis Section 4.x — Implementation, Monte Carlo Architecture
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from picp.bus import PICPBus
from picp.message import PICPContext, PICPEvent
from pillars.base import BasePillar
from pillars.simulation.distributions import CashflowDistributions
from pillars.simulation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResult,
)
from pillars.simulation.scenario_parser import (
    build_multi_scenario,
    parse_reasoning_output,
)

logger = structlog.get_logger()


@dataclass
class SimulationResult:
    """Aggregated output from multi-scenario Monte Carlo simulation.

    Attributes:
        scenarios: Dict mapping scenario name -> MonteCarloResult.
        parsed_reasoning: Extracted parameters from Reasoning Pillar.
        total_elapsed_ms: Total wall-clock time for all scenarios.
        warnings: Accumulated warnings from parsing and simulation.
    """

    scenarios: dict[str, MonteCarloResult] = field(default_factory=dict)
    parsed_reasoning: dict[str, Any] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for PICP context and JSON output."""
        return {
            "scenarios": {
                name: result.to_dict()
                for name, result in self.scenarios.items()
            },
            "parsed_reasoning": self.parsed_reasoning,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "warnings": self.warnings,
        }


class SimulationPillar(BasePillar):
    """Simulation Pillar — runs Monte Carlo cashflow simulations.

    Receives reasoning output and ERP-fitted distributions, then
    runs multi-scenario simulations. No LLM calls — pure computation.

    Args:
        bus: The PICP event bus.
        base_distributions: Fitted distributions from ERP data.
        n_simulations: Number of Monte Carlo paths per scenario.
        random_seed: Base seed for reproducibility.
        initial_balance: Starting cash balance in EUR.
        start_month: Calendar month (1-12) for forecast start.
    """

    def __init__(
        self,
        bus: PICPBus,
        base_distributions: CashflowDistributions | None = None,
        n_simulations: int = 10_000,
        random_seed: int = 42,
        initial_balance: float = 0.0,
        start_month: int = 1,
    ) -> None:
        super().__init__(
            name="simulation",
            bus=bus,
            start_event=PICPEvent.SIMULATION_STARTED,
            complete_event=PICPEvent.SIMULATION_READY,
        )
        self._engine = MonteCarloEngine()
        self._base_distributions = base_distributions or CashflowDistributions()
        self._n_simulations = n_simulations
        self._random_seed = random_seed
        self._initial_balance = initial_balance
        self._start_month = start_month

    @property
    def engine(self) -> MonteCarloEngine:
        """Access the Monte Carlo engine (for testing/inspection)."""
        return self._engine

    @property
    def base_distributions(self) -> CashflowDistributions:
        """Access the base distributions (for testing/inspection)."""
        return self._base_distributions

    def set_distributions(self, distributions: CashflowDistributions) -> None:
        """Update the base distributions (e.g. after ERP data refresh).

        Args:
            distributions: New fitted distributions.
        """
        self._base_distributions = distributions
        logger.info(
            "simulation.distributions_updated",
            revenue_mean=distributions.revenue_mean,
            revenue_std=distributions.revenue_std,
        )

    async def _execute(
        self, context: PICPContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute multi-scenario Monte Carlo simulation.

        Steps:
        1. Extract reasoning output from PICP context.
        2. Parse reasoning into simulation parameters.
        3. Build per-scenario MonteCarloConfigs.
        4. Run Monte Carlo engine for each scenario.
        5. Aggregate and return results.

        Args:
            context: The PICP context with reasoning results.

        Returns:
            Dict with simulation results for all scenarios.
        """
        total_start = time.perf_counter()
        warnings: list[str] = []

        # Step 1: Extract reasoning output
        reasoning_result = context.pillar_results.get("reasoning", {})
        if not reasoning_result:
            warnings.append(
                "No reasoning output found; simulation uses default parameters"
            )
            reasoning_result = {
                "routing": {"query_type": "unknown"},
                "skill_result": {"success": False, "parsed_output": {}},
            }

        # Step 2: Parse reasoning output
        parsed = parse_reasoning_output(reasoning_result)
        warnings.extend(parsed.get("warnings", []))

        # Step 3: Build multi-scenario configs
        configs = build_multi_scenario(
            base_distributions=self._base_distributions,
            parsed_reasoning=parsed,
            n_simulations=self._n_simulations,
            random_seed=self._random_seed,
            initial_balance=self._initial_balance,
            start_month=self._start_month,
        )

        # Step 4: Run simulations for each scenario
        scenario_results: dict[str, MonteCarloResult] = {}
        for scenario_name, config in configs.items():
            mc_result = self._engine.run(config, scenario_name=scenario_name)
            scenario_results[scenario_name] = mc_result

            logger.info(
                "simulation.scenario_complete",
                scenario=scenario_name,
                correlation_id=context.correlation_id,
                probability_negative=round(mc_result.probability_negative, 4),
                var_5pct=round(mc_result.var_5pct, 2),
                elapsed_ms=round(mc_result.elapsed_ms, 2),
            )

        total_elapsed = (time.perf_counter() - total_start) * 1000

        # Step 5: Aggregate
        sim_result = SimulationResult(
            scenarios=scenario_results,
            parsed_reasoning=parsed,
            total_elapsed_ms=total_elapsed,
            warnings=warnings,
        )

        logger.info(
            "simulation.all_scenarios_complete",
            correlation_id=context.correlation_id,
            scenario_count=len(scenario_results),
            total_elapsed_ms=round(total_elapsed, 2),
        )

        return sim_result.to_dict()
