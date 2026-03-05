"""Central Orchestrator — wires K -> R -> S -> T pipeline.

The StrategicPartner class coordinates all four pillars through the
PICP lifecycle.  It creates a fresh PICPContext per query, executes
each pillar sequentially, handles graceful degradation on failure,
and returns a PartnerResponse with timing, trust, and explanation.

This is **not** a web framework — no FastAPI / HTTP.  The CLI script
``scripts/run_query.py`` provides the user interface.

Pipeline:
    QUERY_RECEIVED
      -> Knowledge Pillar  (retrieve ERP context via RAG)
      -> Reasoning Pillar  (classify query + execute analytical skill)
      -> Simulation Pillar  (Monte Carlo cashflow forecast — conditional)
      -> Trust Pillar       (evaluate + explain + audit)
    -> RESPONSE_READY

Simulation is skipped for query types that don't need quantitative
forecasting (SWOT_ANALYSIS, CUSTOMER_ANALYSIS, GENERAL).

References:
    Thesis Section 3.4 — PICP Coordination Protocol
    Thesis Section 4.x — Implementation, Orchestration Pipeline
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from pydantic import BaseModel, Field

from picp.bus import PICPBus
from picp.message import PICPContext, PICPEvent
from pillars.knowledge import KnowledgePillar
from pillars.reasoning import ReasoningPillar
from pillars.reasoning.heuristic_policy import QueryType
from pillars.simulation import SimulationPillar
from pillars.simulation.distributions import CashflowDistributions
from pillars.trust import TrustPillar
from utils.llm import LLMClient

logger = structlog.get_logger()

# Query types that should trigger the Simulation Pillar
_SIMULATION_QUERY_TYPES: set[str] = {
    QueryType.CASHFLOW_FORECAST.value,
    QueryType.RISK_ASSESSMENT.value,
}


class PartnerResponse(BaseModel):
    """Complete response from the AI Strategic Partner.

    Attributes:
        query: The original user query.
        query_type: Routing classification from HeuristicPolicy.
        answer: Main user-facing answer (from reasoning or explanation).
        confidence: Trust confidence level ("high"/"medium"/"low").
        trust_score: Overall trust score ∈ [0, 1].
        explanation: Full explanation from Trust Pillar.
        simulation_summary: Key MC metrics (None if simulation skipped).
        factors: SHAP factor contributions.
        caveats: Warnings and limitations.
        pillar_timings: Per-pillar wall-clock times in ms.
        vector_clock: Final PICP vector clock state.
        degradation_flags: Any pillar failures that occurred.
    """

    query: str = ""
    query_type: str = "unknown"
    answer: str = ""
    confidence: str = "low"
    trust_score: float = 0.0
    explanation: dict[str, Any] = Field(default_factory=dict)
    simulation_summary: dict[str, Any] | None = None
    factors: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    pillar_timings: dict[str, float] = Field(default_factory=dict)
    vector_clock: dict[str, int] = Field(default_factory=dict)
    degradation_flags: list[str] = Field(default_factory=list)


class StrategicPartner:
    """Central orchestrator for the AI Strategic Partner PoC.

    Wires all four pillars (Knowledge, Reasoning, Simulation, Trust)
    into a single query pipeline with PICP lifecycle management.

    Args:
        llm_client: LLM client for Knowledge, Reasoning, and Trust.
        bus: The PICP event bus.
        base_distributions: Fitted distributions from ERP data.
        n_simulations: Number of MC paths per scenario.
        random_seed: Base seed for reproducibility.
        audit_dir: Directory for Trust Pillar audit logs.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        bus: PICPBus,
        base_distributions: CashflowDistributions | None = None,
        n_simulations: int = 10_000,
        random_seed: int = 42,
        audit_dir: str = "data/audit/",
    ) -> None:
        self._bus = bus
        self._llm = llm_client

        # Initialise the four pillars
        self._knowledge = KnowledgePillar(bus=bus, llm_client=llm_client)
        self._reasoning = ReasoningPillar(bus=bus, llm_client=llm_client)
        self._simulation = SimulationPillar(
            bus=bus,
            base_distributions=base_distributions,
            n_simulations=n_simulations,
            random_seed=random_seed,
        )
        self._trust = TrustPillar(
            bus=bus,
            llm_client=llm_client,
            audit_dir=audit_dir,
            base_distributions=base_distributions,
        )

    # ── Properties for testing/inspection ────────────────────

    @property
    def knowledge(self) -> KnowledgePillar:
        return self._knowledge

    @property
    def reasoning(self) -> ReasoningPillar:
        return self._reasoning

    @property
    def simulation(self) -> SimulationPillar:
        return self._simulation

    @property
    def trust(self) -> TrustPillar:
        return self._trust

    def set_distributions(self, distributions: CashflowDistributions) -> None:
        """Update distributions on both Simulation and Trust pillars."""
        self._simulation.set_distributions(distributions)
        self._trust.set_distributions(distributions)

    # ── Main query pipeline ──────────────────────────────────

    async def query(self, user_query: str) -> PartnerResponse:
        """Process a user query through the full K -> R -> S -> T pipeline.

        Args:
            user_query: Natural-language business query (Greek or English).

        Returns:
            PartnerResponse with answer, trust score, and metadata.
        """
        total_start = time.perf_counter()
        degradation_flags: list[str] = []

        # Create fresh PICP context
        context = PICPContext.new(query=user_query)

        logger.info(
            "orchestrator.query.start",
            correlation_id=context.correlation_id,
            query=user_query[:100],
        )

        # Publish QUERY_RECEIVED event
        await self._bus.publish(
            PICPEvent.QUERY_RECEIVED, context, source_pillar="orchestrator"
        )

        # ── Step 1: Knowledge Pillar ─────────────────────────
        knowledge_result = await self._run_pillar(
            "knowledge", self._knowledge, context, degradation_flags
        )

        # ── Step 2: Reasoning Pillar ─────────────────────────
        reasoning_result = await self._run_pillar(
            "reasoning", self._reasoning, context, degradation_flags
        )

        # Determine query type from routing
        query_type = self._extract_query_type(reasoning_result)

        # ── Step 3: Simulation Pillar (conditional) ──────────
        simulation_ran = False
        if query_type in _SIMULATION_QUERY_TYPES:
            sim_result = await self._run_pillar(
                "simulation", self._simulation, context, degradation_flags
            )
            simulation_ran = sim_result is not None
        else:
            logger.info(
                "orchestrator.simulation.skipped",
                correlation_id=context.correlation_id,
                query_type=query_type,
                reason="Query type does not require simulation",
            )

        # ── Step 4: Trust Pillar ─────────────────────────────
        trust_result = await self._run_pillar(
            "trust", self._trust, context, degradation_flags
        )

        # ── Build response ───────────────────────────────────
        total_elapsed = (time.perf_counter() - total_start) * 1000

        # Extract timings from context metadata
        timings = context.metadata.get("timings", {})
        timings["total_ms"] = round(total_elapsed, 2)

        # Publish RESPONSE_READY
        await self._bus.publish(
            PICPEvent.RESPONSE_READY, context, source_pillar="orchestrator"
        )

        response = self._build_response(
            query=user_query,
            query_type=query_type,
            context=context,
            trust_result=trust_result,
            simulation_ran=simulation_ran,
            timings=timings,
            degradation_flags=degradation_flags,
        )

        logger.info(
            "orchestrator.query.complete",
            correlation_id=context.correlation_id,
            query_type=query_type,
            trust_score=response.trust_score,
            confidence=response.confidence,
            total_ms=round(total_elapsed, 2),
            degraded=len(degradation_flags) > 0,
        )

        return response

    # ── Internal helpers ─────────────────────────────────────

    async def _run_pillar(
        self,
        name: str,
        pillar: Any,
        context: PICPContext,
        degradation_flags: list[str],
    ) -> dict[str, Any] | None:
        """Run a pillar with graceful degradation on failure.

        Args:
            name: Pillar name for logging.
            pillar: The BasePillar instance.
            context: PICP context.
            degradation_flags: Mutable list to append failure flags.

        Returns:
            Pillar result dict, or None on failure.
        """
        try:
            result = await pillar.process(context)
            return result
        except Exception as e:
            logger.error(
                f"orchestrator.{name}.failed",
                error=str(e),
                correlation_id=context.correlation_id,
            )
            degradation_flags.append(f"{name}_failed")
            return None

    @staticmethod
    def _extract_query_type(reasoning_result: dict[str, Any] | None) -> str:
        """Extract the query type string from reasoning output.

        Args:
            reasoning_result: Reasoning Pillar output dict.

        Returns:
            Query type string (e.g. "cashflow_forecast") or "general".
        """
        if reasoning_result is None:
            return QueryType.GENERAL.value
        routing = reasoning_result.get("routing", {})
        return routing.get("query_type", QueryType.GENERAL.value)

    @staticmethod
    def _build_response(
        query: str,
        query_type: str,
        context: PICPContext,
        trust_result: dict[str, Any] | None,
        simulation_ran: bool,
        timings: dict[str, float],
        degradation_flags: list[str],
    ) -> PartnerResponse:
        """Assemble the final PartnerResponse from pipeline outputs.

        Args:
            query: Original user query.
            query_type: Classified query type.
            context: PICP context with all pillar results.
            trust_result: Trust Pillar output (may be None).
            simulation_ran: Whether simulation was executed.
            timings: Per-pillar timing dict.
            degradation_flags: Any pillar failure flags.

        Returns:
            Fully populated PartnerResponse.
        """
        # Extract answer from reasoning skill result
        reasoning = context.pillar_results.get("reasoning", {})
        skill_result = reasoning.get("skill_result", {})
        parsed_output = skill_result.get("parsed_output", {})

        # Build answer text
        if isinstance(parsed_output, dict):
            answer = parsed_output.get(
                "analysis",
                parsed_output.get(
                    "response",
                    str(parsed_output) if parsed_output else "No analysis available.",
                ),
            )
        else:
            answer = str(parsed_output) if parsed_output else "No analysis available."

        # Extract trust data
        trust_score_val = 0.0
        confidence = "low"
        explanation: dict[str, Any] = {}
        factors: list[dict[str, Any]] = []
        caveats: list[str] = []

        if trust_result is not None:
            ts = trust_result.get("trust_score", {})
            trust_score_val = ts.get("overall", 0.0)
            confidence = ts.get("confidence_level", "low")
            explanation = trust_result.get("explanation", {})
            factors = trust_result.get("shap_factors", [])
            caveats = explanation.get("caveats", [])

        # Simulation summary (compact)
        simulation_summary: dict[str, Any] | None = None
        if simulation_ran:
            sim = context.pillar_results.get("simulation", {})
            scenarios = sim.get("scenarios", {})
            if scenarios:
                simulation_summary = {}
                for name, scenario in scenarios.items():
                    stats = scenario.get("monthly_stats", [])
                    simulation_summary[name] = {
                        "mean_month1": round(stats[0]["mean"], 2) if stats else 0,
                        "probability_negative": round(
                            scenario.get("probability_negative", 0), 4
                        ),
                        "var_5pct": round(scenario.get("var_5pct", 0), 2),
                    }

        # Add degradation caveats
        for flag in degradation_flags:
            caveats.append(f"Pipeline degraded: {flag}")

        return PartnerResponse(
            query=query,
            query_type=query_type,
            answer=answer,
            confidence=confidence,
            trust_score=round(trust_score_val, 4),
            explanation=explanation,
            simulation_summary=simulation_summary,
            factors=factors,
            caveats=caveats,
            pillar_timings=timings,
            vector_clock=dict(context.vector_clock),
            degradation_flags=degradation_flags,
        )
