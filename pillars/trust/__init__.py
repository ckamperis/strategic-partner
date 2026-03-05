"""Trust Pillar — transparency, explainability, and audit logging.

Orchestrates:
- TrustEvaluator: composite trust score (Eq. 3.28)
- SimulatedSHAP: factor contribution analysis via counterfactual MC
- ExplanationGenerator: human-readable explanations (template-based)
- AuditLogger: append-only JSON Lines audit trail

The Trust Pillar is the final pillar in the pipeline. It evaluates
all preceding pillar outputs for trustworthiness, generates explanations,
and logs the complete decision trace.

Pipeline:
    knowledge_result + reasoning_result + simulation_result
      -> TrustEvaluator.evaluate() -> TrustScore
      -> SimulatedSHAP.explain_forecast() -> [FactorContribution]
      -> ExplanationGenerator.generate() -> Explanation
      -> AuditLogger.log() -> audit_id
      -> TrustResult

References:
    Thesis Section 3.3.4 — Trust Pillar
    Thesis Section 4.x — Implementation, Trust Architecture
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from data.pipeline.models import BusinessMetrics
from picp.bus import PICPBus
from picp.message import PICPContext, PICPEvent
from pillars.base import BasePillar
from pillars.simulation.distributions import CashflowDistributions
from pillars.trust.audit import AuditEntry, AuditLogger
from pillars.trust.evaluator import TrustEvaluator, TrustScore
from pillars.trust.explainer import Explanation, ExplanationGenerator
from pillars.trust.shap_explainer import FactorContribution, SimulatedSHAP
from utils.llm import LLMClient

logger = structlog.get_logger()


@dataclass
class TrustResult:
    """Complete output of the Trust Pillar.

    Attributes:
        trust_score: Weighted composite trust score with sub-scores.
        explanation: Human-readable explanation of the result.
        shap_factors: Factor contributions from SimulatedSHAP.
        audit_id: Unique identifier for the audit log entry.
        execution_time_ms: Wall-clock time for trust evaluation.
    """

    trust_score: TrustScore = field(default_factory=TrustScore)
    explanation: Explanation = field(default_factory=Explanation)
    shap_factors: list[FactorContribution] = field(default_factory=list)
    audit_id: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for PICP context and JSON output."""
        return {
            "trust_score": self.trust_score.to_dict(),
            "explanation": self.explanation.to_dict(),
            "shap_factors": [f.to_dict() for f in self.shap_factors],
            "audit_id": self.audit_id,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class TrustPillar(BasePillar):
    """Trust Pillar — evaluates, explains, and audits system outputs.

    The last pillar in the K -> R -> S -> T pipeline. It examines all
    preceding results to produce a trust score, SHAP-like explanations,
    and a full audit trail.

    Args:
        bus: The PICP event bus.
        llm_client: Optional LLM client for enhanced explanations.
        audit_dir: Directory for audit log files.
        base_distributions: Fitted distributions (for SHAP counterfactuals).
        metrics: Business metrics (for customer concentration analysis).
        trust_weights: Tuple of (w_e, w_c, w_a) for Eq. 3.28.
    """

    def __init__(
        self,
        bus: PICPBus,
        llm_client: LLMClient | None = None,
        audit_dir: str = "data/audit/",
        base_distributions: CashflowDistributions | None = None,
        metrics: BusinessMetrics | None = None,
        trust_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    ) -> None:
        super().__init__(
            name="trust",
            bus=bus,
            start_event=PICPEvent.TRUST_STARTED,
            complete_event=PICPEvent.TRUST_VALIDATED,
        )
        self._evaluator = TrustEvaluator(weights=trust_weights)
        self._shap = SimulatedSHAP()
        self._explainer = ExplanationGenerator(llm_client=llm_client)
        self._auditor = AuditLogger(log_dir=audit_dir)
        self._distributions = base_distributions or CashflowDistributions()
        self._metrics = metrics or BusinessMetrics()

    @property
    def evaluator(self) -> TrustEvaluator:
        """Access the trust evaluator (for testing)."""
        return self._evaluator

    @property
    def auditor(self) -> AuditLogger:
        """Access the audit logger (for testing)."""
        return self._auditor

    def set_distributions(
        self,
        distributions: CashflowDistributions,
        metrics: BusinessMetrics | None = None,
    ) -> None:
        """Update distributions and metrics for SHAP analysis.

        Args:
            distributions: New fitted distributions.
            metrics: Optional updated business metrics.
        """
        self._distributions = distributions
        if metrics is not None:
            self._metrics = metrics

    async def _execute(
        self, context: PICPContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute the Trust Pillar pipeline.

        Steps:
        1. Extract all pillar results from PICP context.
        2. Evaluate composite trust score.
        3. Generate SHAP factor contributions.
        4. Generate human-readable explanation.
        5. Log audit entry.
        6. Return TrustResult.

        Args:
            context: The PICP context with all preceding pillar results.

        Returns:
            Dict with trust score, explanation, SHAP factors, and audit_id.
        """
        start = time.perf_counter()

        # Step 1: Extract pillar results
        knowledge_result = context.pillar_results.get("knowledge", {})
        reasoning_result = context.pillar_results.get("reasoning", {})
        simulation_result = context.pillar_results.get("simulation", {})

        # Step 2: Evaluate trust score (Eq. 3.28)
        trust_score = self._evaluator.evaluate(
            knowledge_result=knowledge_result,
            reasoning_result=reasoning_result,
            simulation_result=simulation_result,
        )

        # Step 3: Generate SHAP factor contributions
        shap_factors: list[FactorContribution] = []
        if simulation_result.get("scenarios"):
            try:
                shap_factors = self._shap.explain_forecast(
                    simulation_result=simulation_result,
                    distributions=self._distributions,
                    metrics=self._metrics,
                )
            except Exception as e:
                logger.warning(
                    "trust.shap.failed",
                    error=str(e),
                    correlation_id=context.correlation_id,
                )
                trust_score.flags.append("shap_analysis_failed")

        # Step 4: Generate explanation
        explanation = self._explainer.generate(
            trust_score=trust_score,
            reasoning_result=reasoning_result,
            simulation_result=simulation_result,
            shap_factors=shap_factors,
        )

        # Step 5: Log audit entry
        audit_entry = self._build_audit_entry(
            context=context,
            knowledge=knowledge_result,
            reasoning=reasoning_result,
            simulation=simulation_result,
            trust_score=trust_score,
            explanation=explanation,
        )
        audit_id = self._auditor.log(audit_entry)

        elapsed = (time.perf_counter() - start) * 1000

        # Step 6: Build result
        trust_result = TrustResult(
            trust_score=trust_score,
            explanation=explanation,
            shap_factors=shap_factors,
            audit_id=audit_id,
            execution_time_ms=elapsed,
        )

        logger.info(
            "trust.pipeline.complete",
            correlation_id=context.correlation_id,
            overall_score=round(trust_score.overall, 4),
            confidence=trust_score.confidence_level,
            audit_id=audit_id,
            elapsed_ms=round(elapsed, 2),
        )

        return trust_result.to_dict()

    @staticmethod
    def _build_audit_entry(
        context: PICPContext,
        knowledge: dict[str, Any],
        reasoning: dict[str, Any],
        simulation: dict[str, Any],
        trust_score: TrustScore,
        explanation: Explanation,
    ) -> AuditEntry:
        """Build an audit entry from pipeline state.

        Args:
            context: PICP context.
            knowledge: Knowledge Pillar output.
            reasoning: Reasoning Pillar output.
            simulation: Simulation Pillar output.
            trust_score: Evaluated trust score.
            explanation: Generated explanation.

        Returns:
            AuditEntry ready for logging.
        """
        routing = reasoning.get("routing", {})

        # Simulation summary (compact)
        sim_summary: dict[str, Any] = {}
        scenarios = simulation.get("scenarios", {})
        for name, scenario in scenarios.items():
            stats = scenario.get("monthly_stats", [])
            sim_summary[name] = {
                "mean_month1": stats[0].get("mean", 0) if stats else 0,
                "prob_neg": scenario.get("probability_negative", 0),
                "var_5pct": scenario.get("var_5pct", 0),
            }

        return AuditEntry(
            query=context.query,
            query_type=routing.get("query_type", "unknown"),
            knowledge_summary={
                "chunk_count": len(knowledge.get("chunks", [])),
                "relevance_score": knowledge.get("final_score", knowledge.get("relevance_score", 0)),
            },
            reasoning_summary={
                "skill_name": routing.get("skill_name", ""),
                "confidence": routing.get("confidence", 0),
                "query_type": routing.get("query_type", ""),
            },
            simulation_summary=sim_summary,
            trust_score=trust_score.to_dict(),
            explanation_summary=explanation.summary[:200],
            vector_clock=dict(context.vector_clock),
            execution_time_ms=0.0,
        )
