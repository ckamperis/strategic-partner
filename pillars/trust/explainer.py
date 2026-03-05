"""Explanation Generator — human-readable explanations of system output.

Generates structured explanations in Greek and English, either:
1. Template mode (default, deterministic) — no LLM needed
2. LLM-enhanced mode (optional) — polished natural language

For unit tests and reproducibility, template mode is always used.
LLM mode is reserved for integration/experiment runs.

References:
    Thesis Section 3.3.4 — Trust Pillar, Explainability
    Thesis Section 4.x — Implementation, Explanation Generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from pillars.trust.evaluator import TrustScore
from pillars.trust.shap_explainer import FactorContribution
from utils.llm import LLMClient

logger = structlog.get_logger()


@dataclass
class Explanation:
    """Complete human-readable explanation of a system output.

    Attributes:
        summary: One-sentence answer (Greek).
        reasoning_trace: Step-by-step how the system reached its conclusion.
        key_factors: Top contributing factors from SimulatedSHAP.
        data_sources: Which data sources were used.
        confidence: "high" / "medium" / "low".
        caveats: Limitations and estimated parameters.
        methodology: Brief description of analysis method.
    """

    summary: str = ""
    reasoning_trace: str = ""
    key_factors: list[FactorContribution] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    confidence: str = "low"
    caveats: list[str] = field(default_factory=list)
    methodology: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "reasoning_trace": self.reasoning_trace,
            "key_factors": [f.to_dict() for f in self.key_factors],
            "data_sources": self.data_sources,
            "confidence": self.confidence,
            "caveats": self.caveats,
            "methodology": self.methodology,
        }


class ExplanationGenerator:
    """Generates human-readable explanations from trust evaluation results.

    Template mode (no LLM) is the default and is fully deterministic.
    LLM mode can optionally polish the output for production quality.

    Args:
        llm_client: Optional LLM client for enhanced mode.
            If None, uses template-based explanations only.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client

    def generate(
        self,
        trust_score: TrustScore,
        reasoning_result: dict[str, Any] | None = None,
        simulation_result: dict[str, Any] | None = None,
        shap_factors: list[FactorContribution] | None = None,
    ) -> Explanation:
        """Generate explanation from trust evaluation results.

        Always uses template mode. LLM enhancement is not called
        in the current implementation (reserved for future integration).

        Args:
            trust_score: The evaluated trust score.
            reasoning_result: Reasoning Pillar output.
            simulation_result: Simulation Pillar output.
            shap_factors: Factor contributions from SimulatedSHAP.

        Returns:
            Explanation with all fields populated.
        """
        reasoning = reasoning_result or {}
        simulation = simulation_result or {}
        factors = shap_factors or []

        summary = self._build_summary(simulation, trust_score)
        reasoning_trace = self._build_reasoning_trace(
            reasoning, simulation, trust_score
        )
        data_sources = self._extract_data_sources(reasoning, simulation)
        caveats = self._build_caveats(trust_score)
        methodology = self._build_methodology(simulation)

        explanation = Explanation(
            summary=summary,
            reasoning_trace=reasoning_trace,
            key_factors=factors,
            data_sources=data_sources,
            confidence=trust_score.confidence_level,
            caveats=caveats,
            methodology=methodology,
        )

        logger.info(
            "explainer.generate.complete",
            confidence=trust_score.confidence_level,
            caveat_count=len(caveats),
            factor_count=len(factors),
        )

        return explanation

    # ------------------------------------------------------------------
    # Template builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        simulation: dict[str, Any],
        trust_score: TrustScore,
    ) -> str:
        """Build one-sentence summary in Greek."""
        scenarios = simulation.get("scenarios", {})
        base = scenarios.get("base", {})
        monthly_stats = base.get("monthly_stats", [])
        cumulative_stats = base.get("cumulative_stats", [])
        prob_neg = base.get("probability_negative", 0)

        if not monthly_stats:
            return (
                "Δεν ήταν δυνατή η παραγωγή πρόβλεψης ταμειακών ροών "
                "λόγω ελλιπών δεδομένων προσομοίωσης."
            )

        n_months = len(monthly_stats)
        mean_monthly = monthly_stats[0].get("mean", 0)
        p5 = monthly_stats[0].get("p5", 0)
        p95 = monthly_stats[0].get("p95", 0)

        return (
            f"Η πρόβλεψη ταμειακών ροών {n_months} μηνών δείχνει "
            f"μέσο μηνιαίο cashflow €{mean_monthly / 1000:,.1f}K "
            f"(90% CI: €{p5 / 1000:,.1f}K — €{p95 / 1000:,.1f}K). "
            f"Πιθανότητα αρνητικού υπολοίπου: {prob_neg * 100:.1f}%."
        )

    @staticmethod
    def _build_reasoning_trace(
        reasoning: dict[str, Any],
        simulation: dict[str, Any],
        trust_score: TrustScore,
    ) -> str:
        """Build step-by-step reasoning trace."""
        # Extract knowledge metadata
        routing = reasoning.get("routing", {})
        skill_result = reasoning.get("skill_result", {})
        skill_name = skill_result.get("skill_name", "unknown")
        confidence = routing.get("confidence", 0)

        # Simulation metadata
        scenarios = simulation.get("scenarios", {})
        base = scenarios.get("base", {})
        n_sims = base.get("n_simulations", 0)
        n_scenarios = len(scenarios)

        lines = [
            f"1. Δρομολόγηση: skill '{skill_name}', confidence {confidence:.2f}",
            f"2. Προσομοίωση: {n_sims:,} Monte Carlo paths, {n_scenarios} σενάρια",
            f"3. Αξιολόγηση: Trust score {trust_score.overall:.2f} "
            f"({trust_score.confidence_level})",
        ]

        return "\n".join(lines)

    @staticmethod
    def _extract_data_sources(
        reasoning: dict[str, Any],
        simulation: dict[str, Any],
    ) -> list[str]:
        """Identify data sources used in the analysis."""
        sources: list[str] = []

        routing = reasoning.get("routing", {})
        query_type = routing.get("query_type", "")
        if query_type:
            sources.append(f"Reasoning skill: {query_type}")

        scenarios = simulation.get("scenarios", {})
        if scenarios:
            sources.append("Monte Carlo simulation (base/optimistic/stress)")

        base = scenarios.get("base", {})
        config = base.get("config_snapshot", {})
        if config.get("distributions"):
            sources.append("ERP-fitted distributions")

        return sources if sources else ["No data sources identified"]

    @staticmethod
    def _build_caveats(trust_score: TrustScore) -> list[str]:
        """Convert trust flags into human-readable caveats (Greek)."""
        flag_map = {
            "estimated_expense_ratio": (
                "Ο δείκτης εξόδων (0.72) είναι εκτιμώμενος — "
                "δεν υπάρχουν τιμολόγια προμηθευτών στα δεδομένα."
            ),
            "estimated_collection_delay": (
                "Η καθυστέρηση είσπραξης (52 ημέρες) είναι εκτιμώμενη — "
                "DUEDATE = TRNDATE στο dataset."
            ),
            "estimated_customer_loss_rate": (
                "Ο ρυθμός απώλειας πελατών (2%/τρίμηνο) είναι εκτιμώμενος — "
                "δεν υπάρχει διαχρονική παρακολούθηση πελατών."
            ),
            "low_rag_relevance": (
                "Χαμηλή σχετικότητα ανάκτησης δεδομένων — "
                "τα αποτελέσματα μπορεί να μην αντικατοπτρίζουν πλήρως "
                "τα διαθέσιμα δεδομένα."
            ),
            "general_routing": (
                "Το ερώτημα δεν αντιστοιχήθηκε σε εξειδικευμένη δεξιότητα — "
                "χρησιμοποιήθηκε γενική ανάλυση."
            ),
            "low_simulation_count": (
                "Χαμηλός αριθμός προσομοιώσεων — "
                "τα στατιστικά αποτελέσματα ενδέχεται να μην έχουν συγκλίνει."
            ),
            "risk_simulation_mismatch": (
                "Ασυμφωνία μεταξύ εκτίμησης κινδύνου και αποτελεσμάτων "
                "προσομοίωσης."
            ),
            "insufficient_data_for_consistency": (
                "Ανεπαρκή δεδομένα για πλήρη έλεγχο συνέπειας."
            ),
            "limited_data_coverage": (
                "Περιορισμένη κάλυψη δεδομένων — "
                "λιγότερα από 12 μήνες δεδομένων."
            ),
            "anomalous_credit_note_ratio": (
                "Ασυνήθιστα υψηλός δείκτης πιστωτικών σημειωμάτων (>20%)."
            ),
        }

        caveats: list[str] = []
        for flag in trust_score.flags:
            if flag in flag_map:
                caveats.append(flag_map[flag])
            else:
                caveats.append(f"Σημαία: {flag}")

        return caveats

    @staticmethod
    def _build_methodology(simulation: dict[str, Any]) -> str:
        """Build brief methodology description."""
        scenarios = simulation.get("scenarios", {})
        base = scenarios.get("base", {})
        n_sims = base.get("n_simulations", 0)
        n_months = len(base.get("monthly_stats", []))

        if not scenarios:
            return "Δεν εκτελέστηκε προσομοίωση."

        return (
            f"Monte Carlo προσομοίωση με {n_sims:,} διαδρομές "
            f"σε ορίζοντα {n_months} μηνών, "
            f"{len(scenarios)} σενάρια (βασικό/αισιόδοξο/πιεστικό). "
            f"Κατανομές εξαγμένες από ιστορικά δεδομένα ERP."
        )
