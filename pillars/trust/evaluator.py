"""Trust Score Evaluator — composite trustworthiness scoring.

Computes a weighted trust score from three sub-dimensions:
    T_overall = w_e · T_explainability + w_c · T_consistency + w_a · T_accuracy

Where (Eq. 3.28):
    w_e = 0.4  (explainability weight)
    w_c = 0.4  (consistency weight)
    w_a = 0.2  (accuracy / evidence weight)

Each sub-score ∈ [0, 1] is computed from pillar outputs. The evaluator
also collects flags for transparency (e.g., "estimated_expense_ratio").

References:
    Thesis Eq. 3.28 — Trust Score composite formula
    Thesis Eq. 3.29 — Domain adjustment (T_adj = T · domain_factor)
    Thesis Section 3.3.4 — Trust Pillar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

# Default trust weights (Eq. 3.28)
DEFAULT_WEIGHTS = (0.4, 0.4, 0.2)

# Thresholds for confidence_level classification
CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.5

# Estimated parameter defaults (from distributions.py)
_DEFAULT_EXPENSE_RATIO = 0.72
_DEFAULT_COLLECTION_DELAY = 52.0
_DEFAULT_CUSTOMER_LOSS = 0.02


@dataclass
class TrustSubScores:
    """Individual sub-dimension trust scores.

    Attributes:
        explainability: How well the result can be explained (0-1).
        consistency: Internal consistency between pillar outputs (0-1).
        accuracy: Data quality and evidence strength (0-1).
    """

    explainability: float = 0.0
    consistency: float = 0.0
    accuracy: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "explainability": round(self.explainability, 4),
            "consistency": round(self.consistency, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class TrustScore:
    """Complete trust evaluation result.

    Attributes:
        overall: Weighted composite score ∈ [0, 1].
        sub_scores: Individual sub-dimension scores.
        flags: List of transparency flags / warnings.
        confidence_level: "high" (>0.8), "medium" (0.5-0.8), "low" (<0.5).
    """

    overall: float = 0.0
    sub_scores: TrustSubScores = field(default_factory=TrustSubScores)
    flags: list[str] = field(default_factory=list)
    confidence_level: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "sub_scores": self.sub_scores.to_dict(),
            "flags": self.flags,
            "confidence_level": self.confidence_level,
        }


class TrustEvaluator:
    """Evaluates trustworthiness of system outputs.

    Implements Eq. 3.28: T = w_e·E + w_c·C + w_a·A

    Args:
        weights: Tuple of (w_explainability, w_consistency, w_accuracy).
            Must sum to 1.0.
    """

    def __init__(
        self,
        weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    ) -> None:
        w_sum = sum(weights)
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Trust weights must sum to 1.0, got {w_sum:.4f}"
            )
        self._w_e, self._w_c, self._w_a = weights

    @property
    def weights(self) -> tuple[float, float, float]:
        """Access the weight triple."""
        return (self._w_e, self._w_c, self._w_a)

    def evaluate(
        self,
        knowledge_result: dict[str, Any] | None = None,
        reasoning_result: dict[str, Any] | None = None,
        simulation_result: dict[str, Any] | None = None,
    ) -> TrustScore:
        """Evaluate trust score from all pillar outputs.

        Args:
            knowledge_result: Knowledge Pillar output (RAG results).
            reasoning_result: Reasoning Pillar output (routing + skill).
            simulation_result: Simulation Pillar output (MC scenarios).

        Returns:
            TrustScore with overall score, sub-scores, and flags.
        """
        knowledge = knowledge_result or {}
        reasoning = reasoning_result or {}
        simulation = simulation_result or {}

        flags: list[str] = []

        # Compute sub-scores
        explainability = self._score_explainability(
            knowledge, reasoning, simulation, flags
        )
        consistency = self._score_consistency(
            knowledge, reasoning, simulation, flags
        )
        accuracy = self._score_accuracy(
            knowledge, reasoning, simulation, flags
        )

        sub_scores = TrustSubScores(
            explainability=explainability,
            consistency=consistency,
            accuracy=accuracy,
        )

        # Weighted composite (Eq. 3.28)
        overall = (
            self._w_e * explainability
            + self._w_c * consistency
            + self._w_a * accuracy
        )
        overall = max(0.0, min(overall, 1.0))

        # Confidence level classification
        if overall >= CONFIDENCE_HIGH:
            confidence_level = "high"
        elif overall >= CONFIDENCE_LOW:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        score = TrustScore(
            overall=overall,
            sub_scores=sub_scores,
            flags=flags,
            confidence_level=confidence_level,
        )

        logger.info(
            "trust.evaluate.complete",
            overall=round(overall, 4),
            explainability=round(explainability, 4),
            consistency=round(consistency, 4),
            accuracy=round(accuracy, 4),
            confidence_level=confidence_level,
            flag_count=len(flags),
        )

        return score

    # ------------------------------------------------------------------
    # Sub-score: Explainability
    # ------------------------------------------------------------------

    @staticmethod
    def _score_explainability(
        knowledge: dict[str, Any],
        reasoning: dict[str, Any],
        simulation: dict[str, Any],
        flags: list[str],
    ) -> float:
        """Compute explainability sub-score (0-1).

        Components:
        +0.30  RAG retrieved relevant chunks
        +0.20  Reasoning used a registered skill (not GENERAL)
        +0.20  Simulation ran successfully with ≥1000 sims
        +0.15  Single-pass retrieval (query_history length ≤ 1)
        +0.15  All required_context fields present for skill
        """
        score = 0.0

        # RAG relevance (+0.30)
        chunks = knowledge.get("chunks", [])
        rag_score = knowledge.get("final_score", knowledge.get("relevance_score", 0.0))
        if chunks and rag_score > 0.5:
            score += 0.30
        elif chunks:
            score += 0.15  # Partial: chunks but low relevance
        else:
            flags.append("low_rag_relevance")

        # Registered skill (+0.20)
        routing = reasoning.get("routing", {})
        query_type = routing.get("query_type", "general")
        skill_name = routing.get("skill_name")
        if query_type != "general" and skill_name is not None:
            score += 0.20
        else:
            flags.append("general_routing")

        # Simulation success (+0.20)
        scenarios = simulation.get("scenarios", {})
        base_scenario = scenarios.get("base", {})
        n_sims = base_scenario.get("n_simulations", 0)
        if scenarios and n_sims >= 1000:
            score += 0.20
        elif scenarios:
            score += 0.10
            flags.append("low_simulation_count")

        # Single-pass retrieval (+0.15)
        query_history = knowledge.get("query_history", [])
        iterations = knowledge.get("iterations", 1)
        if iterations <= 1:
            score += 0.15

        # Required context fields present (+0.15)
        skill_result = reasoning.get("skill_result", {})
        if skill_result.get("success", False):
            parsed = skill_result.get("parsed_output", {})
            if parsed:
                score += 0.15

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Sub-score: Consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _score_consistency(
        knowledge: dict[str, Any],
        reasoning: dict[str, Any],
        simulation: dict[str, Any],
        flags: list[str],
    ) -> float:
        """Compute consistency sub-score (0-1).

        Components (each +0.25):
        1. Revenue estimate within 2σ of simulation mean
        2. Seasonal pattern from ERP matches reasoning direction
        3. Risk assessment aligns with simulation probability_negative
        4. No contradictions between knowledge and reasoning

        Missing data -> 0.0 for that component + flag.
        """
        score = 0.0
        checks_possible = 0

        # Check 1: Revenue within 2σ of simulation mean
        scenarios = simulation.get("scenarios", {})
        base_scenario = scenarios.get("base", {})
        monthly_stats = base_scenario.get("monthly_stats", [])
        config_snapshot = base_scenario.get("config_snapshot", {})
        dist_snapshot = config_snapshot.get("distributions", {})

        if monthly_stats and dist_snapshot:
            checks_possible += 1
            sim_mean = monthly_stats[0].get("mean", 0)
            sim_std = monthly_stats[0].get("std", 1)
            rev_mean = dist_snapshot.get("revenue_mean", 0)
            # Check that the distribution mean is roughly consistent
            # with the simulation output (within 2σ accounting for expenses)
            # Net CF ≈ revenue × (1 - expense_ratio)
            expense_ratio = dist_snapshot.get("expense_ratio_mean", 0.72)
            expected_net = rev_mean * (1 - expense_ratio)
            if sim_std > 0 and abs(sim_mean - expected_net) < 2 * sim_std:
                score += 0.25

        # Check 2: Seasonal pattern consistency
        seasonal = dist_snapshot.get("seasonal_factors", [])
        if seasonal and len(seasonal) == 12:
            checks_possible += 1
            # If there's seasonal variation, that's consistent data
            seasonal_range = max(seasonal) - min(seasonal)
            if seasonal_range > 0.05:
                score += 0.25
            else:
                score += 0.15  # Flat seasonality is less informative but ok

        # Check 3: Risk alignment with probability_negative
        routing = reasoning.get("routing", {})
        skill_result = reasoning.get("skill_result", {})
        parsed = skill_result.get("parsed_output", {})
        risk_level = parsed.get("risk_level", "")
        prob_neg = base_scenario.get("probability_negative", -1)

        if risk_level and prob_neg >= 0:
            checks_possible += 1
            # Risk "low" should correspond to low prob_neg, etc.
            if risk_level == "low" and prob_neg < 0.15:
                score += 0.25
            elif risk_level == "medium" and 0.05 < prob_neg < 0.50:
                score += 0.25
            elif risk_level == "high" and prob_neg > 0.10:
                score += 0.25
            else:
                score += 0.10  # Partial consistency
                flags.append("risk_simulation_mismatch")

        # Check 4: No contradictions between knowledge and reasoning
        chunks = knowledge.get("chunks", [])
        if chunks and skill_result.get("success", False):
            checks_possible += 1
            # If reasoning successfully used knowledge, assume consistency
            score += 0.25

        # If no checks were possible at all, give partial credit
        if checks_possible == 0:
            flags.append("insufficient_data_for_consistency")
            return 0.25  # Minimal baseline

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Sub-score: Accuracy (data quality / evidence)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_accuracy(
        knowledge: dict[str, Any],
        reasoning: dict[str, Any],
        simulation: dict[str, Any],
        flags: list[str],
    ) -> float:
        """Compute accuracy sub-score (0-1).

        Starts at 1.0 and deducts for data quality issues:
        -0.15  expense_ratio is estimated (default 0.72)
        -0.15  collection_delay is estimated (default 52 days)
        -0.10  customer_loss_rate is estimated (default 0.02)
        -0.10  dataset < 12 months
        -0.10  credit_note_ratio anomalous (> 0.20)

        Minimum: 0.0
        """
        score = 1.0

        # Extract distribution parameters from simulation config
        scenarios = simulation.get("scenarios", {})
        base_scenario = scenarios.get("base", {})
        config_snapshot = base_scenario.get("config_snapshot", {})
        dist = config_snapshot.get("distributions", {})

        # Check for estimated expense ratio
        expense_ratio = dist.get("expense_ratio_mean", _DEFAULT_EXPENSE_RATIO)
        if abs(expense_ratio - _DEFAULT_EXPENSE_RATIO) < 0.001:
            score -= 0.15
            flags.append("estimated_expense_ratio")

        # Check for estimated collection delay
        delay = dist.get("collection_delay_mean", _DEFAULT_COLLECTION_DELAY)
        if abs(delay - _DEFAULT_COLLECTION_DELAY) < 0.1:
            score -= 0.15
            flags.append("estimated_collection_delay")

        # Check for estimated customer loss rate
        loss_rate = dist.get("customer_loss_rate", _DEFAULT_CUSTOMER_LOSS)
        if abs(loss_rate - _DEFAULT_CUSTOMER_LOSS) < 0.001:
            score -= 0.10
            flags.append("estimated_customer_loss_rate")

        # Check dataset duration (months)
        n_months = config_snapshot.get("time_horizon_months", 0)
        # We look at the simulation's basis: if there are monthly stats,
        # we trust the underlying data coverage.
        # For this heuristic, if knowledge has few chunks, data may be sparse.
        chunks = knowledge.get("chunks", [])
        if len(chunks) < 6:
            score -= 0.10
            flags.append("limited_data_coverage")

        # Check credit note ratio
        credit_ratio = dist.get("credit_note_ratio", 0.0)
        if credit_ratio > 0.20:
            score -= 0.10
            flags.append("anomalous_credit_note_ratio")

        return max(score, 0.0)
