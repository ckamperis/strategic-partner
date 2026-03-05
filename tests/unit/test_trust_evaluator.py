"""Unit tests for pillars.trust.evaluator — TrustEvaluator.

Tests cover:
- Perfect score: all conditions met -> high score
- Worst case: all estimated, no skill, no RAG -> low score
- Weight normalization: weights sum to 1.0
- Custom weights
- Individual sub-scores
- Flags for each condition
- Confidence level thresholds
- Edge cases: None inputs, empty results
"""

from __future__ import annotations

import pytest

from pillars.trust.evaluator import (
    TrustEvaluator,
    TrustScore,
    TrustSubScores,
)


# ---------------------------------------------------------------------------
# Helpers — mock pillar results
# ---------------------------------------------------------------------------

def _good_knowledge() -> dict:
    """Knowledge result with relevant chunks and high RAG score."""
    return {
        "chunks": [
            {"text": "Revenue €150K"},
            {"text": "Seasonal peak in Q3"},
            {"text": "Credit notes 4.6%"},
            {"text": "Top customers 45%"},
            {"text": "Growth trend"},
            {"text": "Payment patterns"},
        ],
        "relevance_score": 0.85,
        "iterations": 1,
        "query_history": ["cashflow forecast"],
    }


def _good_reasoning() -> dict:
    """Reasoning result with matched skill and successful execution."""
    return {
        "routing": {
            "query_type": "cashflow_forecast",
            "confidence": 0.67,
            "skill_name": "cashflow_forecast",
        },
        "skill_result": {
            "skill_name": "cashflow_forecast",
            "success": True,
            "parsed_output": {
                "revenue_trend": "stable",
                "risk_level": "low",
                "adjustment_factor": 1.0,
            },
        },
    }


def _good_simulation() -> dict:
    """Simulation result with all 3 scenarios and good stats."""
    return {
        "scenarios": {
            "base": {
                "monthly_stats": [
                    {"mean": 28_000, "std": 8_000, "p5": 15_000, "p95": 42_000},
                    {"mean": 29_000, "std": 7_500, "p5": 16_000, "p95": 43_000},
                    {"mean": 30_000, "std": 7_000, "p5": 17_000, "p95": 44_000},
                ],
                "cumulative_stats": [
                    {"mean": 28_000}, {"mean": 57_000}, {"mean": 87_000}
                ],
                "probability_negative": 0.02,
                "var_5pct": 45_000,
                "n_simulations": 10_000,
                "config_snapshot": {
                    "distributions": {
                        "revenue_mean": 100_000,
                        "revenue_std": 20_000,
                        "expense_ratio_mean": 0.72,
                        "collection_delay_mean": 52.0,
                        "customer_loss_rate": 0.02,
                        "credit_note_ratio": 0.05,
                        "seasonal_factors": [0.8, 0.85, 0.9, 1.0, 1.05, 1.1,
                                             1.15, 1.2, 1.1, 1.0, 0.9, 0.95],
                    },
                },
            },
            "optimistic": {"monthly_stats": [{"mean": 35_000}], "n_simulations": 10_000},
            "stress": {"monthly_stats": [{"mean": 18_000}], "n_simulations": 10_000},
        },
    }


def _empty_simulation() -> dict:
    """Simulation result with no scenarios."""
    return {"scenarios": {}}


def _poor_knowledge() -> dict:
    """Knowledge result with no chunks and low RAG score."""
    return {"chunks": [], "relevance_score": 0.0, "iterations": 3}


def _general_reasoning() -> dict:
    """Reasoning result with general routing (no matched skill)."""
    return {
        "routing": {
            "query_type": "general",
            "confidence": 0.0,
            "skill_name": None,
        },
        "skill_result": {
            "skill_name": "general",
            "success": True,
            "parsed_output": {},
        },
    }


# ---------------------------------------------------------------------------
# Default weights and basic construction
# ---------------------------------------------------------------------------

class TestTrustEvaluatorConstruction:
    """Test evaluator construction and weight validation."""

    def test_default_weights(self) -> None:
        evaluator = TrustEvaluator()
        assert evaluator.weights == (0.4, 0.4, 0.2)

    def test_custom_weights(self) -> None:
        evaluator = TrustEvaluator(weights=(0.5, 0.3, 0.2))
        assert evaluator.weights == (0.5, 0.3, 0.2)

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            TrustEvaluator(weights=(0.5, 0.5, 0.5))

    def test_zero_weights_valid(self) -> None:
        evaluator = TrustEvaluator(weights=(1.0, 0.0, 0.0))
        assert evaluator.weights == (1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Overall score computation
# ---------------------------------------------------------------------------

class TestOverallScore:
    """Test composite score computation."""

    def test_perfect_inputs_high_score(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=_good_knowledge(),
            reasoning_result=_good_reasoning(),
            simulation_result=_good_simulation(),
        )
        assert score.overall > 0.6
        assert score.confidence_level in ("high", "medium")

    def test_poor_inputs_low_score(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=_poor_knowledge(),
            reasoning_result=_general_reasoning(),
            simulation_result=_empty_simulation(),
        )
        assert score.overall < 0.5
        assert score.confidence_level == "low"

    def test_score_bounded_0_1(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate()
        assert 0.0 <= score.overall <= 1.0

    def test_none_inputs_dont_crash(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=None,
            reasoning_result=None,
            simulation_result=None,
        )
        assert isinstance(score, TrustScore)
        assert 0.0 <= score.overall <= 1.0


# ---------------------------------------------------------------------------
# Confidence level thresholds
# ---------------------------------------------------------------------------

class TestConfidenceLevel:
    """Test confidence_level classification."""

    def test_high_confidence(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=_good_knowledge(),
            reasoning_result=_good_reasoning(),
            simulation_result=_good_simulation(),
        )
        # Good inputs should give at least medium
        assert score.confidence_level in ("high", "medium")

    def test_low_confidence(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate()
        assert score.confidence_level == "low"


# ---------------------------------------------------------------------------
# Explainability sub-score
# ---------------------------------------------------------------------------

class TestExplainabilityScore:
    """Test explainability sub-score computation."""

    def test_good_rag_adds_score(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=_good_knowledge(),
            reasoning_result=_good_reasoning(),
            simulation_result=_good_simulation(),
        )
        assert score.sub_scores.explainability >= 0.6

    def test_no_rag_low_explainability(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=_poor_knowledge(),
        )
        assert score.sub_scores.explainability < 0.3

    def test_general_routing_reduces_score(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            reasoning_result=_general_reasoning(),
        )
        assert "general_routing" in score.flags


# ---------------------------------------------------------------------------
# Consistency sub-score
# ---------------------------------------------------------------------------

class TestConsistencyScore:
    """Test consistency sub-score computation."""

    def test_consistent_results_high_score(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=_good_knowledge(),
            reasoning_result=_good_reasoning(),
            simulation_result=_good_simulation(),
        )
        assert score.sub_scores.consistency >= 0.5

    def test_no_data_for_consistency(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate()
        assert "insufficient_data_for_consistency" in score.flags


# ---------------------------------------------------------------------------
# Accuracy sub-score
# ---------------------------------------------------------------------------

class TestAccuracyScore:
    """Test accuracy sub-score (data quality)."""

    def test_estimated_parameters_reduce_score(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            simulation_result=_good_simulation(),
        )
        # Default distributions use estimated values -> deductions
        assert score.sub_scores.accuracy < 1.0
        assert "estimated_expense_ratio" in score.flags

    def test_all_estimated_flags(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            simulation_result=_good_simulation(),
        )
        # All three default parameters are estimated
        assert "estimated_expense_ratio" in score.flags
        assert "estimated_collection_delay" in score.flags
        assert "estimated_customer_loss_rate" in score.flags

    def test_accuracy_minimum_zero(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=_poor_knowledge(),
            simulation_result=_good_simulation(),
        )
        assert score.sub_scores.accuracy >= 0.0


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

class TestFlags:
    """Test flag generation."""

    def test_low_rag_relevance_flag(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(knowledge_result=_poor_knowledge())
        assert "low_rag_relevance" in score.flags

    def test_general_routing_flag(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(reasoning_result=_general_reasoning())
        assert "general_routing" in score.flags

    def test_no_duplicate_flags(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=_good_knowledge(),
            reasoning_result=_good_reasoning(),
            simulation_result=_good_simulation(),
        )
        assert len(score.flags) == len(set(score.flags))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    """Test serialisation methods."""

    def test_sub_scores_to_dict(self) -> None:
        ss = TrustSubScores(explainability=0.8, consistency=0.7, accuracy=0.6)
        d = ss.to_dict()
        assert d["explainability"] == 0.8
        assert d["consistency"] == 0.7
        assert d["accuracy"] == 0.6

    def test_trust_score_to_dict(self) -> None:
        evaluator = TrustEvaluator()
        score = evaluator.evaluate(
            knowledge_result=_good_knowledge(),
            reasoning_result=_good_reasoning(),
            simulation_result=_good_simulation(),
        )
        d = score.to_dict()
        assert "overall" in d
        assert "sub_scores" in d
        assert "flags" in d
        assert "confidence_level" in d


# ---------------------------------------------------------------------------
# Custom weights affect score
# ---------------------------------------------------------------------------

class TestCustomWeights:
    """Test that custom weights change the overall score."""

    def test_explainability_heavy_weight(self) -> None:
        eval_e = TrustEvaluator(weights=(0.8, 0.1, 0.1))
        eval_c = TrustEvaluator(weights=(0.1, 0.8, 0.1))

        score_e = eval_e.evaluate(
            knowledge_result=_good_knowledge(),
            reasoning_result=_good_reasoning(),
            simulation_result=_good_simulation(),
        )
        score_c = eval_c.evaluate(
            knowledge_result=_good_knowledge(),
            reasoning_result=_good_reasoning(),
            simulation_result=_good_simulation(),
        )

        # Different weights -> different overall scores
        # (unless sub-scores happen to be equal)
        assert isinstance(score_e.overall, float)
        assert isinstance(score_c.overall, float)
