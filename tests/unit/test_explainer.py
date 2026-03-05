"""Unit tests for pillars.trust.explainer — ExplanationGenerator.

Tests cover:
- Template mode: no LLM, produces valid Explanation
- All fields populated
- Caveats include trust flags
- Greek text in summary
- Missing simulation (degraded mode)
- Data sources extraction
- Methodology text
"""

from __future__ import annotations

import pytest

from pillars.trust.evaluator import TrustScore, TrustSubScores
from pillars.trust.explainer import Explanation, ExplanationGenerator
from pillars.trust.shap_explainer import FactorContribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trust_score(
    overall: float = 0.72,
    flags: list[str] | None = None,
    confidence: str = "medium",
) -> TrustScore:
    return TrustScore(
        overall=overall,
        sub_scores=TrustSubScores(
            explainability=0.8, consistency=0.7, accuracy=0.6,
        ),
        flags=flags or [],
        confidence_level=confidence,
    )


def _make_simulation() -> dict:
    return {
        "scenarios": {
            "base": {
                "monthly_stats": [
                    {"mean": 28_000, "std": 8_000, "p5": 15_000, "p95": 42_000},
                    {"mean": 29_000, "std": 7_500, "p5": 16_000, "p95": 43_000},
                    {"mean": 30_000, "std": 7_000, "p5": 17_000, "p95": 44_000},
                ],
                "probability_negative": 0.02,
                "n_simulations": 10_000,
                "config_snapshot": {
                    "distributions": {"revenue_mean": 100_000}
                },
            },
            "optimistic": {"monthly_stats": [{"mean": 35_000}]},
            "stress": {"monthly_stats": [{"mean": 18_000}]},
        },
    }


def _make_reasoning() -> dict:
    return {
        "routing": {
            "query_type": "cashflow_forecast",
            "confidence": 0.67,
            "skill_name": "cashflow_forecast",
        },
        "skill_result": {
            "skill_name": "cashflow_forecast",
            "success": True,
        },
    }


def _make_factors() -> list[FactorContribution]:
    return [
        FactorContribution(
            factor_name="expense_ratio", impact=-7000,
            direction="negative", magnitude="high",
            evidence="Expense ratio: 0.72",
        ),
        FactorContribution(
            factor_name="seasonal_pattern", impact=2000,
            direction="positive", magnitude="medium",
            evidence="Seasonal range: 0.40",
        ),
    ]


# ---------------------------------------------------------------------------
# Template mode
# ---------------------------------------------------------------------------

class TestTemplateMode:
    """Template-based explanation (no LLM)."""

    def test_produces_valid_explanation(self) -> None:
        gen = ExplanationGenerator(llm_client=None)
        exp = gen.generate(
            trust_score=_make_trust_score(),
            reasoning_result=_make_reasoning(),
            simulation_result=_make_simulation(),
            shap_factors=_make_factors(),
        )
        assert isinstance(exp, Explanation)

    def test_all_fields_populated(self) -> None:
        gen = ExplanationGenerator()
        exp = gen.generate(
            trust_score=_make_trust_score(),
            reasoning_result=_make_reasoning(),
            simulation_result=_make_simulation(),
            shap_factors=_make_factors(),
        )
        assert exp.summary
        assert exp.reasoning_trace
        assert len(exp.key_factors) == 2
        assert exp.data_sources
        assert exp.confidence == "medium"
        assert exp.methodology

    def test_summary_contains_greek(self) -> None:
        gen = ExplanationGenerator()
        exp = gen.generate(
            trust_score=_make_trust_score(),
            simulation_result=_make_simulation(),
        )
        # Greek text should be present
        assert "πρόβλεψη" in exp.summary or "ταμειακ" in exp.summary

    def test_summary_contains_numbers(self) -> None:
        gen = ExplanationGenerator()
        exp = gen.generate(
            trust_score=_make_trust_score(),
            simulation_result=_make_simulation(),
        )
        # Should contain €K values
        assert "€" in exp.summary or "K" in exp.summary


# ---------------------------------------------------------------------------
# Caveats from flags
# ---------------------------------------------------------------------------

class TestCaveats:
    """Caveats should map from trust flags."""

    def test_estimated_expense_ratio_caveat(self) -> None:
        gen = ExplanationGenerator()
        ts = _make_trust_score(flags=["estimated_expense_ratio"])
        exp = gen.generate(trust_score=ts)
        assert len(exp.caveats) == 1
        assert "εξόδων" in exp.caveats[0] or "0.72" in exp.caveats[0]

    def test_estimated_collection_delay_caveat(self) -> None:
        gen = ExplanationGenerator()
        ts = _make_trust_score(flags=["estimated_collection_delay"])
        exp = gen.generate(trust_score=ts)
        assert len(exp.caveats) == 1
        assert "είσπραξη" in exp.caveats[0] or "52" in exp.caveats[0]

    def test_multiple_flags_multiple_caveats(self) -> None:
        gen = ExplanationGenerator()
        ts = _make_trust_score(
            flags=["estimated_expense_ratio", "low_rag_relevance", "general_routing"]
        )
        exp = gen.generate(trust_score=ts)
        assert len(exp.caveats) == 3

    def test_unknown_flag_still_produces_caveat(self) -> None:
        gen = ExplanationGenerator()
        ts = _make_trust_score(flags=["some_unknown_flag"])
        exp = gen.generate(trust_score=ts)
        assert len(exp.caveats) == 1
        assert "some_unknown_flag" in exp.caveats[0]


# ---------------------------------------------------------------------------
# Degraded mode
# ---------------------------------------------------------------------------

class TestDegradedMode:
    """Handle missing pillar results gracefully."""

    def test_no_simulation(self) -> None:
        gen = ExplanationGenerator()
        exp = gen.generate(
            trust_score=_make_trust_score(),
            simulation_result={},
        )
        assert "ελλιπ" in exp.summary.lower() or "δεν" in exp.summary.lower()

    def test_no_reasoning(self) -> None:
        gen = ExplanationGenerator()
        exp = gen.generate(
            trust_score=_make_trust_score(),
            reasoning_result={},
            simulation_result=_make_simulation(),
        )
        assert exp.reasoning_trace

    def test_empty_everything(self) -> None:
        gen = ExplanationGenerator()
        exp = gen.generate(trust_score=_make_trust_score())
        assert isinstance(exp, Explanation)
        assert exp.confidence == "medium"


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

class TestDataSources:
    """Test data source extraction."""

    def test_sources_include_reasoning(self) -> None:
        gen = ExplanationGenerator()
        exp = gen.generate(
            trust_score=_make_trust_score(),
            reasoning_result=_make_reasoning(),
            simulation_result=_make_simulation(),
        )
        assert any("cashflow" in s.lower() for s in exp.data_sources)

    def test_sources_include_simulation(self) -> None:
        gen = ExplanationGenerator()
        exp = gen.generate(
            trust_score=_make_trust_score(),
            simulation_result=_make_simulation(),
        )
        assert any("monte carlo" in s.lower() for s in exp.data_sources)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestExplanationSerialisation:
    """Test Explanation.to_dict()."""

    def test_to_dict_keys(self) -> None:
        gen = ExplanationGenerator()
        exp = gen.generate(
            trust_score=_make_trust_score(),
            simulation_result=_make_simulation(),
            shap_factors=_make_factors(),
        )
        d = exp.to_dict()
        expected_keys = {
            "summary", "reasoning_trace", "key_factors",
            "data_sources", "confidence", "caveats", "methodology",
        }
        assert set(d.keys()) == expected_keys
        assert len(d["key_factors"]) == 2
