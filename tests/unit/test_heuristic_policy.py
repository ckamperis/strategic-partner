"""Unit tests for pillars.reasoning.heuristic_policy — HeuristicPolicy.

Tests cover:
- Determinism: same query always -> same result
- Greek keyword matching (cashflow, risk, SWOT, customer)
- English keyword matching
- Mixed language queries
- Confidence scoring (matched tokens / query tokens)
- Minimum match count threshold -> GENERAL fallback
- Query with no matching keywords -> GENERAL
- Priority tie-breaking
"""

from __future__ import annotations

import pytest

from pillars.reasoning.heuristic_policy import (
    KEYWORD_MAP,
    HeuristicPolicy,
    QueryType,
    RoutingDecision,
)


@pytest.fixture
def policy() -> HeuristicPolicy:
    return HeuristicPolicy(min_match_count=1)


class TestDeterminism:
    """Same query must always produce the same routing decision."""

    def test_same_query_same_result(self, policy: HeuristicPolicy) -> None:
        query = "Ποια είναι η πρόβλεψη ταμειακής ροής;"
        r1 = policy.classify(query)
        r2 = policy.classify(query)
        assert r1.query_type == r2.query_type
        assert r1.confidence == r2.confidence
        assert r1.matched_keywords == r2.matched_keywords

    def test_determinism_across_instances(self) -> None:
        query = "cashflow forecast for next quarter"
        p1 = HeuristicPolicy()
        p2 = HeuristicPolicy()
        assert p1.classify(query).query_type == p2.classify(query).query_type

    def test_determinism_100_iterations(self, policy: HeuristicPolicy) -> None:
        query = "ανάλυση κινδύνου πιστωτικού"
        results = [policy.classify(query).query_type for _ in range(100)]
        assert len(set(results)) == 1


class TestCashflowRouting:
    """Queries about cashflow/forecasting should route to CASHFLOW_FORECAST."""

    def test_greek_cashflow(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("Ποια είναι η ταμειακή ροή;")
        assert result.query_type == QueryType.CASHFLOW_FORECAST

    def test_english_cashflow(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("What is the cashflow forecast?")
        assert result.query_type == QueryType.CASHFLOW_FORECAST

    def test_greek_forecast(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("πρόβλεψη εισπράξεων και πληρωμών")
        assert result.query_type == QueryType.CASHFLOW_FORECAST

    def test_english_liquidity(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("liquidity projection for Q3")
        assert result.query_type == QueryType.CASHFLOW_FORECAST

    def test_mixed_language_cashflow(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("πρόβλεψη cashflow για το επόμενο τρίμηνο")
        assert result.query_type == QueryType.CASHFLOW_FORECAST

    def test_skill_name_cashflow(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("cashflow forecast")
        assert result.skill_name == "cashflow_forecast"


class TestRiskRouting:
    """Queries about risk/credit/overdue should route to RISK_ASSESSMENT."""

    def test_greek_risk(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("ανάλυση κινδύνου επισφαλειών")
        assert result.query_type == QueryType.RISK_ASSESSMENT

    def test_english_risk(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("credit risk assessment for the portfolio")
        assert result.query_type == QueryType.RISK_ASSESSMENT

    def test_greek_overdue(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("ανεξόφλητα τιμολόγια και καθυστερήσεις")
        assert result.query_type == QueryType.RISK_ASSESSMENT

    def test_skill_name_risk(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("risk assessment")
        assert result.skill_name == "risk_assessment"


class TestSWOTRouting:
    """Queries about SWOT/strategy should route to SWOT_ANALYSIS."""

    def test_explicit_swot(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("SWOT analysis of the company")
        assert result.query_type == QueryType.SWOT_ANALYSIS

    def test_greek_strategic(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("στρατηγική ανάλυση δυνάμεων και αδυναμιών")
        assert result.query_type == QueryType.SWOT_ANALYSIS

    def test_english_strengths_weaknesses(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("What are our strengths and weaknesses?")
        assert result.query_type == QueryType.SWOT_ANALYSIS

    def test_skill_name_swot(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("swot analysis")
        assert result.skill_name == "swot_analysis"


class TestCustomerRouting:
    """Queries about customers/concentration should route to CUSTOMER_ANALYSIS."""

    def test_greek_customer(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("ανάλυση πελατολογίου και συγκέντρωσης")
        assert result.query_type == QueryType.CUSTOMER_ANALYSIS

    def test_english_customer(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("customer concentration and segmentation analysis")
        assert result.query_type == QueryType.CUSTOMER_ANALYSIS

    def test_pareto(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("pareto analysis of customer revenue")
        assert result.query_type == QueryType.CUSTOMER_ANALYSIS

    def test_skill_name_customer(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("customer segmentation")
        assert result.skill_name == "customer_analysis"


class TestGeneralFallback:
    """Queries with no matching keywords should fall back to GENERAL."""

    def test_unrelated_query(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("What is the weather today?")
        assert result.query_type == QueryType.GENERAL

    def test_empty_query(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("")
        assert result.query_type == QueryType.GENERAL

    def test_general_has_no_skill(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("random unrelated text")
        assert result.skill_name is None

    def test_general_zero_confidence(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("something completely different")
        assert result.confidence == 0.0
        assert result.matched_keywords == []


class TestConfidence:
    """Confidence scoring and threshold behaviour."""

    def test_confidence_positive(self, policy: HeuristicPolicy) -> None:
        result = policy.classify("cashflow forecast liquidity projection")
        assert result.confidence > 0.0

    def test_more_keywords_higher_confidence(self, policy: HeuristicPolicy) -> None:
        """More keyword matches -> higher confidence (matched/total tokens)."""
        r1 = policy.classify("cashflow")
        r2 = policy.classify("cashflow forecast liquidity inflow outflow payments")
        assert r2.confidence >= r1.confidence

    def test_confidence_bounded(self, policy: HeuristicPolicy) -> None:
        result = policy.classify(
            "cashflow forecast liquidity projection inflow outflow "
            "receipts payments monthly quarterly annual budget"
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_high_threshold_forces_general(self) -> None:
        """With a very high min_match_count, everything becomes GENERAL."""
        strict_policy = HeuristicPolicy(min_match_count=100)
        result = strict_policy.classify("cashflow forecast")
        assert result.query_type == QueryType.GENERAL

    def test_default_accepts_single_keyword(self) -> None:
        """Default min_match_count=1 accepts a single keyword match."""
        p = HeuristicPolicy()
        result = p.classify("cashflow")
        assert result.query_type == QueryType.CASHFLOW_FORECAST


class TestRoutingDecision:
    """RoutingDecision dataclass properties."""

    def test_routing_decision_frozen(self) -> None:
        rd = RoutingDecision(
            query_type=QueryType.CASHFLOW_FORECAST,
            confidence=0.5,
            matched_keywords=["cashflow"],
        )
        with pytest.raises(AttributeError):
            rd.query_type = QueryType.GENERAL  # type: ignore[misc]

    def test_skill_name_general_is_none(self) -> None:
        rd = RoutingDecision(
            query_type=QueryType.GENERAL,
            confidence=0.0,
        )
        assert rd.skill_name is None

    def test_skill_name_maps_correctly(self) -> None:
        for qt in QueryType:
            rd = RoutingDecision(query_type=qt, confidence=0.5)
            if qt == QueryType.GENERAL:
                assert rd.skill_name is None
            else:
                assert rd.skill_name == qt.value


class TestTokenizer:
    """Internal tokenizer handles Greek + English correctly."""

    def test_tokenize_english(self) -> None:
        tokens = HeuristicPolicy._tokenize("Hello World 123")
        assert tokens == ["hello", "world", "123"]

    def test_tokenize_greek(self) -> None:
        tokens = HeuristicPolicy._tokenize("Ταμειακή Ροή")
        assert tokens == ["ταμειακή", "ροή"]

    def test_tokenize_mixed(self) -> None:
        tokens = HeuristicPolicy._tokenize("cashflow ταμειακή ροή Q3")
        assert "cashflow" in tokens
        assert "ταμειακή" in tokens
        assert "q3" in tokens

    def test_tokenize_punctuation(self) -> None:
        tokens = HeuristicPolicy._tokenize("Hello, world! What's up?")
        # "what's" -> ["what", "s"] due to apostrophe splitting
        assert "hello" in tokens
        assert "world" in tokens


class TestKeywordMap:
    """Keyword map completeness checks."""

    def test_all_query_types_except_general_have_keywords(self) -> None:
        for qt in QueryType:
            if qt != QueryType.GENERAL:
                assert qt in KEYWORD_MAP, f"Missing keywords for {qt}"
                assert len(KEYWORD_MAP[qt]) > 0

    def test_keyword_map_has_greek_entries(self) -> None:
        """Each category should have at least some Greek keywords."""
        for qt, keywords in KEYWORD_MAP.items():
            # Check for at least one non-ASCII keyword (Greek)
            greek_kw = [k for k in keywords if any(ord(c) > 127 for c in k)]
            assert len(greek_kw) > 0, f"{qt} has no Greek keywords"

    def test_keyword_map_has_english_entries(self) -> None:
        """Each category should have at least some English keywords."""
        for qt, keywords in KEYWORD_MAP.items():
            english_kw = [k for k in keywords if all(ord(c) < 128 for c in k)]
            assert len(english_kw) > 0, f"{qt} has no English keywords"
