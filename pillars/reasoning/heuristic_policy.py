"""Heuristic policy — deterministic query router.

Classifies natural-language queries into skill types using keyword matching.
Determinism is critical for reproducibility: same query -> same skill every time.

Uses a two-phase approach:
1. Keyword scan: Greek + English keywords map to QueryType
2. Confidence scoring: fraction of matched keywords vs total tokens

Design decision: Heuristic policy over DRL because:
- Deterministic O(1) routing (no model inference)
- Transparent decision-making (fully explainable)
- No training data required
- Sufficient for PoC scope

References:
    Thesis Section 3.3.2 — Reasoning Pillar, Heuristic Policy
    Thesis Section 4.x — Implementation, Deterministic Routing
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger()


class QueryType(str, Enum):
    """Supported analytical skill types.

    Each maps to a YAML skill definition in pillars/reasoning/skills/.
    """

    CASHFLOW_FORECAST = "cashflow_forecast"
    RISK_ASSESSMENT = "risk_assessment"
    SWOT_ANALYSIS = "swot_analysis"
    CUSTOMER_ANALYSIS = "customer_analysis"
    GENERAL = "general"  # Fallback — no specialised skill


@dataclass(frozen=True)
class RoutingDecision:
    """Result of the heuristic routing process.

    Attributes:
        query_type: The classified query type.
        confidence: Confidence score ∈ [0, 1] based on keyword overlap.
        matched_keywords: Keywords from the query that triggered this route.
        skill_name: Name of the YAML skill to execute (or None for GENERAL).
    """

    query_type: QueryType
    confidence: float
    matched_keywords: list[str] = field(default_factory=list)

    @property
    def skill_name(self) -> str | None:
        """Map query type to skill definition filename (without .yaml)."""
        if self.query_type == QueryType.GENERAL:
            return None
        return self.query_type.value


# ── Keyword Maps ─────────────────────────────────────────────────────
# Each key is a QueryType, value is a set of Greek + English keywords.
# Lowercase matching is used. Greek accented forms included for coverage.

KEYWORD_MAP: dict[QueryType, set[str]] = {
    QueryType.CASHFLOW_FORECAST: {
        # Greek (nominative + genitive + accusative forms)
        "ταμειακή", "ταμειακής", "ταμειακές", "ταμείο", "ταμείου",
        "ροή", "ροής", "ροές", "ροών",
        "πρόβλεψη", "πρόβλεψης", "προβλέψεις", "προβλέψεων",
        "ρευστότητα", "ρευστότητας",
        "εισπράξεις", "εισπράξεων", "είσπραξη",
        "πληρωμές", "πληρωμών", "πληρωμή",
        "εισροές", "εισροών", "εκροές", "εκροών",
        "μηνιαίο", "μηνιαία", "τριμηνιαίο", "τριμηνιαία",
        "ετήσιο", "ετήσια", "προϋπολογισμός", "προϋπολογισμού",
        "ισοζύγιο", "ισοζυγίου",
        # English
        "cashflow", "cash", "flow", "forecast", "forecasting", "projection",
        "liquidity", "inflow", "outflow", "receipts", "payments",
        "monthly", "quarterly", "annual", "budget",
    },
    QueryType.RISK_ASSESSMENT: {
        # Greek (nominative + genitive + accusative forms)
        "κίνδυνος", "κινδύνου", "κίνδυνο", "κίνδυνοι", "κινδύνων",
        "ρίσκο", "ρίσκα",
        "απειλή", "απειλής", "απειλές", "απειλών",
        "πιστωτικός", "πιστωτικό", "πιστωτικού", "πιστωτικά",
        "ανεξόφλητο", "ανεξόφλητα", "ανεξόφλητων",
        "καθυστέρηση", "καθυστέρησης", "καθυστερήσεις", "καθυστερήσεων",
        "επισφάλεια", "επισφάλειας", "επισφαλειών", "επισφαλής",
        "εκκρεμότητα", "εκκρεμότητας",
        "αβεβαιότητα", "αβεβαιότητας",
        "ρήτρα", "αθέτηση", "αθέτησης",
        # English
        "risk", "risks", "assessment", "threat", "threats", "credit",
        "overdue", "default", "exposure", "vulnerability", "hazard",
        "delinquency", "bad_debt", "collection",
    },
    QueryType.SWOT_ANALYSIS: {
        # Greek (nominative + genitive + accusative forms)
        # Note: "ανάλυση" is intentionally omitted — too generic
        "swot",
        "δυνάμεις", "δυνάμεων", "δύναμη", "δύναμης",
        "αδυναμίες", "αδυναμιών", "αδυναμία", "αδυναμίας",
        "ευκαιρίες", "ευκαιριών", "ευκαιρία", "ευκαιρίας",
        "στρατηγική", "στρατηγικής", "στρατηγικό", "στρατηγικός",
        "ανταγωνισμός", "ανταγωνισμού", "ανταγωνιστικό", "ανταγωνιστικά",
        "πλεονέκτημα", "πλεονεκτήματα", "μειονέκτημα", "μειονεκτήματα",
        "αξιολόγηση", "αξιολόγησης",
        "θέση", "θέσης",
        # English
        "swot", "strengths", "weaknesses", "opportunities",
        "strategic", "strategy", "competitive", "advantage",
        "disadvantage", "positioning",
    },
    QueryType.CUSTOMER_ANALYSIS: {
        # Greek (nominative + genitive + accusative forms)
        "πελάτης", "πελάτη", "πελάτες", "πελατών",
        "πελατολόγιο", "πελατολογίου",
        "συγκέντρωση", "συγκέντρωσης", "συγκεντρώσεις",
        "κατανομή", "κατανομής",
        "αξία", "αξίας",
        "τζίρος", "τζίρου",
        "έσοδα", "εσόδων",
        "τμηματοποίηση", "τμηματοποίησης",
        "ομάδα", "ομάδας", "ομάδες", "ομάδων",
        "πιστότητα", "πιστότητας",
        "διατήρηση", "διατήρησης",
        # English
        "customer", "customers", "client", "clients", "concentration",
        "distribution", "revenue", "pareto", "segmentation",
        "segment", "rfm", "loyalty", "retention", "churn", "ltv",
        "lifetime", "value", "top",
    },
}


class HeuristicPolicy:
    """Deterministic keyword-based query classifier.

    Scans the query for known keywords and selects the QueryType
    with the highest keyword overlap. Ties are broken by priority
    order (cashflow > risk > swot > customer > general).

    Scoring uses matched keyword count as the primary metric, with
    confidence computed as the fraction of query tokens that are
    relevant keywords (i.e., topic coverage of the query).

    Args:
        min_match_count: Minimum number of matched keywords to accept
            a classification. Below this, falls back to GENERAL. Default: 1.
    """

    # Priority order for tie-breaking (lower index = higher priority)
    _PRIORITY_ORDER: list[QueryType] = [
        QueryType.CASHFLOW_FORECAST,
        QueryType.RISK_ASSESSMENT,
        QueryType.SWOT_ANALYSIS,
        QueryType.CUSTOMER_ANALYSIS,
    ]

    def __init__(self, min_match_count: int = 1) -> None:
        self._min_match_count = min_match_count

    def classify(self, query: str) -> RoutingDecision:
        """Classify a query into a QueryType.

        Algorithm:
        1. Tokenize query (lowercase, split on non-alphanumeric).
        2. For each QueryType, count keyword matches.
        3. Select the type with the most matched keywords.
        4. Break ties using priority order.
        5. Confidence = matched_count / total_query_tokens.

        Args:
            query: The user's natural-language query (Greek or English).

        Returns:
            A RoutingDecision with type, confidence, and matched keywords.
        """
        tokens = self._tokenize(query)
        token_set = set(tokens)
        n_tokens = max(len(token_set), 1)  # avoid div-by-zero

        best_type = QueryType.GENERAL
        best_count = 0
        best_matched: list[str] = []

        for query_type in self._PRIORITY_ORDER:
            keywords = KEYWORD_MAP[query_type]
            matched = token_set & keywords
            match_count = len(matched)

            if match_count == 0:
                continue

            # Primary: highest match count wins
            # Secondary: priority order (earlier in list = higher priority)
            if match_count > best_count:
                best_count = match_count
                best_type = query_type
                best_matched = sorted(matched)

        # Apply minimum match count threshold
        if best_count < self._min_match_count:
            best_type = QueryType.GENERAL
            best_count = 0
            best_matched = []

        # Confidence = fraction of query tokens that matched
        confidence = round(best_count / n_tokens, 4) if best_count > 0 else 0.0

        decision = RoutingDecision(
            query_type=best_type,
            confidence=confidence,
            matched_keywords=best_matched,
        )

        logger.info(
            "heuristic_policy.classify",
            query_type=decision.query_type.value,
            confidence=decision.confidence,
            matched_count=len(decision.matched_keywords),
            matched_keywords=decision.matched_keywords,
        )

        return decision

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase tokens.

        Handles Greek + English text. Splits on whitespace and
        non-alphanumeric characters (preserving Unicode letters).

        Args:
            text: Input text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        # Split on anything that is not a Unicode letter or digit
        tokens = re.findall(r"[\w]+", text.lower(), re.UNICODE)
        return tokens
