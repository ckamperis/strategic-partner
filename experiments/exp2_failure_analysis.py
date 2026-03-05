"""
Exp2 Failure Pattern Analysis — Self-Correcting RAG.

Analyses the 5 queries that failed to exceed the relevance threshold (0.75)
after 3 Self-Correcting RAG iterations. Categorises each failure and
identifies the fundamental boundary between retrieval and reasoning.

Thesis reference: Section 5.4 — Self-Correcting RAG, Section 6.X — Limitations
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXP2_FILE = BASE_DIR / "data" / "results" / "exp2_rag_iterations.json"
OUTPUT_FILE = BASE_DIR / "data" / "results" / "exp2_failure_analysis.json"

# ── Failure categories ──────────────────────────────────────────────
FAILURE_CATEGORIES = {
    "cross_temporal": "Queries requiring comparison of data across multiple time periods",
    "inferential": "Queries asking for conclusions or recommendations not explicitly in the data",
    "correlation": "Queries about relationships between variables requiring computation",
    "data_gap": "The specific data requested does not exist in the embedded chunks",
}

# ── Human-annotated failure analysis ────────────────────────────────
# Each failed query is categorised with explanation and refinement assessment.
# The query_history from the experiment shows how the LLM refined each query.

FAILURE_ANNOTATIONS = {
    8: {
        "query": "Πώς επηρεάζει η εποχικότητα τον κίνδυνο ελλειμμάτων;",
        "final_score": 0.70,
        "failure_category": "cross_temporal",
        "explanation": (
            "Requires linking seasonality data with deficit risk across "
            "multiple time periods. Seasonality information is distributed "
            "across monthly narrative chunks — no single chunk contains the "
            "full seasonal pattern AND deficit analysis. The answer requires "
            "synthesising 12+ monthly data points into a causal narrative."
        ),
        "refinement_pattern": (
            "Iteration 1: Original query (0.70). "
            "Iteration 2: Added 'πωλήσεις' (sales) to broaden retrieval — "
            "score unchanged because broadening retrieves different months "
            "but still not all relevant ones. "
            "Iteration 3: Added 'διαφορετικές περιόδους' (different periods) — "
            "same problem, k=5 chunks insufficient for 12-month pattern."
        ),
        "refinement_helped": False,
        "notes": (
            "Query refinement broadened search but didn't help because "
            "the answer requires synthesis across many chunks (12+ months), "
            "not retrieval of a single relevant passage."
        ),
    },
    9: {
        "query": "Σύγκριση του δεύτερου τριμήνου με το πρώτο σε επίπεδο ταμειακών ροών",
        "final_score": 0.705,
        "failure_category": "cross_temporal",
        "explanation": (
            "Explicitly asks to COMPARE Q2 vs Q1 cashflows. This requires "
            "retrieving data from at least 6 months (Jan-Jun) and computing "
            "differences. With k=5, the retriever can fetch at most 5 chunks, "
            "which may not cover both quarters completely. The comparison "
            "itself is a computation, not a retrieval task."
        ),
        "refinement_pattern": (
            "Iteration 1: Original query (0.705). "
            "Iteration 2: Simplified to 'Σύγκριση ταμειακών ροών δεύτερου "
            "τριμήνου με πρώτο τριμήνου' — slightly more targeted but "
            "same fundamental limitation. "
            "Iteration 3: Added year '2022' — narrows to specific year "
            "but still needs 6 months of data and a comparison computation."
        ),
        "refinement_helped": False,
        "notes": (
            "Even with perfect retrieval, comparing two quarters requires "
            "arithmetic that RAG cannot perform. This is a reasoning task "
            "disguised as a retrieval query."
        ),
    },
    10: {
        "query": "Ποια η σχέση μεταξύ αριθμού πελατών και μέσου τιμολογίου;",
        "final_score": 0.71,
        "failure_category": "correlation",
        "explanation": (
            "Asks about the RELATIONSHIP between number of customers and "
            "average invoice value. This requires: (1) computing customer "
            "count per period, (2) computing average invoice per period, "
            "(3) correlating the two. None of these are stored as "
            "pre-computed values in the narrative chunks."
        ),
        "refinement_pattern": (
            "Iteration 1: Original query (0.71). "
            "Iteration 2: Rephrased to 'Ποια η μέση τιμή τιμολογίου ανά "
            "πελάτη;' (average invoice per customer) — different question, "
            "partially answerable. "
            "Iteration 3: Added time range '2019-2023' — makes query more "
            "specific but the correlation analysis still cannot be retrieved."
        ),
        "refinement_helped": False,
        "notes": (
            "The LLM refinement shifted the query from 'correlation between "
            "two variables' to 'average value' — a simpler but different "
            "question. This shows the LLM understands the difficulty but "
            "cannot actually make the original query retrievable."
        ),
    },
    12: {
        "query": "Ποιος μήνας έχει τον μεγαλύτερο κίνδυνο αρνητικής ταμειακής ροής;",
        "final_score": 0.557,
        "failure_category": "cross_temporal",
        "explanation": (
            "Finding the WORST month requires comparing all 12 months' "
            "cashflow risk — a superlative query across the entire dataset. "
            "With k=5, only 5 of 12 months can be retrieved. Even if the "
            "right months are retrieved, determining the 'maximum risk' "
            "month requires ranking across all periods. This is the lowest-"
            "scoring failed query (0.557), confirming that superlative "
            "queries over distributed data are the hardest for RAG."
        ),
        "refinement_pattern": (
            "Iteration 1: Singular 'Ποιος μήνας' (which month) — score 0.557. "
            "Iteration 2: Plural 'Ποιοι μήνες' (which months) — broadens "
            "but doesn't improve because the problem is coverage, not query "
            "formulation. "
            "Iteration 3: Added 'και γιατί;' (and why?) — asks for "
            "explanation but still can't retrieve all 12 months for ranking."
        ),
        "refinement_helped": False,
        "notes": (
            "Lowest score among all failures (0.557). Superlative queries "
            "('which month has the MOST...') are fundamentally incompatible "
            "with top-k retrieval — you need ALL data to find the maximum."
        ),
    },
    13: {
        "query": "Πώς θα μπορούσε να βελτιωθεί η είσπραξη;",
        "final_score": 0.70,
        "failure_category": "inferential",
        "explanation": (
            "Asks 'how could collection be improved?' — a recommendation "
            "question. The ERP data contains transaction records, not "
            "improvement strategies. The answer requires domain knowledge "
            "about collection practices that is NOT in the embedded data. "
            "RAG retrieves payment-related chunks but they describe WHAT "
            "happened, not HOW to improve."
        ),
        "refinement_pattern": (
            "Iteration 1: Original question (0.70). "
            "Iteration 2: Rephrased to 'Ποιες είναι οι στρατηγικές για τη "
            "βελτίωση της είσπραξης;' (what are the strategies) — more "
            "specific but strategies are not in ERP data. "
            "Iteration 3: Added 'διαχείρισης πιστώσεων' (credit management) "
            "— retrieves credit-related chunks but still no actionable "
            "recommendations in the data."
        ),
        "refinement_helped": False,
        "notes": (
            "This is a pure reasoning task: the answer must be GENERATED "
            "from understanding, not RETRIEVED from documents. The Reasoning "
            "Pillar (skill execution) is better suited for this type of query."
        ),
    },
}


def analyse_passed_queries(per_query: list[dict]) -> list[dict]:
    """Characterise passed queries to identify success patterns."""
    passed = []
    for q in per_query:
        if q["above_threshold"]:
            # Classify query type
            query_text = q["query"].lower()
            if any(w in query_text for w in ["συνολικές", "πόσοι", "ποιο είναι"]):
                query_nature = "factual_lookup"
            elif any(w in query_text for w in ["πρόβλεψη", "τάση", "ανάλυση"]):
                query_nature = "analytical_single_topic"
            elif any(w in query_text for w in ["κατανομή", "συγκέντρωση"]):
                query_nature = "distribution_query"
            else:
                query_nature = "domain_specific"

            passed.append({
                "query_index": q["query_index"],
                "query": q["query"],
                "iterations_used": q["iterations_used"],
                "final_score": q["final_score"],
                "query_nature": query_nature,
            })
    return passed


def main() -> None:
    print("=" * 70)
    print("Exp2 Failure Pattern Analysis — Self-Correcting RAG")
    print("=" * 70)

    # Load experiment data
    with open(EXP2_FILE) as f:
        data = json.load(f)

    per_query = data["per_query_results"]
    config = data["config"]

    total = len(per_query)
    passed_queries = [q for q in per_query if q["above_threshold"]]
    failed_queries = [q for q in per_query if not q["above_threshold"]]
    n_passed = len(passed_queries)
    n_failed = len(failed_queries)
    pass_rate = round(n_passed / total * 100, 1)

    print(f"\nTotal queries: {total}")
    print(f"Passed (>={config['threshold']}): {n_passed} ({pass_rate}%)")
    print(f"Failed: {n_failed}")

    # ── Analyse failed queries ──────────────────────────────────────
    failure_analysis = []
    category_counts: dict[str, int] = {cat: 0 for cat in FAILURE_CATEGORIES}

    print(f"\n{'Idx':>3} {'Score':>6} {'Category':>16} Query")
    print("-" * 70)

    for q in failed_queries:
        idx = q["query_index"]
        annotation = FAILURE_ANNOTATIONS[idx]

        # Verify data consistency
        assert annotation["query"] == q["query"], (
            f"Query mismatch at index {idx}"
        )
        assert annotation["final_score"] == q["final_score"], (
            f"Score mismatch at index {idx}: "
            f"{annotation['final_score']} vs {q['final_score']}"
        )

        category = annotation["failure_category"]
        category_counts[category] += 1

        entry = {
            "query_index": idx,
            "query": q["query"],
            "final_score": q["final_score"],
            "iterations_used": q["iterations_used"],
            "query_history": q["query_history"],
            "failure_category": category,
            "explanation": annotation["explanation"],
            "refinement_pattern": annotation["refinement_pattern"],
            "refinement_helped": annotation["refinement_helped"],
            "notes": annotation["notes"],
        }
        failure_analysis.append(entry)

        print(f"  {idx:>3} {q['final_score']:>6.3f} {category:>16} "
              f"{q['query'][:40]}...")

    # ── Category summary ────────────────────────────────────────────
    # Remove categories with 0 count
    category_summary = {
        cat: {
            "count": count,
            "description": FAILURE_CATEGORIES[cat],
        }
        for cat, count in category_counts.items()
        if count > 0
    }

    print(f"\nCategory breakdown:")
    for cat, info in category_summary.items():
        print(f"  {cat}: {info['count']} — {info['description']}")

    # ── Analyse passed queries for contrast ──────────────────────────
    passed_analysis = analyse_passed_queries(per_query)

    # Score distribution
    passed_scores = [q["final_score"] for q in passed_queries]
    failed_scores = [q["final_score"] for q in failed_queries]

    avg_passed = sum(passed_scores) / len(passed_scores) if passed_scores else 0
    avg_failed = sum(failed_scores) / len(failed_scores) if failed_scores else 0

    # Iteration distribution
    single_iter_passed = sum(1 for q in passed_queries if q["iterations_used"] == 1)
    multi_iter_passed = sum(1 for q in passed_queries if q["iterations_used"] > 1)

    print(f"\nScore comparison:")
    print(f"  Passed avg: {avg_passed:.3f} (range: "
          f"{min(passed_scores):.3f}–{max(passed_scores):.3f})")
    print(f"  Failed avg: {avg_failed:.3f} (range: "
          f"{min(failed_scores):.3f}–{max(failed_scores):.3f})")
    print(f"  Gap: {avg_passed - avg_failed:.3f}")

    print(f"\nIteration pattern (passed queries):")
    print(f"  1 iteration: {single_iter_passed}/10 — factual/direct queries")
    print(f"  2+ iterations: {multi_iter_passed}/10 — needed refinement but succeeded")
    print(f"  All 5 failures used exactly 3 iterations (max)")

    # ── Refinement effectiveness ────────────────────────────────────
    any_refinement_helped = any(a["refinement_helped"] for a in failure_analysis)

    print(f"\nRefinement effectiveness for failed queries: "
          f"{'some helped' if any_refinement_helped else 'NONE helped'}")
    print("  -> Query refinement cannot fix fundamental retrieval-vs-reasoning gap")

    # ── Build output ────────────────────────────────────────────────
    output = {
        "description": (
            "Failure pattern analysis for Self-Correcting RAG (Exp2). "
            "Categorises the 5 failed queries and identifies the fundamental "
            "boundary between retrieval and reasoning."
        ),
        "total_queries": total,
        "passed": n_passed,
        "failed": n_failed,
        "pass_rate": pass_rate,
        "threshold": config["threshold"],
        "max_iterations": config["max_iterations"],
        "score_statistics": {
            "passed_avg": round(avg_passed, 4),
            "passed_min": min(passed_scores),
            "passed_max": max(passed_scores),
            "failed_avg": round(avg_failed, 4),
            "failed_min": min(failed_scores),
            "failed_max": max(failed_scores),
            "gap": round(avg_passed - avg_failed, 4),
        },
        "iteration_pattern": {
            "single_iteration_passed": single_iter_passed,
            "multi_iteration_passed": multi_iter_passed,
            "all_failures_used_max_iterations": True,
            "bimodal_note": (
                "8/15 queries pass in 1 iteration, 6/15 exhaust all 3, "
                "only 1 uses exactly 2. This bimodal pattern reflects "
                "a sharp boundary between retrievable and non-retrievable queries."
            ),
        },
        "failure_analysis": failure_analysis,
        "category_summary": category_summary,
        "passed_query_summary": passed_analysis,
        "refinement_effectiveness": {
            "any_failure_benefited": any_refinement_helped,
            "note": (
                "Query refinement did NOT rescue any of the 5 failed queries. "
                "The LLM correctly identified search terms but the fundamental "
                "problem — needing synthesis/computation rather than retrieval "
                "— cannot be solved by better search queries."
            ),
        },
        "insight": (
            "Self-Correcting RAG succeeds on factual retrieval queries "
            "(direct lookup, single-topic analysis) but fails on analytical "
            "queries requiring synthesis across multiple data points. "
            "The bimodal iteration distribution (8 single-pass vs 6 "
            "max-iterations) reflects this fundamental boundary between "
            "retrieval and reasoning. Failed queries fall into three "
            "categories: cross-temporal comparison (3/5), correlation "
            "analysis (1/5), and inferential reasoning (1/5)."
        ),
        "thesis_implications": {
            "chapter_5": (
                "Document failure categories in Section 5.4 results. "
                "The 66.7% pass rate is honest — 5 failures are "
                "structurally non-retrievable, not quality issues."
            ),
            "chapter_6": (
                "Discuss as fundamental RAG limitation: retrieval != reasoning. "
                "Self-Correcting RAG pushes the boundary but cannot cross it. "
                "The correction mechanism helps with query formulation "
                "(rescued 2 marginal queries) but not with missing capabilities."
            ),
            "future_work": (
                "Agentic RAG with multi-step reasoning and intermediate "
                "computation, or knowledge graph for cross-temporal queries. "
                "Also: increase k from 5 to 10+ for temporal coverage."
            ),
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
