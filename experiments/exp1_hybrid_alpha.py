"""Experiment 1 — Hybrid Search α Parameter Sweep with LLM-as-Judge.

Uses LLM-as-Judge (gpt-4o-mini) to grade each retrieved chunk as
0/1/2 (irrelevant/partial/relevant). Then computes proper IR metrics:
Precision@3, Precision@5, nDCG@5, MRR — independent of score magnitudes.

Usage:
    python experiments/exp1_hybrid_alpha.py

Produces:
    data/results/exp1_hybrid_alpha.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import structlog

from config.settings import get_settings
from data.pipeline.transformer import ERPTransformer
from pillars.knowledge.hybrid_search import HybridSearcher
from pillars.knowledge.vector_store import VectorRetriever
from utils.llm import get_llm_client

logger = structlog.get_logger()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# 10 diverse Greek business queries for retrieval evaluation
QUERIES = [
    "Πρόβλεψη ταμειακών ροών για τους επόμενους 3 μήνες",
    "Ποια είναι η εποχικότητα στις πωλήσεις;",
    "Ποιοι είναι οι top πελάτες σε τζίρο;",
    "Πώς κατανέμονται τα πιστωτικά τιμολόγια;",
    "Ποιος είναι ο μέσος χρόνος είσπραξης;",
    "Ποια είναι η τάση εσόδων ανά μήνα;",
    "Ανάλυση κινδύνου ταμειακών ελλειμμάτων",
    "Σύγκριση πωλήσεων πρώτου και δεύτερου εξαμήνου",
    "Ποιο είναι το ποσοστό ΦΠΑ στα τιμολόγια;",
    "Συνολική εικόνα οικονομικής υγείας της εταιρείας",
]

# 11 α values from 0.0 (pure BM25) to 1.0 (pure cosine)
ALPHA_VALUES = [round(i * 0.1, 1) for i in range(11)]

RELEVANCE_PROMPT = """You are evaluating whether a retrieved text chunk is relevant
to a business query. The chunks come from a Greek ERP dataset and contain monthly
financial summaries in Greek.

Query: {query}

Retrieved chunk:
---
{chunk_text}
---

Is this chunk relevant to answering the query?
Rate: 2 = highly relevant (directly answers the query),
      1 = partially relevant (contains useful context),
      0 = not relevant (unrelated or noise)

Respond with ONLY the number (0, 1, or 2)."""


def ndcg_at_k(relevance_scores: list[int], k: int = 5) -> float:
    """Compute nDCG@k from graded relevance scores (0, 1, 2)."""
    relevance = np.array(relevance_scores[:k], dtype=float)
    if len(relevance) == 0:
        return 0.0
    # Pad if fewer than k results
    if len(relevance) < k:
        relevance = np.pad(relevance, (0, k - len(relevance)))
    # DCG
    discounts = np.log2(np.arange(2, k + 2))  # log2(2), log2(3), ..., log2(k+1)
    dcg = np.sum(relevance / discounts)
    # Ideal DCG (sort scores descending)
    ideal = np.sort(relevance)[::-1]
    idcg = np.sum(ideal / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def precision_at_k(relevance_scores: list[int], k: int) -> float:
    """Compute Precision@k. Relevant = score >= 1."""
    top_k = relevance_scores[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for s in top_k if s >= 1)
    return relevant / k


def mrr(relevance_scores: list[int]) -> float:
    """Mean Reciprocal Rank. First relevant = score >= 1."""
    for i, s in enumerate(relevance_scores):
        if s >= 1:
            return 1.0 / (i + 1)
    return 0.0


async def judge_relevance(
    llm_client,
    query: str,
    chunk_text: str,
) -> int:
    """Ask LLM to judge chunk relevance (0/1/2)."""
    prompt = RELEVANCE_PROMPT.format(query=query, chunk_text=chunk_text)
    response = await llm_client.complete(
        prompt=prompt,
        temperature=0.0,
    )
    response = response.strip()
    # Parse the integer
    for char in response:
        if char in ("0", "1", "2"):
            return int(char)
    # Default to 0 if parsing fails
    logger.warning("judge_relevance.parse_failed", response=response)
    return 0


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    print("=" * 60)
    print("EXPERIMENT 1 v2: Hybrid Search α Sweep — LLM-as-Judge")
    print(f"  α values: {ALPHA_VALUES}")
    print(f"  Queries: {len(QUERIES)}")
    print(f"  Methodology: gpt-4o-mini relevance grading (0/1/2)")
    print(f"  Metrics: P@3, P@5, nDCG@5, MRR")
    print("=" * 60)

    # ── Setup: Ingest data ───────────────────────────────────
    settings = get_settings()
    llm_client = get_llm_client(settings)

    print("\n[1/4] Loading and ingesting ERP data...")
    transformer = ERPTransformer(recent_years=5)
    pipeline_result = transformer.run_pipeline(
        str(PROJECT_ROOT / "data" / "raw" / "cashflow_dataset.xlsx")
    )
    chunks = pipeline_result.text_chunks
    print(f"  Chunks: {len(chunks)}")

    # Ingest once with real embeddings
    retriever = VectorRetriever(
        llm_client=llm_client,
        collection_name="exp1_alpha_sweep_v2",
        persist_dir=str(PROJECT_ROOT / "data" / "embedded" / "exp1_v2"),
    )
    n_ingested = await retriever.ingest(chunks)
    print(f"  Ingested: {n_ingested} chunks with real embeddings")

    # ── Run sweep & collect (query, chunk) pairs ────────────
    print("\n[2/4] Running α sweep and collecting unique (query, chunk) pairs...")

    # Store search results per (alpha, query)
    all_search_results: dict[tuple[float, int], list] = {}

    for alpha in ALPHA_VALUES:
        searcher = HybridSearcher(vector_retriever=retriever, alpha=alpha)
        searcher.build_bm25_index(chunks)

        for qi, query in enumerate(QUERIES):
            q_start = time.perf_counter()
            search_results = await searcher.search(query, k=5, alpha=alpha)
            q_elapsed = (time.perf_counter() - q_start) * 1000
            all_search_results[(alpha, qi)] = search_results

    print(f"  Total search executions: {len(all_search_results)}")

    # ── Collect unique (query_index, chunk_id) pairs ────────
    unique_pairs: dict[tuple[int, str], str] = {}  # (qi, chunk_id) -> chunk_text
    for (alpha, qi), results in all_search_results.items():
        for r in results:
            key = (qi, r.chunk_id)
            if key not in unique_pairs:
                unique_pairs[key] = r.chunk_text

    print(f"  Unique (query, chunk) pairs to judge: {len(unique_pairs)}")

    # ── LLM-as-Judge: grade each unique pair ────────────────
    print("\n[3/4] Running LLM-as-Judge relevance grading...")
    judgment_cache: dict[tuple[int, str], int] = {}
    judge_count = 0

    for (qi, chunk_id), chunk_text in unique_pairs.items():
        score = await judge_relevance(llm_client, QUERIES[qi], chunk_text)
        judgment_cache[(qi, chunk_id)] = score
        judge_count += 1
        if judge_count % 50 == 0:
            print(f"  Judged {judge_count}/{len(unique_pairs)} pairs...")

    print(f"  Total LLM judgments: {judge_count}")

    # ── Compute IR metrics per (α, query) ───────────────────
    print("\n[4/4] Computing IR metrics...")
    per_query_results: list[dict] = []

    for alpha in ALPHA_VALUES:
        for qi, query in enumerate(QUERIES):
            results = all_search_results[(alpha, qi)]
            # Look up relevance grades for each retrieved chunk
            relevance_scores = []
            for r in results:
                grade = judgment_cache.get((qi, r.chunk_id), 0)
                relevance_scores.append(grade)

            row = {
                "alpha": alpha,
                "query_index": qi,
                "query": query,
                "precision_at_3": round(precision_at_k(relevance_scores, 3), 4),
                "precision_at_5": round(precision_at_k(relevance_scores, 5), 4),
                "ndcg_at_5": round(ndcg_at_k(relevance_scores, 5), 4),
                "mrr": round(mrr(relevance_scores), 4),
                "relevance_grades": relevance_scores,
                "n_results": len(results),
            }
            per_query_results.append(row)

    # ── Aggregate per α ─────────────────────────────────────
    alpha_summary = []
    for alpha in ALPHA_VALUES:
        rows = [r for r in per_query_results if r["alpha"] == alpha]
        alpha_summary.append({
            "alpha": alpha,
            "avg_precision_at_3": round(_mean([r["precision_at_3"] for r in rows]), 4),
            "avg_precision_at_5": round(_mean([r["precision_at_5"] for r in rows]), 4),
            "avg_ndcg_at_5": round(_mean([r["ndcg_at_5"] for r in rows]), 4),
            "avg_mrr": round(_mean([r["mrr"] for r in rows]), 4),
        })

    # Find optimal α by nDCG@5
    best = max(alpha_summary, key=lambda x: x["avg_ndcg_at_5"])

    total_elapsed = (time.perf_counter() - total_start) * 1000

    output = {
        "experiment": "exp1_hybrid_alpha",
        "description": "Hybrid search α parameter sweep with LLM-as-Judge relevance grading",
        "methodology": "LLM-as-judge relevance (gpt-4o-mini, graded 0/1/2). "
                        "Metrics: Precision@3, Precision@5, nDCG@5, MRR.",
        "correction_note": "v1 compared fused score magnitudes which artificially favoured "
                           "high α. v2 uses external relevance judgments independent of score scale.",
        "config": {
            "alpha_values": ALPHA_VALUES,
            "n_queries": len(QUERIES),
            "n_chunks": len(chunks),
            "k": 5,
            "judge_model": "gpt-4o-mini",
            "temperature": 0.0,
        },
        "alpha_summary": alpha_summary,
        "best_alpha_by_ndcg": best["alpha"],
        "best_avg_ndcg": best["avg_ndcg_at_5"],
        "judgment_cache_size": len(judgment_cache),
        "total_llm_calls": judge_count,
        "per_query_results": per_query_results,
        "total_elapsed_ms": round(total_elapsed, 2),
    }

    out_path = RESULTS_DIR / "exp1_hybrid_alpha.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Best α (by nDCG@5): {best['alpha']} (nDCG@5={best['avg_ndcg_at_5']:.4f})")
    print(f"  LLM judgments: {judge_count}")
    print(f"  Total time: {total_elapsed:.0f} ms")
    print(f"  Results: {out_path}")
    print("=" * 60)

    # Print summary table
    print("\n  α    | P@3   | P@5   | nDCG@5 | MRR")
    print("  " + "-" * 42)
    for row in alpha_summary:
        marker = " ← best" if row["alpha"] == best["alpha"] else ""
        print(f"  {row['alpha']:.1f}  | {row['avg_precision_at_3']:.3f} | "
              f"{row['avg_precision_at_5']:.3f} | {row['avg_ndcg_at_5']:.4f} | "
              f"{row['avg_mrr']:.3f}{marker}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    asyncio.run(main())
