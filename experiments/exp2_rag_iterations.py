"""Experiment 2 — Self-Correcting RAG Iteration Analysis.

Measures how many iterations the Self-Correcting RAG loop needs to
reach the relevance threshold (0.75) for different query types, using
the REAL LLM judge (gpt-4o-mini).

Metrics per query:
    - iterations_used (1–3)
    - per-iteration relevance score
    - per-iteration latency (LLM judge + search)
    - final relevance score
    - query history (original -> refined)

Usage:
    python experiments/exp2_rag_iterations.py

Produces:
    data/results/exp2_rag_iterations.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from config.settings import get_settings
from data.pipeline.transformer import ERPTransformer
from picp.bus import PICPBus
from pillars.knowledge import KnowledgePillar
from pillars.knowledge.hybrid_search import HybridSearcher
from pillars.knowledge.rag import SelfCorrectingRAG
from pillars.knowledge.vector_store import VectorRetriever
from utils.llm import get_llm_client

logger = structlog.get_logger()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# 15 queries of varying difficulty for RAG evaluation
QUERIES = [
    # Easy — should hit on first iteration
    "Ποιες είναι οι συνολικές πωλήσεις;",
    "Ποια είναι η εποχικότητα;",
    "Πόσοι πελάτες υπάρχουν;",
    # Medium — may need 1 refinement
    "Πρόβλεψη ταμειακών ροών για 3 μήνες",
    "Ανάλυση κινδύνου ρευστότητας",
    "Τάση εσόδων και εξόδων",
    "Ποιοι πελάτες αποτελούν τον κύριο κύκλο εργασιών;",
    "Ποιο είναι το ποσοστό πιστωτικών τιμολογίων;",
    # Hard — likely needs multiple refinements
    "Πώς επηρεάζει η εποχικότητα τον κίνδυνο ελλειμμάτων;",
    "Σύγκριση του δεύτερου τριμήνου με το πρώτο σε επίπεδο ταμειακών ροών",
    "Ποια η σχέση μεταξύ αριθμού πελατών και μέσου τιμολογίου;",
    "Τι δείχνει η κατανομή τιμολογίων για τη σταθερότητα εσόδων;",
    "Ποιος μήνας έχει τον μεγαλύτερο κίνδυνο αρνητικής ταμειακής ροής;",
    "Πώς θα μπορούσε να βελτιωθεί η είσπραξη;",
    "Ανάλυση συγκέντρωσης πελατών και κινδύνου εξάρτησης",
]


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    print("=" * 60)
    print("EXPERIMENT 2: Self-Correcting RAG Iteration Analysis")
    print(f"  Queries: {len(QUERIES)}")
    print(f"  Max iterations: 3, Threshold: 0.75")
    print("=" * 60)

    # ── Setup ────────────────────────────────────────────────
    settings = get_settings()
    llm_client = get_llm_client(settings)
    bus = PICPBus(redis=None)

    print("\n[1/3] Ingesting ERP data...")
    transformer = ERPTransformer(recent_years=5)
    pipeline_result = transformer.run_pipeline(
        str(PROJECT_ROOT / "data" / "raw" / "cashflow_dataset.xlsx")
    )
    chunks = pipeline_result.text_chunks

    retriever = VectorRetriever(
        llm_client=llm_client,
        collection_name="exp2_rag_iterations",
        persist_dir=str(PROJECT_ROOT / "data" / "embedded" / "exp2"),
    )
    await retriever.ingest(chunks)

    searcher = HybridSearcher(vector_retriever=retriever)
    searcher.build_bm25_index(chunks)
    print(f"  Ingested {len(chunks)} chunks")

    # ── Run queries ──────────────────────────────────────────
    print("\n[2/3] Running RAG queries with real LLM judge...")
    results: list[dict] = []

    for qi, query in enumerate(QUERIES):
        rag = SelfCorrectingRAG(
            hybrid_searcher=searcher,
            llm_client=llm_client,
            max_iterations=3,
            threshold=0.75,
            k=5,
        )

        q_start = time.perf_counter()
        rag_result = await rag.retrieve(query)
        q_elapsed = (time.perf_counter() - q_start) * 1000

        row = {
            "query_index": qi,
            "query": query,
            "iterations_used": rag_result.iterations_used,
            "final_score": round(rag_result.final_score, 4),
            "query_history": rag_result.query_history,
            "above_threshold": rag_result.final_score >= 0.75,
            "total_latency_ms": round(q_elapsed, 2),
            "timing_ms": round(rag_result.timing_ms, 2),
            "n_chunks_returned": len(rag_result.chunks),
            "warnings": rag_result.warnings or [],
        }
        results.append(row)
        status = "✓" if row["above_threshold"] else "✗"
        print(f"  [{qi+1:2d}/{len(QUERIES)}] {status} iters={row['iterations_used']} "
              f"score={row['final_score']:.3f} {q_elapsed:.0f}ms — {query[:50]}")

    # ── Aggregate ────────────────────────────────────────────
    print("\n[3/3] Computing aggregates...")

    iters_list = [r["iterations_used"] for r in results]
    scores_list = [r["final_score"] for r in results]
    above_list = [r["above_threshold"] for r in results]

    # Distribution of iterations
    iter_dist = {i: iters_list.count(i) for i in [1, 2, 3]}

    total_elapsed = (time.perf_counter() - total_start) * 1000

    output = {
        "experiment": "exp2_rag_iterations",
        "description": "Self-Correcting RAG iteration analysis with real LLM judge",
        "config": {
            "n_queries": len(QUERIES),
            "max_iterations": 3,
            "threshold": 0.75,
            "k": 5,
            "n_chunks": len(chunks),
        },
        "summary": {
            "avg_iterations": round(sum(iters_list) / len(iters_list), 2),
            "avg_final_score": round(sum(scores_list) / len(scores_list), 4),
            "pct_above_threshold": round(sum(above_list) / len(above_list) * 100, 1),
            "iteration_distribution": iter_dist,
            "avg_latency_ms": round(sum(r["total_latency_ms"] for r in results) / len(results), 2),
        },
        "per_query_results": results,
        "total_elapsed_ms": round(total_elapsed, 2),
    }

    out_path = RESULTS_DIR / "exp2_rag_iterations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Avg iterations: {output['summary']['avg_iterations']}")
    print(f"  Avg final score: {output['summary']['avg_final_score']}")
    print(f"  Above threshold: {output['summary']['pct_above_threshold']}%")
    print(f"  Iteration distribution: {iter_dist}")
    print(f"  Total time: {total_elapsed:.0f} ms")
    print(f"  Results: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
