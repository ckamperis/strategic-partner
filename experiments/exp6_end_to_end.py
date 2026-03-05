"""Experiment 6 — End-to-End Query Performance.

Runs 20 diverse Greek business queries through the COMPLETE
K -> R -> S -> T pipeline with REAL OpenAI API calls and measures
per-query quality and performance metrics.

Metrics per query:
    - query_type, trust_score, confidence
    - per-pillar timing
    - simulation_ran, answer length
    - degradation_flags

Usage:
    python experiments/exp6_end_to_end.py

Produces:
    data/results/exp6_end_to_end.json
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
from orchestrator import StrategicPartner
from picp.bus import PICPBus
from pillars.simulation.distributions import fit_from_erp_data
from utils.llm import get_llm_client

logger = structlog.get_logger()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# 20 diverse queries (4 per query type)
QUERIES = [
    # Cashflow Forecast (4)
    ("Πρόβλεψη ταμειακών ροών 3 μηνών", "cashflow_forecast"),
    ("Πώς θα εξελιχθεί η ρευστότητα;", "cashflow_forecast"),
    ("Ταμειακή πρόβλεψη για το τρίμηνο", "cashflow_forecast"),
    ("Εκτίμηση εισπράξεων και πληρωμών", "cashflow_forecast"),
    # Risk Assessment (4)
    ("Ποιοι είναι οι βασικοί κίνδυνοι;", "risk_assessment"),
    ("Ανάλυση κινδύνου ρευστότητας", "risk_assessment"),
    ("Πιθανότητα αρνητικών ταμειακών ροών", "risk_assessment"),
    ("Αξιολόγηση κινδύνου ελλειμμάτων", "risk_assessment"),
    # SWOT Analysis (4)
    ("Κάνε SWOT ανάλυση", "swot_analysis"),
    ("Δυνάμεις και αδυναμίες εταιρείας", "swot_analysis"),
    ("Ανάλυση ευκαιριών και απειλών", "swot_analysis"),
    ("SWOT στρατηγική ανάλυση", "swot_analysis"),
    # Customer Analysis (4)
    ("Ανάλυση πελατολογίου", "customer_analysis"),
    ("Ποιοι είναι οι κύριοι πελάτες;", "customer_analysis"),
    ("Κατανομή τζίρου ανά πελάτη", "customer_analysis"),
    ("Αξιολόγηση πελατειακής βάσης", "customer_analysis"),
    # General (4)
    ("Καλημέρα, πώς λειτουργείς;", "general"),
    ("Τι μπορείς να κάνεις;", "general"),
    ("Γενική εικόνα εταιρείας", "general"),
    ("Πόσα δεδομένα έχεις;", "general"),
]


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    print("=" * 60)
    print("EXPERIMENT 6: End-to-End Query Performance")
    print(f"  Queries: {len(QUERIES)} (4 per type × 5 types)")
    print("=" * 60)

    # ── Setup ────────────────────────────────────────────────
    settings = get_settings()
    llm_client = get_llm_client(settings)

    print("\n[1/3] Loading real ERP data...")
    transformer = ERPTransformer(recent_years=5)
    pipeline_result = transformer.run_pipeline(
        str(PROJECT_ROOT / "data" / "raw" / "cashflow_dataset.xlsx")
    )
    distributions = fit_from_erp_data(
        pipeline_result.monthly_data,
        pipeline_result.metrics,
    )
    print(f"  Monthly records: {pipeline_result.monthly_data.total_months}")
    print(f"  Text chunks: {len(pipeline_result.text_chunks)}")

    # ── Run queries ──────────────────────────────────────────
    print("\n[2/3] Running end-to-end queries...")
    results: list[dict] = []

    for qi, (query, expected_type) in enumerate(QUERIES):
        # Fresh bus per query
        bus = PICPBus(redis=None)
        audit_dir = str(PROJECT_ROOT / "data" / "results" / "exp6_audit")

        partner = StrategicPartner(
            llm_client=llm_client,
            bus=bus,
            base_distributions=distributions,
            n_simulations=5_000,
            random_seed=42,
            audit_dir=audit_dir,
        )
        await partner.knowledge.ingest(pipeline_result.text_chunks)

        q_start = time.perf_counter()
        response = await partner.query(query)
        q_elapsed = (time.perf_counter() - q_start) * 1000

        type_correct = response.query_type == expected_type

        row = {
            "query_index": qi,
            "query": query,
            "expected_type": expected_type,
            "actual_type": response.query_type,
            "type_correct": type_correct,
            "trust_score": response.trust_score,
            "confidence": response.confidence,
            "answer_length": len(response.answer),
            "simulation_ran": response.simulation_summary is not None,
            "n_factors": len(response.factors),
            "n_caveats": len(response.caveats),
            "degradation_flags": response.degradation_flags,
            "degraded": len(response.degradation_flags) > 0,
            "pillar_timings": response.pillar_timings,
            "total_ms": round(response.pillar_timings.get("total_ms", q_elapsed), 2),
            "vector_clock": response.vector_clock,
        }
        results.append(row)

        marker = "✓" if type_correct else "✗"
        sim = "S" if row["simulation_ran"] else "-"
        print(f"  [{qi+1:2d}/{len(QUERIES)}] {marker}[{sim}] {row['actual_type']:20s} "
              f"T={row['trust_score']:.3f} [{row['confidence']:6s}] "
              f"{row['total_ms']:7.0f}ms — {query[:45]}")

    # ── Aggregate ────────────────────────────────────────────
    print("\n[3/3] Computing aggregates...")

    # Routing accuracy
    total_correct = sum(1 for r in results if r["type_correct"])
    routing_accuracy = total_correct / len(results)

    # Per-type summaries
    type_summary = {}
    for qt in ["cashflow_forecast", "risk_assessment", "swot_analysis", "customer_analysis", "general"]:
        rows = [r for r in results if r["expected_type"] == qt]
        type_summary[qt] = {
            "count": len(rows),
            "routing_accuracy": round(sum(1 for r in rows if r["type_correct"]) / len(rows), 2) if rows else 0,
            "avg_trust_score": round(_mean([r["trust_score"] for r in rows]), 4),
            "avg_total_ms": round(_mean([r["total_ms"] for r in rows]), 2),
            "avg_answer_length": round(_mean([r["answer_length"] for r in rows]), 0),
            "degraded_count": sum(1 for r in rows if r["degraded"]),
        }

    # Overall
    overall = {
        "routing_accuracy": round(routing_accuracy, 4),
        "avg_trust_score": round(_mean([r["trust_score"] for r in results]), 4),
        "avg_total_ms": round(_mean([r["total_ms"] for r in results]), 2),
        "median_total_ms": round(sorted([r["total_ms"] for r in results])[len(results) // 2], 2),
        "max_total_ms": round(max(r["total_ms"] for r in results), 2),
        "avg_answer_length": round(_mean([r["answer_length"] for r in results]), 0),
        "degraded_count": sum(1 for r in results if r["degraded"]),
        "confidence_distribution": {
            "high": sum(1 for r in results if r["confidence"] == "high"),
            "medium": sum(1 for r in results if r["confidence"] == "medium"),
            "low": sum(1 for r in results if r["confidence"] == "low"),
        },
    }

    total_elapsed = (time.perf_counter() - total_start) * 1000

    output = {
        "experiment": "exp6_end_to_end",
        "description": "End-to-end query performance with 20 real queries",
        "config": {
            "n_queries": len(QUERIES),
            "n_simulations": 5_000,
            "random_seed": 42,
        },
        "overall_summary": overall,
        "by_query_type": type_summary,
        "per_query_results": results,
        "total_elapsed_ms": round(total_elapsed, 2),
    }

    out_path = RESULTS_DIR / "exp6_end_to_end.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Routing accuracy: {routing_accuracy:.0%}")
    print(f"  Avg trust score: {overall['avg_trust_score']:.4f}")
    print(f"  Avg latency: {overall['avg_total_ms']:.0f} ms")
    print(f"  Confidence: H={overall['confidence_distribution']['high']} "
          f"M={overall['confidence_distribution']['medium']} "
          f"L={overall['confidence_distribution']['low']}")
    print(f"  Degraded: {overall['degraded_count']}/{len(results)}")
    print(f"  Total time: {total_elapsed:.0f} ms")
    print(f"  Results: {out_path}")
    print("=" * 60)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    asyncio.run(main())
