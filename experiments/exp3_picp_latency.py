"""Experiment 3 — PICP Latency Profiling.

Measures wall-clock time for each PICP component across 20 real queries
through the full K -> R -> S -> T pipeline. Identifies bottlenecks.

Metrics per query:
    - Per-pillar latency (knowledge, reasoning, simulation, trust)
    - Total pipeline latency
    - PICP overhead (total - sum of pillars)
    - Vector clock state
    - Event count

Usage:
    python experiments/exp3_picp_latency.py

Produces:
    data/results/exp3_picp_latency.json
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

# 20 diverse queries (mix of all 5 query types)
QUERIES = [
    # Cashflow (triggers simulation)
    "Πρόβλεψη ταμειακών ροών 3 μηνών",
    "Πώς θα εξελιχθεί η ρευστότητα;",
    "Ταμειακή πρόβλεψη για το επόμενο τρίμηνο",
    "Εκτίμηση εισπράξεων και πληρωμών",
    # Risk (triggers simulation)
    "Ποιοι είναι οι βασικοί κίνδυνοι;",
    "Ανάλυση κινδύνου ρευστότητας",
    "Πιθανότητα αρνητικών ταμειακών ροών",
    "Αξιολόγηση κινδύνου ελλειμμάτων",
    # SWOT (no simulation)
    "Κάνε SWOT ανάλυση",
    "Δυνάμεις και αδυναμίες της εταιρείας",
    "Ανάλυση ευκαιριών και απειλών",
    "SWOT για στρατηγικό σχεδιασμό",
    # Customer (no simulation)
    "Ανάλυση πελατολογίου",
    "Ποιοι είναι οι κύριοι πελάτες;",
    "Κατανομή τζίρου ανά πελάτη",
    "Αξιολόγηση πελατειακής βάσης",
    # General (no simulation)
    "Καλημέρα, πώς λειτουργείς;",
    "Τι μπορείς να κάνεις;",
    "Γενική εικόνα εταιρείας",
    "Πόσα δεδομένα έχεις διαθέσιμα;",
]


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    print("=" * 60)
    print("EXPERIMENT 3: PICP Latency Profiling")
    print(f"  Queries: {len(QUERIES)}")
    print("=" * 60)

    # ── Setup ────────────────────────────────────────────────
    settings = get_settings()
    llm_client = get_llm_client(settings)

    print("\n[1/3] Setting up pipeline with real data...")
    transformer = ERPTransformer(recent_years=5)
    pipeline_result = transformer.run_pipeline(
        str(PROJECT_ROOT / "data" / "raw" / "cashflow_dataset.xlsx")
    )
    distributions = fit_from_erp_data(
        pipeline_result.monthly_data,
        pipeline_result.metrics,
    )

    # ── Run queries ──────────────────────────────────────────
    print("\n[2/3] Running queries through full pipeline...")
    results: list[dict] = []

    for qi, query in enumerate(QUERIES):
        # Fresh bus per query to get clean event log
        bus = PICPBus(redis=None)
        audit_dir = str(PROJECT_ROOT / "data" / "results" / "exp3_audit")

        partner = StrategicPartner(
            llm_client=llm_client,
            bus=bus,
            base_distributions=distributions,
            n_simulations=5_000,
            random_seed=42,
            audit_dir=audit_dir,
        )

        # Ingest chunks for knowledge pillar
        await partner.knowledge.ingest(pipeline_result.text_chunks)

        q_start = time.perf_counter()
        response = await partner.query(query)
        q_elapsed = (time.perf_counter() - q_start) * 1000

        # Extract pillar timings
        timings = response.pillar_timings
        pillar_sum = sum(
            v for k, v in timings.items() if k != "total_ms"
        )
        overhead = timings.get("total_ms", 0) - pillar_sum

        # Event count
        events = await bus.get_event_log()

        row = {
            "query_index": qi,
            "query": query,
            "query_type": response.query_type,
            "knowledge_ms": round(timings.get("knowledge", 0), 2),
            "reasoning_ms": round(timings.get("reasoning", 0), 2),
            "simulation_ms": round(timings.get("simulation", 0), 2),
            "trust_ms": round(timings.get("trust", 0), 2),
            "total_ms": round(timings.get("total_ms", 0), 2),
            "picp_overhead_ms": round(overhead, 2),
            "pillar_sum_ms": round(pillar_sum, 2),
            "overhead_pct": round(overhead / timings.get("total_ms", 1) * 100, 1),
            "event_count": len(events),
            "vector_clock": response.vector_clock,
            "trust_score": response.trust_score,
            "simulation_ran": response.simulation_summary is not None,
            "degraded": len(response.degradation_flags) > 0,
        }
        results.append(row)
        sim_marker = "S" if row["simulation_ran"] else "-"
        print(f"  [{qi+1:2d}/{len(QUERIES)}] [{sim_marker}] {row['query_type']:20s} "
              f"K={row['knowledge_ms']:7.1f} R={row['reasoning_ms']:7.1f} "
              f"S={row['simulation_ms']:7.1f} T={row['trust_ms']:7.1f} "
              f"Σ={row['total_ms']:7.1f} ms")

    # ── Aggregate ────────────────────────────────────────────
    print("\n[3/3] Computing aggregates...")

    # Per query type
    type_summary = {}
    for qt in set(r["query_type"] for r in results):
        rows = [r for r in results if r["query_type"] == qt]
        type_summary[qt] = {
            "count": len(rows),
            "avg_knowledge_ms": round(_mean([r["knowledge_ms"] for r in rows]), 2),
            "avg_reasoning_ms": round(_mean([r["reasoning_ms"] for r in rows]), 2),
            "avg_simulation_ms": round(_mean([r["simulation_ms"] for r in rows]), 2),
            "avg_trust_ms": round(_mean([r["trust_ms"] for r in rows]), 2),
            "avg_total_ms": round(_mean([r["total_ms"] for r in rows]), 2),
            "avg_overhead_pct": round(_mean([r["overhead_pct"] for r in rows]), 1),
        }

    # Overall
    overall = {
        "avg_total_ms": round(_mean([r["total_ms"] for r in results]), 2),
        "avg_knowledge_ms": round(_mean([r["knowledge_ms"] for r in results]), 2),
        "avg_reasoning_ms": round(_mean([r["reasoning_ms"] for r in results]), 2),
        "avg_simulation_ms": round(_mean([r["simulation_ms"] for r in results if r["simulation_ran"]]), 2),
        "avg_trust_ms": round(_mean([r["trust_ms"] for r in results]), 2),
        "avg_overhead_pct": round(_mean([r["overhead_pct"] for r in results]), 1),
        "max_total_ms": round(max(r["total_ms"] for r in results), 2),
        "min_total_ms": round(min(r["total_ms"] for r in results), 2),
    }

    # Bottleneck analysis: which pillar dominates?
    bottleneck_counts = {"knowledge": 0, "reasoning": 0, "simulation": 0, "trust": 0}
    for r in results:
        pillars = {
            "knowledge": r["knowledge_ms"],
            "reasoning": r["reasoning_ms"],
            "simulation": r["simulation_ms"],
            "trust": r["trust_ms"],
        }
        bottleneck = max(pillars, key=pillars.get)
        bottleneck_counts[bottleneck] += 1

    total_elapsed = (time.perf_counter() - total_start) * 1000

    output = {
        "experiment": "exp3_picp_latency",
        "description": "PICP latency profiling across 20 real queries",
        "config": {
            "n_queries": len(QUERIES),
            "n_simulations": 5_000,
            "random_seed": 42,
        },
        "overall_summary": overall,
        "by_query_type": type_summary,
        "bottleneck_counts": bottleneck_counts,
        "per_query_results": results,
        "total_elapsed_ms": round(total_elapsed, 2),
    }

    out_path = RESULTS_DIR / "exp3_picp_latency.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Avg total: {overall['avg_total_ms']:.1f} ms")
    print(f"  Bottleneck pillar: {max(bottleneck_counts, key=bottleneck_counts.get)}")
    print(f"  Avg PICP overhead: {overall['avg_overhead_pct']:.1f}%")
    print(f"  Total time: {total_elapsed:.0f} ms")
    print(f"  Results: {out_path}")
    print("=" * 60)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    asyncio.run(main())
