"""Setup — Ingest real ERP data for experiments.

Pre-requisite for all Chapter 5 experiments. Loads the real ERP dataset
through ERPTransformer, ingests text chunks into the Knowledge Pillar
with real OpenAI embeddings, fits distributions, and saves artefacts.

Usage:
    python experiments/setup_data.py

Produces:
    data/results/setup_pipeline_result.json
    data/results/setup_distributions.json
    data/results/setup_metrics.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from config.settings import get_settings
from data.pipeline.models import PipelineResult
from data.pipeline.transformer import ERPTransformer
from picp.bus import PICPBus
from pillars.knowledge import KnowledgePillar
from pillars.simulation.distributions import CashflowDistributions, fit_from_erp_data
from utils.llm import get_llm_client

logger = structlog.get_logger()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "cashflow_dataset.xlsx"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    print("=" * 60)
    print("SETUP: Ingesting Real ERP Data")
    print("=" * 60)

    # ── Step 1: Run ERP pipeline ─────────────────────────────
    print(f"\n[1/4] Loading ERP data from: {DATA_RAW}")
    transformer = ERPTransformer(recent_years=5)
    pipeline_result: PipelineResult = transformer.run_pipeline(str(DATA_RAW))

    print(f"  Rows processed: {pipeline_result.total_rows_processed}")
    print(f"  Monthly records: {pipeline_result.monthly_data.total_months}")
    print(f"  Text chunks: {len(pipeline_result.text_chunks)}")
    print(f"  Warnings: {len(pipeline_result.warnings)}")
    for w in pipeline_result.warnings:
        print(f"    - {w}")

    # ── Step 2: Fit distributions ────────────────────────────
    print("\n[2/4] Fitting distributions from ERP data...")
    distributions = fit_from_erp_data(
        monthly_data=pipeline_result.monthly_data,
        metrics=pipeline_result.metrics,
    )
    print(f"  Revenue mean: €{distributions.revenue_mean:,.2f}")
    print(f"  Revenue std:  €{distributions.revenue_std:,.2f}")
    print(f"  Seasonal factors: {[round(s, 3) for s in distributions.seasonal_factors]}")
    print(f"  Credit note ratio: {distributions.credit_note_ratio:.4f}")

    # ── Step 3: Ingest into Knowledge Pillar ─────────────────
    print("\n[3/4] Ingesting into Knowledge Pillar (real OpenAI embeddings)...")
    settings = get_settings()
    llm_client = get_llm_client(settings)
    bus = PICPBus(redis=None)
    knowledge = KnowledgePillar(
        bus=bus,
        llm_client=llm_client,
        collection_name="erp_knowledge_experiments",
    )

    embed_start = time.perf_counter()
    n_ingested = await knowledge.ingest(pipeline_result.text_chunks)
    embed_elapsed = (time.perf_counter() - embed_start) * 1000
    print(f"  Chunks ingested: {n_ingested}")
    print(f"  Embedding time: {embed_elapsed:.0f} ms")

    # ── Step 4: Save artefacts ───────────────────────────────
    print("\n[4/4] Saving artefacts...")

    # Pipeline result (serialisable subset)
    pipeline_json = {
        "total_rows_processed": pipeline_result.total_rows_processed,
        "processing_time_ms": pipeline_result.processing_time_ms,
        "total_months": pipeline_result.monthly_data.total_months,
        "total_chunks": len(pipeline_result.text_chunks),
        "warnings": pipeline_result.warnings,
        "monthly_records": [
            {
                "year": r.year,
                "month": r.month,
                "sales_gross": r.sales_gross,
                "sales_net": r.sales_net,
                "credit_notes": r.credit_notes,
                "payments_out": r.payments_out,
                "receipts_in": r.receipts_in,
                "net_cashflow": r.net_cashflow,
                "transaction_count": r.transaction_count,
                "unique_customers": r.unique_customers,
            }
            for r in pipeline_result.monthly_data.records
        ],
        "embed_time_ms": round(embed_elapsed, 2),
    }
    _save_json(pipeline_json, RESULTS_DIR / "setup_pipeline_result.json")

    # Distributions
    _save_json(distributions.to_dict(), RESULTS_DIR / "setup_distributions.json")

    # Metrics
    metrics_json = {
        "total_revenue_gross": pipeline_result.metrics.total_revenue_gross,
        "total_revenue_net": pipeline_result.metrics.total_revenue_net,
        "vat_rate": pipeline_result.metrics.vat_rate,
        "collection_rate": pipeline_result.metrics.collection_rate,
        "credit_note_ratio": pipeline_result.metrics.credit_note_ratio,
        "avg_payment_days": pipeline_result.metrics.avg_payment_days,
        "seasonal_indices": pipeline_result.metrics.seasonal_indices,
        "date_range_start": pipeline_result.metrics.date_range_start,
        "date_range_end": pipeline_result.metrics.date_range_end,
        "customer_concentration": {
            "top5_pct": pipeline_result.metrics.customer_concentration.top5_pct,
            "top10_pct": pipeline_result.metrics.customer_concentration.top10_pct,
            "top20_pct": pipeline_result.metrics.customer_concentration.top20_pct,
            "total_customers": pipeline_result.metrics.customer_concentration.total_customers,
        },
        "invoice_distribution": {
            "mean": pipeline_result.metrics.invoice_distribution.mean,
            "median": pipeline_result.metrics.invoice_distribution.median,
            "std": pipeline_result.metrics.invoice_distribution.std,
            "p95": pipeline_result.metrics.invoice_distribution.p95,
            "count": pipeline_result.metrics.invoice_distribution.count,
        },
    }
    _save_json(metrics_json, RESULTS_DIR / "setup_metrics.json")

    total_elapsed = (time.perf_counter() - start) * 1000
    print(f"\nSetup complete in {total_elapsed:.0f} ms")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


def _save_json(data: dict, path: Path) -> None:
    """Write dict as JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {path.name}")


if __name__ == "__main__":
    asyncio.run(main())
