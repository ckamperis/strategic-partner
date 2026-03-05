"""One-time ingestion script: ERP Excel -> structured + embedded.

Usage:
    python scripts/ingest_erp_data.py --input data/raw/cashflow_dataset.xlsx

Steps:
1. Load ERP Excel export
2. Classify transactions by DOCCODE prefix
3. Filter and aggregate monthly data
4. Compute business metrics
5. Generate text chunks
6. Embed and store in ChromaDB
7. Save structured outputs to data/processed/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import structlog

from config.settings import get_settings
from data.pipeline.transformer import ERPTransformer
from pillars.knowledge import KnowledgePillar
from picp.bus import PICPBus
from utils.llm import get_llm_client

logger = structlog.get_logger()


async def main(input_path: str, recent_years: int = 5) -> None:
    """Run the full ingestion pipeline."""
    start = time.perf_counter()
    settings = get_settings()

    logger.info("ingestion.start", input=input_path, provider=settings.llm_provider)

    # Step 1-5: Run data pipeline
    transformer = ERPTransformer(recent_years=recent_years)
    result = transformer.run_pipeline(input_path)

    logger.info(
        "ingestion.pipeline_complete",
        months=result.monthly_data.total_months,
        chunks=len(result.text_chunks),
        rows=result.total_rows_processed,
        pipeline_ms=result.processing_time_ms,
    )

    # Save structured outputs
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    monthly_path = processed_dir / "monthly_data.json"
    monthly_path.write_text(
        result.monthly_data.model_dump_json(indent=2),
        encoding="utf-8",
    )

    metrics_path = processed_dir / "business_metrics.json"
    metrics_path.write_text(
        result.metrics.model_dump_json(indent=2),
        encoding="utf-8",
    )

    logger.info("ingestion.saved_structured", monthly=str(monthly_path), metrics=str(metrics_path))

    # Step 6: Embed and store in ChromaDB
    llm_client = get_llm_client(settings)
    bus = PICPBus(redis=None)  # In-memory for ingestion
    pillar = KnowledgePillar(bus=bus, llm_client=llm_client)
    count = await pillar.ingest(result.text_chunks)

    elapsed = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "ingestion.complete",
        chunks_embedded=count,
        total_elapsed_ms=elapsed,
        warnings=result.warnings,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"Ingestion Complete")
    print(f"{'='*60}")
    print(f"  Rows processed:    {result.total_rows_processed:,}")
    print(f"  Months:            {result.monthly_data.total_months}")
    print(f"  Text chunks:       {len(result.text_chunks)}")
    print(f"  Chunks embedded:   {count}")
    print(f"  Pipeline time:     {result.processing_time_ms:.0f}ms")
    print(f"  Total time:        {elapsed:.0f}ms")
    print(f"  Saved to:          {processed_dir}/")
    if result.warnings:
        print(f"\n  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ERP data into the Knowledge Pillar")
    parser.add_argument("--input", required=True, help="Path to ERP Excel export")
    parser.add_argument("--years", type=int, default=5, help="Number of recent years to process")
    args = parser.parse_args()

    asyncio.run(main(args.input, args.years))
