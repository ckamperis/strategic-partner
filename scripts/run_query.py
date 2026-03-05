"""CLI to test the AI Strategic Partner.

Usage:
    python scripts/run_query.py "Πώς θα είναι οι ταμειακές ροές τους επόμενους 3 μήνες;"
    python scripts/run_query.py --mock "Πρόβλεψη ταμειακών ροών"
    python scripts/run_query.py --mock --verbose "SWOT ανάλυση"

The ``--mock`` flag uses MockLLMClient (no API key required).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator import StrategicPartner  # noqa: E402
from picp.bus import PICPBus  # noqa: E402
from pillars.simulation.distributions import CashflowDistributions  # noqa: E402
from utils.llm import MockLLMClient, get_llm_client  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Strategic Partner — Query Interface"
    )
    parser.add_argument("query", help="Business query in Greek or English")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLMClient (no API key needed)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full response JSON",
    )
    args = parser.parse_args()

    # Create LLM client
    if args.mock:
        llm_client = MockLLMClient()
    else:
        llm_client = get_llm_client()

    # Create PICP bus (in-memory for CLI)
    bus = PICPBus(redis=None)

    # Create partner with default distributions
    partner = StrategicPartner(
        llm_client=llm_client,
        bus=bus,
        base_distributions=CashflowDistributions(),
    )

    # Run query
    response = await partner.query(args.query)

    # Pretty print
    print(f"\n{'=' * 60}")
    print(f"Query:      {response.query}")
    print(f"Type:       {response.query_type}")
    print(f"Confidence: {response.confidence}")
    print(f"Trust:      {response.trust_score:.4f}")
    print(f"\nAnswer:\n  {response.answer}")

    if response.simulation_summary:
        print("\nSimulation:")
        for name, metrics in response.simulation_summary.items():
            print(f"  {name}:")
            for k, v in metrics.items():
                print(f"    {k}: {v}")

    if response.caveats:
        print("\nCaveats:")
        for c in response.caveats:
            print(f"  - {c}")

    print(f"\nTimings: {response.pillar_timings}")
    print(f"Vector Clock: {response.vector_clock}")

    if response.degradation_flags:
        print(f"Degradation: {response.degradation_flags}")

    if args.verbose:
        print(f"\nFull Response:")
        print(
            json.dumps(
                response.model_dump(),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
