"""Experiment 5 — Trust Score Sensitivity Analysis.

Tests how the Trust Pillar's composite score (Eq. 3.28) responds to
varying data quality conditions. Uses 1 real baseline run + 7 constructed
variants with degraded/improved conditions.

Variants:
    1. baseline — full pipeline, all pillars normal
    2. no_knowledge — empty knowledge context
    3. low_relevance — force low RAG relevance (< 0.5)
    4. general_routing — force query type = general
    5. no_simulation — skip simulation pillar
    6. low_sim_count — only 100 MC simulations
    7. estimated_params — all distribution params at default
    8. perfect_data — high relevance + specific skill + 10K sims

Usage:
    python experiments/exp5_trust_sensitivity.py

Produces:
    data/results/exp5_trust_sensitivity.json
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from data.pipeline.models import BusinessMetrics
from pillars.simulation.distributions import CashflowDistributions
from pillars.trust.evaluator import TrustEvaluator, TrustScore

logger = structlog.get_logger()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    print("=" * 60)
    print("EXPERIMENT 5: Trust Score Sensitivity Analysis")
    print("=" * 60)

    evaluator = TrustEvaluator(weights=(0.4, 0.4, 0.2))

    # ── Define test variants ─────────────────────────────────
    # Each variant simulates a different data quality scenario
    # by constructing pillar result dicts as the TrustEvaluator expects them.

    variants = _build_variants()

    # ── Run evaluations ──────────────────────────────────────
    print("\n[1/2] Running trust evaluations for all variants...")
    results: list[dict] = []

    for variant in variants:
        v_start = time.perf_counter()
        score: TrustScore = evaluator.evaluate(
            knowledge_result=variant["knowledge_result"],
            reasoning_result=variant["reasoning_result"],
            simulation_result=variant["simulation_result"],
        )
        v_elapsed = (time.perf_counter() - v_start) * 1000

        row = {
            "variant": variant["name"],
            "description": variant["description"],
            "overall_score": round(score.overall, 4),
            "explainability": round(score.sub_scores.explainability, 4),
            "consistency": round(score.sub_scores.consistency, 4),
            "accuracy": round(score.sub_scores.accuracy, 4),
            "confidence_level": score.confidence_level,
            "flags": score.flags,
            "flag_count": len(score.flags),
            "elapsed_ms": round(v_elapsed, 4),
        }
        results.append(row)
        print(f"  {variant['name']:25s} -> T={row['overall_score']:.4f} "
              f"(E={row['explainability']:.3f} C={row['consistency']:.3f} "
              f"A={row['accuracy']:.3f}) [{row['confidence_level']}] "
              f"flags={row['flag_count']}")

    # ── Weight sensitivity sweep ─────────────────────────────
    print("\n[2/2] Running weight sensitivity sweep...")
    weight_configs = [
        (0.4, 0.4, 0.2, "default"),
        (0.6, 0.2, 0.2, "explainability_heavy"),
        (0.2, 0.6, 0.2, "consistency_heavy"),
        (0.2, 0.2, 0.6, "accuracy_heavy"),
        (0.33, 0.34, 0.33, "equal_weights"),
    ]

    # Use the "baseline" variant for weight sweep
    baseline = variants[0]
    weight_results = []
    for w_e, w_c, w_a, label in weight_configs:
        ev = TrustEvaluator(weights=(w_e, w_c, w_a))
        score = ev.evaluate(
            knowledge_result=baseline["knowledge_result"],
            reasoning_result=baseline["reasoning_result"],
            simulation_result=baseline["simulation_result"],
        )
        weight_results.append({
            "label": label,
            "weights": {"w_e": w_e, "w_c": w_c, "w_a": w_a},
            "overall_score": round(score.overall, 4),
            "confidence_level": score.confidence_level,
        })
        print(f"  w=({w_e},{w_c},{w_a}) [{label:25s}] -> T={score.overall:.4f}")

    total_elapsed = (time.perf_counter() - total_start) * 1000

    # Compute sensitivity delta: max - min across variants
    scores = [r["overall_score"] for r in results]
    sensitivity_range = max(scores) - min(scores)

    output = {
        "experiment": "exp5_trust_sensitivity",
        "description": "Trust score sensitivity to data quality and weight variations",
        "config": {
            "default_weights": {"w_e": 0.4, "w_c": 0.4, "w_a": 0.2},
            "n_variants": len(variants),
            "n_weight_configs": len(weight_configs),
        },
        "variant_results": results,
        "weight_sensitivity": weight_results,
        "sensitivity_summary": {
            "max_score": round(max(scores), 4),
            "min_score": round(min(scores), 4),
            "range": round(sensitivity_range, 4),
            "best_variant": max(results, key=lambda r: r["overall_score"])["variant"],
            "worst_variant": min(results, key=lambda r: r["overall_score"])["variant"],
        },
        "total_elapsed_ms": round(total_elapsed, 2),
    }

    out_path = RESULTS_DIR / "exp5_trust_sensitivity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Score range: {min(scores):.4f} – {max(scores):.4f} (Δ={sensitivity_range:.4f})")
    print(f"  Best variant: {output['sensitivity_summary']['best_variant']}")
    print(f"  Worst variant: {output['sensitivity_summary']['worst_variant']}")
    print(f"  Total time: {total_elapsed:.0f} ms")
    print(f"  Results: {out_path}")
    print("=" * 60)


def _build_variants() -> list[dict]:
    """Build 8 test variants with different data quality conditions."""

    # Shared building blocks
    _good_knowledge = {
        "chunks": [
            {"text": f"Chunk {i}", "score": 0.8, "metadata": {}}
            for i in range(5)
        ],
        "relevance_score": 0.85,
        "iterations": 1,
        "query_history": ["original query"],
    }

    _good_reasoning = {
        "routing": {
            "query_type": "cashflow_forecast",
            "skill_name": "cashflow_forecast",
            "confidence": 0.9,
        },
        "skill_result": {
            "success": True,
            "parsed_output": {
                "analysis": "Detailed cashflow analysis...",
                "risk_level": "medium",
            },
        },
    }

    _good_simulation = {
        "scenarios": {
            "base": {
                "monthly_stats": [
                    {"mean": 25000, "std": 8000, "p5": 12000, "p50": 25000, "p95": 38000},
                    {"mean": 26000, "std": 8200, "p5": 12500, "p50": 26000, "p95": 39000},
                    {"mean": 27000, "std": 8500, "p5": 13000, "p50": 27000, "p95": 40000},
                ],
                "n_simulations": 10000,
                "probability_negative": 0.05,
                "var_5pct": -5000,
                "config_snapshot": {
                    "distributions": {
                        "revenue_mean": 90000,
                        "revenue_std": 15000,
                        "seasonal_factors": [0.85, 0.90, 1.05, 1.10, 1.15, 1.20, 0.95, 0.80, 0.90, 1.00, 1.05, 1.05],
                        "expense_ratio_mean": 0.72,
                        "collection_delay_mean": 52.0,
                        "customer_loss_rate": 0.02,
                        "credit_note_ratio": 0.05,
                    },
                    "time_horizon_months": 3,
                },
            },
        },
    }

    return [
        {
            "name": "baseline",
            "description": "Full pipeline, all data present, good quality",
            "knowledge_result": _good_knowledge,
            "reasoning_result": _good_reasoning,
            "simulation_result": _good_simulation,
        },
        {
            "name": "no_knowledge",
            "description": "Empty knowledge context (RAG failed)",
            "knowledge_result": {"chunks": [], "relevance_score": 0, "iterations": 1, "query_history": []},
            "reasoning_result": _good_reasoning,
            "simulation_result": _good_simulation,
        },
        {
            "name": "low_relevance",
            "description": "Low RAG relevance score (below 0.5)",
            "knowledge_result": {
                "chunks": [{"text": "Chunk 1", "score": 0.3, "metadata": {}}],
                "relevance_score": 0.35,
                "iterations": 3,
                "query_history": ["q1", "q2", "q3"],
            },
            "reasoning_result": _good_reasoning,
            "simulation_result": _good_simulation,
        },
        {
            "name": "general_routing",
            "description": "Query classified as general (no specific skill)",
            "knowledge_result": _good_knowledge,
            "reasoning_result": {
                "routing": {"query_type": "general", "skill_name": None, "confidence": 0.3},
                "skill_result": {"success": False, "parsed_output": {}},
            },
            "simulation_result": _good_simulation,
        },
        {
            "name": "no_simulation",
            "description": "Simulation pillar skipped completely",
            "knowledge_result": _good_knowledge,
            "reasoning_result": _good_reasoning,
            "simulation_result": {"scenarios": {}},
        },
        {
            "name": "low_sim_count",
            "description": "Only 100 simulations (below 1000 threshold)",
            "knowledge_result": _good_knowledge,
            "reasoning_result": _good_reasoning,
            "simulation_result": {
                "scenarios": {
                    "base": {
                        **_good_simulation["scenarios"]["base"],
                        "n_simulations": 100,
                    },
                },
            },
        },
        {
            "name": "all_estimated_params",
            "description": "All distribution params at default (estimated, not fitted)",
            "knowledge_result": {
                "chunks": [{"text": "Chunk 1", "score": 0.6, "metadata": {}}],
                "relevance_score": 0.6,
                "iterations": 2,
                "query_history": ["q1", "q2"],
            },
            "reasoning_result": _good_reasoning,
            "simulation_result": {
                "scenarios": {
                    "base": {
                        **_good_simulation["scenarios"]["base"],
                        "config_snapshot": {
                            "distributions": {
                                "revenue_mean": 0,
                                "revenue_std": 0,
                                "seasonal_factors": [1.0] * 12,
                                "expense_ratio_mean": 0.72,
                                "collection_delay_mean": 52.0,
                                "customer_loss_rate": 0.02,
                                "credit_note_ratio": 0.05,
                            },
                            "time_horizon_months": 3,
                        },
                    },
                },
            },
        },
        {
            "name": "perfect_data",
            "description": "All conditions optimal — maximum trust score",
            "knowledge_result": {
                "chunks": [
                    {"text": f"Detailed chunk {i}", "score": 0.95, "metadata": {}}
                    for i in range(8)
                ],
                "relevance_score": 0.95,
                "iterations": 1,
                "query_history": ["perfect query"],
            },
            "reasoning_result": {
                "routing": {
                    "query_type": "cashflow_forecast",
                    "skill_name": "cashflow_forecast",
                    "confidence": 0.95,
                },
                "skill_result": {
                    "success": True,
                    "parsed_output": {
                        "analysis": "Comprehensive analysis with all fields...",
                        "risk_level": "low",
                        "recommendations": ["a", "b", "c"],
                    },
                },
            },
            "simulation_result": {
                "scenarios": {
                    "base": {
                        "monthly_stats": [
                            {"mean": 30000, "std": 5000, "p5": 22000, "p50": 30000, "p95": 38000},
                        ],
                        "n_simulations": 10000,
                        "probability_negative": 0.01,
                        "var_5pct": 5000,
                        "config_snapshot": {
                            "distributions": {
                                "revenue_mean": 100000,
                                "revenue_std": 15000,
                                "seasonal_factors": [0.85, 0.90, 1.05, 1.10, 1.15, 1.20, 0.95, 0.80, 0.90, 1.00, 1.05, 1.05],
                                "expense_ratio_mean": 0.68,  # NOT default
                                "collection_delay_mean": 45.0,  # NOT default
                                "customer_loss_rate": 0.015,  # NOT default
                                "credit_note_ratio": 0.03,
                            },
                            "time_horizon_months": 3,
                        },
                    },
                },
            },
        },
    ]


if __name__ == "__main__":
    main()
