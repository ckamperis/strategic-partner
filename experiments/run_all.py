"""Run All Experiments — Chapter 5 Experiment Suite.

Runs setup_data.py first, then all 7 experiments sequentially.
Reports total time and cost estimate.

Usage:
    python experiments/run_all.py              # Run all
    python experiments/run_all.py --skip-setup  # Skip data setup
    python experiments/run_all.py --only 1 4 7  # Run specific experiments

Produces:
    data/results/run_all_summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
EXPERIMENTS_DIR = Path(__file__).resolve().parent

# Experiment manifest
EXPERIMENTS = [
    ("setup_data.py", "Setup: Ingest Real ERP Data"),
    ("exp1_hybrid_alpha.py", "Exp 1: Hybrid Search α Sweep"),
    ("exp2_rag_iterations.py", "Exp 2: Self-Correcting RAG Iterations"),
    ("exp3_picp_latency.py", "Exp 3: PICP Latency Profiling"),
    ("exp4_monte_carlo.py", "Exp 4: Monte Carlo Convergence & Backtesting"),
    ("exp5_trust_sensitivity.py", "Exp 5: Trust Score Sensitivity"),
    ("exp6_end_to_end.py", "Exp 6: End-to-End Query Performance"),
    ("exp7_degradation.py", "Exp 7: Graceful Degradation"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 5 experiments")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip setup_data.py (use existing data)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        help="Run only specific experiments (e.g., --only 1 4 7)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    print("=" * 70)
    print("  CHAPTER 5 — EXPERIMENTAL EVALUATION SUITE")
    print("=" * 70)

    # Determine which experiments to run
    experiments_to_run = []
    for idx, (script, description) in enumerate(EXPERIMENTS):
        if idx == 0 and args.skip_setup:
            print(f"  [SKIP] {description}")
            continue
        if idx > 0 and args.only and idx not in args.only:
            print(f"  [SKIP] {description}")
            continue
        experiments_to_run.append((idx, script, description))

    print(f"\n  Running {len(experiments_to_run)} experiments...\n")

    run_results: list[dict] = []

    for run_idx, (exp_idx, script, description) in enumerate(experiments_to_run):
        print(f"\n{'─' * 70}")
        print(f"  [{run_idx+1}/{len(experiments_to_run)}] {description}")
        print(f"{'─' * 70}\n")

        script_path = EXPERIMENTS_DIR / script
        exp_start = time.perf_counter()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=False,
                timeout=600,  # 10 min timeout per experiment
            )
            success = result.returncode == 0
            error = "" if success else f"Exit code: {result.returncode}"
        except subprocess.TimeoutExpired:
            success = False
            error = "Timeout (600s)"
        except Exception as e:
            success = False
            error = str(e)

        exp_elapsed = (time.perf_counter() - exp_start) * 1000

        run_results.append({
            "index": exp_idx,
            "script": script,
            "description": description,
            "success": success,
            "error": error,
            "elapsed_ms": round(exp_elapsed, 2),
        })

        status = "OK" if success else "FAIL"
        print(f"\n  -> [{status}] {description} ({exp_elapsed/1000:.1f}s)")

    # ── Summary ──────────────────────────────────────────────
    total_elapsed = (time.perf_counter() - total_start) * 1000

    n_success = sum(1 for r in run_results if r["success"])
    n_fail = sum(1 for r in run_results if not r["success"])

    summary = {
        "experiment_suite": "chapter5_experimental_evaluation",
        "total_experiments": len(run_results),
        "succeeded": n_success,
        "failed": n_fail,
        "total_elapsed_ms": round(total_elapsed, 2),
        "total_elapsed_min": round(total_elapsed / 60000, 2),
        "per_experiment": run_results,
    }

    # Check which result files exist
    result_files = list(RESULTS_DIR.glob("*.json"))
    summary["result_files"] = [f.name for f in sorted(result_files)]

    out_path = RESULTS_DIR / "run_all_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT SUITE COMPLETE")
    print(f"  Succeeded: {n_success}/{len(run_results)}")
    print(f"  Failed:    {n_fail}/{len(run_results)}")
    print(f"  Total time: {total_elapsed/1000:.1f}s ({total_elapsed/60000:.1f}min)")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Summary: {out_path}")
    if n_fail > 0:
        print(f"\n  FAILURES:")
        for r in run_results:
            if not r["success"]:
                print(f"    - {r['description']}: {r['error']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
