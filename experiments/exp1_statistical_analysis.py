"""
Exp1 Post-Hoc Statistical Analysis: Bootstrap CIs + Wilcoxon Signed-Rank Tests.

Loads per-query nDCG@5 scores from exp1_hybrid_alpha.json (v2, LLM-as-Judge)
and computes:
  1. Bootstrap 95% confidence intervals for mean nDCG@5 per alpha value
  2. Wilcoxon signed-rank tests (paired, non-parametric) for key comparisons
  3. Cohen's d effect sizes

Methodological justifications:
  - Bootstrap (B=10,000): n=10 queries is too small for parametric CI
    (CLT requires n≥30 for reliable normal approximation). Bootstrap makes
    no distributional assumptions and produces valid CIs for any sample size.
  - Wilcoxon signed-rank: Non-parametric paired test. Does not assume
    normality of differences — appropriate because with n=10 we cannot
    reliably verify normality (Shapiro-Wilk has low power at small n).
  - NOT paired t-test: Cannot verify normality assumption with n=10.
    Shapiro-Wilk is unreliable below n≈20 (high Type II error rate).
  - Cohen's d: Provides standardized effect size independent of sample size,
    allowing comparison even when p-values are inflated by low n.

Thesis reference: Section 5.3 — Hybrid Search Evaluation (Eq. 3.17)
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "data" / "results"
INPUT_FILE = RESULTS_DIR / "exp1_hybrid_alpha.json"
OUTPUT_FILE = RESULTS_DIR / "exp1_statistical_analysis.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95


def load_per_query_ndcg(path: Path) -> dict[float, list[float]]:
    """Load per-query nDCG@5 scores grouped by alpha value.

    Returns:
        Dict mapping alpha (float) to list of nDCG@5 scores (one per query).
    """
    with open(path) as f:
        data = json.load(f)

    grouped: dict[float, list[float]] = defaultdict(list)
    for entry in data["per_query_results"]:
        alpha = round(float(entry["alpha"]), 1)
        grouped[alpha].append(float(entry["ndcg_at_5"]))

    return dict(sorted(grouped.items()))


def bootstrap_ci(
    scores: list[float],
    *,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = CI_LEVEL,
) -> dict:
    """Compute bootstrap confidence interval for the mean.

    Uses the percentile method: resample with replacement B times,
    compute the mean of each resample, take the (α/2, 1-α/2) percentiles
    of the bootstrap distribution.

    Args:
        scores: Observed sample (n=10 per-query nDCG@5 scores).
        b: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        ci_level: Confidence level (0.95 for 95% CI).

    Returns:
        Dict with mean, std, ci_lower, ci_upper, n.
    """
    arr = np.array(scores, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)

    # Generate all bootstrap indices at once: (B, n) matrix
    indices = rng.integers(0, n, size=(b, n))
    boot_means = arr[indices].mean(axis=1)

    alpha_half = (1 - ci_level) / 2
    ci_lower = float(np.percentile(boot_means, alpha_half * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha_half) * 100))

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "n_queries": n,
    }


def cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d for paired samples.

    Uses the pooled standard deviation:
        d = (mean_a - mean_b) / s_pooled
    where s_pooled = sqrt((s_a^2 + s_b^2) / 2).

    Interpretation thresholds (Cohen 1988):
        |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, >0.8: large
    """
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    s_a = float(np.std(arr_a, ddof=1))
    s_b = float(np.std(arr_b, ddof=1))
    s_pooled = math.sqrt((s_a**2 + s_b**2) / 2)
    if s_pooled < 1e-12:
        return 0.0
    return float((np.mean(arr_a) - np.mean(arr_b)) / s_pooled)


def classify_effect(d: float) -> str:
    """Classify effect size magnitude per Cohen (1988)."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def wilcoxon_test(
    scores_a: list[float], scores_b: list[float]
) -> dict:
    """Run Wilcoxon signed-rank test on paired samples.

    Handles the edge case where all differences are zero
    (scores are identical) — returns statistic=0, p=1.0.

    Args:
        scores_a: nDCG@5 scores for condition A (e.g., α=0.3).
        scores_b: nDCG@5 scores for condition B (e.g., α=0.0).

    Returns:
        Dict with test statistic, p-value, and significance flag.
    """
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    diffs = a - b

    # If all differences are zero, Wilcoxon is undefined
    if np.all(np.abs(diffs) < 1e-12):
        return {
            "wilcoxon_statistic": 0.0,
            "p_value": 1.0,
            "note": "All paired differences are zero — no test possible",
        }

    # zero_method="wilcox": discard zero-differences (classic Wilcoxon)
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {
        "wilcoxon_statistic": float(stat),
        "p_value": round(float(p), 6),
    }


def run_comparison(
    label: str,
    description: str,
    scores_a: list[float],
    scores_b: list[float],
    alpha_a: float,
    alpha_b: float,
) -> dict:
    """Run full statistical comparison between two alpha values."""
    wtest = wilcoxon_test(scores_a, scores_b)
    d = cohens_d(scores_a, scores_b)
    p = wtest["p_value"]
    sig = p < 0.05

    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))
    diff = mean_a - mean_b

    if sig:
        direction = "higher" if diff > 0 else "lower"
        interp = (
            f"α={alpha_a} has significantly {direction} nDCG@5 than α={alpha_b} "
            f"(Wilcoxon p={p:.4f}, Cohen's d={d:.3f} [{classify_effect(d)}]). "
            f"Mean difference: {diff:+.4f}."
        )
    else:
        interp = (
            f"No statistically significant difference between α={alpha_a} and α={alpha_b} "
            f"(Wilcoxon p={p:.4f}, Cohen's d={d:.3f} [{classify_effect(d)}]). "
            f"Mean difference: {diff:+.4f}. "
            f"With n=10, statistical power is limited — the effect may exist but is undetectable."
        )

    result = {
        "description": description,
        "alpha_a": alpha_a,
        "alpha_b": alpha_b,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "mean_difference": round(diff, 4),
        **wtest,
        "cohens_d": round(d, 4),
        "effect_size_class": classify_effect(d),
        "significant_at_005": sig,
        "interpretation": interp,
    }
    return result


def main() -> None:
    print("=" * 70)
    print("Exp1 Post-Hoc Statistical Analysis")
    print("Bootstrap CIs + Wilcoxon Signed-Rank Tests")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    alpha_scores = load_per_query_ndcg(INPUT_FILE)
    n_alphas = len(alpha_scores)
    n_queries = len(next(iter(alpha_scores.values())))
    print(f"\nLoaded {n_alphas} alpha values × {n_queries} queries")

    # ------------------------------------------------------------------
    # Part 1: Bootstrap 95% CIs per alpha
    # ------------------------------------------------------------------
    print(f"\n--- Bootstrap 95% CIs (B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED}) ---")
    per_alpha_ci = []
    for alpha, scores in alpha_scores.items():
        ci = bootstrap_ci(scores)
        ci["alpha"] = alpha
        per_alpha_ci.append(ci)
        print(
            f"  α={alpha:.1f}: mean={ci['mean']:.4f} "
            f"[{ci['ci_95_lower']:.4f}, {ci['ci_95_upper']:.4f}] "
            f"(std={ci['std']:.4f}, n={ci['n_queries']})"
        )

    # ------------------------------------------------------------------
    # Part 2: Wilcoxon signed-rank tests + Cohen's d
    # ------------------------------------------------------------------
    print("\n--- Statistical Tests ---")

    scores_03 = alpha_scores[0.3]
    scores_00 = alpha_scores[0.0]
    scores_10 = alpha_scores[1.0]

    test_03_vs_00 = run_comparison(
        label="alpha03_vs_alpha00",
        description="Best hybrid (α=0.3) vs pure BM25 (α=0.0)",
        scores_a=scores_03,
        scores_b=scores_00,
        alpha_a=0.3,
        alpha_b=0.0,
    )
    print(f"\n  α=0.3 vs α=0.0:")
    print(f"    {test_03_vs_00['interpretation']}")

    test_03_vs_10 = run_comparison(
        label="alpha03_vs_alpha10",
        description="Best hybrid (α=0.3) vs pure cosine (α=1.0)",
        scores_a=scores_03,
        scores_b=scores_10,
        alpha_a=0.3,
        alpha_b=1.0,
    )
    print(f"\n  α=0.3 vs α=1.0:")
    print(f"    {test_03_vs_10['interpretation']}")

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    output = {
        "description": "Post-hoc statistical analysis of Exp1 hybrid search α-sweep",
        "method": "Bootstrap resampling (B=10000) + Wilcoxon signed-rank",
        "config": {
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "significance_level": 0.05,
            "n_queries": n_queries,
            "n_alpha_values": n_alphas,
        },
        "per_alpha_ci": per_alpha_ci,
        "statistical_tests": {
            "alpha03_vs_alpha00": test_03_vs_00,
            "alpha03_vs_alpha10": test_03_vs_10,
        },
        "notes": (
            "With n=10 queries, statistical power is limited. CIs are wide but "
            "provide honest uncertainty quantification. Wilcoxon signed-rank is "
            "the appropriate non-parametric paired test when normality cannot be "
            "verified (Shapiro-Wilk is unreliable at n<20). Cohen's d provides "
            "effect size independent of sample size."
        ),
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
