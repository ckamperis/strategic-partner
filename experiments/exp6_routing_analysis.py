"""
Exp6 Routing Analysis — Confusion Matrix + Per-Class Metrics.

Evaluates the heuristic policy's query routing accuracy against
human-annotated ground truth for the 20 queries used in Exp3/Exp6.

Produces:
  - Confusion matrix (5×5)
  - Overall accuracy
  - Per-class precision, recall, F1
  - Misrouting analysis with keyword explanations

Thesis reference: Section 5.6 — End-to-End Evaluation, Routing Accuracy
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXP3_FILE = BASE_DIR / "data" / "results" / "exp3_picp_latency.json"
OUTPUT_FILE = BASE_DIR / "data" / "results" / "exp6_routing_analysis.json"

# ── Class labels (same order as HeuristicPolicy priority) ────────────
CLASSES = [
    "cashflow_forecast",
    "risk_assessment",
    "swot_analysis",
    "customer_analysis",
    "general",
]

# ── Human-annotated ground truth ────────────────────────────────────
# Each entry: (query_text, ground_truth_label, justification)
#
# Justification documents WHY this ground truth was chosen.
# Some queries are genuinely ambiguous — noted explicitly.

GROUND_TRUTH = [
    # -- Cashflow queries (0-3): unambiguous --
    (
        "Πρόβλεψη ταμειακών ροών 3 μηνών",
        "cashflow_forecast",
        "Explicitly asks for cashflow forecast for 3 months."
    ),
    (
        "Πώς θα εξελιχθεί η ρευστότητα;",
        "cashflow_forecast",
        "Asks about liquidity evolution — core cashflow question."
    ),
    (
        "Ταμειακή πρόβλεψη για το επόμενο τρίμηνο",
        "cashflow_forecast",
        "Explicitly asks for cashflow forecast for next quarter."
    ),
    (
        "Εκτίμηση εισπράξεων και πληρωμών",
        "cashflow_forecast",
        "Asks for estimate of receipts and payments — cashflow components."
    ),

    # -- Risk queries (4-7): #5 and #6 are ambiguous --
    (
        "Ποιοι είναι οι βασικοί κίνδυνοι;",
        "risk_assessment",
        "Explicitly asks 'what are the main risks?' — unambiguous risk query."
    ),
    (
        "Ανάλυση κινδύνου ρευστότητας",
        "risk_assessment",
        "AMBIGUOUS: 'κινδύνου' (risk) + 'ρευστότητας' (liquidity). "
        "Intent is risk analysis OF liquidity, not liquidity forecasting. "
        "The primary noun is 'κίνδυνος' (risk); 'ρευστότητα' is the domain."
    ),
    (
        "Πιθανότητα αρνητικών ταμειακών ροών",
        "risk_assessment",
        "AMBIGUOUS: 'probability of negative cashflows'. Asks about probability "
        "of adverse outcome — inherently a risk question. However, the Monte Carlo "
        "cashflow simulation also produces this metric. Classified as risk because "
        "the framing is about evaluating a threat, not generating a forecast."
    ),
    (
        "Αξιολόγηση κινδύνου ελλειμμάτων",
        "risk_assessment",
        "Asks for risk assessment of deficits — unambiguous risk query."
    ),

    # -- SWOT queries (8-11): #10 is ambiguous --
    (
        "Κάνε SWOT ανάλυση",
        "swot_analysis",
        "Explicitly requests SWOT analysis — unambiguous."
    ),
    (
        "Δυνάμεις και αδυναμίες της εταιρείας",
        "swot_analysis",
        "Asks for strengths and weaknesses — S and W from SWOT."
    ),
    (
        "Ανάλυση ευκαιριών και απειλών",
        "swot_analysis",
        "AMBIGUOUS: 'opportunities and threats' are the O and T of SWOT. "
        "However, 'απειλών' (threats) also appears in RISK keywords. "
        "Classified as SWOT because the pair (opportunities + threats) "
        "is the canonical second half of a SWOT analysis."
    ),
    (
        "SWOT για στρατηγικό σχεδιασμό",
        "swot_analysis",
        "Explicitly mentions SWOT for strategic planning — unambiguous."
    ),

    # -- Customer queries (12-15): #15 is ambiguous --
    (
        "Ανάλυση πελατολογίου",
        "customer_analysis",
        "Asks for customer portfolio analysis — unambiguous."
    ),
    (
        "Ποιοι είναι οι κύριοι πελάτες;",
        "customer_analysis",
        "Asks 'who are the main customers?' — unambiguous."
    ),
    (
        "Κατανομή τζίρου ανά πελάτη",
        "customer_analysis",
        "Asks for revenue distribution per customer — unambiguous."
    ),
    (
        "Αξιολόγηση πελατειακής βάσης",
        "customer_analysis",
        "AMBIGUOUS: 'αξιολόγηση' (evaluation) matches SWOT keywords, "
        "but 'πελατειακής βάσης' (customer base) is the subject. "
        "Classified as customer_analysis because the intent is evaluating "
        "the customer base, not performing a strategic SWOT."
    ),

    # -- General queries (16-19): unambiguous --
    (
        "Καλημέρα, πώς λειτουργείς;",
        "general",
        "Greeting / how-do-you-work question — no analytical intent."
    ),
    (
        "Τι μπορείς να κάνεις;",
        "general",
        "Capability question — no analytical intent."
    ),
    (
        "Γενική εικόνα εταιρείας",
        "general",
        "Asks for general company overview — too broad for any specific skill."
    ),
    (
        "Πόσα δεδομένα έχεις διαθέσιμα;",
        "general",
        "Asks about available data — meta-question, no analytical intent."
    ),
]

# ── Keyword explanations for misroutings ────────────────────────────
# Pre-computed from heuristic_policy.py KEYWORD_MAP analysis

MISROUTING_EXPLANATIONS = {
    5: {
        "query": "Ανάλυση κινδύνου ρευστότητας",
        "ground_truth": "risk_assessment",
        "predicted": "cashflow_forecast",
        "keyword_analysis": {
            "cashflow_matches": ["ρευστότητας"],
            "risk_matches": ["κινδύνου"],
            "note": "'ανάλυση' intentionally excluded from all keyword maps (too generic). "
                    "Both CASHFLOW and RISK score 1 match. Tie broken by priority order "
                    "(cashflow=1 > risk=2), so cashflow wins.",
        },
        "root_cause": "priority_tiebreak",
        "fix_suggestion": "Add compound-term detection: 'κίνδυνος ρευστότητας' -> risk."
    },
    6: {
        "query": "Πιθανότητα αρνητικών ταμειακών ροών",
        "ground_truth": "risk_assessment",
        "predicted": "cashflow_forecast",
        "keyword_analysis": {
            "cashflow_matches": ["ροών"],
            "risk_matches": [],
            "note": "'ροών' matches CASHFLOW (1 match). 'ταμειακών' (genitive plural) "
                    "is NOT in CASHFLOW keywords (only ταμειακή/ής/ές). 'πιθανότητα' "
                    "and 'αρνητικών' not in any keyword map. RISK gets 0 matches.",
        },
        "root_cause": "missing_risk_keywords",
        "fix_suggestion": "Add 'πιθανότητα', 'αρνητικός/ή/ό/ών' to RISK keywords."
    },
    10: {
        "query": "Ανάλυση ευκαιριών και απειλών",
        "ground_truth": "swot_analysis",
        "predicted": "risk_assessment",
        "keyword_analysis": {
            "swot_matches": ["ευκαιριών"],
            "risk_matches": ["απειλών"],
            "note": "'ευκαιριών' matches SWOT (1 match). 'απειλών' matches RISK (1 match). "
                    "'ανάλυση' excluded from all maps. Tie broken by priority order "
                    "(risk=2 > swot=3), so risk wins.",
        },
        "root_cause": "priority_tiebreak",
        "fix_suggestion": "Remove 'απειλή/ών' from RISK keywords (it's SWOT terminology), "
                          "or add compound detection for O+T pairs."
    },
    15: {
        "query": "Αξιολόγηση πελατειακής βάσης",
        "ground_truth": "customer_analysis",
        "predicted": "swot_analysis",
        "keyword_analysis": {
            "swot_matches": ["αξιολόγηση"],
            "customer_matches": [],
            "note": "'αξιολόγηση' (evaluation) matches SWOT (1 match). "
                    "'πελατειακής' (adjective: customer-base-related) is NOT in CUSTOMER "
                    "keywords — only noun forms (πελάτης/πελάτη/πελάτες/πελατών/πελατολόγιο). "
                    "CUSTOMER gets 0 matches.",
        },
        "root_cause": "missing_adjective_form",
        "fix_suggestion": "Add 'πελατειακός/ή/ό/ής' adjective forms to CUSTOMER keywords, "
                          "or move 'αξιολόγηση' out of SWOT (it's too generic)."
    },
}


def load_predictions() -> list[dict]:
    """Load actual routing predictions from exp3_picp_latency.json."""
    with open(EXP3_FILE) as f:
        data = json.load(f)
    return data["per_query_results"]


def build_confusion_matrix(
    y_true: list[str], y_pred: list[str]
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix as nested dict: matrix[true_label][pred_label] = count."""
    matrix: dict[str, dict[str, int]] = {}
    for cls in CLASSES:
        matrix[cls] = {c: 0 for c in CLASSES}

    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1

    return matrix


def compute_per_class_metrics(
    matrix: dict[str, dict[str, int]]
) -> list[dict]:
    """Compute precision, recall, F1 per class from confusion matrix."""
    metrics = []

    for cls in CLASSES:
        tp = matrix[cls][cls]
        fp = sum(matrix[other][cls] for other in CLASSES if other != cls)
        fn = sum(matrix[cls][other] for other in CLASSES if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        support = sum(matrix[cls].values())

        metrics.append({
            "class": cls,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        })

    return metrics


def main() -> None:
    print("=" * 70)
    print("Exp6 Routing Analysis — Confusion Matrix + Per-Class Metrics")
    print("=" * 70)

    # Load actual predictions
    predictions = load_predictions()

    # Validate alignment
    assert len(predictions) == len(GROUND_TRUTH), (
        f"Expected {len(GROUND_TRUTH)} queries, got {len(predictions)}"
    )

    # Verify query text alignment
    for i, (gt_query, gt_label, _) in enumerate(GROUND_TRUTH):
        pred_query = predictions[i]["query"]
        assert gt_query == pred_query, (
            f"Query {i} mismatch: GT='{gt_query}' vs Pred='{pred_query}'"
        )

    # Extract labels
    y_true = [gt[1] for gt in GROUND_TRUTH]
    y_pred = [p["query_type"] for p in predictions]

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
    print(f"\nAccuracy: {correct}/{len(y_true)} = {accuracy:.1%}")

    # Confusion matrix
    matrix = build_confusion_matrix(y_true, y_pred)

    print(f"\nConfusion Matrix (rows=ground truth, cols=predicted):")
    header = f"{'':>22}" + "".join(f"{c[:8]:>10}" for c in CLASSES)
    print(header)
    print("-" * len(header))
    for true_cls in CLASSES:
        row = f"{true_cls:>22}"
        for pred_cls in CLASSES:
            count = matrix[true_cls][pred_cls]
            marker = " *" if true_cls != pred_cls and count > 0 else "  "
            row += f"{count:>8}{marker}"
        print(row)

    # Per-class metrics
    per_class = compute_per_class_metrics(matrix)

    print(f"\n{'Class':>22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print("-" * 55)
    for m in per_class:
        print(f"{m['class']:>22} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>8}")

    # Macro and weighted averages
    macro_precision = sum(m["precision"] for m in per_class) / len(per_class)
    macro_recall = sum(m["recall"] for m in per_class) / len(per_class)
    macro_f1 = sum(m["f1"] for m in per_class) / len(per_class)

    total_support = sum(m["support"] for m in per_class)
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class) / total_support

    print("-" * 55)
    print(f"{'macro avg':>22} {macro_precision:>6.3f} {macro_recall:>6.3f} "
          f"{macro_f1:>6.3f} {total_support:>8}")
    print(f"{'weighted avg':>22} {'':>6} {'':>6} "
          f"{weighted_f1:>6.3f} {total_support:>8}")

    # Misrouted queries
    misrouted = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            gt_query, gt_label, gt_justification = GROUND_TRUTH[i]
            entry = {
                "query_index": i,
                "query": gt_query,
                "ground_truth": t,
                "predicted": p,
                "justification": gt_justification,
            }
            if i in MISROUTING_EXPLANATIONS:
                entry["keyword_analysis"] = MISROUTING_EXPLANATIONS[i]["keyword_analysis"]
                entry["root_cause"] = MISROUTING_EXPLANATIONS[i]["root_cause"]
                entry["fix_suggestion"] = MISROUTING_EXPLANATIONS[i]["fix_suggestion"]
            misrouted.append(entry)

    print(f"\nMisrouted queries: {len(misrouted)}")
    for m in misrouted:
        print(f"  #{m['query_index']}: '{m['query']}'")
        print(f"    GT={m['ground_truth']}, Pred={m['predicted']}")
        print(f"    Cause: {m.get('root_cause', 'unknown')}")

    # Root cause summary
    root_causes = defaultdict(int)
    for m in misrouted:
        root_causes[m.get("root_cause", "unknown")] += 1

    print(f"\nRoot cause summary:")
    for cause, count in sorted(root_causes.items(), key=lambda x: -x[1]):
        print(f"  {cause}: {count}")

    # Count ambiguous ground truth annotations
    ambiguous = sum(1 for _, _, j in GROUND_TRUTH if "AMBIGUOUS" in j)
    print(f"\nAmbiguous queries: {ambiguous}/20")

    # ── Build output ────────────────────────────────────────────────
    # Flatten confusion matrix to 2D list for JSON
    matrix_list = [[matrix[t][p] for p in CLASSES] for t in CLASSES]

    output = {
        "description": "Exp6 routing analysis — confusion matrix and per-class metrics",
        "n_queries": len(y_true),
        "n_classes": len(CLASSES),
        "class_labels": CLASSES,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "incorrect": len(y_true) - correct,
        "confusion_matrix": matrix_list,
        "confusion_matrix_labels": "rows=ground_truth, cols=predicted",
        "per_class_metrics": per_class,
        "macro_avg": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "weighted_avg_f1": round(weighted_f1, 4),
        "misrouted_queries": misrouted,
        "root_cause_summary": dict(root_causes),
        "ambiguous_queries": ambiguous,
        "ground_truth_annotations": [
            {
                "index": i,
                "query": gt[0],
                "ground_truth": gt[1],
                "justification": gt[2],
            }
            for i, gt in enumerate(GROUND_TRUTH)
        ],
        "notes": (
            "Ground truth annotated by human judgment. 4/20 queries are genuinely "
            "ambiguous (marked with AMBIGUOUS in justification). Misroutings stem from "
            "two root causes: (1) priority tie-breaking when keywords match multiple "
            "categories equally, and (2) missing keyword forms (adjective forms, "
            "probability-related terms). Both are addressable without architectural changes."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
