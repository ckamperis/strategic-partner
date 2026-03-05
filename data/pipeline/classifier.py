"""ERP document type classifier.

Maps Greek DOCCODE prefixes to structured TransactionType categories.
The DOCCODE field in the SoftOne ERP export uses Greek abbreviations
that indicate the financial nature of each transaction.

In the dataset, DOCCODEs may appear with or without a leading "00" prefix
and may include trailing numbers/spaces (e.g. "00ΤΠΝ 1", "ΤΠΝ").

References:
    CLAUDE.md — DOC_TYPE_MAP
    Thesis Chapter 4 — Data Pipeline, Classification step
"""

from __future__ import annotations

import re

from data.pipeline.models import TransactionType

# ── Master classification map ───────────────────────────────
# Keys are the canonical Greek prefixes (without "00" or trailing numbers).
DOC_TYPE_MAP: dict[str, TransactionType] = {
    # Inflows (sales / revenue)
    "ΤΠΝ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Πώλησης"),
    "ΤΔΝ": TransactionType(type="sale", direction="inflow", description="Τιμ/γιο Δελτίο Αποστολής"),
    "ΤΙΜ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο"),
    "ΤΠΕ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Πώλησης Ε"),
    "ΤΠΗ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Πώλησης Η"),
    "ΤΠΚ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Πώλησης Κ"),
    "ΤΠΠ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Πώλησης Π"),
    "ΤΠΥ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Παροχής Υπηρεσιών"),
    "ΤΕΝ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Εντός"),
    "ΤΙΠ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Πώλησης"),
    "ΤΕΠ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Εξωτερικού Πώλησης"),
    "ΤΝΕ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Νέο"),
    "ΤΔΕ": TransactionType(type="sale", direction="inflow", description="Τιμ/γιο Δελτίο Ε"),
    "ΤΙΧ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Χρέωσης"),
    "ΤΧΡ": TransactionType(type="sale", direction="inflow", description="Τιμολόγιο Χρέωσης"),

    # Reductions (credit notes)
    "ΠΤΝ": TransactionType(type="credit_note", direction="reduction", description="Πιστωτικό Τιμολόγιο"),
    "ΠΤΕ": TransactionType(type="credit_note", direction="reduction", description="Πιστωτικό"),
    "ΠΤΖ": TransactionType(type="credit_note", direction="reduction", description="Πιστωτικό Ζ"),
    "ΠΤΠ": TransactionType(type="credit_note", direction="reduction", description="Πιστωτικό Τιμ. Πώλησης"),
    "ΠΧΑ": TransactionType(type="credit_note", direction="reduction", description="Πιστωτικό Χρέωσης Α"),

    # Outflows (payments)
    "ΧΠΑ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Α"),
    "ΧΠΒ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Β"),
    "ΧΠΓ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Γ"),
    "ΧΠΔ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Δ"),
    "ΧΠΕ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Ε"),
    "ΧΠΖ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Ζ"),
    "ΧΠΗ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Η"),
    "ΧΠΘ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Θ"),
    "ΧΠΙ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Ι"),
    "ΧΠΚ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Κ"),
    "ΧΠΛ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Λ"),
    "ΧΠΜ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Μ"),
    "ΧΠΝ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Ν"),
    "ΧΠΞ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Ξ"),
    "ΧΠΟ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Ο"),
    "ΧΠΠ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Π"),
    "ΧΠΡ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Ρ"),
    "ΧΠΤ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Τ"),
    "ΧΠΩ": TransactionType(type="payment", direction="outflow", description="Χρηματική Πληρωμή Ω"),
    "ΧΔΝ": TransactionType(type="payment", direction="outflow", description="Χρεωστικό Δελτίο"),
    "ΧΤΑ": TransactionType(type="payment", direction="outflow", description="Χρηματική Ταμείου Α"),
    "ΧΤΕ": TransactionType(type="payment", direction="outflow", description="Χρηματική Ταμείου Ε"),
    "ΧΤΙ": TransactionType(type="payment", direction="outflow", description="Χρηματική Ταμείου Ι"),
    "ΧΤΜ": TransactionType(type="payment", direction="outflow", description="Χρηματική Ταμείου Μ"),
    "ΧΧΛ": TransactionType(type="payment", direction="outflow", description="Χρηματική Χρεωστική Λ"),
    "ΧΚΧ": TransactionType(type="payment", direction="outflow", description="Χρηματική Κατάθεση Χ"),
    "ΧΜΝ": TransactionType(type="payment", direction="outflow", description="Χρηματική Μεταφορά Ν"),
    "ΧΖΖ": TransactionType(type="payment", direction="outflow", description="Χρηματική Ζ"),

    # Cash receipts
    "ΑΤΝ": TransactionType(type="receipt", direction="inflow", description="Απόδειξη Ταμείου Ν"),
    "ΑΤΔ": TransactionType(type="receipt", direction="inflow", description="Απόδειξη Ταμείου Δ"),
    "ΑΤΙ": TransactionType(type="receipt", direction="inflow", description="Απόδειξη Ταμείου Ι"),

    # Reversals
    "ΑΚΠ": TransactionType(type="reversal", direction="reversal", description="Ακύρωση Πληρωμής"),

    # Opening balances
    "ΑΠΧ": TransactionType(type="opening", direction="none", description="Απογραφή"),

    # Other internal / non-cashflow
    "ΔΠΣ": TransactionType(type="other", direction="none", description="Δελτίο Παραγγελίας"),
    "ΠΡΟ": TransactionType(type="other", direction="none", description="Προσφορά"),
    "ΕΜΒ": TransactionType(type="other", direction="none", description="Εμβάσματα"),
    "ΠΑΡ": TransactionType(type="other", direction="none", description="Παραστατικό"),
    "ΤΥΔ": TransactionType(type="other", direction="none", description="Τυποποίηση Δεδομένων"),
    "ΑΠΡ": TransactionType(type="other", direction="none", description="Απόδειξη Παραλαβής"),
    "ΣΥΜ": TransactionType(type="other", direction="none", description="Σύμβαση"),
    "ΜΑΕ": TransactionType(type="other", direction="none", description="Μεταφορά Αποθέματος"),
    "ΑΠΟ": TransactionType(type="other", direction="none", description="Απόδειξη"),
    "ΕΜΑ": TransactionType(type="other", direction="none", description="Εμβάσματα Α"),
    "ΕΜΠ": TransactionType(type="other", direction="none", description="Εμβάσματα Πληρωμής"),
}

# Fallback for unrecognised prefixes
_UNKNOWN_TYPE = TransactionType(type="unknown", direction="none", description="Άγνωστος τύπος")

# Regex to extract the Greek letter prefix from a DOCCODE
_PREFIX_RE = re.compile(r"^(?:00|01)?([Α-ΩA-Z]{2,4})")


def extract_prefix(doccode: str) -> str:
    """Extract the canonical Greek prefix from a DOCCODE string.

    Handles patterns like: "00ΤΠΝ 1", "ΤΠΝ", "00ΑΠΧ0000001", "01ΑΠΠ 3".

    Args:
        doccode: Raw DOCCODE from the ERP export.

    Returns:
        The canonical prefix (e.g. "ΤΠΝ", "ΧΠΑ", "ΑΠΧ").
    """
    code = str(doccode).strip()
    m = _PREFIX_RE.match(code)
    if m:
        return m.group(1)
    return ""


def classify_transaction(doccode: str) -> TransactionType:
    """Classify a DOCCODE into a TransactionType.

    Args:
        doccode: Raw DOCCODE string from the ERP export.

    Returns:
        The matching TransactionType, or UNKNOWN if not recognised.
    """
    prefix = extract_prefix(doccode)
    return DOC_TYPE_MAP.get(prefix, _UNKNOWN_TYPE)
