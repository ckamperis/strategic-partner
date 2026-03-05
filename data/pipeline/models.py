"""Data pipeline Pydantic models.

Defines the structured data types flowing through the ERP -> Knowledge pipeline:
TransactionType -> MonthlyRecord -> MonthlyData -> BusinessMetrics -> TextChunk -> PipelineResult

References:
    Thesis Chapter 4 — Data Pipeline, Section 4.2
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TransactionType(BaseModel):
    """Classification of an ERP document type.

    Attributes:
        type: Category (sale, payment, credit_note, receipt, reversal, opening, other).
        direction: Cash-flow direction (inflow, outflow, reduction, reversal, none).
        description: Human-readable Greek description.
    """

    type: str
    direction: str
    description: str


class MonthlyRecord(BaseModel):
    """Aggregated financial data for a single month.

    All amounts in EUR, gross (including VAT where applicable).
    """

    year: int
    month: int
    sales_gross: float = 0.0
    sales_net: float = 0.0
    credit_notes: float = 0.0
    payments_out: float = 0.0
    receipts_in: float = 0.0
    reversals: float = 0.0
    transaction_count: int = 0
    unique_customers: int = 0

    @property
    def net_cashflow(self) -> float:
        """Net cashflow = inflows - outflows - credit notes."""
        return (self.sales_gross + self.receipts_in) - self.payments_out - self.credit_notes

    @property
    def period_label(self) -> str:
        """Human-readable period label, e.g. '2023-01'."""
        return f"{self.year}-{self.month:02d}"


class MonthlyData(BaseModel):
    """Collection of monthly records spanning the dataset period."""

    records: list[MonthlyRecord] = Field(default_factory=list)

    @property
    def total_months(self) -> int:
        return len(self.records)

    @property
    def years(self) -> list[int]:
        return sorted({r.year for r in self.records})


class CustomerConcentration(BaseModel):
    """Customer concentration metrics (Herfindahl-style)."""

    top5_pct: float = 0.0
    top10_pct: float = 0.0
    top20_pct: float = 0.0
    total_customers: int = 0


class InvoiceDistribution(BaseModel):
    """Statistical distribution of invoice amounts."""

    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    p95: float = 0.0
    count: int = 0


class BusinessMetrics(BaseModel):
    """Computed business intelligence metrics from ERP data.

    Used by Reasoning and Simulation pillars for analysis.
    """

    seasonal_indices: list[float] = Field(default_factory=list)
    customer_concentration: CustomerConcentration = Field(default_factory=CustomerConcentration)
    invoice_distribution: InvoiceDistribution = Field(default_factory=InvoiceDistribution)
    credit_note_ratio: float = 0.0
    vat_rate: float = 0.0
    collection_rate: float = 0.0
    avg_payment_days: float = 0.0
    total_revenue_gross: float = 0.0
    total_revenue_net: float = 0.0
    date_range_start: str = ""
    date_range_end: str = ""


class TextChunk(BaseModel):
    """A text chunk prepared for embedding and RAG retrieval.

    Attributes:
        text: The chunk content (target ~500 tokens).
        metadata: Structured metadata for filtering/attribution.
        chunk_id: Unique identifier.
        chunk_type: Category (monthly_summary, customer_analysis, seasonal, risk, metrics).
    """

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_id: str = ""
    chunk_type: str = "general"


class PipelineResult(BaseModel):
    """Complete output of the ERP data pipeline.

    Contains both structured data (for Simulation pillar) and
    text chunks (for Knowledge pillar / RAG).
    """

    monthly_data: MonthlyData = Field(default_factory=MonthlyData)
    metrics: BusinessMetrics = Field(default_factory=BusinessMetrics)
    text_chunks: list[TextChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    total_rows_processed: int = 0
