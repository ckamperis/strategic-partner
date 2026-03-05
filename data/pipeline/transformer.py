"""ERP data transformer — Excel -> structured + text chunks.

Loads the SoftOne ERP export, classifies transactions, computes
monthly aggregates and business metrics, and generates text chunks
suitable for embedding and RAG retrieval.

References:
    Thesis Chapter 4 — Data Pipeline
    CLAUDE.md — ERP Dataset Structure, DOC_TYPE_MAP
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from data.pipeline.classifier import classify_transaction, extract_prefix
from data.pipeline.models import (
    BusinessMetrics,
    CustomerConcentration,
    InvoiceDistribution,
    MonthlyData,
    MonthlyRecord,
    PipelineResult,
    TextChunk,
)

logger = structlog.get_logger()


class ERPTransformer:
    """Transforms raw ERP Excel data into structured analytics and text chunks.

    Pipeline steps:
    1. load_excel — read and validate columns
    2. classify — add doc_type / direction columns
    3. filter_opening_balances — remove 00ΑΠΧ entries
    4. aggregate_monthly — group by year-month, sum by direction
    5. compute_metrics — seasonal indices, concentration, distributions
    6. generate_text_chunks — narrative chunks for RAG

    Args:
        recent_years: Only process data from these years (default: last 5).
    """

    def __init__(self, recent_years: int = 5) -> None:
        self._recent_years = recent_years

    def run_pipeline(self, excel_path: str | Path) -> PipelineResult:
        """Execute the full transformation pipeline.

        Args:
            excel_path: Path to the SoftOne ERP Excel export.

        Returns:
            PipelineResult with structured data, metrics, and text chunks.
        """
        start = time.perf_counter()
        warnings: list[str] = []

        # Step 1: Load
        df = self.load_excel(excel_path)
        total_rows = len(df)
        logger.info("pipeline.loaded", rows=total_rows, columns=len(df.columns))

        # Step 2: Classify
        df = self.classify(df)

        # Step 3: Filter opening balances and non-cashflow types
        df_cashflow = self.filter_for_cashflow(df)
        logger.info("pipeline.filtered", remaining=len(df_cashflow), removed=total_rows - len(df_cashflow))

        # Filter to recent years
        if self._recent_years:
            max_year = df_cashflow["year"].max()
            min_year = max_year - self._recent_years + 1
            df_cashflow = df_cashflow[df_cashflow["year"] >= min_year]
            logger.info("pipeline.year_filter", min_year=int(min_year), max_year=int(max_year), rows=len(df_cashflow))

        if df_cashflow.empty:
            warnings.append("No cashflow transactions found after filtering")
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            return PipelineResult(warnings=warnings, processing_time_ms=elapsed, total_rows_processed=total_rows)

        # Step 4: Aggregate monthly
        monthly = self.aggregate_monthly(df_cashflow)

        # Step 5: Compute metrics (using full classified df for broader stats)
        metrics = self.compute_metrics(df_cashflow, monthly)

        # Step 6: Generate text chunks
        chunks = self.generate_text_chunks(monthly, metrics)

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.info("pipeline.complete", months=monthly.total_months, chunks=len(chunks), elapsed_ms=elapsed)

        # Known data gaps -> warnings
        warnings.append("DUEDATE equals TRNDATE in export — payment terms modelled stochastically (Normal μ=52, σ=15 days)")
        warnings.append("Limited payment records relative to sales — collection rate modelled stochastically")
        warnings.append("No supplier invoices in dataset — expenses modelled as ratio of revenue (0.70-0.75)")

        return PipelineResult(
            monthly_data=monthly,
            metrics=metrics,
            text_chunks=chunks,
            warnings=warnings,
            processing_time_ms=elapsed,
            total_rows_processed=total_rows,
        )

    # ── Step 1: Load ────────────────────────────────────────

    def load_excel(self, path: str | Path) -> pd.DataFrame:
        """Load and validate the ERP Excel export.

        Args:
            path: Path to the Excel file.

        Returns:
            DataFrame with parsed dates and validated columns.
        """
        df = pd.read_excel(path)

        required = {"TRNDATE", "TRNVALUE", "TURNOVER", "DOCCODE", "PERID"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Parse dates
        df["TRNDATE"] = pd.to_datetime(df["TRNDATE"], errors="coerce")
        df = df.dropna(subset=["TRNDATE"])

        # Filter out 1970 epoch artefacts
        df = df[df["TRNDATE"].dt.year >= 2000]

        # Extract year/month for grouping
        df["year"] = df["TRNDATE"].dt.year
        df["month"] = df["TRNDATE"].dt.month

        return df

    # ── Step 2: Classify ────────────────────────────────────

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add doc_type, direction, and prefix columns.

        Args:
            df: DataFrame with DOCCODE column.

        Returns:
            DataFrame with added classification columns.
        """
        df = df.copy()
        df["prefix"] = df["DOCCODE"].apply(lambda x: extract_prefix(str(x)))

        classifications = df["DOCCODE"].apply(lambda x: classify_transaction(str(x)))
        df["doc_type"] = classifications.apply(lambda t: t.type)
        df["direction"] = classifications.apply(lambda t: t.direction)

        return df

    # ── Step 3: Filter ──────────────────────────────────────

    def filter_for_cashflow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove opening balances and non-cashflow transactions.

        Keeps: sale, payment, credit_note, receipt, reversal.
        Removes: opening, other, unknown.
        """
        cashflow_types = {"sale", "payment", "credit_note", "receipt", "reversal"}
        return df[df["doc_type"].isin(cashflow_types)].copy()

    # ── Step 4: Aggregate ───────────────────────────────────

    def aggregate_monthly(self, df: pd.DataFrame) -> MonthlyData:
        """Aggregate transactions into monthly records.

        Args:
            df: Classified DataFrame with direction column.

        Returns:
            MonthlyData with one record per year-month.
        """
        records: list[MonthlyRecord] = []

        grouped = df.groupby(["year", "month"])
        for (year, month), group in sorted(grouped):
            sales = group[group["direction"] == "inflow"]
            payments = group[group["direction"] == "outflow"]
            credits = group[group["direction"] == "reduction"]
            receipts = group[group["doc_type"] == "receipt"]
            reversals = group[group["direction"] == "reversal"]

            rec = MonthlyRecord(
                year=int(year),
                month=int(month),
                sales_gross=float(sales["TRNVALUE"].sum()),
                sales_net=float(sales["TURNOVER"].sum()) if "TURNOVER" in sales.columns else 0.0,
                credit_notes=float(credits["TRNVALUE"].sum()),
                payments_out=float(payments["TRNVALUE"].sum()),
                receipts_in=float(receipts["TRNVALUE"].sum()),
                reversals=float(reversals["TRNVALUE"].sum()),
                transaction_count=len(group),
                unique_customers=int(group["PERID"].nunique()),
            )
            records.append(rec)

        return MonthlyData(records=records)

    # ── Step 5: Metrics ─────────────────────────────────────

    def compute_metrics(self, df: pd.DataFrame, monthly: MonthlyData) -> BusinessMetrics:
        """Compute business intelligence metrics.

        Args:
            df: Classified, filtered DataFrame.
            monthly: Aggregated monthly data.

        Returns:
            BusinessMetrics with seasonal indices, concentration, etc.
        """
        # Seasonal indices (12 values, mean=1.0)
        seasonal = self._compute_seasonal_indices(monthly)

        # Customer concentration
        concentration = self._compute_customer_concentration(df)

        # Invoice distribution (sales only)
        sales_df = df[df["direction"] == "inflow"]
        invoice_dist = self._compute_invoice_distribution(sales_df)

        # Credit note ratio
        total_sales = sales_df["TRNVALUE"].sum()
        total_credits = df[df["direction"] == "reduction"]["TRNVALUE"].sum()
        credit_ratio = float(total_credits / total_sales) if total_sales > 0 else 0.0

        # VAT rate detection
        if "TURNOVER" in sales_df.columns:
            mask = (sales_df["TURNOVER"] > 0) & (sales_df["TRNVALUE"] > 0)
            valid = sales_df[mask]
            if len(valid) > 0:
                ratios = (valid["TRNVALUE"] / valid["TURNOVER"]) - 1.0
                vat_rate = float(ratios.median())
            else:
                vat_rate = 0.24
        else:
            vat_rate = 0.24

        # Collection rate (payments / sales)
        total_payments = df[df["direction"] == "outflow"]["TRNVALUE"].sum()
        collection = float(total_payments / total_sales) if total_sales > 0 else 0.0

        # Date range
        dates = df["TRNDATE"].dropna()
        date_start = str(dates.min().date()) if len(dates) > 0 else ""
        date_end = str(dates.max().date()) if len(dates) > 0 else ""

        return BusinessMetrics(
            seasonal_indices=seasonal,
            customer_concentration=concentration,
            invoice_distribution=invoice_dist,
            credit_note_ratio=round(credit_ratio, 4),
            vat_rate=round(vat_rate, 4),
            collection_rate=round(collection, 4),
            total_revenue_gross=round(float(total_sales), 2),
            total_revenue_net=round(float(sales_df["TURNOVER"].sum()) if "TURNOVER" in sales_df.columns else 0.0, 2),
            date_range_start=date_start,
            date_range_end=date_end,
        )

    def _compute_seasonal_indices(self, monthly: MonthlyData) -> list[float]:
        """Compute 12 seasonal indices (mean = 1.0) from monthly sales."""
        month_totals: dict[int, list[float]] = {m: [] for m in range(1, 13)}
        for rec in monthly.records:
            month_totals[rec.month].append(rec.sales_gross)

        # Average sales per calendar month
        month_avgs = []
        for m in range(1, 13):
            vals = month_totals[m]
            month_avgs.append(np.mean(vals) if vals else 0.0)

        overall_mean = np.mean(month_avgs) if any(a > 0 for a in month_avgs) else 1.0
        if overall_mean == 0:
            return [1.0] * 12

        indices = [round(float(a / overall_mean), 4) for a in month_avgs]
        return indices

    def _compute_customer_concentration(self, df: pd.DataFrame) -> CustomerConcentration:
        """Compute customer concentration (top N% of revenue)."""
        sales = df[df["direction"] == "inflow"]
        if sales.empty:
            return CustomerConcentration()

        customer_rev = sales.groupby("PERID")["TRNVALUE"].sum().sort_values(ascending=False)
        total = customer_rev.sum()
        n_customers = len(customer_rev)

        if total == 0 or n_customers == 0:
            return CustomerConcentration(total_customers=n_customers)

        cumulative = customer_rev.cumsum()
        top5_n = max(1, int(np.ceil(n_customers * 0.05)))
        top10_n = max(1, int(np.ceil(n_customers * 0.10)))
        top20_n = max(1, int(np.ceil(n_customers * 0.20)))

        return CustomerConcentration(
            top5_pct=round(float(customer_rev.iloc[:top5_n].sum() / total) * 100, 2),
            top10_pct=round(float(customer_rev.iloc[:top10_n].sum() / total) * 100, 2),
            top20_pct=round(float(customer_rev.iloc[:top20_n].sum() / total) * 100, 2),
            total_customers=n_customers,
        )

    def _compute_invoice_distribution(self, sales_df: pd.DataFrame) -> InvoiceDistribution:
        """Compute statistical distribution of invoice amounts."""
        if sales_df.empty:
            return InvoiceDistribution()

        values = sales_df["TRNVALUE"].dropna()
        values = values[values > 0]

        if values.empty:
            return InvoiceDistribution()

        return InvoiceDistribution(
            mean=round(float(values.mean()), 2),
            median=round(float(values.median()), 2),
            std=round(float(values.std()), 2),
            p95=round(float(np.percentile(values, 95)), 2),
            count=len(values),
        )

    # ── Step 6: Text chunks ─────────────────────────────────

    def generate_text_chunks(
        self, monthly: MonthlyData, metrics: BusinessMetrics
    ) -> list[TextChunk]:
        """Generate narrative text chunks for RAG embedding.

        Creates:
        - One chunk per month (summary narrative)
        - One chunk for customer concentration analysis
        - One chunk for seasonal patterns
        - One chunk for risk factors / data quality
        - One chunk for overall metrics summary

        Args:
            monthly: Aggregated monthly data.
            metrics: Computed business metrics.

        Returns:
            List of TextChunk objects ready for embedding.
        """
        chunks: list[TextChunk] = []

        # Monthly summaries
        for rec in monthly.records:
            text = (
                f"Μηνιαία σύνοψη {rec.period_label}: "
                f"Πωλήσεις (ακαθάριστες) €{rec.sales_gross:,.2f}, "
                f"Πωλήσεις (καθαρές) €{rec.sales_net:,.2f}, "
                f"Πιστωτικά τιμολόγια €{rec.credit_notes:,.2f}, "
                f"Πληρωμές εξόδων €{rec.payments_out:,.2f}, "
                f"Εισπράξεις ταμείου €{rec.receipts_in:,.2f}, "
                f"Καθαρή ταμειακή ροή €{rec.net_cashflow:,.2f}. "
                f"Αριθμός συναλλαγών: {rec.transaction_count}, "
                f"Μοναδικοί πελάτες: {rec.unique_customers}."
            )
            chunks.append(TextChunk(
                text=text,
                metadata={"year": rec.year, "month": rec.month, "period": rec.period_label},
                chunk_id=f"monthly_{rec.period_label}",
                chunk_type="monthly_summary",
            ))

        # Customer concentration
        cc = metrics.customer_concentration
        chunks.append(TextChunk(
            text=(
                f"Ανάλυση συγκέντρωσης πελατών: "
                f"Το top 5% των πελατών ({max(1, int(cc.total_customers * 0.05))} πελάτες) "
                f"αντιπροσωπεύει {cc.top5_pct:.1f}% των εσόδων. "
                f"Το top 10% αντιπροσωπεύει {cc.top10_pct:.1f}% και "
                f"το top 20% αντιπροσωπεύει {cc.top20_pct:.1f}%. "
                f"Σύνολο πελατών: {cc.total_customers}. "
                f"{'Υψηλή συγκέντρωση — κίνδυνος εξάρτησης από λίγους πελάτες.' if cc.top5_pct > 40 else 'Μέτρια κατανομή εσόδων.'}"
            ),
            metadata={"total_customers": cc.total_customers},
            chunk_id="customer_concentration",
            chunk_type="customer_analysis",
        ))

        # Seasonal patterns
        months_gr = ["Ιαν", "Φεβ", "Μαρ", "Απρ", "Μαι", "Ιουν", "Ιουλ", "Αυγ", "Σεπ", "Οκτ", "Νοε", "Δεκ"]
        seasonal_text = "Εποχιακοί δείκτες πωλήσεων: "
        for i, (name, idx) in enumerate(zip(months_gr, metrics.seasonal_indices)):
            seasonal_text += f"{name}={idx:.2f}"
            if i < 11:
                seasonal_text += ", "
        seasonal_text += ". "
        peak_month = months_gr[np.argmax(metrics.seasonal_indices)] if metrics.seasonal_indices else "N/A"
        trough_month = months_gr[np.argmin(metrics.seasonal_indices)] if metrics.seasonal_indices else "N/A"
        seasonal_text += f"Κορυφαίος μήνας: {peak_month}. Χαμηλότερος μήνας: {trough_month}."

        chunks.append(TextChunk(
            text=seasonal_text,
            metadata={"indices": metrics.seasonal_indices},
            chunk_id="seasonal_patterns",
            chunk_type="seasonal",
        ))

        # Risk factors / data quality
        chunks.append(TextChunk(
            text=(
                f"Παράγοντες κινδύνου και ποιότητα δεδομένων: "
                f"Ποσοστό πιστωτικών σημειωμάτων: {metrics.credit_note_ratio*100:.1f}% των πωλήσεων. "
                f"Ανιχνευμένος ΦΠΑ: {metrics.vat_rate*100:.1f}%. "
                f"Ποσοστό είσπραξης: {metrics.collection_rate*100:.1f}%. "
                f"Γνωστοί περιορισμοί δεδομένων: "
                f"(1) DUEDATE = TRNDATE — μοντελοποίηση όρων πληρωμής στοχαστικά, "
                f"(2) Περιορισμένα αρχεία πληρωμών σε σχέση με πωλήσεις, "
                f"(3) Δεν υπάρχουν τιμολόγια προμηθευτών — μοντελοποίηση εξόδων ως αναλογία εσόδων."
            ),
            metadata={},
            chunk_id="risk_factors",
            chunk_type="risk",
        ))

        # Overall metrics
        inv = metrics.invoice_distribution
        chunks.append(TextChunk(
            text=(
                f"Συνολικά στατιστικά: "
                f"Ακαθάριστα έσοδα €{metrics.total_revenue_gross:,.2f}, "
                f"Καθαρά έσοδα €{metrics.total_revenue_net:,.2f}. "
                f"Περίοδος: {metrics.date_range_start} έως {metrics.date_range_end}. "
                f"Κατανομή τιμολογίων: μέσος €{inv.mean:,.2f}, "
                f"διάμεσος €{inv.median:,.2f}, "
                f"τυπική απόκλιση €{inv.std:,.2f}, "
                f"95ο εκατοστημόριο €{inv.p95:,.2f} "
                f"({inv.count} τιμολόγια)."
            ),
            metadata={
                "total_gross": metrics.total_revenue_gross,
                "total_net": metrics.total_revenue_net,
            },
            chunk_id="overall_metrics",
            chunk_type="metrics",
        ))

        return chunks
