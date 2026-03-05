"""Unit tests for data.pipeline — classifier, transformer, models.

Tests cover:
- Classifier: known DOCCODE prefix extraction and classification
- TransactionType model
- ERPTransformer with sample data (first 200 rows of real Excel)
- Monthly aggregation correctness
- Metric computation (seasonal indices mean ≈ 1.0)
- Text chunk generation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.pipeline.classifier import (
    DOC_TYPE_MAP,
    classify_transaction,
    extract_prefix,
)
from data.pipeline.models import (
    BusinessMetrics,
    MonthlyData,
    MonthlyRecord,
    PipelineResult,
    TextChunk,
    TransactionType,
)
from data.pipeline.transformer import ERPTransformer

# Path to the real dataset (for tests that need it)
DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "cashflow_dataset.xlsx"


class TestClassifier:
    """DOCCODE prefix extraction and classification."""

    @pytest.mark.parametrize(
        "doccode,expected_prefix",
        [
            ("00ΤΠΝ 1", "ΤΠΝ"),
            ("00ΤΔΝ 5", "ΤΔΝ"),
            ("00ΧΠΑ 3", "ΧΠΑ"),
            ("00ΠΤΝ 2", "ΠΤΝ"),
            ("00ΑΤΝ 1", "ΑΤΝ"),
            ("00ΑΠΧ 1", "ΑΠΧ"),
            ("00ΑΚΠ 1", "ΑΚΠ"),
            ("ΤΠΝ", "ΤΠΝ"),
            ("ΧΠΒ", "ΧΠΒ"),
            ("01ΑΠΠ 3", "ΑΠΠ"),
            ("00ΔΠΣ 1", "ΔΠΣ"),
        ],
    )
    def test_extract_prefix(self, doccode: str, expected_prefix: str) -> None:
        assert extract_prefix(doccode) == expected_prefix

    def test_classify_sale(self) -> None:
        t = classify_transaction("00ΤΠΝ 1")
        assert t.type == "sale"
        assert t.direction == "inflow"

    def test_classify_payment(self) -> None:
        t = classify_transaction("00ΧΠΑ 5")
        assert t.type == "payment"
        assert t.direction == "outflow"

    def test_classify_credit_note(self) -> None:
        t = classify_transaction("00ΠΤΝ 2")
        assert t.type == "credit_note"
        assert t.direction == "reduction"

    def test_classify_receipt(self) -> None:
        t = classify_transaction("00ΑΤΝ 1")
        assert t.type == "receipt"
        assert t.direction == "inflow"

    def test_classify_opening(self) -> None:
        t = classify_transaction("00ΑΠΧ 1")
        assert t.type == "opening"
        assert t.direction == "none"

    def test_classify_reversal(self) -> None:
        t = classify_transaction("00ΑΚΠ 1")
        assert t.type == "reversal"
        assert t.direction == "reversal"

    def test_classify_unknown(self) -> None:
        t = classify_transaction("XXXX")
        assert t.type == "unknown"

    def test_doc_type_map_has_key_types(self) -> None:
        assert "ΤΠΝ" in DOC_TYPE_MAP
        assert "ΧΠΑ" in DOC_TYPE_MAP
        assert "ΠΤΝ" in DOC_TYPE_MAP
        assert "ΑΤΝ" in DOC_TYPE_MAP


class TestModels:
    """Pydantic model correctness."""

    def test_monthly_record_net_cashflow(self) -> None:
        rec = MonthlyRecord(
            year=2023, month=1,
            sales_gross=100_000, receipts_in=5_000,
            payments_out=60_000, credit_notes=10_000,
        )
        assert rec.net_cashflow == 35_000

    def test_monthly_record_period_label(self) -> None:
        rec = MonthlyRecord(year=2023, month=3)
        assert rec.period_label == "2023-03"

    def test_monthly_data_properties(self) -> None:
        data = MonthlyData(records=[
            MonthlyRecord(year=2022, month=11),
            MonthlyRecord(year=2022, month=12),
            MonthlyRecord(year=2023, month=1),
        ])
        assert data.total_months == 3
        assert data.years == [2022, 2023]

    def test_pipeline_result_defaults(self) -> None:
        result = PipelineResult()
        assert result.total_rows_processed == 0
        assert result.text_chunks == []


class TestERPTransformerWithSampleData:
    """Tests using a synthetic DataFrame that mimics the real ERP export."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a small synthetic dataset mimicking the ERP export."""
        rows = []
        # 12 months of sales
        for month in range(1, 13):
            for i in range(10):
                rows.append({
                    "ID": len(rows) + 1,
                    "TRNDATE": pd.Timestamp(2023, month, 15),
                    "TRNVALUE": 1000.0 + month * 100 + i * 10,
                    "TURNOVER": (1000.0 + month * 100 + i * 10) / 1.24,
                    "DOCCODE": "00ΤΠΝ 1",
                    "PERID": i + 1,
                })
            # 3 payments per month
            for i in range(3):
                rows.append({
                    "ID": len(rows) + 1,
                    "TRNDATE": pd.Timestamp(2023, month, 20),
                    "TRNVALUE": 500.0 + month * 50,
                    "TURNOVER": 0,
                    "DOCCODE": "00ΧΠΑ 1",
                    "PERID": i + 1,
                })
            # 1 credit note per month
            rows.append({
                "ID": len(rows) + 1,
                "TRNDATE": pd.Timestamp(2023, month, 25),
                "TRNVALUE": 200.0,
                "TURNOVER": 200.0 / 1.24,
                "DOCCODE": "00ΠΤΝ 1",
                "PERID": 1,
            })
        # Opening balance (should be filtered out)
        rows.append({
            "ID": len(rows) + 1,
            "TRNDATE": pd.Timestamp(2003, 12, 31),
            "TRNVALUE": 5000.0,
            "TURNOVER": 0,
            "DOCCODE": "00ΑΠΧ 1",
            "PERID": 1,
        })

        return pd.DataFrame(rows)

    def test_classify_adds_columns(self, sample_df: pd.DataFrame) -> None:
        transformer = ERPTransformer(recent_years=0)
        df = transformer.load_excel.__wrapped__(transformer, sample_df) if hasattr(transformer.load_excel, '__wrapped__') else sample_df.copy()
        # Manually prep
        df["TRNDATE"] = pd.to_datetime(df["TRNDATE"])
        df["year"] = df["TRNDATE"].dt.year
        df["month"] = df["TRNDATE"].dt.month
        classified = transformer.classify(df)
        assert "doc_type" in classified.columns
        assert "direction" in classified.columns
        assert "prefix" in classified.columns

    def test_filter_removes_openings(self, sample_df: pd.DataFrame) -> None:
        transformer = ERPTransformer(recent_years=0)
        df = sample_df.copy()
        df["TRNDATE"] = pd.to_datetime(df["TRNDATE"])
        df["year"] = df["TRNDATE"].dt.year
        df["month"] = df["TRNDATE"].dt.month
        classified = transformer.classify(df)
        filtered = transformer.filter_for_cashflow(classified)
        assert len(filtered) < len(classified)
        assert "opening" not in filtered["doc_type"].values

    def test_aggregate_monthly(self, sample_df: pd.DataFrame) -> None:
        transformer = ERPTransformer(recent_years=0)
        df = sample_df.copy()
        df["TRNDATE"] = pd.to_datetime(df["TRNDATE"])
        df["year"] = df["TRNDATE"].dt.year
        df["month"] = df["TRNDATE"].dt.month
        classified = transformer.classify(df)
        filtered = transformer.filter_for_cashflow(classified)
        filtered = filtered[filtered["year"] >= 2023]
        monthly = transformer.aggregate_monthly(filtered)
        assert monthly.total_months == 12
        assert all(r.year == 2023 for r in monthly.records)
        # Each month has 10 sales + 3 payments + 1 credit note = 14 transactions
        assert all(r.transaction_count == 14 for r in monthly.records)

    def test_seasonal_indices_mean_approximately_one(self, sample_df: pd.DataFrame) -> None:
        transformer = ERPTransformer(recent_years=0)
        df = sample_df.copy()
        df["TRNDATE"] = pd.to_datetime(df["TRNDATE"])
        df["year"] = df["TRNDATE"].dt.year
        df["month"] = df["TRNDATE"].dt.month
        classified = transformer.classify(df)
        filtered = transformer.filter_for_cashflow(classified)
        filtered = filtered[filtered["year"] >= 2023]
        monthly = transformer.aggregate_monthly(filtered)
        metrics = transformer.compute_metrics(filtered, monthly)
        assert len(metrics.seasonal_indices) == 12
        mean_idx = np.mean(metrics.seasonal_indices)
        assert abs(mean_idx - 1.0) < 0.1, f"Seasonal mean {mean_idx} not close to 1.0"

    def test_generate_chunks(self, sample_df: pd.DataFrame) -> None:
        transformer = ERPTransformer(recent_years=0)
        df = sample_df.copy()
        df["TRNDATE"] = pd.to_datetime(df["TRNDATE"])
        df["year"] = df["TRNDATE"].dt.year
        df["month"] = df["TRNDATE"].dt.month
        classified = transformer.classify(df)
        filtered = transformer.filter_for_cashflow(classified)
        filtered = filtered[filtered["year"] >= 2023]
        monthly = transformer.aggregate_monthly(filtered)
        metrics = transformer.compute_metrics(filtered, monthly)
        chunks = transformer.generate_text_chunks(monthly, metrics)
        # 12 monthly + customer + seasonal + risk + overall = 16
        assert len(chunks) == 16
        # Check types
        types = {c.chunk_type for c in chunks}
        assert "monthly_summary" in types
        assert "customer_analysis" in types
        assert "seasonal" in types
        assert "risk" in types
        assert "metrics" in types

    def test_chunks_contain_greek(self, sample_df: pd.DataFrame) -> None:
        transformer = ERPTransformer(recent_years=0)
        df = sample_df.copy()
        df["TRNDATE"] = pd.to_datetime(df["TRNDATE"])
        df["year"] = df["TRNDATE"].dt.year
        df["month"] = df["TRNDATE"].dt.month
        classified = transformer.classify(df)
        filtered = transformer.filter_for_cashflow(classified)
        filtered = filtered[filtered["year"] >= 2023]
        monthly = transformer.aggregate_monthly(filtered)
        metrics = transformer.compute_metrics(filtered, monthly)
        chunks = transformer.generate_text_chunks(monthly, metrics)
        # Monthly chunks should contain Greek text
        monthly_chunks = [c for c in chunks if c.chunk_type == "monthly_summary"]
        assert any("Πωλήσεις" in c.text for c in monthly_chunks)
