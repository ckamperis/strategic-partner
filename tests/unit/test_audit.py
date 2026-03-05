"""Unit tests for pillars.trust.audit — AuditLogger.

Tests cover:
- Log + retrieve roundtrip
- JSON serialization
- File naming by date
- Empty log
- Multiple entries
- AuditEntry from_dict
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pillars.trust.audit import AuditEntry, AuditLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_audit_dir(tmp_path: Path) -> str:
    """Use pytest's tmp_path for audit logs."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    return str(audit_dir)


@pytest.fixture
def logger(tmp_audit_dir: str) -> AuditLogger:
    return AuditLogger(log_dir=tmp_audit_dir)


def _make_entry(**overrides) -> AuditEntry:
    """Create a test audit entry."""
    defaults = {
        "query": "cashflow forecast",
        "query_type": "cashflow_forecast",
        "knowledge_summary": {"chunk_count": 5, "relevance_score": 0.85},
        "reasoning_summary": {"skill_name": "cashflow_forecast", "confidence": 0.67},
        "simulation_summary": {"base": {"mean": 28_000, "prob_neg": 0.02}},
        "trust_score": {"overall": 0.72, "confidence_level": "medium"},
        "explanation_summary": "Test summary",
        "vector_clock": {"knowledge": 1, "reasoning": 1, "simulation": 1, "trust": 1},
        "execution_time_ms": 150.5,
    }
    defaults.update(overrides)
    return AuditEntry(**defaults)


# ---------------------------------------------------------------------------
# Log + Retrieve roundtrip
# ---------------------------------------------------------------------------

class TestLogRetrieve:
    """Test writing and reading audit entries."""

    def test_log_returns_audit_id(self, logger: AuditLogger) -> None:
        entry = _make_entry()
        audit_id = logger.log(entry)
        assert audit_id
        assert len(audit_id) > 0

    def test_log_creates_file(self, logger: AuditLogger) -> None:
        entry = _make_entry()
        logger.log(entry)
        files = list(logger.log_dir.glob("audit_*.jsonl"))
        assert len(files) == 1

    def test_retrieve_after_log(self, logger: AuditLogger) -> None:
        entry = _make_entry()
        audit_id = logger.log(entry)
        recent = logger.get_recent(n=1)
        assert len(recent) == 1
        assert recent[0].audit_id == audit_id
        assert recent[0].query == "cashflow forecast"

    def test_roundtrip_preserves_data(self, logger: AuditLogger) -> None:
        entry = _make_entry(query="test roundtrip")
        logger.log(entry)
        recent = logger.get_recent(n=1)
        assert recent[0].query == "test roundtrip"
        assert recent[0].query_type == "cashflow_forecast"
        assert recent[0].knowledge_summary["chunk_count"] == 5
        assert recent[0].vector_clock["knowledge"] == 1


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

class TestJsonSerialisation:
    """Test JSON format in log files."""

    def test_file_contains_valid_json(self, logger: AuditLogger) -> None:
        entry = _make_entry()
        logger.log(entry)
        files = list(logger.log_dir.glob("audit_*.jsonl"))
        with open(files[0], "r", encoding="utf-8") as f:
            line = f.readline().strip()
        data = json.loads(line)
        assert data["query"] == "cashflow forecast"

    def test_to_dict_roundtrip(self) -> None:
        entry = _make_entry()
        entry.audit_id = "test-id-123"
        entry.timestamp = "2026-02-19T12:00:00Z"
        d = entry.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored.audit_id == "test-id-123"
        assert restored.query == "cashflow forecast"
        assert restored.execution_time_ms == 150.5


# ---------------------------------------------------------------------------
# Multiple entries
# ---------------------------------------------------------------------------

class TestMultipleEntries:
    """Test logging and retrieving multiple entries."""

    def test_multiple_entries_in_order(self, logger: AuditLogger) -> None:
        for i in range(5):
            entry = _make_entry(query=f"query_{i}")
            logger.log(entry)
        recent = logger.get_recent(n=5)
        assert len(recent) == 5
        # Most recent first
        assert recent[0].query == "query_4"
        assert recent[4].query == "query_0"

    def test_get_recent_limits(self, logger: AuditLogger) -> None:
        for i in range(10):
            logger.log(_make_entry(query=f"query_{i}"))
        recent = logger.get_recent(n=3)
        assert len(recent) == 3

    def test_append_only(self, logger: AuditLogger) -> None:
        """Multiple logs should append, not overwrite."""
        logger.log(_make_entry(query="first"))
        logger.log(_make_entry(query="second"))
        files = list(logger.log_dir.glob("audit_*.jsonl"))
        assert len(files) == 1  # Same day = same file
        with open(files[0], "r") as f:
            lines = f.readlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Empty log
# ---------------------------------------------------------------------------

class TestEmptyLog:
    """Test behaviour with empty logs."""

    def test_get_recent_empty(self, logger: AuditLogger) -> None:
        recent = logger.get_recent(n=10)
        assert recent == []


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------

class TestFileNaming:
    """Test file naming by date."""

    def test_file_name_format(self, logger: AuditLogger) -> None:
        logger.log(_make_entry())
        files = list(logger.log_dir.glob("audit_*.jsonl"))
        assert len(files) == 1
        name = files[0].name
        assert name.startswith("audit_")
        assert name.endswith(".jsonl")
        # Should contain a date: audit_YYYY-MM-DD.jsonl
        date_part = name.replace("audit_", "").replace(".jsonl", "")
        assert len(date_part) == 10  # YYYY-MM-DD


# ---------------------------------------------------------------------------
# Auto-generated fields
# ---------------------------------------------------------------------------

class TestAutoFields:
    """Test auto-generated timestamp and audit_id."""

    def test_auto_timestamp(self, logger: AuditLogger) -> None:
        entry = _make_entry()
        assert entry.timestamp == ""
        logger.log(entry)
        assert entry.timestamp != ""

    def test_auto_audit_id(self, logger: AuditLogger) -> None:
        entry = _make_entry()
        assert entry.audit_id == ""
        audit_id = logger.log(entry)
        assert entry.audit_id == audit_id

    def test_custom_timestamp_preserved(self, logger: AuditLogger) -> None:
        entry = _make_entry()
        entry.timestamp = "2026-01-01T00:00:00Z"
        logger.log(entry)
        assert entry.timestamp == "2026-01-01T00:00:00Z"

    def test_custom_audit_id_preserved(self, logger: AuditLogger) -> None:
        entry = _make_entry()
        entry.audit_id = "custom-id"
        returned_id = logger.log(entry)
        assert returned_id == "custom-id"
