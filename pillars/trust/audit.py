"""Audit Logger — append-only JSON Lines audit trail.

Records full decision traces for every query processed by the system.
Each entry contains the complete pipeline state: query, knowledge,
reasoning, simulation, trust score, explanation, and PICP vector clock.

Files: audit_YYYY-MM-DD.jsonl (one JSON object per line).
Append-only: past entries are never modified.

References:
    Thesis Section 3.3.4 — Trust Pillar, Audit Trail
    Thesis Section 4.x — Implementation, Reproducibility
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class AuditEntry:
    """A single audit log entry capturing the full pipeline state.

    Attributes:
        timestamp: ISO-8601 UTC timestamp.
        audit_id: Unique identifier for this entry.
        query: The user's original query.
        query_type: Classified query type from routing.
        knowledge_summary: RAG results summary.
        reasoning_summary: Skill and confidence summary.
        simulation_summary: Scenario key stats.
        trust_score: Overall + sub-scores + flags.
        explanation_summary: Short explanation text.
        vector_clock: PICP vector clock state at completion.
        execution_time_ms: Total pipeline execution time.
    """

    timestamp: str = ""
    audit_id: str = ""
    query: str = ""
    query_type: str = ""
    knowledge_summary: dict[str, Any] = field(default_factory=dict)
    reasoning_summary: dict[str, Any] = field(default_factory=dict)
    simulation_summary: dict[str, Any] = field(default_factory=dict)
    trust_score: dict[str, Any] = field(default_factory=dict)
    explanation_summary: str = ""
    vector_clock: dict[str, int] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dictionary."""
        return {
            "timestamp": self.timestamp,
            "audit_id": self.audit_id,
            "query": self.query,
            "query_type": self.query_type,
            "knowledge_summary": self.knowledge_summary,
            "reasoning_summary": self.reasoning_summary,
            "simulation_summary": self.simulation_summary,
            "trust_score": self.trust_score,
            "explanation_summary": self.explanation_summary,
            "vector_clock": self.vector_clock,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        """Reconstruct from a JSON-compatible dictionary."""
        return cls(
            timestamp=data.get("timestamp", ""),
            audit_id=data.get("audit_id", ""),
            query=data.get("query", ""),
            query_type=data.get("query_type", ""),
            knowledge_summary=data.get("knowledge_summary", {}),
            reasoning_summary=data.get("reasoning_summary", {}),
            simulation_summary=data.get("simulation_summary", {}),
            trust_score=data.get("trust_score", {}),
            explanation_summary=data.get("explanation_summary", ""),
            vector_clock=data.get("vector_clock", {}),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )


class AuditLogger:
    """Append-only JSON Lines audit logger.

    Each day gets its own file: ``audit_YYYY-MM-DD.jsonl``.
    Entries are JSON objects, one per line.

    Args:
        log_dir: Directory to write audit files. Created if missing.
    """

    def __init__(self, log_dir: str = "data/audit/") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self) -> Path:
        """Access the audit log directory."""
        return self._log_dir

    def _get_file_path(self, date: str | None = None) -> Path:
        """Get the audit file path for a given date.

        Args:
            date: Date string in YYYY-MM-DD format.
                If None, uses today's date.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._log_dir / f"audit_{date}.jsonl"

    def log(self, entry: AuditEntry) -> str:
        """Append an audit entry to the log file.

        Args:
            entry: The audit entry to log.

        Returns:
            The audit_id of the logged entry.
        """
        # Ensure timestamp and audit_id
        if not entry.timestamp:
            entry.timestamp = datetime.now(timezone.utc).isoformat()
        if not entry.audit_id:
            entry.audit_id = str(uuid.uuid4())

        file_path = self._get_file_path()
        line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(line)

        logger.info(
            "audit.log.written",
            audit_id=entry.audit_id,
            file=str(file_path),
        )

        return entry.audit_id

    def get_recent(self, n: int = 10) -> list[AuditEntry]:
        """Read the most recent N audit entries.

        Reads from today's file first, then previous days if needed.

        Args:
            n: Number of entries to return (most recent first).

        Returns:
            List of AuditEntry, ordered most recent first.
        """
        entries: list[AuditEntry] = []

        # Collect all .jsonl files, sorted by name (date) descending
        files = sorted(self._log_dir.glob("audit_*.jsonl"), reverse=True)

        for file_path in files:
            if len(entries) >= n:
                break
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # Read in reverse order (most recent last in file)
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entries.append(AuditEntry.from_dict(data))
                    except json.JSONDecodeError:
                        logger.warning(
                            "audit.read.invalid_json",
                            file=str(file_path),
                        )
                    if len(entries) >= n:
                        break
            except OSError as e:
                logger.warning(
                    "audit.read.file_error",
                    file=str(file_path),
                    error=str(e),
                )

        return entries[:n]
