"""PICP message and context data structures.

Defines the core protocol messages exchanged between pillars:
- PICPEvent: enumeration of all protocol events
- PICPMessage: individual event message with metadata
- PICPContext: per-query context that flows through the pillar pipeline

References:
    Section 3.4 — Pillar Integration Context Protocol
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PICPEvent(str, Enum):
    """Protocol events emitted during query processing.

    Event flow for every query (Section 3.4):
        QUERY_RECEIVED -> KNOWLEDGE_UPDATED -> REASONING_COMPLETE
        -> SIMULATION_READY -> TRUST_VALIDATED -> RESPONSE_READY

    PILLAR_ERROR may be emitted at any stage for graceful degradation.
    """

    QUERY_RECEIVED = "query_received"
    KNOWLEDGE_STARTED = "knowledge_started"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    REASONING_STARTED = "reasoning_started"
    REASONING_COMPLETE = "reasoning_complete"
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_READY = "simulation_ready"
    TRUST_STARTED = "trust_started"
    TRUST_VALIDATED = "trust_validated"
    RESPONSE_READY = "response_ready"
    PILLAR_ERROR = "pillar_error"


class PICPMessage(BaseModel):
    """A single PICP protocol message.

    Attributes:
        event: The protocol event type.
        source_pillar: Name of the originating pillar (or "orchestrator").
        target_pillar: Name of the destination pillar (or "all" for broadcast).
        payload: Arbitrary data carried by the message.
        timestamp: UTC timestamp of message creation.
        vector_clock_snapshot: Serialised vector clock state at emission time.
        correlation_id: Unique query identifier for tracing.
    """

    event: PICPEvent
    source_pillar: str
    target_pillar: str = "all"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    vector_clock_snapshot: dict[str, int] = Field(default_factory=dict)
    correlation_id: str = ""


class PICPContext(BaseModel):
    """Per-query context that accumulates results as it flows through pillars.

    Created by the orchestrator at QUERY_RECEIVED and passed sequentially
    through Knowledge -> Reasoning -> Simulation -> Trust.

    Attributes:
        correlation_id: Unique identifier for this query (UUID4).
        query: The user's original natural-language query.
        vector_clock: Current vector clock state (serialised dict).
        created_at: UTC timestamp when the context was created.
        pillar_results: Accumulator for each pillar's output.
        metadata: Additional tracking data (timings, flags).
    """

    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    vector_clock: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pillar_results: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def new(cls, query: str) -> PICPContext:
        """Factory to create a new context with initialised vector clock.

        Args:
            query: The user's natural-language query.

        Returns:
            A fresh PICPContext ready for pipeline processing.
        """
        from picp.vector_clock import VectorClock

        vc = VectorClock()
        return cls(
            query=query,
            vector_clock=vc.to_dict(),
        )
