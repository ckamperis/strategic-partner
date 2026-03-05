"""Unit tests for picp.bus — PICPBus (in-memory fallback).

Tests cover:
- Publish/subscribe with in-memory fallback
- Event ordering in the log
- Correlation ID filtering
- Multiple subscribers for same event
- Callback error isolation
"""

from __future__ import annotations

import pytest

from picp.bus import PICPBus
from picp.message import PICPContext, PICPEvent, PICPMessage


@pytest.fixture
def bus() -> PICPBus:
    """PICPBus using in-memory fallback (no Redis)."""
    return PICPBus(redis=None)


@pytest.fixture
def context() -> PICPContext:
    return PICPContext.new(query="test query")


class TestPublishSubscribe:
    """In-memory pub/sub."""

    @pytest.mark.asyncio
    async def test_subscriber_receives_message(self, bus: PICPBus, context: PICPContext) -> None:
        received: list[PICPMessage] = []

        async def handler(msg: PICPMessage) -> None:
            received.append(msg)

        await bus.subscribe(PICPEvent.KNOWLEDGE_UPDATED, handler)
        await bus.publish(PICPEvent.KNOWLEDGE_UPDATED, context, source_pillar="knowledge")

        assert len(received) == 1
        assert received[0].event == PICPEvent.KNOWLEDGE_UPDATED
        assert received[0].source_pillar == "knowledge"

    @pytest.mark.asyncio
    async def test_no_crosstalk_between_events(self, bus: PICPBus, context: PICPContext) -> None:
        received: list[PICPMessage] = []

        async def handler(msg: PICPMessage) -> None:
            received.append(msg)

        await bus.subscribe(PICPEvent.KNOWLEDGE_UPDATED, handler)
        await bus.publish(PICPEvent.REASONING_COMPLETE, context, source_pillar="reasoning")

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus: PICPBus, context: PICPContext) -> None:
        count = {"a": 0, "b": 0}

        async def handler_a(msg: PICPMessage) -> None:
            count["a"] += 1

        async def handler_b(msg: PICPMessage) -> None:
            count["b"] += 1

        await bus.subscribe(PICPEvent.SIMULATION_READY, handler_a)
        await bus.subscribe(PICPEvent.SIMULATION_READY, handler_b)
        await bus.publish(PICPEvent.SIMULATION_READY, context, source_pillar="simulation")

        assert count["a"] == 1
        assert count["b"] == 1

    @pytest.mark.asyncio
    async def test_callback_error_isolation(self, bus: PICPBus, context: PICPContext) -> None:
        """A failing callback should not prevent other subscribers."""
        results: list[str] = []

        async def failing_handler(msg: PICPMessage) -> None:
            raise RuntimeError("boom")

        async def good_handler(msg: PICPMessage) -> None:
            results.append("ok")

        await bus.subscribe(PICPEvent.TRUST_VALIDATED, failing_handler)
        await bus.subscribe(PICPEvent.TRUST_VALIDATED, good_handler)
        await bus.publish(PICPEvent.TRUST_VALIDATED, context, source_pillar="trust")

        assert results == ["ok"]


class TestEventLog:
    """Ordered event log (audit trail)."""

    @pytest.mark.asyncio
    async def test_events_logged_in_order(self, bus: PICPBus, context: PICPContext) -> None:
        await bus.publish(PICPEvent.QUERY_RECEIVED, context, source_pillar="orchestrator")
        await bus.publish(PICPEvent.KNOWLEDGE_UPDATED, context, source_pillar="knowledge")
        await bus.publish(PICPEvent.REASONING_COMPLETE, context, source_pillar="reasoning")

        log = await bus.get_event_log()
        assert len(log) == 3
        assert log[0]["event"] == "query_received"
        assert log[1]["event"] == "knowledge_updated"
        assert log[2]["event"] == "reasoning_complete"

    @pytest.mark.asyncio
    async def test_filter_by_correlation_id(self, bus: PICPBus) -> None:
        ctx1 = PICPContext.new(query="query1")
        ctx2 = PICPContext.new(query="query2")

        await bus.publish(PICPEvent.QUERY_RECEIVED, ctx1, source_pillar="orchestrator")
        await bus.publish(PICPEvent.QUERY_RECEIVED, ctx2, source_pillar="orchestrator")

        log1 = await bus.get_event_log(correlation_id=ctx1.correlation_id)
        assert len(log1) == 1
        assert log1[0]["correlation_id"] == ctx1.correlation_id

    @pytest.mark.asyncio
    async def test_event_log_limit(self, bus: PICPBus, context: PICPContext) -> None:
        for _ in range(10):
            await bus.publish(PICPEvent.PILLAR_ERROR, context, source_pillar="test")

        log = await bus.get_event_log(limit=3)
        assert len(log) == 3

    @pytest.mark.asyncio
    async def test_event_log_contains_correlation_id(
        self, bus: PICPBus, context: PICPContext
    ) -> None:
        await bus.publish(PICPEvent.RESPONSE_READY, context, source_pillar="orchestrator")
        log = await bus.get_event_log()
        assert log[0]["correlation_id"] == context.correlation_id

    @pytest.mark.asyncio
    async def test_event_log_contains_vector_clock(
        self, bus: PICPBus, context: PICPContext
    ) -> None:
        await bus.publish(PICPEvent.QUERY_RECEIVED, context, source_pillar="orchestrator")
        log = await bus.get_event_log()
        assert "vector_clock" in log[0]
