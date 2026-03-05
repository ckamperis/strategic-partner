"""PICP event bus — Redis pub/sub and Streams for inter-pillar messaging.

Provides two messaging primitives:
1. **Pub/sub** (``publish`` / ``subscribe``) — real-time event notifications
   between pillars, routed through Redis channels ``picp:{event_name}``.
2. **Streams** (``log_event``) — append-only ordered event log used for
   audit trail and experiment replay (Redis Streams).

If Redis is unavailable, falls back to an in-memory asyncio.Queue-based
implementation with a warning log.

References:
    Thesis Section 3.4 — PICP event routing
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any, Callable, Coroutine

import structlog

from picp.message import PICPContext, PICPEvent, PICPMessage

logger = structlog.get_logger()

# Type alias for event callback
EventCallback = Callable[[PICPMessage], Coroutine[Any, Any, None]]


class PICPBus:
    """Event bus for PICP inter-pillar communication.

    Supports both Redis-backed and in-memory operation.

    Args:
        redis: An async Redis client or ``None`` for in-memory mode.
        stream_key: Redis Stream key for the audit log.
    """

    def __init__(self, redis: Any | None = None, stream_key: str = "picp:events") -> None:
        self._redis = redis
        self._stream_key = stream_key
        self._using_fallback = redis is None

        # In-memory fallback structures
        self._subscribers: dict[str, list[EventCallback]] = defaultdict(list)
        self._event_log: list[dict[str, Any]] = []

        # Track pub/sub subscription task (Redis mode)
        self._pubsub: Any | None = None
        self._listener_task: asyncio.Task[None] | None = None

        if self._using_fallback:
            logger.warning("picp_bus.fallback", reason="Redis unavailable, using in-memory event bus")

    # ── Publish ─────────────────────────────────────────────

    async def publish(
        self,
        event: PICPEvent,
        context: PICPContext,
        source_pillar: str = "orchestrator",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Publish a PICP event to all subscribers.

        Also appends to the ordered event log (Stream or in-memory).

        Args:
            event: The protocol event type.
            context: Current pipeline context.
            source_pillar: The pillar emitting the event.
            payload: Optional additional data.
        """
        start = time.perf_counter()

        message = PICPMessage(
            event=event,
            source_pillar=source_pillar,
            payload=payload or {},
            vector_clock_snapshot=dict(context.vector_clock),
            correlation_id=context.correlation_id,
        )

        channel = f"picp:{event.value}"

        if self._using_fallback:
            await self._publish_local(channel, message)
        else:
            await self._publish_redis(channel, message)

        # Log to stream / audit log
        await self._log_event(message)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "picp_bus.published",
            picp_event=event.value,
            source=source_pillar,
            correlation_id=context.correlation_id,
            elapsed_ms=elapsed_ms,
        )

    # ── Subscribe ───────────────────────────────────────────

    async def subscribe(self, event: PICPEvent, callback: EventCallback) -> None:
        """Register a callback for a specific event type.

        Args:
            event: The event type to listen for.
            callback: Async callable invoked when the event fires.
        """
        channel = f"picp:{event.value}"

        if self._using_fallback:
            self._subscribers[channel].append(callback)
        else:
            await self._subscribe_redis(channel, callback)

        logger.debug("picp_bus.subscribed", picp_event=event.value)

    # ── Event log / audit ───────────────────────────────────

    async def get_event_log(
        self, correlation_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Retrieve events from the ordered log.

        Args:
            correlation_id: Filter by query ID.  ``None`` returns all.
            limit: Maximum number of events to return.

        Returns:
            List of event dicts, oldest first.
        """
        if self._using_fallback:
            events = self._event_log
            if correlation_id:
                events = [e for e in events if e.get("correlation_id") == correlation_id]
            return events[-limit:]

        return await self._read_stream(correlation_id, limit)

    # ── Shutdown ────────────────────────────────────────────

    async def close(self) -> None:
        """Clean up subscriptions and connections."""
        if self._pubsub is not None:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
        if self._listener_task is not None:
            self._listener_task.cancel()
        logger.info("picp_bus.closed")

    # ── Redis implementation ────────────────────────────────

    async def _publish_redis(self, channel: str, message: PICPMessage) -> None:
        """Publish via Redis pub/sub."""
        data = message.model_dump_json()
        await self._redis.publish(channel, data)

    async def _subscribe_redis(self, channel: str, callback: EventCallback) -> None:
        """Subscribe via Redis pub/sub."""
        if self._pubsub is None:
            self._pubsub = self._redis.pubsub()

        await self._pubsub.subscribe(channel)
        self._subscribers[channel].append(callback)

        # Start listener if not running
        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._redis_listener())

    async def _redis_listener(self) -> None:
        """Background task that dispatches Redis pub/sub messages."""
        try:
            while True:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode("utf-8")
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")

                    picp_msg = PICPMessage.model_validate_json(data)
                    for cb in self._subscribers.get(channel, []):
                        try:
                            await cb(picp_msg)
                        except Exception:
                            logger.exception("picp_bus.callback_error", channel=channel)

                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass

    async def _log_event(self, message: PICPMessage) -> None:
        """Append event to the ordered log."""
        entry = {
            "event": message.event.value,
            "source_pillar": message.source_pillar,
            "correlation_id": message.correlation_id,
            "timestamp": message.timestamp.isoformat(),
            "vector_clock": message.vector_clock_snapshot,
            "payload_keys": list(message.payload.keys()),
        }

        if self._using_fallback:
            self._event_log.append(entry)
            return

        # Redis Streams: XADD
        flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in entry.items()}
        await self._redis.xadd(self._stream_key, flat)

    async def _read_stream(
        self, correlation_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """Read events from Redis Stream."""
        raw = await self._redis.xrange(self._stream_key, count=limit * 5)
        events: list[dict[str, Any]] = []
        for _stream_id, fields in raw:
            # Decode bytes
            decoded = {}
            for k, v in fields.items():
                key = k.decode("utf-8") if isinstance(k, bytes) else k
                val = v.decode("utf-8") if isinstance(v, bytes) else v
                # Try to parse JSON fields
                try:
                    decoded[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    decoded[key] = val

            if correlation_id and decoded.get("correlation_id") != correlation_id:
                continue
            events.append(decoded)
            if len(events) >= limit:
                break

        return events

    # ── In-memory implementation ────────────────────────────

    async def _publish_local(self, channel: str, message: PICPMessage) -> None:
        """Dispatch to in-memory subscribers."""
        for cb in self._subscribers.get(channel, []):
            try:
                await cb(message)
            except Exception:
                logger.exception("picp_bus.callback_error", channel=channel)
