"""Abstract base class for all PICP pillars.

Every pillar must:
1. Implement ``process()`` — the main entry point for the orchestrator.
2. Increment its vector clock component on each operation.
3. Publish PICP events at start and completion.
4. Log structured timing data for experiments.

References:
    Thesis Section 3.3 — Four-pillar architecture
    Thesis Section 3.4 — PICP coordination protocol
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

from picp.bus import PICPBus
from picp.message import PICPContext, PICPEvent
from picp.vector_clock import VectorClock

logger = structlog.get_logger()


class BasePillar(ABC):
    """Abstract base for the four pillars (Knowledge, Reasoning, Simulation, Trust).

    Provides lifecycle hooks (start/complete events) and timing instrumentation.
    Subclasses only need to implement :meth:`_execute`.

    Args:
        name: Pillar identifier (e.g. "knowledge").
        bus: The PICP event bus for publishing events.
        start_event: Event to publish when processing starts.
        complete_event: Event to publish when processing completes.
    """

    def __init__(
        self,
        name: str,
        bus: PICPBus,
        start_event: PICPEvent,
        complete_event: PICPEvent,
    ) -> None:
        self.name = name
        self.bus = bus
        self._start_event = start_event
        self._complete_event = complete_event

    async def process(self, context: PICPContext, **kwargs: Any) -> dict[str, Any]:
        """Run the pillar's processing pipeline with PICP instrumentation.

        This method:
        1. Increments the pillar's vector clock component.
        2. Publishes a start event.
        3. Calls :meth:`_execute` (subclass implementation).
        4. Stores results in ``context.pillar_results``.
        5. Publishes a completion event.

        Args:
            context: The PICP context flowing through the pipeline.
            **kwargs: Additional arguments forwarded to :meth:`_execute`.

        Returns:
            The pillar's result dictionary.
        """
        start = time.perf_counter()

        # Step 1: Increment vector clock (Eq. 3.22)
        vc = VectorClock.from_dict(context.vector_clock)
        vc = vc.increment(self.name)
        context.vector_clock = vc.to_dict()

        # Step 2: Publish start event
        await self.bus.publish(
            self._start_event, context, source_pillar=self.name
        )

        try:
            # Step 3: Execute pillar-specific logic
            result = await self._execute(context, **kwargs)

            # Step 4: Store result
            context.pillar_results[self.name] = result

            elapsed = time.perf_counter() - start
            logger.info(
                f"{self.name}.complete",
                correlation_id=context.correlation_id,
                elapsed_ms=round(elapsed * 1000, 2),
            )

            # Step 5: Publish completion event
            await self.bus.publish(
                self._complete_event, context, source_pillar=self.name
            )

            # Store timing in metadata
            if "timings" not in context.metadata:
                context.metadata["timings"] = {}
            context.metadata["timings"][self.name] = round(elapsed * 1000, 2)

            return result

        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(
                f"{self.name}.failed",
                error=str(e),
                correlation_id=context.correlation_id,
                elapsed_ms=round(elapsed * 1000, 2),
            )
            await self.bus.publish(
                PICPEvent.PILLAR_ERROR,
                context,
                source_pillar=self.name,
                payload={"error": str(e)},
            )
            raise

    @abstractmethod
    async def _execute(self, context: PICPContext, **kwargs: Any) -> dict[str, Any]:
        """Pillar-specific processing logic.

        Subclasses must implement this method.

        Args:
            context: The PICP context.
            **kwargs: Additional arguments.

        Returns:
            A dict of pillar results to be stored in context.pillar_results.
        """
        ...
