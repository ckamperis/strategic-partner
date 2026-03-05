"""PICP priority hierarchy enforcer with deadlock detection.

Enforces the pillar call hierarchy defined in Section 3.4:
    Reasoning (1) > Simulation (2) > Trust (3) > Knowledge (4)

A higher-priority pillar (lower number) may invoke a lower-priority
pillar, but **not** vice versa.  This prevents circular dependencies
and ensures a DAG-structured execution flow.

Additionally provides DFS-based cycle detection on a wait-for graph
to prevent deadlocks in distributed lock acquisition.

References:
    Thesis Section 3.4 — PICP priority hierarchy
"""

from __future__ import annotations

import structlog

from picp.exceptions import PICPViolationError

logger = structlog.get_logger()

# Priority map: lower number = higher priority
DEFAULT_PRIORITY_MAP: dict[str, int] = {
    "reasoning": 1,
    "simulation": 2,
    "trust": 3,
    "knowledge": 4,
}


class PICPEnforcer:
    """Enforces the PICP pillar priority hierarchy.

    Validates inter-pillar calls and detects potential deadlocks
    using a wait-for graph with DFS cycle detection.

    Attributes:
        priority_map: Mapping of pillar name to priority (lower = higher).
    """

    def __init__(self, priority_map: dict[str, int] | None = None) -> None:
        """Initialise the enforcer.

        Args:
            priority_map: Custom priority mapping. Defaults to the thesis hierarchy.
        """
        self._priority_map = priority_map or dict(DEFAULT_PRIORITY_MAP)
        # Wait-for graph: source -> set of targets it is waiting on
        self._wait_for: dict[str, set[str]] = {}

    @property
    def priority_map(self) -> dict[str, int]:
        """Read-only access to the priority mapping."""
        return dict(self._priority_map)

    def validate_call(self, source: str, target: str) -> bool:
        """Validate that *source* is allowed to invoke *target*.

        The rule is: a higher-priority pillar (lower number) may call
        a lower-priority pillar (higher number), but not vice versa.
        Equal priority is also forbidden (a pillar cannot call itself).

        Args:
            source: The calling pillar name.
            target: The target pillar name.

        Returns:
            ``True`` if the call is permitted.

        Raises:
            PICPViolationError: If the call violates the priority hierarchy.
            KeyError: If either pillar name is unknown.
        """
        if source not in self._priority_map:
            raise KeyError(f"Unknown source pillar '{source}'")
        if target not in self._priority_map:
            raise KeyError(f"Unknown target pillar '{target}'")

        source_priority = self._priority_map[source]
        target_priority = self._priority_map[target]

        if source_priority >= target_priority:
            logger.warning(
                "picp.violation",
                source=source,
                target=target,
                source_priority=source_priority,
                target_priority=target_priority,
            )
            raise PICPViolationError(source=source, target=target)

        logger.debug(
            "picp.call_validated",
            source=source,
            target=target,
            source_priority=source_priority,
            target_priority=target_priority,
        )
        return True

    # ── Orchestrator calls bypass hierarchy ──────────────────

    def validate_orchestrator_call(self, target: str) -> bool:
        """Validate that the orchestrator can call any pillar.

        The orchestrator sits outside the priority hierarchy and may
        invoke any pillar directly.

        Args:
            target: The target pillar name.

        Returns:
            ``True`` always (orchestrator has universal access).

        Raises:
            KeyError: If the target pillar name is unknown.
        """
        if target not in self._priority_map:
            raise KeyError(f"Unknown target pillar '{target}'")
        return True

    # ── Wait-for graph deadlock detection ───────────────────

    def register_wait(self, source: str, target: str) -> None:
        """Register that *source* is waiting for *target* to release a resource.

        Args:
            source: The waiting pillar.
            target: The pillar holding the resource.

        Raises:
            PICPViolationError: If adding this edge would create a cycle (deadlock).
        """
        # Check if adding this edge creates a cycle
        if self._would_create_cycle(source, target):
            logger.error(
                "picp.deadlock_detected",
                source=source,
                target=target,
                wait_for_graph=self._serialise_graph(),
            )
            raise PICPViolationError(source=source, target=target)

        if source not in self._wait_for:
            self._wait_for[source] = set()
        self._wait_for[source].add(target)

        logger.debug("picp.wait_registered", source=source, target=target)

    def clear_wait(self, source: str, target: str) -> None:
        """Remove a wait-for edge (resource was released).

        Args:
            source: The previously waiting pillar.
            target: The pillar that released the resource.
        """
        if source in self._wait_for:
            self._wait_for[source].discard(target)
            if not self._wait_for[source]:
                del self._wait_for[source]

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """DFS-based cycle detection on the wait-for graph.

        Checks if adding edge source -> target would create a cycle.
        """
        # If target can reach source via existing edges, cycle exists
        visited: set[str] = set()

        def dfs(node: str) -> bool:
            if node == source:
                return True
            if node in visited:
                return False
            visited.add(node)
            for neighbour in self._wait_for.get(node, set()):
                if dfs(neighbour):
                    return True
            return False

        return dfs(target)

    def _serialise_graph(self) -> dict[str, list[str]]:
        """Serialise the wait-for graph for logging."""
        return {k: sorted(v) for k, v in self._wait_for.items()}
