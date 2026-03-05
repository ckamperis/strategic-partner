"""Vector clock implementation for PICP causality tracking.

Implements the vector clock mechanism described in Section 3.4:
- Eq. 3.22: VC[i] = VC[i] + 1  (local increment on event)
- Eq. 3.23: VC[i] = max(VC_local[i], VC_received[i])  (merge on receive)

Each pillar maintains its own logical clock component. The vector clock
enables causality ordering between pillar events without a global clock.

References:
    Fidge (1988), Mattern (1989) — vector clock theory
    Thesis Section 3.4, Equations 3.22–3.23
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Default pillars used across the system
DEFAULT_PILLARS: list[str] = ["knowledge", "reasoning", "simulation", "trust"]


class CausalOrder(str, Enum):
    """Result of comparing two vector clocks.

    BEFORE: self happened-before other
    AFTER: other happened-before self
    CONCURRENT: neither happened-before the other
    EQUAL: identical timestamps
    """

    BEFORE = "before"
    AFTER = "after"
    CONCURRENT = "concurrent"
    EQUAL = "equal"


class VectorClock(BaseModel):
    """Logical vector clock for PICP causality tracking.

    Each pillar has a counter that is incremented on local events (Eq. 3.22)
    and merged with received clocks on message receipt (Eq. 3.23).

    The clock is immutable-safe: merge() returns a new instance.

    Attributes:
        clocks: Mapping of pillar name -> logical timestamp.
    """

    clocks: dict[str, int] = Field(default_factory=dict)

    def __init__(self, pillars: list[str] | None = None, **data: Any) -> None:
        """Initialise a vector clock.

        Args:
            pillars: List of pillar names. Defaults to the four system pillars.
            **data: Pydantic model fields (e.g. clocks dict for deserialisation).
        """
        super().__init__(**data)
        if not self.clocks:
            names = pillars if pillars is not None else DEFAULT_PILLARS
            self.clocks = {name: 0 for name in names}

    # ── Eq. 3.22 — Local increment ─────────────────────────
    def increment(self, pillar: str) -> VectorClock:
        """Increment the clock component for *pillar* (Eq. 3.22).

        Args:
            pillar: The pillar performing a local event.

        Returns:
            A **new** VectorClock with the incremented value.

        Raises:
            KeyError: If *pillar* is not in this clock.
        """
        if pillar not in self.clocks:
            raise KeyError(f"Unknown pillar '{pillar}' — known: {list(self.clocks)}")
        new_clocks = dict(self.clocks)
        new_clocks[pillar] += 1
        return VectorClock(clocks=new_clocks)

    # ── Eq. 3.23 — Merge on receive ────────────────────────
    def merge(self, other: VectorClock) -> VectorClock:
        """Merge with another clock using element-wise max (Eq. 3.23).

        Args:
            other: The received vector clock.

        Returns:
            A **new** VectorClock with merged (max) values.
        """
        all_pillars = set(self.clocks) | set(other.clocks)
        merged = {
            p: max(self.clocks.get(p, 0), other.clocks.get(p, 0))
            for p in all_pillars
        }
        return VectorClock(clocks=merged)

    # ── Causality comparison ────────────────────────────────
    def compare(self, other: VectorClock) -> CausalOrder:
        """Compare two vector clocks for causal ordering.

        Returns:
            CausalOrder.BEFORE  — self happened-before other
            CausalOrder.AFTER   — other happened-before self
            CausalOrder.CONCURRENT — neither dominates
            CausalOrder.EQUAL   — identical clocks
        """
        all_pillars = set(self.clocks) | set(other.clocks)

        has_less = False
        has_greater = False

        for p in all_pillars:
            a = self.clocks.get(p, 0)
            b = other.clocks.get(p, 0)
            if a < b:
                has_less = True
            elif a > b:
                has_greater = True

        if has_less and has_greater:
            return CausalOrder.CONCURRENT
        if has_less:
            return CausalOrder.BEFORE
        if has_greater:
            return CausalOrder.AFTER
        return CausalOrder.EQUAL

    # ── Serialisation ───────────────────────────────────────
    def to_dict(self) -> dict[str, int]:
        """Serialise to a plain dict (for Redis / JSON storage)."""
        return dict(self.clocks)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> VectorClock:
        """Deserialise from a plain dict.

        Args:
            data: Mapping of pillar name -> clock value.
        """
        return cls(clocks=dict(data))

    # ── Dunder helpers ──────────────────────────────────────
    def __getitem__(self, pillar: str) -> int:
        return self.clocks[pillar]

    def __repr__(self) -> str:
        entries = ", ".join(f"{k}={v}" for k, v in sorted(self.clocks.items()))
        return f"VectorClock({entries})"
