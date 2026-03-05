"""PICP-specific exceptions.

Hierarchy:
    PICPError
    ├── PICPViolationError   — priority hierarchy violation
    ├── PICPLockTimeout      — lock acquisition timeout
    └── PillarDegradedError  — pillar operating in degraded mode
"""


class PICPError(Exception):
    """Base exception for all PICP protocol errors."""

    pass


class PICPViolationError(PICPError):
    """Raised when a lower-priority pillar attempts to call a higher-priority one.

    The PICP priority hierarchy (Section 3.4) is:
        Reasoning (1) > Simulation (2) > Trust (3) > Knowledge (4)
    Higher-priority pillars may invoke lower-priority ones, but not vice versa.
    """

    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        super().__init__(
            f"PICP violation: '{source}' (lower priority) cannot call "
            f"'{target}' (higher priority)"
        )


class PICPLockTimeout(PICPError):
    """Raised when a distributed lock cannot be acquired within the TTL.

    Implements timeout semantics for the Redlock algorithm (Eq. 3.24).
    """

    def __init__(self, resource: str, ttl_ms: int) -> None:
        self.resource = resource
        self.ttl_ms = ttl_ms
        super().__init__(
            f"Failed to acquire lock on '{resource}' within {ttl_ms}ms"
        )


class PillarDegradedError(PICPError):
    """Raised when a pillar is operating in degraded mode.

    The pipeline continues with reduced functionality; this error
    is logged and reflected in the Trust score.
    """

    def __init__(self, pillar: str, reason: str) -> None:
        self.pillar = pillar
        self.reason = reason
        super().__init__(
            f"Pillar '{pillar}' degraded: {reason}"
        )
