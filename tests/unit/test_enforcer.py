"""Unit tests for picp.enforcer — PICPEnforcer priority enforcement.

Tests cover:
- Valid calls (higher priority calling lower priority)
- Violations (lower calling higher, or same priority)
- Unknown pillar handling
- Orchestrator bypass
- Wait-for graph deadlock detection
- Cycle detection via DFS
"""

from __future__ import annotations

import pytest

from picp.enforcer import PICPEnforcer
from picp.exceptions import PICPViolationError


@pytest.fixture
def enforcer() -> PICPEnforcer:
    """Enforcer with default thesis priority hierarchy."""
    return PICPEnforcer()


class TestValidCalls:
    """Higher priority (lower number) may call lower priority (higher number)."""

    def test_reasoning_can_call_simulation(self, enforcer: PICPEnforcer) -> None:
        assert enforcer.validate_call("reasoning", "simulation") is True

    def test_reasoning_can_call_trust(self, enforcer: PICPEnforcer) -> None:
        assert enforcer.validate_call("reasoning", "trust") is True

    def test_reasoning_can_call_knowledge(self, enforcer: PICPEnforcer) -> None:
        assert enforcer.validate_call("reasoning", "knowledge") is True

    def test_simulation_can_call_trust(self, enforcer: PICPEnforcer) -> None:
        assert enforcer.validate_call("simulation", "trust") is True

    def test_simulation_can_call_knowledge(self, enforcer: PICPEnforcer) -> None:
        assert enforcer.validate_call("simulation", "knowledge") is True

    def test_trust_can_call_knowledge(self, enforcer: PICPEnforcer) -> None:
        assert enforcer.validate_call("trust", "knowledge") is True


class TestViolations:
    """Lower priority cannot call higher priority; same cannot call same."""

    def test_knowledge_cannot_call_reasoning(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(PICPViolationError):
            enforcer.validate_call("knowledge", "reasoning")

    def test_knowledge_cannot_call_simulation(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(PICPViolationError):
            enforcer.validate_call("knowledge", "simulation")

    def test_simulation_cannot_call_reasoning(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(PICPViolationError):
            enforcer.validate_call("simulation", "reasoning")

    def test_trust_cannot_call_reasoning(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(PICPViolationError):
            enforcer.validate_call("trust", "reasoning")

    def test_trust_cannot_call_simulation(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(PICPViolationError):
            enforcer.validate_call("trust", "simulation")

    def test_self_call_raises(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(PICPViolationError):
            enforcer.validate_call("reasoning", "reasoning")

    def test_violation_error_attributes(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(PICPViolationError) as exc_info:
            enforcer.validate_call("knowledge", "reasoning")
        err = exc_info.value
        assert err.source == "knowledge"
        assert err.target == "reasoning"


class TestUnknownPillar:
    """Unknown pillar names raise KeyError."""

    def test_unknown_source(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(KeyError, match="Unknown source pillar"):
            enforcer.validate_call("nonexistent", "knowledge")

    def test_unknown_target(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(KeyError, match="Unknown target pillar"):
            enforcer.validate_call("reasoning", "nonexistent")


class TestOrchestratorBypass:
    """Orchestrator can call any pillar."""

    def test_orchestrator_can_call_reasoning(self, enforcer: PICPEnforcer) -> None:
        assert enforcer.validate_orchestrator_call("reasoning") is True

    def test_orchestrator_can_call_knowledge(self, enforcer: PICPEnforcer) -> None:
        assert enforcer.validate_orchestrator_call("knowledge") is True

    def test_orchestrator_unknown_pillar_raises(self, enforcer: PICPEnforcer) -> None:
        with pytest.raises(KeyError):
            enforcer.validate_orchestrator_call("nonexistent")


class TestDeadlockDetection:
    """DFS-based cycle detection in the wait-for graph."""

    def test_no_deadlock_linear(self, enforcer: PICPEnforcer) -> None:
        """A -> B -> C should not be a cycle."""
        enforcer.register_wait("reasoning", "simulation")
        enforcer.register_wait("simulation", "trust")
        # No exception raised

    def test_deadlock_simple_cycle(self, enforcer: PICPEnforcer) -> None:
        """A -> B, then B -> A is a cycle."""
        enforcer.register_wait("reasoning", "simulation")
        with pytest.raises(PICPViolationError):
            enforcer.register_wait("simulation", "reasoning")

    def test_deadlock_transitive_cycle(self, enforcer: PICPEnforcer) -> None:
        """A -> B -> C, then C -> A is a cycle."""
        enforcer.register_wait("reasoning", "simulation")
        enforcer.register_wait("simulation", "trust")
        with pytest.raises(PICPViolationError):
            enforcer.register_wait("trust", "reasoning")

    def test_clear_wait_removes_edge(self, enforcer: PICPEnforcer) -> None:
        """After clearing A -> B, B -> A should be allowed."""
        enforcer.register_wait("reasoning", "simulation")
        enforcer.clear_wait("reasoning", "simulation")
        # Now the reverse should not create a cycle
        enforcer.register_wait("simulation", "reasoning")

    def test_self_deadlock(self, enforcer: PICPEnforcer) -> None:
        """A -> A is a cycle."""
        with pytest.raises(PICPViolationError):
            enforcer.register_wait("reasoning", "reasoning")


class TestPriorityMap:
    """Priority map access and custom maps."""

    def test_default_priorities(self, enforcer: PICPEnforcer) -> None:
        p = enforcer.priority_map
        assert p["reasoning"] == 1
        assert p["simulation"] == 2
        assert p["trust"] == 3
        assert p["knowledge"] == 4

    def test_custom_priority_map(self) -> None:
        custom = PICPEnforcer(priority_map={"alpha": 1, "beta": 2})
        assert custom.validate_call("alpha", "beta") is True
        with pytest.raises(PICPViolationError):
            custom.validate_call("beta", "alpha")

    def test_priority_map_is_copy(self, enforcer: PICPEnforcer) -> None:
        """Modifying returned map should not affect enforcer."""
        p = enforcer.priority_map
        p["reasoning"] = 999
        assert enforcer.priority_map["reasoning"] == 1
