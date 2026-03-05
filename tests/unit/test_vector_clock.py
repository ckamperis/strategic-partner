"""Unit tests for picp.vector_clock — VectorClock implementation.

Tests cover:
- Initialisation with default and custom pillars
- Increment (Eq. 3.22)
- Merge (Eq. 3.23)
- Causal comparison (BEFORE, AFTER, CONCURRENT, EQUAL)
- Serialisation round-trip (to_dict / from_dict)
- Immutability guarantees
- Error handling for unknown pillars
"""

from __future__ import annotations

import pytest

from picp.vector_clock import CausalOrder, VectorClock


class TestVectorClockInit:
    """Initialisation and defaults."""

    def test_default_pillars(self) -> None:
        vc = VectorClock()
        assert set(vc.clocks.keys()) == {"knowledge", "reasoning", "simulation", "trust"}
        assert all(v == 0 for v in vc.clocks.values())

    def test_custom_pillars(self) -> None:
        vc = VectorClock(pillars=["a", "b", "c"])
        assert set(vc.clocks.keys()) == {"a", "b", "c"}

    def test_from_existing_clocks(self) -> None:
        vc = VectorClock(clocks={"x": 3, "y": 7})
        assert vc["x"] == 3
        assert vc["y"] == 7


class TestIncrement:
    """Eq. 3.22 — VC[i] = VC[i] + 1."""

    def test_increment_single(self, vector_clock: VectorClock) -> None:
        vc2 = vector_clock.increment("knowledge")
        assert vc2["knowledge"] == 1
        assert vc2["reasoning"] == 0

    def test_increment_multiple(self, vector_clock: VectorClock) -> None:
        vc = vector_clock.increment("knowledge").increment("knowledge").increment("reasoning")
        assert vc["knowledge"] == 2
        assert vc["reasoning"] == 1

    def test_increment_returns_new_instance(self, vector_clock: VectorClock) -> None:
        vc2 = vector_clock.increment("knowledge")
        assert vc2 is not vector_clock
        assert vector_clock["knowledge"] == 0  # original unchanged

    def test_increment_unknown_pillar_raises(self, vector_clock: VectorClock) -> None:
        with pytest.raises(KeyError, match="Unknown pillar 'nonexistent'"):
            vector_clock.increment("nonexistent")


class TestMerge:
    """Eq. 3.23 — VC[i] = max(VC_local[i], VC_received[i])."""

    def test_merge_takes_max(self) -> None:
        vc1 = VectorClock(clocks={"a": 3, "b": 1, "c": 0})
        vc2 = VectorClock(clocks={"a": 1, "b": 5, "c": 2})
        merged = vc1.merge(vc2)
        assert merged["a"] == 3
        assert merged["b"] == 5
        assert merged["c"] == 2

    def test_merge_returns_new_instance(self) -> None:
        vc1 = VectorClock(clocks={"a": 1})
        vc2 = VectorClock(clocks={"a": 2})
        merged = vc1.merge(vc2)
        assert merged is not vc1
        assert merged is not vc2
        assert vc1["a"] == 1  # original unchanged

    def test_merge_with_disjoint_pillars(self) -> None:
        vc1 = VectorClock(clocks={"a": 3})
        vc2 = VectorClock(clocks={"b": 5})
        merged = vc1.merge(vc2)
        assert merged["a"] == 3
        assert merged["b"] == 5

    def test_merge_symmetric(self) -> None:
        vc1 = VectorClock(clocks={"a": 3, "b": 1})
        vc2 = VectorClock(clocks={"a": 1, "b": 5})
        assert vc1.merge(vc2).clocks == vc2.merge(vc1).clocks


class TestCompare:
    """Causality ordering via vector clock comparison."""

    def test_equal(self) -> None:
        vc1 = VectorClock(clocks={"a": 1, "b": 2})
        vc2 = VectorClock(clocks={"a": 1, "b": 2})
        assert vc1.compare(vc2) == CausalOrder.EQUAL

    def test_before(self) -> None:
        vc1 = VectorClock(clocks={"a": 1, "b": 2})
        vc2 = VectorClock(clocks={"a": 2, "b": 3})
        assert vc1.compare(vc2) == CausalOrder.BEFORE

    def test_after(self) -> None:
        vc1 = VectorClock(clocks={"a": 3, "b": 4})
        vc2 = VectorClock(clocks={"a": 1, "b": 2})
        assert vc1.compare(vc2) == CausalOrder.AFTER

    def test_concurrent(self) -> None:
        vc1 = VectorClock(clocks={"a": 3, "b": 1})
        vc2 = VectorClock(clocks={"a": 1, "b": 5})
        assert vc1.compare(vc2) == CausalOrder.CONCURRENT

    def test_before_with_some_equal(self) -> None:
        vc1 = VectorClock(clocks={"a": 1, "b": 2, "c": 3})
        vc2 = VectorClock(clocks={"a": 1, "b": 2, "c": 4})
        assert vc1.compare(vc2) == CausalOrder.BEFORE

    def test_concurrent_detection_is_correct(self) -> None:
        """Concurrent: a > b on one component, a < b on another."""
        vc_knowledge = VectorClock().increment("knowledge")
        vc_reasoning = VectorClock().increment("reasoning")
        assert vc_knowledge.compare(vc_reasoning) == CausalOrder.CONCURRENT


class TestSerialisation:
    """Round-trip to/from dict for Redis/JSON storage."""

    def test_to_dict(self, vector_clock: VectorClock) -> None:
        d = vector_clock.to_dict()
        assert isinstance(d, dict)
        assert d == {"knowledge": 0, "reasoning": 0, "simulation": 0, "trust": 0}

    def test_from_dict(self) -> None:
        data = {"knowledge": 5, "reasoning": 3, "simulation": 1, "trust": 0}
        vc = VectorClock.from_dict(data)
        assert vc["knowledge"] == 5
        assert vc["trust"] == 0

    def test_round_trip(self, vector_clock: VectorClock) -> None:
        vc = vector_clock.increment("knowledge").increment("reasoning")
        restored = VectorClock.from_dict(vc.to_dict())
        assert restored.clocks == vc.clocks

    def test_repr(self) -> None:
        vc = VectorClock(clocks={"a": 1, "b": 2})
        assert "a=1" in repr(vc)
        assert "b=2" in repr(vc)
