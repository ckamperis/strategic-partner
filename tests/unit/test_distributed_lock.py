"""Unit tests for picp.distributed_lock — RedlockManager (in-memory fallback).

Tests cover:
- Acquire and release lifecycle
- Token-based ownership
- Double-acquire detection (second acquire blocks until release)
- Timeout handling (PICPLockTimeout)
- Extend (ownership verification)
- Release with wrong token
"""

from __future__ import annotations

import pytest

from picp.distributed_lock import RedlockManager
from picp.exceptions import PICPLockTimeout


@pytest.fixture
def lock_manager() -> RedlockManager:
    """RedlockManager using in-memory fallback (no Redis)."""
    return RedlockManager(redis=None, ttl_ms=1000, retry_delay_ms=10, max_retries=3)


class TestAcquireRelease:
    """Basic lock acquire/release lifecycle."""

    @pytest.mark.asyncio
    async def test_acquire_returns_token(self, lock_manager: RedlockManager) -> None:
        token = await lock_manager.acquire("resource:test")
        assert isinstance(token, str)
        assert len(token) > 0

    @pytest.mark.asyncio
    async def test_release_with_correct_token(self, lock_manager: RedlockManager) -> None:
        token = await lock_manager.acquire("resource:test")
        released = await lock_manager.release("resource:test", token)
        assert released is True

    @pytest.mark.asyncio
    async def test_release_with_wrong_token(self, lock_manager: RedlockManager) -> None:
        token = await lock_manager.acquire("resource:test")
        released = await lock_manager.release("resource:test", "wrong-token")
        assert released is False
        # Original token should still work
        released = await lock_manager.release("resource:test", token)
        assert released is True

    @pytest.mark.asyncio
    async def test_acquire_different_resources(self, lock_manager: RedlockManager) -> None:
        token1 = await lock_manager.acquire("resource:a")
        token2 = await lock_manager.acquire("resource:b")
        assert token1 != token2
        await lock_manager.release("resource:a", token1)
        await lock_manager.release("resource:b", token2)


class TestDoubleAcquire:
    """Contention: second acquire on same resource."""

    @pytest.mark.asyncio
    async def test_double_acquire_timeout(self, lock_manager: RedlockManager) -> None:
        """Second acquire should fail with PICPLockTimeout when resource is held."""
        await lock_manager.acquire("resource:contested")
        with pytest.raises(PICPLockTimeout):
            await lock_manager.acquire("resource:contested")

    @pytest.mark.asyncio
    async def test_acquire_after_release(self, lock_manager: RedlockManager) -> None:
        """After release, a new acquire should succeed."""
        token1 = await lock_manager.acquire("resource:reuse")
        await lock_manager.release("resource:reuse", token1)
        token2 = await lock_manager.acquire("resource:reuse")
        assert isinstance(token2, str)
        await lock_manager.release("resource:reuse", token2)


class TestExtend:
    """Lock TTL extension."""

    @pytest.mark.asyncio
    async def test_extend_with_correct_token(self, lock_manager: RedlockManager) -> None:
        token = await lock_manager.acquire("resource:extend")
        extended = await lock_manager.extend("resource:extend", token)
        assert extended is True
        await lock_manager.release("resource:extend", token)

    @pytest.mark.asyncio
    async def test_extend_with_wrong_token(self, lock_manager: RedlockManager) -> None:
        await lock_manager.acquire("resource:extend")
        extended = await lock_manager.extend("resource:extend", "wrong-token")
        assert extended is False


class TestTimeout:
    """Lock timeout parameters."""

    @pytest.mark.asyncio
    async def test_custom_ttl(self) -> None:
        """Lock with very short retry should timeout quickly."""
        mgr = RedlockManager(redis=None, ttl_ms=100, retry_delay_ms=5, max_retries=2)
        await mgr.acquire("resource:timeout")
        with pytest.raises(PICPLockTimeout) as exc_info:
            await mgr.acquire("resource:timeout")
        assert exc_info.value.resource == "resource:timeout"

    @pytest.mark.asyncio
    async def test_timeout_error_attributes(self) -> None:
        mgr = RedlockManager(redis=None, ttl_ms=500, retry_delay_ms=5, max_retries=1)
        await mgr.acquire("resource:attrs")
        with pytest.raises(PICPLockTimeout) as exc_info:
            await mgr.acquire("resource:attrs")
        err = exc_info.value
        assert err.resource == "resource:attrs"
        assert err.ttl_ms == 500
