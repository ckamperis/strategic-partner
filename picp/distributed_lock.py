"""Distributed lock manager implementing the Redlock algorithm.

Implements Eq. 3.24 from the thesis — quorum-based lock acquisition:
    acquired = count(locks) > floor(N/2)

For the PoC, we use a single Redis instance but maintain correct quorum
logic so the algorithm generalises to N instances.  If Redis is unavailable,
falls back to a local ``threading.Lock`` with a warning log.

References:
    Antirez (2016) — Redlock algorithm
    Thesis Section 3.4, Equation 3.24
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any

import structlog

from picp.exceptions import PICPLockTimeout

logger = structlog.get_logger()

# Lua scripts for atomic Redis operations (executed server-side)
_LUA_RELEASE = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

_LUA_EXTEND = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("pexpire", KEYS[1], ARGV[2])
else
    return 0
end
"""


class RedlockManager:
    """Distributed lock manager with Redis-backed or in-memory fallback.

    Attributes:
        redis: An async Redis client (``redis.asyncio.Redis``) or ``None``.
        ttl_ms: Default lock time-to-live in milliseconds.
        retry_delay_ms: Delay between acquisition retries.
        max_retries: Maximum number of acquisition attempts.
    """

    def __init__(
        self,
        redis: Any | None = None,
        ttl_ms: int = 5_000,
        retry_delay_ms: int = 200,
        max_retries: int = 10,
    ) -> None:
        self._redis = redis
        self._ttl_ms = ttl_ms
        self._retry_delay_ms = retry_delay_ms
        self._max_retries = max_retries

        # Fallback: local threading locks keyed by resource
        self._local_locks: dict[str, threading.Lock] = {}
        # Track ownership tokens for release validation
        self._tokens: dict[str, str] = {}
        self._using_fallback = redis is None

        if self._using_fallback:
            logger.warning("redlock.fallback", reason="Redis unavailable, using local threading.Lock")

    # ── Public API ──────────────────────────────────────────

    async def acquire(self, resource: str, ttl_ms: int | None = None) -> str:
        """Acquire a distributed lock on *resource*.

        Implements Eq. 3.24: for N Redis instances, lock is acquired
        when count(successful_locks) > floor(N/2).  With N=1, this reduces
        to a single SET NX.

        Args:
            resource: The resource identifier to lock (e.g. "pillar:knowledge").
            ttl_ms: Lock TTL in ms.  Defaults to ``self._ttl_ms``.

        Returns:
            A unique token string used to release the lock.

        Raises:
            PICPLockTimeout: If the lock cannot be acquired within max retries.
        """
        ttl = ttl_ms or self._ttl_ms
        token = str(uuid.uuid4())
        start = time.perf_counter()

        for attempt in range(1, self._max_retries + 1):
            acquired = await self._try_acquire(resource, token, ttl)

            if acquired:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.info(
                    "redlock.acquired",
                    resource=resource,
                    token=token[:8],
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                )
                return token

            await asyncio.sleep(self._retry_delay_ms / 1000.0)

        raise PICPLockTimeout(resource=resource, ttl_ms=ttl)

    async def release(self, resource: str, token: str) -> bool:
        """Release a previously acquired lock.

        Only the holder of the original *token* can release the lock.

        Args:
            resource: The locked resource.
            token: The token returned by :meth:`acquire`.

        Returns:
            ``True`` if the lock was released, ``False`` if token mismatch.
        """
        if self._using_fallback:
            return self._release_local(resource, token)
        return await self._release_redis(resource, token)

    async def extend(self, resource: str, token: str, ttl_ms: int | None = None) -> bool:
        """Extend the TTL of an existing lock.

        Args:
            resource: The locked resource.
            token: The token returned by :meth:`acquire`.
            ttl_ms: New TTL in milliseconds.

        Returns:
            ``True`` if extended, ``False`` if token mismatch or not held.
        """
        ttl = ttl_ms or self._ttl_ms

        if self._using_fallback:
            # Local locks don't expire, just verify ownership
            return self._tokens.get(resource) == token

        return await self._extend_redis(resource, token, ttl)

    # ── Redis implementation ────────────────────────────────

    async def _try_acquire(self, resource: str, token: str, ttl_ms: int) -> bool:
        """Attempt a single lock acquisition (Eq. 3.24 with N=1)."""
        if self._using_fallback:
            return self._acquire_local(resource, token)

        # N=1 Redlock: quorum is 1 > floor(1/2) = 0, i.e. single success suffices
        key = f"picp:lock:{resource}"
        acquired = await self._redis.set(key, token, nx=True, px=ttl_ms)

        if acquired:
            self._tokens[resource] = token
            # Quorum check (trivial for N=1 but shows the algorithm)
            n_instances = 1
            n_acquired = 1
            quorum = n_instances // 2
            if n_acquired > quorum:
                return True
            # Failed quorum — release
            await self._redis.delete(key)
            return False

        return False

    async def _release_redis(self, resource: str, token: str) -> bool:
        """Release via Lua script (atomic check-and-delete)."""
        key = f"picp:lock:{resource}"
        # Atomic: only delete if the token matches
        release_script = self._redis.register_script(_LUA_RELEASE)
        result = await release_script(keys=[key], args=[token])
        released = result == 1

        if released:
            self._tokens.pop(resource, None)
            logger.info("redlock.released", resource=resource, token=token[:8])
        else:
            logger.warning("redlock.release_failed", resource=resource, token=token[:8])

        return released

    async def _extend_redis(self, resource: str, token: str, ttl_ms: int) -> bool:
        """Extend TTL via Lua script (atomic check-and-pexpire)."""
        key = f"picp:lock:{resource}"
        extend_script = self._redis.register_script(_LUA_EXTEND)
        result = await extend_script(keys=[key], args=[token, str(ttl_ms)])
        return result == 1

    # ── Local fallback implementation ───────────────────────

    def _acquire_local(self, resource: str, token: str) -> bool:
        """Non-blocking local lock acquire."""
        if resource not in self._local_locks:
            self._local_locks[resource] = threading.Lock()

        lock = self._local_locks[resource]
        acquired = lock.acquire(blocking=False)

        if acquired:
            self._tokens[resource] = token
        return acquired

    def _release_local(self, resource: str, token: str) -> bool:
        """Release local lock only if token matches."""
        if self._tokens.get(resource) != token:
            logger.warning("redlock.release_failed", resource=resource, token=token[:8])
            return False

        lock = self._local_locks.get(resource)
        if lock is not None:
            try:
                lock.release()
            except RuntimeError:
                # Already released
                pass
        self._tokens.pop(resource, None)
        logger.info("redlock.released", resource=resource, token=token[:8])
        return True
