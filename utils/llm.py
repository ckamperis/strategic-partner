"""LLM abstraction layer — provider-agnostic Strategy pattern.

Provides a unified interface for text completions and embeddings.
OpenAI selected for PoC due to unified completions+embeddings API,
cost-effectiveness (gpt-4o-mini), and mature SDK.  Architecture
supports provider swap without code changes.

References:
    Thesis Chapter 4 — Implementation, LLM Abstraction Layer
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import structlog

from config.settings import Settings, get_settings

logger = structlog.get_logger()


class LLMClient(ABC):
    """Abstract base for LLM providers.

    All pillars use this interface — never call OpenAI directly.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The user message / prompt.
            system: Optional system message.
            model: Override the default model.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            The model's text response.
        """
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of strings to embed (max 100 per call).

        Returns:
            List of embedding vectors (each 1536-dim for text-embedding-3-small).
        """
        ...


class OpenAIClient(LLMClient):
    """OpenAI API client with exponential backoff and structured logging.

    Uses two model tiers:
    - fast: gpt-4o-mini (cheap, low latency — routing, relevance judging)
    - strong: gpt-4o (higher quality — analysis, synthesis)

    Attributes:
        settings: Application settings.
        _client: The openai.AsyncOpenAI instance.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        import openai

        self._client = openai.AsyncOpenAI(api_key=self._settings.openai_api_key)
        self._max_retries = 3
        self._base_delay = 1.0  # seconds

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate completion with exponential backoff on rate limits."""
        model = model or self._settings.llm_model_fast
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(1, self._max_retries + 1):
            start = time.perf_counter()
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

                usage = response.usage
                logger.info(
                    "llm.complete",
                    model=model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=elapsed_ms,
                    attempt=attempt,
                )

                return response.choices[0].message.content or ""

            except Exception as e:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                delay = self._base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "llm.complete.retry",
                    model=model,
                    attempt=attempt,
                    error=str(e),
                    retry_delay_s=delay,
                    elapsed_ms=elapsed_ms,
                )
                if attempt == self._max_retries:
                    raise
                await asyncio.sleep(delay)

        return ""  # unreachable, satisfies type checker

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings in batches of up to 100 texts."""
        all_embeddings: list[list[float]] = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            for attempt in range(1, self._max_retries + 1):
                start = time.perf_counter()
                try:
                    response = await self._client.embeddings.create(
                        model=self._settings.embedding_model,
                        input=batch,
                    )
                    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

                    logger.info(
                        "llm.embed",
                        model=self._settings.embedding_model,
                        batch_size=len(batch),
                        batch_index=i // batch_size,
                        latency_ms=elapsed_ms,
                    )

                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    delay = self._base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "llm.embed.retry",
                        attempt=attempt,
                        error=str(e),
                        retry_delay_s=delay,
                    )
                    if attempt == self._max_retries:
                        raise
                    await asyncio.sleep(delay)

        return all_embeddings


class MockLLMClient(LLMClient):
    """Deterministic mock LLM for testing without API keys.

    - ``complete()`` returns deterministic JSON based on keywords in prompt.
    - ``embed()`` returns deterministic 1536-dim vectors using numpy RandomState(seed=42).
    """

    def __init__(self, embedding_dim: int = 1536) -> None:
        self._embedding_dim = embedding_dim
        self._rng = np.random.RandomState(42)
        self._call_count = 0

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return deterministic JSON responses based on prompt keywords."""
        self._call_count += 1
        prompt_lower = prompt.lower()

        # Relevance judging (used by SelfCorrectingRAG)
        if "relevance" in prompt_lower or "rate overall" in prompt_lower:
            # Simulate improving scores on successive calls
            score = min(0.5 + self._call_count * 0.2, 0.95)
            return (
                f'{{"score": {score:.2f}, "reasoning": "Mock relevance assessment", '
                f'"refined_query": "refined: {prompt[:50]}"}}'
            )

        # Cashflow / financial analysis
        if "cashflow" in prompt_lower or "forecast" in prompt_lower:
            return (
                '{"analysis": "Mock cashflow analysis based on ERP data", '
                '"inflows": 150000.0, "outflows": 95000.0, "net": 55000.0, '
                '"confidence": 0.75}'
            )

        # SWOT / risk assessment
        if "swot" in prompt_lower or "risk" in prompt_lower:
            return (
                '{"strengths": ["Diverse customer base"], '
                '"weaknesses": ["High customer concentration"], '
                '"opportunities": ["Market expansion"], '
                '"threats": ["Economic downturn"]}'
            )

        # Default structured response
        return (
            '{"response": "Mock LLM response", '
            f'"prompt_length": {len(prompt)}, '
            f'"call_number": {self._call_count}}}'
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings seeded by text content."""
        embeddings: list[list[float]] = []
        for text in texts:
            # Seed based on text hash for determinism per unique text
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._embedding_dim).astype(np.float64)
            # Normalize to unit length (cosine similarity friendly)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings

    def reset(self) -> None:
        """Reset call counter (useful between tests)."""
        self._call_count = 0


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Factory function — returns the configured LLM client.

    Args:
        settings: Application settings. Uses defaults if None.

    Returns:
        An LLMClient instance based on ``settings.llm_provider``.
    """
    s = settings or get_settings()

    if s.llm_provider == "mock":
        logger.info("llm.factory", provider="mock")
        return MockLLMClient(embedding_dim=s.embedding_dimensions)

    if s.llm_provider == "openai":
        logger.info("llm.factory", provider="openai")
        return OpenAIClient(settings=s)

    raise ValueError(f"Unknown LLM provider: {s.llm_provider}")
