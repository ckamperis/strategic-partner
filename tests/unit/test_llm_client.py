"""Unit tests for utils.llm — LLM abstraction layer.

Tests cover:
- MockLLMClient: deterministic outputs, embedding dimensions
- Factory function: returns correct client type
- OpenAIClient initialisation (without API call)
"""

from __future__ import annotations

import pytest

from config.settings import Settings
from utils.llm import LLMClient, MockLLMClient, OpenAIClient, get_llm_client


class TestMockLLMClient:
    """MockLLMClient deterministic behaviour."""

    @pytest.fixture
    def mock(self) -> MockLLMClient:
        return MockLLMClient(embedding_dim=1536)

    @pytest.mark.asyncio
    async def test_complete_returns_string(self, mock: MockLLMClient) -> None:
        result = await mock.complete("test prompt")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_complete_relevance_returns_json(self, mock: MockLLMClient) -> None:
        result = await mock.complete("Rate overall relevance")
        assert '"score"' in result
        assert '"reasoning"' in result

    @pytest.mark.asyncio
    async def test_complete_cashflow_returns_json(self, mock: MockLLMClient) -> None:
        result = await mock.complete("cashflow forecast")
        assert '"analysis"' in result

    @pytest.mark.asyncio
    async def test_embed_correct_dimensions(self, mock: MockLLMClient) -> None:
        embeddings = await mock.embed(["hello", "world"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536

    @pytest.mark.asyncio
    async def test_embed_deterministic(self, mock: MockLLMClient) -> None:
        """Same text -> same embedding."""
        e1 = await mock.embed(["test"])
        e2 = await mock.embed(["test"])
        assert e1[0] == e2[0]

    @pytest.mark.asyncio
    async def test_embed_different_texts(self, mock: MockLLMClient) -> None:
        """Different texts -> different embeddings."""
        embeddings = await mock.embed(["alpha", "beta"])
        assert embeddings[0] != embeddings[1]

    @pytest.mark.asyncio
    async def test_embed_normalised(self, mock: MockLLMClient) -> None:
        """Embeddings should be unit-normalised."""
        import numpy as np

        embeddings = await mock.embed(["test"])
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-6

    def test_reset_call_counter(self, mock: MockLLMClient) -> None:
        mock._call_count = 5
        mock.reset()
        assert mock._call_count == 0

    @pytest.mark.asyncio
    async def test_improving_relevance_scores(self, mock: MockLLMClient) -> None:
        """Successive relevance calls should return increasing scores."""
        import json

        mock.reset()
        scores = []
        for _ in range(3):
            result = await mock.complete("Rate overall relevance of search results")
            data = json.loads(result)
            scores.append(data["score"])

        # Scores should increase
        assert scores[1] >= scores[0]
        assert scores[2] >= scores[1]


class TestFactory:
    """get_llm_client factory function."""

    def test_mock_provider(self) -> None:
        settings = Settings(llm_provider="mock")
        client = get_llm_client(settings)
        assert isinstance(client, MockLLMClient)

    def test_openai_provider(self) -> None:
        settings = Settings(llm_provider="openai", openai_api_key="sk-test")
        client = get_llm_client(settings)
        assert isinstance(client, OpenAIClient)

    def test_unknown_provider_raises(self) -> None:
        settings = Settings(llm_provider="unknown")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_client(settings)

    def test_mock_custom_dimensions(self) -> None:
        settings = Settings(llm_provider="mock", embedding_dimensions=768)
        client = get_llm_client(settings)
        assert isinstance(client, MockLLMClient)
        assert client._embedding_dim == 768


class TestOpenAIClientInit:
    """OpenAIClient initialisation (no API calls)."""

    def test_init_stores_settings(self) -> None:
        settings = Settings(openai_api_key="sk-test", llm_model_fast="gpt-4o-mini")
        client = OpenAIClient(settings=settings)
        assert client._settings.llm_model_fast == "gpt-4o-mini"

    def test_is_llm_client(self) -> None:
        settings = Settings(openai_api_key="sk-test")
        client = OpenAIClient(settings=settings)
        assert isinstance(client, LLMClient)
