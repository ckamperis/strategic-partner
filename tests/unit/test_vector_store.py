"""Unit tests for pillars.knowledge.vector_store — VectorRetriever.

Uses MockLLMClient for embeddings (no API keys needed).

Tests cover:
- Ingest: correct number of documents stored
- Retrieve: returns top-k with scores
- Empty collection handling
"""

from __future__ import annotations

import pytest

from data.pipeline.models import TextChunk
from pillars.knowledge.vector_store import VectorRetriever
from utils.llm import MockLLMClient


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient(embedding_dim=1536)


@pytest.fixture
def sample_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            text="Μηνιαία σύνοψη 2023-01: Πωλήσεις €150,000",
            metadata={"year": 2023, "month": 1},
            chunk_id="monthly_2023-01",
            chunk_type="monthly_summary",
        ),
        TextChunk(
            text="Μηνιαία σύνοψη 2023-02: Πωλήσεις €120,000",
            metadata={"year": 2023, "month": 2},
            chunk_id="monthly_2023-02",
            chunk_type="monthly_summary",
        ),
        TextChunk(
            text="Ανάλυση συγκέντρωσης πελατών: Top 5% = 45% εσόδων",
            metadata={"total_customers": 100},
            chunk_id="customer_concentration",
            chunk_type="customer_analysis",
        ),
        TextChunk(
            text="Εποχιακοί δείκτες πωλήσεων: Ιαν=0.85, Φεβ=0.90",
            metadata={},
            chunk_id="seasonal_patterns",
            chunk_type="seasonal",
        ),
        TextChunk(
            text="Παράγοντες κινδύνου: Ποσοστό πιστωτικών 8.5%",
            metadata={},
            chunk_id="risk_factors",
            chunk_type="risk",
        ),
    ]


class TestVectorRetrieverIngest:
    """Ingestion into ChromaDB."""

    @pytest.mark.asyncio
    async def test_ingest_returns_count(
        self, mock_llm: MockLLMClient, sample_chunks: list[TextChunk]
    ) -> None:
        retriever = VectorRetriever(mock_llm, collection_name="test_ingest")
        count = await retriever.ingest(sample_chunks)
        assert count == 5

    @pytest.mark.asyncio
    async def test_ingest_updates_collection_count(
        self, mock_llm: MockLLMClient, sample_chunks: list[TextChunk]
    ) -> None:
        retriever = VectorRetriever(mock_llm, collection_name="test_count")
        await retriever.ingest(sample_chunks)
        assert retriever.count == 5

    @pytest.mark.asyncio
    async def test_ingest_idempotent(
        self, mock_llm: MockLLMClient, sample_chunks: list[TextChunk]
    ) -> None:
        """Upserting same chunks twice should not duplicate."""
        retriever = VectorRetriever(mock_llm, collection_name="test_idempotent")
        await retriever.ingest(sample_chunks)
        await retriever.ingest(sample_chunks)
        assert retriever.count == 5


class TestVectorRetrieverRetrieve:
    """Retrieval from ChromaDB."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_results(
        self, mock_llm: MockLLMClient, sample_chunks: list[TextChunk]
    ) -> None:
        retriever = VectorRetriever(mock_llm, collection_name="test_retrieve")
        await retriever.ingest(sample_chunks)
        results = await retriever.retrieve("πωλήσεις Ιανουάριος", k=3)
        assert len(results) == 3
        assert all(r.chunk_text for r in results)
        assert all(isinstance(r.score, float) for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_respects_k(
        self, mock_llm: MockLLMClient, sample_chunks: list[TextChunk]
    ) -> None:
        retriever = VectorRetriever(mock_llm, collection_name="test_k")
        await retriever.ingest(sample_chunks)
        results = await retriever.retrieve("test", k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_empty_collection(self, mock_llm: MockLLMClient) -> None:
        retriever = VectorRetriever(mock_llm, collection_name="test_empty")
        results = await retriever.retrieve("anything", k=3)
        assert len(results) == 0
