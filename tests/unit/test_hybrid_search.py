"""Unit tests for pillars.knowledge.hybrid_search — HybridSearcher.

Tests cover:
- Min-max normalisation
- Fusion math (Eq. 3.17)
- Alpha extremes: α=0 -> pure BM25, α=1 -> pure cosine
- Ranking correctness
- End-to-end search with mock LLM
"""

from __future__ import annotations

import pytest

from data.pipeline.models import TextChunk
from pillars.knowledge.hybrid_search import HybridSearcher, min_max_normalize
from pillars.knowledge.vector_store import VectorRetriever
from utils.llm import MockLLMClient


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient(embedding_dim=1536)


@pytest.fixture
def chunks() -> list[TextChunk]:
    return [
        TextChunk(text="Πωλήσεις Ιανουαρίου 2023 ήταν €150,000", chunk_id="jan", chunk_type="monthly_summary"),
        TextChunk(text="Πωλήσεις Φεβρουαρίου 2023 ήταν €120,000", chunk_id="feb", chunk_type="monthly_summary"),
        TextChunk(text="Ανάλυση πελατών: υψηλή συγκέντρωση εσόδων", chunk_id="cust", chunk_type="customer_analysis"),
        TextChunk(text="Ταμειακή ροή και πληρωμές τρίτου τριμήνου", chunk_id="cf", chunk_type="monthly_summary"),
        TextChunk(text="Εποχιακοί δείκτες: κορυφή Δεκέμβριος", chunk_id="season", chunk_type="seasonal"),
    ]


class TestMinMaxNormalize:
    """Min-max normalisation utility."""

    def test_basic(self) -> None:
        result = min_max_normalize([1.0, 2.0, 3.0])
        assert result == [0.0, 0.5, 1.0]

    def test_single_value(self) -> None:
        result = min_max_normalize([5.0])
        assert result == [0.0]

    def test_all_same(self) -> None:
        result = min_max_normalize([3.0, 3.0, 3.0])
        assert result == [0.0, 0.0, 0.0]

    def test_empty(self) -> None:
        result = min_max_normalize([])
        assert result == []

    def test_negative_values(self) -> None:
        result = min_max_normalize([-2.0, 0.0, 2.0])
        assert result[0] == 0.0
        assert result[2] == 1.0

    def test_output_in_01(self) -> None:
        result = min_max_normalize([10.0, 20.0, 50.0, 100.0])
        assert all(0.0 <= v <= 1.0 for v in result)


class TestHybridSearcher:
    """End-to-end hybrid search with mock LLM."""

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, mock_llm: MockLLMClient, chunks: list[TextChunk]
    ) -> None:
        retriever = VectorRetriever(mock_llm, collection_name="test_hybrid")
        await retriever.ingest(chunks)

        searcher = HybridSearcher(retriever, alpha=0.7)
        searcher.build_bm25_index(chunks)

        results = await searcher.search("πωλήσεις Ιανουαρίου", k=3)
        assert len(results) <= 3
        assert all(r.fused_score >= 0 for r in results)

    @pytest.mark.asyncio
    async def test_search_ordered_by_fused_score(
        self, mock_llm: MockLLMClient, chunks: list[TextChunk]
    ) -> None:
        retriever = VectorRetriever(mock_llm, collection_name="test_order")
        await retriever.ingest(chunks)

        searcher = HybridSearcher(retriever, alpha=0.7)
        searcher.build_bm25_index(chunks)

        results = await searcher.search("πωλήσεις", k=5)
        scores = [r.fused_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_alpha_zero_pure_bm25(
        self, mock_llm: MockLLMClient, chunks: list[TextChunk]
    ) -> None:
        """α=0 -> fused = bm25 only."""
        retriever = VectorRetriever(mock_llm, collection_name="test_alpha0")
        await retriever.ingest(chunks)

        searcher = HybridSearcher(retriever, alpha=0.0)
        searcher.build_bm25_index(chunks)

        results = await searcher.search("πωλήσεις", k=3)
        for r in results:
            assert r.fused_score == pytest.approx(r.bm25_score, abs=1e-5)

    @pytest.mark.asyncio
    async def test_alpha_one_pure_cosine(
        self, mock_llm: MockLLMClient, chunks: list[TextChunk]
    ) -> None:
        """α=1 -> fused = cosine only."""
        retriever = VectorRetriever(mock_llm, collection_name="test_alpha1")
        await retriever.ingest(chunks)

        searcher = HybridSearcher(retriever, alpha=1.0)
        searcher.build_bm25_index(chunks)

        results = await searcher.search("πωλήσεις", k=3)
        for r in results:
            assert r.fused_score == pytest.approx(r.cosine_score, abs=1e-5)

    @pytest.mark.asyncio
    async def test_fusion_formula(
        self, mock_llm: MockLLMClient, chunks: list[TextChunk]
    ) -> None:
        """Verify Eq. 3.17: S = α·cos + (1-α)·bm25."""
        retriever = VectorRetriever(mock_llm, collection_name="test_fusion")
        await retriever.ingest(chunks)

        alpha = 0.7
        searcher = HybridSearcher(retriever, alpha=alpha)
        searcher.build_bm25_index(chunks)

        results = await searcher.search("ταμειακή ροή", k=5)
        for r in results:
            expected = alpha * r.cosine_score + (1 - alpha) * r.bm25_score
            assert r.fused_score == pytest.approx(expected, abs=1e-5)
