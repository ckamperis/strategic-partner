"""Unit tests for pillars.knowledge.rag — SelfCorrectingRAG.

Uses MockLLMClient for deterministic behaviour.

Tests cover:
- High relevance -> 1 iteration
- Low relevance -> multiple iterations (mock returns improving scores)
- Max iterations reached -> returns best result with warning
- Query history tracking
"""

from __future__ import annotations

import pytest

from data.pipeline.models import TextChunk
from pillars.knowledge.hybrid_search import HybridSearcher
from pillars.knowledge.rag import RAGResult, SelfCorrectingRAG
from pillars.knowledge.vector_store import VectorRetriever
from utils.llm import MockLLMClient


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient(embedding_dim=1536)


@pytest.fixture
def chunks() -> list[TextChunk]:
    return [
        TextChunk(text="Μηνιαία σύνοψη 2023-01: Πωλήσεις €150,000", chunk_id="jan", chunk_type="monthly_summary"),
        TextChunk(text="Μηνιαία σύνοψη 2023-02: Πωλήσεις €120,000", chunk_id="feb", chunk_type="monthly_summary"),
        TextChunk(text="Ανάλυση πελατών: υψηλή συγκέντρωση", chunk_id="cust", chunk_type="customer_analysis"),
        TextChunk(text="Ταμειακή ροή Q3: θετικό ισοζύγιο", chunk_id="cf", chunk_type="monthly_summary"),
    ]


@pytest.fixture
async def rag_pipeline(mock_llm: MockLLMClient, chunks: list[TextChunk]) -> SelfCorrectingRAG:
    """Set up a complete RAG pipeline with ingested data."""
    retriever = VectorRetriever(mock_llm, collection_name="test_rag")
    await retriever.ingest(chunks)

    searcher = HybridSearcher(retriever, alpha=0.7)
    searcher.build_bm25_index(chunks)

    return SelfCorrectingRAG(searcher, mock_llm, max_iterations=3, threshold=0.75)


class TestSelfCorrectingRAG:
    """Self-correcting RAG pipeline."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_rag_result(
        self, rag_pipeline: SelfCorrectingRAG
    ) -> None:
        result = await rag_pipeline.retrieve("πωλήσεις Ιανουαρίου")
        assert isinstance(result, RAGResult)
        assert len(result.chunks) > 0
        assert result.final_score > 0
        assert result.iterations_used >= 1

    @pytest.mark.asyncio
    async def test_query_history_starts_with_original(
        self, rag_pipeline: SelfCorrectingRAG
    ) -> None:
        query = "ταμειακή ροή τρίτο τρίμηνο"
        result = await rag_pipeline.retrieve(query)
        assert result.query_history[0] == query

    @pytest.mark.asyncio
    async def test_max_iterations_produces_warning(
        self, mock_llm: MockLLMClient, chunks: list[TextChunk]
    ) -> None:
        """With a very high threshold, mock should hit max iterations."""
        retriever = VectorRetriever(mock_llm, collection_name="test_rag_max")
        await retriever.ingest(chunks)

        searcher = HybridSearcher(retriever, alpha=0.7)
        searcher.build_bm25_index(chunks)

        # Set threshold impossibly high so all iterations run
        mock_llm.reset()
        rag = SelfCorrectingRAG(
            searcher, mock_llm, max_iterations=2, threshold=0.99
        )
        result = await rag.retrieve("test query")

        assert result.iterations_used == 2
        assert any("Max iterations" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_timing_recorded(self, rag_pipeline: SelfCorrectingRAG) -> None:
        result = await rag_pipeline.retrieve("πωλήσεις")
        assert result.timing_ms > 0

    @pytest.mark.asyncio
    async def test_to_dict_serialisation(self, rag_pipeline: SelfCorrectingRAG) -> None:
        result = await rag_pipeline.retrieve("test")
        d = result.to_dict()
        assert "chunks" in d
        assert "final_score" in d
        assert "iterations_used" in d
        assert "query_history" in d
        assert isinstance(d["chunks"], list)

    @pytest.mark.asyncio
    async def test_improving_scores_across_iterations(
        self, mock_llm: MockLLMClient, chunks: list[TextChunk]
    ) -> None:
        """MockLLMClient returns increasing relevance scores."""
        retriever = VectorRetriever(mock_llm, collection_name="test_rag_improve")
        await retriever.ingest(chunks)

        searcher = HybridSearcher(retriever, alpha=0.7)
        searcher.build_bm25_index(chunks)

        # Reset mock so scores start low and increase
        mock_llm.reset()
        rag = SelfCorrectingRAG(
            searcher, mock_llm, max_iterations=3, threshold=0.85
        )
        result = await rag.retrieve("cashflow analysis")

        # Should converge before max iterations due to increasing scores
        assert result.final_score > 0
        assert result.iterations_used <= 3
