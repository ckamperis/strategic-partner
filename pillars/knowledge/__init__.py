"""Knowledge Pillar — Deep Knowledge via RAG + Hybrid Search.

Orchestrates:
- VectorRetriever (ChromaDB cosine similarity)
- HybridSearcher (BM25 + cosine fusion, Eq. 3.17)
- SelfCorrectingRAG (iterative retrieve-evaluate-refine)

References:
    Thesis Section 3.3.1 — Knowledge Pillar
"""

from __future__ import annotations

from typing import Any

from data.pipeline.models import TextChunk
from picp.bus import PICPBus
from picp.message import PICPContext, PICPEvent
from pillars.base import BasePillar
from pillars.knowledge.hybrid_search import HybridSearcher
from pillars.knowledge.rag import SelfCorrectingRAG
from pillars.knowledge.vector_store import VectorRetriever
from utils.llm import LLMClient


class KnowledgePillar(BasePillar):
    """Knowledge Pillar — retrieves relevant ERP context for queries.

    Wires together VectorRetriever + HybridSearcher + SelfCorrectingRAG.
    Integrates with PICP via BasePillar lifecycle hooks.

    Args:
        bus: The PICP event bus.
        llm_client: LLM client for embeddings and relevance judging.
        collection_name: ChromaDB collection name.
    """

    def __init__(
        self,
        bus: PICPBus,
        llm_client: LLMClient,
        collection_name: str = "erp_knowledge",
    ) -> None:
        super().__init__(
            name="knowledge",
            bus=bus,
            start_event=PICPEvent.KNOWLEDGE_STARTED,
            complete_event=PICPEvent.KNOWLEDGE_UPDATED,
        )
        self._llm = llm_client
        self._retriever = VectorRetriever(llm_client, collection_name=collection_name)
        self._searcher = HybridSearcher(self._retriever)
        self._rag = SelfCorrectingRAG(self._searcher, llm_client)

    async def ingest(self, chunks: list[TextChunk]) -> int:
        """Ingest text chunks into both vector store and BM25 index.

        Args:
            chunks: Text chunks from the data pipeline.

        Returns:
            Number of chunks ingested.
        """
        count = await self._retriever.ingest(chunks)
        self._searcher.build_bm25_index(chunks)
        return count

    async def _execute(self, context: PICPContext, **kwargs: Any) -> dict[str, Any]:
        """Execute the Knowledge Pillar retrieval pipeline.

        Args:
            context: The PICP context with the user's query.

        Returns:
            Dict with RAG results: chunks, score, iterations, timing.
        """
        rag_result = await self._rag.retrieve(context.query)
        return rag_result.to_dict()
