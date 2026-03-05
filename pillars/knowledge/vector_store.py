"""ChromaDB-backed vector store for Knowledge Pillar retrieval.

Stores embedded text chunks from the ERP data pipeline and
retrieves the most similar chunks for a given query using
cosine similarity.

References:
    Thesis Chapter 4 — Knowledge Pillar, Vector Store
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from config.settings import Settings, get_settings
from data.pipeline.models import TextChunk
from utils.llm import LLMClient

logger = structlog.get_logger()


class RetrievalResult:
    """A single retrieval result from the vector store.

    Attributes:
        chunk_text: The text content of the chunk.
        score: Cosine similarity score (higher = more similar).
        metadata: Chunk metadata (month, type, etc.).
        chunk_id: The unique chunk identifier.
    """

    def __init__(
        self,
        chunk_text: str,
        score: float,
        metadata: dict[str, Any],
        chunk_id: str = "",
    ) -> None:
        self.chunk_text = chunk_text
        self.score = score
        self.metadata = metadata
        self.chunk_id = chunk_id

    def __repr__(self) -> str:
        return f"RetrievalResult(id={self.chunk_id!r}, score={self.score:.4f})"


class VectorRetriever:
    """ChromaDB-backed vector retriever.

    Args:
        llm_client: The LLM client used for generating embeddings.
        collection_name: ChromaDB collection name.
        persist_dir: Directory for ChromaDB persistent storage.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        collection_name: str = "erp_knowledge",
        persist_dir: str | None = None,
    ) -> None:
        self._llm = llm_client
        self._collection_name = collection_name

        settings = get_settings()
        self._persist_dir = persist_dir or settings.chroma_persist_dir

        # Lazy-init ChromaDB
        self._client: Any = None
        self._collection: Any = None

    def _ensure_collection(self) -> None:
        """Lazily initialise ChromaDB client and collection."""
        if self._collection is not None:
            return

        import chromadb

        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store.init",
            collection=self._collection_name,
            count=self._collection.count(),
        )

    async def ingest(self, chunks: list[TextChunk]) -> int:
        """Embed and store text chunks in ChromaDB.

        Args:
            chunks: List of TextChunk objects from the data pipeline.

        Returns:
            Number of chunks ingested.
        """
        self._ensure_collection()
        start = time.perf_counter()

        texts = [c.text for c in chunks]
        ids = [c.chunk_id or f"chunk_{i}" for i, c in enumerate(chunks)]
        metadatas = []
        for c in chunks:
            # ChromaDB requires flat string/int/float metadata
            flat_meta: dict[str, Any] = {"chunk_type": c.chunk_type}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_meta[k] = v
            metadatas.append(flat_meta)

        # Generate embeddings
        embeddings = await self._llm.embed(texts)

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "vector_store.ingested",
            count=len(chunks),
            elapsed_ms=elapsed_ms,
        )
        return len(chunks)

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k most similar chunks for a query.

        Args:
            query: The natural-language query string.
            k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by descending similarity.
        """
        self._ensure_collection()
        start = time.perf_counter()

        # Embed the query
        query_embedding = (await self._llm.embed([query]))[0]

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()) if self._collection.count() > 0 else k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to RetrievalResult objects
        retrieval_results: list[RetrievalResult] = []

        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            ids = results["ids"][0] if results["ids"] else [""] * len(docs)

            for doc, meta, dist, chunk_id in zip(docs, metas, distances, ids):
                # ChromaDB returns distance; convert to similarity
                # For cosine space: similarity = 1 - distance
                score = 1.0 - dist
                retrieval_results.append(
                    RetrievalResult(
                        chunk_text=doc,
                        score=score,
                        metadata=meta or {},
                        chunk_id=chunk_id,
                    )
                )

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "vector_store.retrieved",
            query_len=len(query),
            k=k,
            results=len(retrieval_results),
            elapsed_ms=elapsed_ms,
        )

        return retrieval_results

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        self._ensure_collection()
        return self._collection.count()
