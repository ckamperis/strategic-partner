"""Hybrid search combining BM25 and cosine similarity.

Implements Eq. 3.17 from the thesis:
    S = α · cos_sim_norm + (1 - α) · bm25_norm

where both scores are min-max normalised to [0, 1] before fusion.

BM25 provides exact keyword matching (good for Greek business terms),
while cosine similarity captures semantic similarity.  The fusion
parameter α controls the balance (default 0.7 = cosine-heavy).

References:
    Thesis Section 3.3.1, Equation 3.17
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from config.settings import get_settings
from data.pipeline.models import TextChunk
from pillars.knowledge.vector_store import RetrievalResult, VectorRetriever

logger = structlog.get_logger()


class SearchResult:
    """A fused search result combining BM25 and cosine scores.

    Attributes:
        chunk_text: The text content.
        fused_score: The combined score (Eq. 3.17).
        cosine_score: Normalised cosine similarity score.
        bm25_score: Normalised BM25 score.
        metadata: Chunk metadata.
        chunk_id: Unique chunk identifier.
    """

    def __init__(
        self,
        chunk_text: str,
        fused_score: float,
        cosine_score: float,
        bm25_score: float,
        metadata: dict[str, Any],
        chunk_id: str = "",
    ) -> None:
        self.chunk_text = chunk_text
        self.fused_score = fused_score
        self.cosine_score = cosine_score
        self.bm25_score = bm25_score
        self.metadata = metadata
        self.chunk_id = chunk_id

    def __repr__(self) -> str:
        return (
            f"SearchResult(id={self.chunk_id!r}, fused={self.fused_score:.4f}, "
            f"cos={self.cosine_score:.4f}, bm25={self.bm25_score:.4f})"
        )


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for Greek text."""
    # Lowercase and split on non-alphanumeric (preserving Greek chars)
    import re

    tokens = re.findall(r"[\w]+", text.lower(), re.UNICODE)
    return tokens


def min_max_normalize(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1].

    If all scores are identical, returns all 0.0.

    Args:
        scores: Raw scores to normalise.

    Returns:
        Normalised scores in [0, 1].
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)
    span = max_s - min_s

    if span == 0:
        return [0.0] * len(scores)

    return [(s - min_s) / span for s in scores]


class HybridSearcher:
    """Hybrid BM25 + cosine similarity search engine.

    Builds a BM25 index at ingestion time alongside the vector store.
    At query time, retrieves candidates from both and fuses scores.

    Args:
        vector_retriever: The ChromaDB-backed vector retriever.
        alpha: Fusion weight for cosine similarity (Eq. 3.17). Default 0.7.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        alpha: float | None = None,
    ) -> None:
        self._retriever = vector_retriever
        self._alpha = alpha if alpha is not None else get_settings().hybrid_search_alpha

        # BM25 index (built at ingestion time)
        self._bm25: BM25Okapi | None = None
        self._corpus_texts: list[str] = []
        self._corpus_ids: list[str] = []
        self._corpus_metadata: list[dict[str, Any]] = []

    def build_bm25_index(self, chunks: list[TextChunk]) -> None:
        """Build the BM25 index from text chunks.

        Called once after data ingestion, before any queries.

        Args:
            chunks: The same chunks ingested into the vector store.
        """
        start = time.perf_counter()

        self._corpus_texts = [c.text for c in chunks]
        self._corpus_ids = [c.chunk_id or f"chunk_{i}" for i, c in enumerate(chunks)]
        self._corpus_metadata = [c.metadata for c in chunks]

        tokenized = [_tokenize(text) for text in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info("hybrid_search.bm25_built", corpus_size=len(chunks), elapsed_ms=elapsed_ms)

    async def search(
        self,
        query: str,
        k: int = 5,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """Execute hybrid search combining BM25 and cosine similarity.

        Implements Eq. 3.17: S = α · cos_norm + (1 - α) · bm25_norm

        Args:
            query: Natural-language query string.
            k: Number of results to return.
            alpha: Override fusion weight (default: self._alpha).

        Returns:
            List of SearchResult sorted by fused_score descending.
        """
        start = time.perf_counter()
        a = alpha if alpha is not None else self._alpha

        # Get cosine similarity results from vector store
        cosine_results = await self._retriever.retrieve(query, k=k * 2)

        # Get BM25 scores for ALL corpus documents
        bm25_scores_raw = self._get_bm25_scores(query)

        # Build a unified candidate set
        # Map chunk_id -> (cosine_score, bm25_score, text, metadata)
        candidates: dict[str, dict[str, Any]] = {}

        # Add cosine results
        for r in cosine_results:
            candidates[r.chunk_id] = {
                "text": r.chunk_text,
                "cosine_raw": r.score,
                "bm25_raw": 0.0,
                "metadata": r.metadata,
            }

        # Add/update with BM25 scores
        for i, (text, chunk_id) in enumerate(zip(self._corpus_texts, self._corpus_ids)):
            if chunk_id in candidates:
                candidates[chunk_id]["bm25_raw"] = bm25_scores_raw[i]
            elif bm25_scores_raw[i] > 0:
                candidates[chunk_id] = {
                    "text": text,
                    "cosine_raw": 0.0,
                    "bm25_raw": bm25_scores_raw[i],
                    "metadata": self._corpus_metadata[i] if i < len(self._corpus_metadata) else {},
                }

        if not candidates:
            return []

        # Normalise both score dimensions
        ids = list(candidates.keys())
        cosine_raw = [candidates[cid]["cosine_raw"] for cid in ids]
        bm25_raw = [candidates[cid]["bm25_raw"] for cid in ids]

        cosine_norm = min_max_normalize(cosine_raw)
        bm25_norm = min_max_normalize(bm25_raw)

        # Fuse: Eq. 3.17
        results: list[SearchResult] = []
        for i, cid in enumerate(ids):
            fused = a * cosine_norm[i] + (1 - a) * bm25_norm[i]
            results.append(SearchResult(
                chunk_text=candidates[cid]["text"],
                fused_score=round(fused, 6),
                cosine_score=round(cosine_norm[i], 6),
                bm25_score=round(bm25_norm[i], 6),
                metadata=candidates[cid]["metadata"],
                chunk_id=cid,
            ))

        # Sort descending by fused score
        results.sort(key=lambda r: r.fused_score, reverse=True)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "hybrid_search.complete",
            query_len=len(query),
            k=k,
            candidates=len(candidates),
            alpha=a,
            elapsed_ms=elapsed_ms,
        )

        return results[:k]

    def _get_bm25_scores(self, query: str) -> list[float]:
        """Get BM25 scores for all corpus documents.

        Args:
            query: The search query.

        Returns:
            List of BM25 scores (one per corpus document).
        """
        if self._bm25 is None or not self._corpus_texts:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return [0.0] * len(self._corpus_texts)

        return self._bm25.get_scores(tokens).tolist()
