"""Self-correcting RAG with iterative query refinement.

Implements a retrieve -> evaluate -> refine loop (max 3 iterations):
1. Hybrid search retrieves candidate chunks.
2. An LLM judge (gpt-4o-mini) scores relevance 0–1.
3. If score ≥ threshold -> return.
4. Otherwise, the judge suggests a refined query -> loop.

This self-correction mechanism improves retrieval quality for
domain-specific Greek business queries where initial keyword
matching may miss relevant context.

References:
    Thesis Section 3.3.1 — Self-Correcting RAG
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from pillars.knowledge.hybrid_search import HybridSearcher, SearchResult
from utils.llm import LLMClient

logger = structlog.get_logger()

_RELEVANCE_JUDGE_PROMPT = """You are evaluating search results for relevance to a business query.
Query: {query}
Retrieved chunks:
{chunks_text}

Rate overall relevance from 0.0 to 1.0.
Respond ONLY with JSON: {{"score": float, "reasoning": str, "refined_query": str}}
If score < 0.75, suggest a refined_query that would retrieve more relevant results.
If score >= 0.75, set refined_query to empty string."""


class RAGResult:
    """Output of the self-correcting RAG pipeline.

    Attributes:
        chunks: The final retrieved search results.
        final_score: Relevance score of the final retrieval (0–1).
        iterations_used: Number of retrieve-evaluate cycles.
        query_history: List of queries used (original + refined).
        timing_ms: Total time for the RAG pipeline.
        warnings: Any warnings (e.g. max iterations reached).
    """

    def __init__(
        self,
        chunks: list[SearchResult],
        final_score: float,
        iterations_used: int,
        query_history: list[str],
        timing_ms: float,
        warnings: list[str] | None = None,
    ) -> None:
        self.chunks = chunks
        self.final_score = final_score
        self.iterations_used = iterations_used
        self.query_history = query_history
        self.timing_ms = timing_ms
        self.warnings = warnings or []

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage in PICPContext.pillar_results."""
        return {
            "chunks": [
                {
                    "text": c.chunk_text,
                    "score": c.fused_score,
                    "chunk_id": c.chunk_id,
                    "metadata": c.metadata,
                }
                for c in self.chunks
            ],
            "final_score": self.final_score,
            "iterations_used": self.iterations_used,
            "query_history": self.query_history,
            "timing_ms": self.timing_ms,
            "warnings": self.warnings,
        }


class SelfCorrectingRAG:
    """Self-correcting RAG with iterative query refinement.

    Args:
        hybrid_searcher: The hybrid BM25+cosine search engine.
        llm_client: LLM for relevance judging and query refinement.
        max_iterations: Maximum retrieve-evaluate cycles (default 3).
        threshold: Minimum relevance score to accept results (default 0.75).
        k: Number of chunks to retrieve per iteration.
    """

    def __init__(
        self,
        hybrid_searcher: HybridSearcher,
        llm_client: LLMClient,
        max_iterations: int = 3,
        threshold: float = 0.75,
        k: int = 5,
    ) -> None:
        self._searcher = hybrid_searcher
        self._llm = llm_client
        self._max_iterations = max_iterations
        self._threshold = threshold
        self._k = k

    async def retrieve(self, query: str) -> RAGResult:
        """Execute the self-correcting retrieval pipeline.

        Args:
            query: The user's natural-language query.

        Returns:
            RAGResult with best chunks, score, and iteration history.
        """
        start = time.perf_counter()
        current_query = query
        query_history = [query]
        best_chunks: list[SearchResult] = []
        best_score = 0.0
        warnings: list[str] = []

        for iteration in range(1, self._max_iterations + 1):
            # Step 1: Retrieve
            chunks = await self._searcher.search(current_query, k=self._k)

            if not chunks:
                warnings.append(f"Iteration {iteration}: no chunks retrieved")
                break

            # Step 2: Evaluate relevance
            score, reasoning, refined_query = await self._evaluate_relevance(
                current_query, chunks
            )

            logger.info(
                "rag.iteration",
                iteration=iteration,
                query=current_query[:80],
                score=score,
                chunks=len(chunks),
            )

            # Track best result
            if score > best_score:
                best_score = score
                best_chunks = chunks

            # Step 3: Check threshold
            if score >= self._threshold:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                return RAGResult(
                    chunks=best_chunks,
                    final_score=best_score,
                    iterations_used=iteration,
                    query_history=query_history,
                    timing_ms=elapsed_ms,
                )

            # Step 4: Refine query if not last iteration
            if iteration < self._max_iterations and refined_query:
                current_query = refined_query
                query_history.append(refined_query)

        # Max iterations reached — return best result with warning
        warnings.append(
            f"Max iterations ({self._max_iterations}) reached. "
            f"Best score: {best_score:.2f} (threshold: {self._threshold})"
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        return RAGResult(
            chunks=best_chunks,
            final_score=best_score,
            iterations_used=self._max_iterations,
            query_history=query_history,
            timing_ms=elapsed_ms,
            warnings=warnings,
        )

    async def _evaluate_relevance(
        self, query: str, chunks: list[SearchResult]
    ) -> tuple[float, str, str]:
        """Ask the LLM to judge relevance and optionally refine the query.

        Args:
            query: The current search query.
            chunks: Retrieved search results.

        Returns:
            Tuple of (score, reasoning, refined_query).
        """
        chunks_text = "\n---\n".join(
            f"[{i+1}] (score={c.fused_score:.3f}) {c.chunk_text[:500]}"
            for i, c in enumerate(chunks)
        )

        prompt = _RELEVANCE_JUDGE_PROMPT.format(
            query=query, chunks_text=chunks_text
        )

        response_text = await self._llm.complete(prompt)

        # Parse JSON response
        try:
            data = json.loads(response_text)
            score = float(data.get("score", 0.0))
            reasoning = str(data.get("reasoning", ""))
            refined = str(data.get("refined_query", ""))
            return (min(max(score, 0.0), 1.0), reasoning, refined)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("rag.judge_parse_error", response=response_text[:200])
            return (0.5, "Failed to parse judge response", "")
