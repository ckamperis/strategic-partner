"""Reasoning Pillar — Deterministic routing + LLM skill execution.

Orchestrates:
- HeuristicPolicy: deterministic keyword-based query classification
- SkillRegistry: YAML skill definition loader
- SkillExecutor: LLM-driven skill execution with JSON parsing

The Reasoning Pillar receives the knowledge context from the Knowledge Pillar
and routes the query to the appropriate analytical skill. It uses a
deterministic heuristic policy (no DRL) to ensure same query -> same skill
for reproducibility.

Pipeline:
    query + knowledge_context
      -> HeuristicPolicy.classify(query) -> RoutingDecision
      -> SkillRegistry.get_by_query_type() -> SkillDefinition
      -> SkillExecutor.execute(skill, context, query) -> SkillResult
      -> structured reasoning output

References:
    Thesis Section 3.3.2 — Reasoning Pillar
    Thesis Section 4.x — Implementation, Reasoning Architecture
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from picp.bus import PICPBus
from picp.message import PICPContext, PICPEvent
from pillars.base import BasePillar
from pillars.reasoning.heuristic_policy import HeuristicPolicy, QueryType
from pillars.reasoning.skill_executor import SkillExecutor, SkillResult
from pillars.reasoning.skill_registry import SkillRegistry
from utils.llm import LLMClient

logger = structlog.get_logger()


class ReasoningPillar(BasePillar):
    """Reasoning Pillar — classifies queries and executes analytical skills.

    Integrates HeuristicPolicy + SkillRegistry + SkillExecutor with
    PICP lifecycle hooks (vector clock, events, timing).

    Args:
        bus: The PICP event bus.
        llm_client: LLM client for skill execution.
        skills_dir: Path to the YAML skills directory.
        model_fast: Model for "fast" tier skills.
        model_strong: Model for "strong" tier skills.
        min_match_count: Minimum keyword matches for routing (below -> GENERAL).
    """

    def __init__(
        self,
        bus: PICPBus,
        llm_client: LLMClient,
        skills_dir: Path | None = None,
        model_fast: str = "gpt-4o-mini",
        model_strong: str = "gpt-4o",
        min_match_count: int = 1,
    ) -> None:
        super().__init__(
            name="reasoning",
            bus=bus,
            start_event=PICPEvent.REASONING_STARTED,
            complete_event=PICPEvent.REASONING_COMPLETE,
        )
        self._llm = llm_client
        self._policy = HeuristicPolicy(min_match_count=min_match_count)
        self._registry = SkillRegistry(skills_dir=skills_dir)
        self._executor = SkillExecutor(
            llm_client=llm_client,
            model_fast=model_fast,
            model_strong=model_strong,
        )

    @property
    def policy(self) -> HeuristicPolicy:
        """Access the heuristic policy (for testing/inspection)."""
        return self._policy

    @property
    def registry(self) -> SkillRegistry:
        """Access the skill registry (for testing/inspection)."""
        return self._registry

    async def _execute(self, context: PICPContext, **kwargs: Any) -> dict[str, Any]:
        """Execute the Reasoning Pillar pipeline.

        Steps:
        1. Classify the query using HeuristicPolicy.
        2. Look up the skill in the registry.
        3. Extract knowledge context from pillar_results.
        4. Execute the skill via LLM.
        5. Return structured result.

        Args:
            context: The PICP context with query and knowledge results.

        Returns:
            Dict with routing decision, skill result, and metadata.
        """
        query = context.query

        # Step 1: Classify query
        routing = self._policy.classify(query)

        logger.info(
            "reasoning.routing",
            correlation_id=context.correlation_id,
            query_type=routing.query_type.value,
            confidence=routing.confidence,
            skill_name=routing.skill_name,
        )

        result: dict[str, Any] = {
            "routing": {
                "query_type": routing.query_type.value,
                "confidence": routing.confidence,
                "matched_keywords": routing.matched_keywords,
                "skill_name": routing.skill_name,
            },
        }

        # Step 2: Look up skill
        if routing.skill_name is None:
            # GENERAL query — no specialised skill
            logger.info(
                "reasoning.general_query",
                correlation_id=context.correlation_id,
            )
            result["skill_result"] = {
                "skill_name": "general",
                "success": True,
                "parsed_output": {
                    "response": "Query does not match a specialised skill. "
                    "Passing through knowledge context.",
                },
                "timing_ms": 0.0,
                "warnings": [],
            }
            return result

        skill = self._registry.get_by_query_type(routing.query_type.value)
        if skill is None:
            logger.warning(
                "reasoning.skill_not_found",
                query_type=routing.query_type.value,
                correlation_id=context.correlation_id,
            )
            result["skill_result"] = {
                "skill_name": routing.skill_name,
                "success": False,
                "parsed_output": {},
                "timing_ms": 0.0,
                "warnings": [
                    f"Skill '{routing.skill_name}' not found in registry"
                ],
            }
            return result

        # Step 3: Extract knowledge context
        knowledge_context = self._build_context_string(context)

        # Step 4: Execute skill
        skill_result = await self._executor.execute(
            skill=skill,
            context=knowledge_context,
            query=query,
        )

        result["skill_result"] = skill_result.to_dict()
        return result

    @staticmethod
    def _build_context_string(context: PICPContext) -> str:
        """Extract knowledge context from the PICP context.

        Combines chunks from the Knowledge Pillar results into a
        single text block for the skill prompt template.

        Args:
            context: The PICP context with accumulated pillar results.

        Returns:
            Concatenated text context string.
        """
        knowledge = context.pillar_results.get("knowledge", {})

        # Handle different knowledge result formats
        chunks = knowledge.get("chunks", [])
        if not chunks:
            return "No knowledge context available."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            if isinstance(chunk, dict):
                text = chunk.get("text", chunk.get("chunk_text", ""))
            elif isinstance(chunk, str):
                text = chunk
            else:
                text = str(chunk)
            if text:
                parts.append(f"[{i}] {text}")

        return "\n\n".join(parts) if parts else "No knowledge context available."
