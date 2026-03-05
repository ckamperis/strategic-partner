"""Skill executor — runs a skill definition via LLM.

Takes a SkillDefinition + knowledge context + query and:
1. Renders the prompt template with context and query.
2. Calls the LLM with the system prompt and rendered prompt.
3. Parses the JSON response (with retry on failure).
4. Returns a structured SkillResult.

References:
    Thesis Section 3.3.2 — Reasoning Pillar, Skill Execution
    Thesis Section 4.x — Implementation, LLM-driven Skill Execution
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from pillars.reasoning.skill_registry import SkillDefinition
from utils.llm import LLMClient

logger = structlog.get_logger()


@dataclass
class SkillResult:
    """Result from executing a skill via LLM.

    Attributes:
        skill_name: The skill that was executed.
        raw_response: The raw LLM text response.
        parsed_output: The parsed JSON dict (or empty on failure).
        success: Whether JSON parsing succeeded.
        attempts: Number of LLM calls made (including retries).
        timing_ms: Total execution time in milliseconds.
        warnings: Any warnings generated during execution.
    """

    skill_name: str
    raw_response: str = ""
    parsed_output: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    attempts: int = 0
    timing_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary for PICP context storage."""
        return {
            "skill_name": self.skill_name,
            "parsed_output": self.parsed_output,
            "success": self.success,
            "attempts": self.attempts,
            "timing_ms": self.timing_ms,
            "warnings": self.warnings,
        }


class SkillExecutor:
    """Executes a YAML-defined skill using an LLM client.

    The executor:
    1. Renders the prompt template with the knowledge context and query.
    2. Calls the LLM with the skill's system prompt.
    3. Attempts to parse the response as JSON.
    4. Retries up to ``max_retries`` times on parse failure.

    Args:
        llm_client: The LLM client to use for completions.
        model_fast: Model name for "fast" tier skills.
        model_strong: Model name for "strong" tier skills.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model_fast: str = "gpt-4o-mini",
        model_strong: str = "gpt-4o",
    ) -> None:
        self._llm = llm_client
        self._model_fast = model_fast
        self._model_strong = model_strong

    async def execute(
        self,
        skill: SkillDefinition,
        context: str,
        query: str,
    ) -> SkillResult:
        """Execute a skill definition against the provided context.

        Args:
            skill: The skill definition to execute.
            context: Concatenated knowledge context (RAG chunks as text).
            query: The user's original natural-language query.

        Returns:
            A SkillResult with parsed output and execution metadata.
        """
        start = time.perf_counter()
        max_retries = skill.parameters.max_retries
        model = self._resolve_model(skill.parameters.model_tier)

        # Render the prompt template
        rendered_prompt = skill.render_prompt(context=context, query=query)
        system_prompt = skill.system_prompt

        result = SkillResult(skill_name=skill.name)
        last_response = ""

        for attempt in range(1, max_retries + 1):
            result.attempts = attempt

            try:
                # Adjust prompt on retry to emphasise JSON format
                prompt = rendered_prompt
                if attempt > 1:
                    prompt = (
                        f"{rendered_prompt}\n\n"
                        f"IMPORTANT: Your previous response was not valid JSON. "
                        f"Please output ONLY valid JSON, no markdown or extra text."
                    )

                response = await self._llm.complete(
                    prompt=prompt,
                    system=system_prompt,
                    model=model,
                    temperature=skill.parameters.temperature,
                )

                last_response = response
                result.raw_response = response

                # Try to parse JSON from the response
                parsed = self._extract_json(response)
                if parsed is not None:
                    result.parsed_output = parsed
                    result.success = True

                    elapsed = time.perf_counter() - start
                    result.timing_ms = round(elapsed * 1000, 2)

                    logger.info(
                        "skill_executor.success",
                        skill=skill.name,
                        attempts=attempt,
                        timing_ms=result.timing_ms,
                        model=model,
                    )
                    return result

                # JSON parse failed — retry
                logger.warning(
                    "skill_executor.json_parse_failed",
                    skill=skill.name,
                    attempt=attempt,
                    response_preview=response[:200],
                )

            except Exception as e:
                logger.error(
                    "skill_executor.llm_error",
                    skill=skill.name,
                    attempt=attempt,
                    error=str(e),
                )
                result.warnings.append(f"LLM error on attempt {attempt}: {e}")

        # All retries exhausted — return best-effort result
        elapsed = time.perf_counter() - start
        result.timing_ms = round(elapsed * 1000, 2)
        result.raw_response = last_response
        result.warnings.append(
            f"Failed to parse JSON after {max_retries} attempts"
        )

        logger.warning(
            "skill_executor.exhausted",
            skill=skill.name,
            attempts=max_retries,
            timing_ms=result.timing_ms,
        )

        return result

    def _resolve_model(self, tier: str) -> str:
        """Map a model tier name to an actual model identifier.

        Args:
            tier: "fast" or "strong".

        Returns:
            The model name string.
        """
        if tier == "strong":
            return self._model_strong
        return self._model_fast

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Attempt to extract a JSON object from LLM response text.

        Handles:
        - Pure JSON responses
        - JSON wrapped in markdown code blocks (```json ... ```)
        - JSON embedded in surrounding text

        Args:
            text: The raw LLM response text.

        Returns:
            Parsed dict, or None if parsing failed.
        """
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```" in text:
            # Find content between ```json and ``` (or just ``` and ```)
            import re

            patterns = [
                r"```json\s*\n(.*?)```",
                r"```\s*\n(.*?)```",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1).strip())
                    except json.JSONDecodeError:
                        continue

        # Try finding first { ... } block
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        return None
