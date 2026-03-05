"""Skill registry — loads and validates YAML skill definitions.

The registry discovers skill YAML files from the ``skills/`` directory,
parses them into ``SkillDefinition`` Pydantic models, and provides
lookup by name or query type.

References:
    Thesis Section 3.3.2 — Reasoning Pillar, Skill Registry
    Thesis Section 4.x — Implementation, YAML-driven Skill Architecture
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# Default skills directory (relative to this file)
_DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"


class SkillParameters(BaseModel):
    """Execution parameters for a skill.

    Attributes:
        model_tier: Which LLM tier to use ("fast" or "strong").
        temperature: Sampling temperature for determinism.
        max_retries: How many times to retry on JSON parse failure.
    """

    model_config = {"protected_namespaces": ()}

    model_tier: str = "fast"
    temperature: float = 0.0
    max_retries: int = 2


class SkillDefinition(BaseModel):
    """A complete skill definition loaded from YAML.

    Attributes:
        name: Unique skill identifier (e.g. "cashflow_forecast").
        description: Human-readable description of what the skill does.
        version: Skill definition version.
        query_type: Which QueryType this skill handles.
        required_context: List of required context keys from knowledge.
        system_prompt: System message for the LLM.
        prompt_template: Template with {context} and {query} placeholders.
        output_schema: Expected JSON output schema (for documentation/validation).
        parameters: Execution parameters (model tier, temperature, retries).
    """

    name: str
    description: str = ""
    version: str = "1.0"
    query_type: str = ""
    required_context: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    prompt_template: str = ""
    output_schema: dict[str, Any] = Field(default_factory=dict)
    parameters: SkillParameters = Field(default_factory=SkillParameters)

    def render_prompt(self, context: str, query: str) -> str:
        """Render the prompt template with context and query.

        Args:
            context: The knowledge context string (RAG chunks).
            query: The user's original query.

        Returns:
            The fully rendered prompt ready for LLM.
        """
        return self.prompt_template.format(context=context, query=query)


class SkillRegistry:
    """Registry that discovers and manages YAML skill definitions.

    On initialisation, scans the skills directory for ``.yaml`` files,
    parses each into a ``SkillDefinition``, and indexes by name and query_type.

    Args:
        skills_dir: Path to the directory containing YAML skill files.
            Defaults to ``pillars/reasoning/skills/``.
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills_dir = skills_dir or _DEFAULT_SKILLS_DIR
        self._by_name: dict[str, SkillDefinition] = {}
        self._by_query_type: dict[str, SkillDefinition] = {}
        self._load_skills()

    def _load_skills(self) -> None:
        """Discover and parse all YAML skill files."""
        if not self._skills_dir.exists():
            logger.warning(
                "skill_registry.no_directory",
                path=str(self._skills_dir),
            )
            return

        yaml_files = sorted(self._skills_dir.glob("*.yaml"))
        for yaml_file in yaml_files:
            try:
                self._load_skill_file(yaml_file)
            except Exception as e:
                logger.error(
                    "skill_registry.load_error",
                    file=yaml_file.name,
                    error=str(e),
                )

        logger.info(
            "skill_registry.loaded",
            skill_count=len(self._by_name),
            skill_names=list(self._by_name.keys()),
        )

    def _load_skill_file(self, path: Path) -> None:
        """Parse a single YAML file into a SkillDefinition.

        Args:
            path: Path to the YAML file.
        """
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict in {path.name}, got {type(raw)}")

        # Parse parameters sub-dict if present
        if "parameters" in raw and isinstance(raw["parameters"], dict):
            raw["parameters"] = SkillParameters(**raw["parameters"])

        skill = SkillDefinition(**raw)

        self._by_name[skill.name] = skill
        if skill.query_type:
            self._by_query_type[skill.query_type] = skill

        logger.debug(
            "skill_registry.loaded_skill",
            name=skill.name,
            query_type=skill.query_type,
            version=skill.version,
        )

    def get_by_name(self, name: str) -> SkillDefinition | None:
        """Look up a skill by its unique name.

        Args:
            name: The skill name (e.g. "cashflow_forecast").

        Returns:
            The SkillDefinition, or None if not found.
        """
        return self._by_name.get(name)

    def get_by_query_type(self, query_type: str) -> SkillDefinition | None:
        """Look up a skill by its query type.

        Args:
            query_type: The query type string (e.g. "cashflow_forecast").

        Returns:
            The SkillDefinition, or None if not found.
        """
        return self._by_query_type.get(query_type)

    @property
    def skill_names(self) -> list[str]:
        """List of all registered skill names."""
        return list(self._by_name.keys())

    @property
    def skill_count(self) -> int:
        """Number of registered skills."""
        return len(self._by_name)

    def all_skills(self) -> list[SkillDefinition]:
        """Return all registered skill definitions."""
        return list(self._by_name.values())
