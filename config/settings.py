"""Application settings loaded from environment variables.

Uses pydantic-settings to validate and parse .env configuration.
All thesis-specific constants are defined here for reproducibility.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the AI Strategic Partner PoC."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM Provider ──────────────────────────────────────────
    llm_provider: str = "openai"  # openai | mock
    openai_api_key: str = ""
    llm_model_fast: str = "gpt-4o-mini"
    llm_model_strong: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Redis (PICP Bus) ────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── ChromaDB ────────────────────────────────────────────
    chroma_persist_dir: str = "./data/embedded"

    # ── Logging ─────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Trust Pillar Weights (Eq. 3.28) ─────────────────────
    trust_weight_evidence: float = 0.4
    trust_weight_coherence: float = 0.4
    trust_weight_agreement: float = 0.2

    # ── Hybrid Search (Eq. 3.17) ────────────────────────────
    hybrid_search_alpha: float = 0.7

    # ── Monte Carlo ─────────────────────────────────────────
    monte_carlo_simulations: int = 10_000

    # ── PICP ────────────────────────────────────────────────
    picp_lock_ttl_ms: int = 5_000
    picp_lock_retry_delay_ms: int = 200
    picp_lock_max_retries: int = 10

    # ── Pillar priority map (lower = higher priority) ───────
    @property
    def pillar_priorities(self) -> dict[str, int]:
        """Priority hierarchy enforced by PICP (Section 3.4)."""
        return {
            "reasoning": 1,
            "simulation": 2,
            "trust": 3,
            "knowledge": 4,
        }

    @property
    def pillar_names(self) -> list[str]:
        """Ordered list of pillar identifiers for vector clocks."""
        return ["knowledge", "reasoning", "simulation", "trust"]


def get_settings() -> Settings:
    """Factory for cached settings singleton."""
    return Settings()
