"""Application configuration, loaded from environment variables / .env file.

Nothing secret is ever hardcoded. Every credential arrives via the environment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    Values come from (in priority order): real environment variables, then `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Discord ---
    discord_bot_token: str = Field(default="", description="Discord bot token.")
    discord_command_prefix: str = Field(default="!")

    # --- LLM (provider agnostic, OpenAI-compatible chat completions API) ---
    llm_api_key: str = Field(default="", description="API key for the LLM provider.")
    llm_model: str = Field(default="claude-sonnet-5")
    llm_base_url: str = Field(
        default="https://api.anthropic.com/v1",
        description="Base URL of the provider. Anthropic, OpenAI, Groq, Together, "
        "Ollama and any OpenAI-compatible gateway all work by changing this.",
    )
    llm_provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description="Wire format to speak. 'openai' covers every OpenAI-compatible gateway.",
    )
    llm_timeout_seconds: float = Field(default=90.0)
    llm_max_retries: int = Field(default=2)
    llm_max_tokens: int = Field(default=4096)

    # --- Behaviour ---
    demo_mode: bool = Field(
        default=False,
        description="When true, skip the LLM entirely and use the deterministic "
        "heuristic extractor. Useful for offline demos and CI.",
    )
    max_files_per_session: int = Field(default=10)
    max_file_size_mb: float = Field(default=10.0)
    max_document_chars: int = Field(
        default=20000, description="Documents are truncated to this before prompting."
    )
    session_ttl_minutes: int = Field(default=60)
    log_level: str = Field(default="INFO")

    @field_validator("llm_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)

    @property
    def llm_configured(self) -> bool:
        """True when we have enough config to attempt a real LLM call."""
        return bool(self.llm_api_key) and not self.demo_mode

    def safe_summary(self) -> dict[str, object]:
        """Config summary safe for logging - never includes secrets."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_api_key": "set" if self.llm_api_key else "missing",
            "discord_bot_token": "set" if self.discord_bot_token else "missing",
            "demo_mode": self.demo_mode,
            "max_files_per_session": self.max_files_per_session,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read the environment (used by tests)."""
    get_settings.cache_clear()
    return get_settings()
