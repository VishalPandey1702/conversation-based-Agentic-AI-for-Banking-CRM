"""
Centralized application configuration.

Reads environment variables (with sensible defaults), validates them through
Pydantic, and exposes them as a single `settings` singleton consumed by the
rest of the backend.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Locate and load .env relative to the repository root (one level above /backend)
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH) if _ENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---------- Azure OpenAI ----------
    AZURE_OPENAI_API_KEY: str = Field(default="")
    AZURE_OPENAI_ENDPOINT: str = Field(default="")
    AZURE_OPENAI_DEPLOYMENT: str = Field(default="gpt-5-nano-5")
    AZURE_OPENAI_MODEL: str = Field(default="gpt-5-nano")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-10-21")

    # ---------- Database ----------
    DATABASE_URL: str = Field(default="sqlite:///banking_crm.db")

    # ---------- Application ----------
    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    STREAMLIT_PORT: int = Field(default=8501)
    BACKEND_BASE_URL: str = Field(default="http://localhost:8000")

    # ---------- Workflow ----------
    DEFAULT_TOP_N_CUSTOMERS: int = Field(default=10)
    MIN_CONVERSION_THRESHOLD: float = Field(default=0.55)

    @property
    def llm_configured(self) -> bool:
        """True when an Azure OpenAI key/endpoint pair is available."""
        return bool(self.AZURE_OPENAI_API_KEY and self.AZURE_OPENAI_ENDPOINT)


@lru_cache(maxsize=1)
def _build_settings() -> Settings:
    """Build the settings object once and cache it for the process lifetime."""
    return Settings()


# A single, importable singleton: `from backend.utils.config import settings`
settings: Settings = _build_settings()
