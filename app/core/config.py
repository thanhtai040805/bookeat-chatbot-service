"""Application configuration helpers."""

from functools import lru_cache
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def _resolve_env_file() -> str:
    """Pick the appropriate .env file based on ENV variable."""
    env: Literal["dev", "prod", "test"] = os.getenv("ENV", "dev").lower()  # type: ignore[assignment]
    candidate = f".env.{env}"
    if os.path.exists(candidate):
        return candidate
    return ".env"


ENV_FILE = _resolve_env_file()
load_dotenv(ENV_FILE)
print(f"[CONFIG] Loaded environment from: {ENV_FILE}")


class Settings(BaseSettings):
    """Centralised configuration for the chatbot service."""

    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    SPRING_API_URL: str = Field(..., description="Base URL for Spring Boot backend")

    OPENAI_MODEL: str = Field("gpt-4o-mini", description="Model for OpenAI chat")
    OPENAI_TEMPERATURE: float = Field(0.4, description="Creativity for OpenAI responses")

    model_config = {
        "env_file": ENV_FILE,
        "case_sensitive": True,
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


# Backwards compatible alias for legacy imports.
def load_environment() -> Settings:
    """Legacy helper that returns the singleton settings object."""
    return get_settings()


# Eagerly instantiate for modules that `from app.core.config import settings`.
settings = get_settings()
