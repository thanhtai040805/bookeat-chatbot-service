"""Router package exports for FastAPI app."""

from . import chat, health, sync, vector  # noqa: F401

__all__ = ["chat", "health", "sync", "vector"]

