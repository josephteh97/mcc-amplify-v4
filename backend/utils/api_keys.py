"""
API key resolution helpers.

Priority order for each key:
  1. Environment variable (set in .env or shell)
  2. Plaintext key file next to this backend directory (gitignored)
"""

import os
from pathlib import Path

# Backend root = the directory that contains this utils/ package
_BACKEND_DIR = Path(__file__).resolve().parents[1]


def _read_key_file(filename: str) -> str | None:
    """Read a single-line key file; return None if missing or empty."""
    path = _BACKEND_DIR / filename
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


def _is_valid(key: str | None) -> bool:
    """Reject missing, empty, or placeholder values like [REPLACE_WITH_NEW_KEY]."""
    return bool(key) and not key.startswith("[")


def get_google_api_key() -> str | None:
    """Return the Google API key from env var or google_key.txt."""
    env_key = os.getenv("GOOGLE_API_KEY")
    if _is_valid(env_key):
        return env_key
    return _read_key_file("google_key.txt")


def get_nvidia_api_key() -> str | None:
    """Return the NVIDIA NIM API key from env var or nvidia_key.txt."""
    env_key = os.getenv("NVIDIA_API_KEY")
    if _is_valid(env_key):
        return env_key
    return _read_key_file("nvidia_key.txt")


def get_anthropic_api_key() -> str | None:
    """Return the Anthropic API key from env var or claude_key.txt."""
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if _is_valid(env_key):
        return env_key
    return _read_key_file("claude_key.txt")
