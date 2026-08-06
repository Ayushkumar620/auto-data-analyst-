"""
Configuration - Loads API keys and settings from environment / .env file.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_api_key():
    """Return the LLM API key from environment or .env file, or None."""
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or None


def get_llm_provider():
    """Return the configured LLM provider."""
    return os.environ.get("LLM_PROVIDER", "openai").lower()


def get_llm_base_url():
    """Return the base URL for OpenAI-compatible APIs."""
    return os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")


def get_llm_model():
    """Return the configured LLM model name."""
    return os.environ.get("LLM_MODEL", "gpt-4o-mini")


def has_api_key():
    """Check if an API key is configured."""
    return bool(get_api_key())
