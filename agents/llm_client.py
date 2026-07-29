"""
Centralised LLM client factory for mindFree.
All agents import from here — one place to change models, keys, or settings.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()


def _check_key() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key


def get_llm(max_tokens: int = 512) -> ChatAnthropic:
    """Return a plain ChatAnthropic instance. Key is validated once here."""
    _check_key()
    return ChatAnthropic(model="claude-haiku-4-5", max_tokens=max_tokens)


def get_structured_llm(schema, max_tokens: int = 512):
    """Return a ChatAnthropic instance with structured output bound to `schema`."""
    return get_llm(max_tokens=max_tokens).with_structured_output(schema)
