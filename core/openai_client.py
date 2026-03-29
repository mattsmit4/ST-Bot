"""
Centralized OpenAI client initialization for ST-Bot.

All LLM modules should import get_openai_client() from here
instead of duplicating the initialization logic.

Token tracking: import get_token_usage() to see cumulative usage.
"""

import os
import logging

_logger = logging.getLogger(__name__)

# Lazy-loaded OpenAI client (singleton)
_openai_client = None

# Cumulative token usage tracking
_token_usage = {
    'total_calls': 0,
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_tokens': 0,
}


def get_token_usage() -> dict:
    """Get cumulative token usage across all LLM calls this session."""
    return dict(_token_usage)


def _track_usage(response):
    """Track token usage from an OpenAI response."""
    if hasattr(response, 'usage') and response.usage:
        _token_usage['total_calls'] += 1
        _token_usage['prompt_tokens'] += response.usage.prompt_tokens or 0
        _token_usage['completion_tokens'] += response.usage.completion_tokens or 0
        _token_usage['total_tokens'] += response.usage.total_tokens or 0


class _InstrumentedCompletions:
    """Wrapper that tracks token usage on each create() call."""

    def __init__(self, original_completions):
        self._original = original_completions

    def create(self, **kwargs):
        response = self._original.create(**kwargs)
        _track_usage(response)
        return response


class _InstrumentedChat:
    """Wrapper for chat namespace with instrumented completions."""

    def __init__(self, original_chat):
        self.completions = _InstrumentedCompletions(original_chat.completions)


class _InstrumentedClient:
    """Lightweight wrapper around OpenAI client that tracks token usage."""

    def __init__(self, client):
        self._client = client
        self.chat = _InstrumentedChat(client.chat)

    def __getattr__(self, name):
        return getattr(self._client, name)


def get_openai_client():
    """
    Get or create OpenAI client (lazy initialization).

    Returns the shared OpenAI client instance, creating it on first call.
    Tries environment variable first, falls back to Streamlit secrets.
    Client is instrumented to track token usage automatically.

    Returns:
        OpenAI client instance, or None if API key not available
    """
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                try:
                    import streamlit as st
                    api_key = st.secrets.get('OPENAI_API_KEY')
                except Exception:
                    pass
            if api_key:
                raw_client = OpenAI(api_key=api_key)
                _openai_client = _InstrumentedClient(raw_client)
        except ImportError:
            pass
    return _openai_client
