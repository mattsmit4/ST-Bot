"""Shared configuration utilities."""

import os


def get_config_value(key: str, default: str = 'false') -> str:
    """Get config from os.environ first, then st.secrets (for Streamlit Cloud)."""
    value = os.environ.get(key)
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(key)
        except Exception:
            pass
    return str(value).lower() if value else default
