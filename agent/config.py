# agent/config.py — Environment configuration and API key management

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from project root
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Model Config ──────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"


def is_agent_available() -> bool:
    """
    Check whether both required API keys are configured.

    Returns True only when both GROQ and Tavily keys are non-empty,
    meaning the AI Agent pipeline can run.  When False the Streamlit
    app gracefully falls back to ML-only analysis.
    """
    groq_ok = bool(GROQ_API_KEY and GROQ_API_KEY.strip())
    tavily_ok = bool(TAVILY_API_KEY and TAVILY_API_KEY.strip())

    if not groq_ok:
        logger.info("GROQ_API_KEY not set — AI Agent disabled")
    if not tavily_ok:
        logger.info("TAVILY_API_KEY not set — AI Agent disabled")

    return groq_ok and tavily_ok
