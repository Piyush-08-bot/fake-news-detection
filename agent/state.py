# agent/state.py — LangGraph state schema

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state that flows through every node in the LangGraph pipeline.

    Fields are populated progressively as the graph executes:
      Inputs  → set before the graph runs
      Outputs → written by individual nodes
    """

    # ── Inputs (set by caller) ────────────────────────────────
    news_text: str          # original article text
    ml_prediction: str      # "FAKE" or "REAL"
    ml_confidence: float    # 0-100

    # ── Node outputs ──────────────────────────────────────────
    tone: str               # "sensational" | "neutral" | "urgent"
    domain: str             # "politics" | "health" | "finance" | ...
    confidence_level: str   # "high" | "moderate" | "uncertain"
    should_verify: bool     # whether web verification is needed

    verification_result: str       # summary of Tavily verification
    source_links: list[str]        # URLs from verification search

    explanation: str               # LLM-generated analysis text
    final_verdict: str             # human-readable verdict string

    related_news: list[dict]       # [{title, summary, url}, ...]
