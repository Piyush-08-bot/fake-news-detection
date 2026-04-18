# agent/graph.py — LangGraph state-graph assembly and public runner

import logging
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    understand_news,
    reason_on_ml,
    verify_claims,
    skip_verification,
    generate_explanation,
    fetch_related,
)

logger = logging.getLogger(__name__)


def _route_verification(state: AgentState) -> str:
    """Conditional edge: decide whether to verify or skip."""
    if state.get("should_verify", False):
        return "verify_claims"
    return "skip_verification"


def _route_after_explanation(state: AgentState) -> str:
    """Conditional edge: fetch related news only for REAL predictions."""
    if state.get("ml_prediction") == "REAL":
        return "fetch_related"
    return END


def _build_graph() -> StateGraph:
    """
    Assemble the full LangGraph pipeline:

        understand_news
              │
        reason_on_ml
              │
        ┌─────┴──────┐
        ▼             ▼
    verify_claims  skip_verification
        └─────┬──────┘
              ▼
      generate_explanation
              │
        ┌─────┴──────┐
        ▼             ▼
    fetch_related    END
        │
        ▼
       END
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────
    graph.add_node("understand_news", understand_news)
    graph.add_node("reason_on_ml", reason_on_ml)
    graph.add_node("verify_claims", verify_claims)
    graph.add_node("skip_verification", skip_verification)
    graph.add_node("generate_explanation", generate_explanation)
    graph.add_node("fetch_related", fetch_related)

    # ── Set entry point ───────────────────────────────────────
    graph.set_entry_point("understand_news")

    # ── Fixed edges ───────────────────────────────────────────
    graph.add_edge("understand_news", "reason_on_ml")
    graph.add_edge("verify_claims", "generate_explanation")
    graph.add_edge("skip_verification", "generate_explanation")
    graph.add_edge("fetch_related", END)

    # ── Conditional edges ─────────────────────────────────────
    graph.add_conditional_edges(
        "reason_on_ml",
        _route_verification,
        {
            "verify_claims": "verify_claims",
            "skip_verification": "skip_verification",
        },
    )

    graph.add_conditional_edges(
        "generate_explanation",
        _route_after_explanation,
        {
            "fetch_related": "fetch_related",
            END: END,
        },
    )

    return graph


# Pre-compile the graph once at module level
_compiled_graph = _build_graph().compile()


def run_agent_analysis(
    news_text: str,
    ml_prediction: str,
    ml_confidence: float,
) -> dict:
    """
    Public API — run the full AI agent analysis pipeline.

    Args:
        news_text:      The original news article text.
        ml_prediction:  "FAKE" or "REAL" from the ML model.
        ml_confidence:  Confidence percentage (0-100).

    Returns:
        Final AgentState dict with all fields populated.
    """
    try:
        initial_state: AgentState = {
            "news_text": news_text,
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
        }

        logger.info(
            f"Starting agent analysis: prediction={ml_prediction}, "
            f"confidence={ml_confidence:.1f}%"
        )

        result = _compiled_graph.invoke(initial_state)

        logger.info("Agent analysis completed successfully")
        return dict(result)

    except Exception as e:
        logger.error(f"Agent analysis pipeline failed: {e}")
        # Return a minimal fallback so the UI doesn't crash
        return {
            "news_text": news_text,
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "tone": "unknown",
            "domain": "unknown",
            "confidence_level": "unknown",
            "should_verify": False,
            "verification_result": "Agent analysis failed.",
            "source_links": [],
            "explanation": "The AI analysis pipeline encountered an error. Please try again.",
            "final_verdict": (
                "⚠️ This news is likely misinformation"
                if ml_prediction == "FAKE"
                else "✅ This news appears to be credible"
            ),
            "related_news": [],
        }
