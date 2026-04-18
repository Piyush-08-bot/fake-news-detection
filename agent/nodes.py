# agent/nodes.py — LangGraph node functions for the AI analysis pipeline

import logging
from langchain_groq import ChatGroq
from agent.state import AgentState
from agent.config import GROQ_API_KEY, GROQ_MODEL
from agent.tools import verify_claim, fetch_related_articles

logger = logging.getLogger(__name__)


def _get_llm() -> ChatGroq:
    """Instantiate the Groq LLM client."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3,
        max_tokens=1024,
    )


# ═══════════════════════════════════════════════════════════════
# Node 1 — Understand the News
# ═══════════════════════════════════════════════════════════════

def understand_news(state: AgentState) -> dict:
    """
    Identify the tone (sensational / neutral / urgent) and
    domain (politics, health, finance, tech, etc.) of the article.
    Uses Groq LLM for classification.
    """
    try:
        llm = _get_llm()
        prompt = f"""Analyze the following news article and respond with EXACTLY two lines:
Line 1: TONE: <one of: sensational, neutral, urgent>
Line 2: DOMAIN: <one of: politics, finance, health, technology, entertainment, sports, science, world, other>

Rules:
- "sensational" = clickbait, emotional, exaggerated language
- "neutral" = factual, measured, objective reporting
- "urgent" = breaking news tone, time-sensitive language
- Pick the SINGLE most fitting domain

Article:
\"\"\"
{state["news_text"][:3000]}
\"\"\"
"""
        response = llm.invoke(prompt)
        lines = response.content.strip().split("\n")

        tone = "neutral"
        domain = "other"
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("tone:"):
                raw = line_lower.replace("tone:", "").strip()
                if raw in ("sensational", "neutral", "urgent"):
                    tone = raw
            elif line_lower.startswith("domain:"):
                raw = line_lower.replace("domain:", "").strip()
                domain = raw

        logger.info(f"understand_news → tone={tone}, domain={domain}")
        return {"tone": tone, "domain": domain}

    except Exception as e:
        logger.error(f"understand_news failed: {e}")
        return {"tone": "neutral", "domain": "other"}


# ═══════════════════════════════════════════════════════════════
# Node 2 — Reason on ML Output
# ═══════════════════════════════════════════════════════════════

def reason_on_ml(state: AgentState) -> dict:
    """
    Pure logic node — no LLM or tool calls.
    Evaluates the ML confidence score and decides whether
    external verification is warranted.

    Verification triggers:
      • prediction is FAKE  AND confidence >= 85  (high-confidence fake → verify)
      • prediction is REAL  AND 60 <= confidence < 85 (moderate-confidence real → verify)
    """
    confidence = state["ml_confidence"]
    prediction = state["ml_prediction"]

    # Classify confidence level
    if confidence >= 85:
        level = "high"
    elif confidence >= 60:
        level = "moderate"
    else:
        level = "uncertain"

    # Decide if verification is needed
    should_verify = False
    if prediction == "FAKE" and level == "high":
        should_verify = True
    elif prediction == "REAL" and level == "moderate":
        should_verify = True

    logger.info(
        f"reason_on_ml → level={level}, should_verify={should_verify} "
        f"(pred={prediction}, conf={confidence:.1f})"
    )
    return {
        "confidence_level": level,
        "should_verify": should_verify,
    }


# ═══════════════════════════════════════════════════════════════
# Node 3 — Verify Claims (conditional)
# ═══════════════════════════════════════════════════════════════

def verify_claims(state: AgentState) -> dict:
    """
    Use Tavily to search for fact-check information about the claim.
    Only called when should_verify is True.
    """
    try:
        # Build a concise search query from the first ~200 chars
        text_snippet = state["news_text"][:200].strip()
        result = verify_claim(text_snippet)

        logger.info(f"verify_claims → found {len(result['sources'])} sources")
        return {
            "verification_result": result["summary"],
            "source_links": result["sources"],
        }

    except Exception as e:
        logger.error(f"verify_claims failed: {e}")
        return {
            "verification_result": "Verification could not be completed.",
            "source_links": [],
        }


def skip_verification(state: AgentState) -> dict:
    """Passthrough node when verification is not needed."""
    return {
        "verification_result": "Verification not required for this confidence level.",
        "source_links": [],
    }


# ═══════════════════════════════════════════════════════════════
# Node 4 — Generate Explanation
# ═══════════════════════════════════════════════════════════════

def generate_explanation(state: AgentState) -> dict:
    """
    Use Groq LLM to produce a clear, structured explanation of the
    credibility assessment.  Incorporates all context gathered so far
    (tone, domain, ML score, verification results).
    """
    try:
        llm = _get_llm()

        verification_context = ""
        if state.get("verification_result") and "not required" not in state["verification_result"].lower():
            verification_context = f"""
Web Verification Results:
{state["verification_result"]}
Sources checked: {', '.join(state.get('source_links', [])) or 'None'}
"""

        prompt = f"""You are an expert fact-checker. Based on the analysis below, write a concise
explanation (3-5 bullet points) of why this news article is likely {state['ml_prediction']}.

Context:
- ML Prediction: {state['ml_prediction']} ({state['ml_confidence']:.1f}% confidence)
- Confidence Level: {state.get('confidence_level', 'unknown')}
- Detected Tone: {state.get('tone', 'unknown')}
- Detected Domain: {state.get('domain', 'unknown')}
{verification_context}

Article excerpt:
\"\"\"
{state['news_text'][:2000]}
\"\"\"

Rules:
1. Be concise but informative (each point should be 1-2 sentences max)
2. Mention specific linguistic signals (clickbait, emotional language, source attribution, etc.)
3. If verification was performed, reference whether sources confirmed or contradicted the claim
4. Do NOT fabricate sources or make up facts
5. Write in a professional, neutral tone

Format your response as bullet points starting with "•"
"""
        response = llm.invoke(prompt)
        explanation = response.content.strip()

        # Build the final verdict string
        if state["ml_prediction"] == "FAKE":
            verdict = "⚠️ This news is likely misinformation"
        else:
            verdict = "✅ This news appears to be credible"

        logger.info("generate_explanation → complete")
        return {
            "explanation": explanation,
            "final_verdict": verdict,
        }

    except Exception as e:
        logger.error(f"generate_explanation failed: {e}")
        return {
            "explanation": "Analysis could not be generated due to an error.",
            "final_verdict": (
                "⚠️ This news is likely misinformation"
                if state["ml_prediction"] == "FAKE"
                else "✅ This news appears to be credible"
            ),
        }


# ═══════════════════════════════════════════════════════════════
# Node 5 — Fetch Related News (only for REAL predictions)
# ═══════════════════════════════════════════════════════════════

def fetch_related(state: AgentState) -> dict:
    """
    Fetch 3-5 related articles from credible sources via Tavily.
    Only called when the prediction is REAL.
    """
    try:
        # Use the first ~150 chars of the article as the search query
        query = state["news_text"][:150].strip()
        articles = fetch_related_articles(query, num_results=5)

        logger.info(f"fetch_related → found {len(articles)} related articles")
        return {"related_news": articles}

    except Exception as e:
        logger.error(f"fetch_related failed: {e}")
        return {"related_news": []}
