# agent/nodes.py — LangGraph node functions for the AI analysis pipeline
from __future__ import annotations

import json
import logging
from langchain_groq import ChatGroq
from agent.state import AgentState
from agent.config import GROQ_API_KEY, GROQ_MODEL
from agent.tools import verify_claim, fetch_related_articles

logger = logging.getLogger(__name__)


def _get_llm(max_tokens: int = 1024) -> ChatGroq:
    """Instantiate the Groq LLM client."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3,
        max_tokens=max_tokens,
    )


def _parse_json_safe(text: str) -> list | dict | None:
    """Try to extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object within the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        return None


# ═══════════════════════════════════════════════════════════════
# Node 1 — Understand the News
# ═══════════════════════════════════════════════════════════════

def understand_news(state: AgentState) -> dict:
    """
    Identify the tone (sensational / neutral / urgent / emotional) and
    domain (politics, health, finance, tech, etc.) of the article.
    """
    try:
        llm = _get_llm()
        prompt = f"""Analyze the following news article and respond with EXACTLY two lines:
Line 1: TONE: <one of: sensational, neutral, urgent, emotional>
Line 2: DOMAIN: <one of: politics, finance, health, technology, entertainment, sports, science, world, other>

Rules:
- "sensational" = clickbait, exaggerated language
- "neutral" = factual, measured, objective reporting
- "urgent" = breaking news tone, time-sensitive language
- "emotional" = fear, anger, panic-inducing language
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
                if raw in ("sensational", "neutral", "urgent", "emotional"):
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
# Node 2 — Highlight Suspicious Text
# ═══════════════════════════════════════════════════════════════

def highlight_suspicious(state: AgentState) -> dict:
    """
    Extract specific phrases from the article that may indicate
    misinformation: clickbait, urgency, emotional triggers,
    absolute claims, unsupported numbers.
    """
    try:
        llm = _get_llm(max_tokens=1500)
        prompt = f"""You are a misinformation analyst. Extract suspicious phrases from the article below.

For each phrase, provide:
- "text": the EXACT phrase from the article (must exist in the text)
- "category": one of: clickbait, urgency, emotional, absolute, numeric
- "reason": why this phrase is suspicious (1 sentence)

Categories:
- clickbait: exaggerated or sensational words (e.g., "shocking", "unbelievable", "you won't believe")
- urgency: time-pressure phrases (e.g., "from tomorrow", "breaking", "act now")
- emotional: fear/panic/anger triggers (e.g., "devastating", "terrifying", "outrage")
- absolute: absolute claims without nuance (e.g., "always", "never", "everyone knows")
- numeric: unsupported or vague statistics (e.g., "millions affected", "thousands dead")

Return a JSON array. If no suspicious phrases exist, return an empty array [].

Article:
\"\"\"
{state["news_text"][:3000]}
\"\"\"

Respond with ONLY the JSON array, no other text.
"""
        response = llm.invoke(prompt)
        parsed = _parse_json_safe(response.content)

        highlights = []
        if isinstance(parsed, list):
            for item in parsed[:10]:  # Cap at 10
                if isinstance(item, dict) and "text" in item:
                    highlights.append({
                        "text": str(item.get("text", "")),
                        "category": str(item.get("category", "clickbait")),
                        "reason": str(item.get("reason", "")),
                    })

        logger.info(f"highlight_suspicious → found {len(highlights)} highlights")
        return {"highlights": highlights}

    except Exception as e:
        logger.error(f"highlight_suspicious failed: {e}")
        return {"highlights": []}


# ═══════════════════════════════════════════════════════════════
# Node 3 — Reason on ML Output
# ═══════════════════════════════════════════════════════════════

def reason_on_ml(state: AgentState) -> dict:
    """
    Pure logic node — no LLM or tool calls.
    Evaluates the ML confidence score and decides whether
    external verification is warranted.
    """
    confidence = state["ml_confidence"]
    prediction = state["ml_prediction"]

    if confidence >= 85:
        level = "high"
    elif confidence >= 60:
        level = "moderate"
    else:
        level = "uncertain"

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
# Node 4 — Breakdown Claims
# ═══════════════════════════════════════════════════════════════

def breakdown_claims(state: AgentState) -> dict:
    """
    Split the article into independent factual claims and
    evaluate each one separately.
    """
    try:
        llm = _get_llm(max_tokens=1500)
        prompt = f"""You are a fact-checker. Break down the following news article into independent factual claims.

For each claim, provide:
- "claim": the factual statement (rewrite concisely)
- "verdict": one of: "likely true", "likely false", "uncertain"
- "reason": a 1-sentence justification for the verdict

Rules:
- Extract only verifiable factual claims, not opinions
- Be honest: if you cannot verify a claim, mark it "uncertain"
- Do NOT fabricate evidence
- Return 3-7 claims maximum
- Return a JSON array

Article:
\"\"\"
{state["news_text"][:3000]}
\"\"\"

Respond with ONLY the JSON array, no other text.
"""
        response = llm.invoke(prompt)
        parsed = _parse_json_safe(response.content)

        claims = []
        if isinstance(parsed, list):
            for item in parsed[:7]:  # Cap at 7
                if isinstance(item, dict) and "claim" in item:
                    claims.append({
                        "claim": str(item.get("claim", "")),
                        "verdict": str(item.get("verdict", "uncertain")),
                        "reason": str(item.get("reason", "")),
                    })

        logger.info(f"breakdown_claims → found {len(claims)} claims")
        return {"claims": claims}

    except Exception as e:
        logger.error(f"breakdown_claims failed: {e}")
        return {"claims": []}


# ═══════════════════════════════════════════════════════════════
# Node 5 — Verify Claims (conditional)
# ═══════════════════════════════════════════════════════════════

def verify_claims(state: AgentState) -> dict:
    """Use Tavily to search for fact-check information."""
    try:
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
# Node 6 — Generate Explanation
# ═══════════════════════════════════════════════════════════════

def generate_explanation(state: AgentState) -> dict:
    """
    Produce a clear, structured explanation of the credibility assessment.
    Incorporates all context gathered so far.
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

        highlights_context = ""
        highlights = state.get("highlights", [])
        if highlights:
            h_list = ", ".join([f'"{h["text"]}" ({h["category"]})' for h in highlights[:5]])
            highlights_context = f"\nSuspicious phrases found: {h_list}"

        claims_context = ""
        claims = state.get("claims", [])
        if claims:
            c_list = "\n".join([f'- "{c["claim"]}" → {c["verdict"]}' for c in claims[:5]])
            claims_context = f"\nClaim analysis:\n{c_list}"

        prompt = f"""You are an expert fact-checker. Based on the analysis below, write a concise
explanation (3-5 bullet points) of why this news article is likely {state['ml_prediction']}.

Context:
- ML Prediction: {state['ml_prediction']} ({state['ml_confidence']:.1f}% confidence)
- Confidence Level: {state.get('confidence_level', 'unknown')}
- Detected Tone: {state.get('tone', 'unknown')}
- Detected Domain: {state.get('domain', 'unknown')}
{verification_context}{highlights_context}{claims_context}

Article excerpt:
\"\"\"
{state['news_text'][:2000]}
\"\"\"

Rules:
1. Be concise but informative (each point should be 1-2 sentences max)
2. Mention specific linguistic signals (clickbait, emotional language, source attribution, etc.)
3. If verification was performed, reference whether sources confirmed or contradicted the claim
4. Reference specific suspicious highlights or claim verdicts if available
5. Do NOT fabricate sources or make up facts
6. Write in a professional, neutral tone

Format your response as bullet points starting with "•"
"""
        response = llm.invoke(prompt)
        explanation = response.content.strip()

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
# Node 7 — Fetch Related News (only for REAL predictions)
# ═══════════════════════════════════════════════════════════════

def fetch_related(state: AgentState) -> dict:
    """Fetch 3-5 related articles from credible sources via Tavily."""
    try:
        query = state["news_text"][:150].strip()
        articles = fetch_related_articles(query, num_results=5)
        logger.info(f"fetch_related → found {len(articles)} related articles")
        return {"related_news": articles}
    except Exception as e:
        logger.error(f"fetch_related failed: {e}")
        return {"related_news": []}


# ═══════════════════════════════════════════════════════════════
# Standalone — Compare Two News Articles
# ═══════════════════════════════════════════════════════════════

def compare_news(news_a: str, news_b: str) -> dict:
    """
    Compare two news articles and determine which is more credible.
    Returns {summary, more_credible, reason}.
    This is a standalone function, NOT a graph node.
    """
    try:
        llm = _get_llm(max_tokens=1024)
        prompt = f"""You are an expert fact-checker. Compare News A and News B below.

Your task:
1. Identify contradictions between the two articles
2. Assess which article uses more credible language (source attribution, neutral tone, specificity)
3. Determine which is more credible

News A:
\"\"\"
{news_a[:2000]}
\"\"\"

News B:
\"\"\"
{news_b[:2000]}
\"\"\"

Respond in EXACTLY this JSON format:
{{
  "summary": "<2-3 sentence comparison of the two articles>",
  "more_credible": "<News A | News B | cannot determine>",
  "reason": "<1-2 sentence explanation of why>"
}}

Respond with ONLY the JSON object, no other text.
"""
        response = llm.invoke(prompt)
        parsed = _parse_json_safe(response.content)

        if isinstance(parsed, dict):
            return {
                "summary": str(parsed.get("summary", "")),
                "more_credible": str(parsed.get("more_credible", "cannot determine")),
                "reason": str(parsed.get("reason", "")),
            }

        return {
            "summary": response.content.strip()[:500],
            "more_credible": "cannot determine",
            "reason": "Could not parse structured comparison.",
        }

    except Exception as e:
        logger.error(f"compare_news failed: {e}")
        return {
            "summary": "Comparison could not be completed due to an error.",
            "more_credible": "cannot determine",
            "reason": str(e),
        }


# ═══════════════════════════════════════════════════════════════
# Standalone — Answer Follow-up Question
# ═══════════════════════════════════════════════════════════════

def answer_followup(agent_state: dict, question: str) -> str:
    """
    Answer a follow-up question using the analysis context.
    This is a standalone function, NOT a graph node.
    """
    try:
        llm = _get_llm(max_tokens=512)

        context_parts = [
            f"Prediction: {agent_state.get('ml_prediction', 'unknown')} ({agent_state.get('ml_confidence', 0):.1f}% confidence)",
            f"Tone: {agent_state.get('tone', 'unknown')}",
            f"Domain: {agent_state.get('domain', 'unknown')}",
            f"Confidence Level: {agent_state.get('confidence_level', 'unknown')}",
        ]

        explanation = agent_state.get("explanation", "")
        if explanation:
            context_parts.append(f"Analysis:\n{explanation}")

        verification = agent_state.get("verification_result", "")
        if verification and "not required" not in verification.lower():
            context_parts.append(f"Verification: {verification}")

        highlights = agent_state.get("highlights", [])
        if highlights:
            h_text = ", ".join([f'"{h["text"]}"' for h in highlights[:5]])
            context_parts.append(f"Suspicious phrases: {h_text}")

        claims = agent_state.get("claims", [])
        if claims:
            c_text = "; ".join([f'{c["claim"]} → {c["verdict"]}' for c in claims[:5]])
            context_parts.append(f"Claims: {c_text}")

        context = "\n".join(context_parts)

        prompt = f"""You are a helpful AI fact-checking assistant. A user has analyzed a news article and is asking a follow-up question.

Analysis context:
{context}

Article excerpt:
\"\"\"
{agent_state.get('news_text', '')[:1500]}
\"\"\"

User question: {question}

Rules:
1. Answer based ONLY on the analysis context and article provided
2. Be clear, concise, and helpful (3-5 sentences max)
3. Do NOT fabricate facts or sources
4. If you cannot answer from the available context, say so honestly
"""
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        logger.error(f"answer_followup failed: {e}")
        return "I'm sorry, I couldn't process your question. Please try again."
