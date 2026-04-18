# agent/tools.py — Tavily search wrappers for verification and related news

import logging
from typing import Optional
from tavily import TavilyClient
from agent.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)


def _get_client() -> TavilyClient:
    """Instantiate a Tavily client with the configured API key."""
    return TavilyClient(api_key=TAVILY_API_KEY)


def verify_claim(query: str) -> dict:
    """
    Search the web for fact-check / verification results related to a claim.

    Args:
        query: A search query derived from the news article's core claim.

    Returns:
        {
            "summary": str,        # concise summary of findings
            "sources":  list[str]  # URLs of the sources checked
        }
    """
    try:
        client = _get_client()
        results = client.search(
            query=f"fact check: {query}",
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )

        sources = [r["url"] for r in results.get("results", []) if r.get("url")]
        answer = results.get("answer", "")

        if not answer:
            # Build a summary from the top result snippets
            snippets = [r.get("content", "") for r in results.get("results", [])[:3]]
            answer = " ".join(snippets)[:500] if snippets else "No verification results found."

        return {"summary": answer, "sources": sources}

    except Exception as e:
        logger.error(f"Tavily verification search failed: {e}")
        return {
            "summary": "Verification could not be completed due to a search error.",
            "sources": [],
        }


def fetch_related_articles(query: str, num_results: int = 5) -> list[dict]:
    """
    Fetch related news articles from credible sources.

    Args:
        query:       A search query derived from the news topic.
        num_results: Number of articles to return (default 5).

    Returns:
        List of dicts, each with keys: title, summary, url
    """
    try:
        client = _get_client()
        results = client.search(
            query=query,
            search_depth="basic",
            max_results=num_results,
            include_answer=False,
        )

        articles = []
        for r in results.get("results", []):
            title = r.get("title", "Untitled")
            content = r.get("content", "")
            url = r.get("url", "")

            # Truncate content to a short summary (1-2 lines)
            summary = content[:200].rsplit(" ", 1)[0] + "..." if len(content) > 200 else content

            if url:
                articles.append({
                    "title": title,
                    "summary": summary,
                    "url": url,
                })

        return articles

    except Exception as e:
        logger.error(f"Tavily related-news search failed: {e}")
        return []
