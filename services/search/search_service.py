import os
import re
import httpx

URL = os.getenv("SEARXNG_URL")
user, password = os.getenv("SEARXNG_USER"), os.getenv("SEARXNG_PASS")
AUTH = (user, password) if user and password else None
SEARCH_TRIGGERS = {"latest", "news", "current", "today", "yesterday", "recent", "weather", "price"}


def should_search_web(message: str, points: list | None = None, threshold: float = 0.6) -> bool:
    if points is not None and (not points or points[0].score < threshold):
        return True
    return bool(set(re.findall(r"\w+", message.lower())) & SEARCH_TRIGGERS)


def search(query: str, limit: int = 5):
    if not URL:
        return []
    try:
        res = httpx.get(URL, params={"q": query, "format": "json"}, auth=AUTH, timeout=10)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in res.json().get("results", [])[:limit]
        ]
    except Exception:
        return []


def prepare_message(message: str, limit: int = 3) -> str:
    if should_search_web(message) and (web := search(message, limit=limit)):
        context = "\n\n".join(f"{w['title']}: {w['content']} ({w['url']})" for w in web)
        return f"Context:\nWeb Search Results:\n{context}\n\nQuestion: {message}\nAnswer based on context:"
    return message
