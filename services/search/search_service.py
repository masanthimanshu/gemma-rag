import os
import httpx

URL = os.getenv("SEARXNG_URL")
AUTH = (os.getenv("SEARXNG_USER"), os.getenv("SEARXNG_PASS"))


def search(query: str, limit: int = 5):
    try:
        res = httpx.get(URL, params={"q": query, "format": "json"}, auth=AUTH, timeout=10)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in res.json().get("results", [])[:limit]
        ]
    except Exception:
        return []