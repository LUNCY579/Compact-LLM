from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from pydantic import ValidationError

from rineya_harness.domain.models import SearchResult
from rineya_harness.errors import SecurityViolationError


@dataclass(slots=True)
class SearxNGSearchProvider:
    """Bounded adapter for a trusted, explicitly configured SearXNG endpoint."""

    base_url: str
    timeout_seconds: float = 15.0
    max_response_bytes: int = 1_000_000
    name: str = "searxng"

    async def search(self, query: str, *, limit: int = 8) -> list[SearchResult]:
        endpoint = urljoin(self.base_url.rstrip("/") + "/", "search")
        params = {"q": query, "format": "json", "language": "all", "safesearch": 1}
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            async with client.stream("GET", endpoint, params=params) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                byte_count = 0
                async for chunk in response.aiter_bytes():
                    byte_count += len(chunk)
                    if byte_count > self.max_response_bytes:
                        raise SecurityViolationError(
                            "SEARCH_RESPONSE_TOO_LARGE",
                            f"SearXNG response exceeded {self.max_response_bytes} bytes",
                        )
                    chunks.append(chunk)
        payload = json.loads(b"".join(chunks).decode("utf-8", errors="strict"))
        if not isinstance(payload, dict) or not isinstance(payload.get("results", []), list):
            raise ValueError("SearXNG response must contain a results list")

        results: list[SearchResult] = []
        for item in payload.get("results", [])[:limit]:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not url:
                continue
            try:
                result = SearchResult(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("content") or ""),
                    rank=len(results) + 1,
                    provider=self.name,
                    metadata={
                        "engine": item.get("engine"),
                        "engines": item.get("engines"),
                        "published_date": item.get("publishedDate"),
                    },
                )
            except ValidationError:
                continue
            results.append(result)
        return results
