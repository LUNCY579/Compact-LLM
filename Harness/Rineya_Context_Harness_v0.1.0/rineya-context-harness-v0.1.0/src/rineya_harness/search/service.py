from __future__ import annotations

from dataclasses import dataclass

from rineya_harness.domain.models import SearchResult

from .base import SearchProvider


@dataclass(slots=True)
class SearchService:
    provider: SearchProvider
    max_results: int = 8

    async def search(self, query: str, *, limit: int | None = None) -> list[SearchResult]:
        query = " ".join(query.split())
        if not query:
            return []
        effective_limit = min(limit or self.max_results, self.max_results)
        results = await self.provider.search(query, limit=effective_limit)
        seen: set[str] = set()
        deduplicated: list[SearchResult] = []
        for result in sorted(results, key=lambda item: item.rank):
            key = str(result.url).rstrip("/").casefold()
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(result)
        return deduplicated[:effective_limit]
