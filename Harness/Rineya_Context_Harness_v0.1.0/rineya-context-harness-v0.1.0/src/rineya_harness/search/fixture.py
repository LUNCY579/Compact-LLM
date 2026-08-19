from __future__ import annotations

from dataclasses import dataclass, field

from rineya_harness.domain.models import SearchResult


@dataclass(slots=True)
class FixtureSearchProvider:
    fixtures: dict[str, list[SearchResult]] = field(default_factory=dict)
    name: str = "fixture"

    async def search(self, query: str, *, limit: int = 8) -> list[SearchResult]:
        normalized = " ".join(query.casefold().split())
        for key, results in self.fixtures.items():
            if " ".join(key.casefold().split()) == normalized:
                return results[:limit]
        return []
