from __future__ import annotations

from typing import Protocol

from rineya_harness.domain.models import SearchResult


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, *, limit: int = 8) -> list[SearchResult]: ...
