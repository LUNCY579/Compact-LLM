from __future__ import annotations

from typing import Protocol

from rineya_harness.domain.models import FetchedDocument


class DocumentFetcher(Protocol):
    async def fetch(self, url: str) -> FetchedDocument: ...
