from __future__ import annotations

from dataclasses import dataclass, field

from rineya_harness.domain.models import FetchedDocument
from rineya_harness.errors import NotFoundError


@dataclass(slots=True)
class FixtureDocumentFetcher:
    documents: dict[str, FetchedDocument] = field(default_factory=dict)

    async def fetch(self, url: str) -> FetchedDocument:
        normalized = url.rstrip("/")
        for key, document in self.documents.items():
            if key.rstrip("/") == normalized:
                return document
        raise NotFoundError("fixture_document", url)
