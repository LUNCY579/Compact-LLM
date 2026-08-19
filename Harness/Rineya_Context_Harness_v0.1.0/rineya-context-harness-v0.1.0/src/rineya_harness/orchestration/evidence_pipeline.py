from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rineya_harness.domain.models import (
    EvidencePassage,
    EvidenceRecord,
    FetchedDocument,
    SearchResult,
)
from rineya_harness.evidence.ledger import EvidenceLedgerBuilder
from rineya_harness.evidence.ranker import LexicalPassageRanker
from rineya_harness.search.service import SearchService
from rineya_harness.state.store import SQLiteStateStore
from rineya_harness.web.base import DocumentFetcher


@dataclass(slots=True)
class EvidenceBatch:
    query: str
    search_results: list[SearchResult] = field(default_factory=list)
    documents: list[FetchedDocument] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    reused_evidence_ids: list[str] = field(default_factory=list)
    fetch_failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvidencePipeline:
    search: SearchService
    fetcher: DocumentFetcher
    ranker: LexicalPassageRanker
    ledger: EvidenceLedgerBuilder
    store: SQLiteStateStore
    fetch_concurrency: int = 4
    max_passages_per_source: int = 2

    async def gather(
        self,
        *,
        run_id: str,
        query: str,
        max_results: int,
        max_sources: int,
        passage_limit: int = 4,
    ) -> EvidenceBatch:
        results = await self.search.search(query, limit=max_results)
        selected = results[: max(0, max_sources)]
        semaphore = asyncio.Semaphore(max(1, self.fetch_concurrency))

        async def fetch_one(
            result: SearchResult,
        ) -> tuple[FetchedDocument | None, dict[str, Any] | None]:
            async with semaphore:
                try:
                    return await self.fetcher.fetch(str(result.url)), None
                except Exception as exc:  # Individual sources are best-effort and fully traced.
                    return None, {
                        "url": str(result.url),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }

        fetched = await asyncio.gather(*(fetch_one(result) for result in selected))
        documents = [document for document, _ in fetched if document is not None]
        failures = [failure for _, failure in fetched if failure is not None]
        passages = [passage for document in documents for passage in self.ranker.split(document)]
        ranked_candidates = self.ranker.rank(
            query,
            passages,
            limit=max(passage_limit, passage_limit * 3),
        )
        ranked = self._diversify_passages(ranked_candidates, limit=passage_limit)
        proposed = self.ledger.build_records(
            run_id=run_id,
            query=query,
            documents=documents,
            ranked_passages=ranked,
        )

        existing = self.store.list_evidence(run_id)
        existing_by_key = {
            (str(record.source_uri).rstrip("/"), record.content_hash): record for record in existing
        }
        new_records: list[EvidenceRecord] = []
        reused: list[str] = []
        seen_new: set[tuple[str, str]] = set()
        for record in proposed:
            key = (str(record.source_uri).rstrip("/"), record.content_hash)
            prior = existing_by_key.get(key)
            if prior is not None:
                reused.append(prior.id)
                continue
            if key in seen_new:
                continue
            seen_new.add(key)
            new_records.append(record)
        self.store.add_evidence(new_records)
        return EvidenceBatch(
            query=query,
            search_results=results,
            documents=documents,
            evidence=new_records,
            reused_evidence_ids=sorted(set(reused)),
            fetch_failures=failures,
        )

    def _diversify_passages(
        self, passages: list[EvidencePassage], *, limit: int
    ) -> list[EvidencePassage]:
        if limit <= 0:
            return []
        selected: list[EvidencePassage] = []
        per_source: Counter[str] = Counter()
        for passage in passages:
            if per_source[passage.source_id] >= self.max_passages_per_source:
                continue
            selected.append(passage)
            per_source[passage.source_id] += 1
            if len(selected) >= limit:
                break
        return selected
