from __future__ import annotations

import hashlib
from collections.abc import Iterable

from rineya_harness.domain.enums import MemoryKind, MemoryTemperature
from rineya_harness.domain.models import (
    EvidencePassage,
    EvidenceRecord,
    FetchedDocument,
    MemoryDraft,
)


class EvidenceLedgerBuilder:
    """Creates attributable evidence records and compact model-visible memory records."""

    def record_from_passage(
        self,
        *,
        run_id: str,
        query: str,
        document: FetchedDocument,
        passage: EvidencePassage,
        rank: int,
    ) -> EvidenceRecord:
        return EvidenceRecord(
            run_id=run_id,
            source_uri=str(document.final_url),
            title=document.title or "Untitled source",
            query=query,
            passage=passage.text,
            content_hash=hashlib.sha256(passage.text.encode("utf-8")).hexdigest(),
            rank=rank,
            suspicious_content=bool(document.metadata.get("prompt_injection_suspicious")),
            metadata={
                "source_id": document.id,
                "source_content_sha256": document.content_sha256,
                "passage_id": passage.id,
                "passage_score": passage.score,
                "start_offset": passage.start_offset,
                "end_offset": passage.end_offset,
                "prompt_injection_flags": document.metadata.get("prompt_injection_flags", []),
            },
            retrieved_at=document.fetched_at,
        )

    def memory_from_evidence(self, record: EvidenceRecord) -> MemoryDraft:
        return MemoryDraft(
            id=f"mem_{record.id}",
            kind=MemoryKind.ACTIVE_EVIDENCE,
            temperature=MemoryTemperature.WARM,
            summary=record.passage[:260],
            payload={
                "evidence_id": record.id,
                "title": record.title,
                "source_uri": record.source_uri,
                "passage": record.passage,
                "rank": record.rank,
                "suspicious_content": record.suspicious_content,
            },
            source_ids={record.id},
            metadata={"untrusted_source_text": True},
        )

    def build_records(
        self,
        *,
        run_id: str,
        query: str,
        documents: Iterable[FetchedDocument],
        ranked_passages: Iterable[EvidencePassage],
    ) -> list[EvidenceRecord]:
        document_lookup = {document.id: document for document in documents}
        records: list[EvidenceRecord] = []
        for rank, passage in enumerate(ranked_passages, start=1):
            records.append(
                self.record_from_passage(
                    run_id=run_id,
                    query=query,
                    document=document_lookup[passage.source_id],
                    passage=passage,
                    rank=rank,
                )
            )
        return records
