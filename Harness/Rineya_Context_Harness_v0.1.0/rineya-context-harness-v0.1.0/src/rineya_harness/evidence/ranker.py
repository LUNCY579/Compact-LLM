from __future__ import annotations

import math
import re
from collections import Counter

from rineya_harness.domain.models import EvidencePassage, FetchedDocument

_TOKEN = re.compile(r"[A-Za-z0-9_'-]+")


def _terms(text: str) -> list[str]:
    return [term.casefold() for term in _TOKEN.findall(text)]


class LexicalPassageRanker:
    """Dependency-light BM25-style ranker for local CPU execution."""

    def __init__(self, *, passage_chars: int = 900, overlap_chars: int = 120) -> None:
        if passage_chars <= overlap_chars:
            raise ValueError("passage_chars must exceed overlap_chars")
        self.passage_chars = passage_chars
        self.overlap_chars = overlap_chars

    def split(self, document: FetchedDocument) -> list[EvidencePassage]:
        passages: list[EvidencePassage] = []
        start = 0
        index = 0
        text = document.text
        while start < len(text):
            end = min(len(text), start + self.passage_chars)
            if end < len(text):
                boundary = text.rfind("\n", start + self.passage_chars // 2, end)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                passages.append(
                    EvidencePassage(
                        id=f"{document.id}_p{index:04d}",
                        source_id=document.id,
                        text=chunk,
                        start_offset=start,
                        end_offset=end,
                    )
                )
                index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - self.overlap_chars)
        return passages

    def rank(
        self,
        query: str,
        passages: list[EvidencePassage],
        *,
        limit: int = 12,
    ) -> list[EvidencePassage]:
        query_terms = Counter(_terms(query))
        if not query_terms:
            return passages[:limit]
        documents = [_terms(passage.text) for passage in passages]
        document_frequency: Counter[str] = Counter()
        for terms in documents:
            document_frequency.update(set(terms))
        total_documents = max(1, len(documents))
        average_length = (
            sum(len(document) for document in documents) / total_documents if documents else 1.0
        )
        scored: list[EvidencePassage] = []
        k1, b = 1.2, 0.75
        for passage, terms in zip(passages, documents, strict=True):
            term_frequency = Counter(terms)
            score = 0.0
            for term, query_weight in query_terms.items():
                frequency = term_frequency.get(term, 0)
                if frequency == 0:
                    continue
                df = document_frequency.get(term, 0)
                inverse_document_frequency = math.log(
                    1 + (total_documents - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * (len(terms) / max(1.0, average_length))
                )
                score += query_weight * inverse_document_frequency * (
                    frequency * (k1 + 1) / denominator
                )
            scored.append(passage.model_copy(update={"score": score}))
        scored.sort(key=lambda item: (-item.score, item.source_id, item.start_offset))
        return scored[:limit]
