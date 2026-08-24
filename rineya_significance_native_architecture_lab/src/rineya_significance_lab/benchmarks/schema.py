"""Machine-readable synthetic benchmark records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SyntheticSegment:
    segment_id: str
    text: str
    role: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SyntheticSegment:
        return cls(
            segment_id=str(payload["segment_id"]),
            text=str(payload["text"]),
            role=str(payload["role"]),
        )


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    case_id: str
    family: str
    seed: int
    requested_evidence_depth: float
    actual_evidence_depth: float
    segments: tuple[SyntheticSegment, ...]
    question: str
    answer: str
    evidence_segment_ids: tuple[str, ...]
    distractor_segment_ids: tuple[str, ...]
    stale_segment_ids: tuple[str, ...]
    oracle: Mapping[str, Any]
    generator_version: str = "1.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["segments"] = [asdict(segment) for segment in self.segments]
        payload["evidence_segment_ids"] = list(self.evidence_segment_ids)
        payload["distractor_segment_ids"] = list(self.distractor_segment_ids)
        payload["stale_segment_ids"] = list(self.stale_segment_ids)
        payload["oracle"] = dict(self.oracle)
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SyntheticCase:
        return cls(
            case_id=str(payload["case_id"]),
            family=str(payload["family"]),
            seed=int(payload["seed"]),
            requested_evidence_depth=float(payload["requested_evidence_depth"]),
            actual_evidence_depth=float(payload["actual_evidence_depth"]),
            segments=tuple(SyntheticSegment.from_dict(row) for row in payload["segments"]),
            question=str(payload["question"]),
            answer=str(payload["answer"]),
            evidence_segment_ids=tuple(str(value) for value in payload["evidence_segment_ids"]),
            distractor_segment_ids=tuple(str(value) for value in payload["distractor_segment_ids"]),
            stale_segment_ids=tuple(str(value) for value in payload["stale_segment_ids"]),
            oracle=dict(payload["oracle"]),
            generator_version=str(payload.get("generator_version", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def prompt(self) -> str:
        context = "\n".join(f"[{segment.segment_id}] {segment.text}" for segment in self.segments)
        return f"CONTEXT\n{context}\n\nQUESTION\n{self.question}\n\nANSWER"
