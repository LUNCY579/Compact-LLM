"""Exact, post-hoc token evidence annotation and KV-retention scoring.

This module deliberately has no model/runtime dependency.  An evidence annotation is
frozen from the rendered prompt and tokenizer offsets before conditions run; the scorer
then consumes only the resident absolute token positions produced by a completed policy.
It never returns a selection mask, so oracle annotations have no policy-facing API here.
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar, Protocol, SupportsIndex, cast

EVIDENCE_ANNOTATION_VERSION = "exact-token-evidence-v1"


class EvidenceAnnotationError(ValueError):
    """The supplied semantic evidence annotation is not exact or unambiguous."""


class EvidenceAlignmentError(ValueError):
    """Tokenizer offsets cannot support an exact character-to-token mapping."""


class EvidenceTruncationError(EvidenceAlignmentError):
    """Tokenizer output does not cover the complete rendered prompt."""


class OffsetTokenizer(Protocol):
    """Minimal fast-tokenizer interface needed to freeze evidence annotations."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> Mapping[str, object]: ...


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from exc
    return int(result)


def _items(value: object, *, name: str) -> list[object]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of values, not text")
    try:
        return list(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_token_ids(token_ids: tuple[int, ...]) -> str:
    canonical = json.dumps(token_ids, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class ExactEvidenceSpan:
    """One exact, half-open character span in the final rendered prompt."""

    fact_id: str
    start_char: int
    end_char: int
    exact_text: str

    def __post_init__(self) -> None:
        if not self.fact_id or self.fact_id != self.fact_id.strip():
            raise EvidenceAnnotationError("fact_id must be non-empty with no outer whitespace")
        if isinstance(self.start_char, bool) or isinstance(self.end_char, bool):
            raise TypeError("character offsets must be integers, not bool")
        if not isinstance(self.start_char, int) or not isinstance(self.end_char, int):
            raise TypeError("character offsets must be integers")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise EvidenceAnnotationError("evidence spans must be non-empty half-open ranges")
        if not self.exact_text:
            raise EvidenceAnnotationError("evidence span text must be non-empty")
        if len(self.exact_text) != self.end_char - self.start_char:
            raise EvidenceAnnotationError(
                "evidence span length does not equal its exact_text character length"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExactEvidenceSpan:
        return cls(
            fact_id=str(payload["fact_id"]),
            start_char=_integer(payload["start_char"], name="start_char"),
            end_char=_integer(payload["end_char"], name="end_char"),
            exact_text=str(payload["exact_text"]),
        )


@dataclass(frozen=True, slots=True)
class FrozenEvidenceToken:
    """An immutable required prompt token and its exact tokenizer offset."""

    position: int
    token_id: int
    start_char: int
    end_char: int
    exact_text: str

    def __post_init__(self) -> None:
        if self.position < 0 or self.token_id < 0:
            raise EvidenceAlignmentError("evidence token positions and IDs must be non-negative")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise EvidenceAlignmentError("evidence token offsets must be non-empty")
        if len(self.exact_text) != self.end_char - self.start_char:
            raise EvidenceAlignmentError("evidence token text does not match its offset length")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FrozenEvidenceToken:
        return cls(
            position=_integer(payload["position"], name="position"),
            token_id=_integer(payload["token_id"], name="token_id"),
            start_char=_integer(payload["start_char"], name="start_char"),
            end_char=_integer(payload["end_char"], name="end_char"),
            exact_text=str(payload["exact_text"]),
        )


@dataclass(frozen=True, slots=True)
class FrozenEvidenceFact:
    """A required fact frozen as one atomic, potentially multi-token group."""

    fact_id: str
    character_spans: tuple[ExactEvidenceSpan, ...]
    tokens: tuple[FrozenEvidenceToken, ...]

    def __post_init__(self) -> None:
        if not self.fact_id:
            raise EvidenceAnnotationError("frozen fact_id must be non-empty")
        if not self.character_spans or not self.tokens:
            raise EvidenceAnnotationError("every frozen fact needs a span and at least one token")
        if any(span.fact_id != self.fact_id for span in self.character_spans):
            raise EvidenceAnnotationError("frozen fact contains a span for another fact_id")
        spans = tuple(sorted(self.character_spans, key=lambda span: (span.start_char, span.end_char)))
        if spans != self.character_spans:
            raise EvidenceAnnotationError("frozen fact spans must be sorted")
        if any(left.end_char > right.start_char for left, right in pairwise(spans)):
            raise EvidenceAnnotationError("frozen fact spans overlap")
        positions = tuple(token.position for token in self.tokens)
        if positions != tuple(sorted(positions)) or len(positions) != len(set(positions)):
            raise EvidenceAlignmentError("frozen fact token positions must be sorted and unique")
        for token in self.tokens:
            containing = [
                span
                for span in spans
                if span.start_char <= token.start_char and token.end_char <= span.end_char
            ]
            if len(containing) != 1:
                raise EvidenceAlignmentError(
                    f"fact {self.fact_id!r} token {token.position} is not inside exactly one span"
                )
        for span in spans:
            covered = [
                token
                for token in self.tokens
                if span.start_char <= token.start_char and token.end_char <= span.end_char
            ]
            if not covered or covered[0].start_char != span.start_char:
                raise EvidenceAlignmentError(f"fact {self.fact_id!r} span has an uncovered prefix")
            if covered[-1].end_char != span.end_char:
                raise EvidenceAlignmentError(f"fact {self.fact_id!r} span has an uncovered suffix")
            if any(left.end_char != right.start_char for left, right in pairwise(covered)):
                raise EvidenceAlignmentError(f"fact {self.fact_id!r} span has an offset gap")

    @property
    def token_positions(self) -> tuple[int, ...]:
        return tuple(token.position for token in self.tokens)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FrozenEvidenceFact:
        spans = _items(payload["character_spans"], name="character_spans")
        tokens = _items(payload["tokens"], name="tokens")
        return cls(
            fact_id=str(payload["fact_id"]),
            character_spans=tuple(
                ExactEvidenceSpan.from_dict(cast(Mapping[str, object], row)) for row in spans
            ),
            tokens=tuple(FrozenEvidenceToken.from_dict(cast(Mapping[str, object], row)) for row in tokens),
        )


@dataclass(frozen=True, slots=True)
class FrozenEvidenceAnnotation:
    """Auditable evidence facts tied to one exact prompt/tokenizer rendering."""

    tokenizer_fingerprint: str
    prompt_sha256: str
    token_ids_sha256: str
    token_count: int
    facts: tuple[FrozenEvidenceFact, ...]

    version: ClassVar[str] = EVIDENCE_ANNOTATION_VERSION

    def __post_init__(self) -> None:
        if not self.tokenizer_fingerprint:
            raise EvidenceAnnotationError("tokenizer_fingerprint must be non-empty")
        if not _is_sha256(self.prompt_sha256) or not _is_sha256(self.token_ids_sha256):
            raise EvidenceAnnotationError("annotation hashes must be lowercase SHA-256 values")
        if self.token_count <= 0:
            raise EvidenceAnnotationError("token_count must be positive")
        if not self.facts:
            raise EvidenceAnnotationError("at least one required evidence fact is needed")
        fact_ids = tuple(fact.fact_id for fact in self.facts)
        if len(fact_ids) != len(set(fact_ids)):
            raise EvidenceAnnotationError("frozen fact IDs must be unique")
        span_rows = sorted(
            (span for fact in self.facts for span in fact.character_spans),
            key=lambda span: (span.start_char, span.end_char, span.fact_id),
        )
        if any(left.end_char > right.start_char for left, right in pairwise(span_rows)):
            raise EvidenceAnnotationError("frozen evidence spans overlap")
        positions = [token.position for fact in self.facts for token in fact.tokens]
        if len(positions) != len(set(positions)):
            raise EvidenceAlignmentError("one tokenizer position belongs to multiple evidence facts")
        if min(positions) < 0 or max(positions) >= self.token_count:
            raise EvidenceAlignmentError("evidence token position is outside the rendered prompt")

    @property
    def required_token_positions(self) -> tuple[int, ...]:
        """Sorted union of all required fact-token positions."""

        return tuple(sorted(token.position for fact in self.facts for token in fact.tokens))

    @property
    def required_token_count(self) -> int:
        return len(self.required_token_positions)

    @property
    def required_fact_count(self) -> int:
        return len(self.facts)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FrozenEvidenceAnnotation:
        version = str(payload.get("version", ""))
        if version != EVIDENCE_ANNOTATION_VERSION:
            raise EvidenceAnnotationError(f"unsupported evidence annotation version: {version!r}")
        facts = _items(payload["facts"], name="facts")
        return cls(
            tokenizer_fingerprint=str(payload["tokenizer_fingerprint"]),
            prompt_sha256=str(payload["prompt_sha256"]),
            token_ids_sha256=str(payload["token_ids_sha256"]),
            token_count=_integer(payload["token_count"], name="token_count"),
            facts=tuple(FrozenEvidenceFact.from_dict(cast(Mapping[str, object], row)) for row in facts),
        )


def locate_unique_evidence_spans(
    rendered_prompt: str,
    fact_texts: Mapping[str, str],
) -> tuple[ExactEvidenceSpan, ...]:
    """Locate each exact fact text once, rejecting missing or repeated matches.

    Explicit offsets remain the authoritative annotation.  This convenience function
    creates them only when text matching is provably unique in the rendered prompt.
    """

    if not rendered_prompt:
        raise EvidenceAnnotationError("rendered_prompt must be non-empty")
    if not fact_texts:
        raise EvidenceAnnotationError("fact_texts must be non-empty")
    spans: list[ExactEvidenceSpan] = []
    for fact_id, exact_text in fact_texts.items():
        if not exact_text:
            raise EvidenceAnnotationError(f"fact {fact_id!r} has empty exact text")
        first = rendered_prompt.find(exact_text)
        if first < 0:
            raise EvidenceAnnotationError(f"fact {fact_id!r} text is absent from rendered prompt")
        second = rendered_prompt.find(exact_text, first + 1)
        if second >= 0:
            raise EvidenceAnnotationError(
                f"fact {fact_id!r} text is ambiguous: it occurs more than once"
            )
        spans.append(ExactEvidenceSpan(fact_id, first, first + len(exact_text), exact_text))
    return _validated_spans(rendered_prompt, spans)


def _validated_spans(
    rendered_prompt: str,
    spans: Iterable[ExactEvidenceSpan],
) -> tuple[ExactEvidenceSpan, ...]:
    rows = tuple(sorted(spans, key=lambda span: (span.start_char, span.end_char, span.fact_id)))
    if not rows:
        raise EvidenceAnnotationError("at least one exact evidence span is required")
    for span in rows:
        if span.end_char > len(rendered_prompt):
            raise EvidenceAnnotationError(
                f"fact {span.fact_id!r} span ends outside the rendered prompt"
            )
        observed = rendered_prompt[span.start_char : span.end_char]
        if observed != span.exact_text:
            raise EvidenceAnnotationError(
                f"fact {span.fact_id!r} span text mismatch: {observed!r} != {span.exact_text!r}"
            )
    for left, right in pairwise(rows):
        if left.end_char > right.start_char:
            raise EvidenceAnnotationError(
                f"evidence spans overlap: {left.fact_id!r} and {right.fact_id!r}"
            )
    return rows


def _validated_tokenization(
    rendered_prompt: str,
    token_ids: Iterable[object],
    offset_mapping: Iterable[object],
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    ids = tuple(_integer(value, name="token_id") for value in token_ids)
    raw_offsets = list(offset_mapping)
    offsets: list[tuple[int, int]] = []
    for position, raw in enumerate(raw_offsets):
        pair = _items(raw, name=f"offset_mapping[{position}]")
        if len(pair) != 2:
            raise EvidenceAlignmentError(
                f"offset_mapping[{position}] must contain exactly two integers"
            )
        start = _integer(pair[0], name=f"offset_mapping[{position}][0]")
        end = _integer(pair[1], name=f"offset_mapping[{position}][1]")
        offsets.append((start, end))
    if len(ids) != len(offsets):
        raise EvidenceAlignmentError(
            f"token/offset length mismatch: {len(ids)} token IDs and {len(offsets)} offsets"
        )
    if not ids:
        raise EvidenceTruncationError("tokenizer returned no tokens for a non-empty prompt")
    if any(token_id < 0 for token_id in ids):
        raise EvidenceAlignmentError("token IDs must be non-negative")

    cursor = 0
    for position, (start, end) in enumerate(offsets):
        if start < 0 or end <= start or end > len(rendered_prompt):
            raise EvidenceAlignmentError(
                f"token {position} has invalid or empty character offset {(start, end)}"
            )
        if start < cursor:
            raise EvidenceAlignmentError(
                f"token {position} offset {(start, end)} overlaps the previous token offset"
            )
        if start > cursor:
            raise EvidenceAlignmentError(
                f"tokenizer offsets leave characters [{cursor}, {start}) unmapped"
            )
        cursor = end
    if cursor < len(rendered_prompt):
        raise EvidenceTruncationError(
            f"tokenizer output truncates rendered prompt at character {cursor} of "
            f"{len(rendered_prompt)}"
        )
    return ids, tuple(offsets)


def freeze_evidence_spans(
    rendered_prompt: str,
    spans: Iterable[ExactEvidenceSpan],
    *,
    token_ids: Iterable[object],
    offset_mapping: Iterable[object],
    tokenizer_fingerprint: str,
) -> FrozenEvidenceAnnotation:
    """Freeze exact spans as required token groups using full-prompt offsets.

    Token offsets must form an exact, non-overlapping partition of the complete rendered
    prompt.  An annotation boundary that cuts through a token is rejected rather than
    expanding the oracle span implicitly.
    """

    if not rendered_prompt:
        raise EvidenceAnnotationError("rendered_prompt must be non-empty")
    if not tokenizer_fingerprint:
        raise EvidenceAnnotationError("tokenizer_fingerprint must be non-empty")
    span_rows = _validated_spans(rendered_prompt, spans)
    ids, offsets = _validated_tokenization(rendered_prompt, token_ids, offset_mapping)

    by_fact: defaultdict[str, list[ExactEvidenceSpan]] = defaultdict(list)
    for span in span_rows:
        by_fact[span.fact_id].append(span)

    facts: list[FrozenEvidenceFact] = []
    ordered_fact_ids = sorted(
        by_fact,
        key=lambda fact_id: (by_fact[fact_id][0].start_char, fact_id),
    )
    for fact_id in ordered_fact_ids:
        fact_spans = tuple(by_fact[fact_id])
        fact_tokens: list[FrozenEvidenceToken] = []
        for span in fact_spans:
            overlaps = [
                position
                for position, (start, end) in enumerate(offsets)
                if start < span.end_char and end > span.start_char
            ]
            if not overlaps:
                raise EvidenceAlignmentError(f"fact {fact_id!r} maps to no tokenizer tokens")
            for position in overlaps:
                start, end = offsets[position]
                if start < span.start_char or end > span.end_char:
                    raise EvidenceAlignmentError(
                        f"fact {fact_id!r} boundary cuts token {position} offset {(start, end)}"
                    )
                fact_tokens.append(
                    FrozenEvidenceToken(
                        position=position,
                        token_id=ids[position],
                        start_char=start,
                        end_char=end,
                        exact_text=rendered_prompt[start:end],
                    )
                )
        facts.append(FrozenEvidenceFact(fact_id, fact_spans, tuple(fact_tokens)))

    return FrozenEvidenceAnnotation(
        tokenizer_fingerprint=tokenizer_fingerprint,
        prompt_sha256=_sha256_text(rendered_prompt),
        token_ids_sha256=_sha256_token_ids(ids),
        token_count=len(ids),
        facts=tuple(facts),
    )


def tokenize_and_freeze_evidence(
    rendered_prompt: str,
    spans: Iterable[ExactEvidenceSpan],
    tokenizer: OffsetTokenizer,
    *,
    tokenizer_fingerprint: str,
) -> FrozenEvidenceAnnotation:
    """Tokenize once without truncation and freeze an exact evidence annotation."""

    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer output must be a mapping")
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise EvidenceAlignmentError("tokenizer must return input_ids and offset_mapping")
    return freeze_evidence_spans(
        rendered_prompt,
        spans,
        token_ids=_items(encoded["input_ids"], name="input_ids"),
        offset_mapping=_items(encoded["offset_mapping"], name="offset_mapping"),
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class FactTokenRetention:
    """Retention of one frozen fact within one head or a layer-wide head union."""

    fact_id: str
    retained_token_count: int
    required_token_count: int
    token_retention_fraction: float
    all_fact_tokens_retained: bool


@dataclass(frozen=True, slots=True)
class KVHeadEvidenceRetention:
    """Post-hoc evidence retention for one evaluated KV head."""

    layer_index: int
    kv_head_index: int
    retained_required_token_count: int
    required_token_count: int
    required_token_retention_fraction: float
    complete_fact_count: int
    required_fact_count: int
    required_fact_completion_fraction: float
    all_required_fact_tokens_retained: bool
    facts: tuple[FactTokenRetention, ...]


@dataclass(frozen=True, slots=True)
class LayerEvidenceRetention:
    """Evidence availability after unioning resident positions across KV heads."""

    layer_index: int
    available_required_token_count: int
    required_token_count: int
    required_token_availability_fraction: float
    complete_fact_count: int
    required_fact_count: int
    required_fact_completion_fraction: float
    all_required_fact_tokens_available_in_at_least_one_kv_head: bool
    one_kv_head_contains_all_required_fact_tokens: bool
    kv_heads: tuple[KVHeadEvidenceRetention, ...]
    facts: tuple[FactTokenRetention, ...]


@dataclass(frozen=True, slots=True)
class EvidenceRetentionScore:
    """Strict case endpoint plus transparent head/layer diagnostic fractions."""

    strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer: bool
    evaluated_layer_count: int
    evaluated_kv_head_count_per_layer: int
    required_fact_count: int
    required_token_count: int
    mean_per_kv_head_required_token_retention_fraction: float
    mean_per_kv_head_required_fact_completion_fraction: float
    mean_per_layer_required_token_availability_fraction: float
    minimum_per_layer_required_token_availability_fraction: float
    mean_per_layer_required_fact_completion_fraction: float
    fraction_of_evaluated_layers_with_all_required_fact_tokens_available: float
    fraction_of_evaluated_layers_with_one_kv_head_containing_all_required_fact_tokens: float
    fraction_of_required_facts_complete_in_every_evaluated_layer: float
    layers: tuple[LayerEvidenceRetention, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record with the strict endpoint explicitly named."""

        return {
            "endpoint_definition": (
                "for every evaluated layer and every required fact token, at least one "
                "KV head retains its absolute rendered-prompt token position"
            ),
            "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer": (
                self.strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer
            ),
            "evaluated_layer_count": self.evaluated_layer_count,
            "evaluated_kv_head_count_per_layer": self.evaluated_kv_head_count_per_layer,
            "required_fact_count": self.required_fact_count,
            "required_token_count": self.required_token_count,
            "mean_per_kv_head_required_token_retention_fraction": (
                self.mean_per_kv_head_required_token_retention_fraction
            ),
            "mean_per_kv_head_required_fact_completion_fraction": (
                self.mean_per_kv_head_required_fact_completion_fraction
            ),
            "mean_per_layer_required_token_availability_fraction": (
                self.mean_per_layer_required_token_availability_fraction
            ),
            "minimum_per_layer_required_token_availability_fraction": (
                self.minimum_per_layer_required_token_availability_fraction
            ),
            "mean_per_layer_required_fact_completion_fraction": (
                self.mean_per_layer_required_fact_completion_fraction
            ),
            "fraction_of_evaluated_layers_with_all_required_fact_tokens_available": (
                self.fraction_of_evaluated_layers_with_all_required_fact_tokens_available
            ),
            "fraction_of_evaluated_layers_with_one_kv_head_containing_all_required_fact_tokens": (
                self.fraction_of_evaluated_layers_with_one_kv_head_containing_all_required_fact_tokens
            ),
            "fraction_of_required_facts_complete_in_every_evaluated_layer": (
                self.fraction_of_required_facts_complete_in_every_evaluated_layer
            ),
            "layers": [
                {
                    "layer_index": layer.layer_index,
                    "available_required_token_count": layer.available_required_token_count,
                    "required_token_count": layer.required_token_count,
                    "required_token_availability_fraction": (
                        layer.required_token_availability_fraction
                    ),
                    "complete_fact_count": layer.complete_fact_count,
                    "required_fact_count": layer.required_fact_count,
                    "required_fact_completion_fraction": layer.required_fact_completion_fraction,
                    "all_required_fact_tokens_available_in_at_least_one_kv_head": (
                        layer.all_required_fact_tokens_available_in_at_least_one_kv_head
                    ),
                    "one_kv_head_contains_all_required_fact_tokens": (
                        layer.one_kv_head_contains_all_required_fact_tokens
                    ),
                    "facts": [
                        {
                            "fact_id": fact.fact_id,
                            "retained_token_count": fact.retained_token_count,
                            "required_token_count": fact.required_token_count,
                            "token_retention_fraction": fact.token_retention_fraction,
                            "all_fact_tokens_retained": fact.all_fact_tokens_retained,
                        }
                        for fact in layer.facts
                    ],
                    "kv_heads": [
                        {
                            "kv_head_index": head.kv_head_index,
                            "retained_required_token_count": (
                                head.retained_required_token_count
                            ),
                            "required_token_count": head.required_token_count,
                            "required_token_retention_fraction": (
                                head.required_token_retention_fraction
                            ),
                            "complete_fact_count": head.complete_fact_count,
                            "required_fact_count": head.required_fact_count,
                            "required_fact_completion_fraction": (
                                head.required_fact_completion_fraction
                            ),
                            "all_required_fact_tokens_retained": (
                                head.all_required_fact_tokens_retained
                            ),
                            "facts": [
                                {
                                    "fact_id": fact.fact_id,
                                    "retained_token_count": fact.retained_token_count,
                                    "required_token_count": fact.required_token_count,
                                    "token_retention_fraction": fact.token_retention_fraction,
                                    "all_fact_tokens_retained": fact.all_fact_tokens_retained,
                                }
                                for fact in head.facts
                            ],
                        }
                        for head in layer.kv_heads
                    ],
                }
                for layer in self.layers
            ],
        }


def _fact_retention(
    annotation: FrozenEvidenceAnnotation,
    available: set[int],
) -> tuple[FactTokenRetention, ...]:
    rows: list[FactTokenRetention] = []
    for fact in annotation.facts:
        required = set(fact.token_positions)
        retained = len(required & available)
        total = len(required)
        rows.append(
            FactTokenRetention(
                fact_id=fact.fact_id,
                retained_token_count=retained,
                required_token_count=total,
                token_retention_fraction=retained / total,
                all_fact_tokens_retained=retained == total,
            )
        )
    return tuple(rows)


def _indices(
    supplied: Iterable[object] | None,
    *,
    expected_count: int,
    name: str,
) -> tuple[int, ...]:
    if supplied is None:
        return tuple(range(expected_count))
    values = tuple(_integer(value, name=name) for value in supplied)
    if len(values) != expected_count:
        raise ValueError(f"{name} has {len(values)} values; expected {expected_count}")
    if any(value < 0 for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique non-negative indices")
    return values


def _resident_positions(raw: object, *, label: str) -> tuple[int, ...]:
    values = tuple(_integer(value, name=label) for value in _items(raw, name=label))
    if any(value < 0 for value in values):
        raise ValueError(f"{label} contains a negative/inactive position")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate resident positions")
    return tuple(sorted(values))


def score_posthoc_evidence_retention(
    annotation: FrozenEvidenceAnnotation,
    retained_positions_by_layer_and_kv_head: Iterable[object],
    *,
    evaluated_layer_indices: Iterable[object] | None = None,
    evaluated_kv_head_indices: Iterable[object] | None = None,
) -> EvidenceRetentionScore:
    """Score frozen fact tokens against post-policy resident absolute positions.

    Input shape is ``[evaluated layers][KV heads][resident positions]``.  For the
    strict endpoint, heads are unioned *within* each layer: every required token must
    occur in at least one head in every evaluated layer, but tokens may occur in
    different heads.  Requiring one head to contain the entire set is diagnostic only.
    """

    raw_layers = _items(
        retained_positions_by_layer_and_kv_head,
        name="retained_positions_by_layer_and_kv_head",
    )
    if not raw_layers:
        raise ValueError("at least one evaluated layer is required")
    materialized_heads = [
        _items(layer, name=f"retained_positions_by_layer_and_kv_head[{layer_number}]")
        for layer_number, layer in enumerate(raw_layers)
    ]
    head_counts = {len(heads) for heads in materialized_heads}
    if head_counts == {0}:
        raise ValueError("at least one evaluated KV head is required per layer")
    if len(head_counts) != 1:
        raise ValueError("evaluated layers must all contain the same number of KV heads")
    head_count = head_counts.pop()
    if head_count <= 0:
        raise ValueError("at least one evaluated KV head is required per layer")

    layer_indices = _indices(
        evaluated_layer_indices,
        expected_count=len(raw_layers),
        name="evaluated_layer_indices",
    )
    head_indices = _indices(
        evaluated_kv_head_indices,
        expected_count=head_count,
        name="evaluated_kv_head_indices",
    )
    required_positions = set(annotation.required_token_positions)
    required_token_count = len(required_positions)
    required_fact_count = len(annotation.facts)

    layer_rows: list[LayerEvidenceRetention] = []
    all_head_token_fractions: list[float] = []
    all_head_fact_fractions: list[float] = []
    for layer_offset, raw_heads in enumerate(materialized_heads):
        layer_index = layer_indices[layer_offset]
        head_sets: list[set[int]] = []
        head_rows: list[KVHeadEvidenceRetention] = []
        for head_offset, raw_positions in enumerate(raw_heads):
            head_index = head_indices[head_offset]
            positions = set(
                _resident_positions(
                    raw_positions,
                    label=f"layer {layer_index} KV head {head_index} positions",
                )
            )
            head_sets.append(positions)
            facts = _fact_retention(annotation, positions)
            retained_count = len(required_positions & positions)
            token_fraction = retained_count / required_token_count
            complete_fact_count = sum(fact.all_fact_tokens_retained for fact in facts)
            fact_fraction = complete_fact_count / required_fact_count
            head_rows.append(
                KVHeadEvidenceRetention(
                    layer_index=layer_index,
                    kv_head_index=head_index,
                    retained_required_token_count=retained_count,
                    required_token_count=required_token_count,
                    required_token_retention_fraction=token_fraction,
                    complete_fact_count=complete_fact_count,
                    required_fact_count=required_fact_count,
                    required_fact_completion_fraction=fact_fraction,
                    all_required_fact_tokens_retained=retained_count == required_token_count,
                    facts=facts,
                )
            )
            all_head_token_fractions.append(token_fraction)
            all_head_fact_fractions.append(fact_fraction)

        available = set().union(*head_sets)
        layer_facts = _fact_retention(annotation, available)
        available_count = len(required_positions & available)
        complete_fact_count = sum(fact.all_fact_tokens_retained for fact in layer_facts)
        layer_rows.append(
            LayerEvidenceRetention(
                layer_index=layer_index,
                available_required_token_count=available_count,
                required_token_count=required_token_count,
                required_token_availability_fraction=available_count / required_token_count,
                complete_fact_count=complete_fact_count,
                required_fact_count=required_fact_count,
                required_fact_completion_fraction=complete_fact_count / required_fact_count,
                all_required_fact_tokens_available_in_at_least_one_kv_head=(
                    available_count == required_token_count
                ),
                one_kv_head_contains_all_required_fact_tokens=any(
                    required_positions <= head_positions for head_positions in head_sets
                ),
                kv_heads=tuple(head_rows),
                facts=layer_facts,
            )
        )

    layer_token_fractions = [row.required_token_availability_fraction for row in layer_rows]
    layer_fact_fractions = [row.required_fact_completion_fraction for row in layer_rows]
    complete_layer_count = sum(
        row.all_required_fact_tokens_available_in_at_least_one_kv_head for row in layer_rows
    )
    single_head_complete_layer_count = sum(
        row.one_kv_head_contains_all_required_fact_tokens for row in layer_rows
    )
    facts_complete_everywhere = sum(
        all(layer.facts[fact_offset].all_fact_tokens_retained for layer in layer_rows)
        for fact_offset in range(required_fact_count)
    )
    layer_count = len(layer_rows)
    return EvidenceRetentionScore(
        strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer=(
            complete_layer_count == layer_count
        ),
        evaluated_layer_count=layer_count,
        evaluated_kv_head_count_per_layer=head_count,
        required_fact_count=required_fact_count,
        required_token_count=required_token_count,
        mean_per_kv_head_required_token_retention_fraction=(
            sum(all_head_token_fractions) / len(all_head_token_fractions)
        ),
        mean_per_kv_head_required_fact_completion_fraction=(
            sum(all_head_fact_fractions) / len(all_head_fact_fractions)
        ),
        mean_per_layer_required_token_availability_fraction=(
            sum(layer_token_fractions) / layer_count
        ),
        minimum_per_layer_required_token_availability_fraction=min(layer_token_fractions),
        mean_per_layer_required_fact_completion_fraction=(
            sum(layer_fact_fractions) / layer_count
        ),
        fraction_of_evaluated_layers_with_all_required_fact_tokens_available=(
            complete_layer_count / layer_count
        ),
        fraction_of_evaluated_layers_with_one_kv_head_containing_all_required_fact_tokens=(
            single_head_complete_layer_count / layer_count
        ),
        fraction_of_required_facts_complete_in_every_evaluated_layer=(
            facts_complete_everywhere / required_fact_count
        ),
        layers=tuple(layer_rows),
    )
