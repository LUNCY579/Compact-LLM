from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict

import pytest

from rineya_significance_lab.benchmarks import (
    EVIDENCE_ANNOTATION_VERSION,
    EvidenceAlignmentError,
    EvidenceAnnotationError,
    EvidenceTruncationError,
    ExactEvidenceSpan,
    freeze_evidence_spans,
    locate_unique_evidence_spans,
    score_posthoc_evidence_retention,
    tokenize_and_freeze_evidence,
)


class CharacterOffsetTokenizer:
    """Small offset-contract fake; this is tokenization, never model inference."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        truncation: bool,
    ) -> Mapping[str, object]:
        assert not add_special_tokens
        assert return_offsets_mapping
        assert not truncation
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _annotation():
    prompt = "AA|first fact|BB|second fact|CC"
    spans = locate_unique_evidence_spans(
        prompt,
        {"fact-1": "first fact", "fact-2": "second fact"},
    )
    return tokenize_and_freeze_evidence(
        prompt,
        spans,
        CharacterOffsetTokenizer(),
        tokenizer_fingerprint="character-offset-tokenizer:test-v1",
    )


def test_exact_spans_freeze_deterministic_multi_token_fact_groups() -> None:
    prompt = "AA|first fact|BB|second fact|CC"
    first = locate_unique_evidence_spans(
        prompt,
        {"fact-2": "second fact", "fact-1": "first fact"},
    )
    second = locate_unique_evidence_spans(
        prompt,
        {"fact-1": "first fact", "fact-2": "second fact"},
    )
    left = tokenize_and_freeze_evidence(
        prompt,
        first,
        CharacterOffsetTokenizer(),
        tokenizer_fingerprint="character-offset-tokenizer:test-v1",
    )
    right = tokenize_and_freeze_evidence(
        prompt,
        second,
        CharacterOffsetTokenizer(),
        tokenizer_fingerprint="character-offset-tokenizer:test-v1",
    )

    assert left == right
    assert left.version == EVIDENCE_ANNOTATION_VERSION
    assert left.prompt_sha256 == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert [fact.fact_id for fact in left.facts] == ["fact-1", "fact-2"]
    assert all(len(fact.tokens) > 1 for fact in left.facts)
    for fact in left.facts:
        assert tuple(token.position for token in fact.tokens) == fact.token_positions
        for token in fact.tokens:
            assert prompt[token.start_char : token.end_char] == token.exact_text
    payload = {**asdict(left), "version": left.version}
    assert type(left).from_dict(payload) == left


def test_multiple_disjoint_spans_can_form_one_frozen_fact_group() -> None:
    prompt = "AA|alpha|BB|beta|CC"
    spans = (
        ExactEvidenceSpan("composite", 3, 8, "alpha"),
        ExactEvidenceSpan("composite", 12, 16, "beta"),
    )
    annotation = tokenize_and_freeze_evidence(
        prompt,
        spans,
        CharacterOffsetTokenizer(),
        tokenizer_fingerprint="character-offset-tokenizer:test-v1",
    )

    assert len(annotation.facts) == 1
    assert len(annotation.facts[0].character_spans) == 2
    assert annotation.facts[0].token_positions == (*range(3, 8), *range(12, 16))


@pytest.mark.parametrize(
    ("prompt", "fact_texts", "message"),
    [
        ("fact and fact", {"f": "fact"}, "ambiguous"),
        ("nothing here", {"f": "fact"}, "absent"),
    ],
)
def test_unique_locator_rejects_ambiguous_or_missing_text(
    prompt: str,
    fact_texts: Mapping[str, str],
    message: str,
) -> None:
    with pytest.raises(EvidenceAnnotationError, match=message):
        locate_unique_evidence_spans(prompt, fact_texts)


def test_freezer_rejects_text_mismatch_and_overlapping_annotations() -> None:
    prompt = "abcdef"
    with pytest.raises(EvidenceAnnotationError, match="text mismatch"):
        freeze_evidence_spans(
            prompt,
            [ExactEvidenceSpan("f", 1, 4, "xxx")],
            token_ids=range(len(prompt)),
            offset_mapping=((index, index + 1) for index in range(len(prompt))),
            tokenizer_fingerprint="test",
        )

    with pytest.raises(EvidenceAnnotationError, match="overlap"):
        freeze_evidence_spans(
            prompt,
            [
                ExactEvidenceSpan("f1", 0, 3, "abc"),
                ExactEvidenceSpan("f2", 2, 5, "cde"),
            ],
            token_ids=range(len(prompt)),
            offset_mapping=((index, index + 1) for index in range(len(prompt))),
            tokenizer_fingerprint="test",
        )


def test_freezer_rejects_truncation_offset_overlap_gap_and_cut_token() -> None:
    prompt = "alpha beta"
    span = ExactEvidenceSpan("f", 0, 5, "alpha")
    with pytest.raises(EvidenceTruncationError, match="truncates"):
        freeze_evidence_spans(
            prompt,
            [span],
            token_ids=range(len(prompt) - 1),
            offset_mapping=((index, index + 1) for index in range(len(prompt) - 1)),
            tokenizer_fingerprint="test",
        )
    with pytest.raises(EvidenceAlignmentError, match="overlaps"):
        freeze_evidence_spans(
            prompt,
            [span],
            token_ids=[1, 2],
            offset_mapping=[(0, 6), (5, len(prompt))],
            tokenizer_fingerprint="test",
        )
    with pytest.raises(EvidenceAlignmentError, match="unmapped"):
        freeze_evidence_spans(
            prompt,
            [span],
            token_ids=[1, 2],
            offset_mapping=[(0, 4), (5, len(prompt))],
            tokenizer_fingerprint="test",
        )
    with pytest.raises(EvidenceAlignmentError, match="boundary cuts token"):
        freeze_evidence_spans(
            prompt,
            [ExactEvidenceSpan("f", 1, 5, "lpha")],
            token_ids=[1, 2, 3],
            offset_mapping=[(0, 5), (5, 6), (6, len(prompt))],
            tokenizer_fingerprint="test",
        )


def test_strict_case_endpoint_unions_heads_per_layer_but_not_layers() -> None:
    annotation = _annotation()
    required = annotation.required_token_positions
    result = score_posthoc_evidence_retention(
        annotation,
        [
            [required[::2], required[1::2]],
            [required, ()],
        ],
        evaluated_layer_indices=[3, 17],
        evaluated_kv_head_indices=[2, 7],
    )

    assert (
        result.strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer
        is True
    )
    assert result.fraction_of_evaluated_layers_with_all_required_fact_tokens_available == 1.0
    assert result.fraction_of_evaluated_layers_with_one_kv_head_containing_all_required_fact_tokens == 0.5
    assert result.mean_per_kv_head_required_token_retention_fraction == 0.5
    assert [layer.layer_index for layer in result.layers] == [3, 17]
    assert [head.kv_head_index for head in result.layers[0].kv_heads] == [2, 7]
    assert not result.layers[0].one_kv_head_contains_all_required_fact_tokens
    assert result.layers[0].all_required_fact_tokens_available_in_at_least_one_kv_head
    record = result.to_dict()
    assert (
        record[
            "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer"
        ]
        is True
    )


def test_missing_one_token_fails_strict_endpoint_and_reports_exact_fractions() -> None:
    annotation = _annotation()
    required = annotation.required_token_positions
    result = score_posthoc_evidence_retention(
        annotation,
        [
            [required[::2], required[1::2]],
            [required[1:], ()],
        ],
    )

    assert not (
        result.strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer
    )
    assert result.fraction_of_evaluated_layers_with_all_required_fact_tokens_available == 0.5
    assert result.minimum_per_layer_required_token_availability_fraction == pytest.approx(
        (len(required) - 1) / len(required)
    )
    assert result.fraction_of_required_facts_complete_in_every_evaluated_layer == 0.5


@pytest.mark.parametrize(
    ("resident", "message"),
    [
        ([], "at least one evaluated layer"),
        ([[[0]], []], "same number of KV heads"),
        ([[[0, 0]]], "duplicate resident positions"),
        ([[[-1]]], "negative/inactive"),
    ],
)
def test_scorer_rejects_incomplete_or_invalid_resident_position_shapes(
    resident: list[object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        score_posthoc_evidence_retention(_annotation(), resident)
