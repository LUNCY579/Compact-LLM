from __future__ import annotations

from dataclasses import replace

import pytest

from rineya_significance_lab.benchmarks.evidence import (
    EvidenceRetentionScore,
    LayerEvidenceRetention,
)
from rineya_significance_lab.benchmarks.evidence import (
    FactTokenRetention as RuntimeFactTokenRetention,
)
from rineya_significance_lab.evaluation.evidence_endpoints import (
    EVIDENCE_RETENTION_ENDPOINT_VERSION,
    EvidenceEndpointError,
    FactTokenRecall,
    LayerFactRecall,
    score_evidence_retention_endpoint,
    score_layer_fact_retention,
    score_serialized_evidence_retention,
)


def _layers() -> tuple[LayerFactRecall, ...]:
    return (
        LayerFactRecall(
            layer_index=3,
            facts=(
                FactTokenRecall("fact-a", retained_token_count=2, required_token_count=2),
                FactTokenRecall("fact-b", retained_token_count=1, required_token_count=2),
            ),
        ),
        LayerFactRecall(
            layer_index=17,
            facts=(
                FactTokenRecall("fact-a", retained_token_count=2, required_token_count=2),
                FactTokenRecall("fact-b", retained_token_count=2, required_token_count=2),
            ),
        ),
    )


def test_primary_endpoint_is_exact_layer_fact_macro_completion() -> None:
    score = score_layer_fact_retention(_layers())

    assert score.endpoint_version == EVIDENCE_RETENTION_ENDPOINT_VERSION
    assert score.layer_fact_cell_count == 4
    assert score.layer_fact_exact_completion_rate == 0.75
    assert score.layer_fact_macro_token_recall == 0.875
    assert score.layer_balanced_minimum_fact_token_recall == 0.75
    assert score.minimum_layer_fact_token_recall == 0.5
    assert score.layer_exact_fact_completion_rates == (0.5, 1.0)
    assert score.layer_minimum_fact_token_recalls == (0.5, 1.0)
    assert not score.strict_all_required_fact_tokens_available_in_every_evaluated_layer
    assert score.to_dict()["primary_endpoint"] == "layer_fact_exact_completion_rate"


def test_endpoint_is_invariant_to_fact_order_and_equal_weights_fact_lengths() -> None:
    short = FactTokenRecall("short", retained_token_count=1, required_token_count=1)
    long_partial = FactTokenRecall("long", retained_token_count=9, required_token_count=10)
    left = score_layer_fact_retention([LayerFactRecall(0, (short, long_partial))])
    right = score_layer_fact_retention([LayerFactRecall(0, (long_partial, short))])

    assert left == right
    assert left.layer_fact_exact_completion_rate == 0.5
    assert left.layer_fact_macro_token_recall == 0.95


def test_serialized_projection_recomputes_counts_and_preserves_strict_diagnostic() -> None:
    payload = {
        "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer": False,
        "layers": [
            {
                "layer_index": layer.layer_index,
                "facts": [
                    {
                        "fact_id": fact.fact_id,
                        "retained_token_count": fact.retained_token_count,
                        "required_token_count": fact.required_token_count,
                        "token_retention_fraction": fact.token_recall,
                        "all_fact_tokens_retained": fact.exactly_complete,
                    }
                    for fact in layer.facts
                ],
            }
            for layer in _layers()
        ],
    }

    assert score_serialized_evidence_retention(payload) == score_layer_fact_retention(_layers())

    payload["correct"] = True
    payload["prediction"] = "not consulted"
    assert score_serialized_evidence_retention(payload) == score_layer_fact_retention(_layers())


def test_typed_runtime_score_uses_layer_union_fact_records() -> None:
    runtime_layers = tuple(
        LayerEvidenceRetention(
            layer_index=layer.layer_index,
            available_required_token_count=sum(
                fact.retained_token_count for fact in layer.facts
            ),
            required_token_count=sum(fact.required_token_count for fact in layer.facts),
            required_token_availability_fraction=0.0,
            complete_fact_count=sum(fact.exactly_complete for fact in layer.facts),
            required_fact_count=len(layer.facts),
            required_fact_completion_fraction=0.0,
            all_required_fact_tokens_available_in_at_least_one_kv_head=all(
                fact.exactly_complete for fact in layer.facts
            ),
            one_kv_head_contains_all_required_fact_tokens=False,
            kv_heads=(),
            facts=tuple(
                RuntimeFactTokenRetention(
                    fact_id=fact.fact_id,
                    retained_token_count=fact.retained_token_count,
                    required_token_count=fact.required_token_count,
                    token_retention_fraction=fact.token_recall,
                    all_fact_tokens_retained=fact.exactly_complete,
                )
                for fact in layer.facts
            ),
        )
        for layer in _layers()
    )
    runtime_score = EvidenceRetentionScore(
        strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer=False,
        evaluated_layer_count=2,
        evaluated_kv_head_count_per_layer=8,
        required_fact_count=2,
        required_token_count=4,
        mean_per_kv_head_required_token_retention_fraction=0.0,
        mean_per_kv_head_required_fact_completion_fraction=0.0,
        mean_per_layer_required_token_availability_fraction=0.0,
        minimum_per_layer_required_token_availability_fraction=0.0,
        mean_per_layer_required_fact_completion_fraction=0.0,
        fraction_of_evaluated_layers_with_all_required_fact_tokens_available=0.0,
        fraction_of_evaluated_layers_with_one_kv_head_containing_all_required_fact_tokens=0.0,
        fraction_of_required_facts_complete_in_every_evaluated_layer=0.0,
        layers=runtime_layers,
    )

    assert score_evidence_retention_endpoint(runtime_score) == score_layer_fact_retention(
        _layers()
    )
    with pytest.raises(EvidenceEndpointError, match="evaluated_layer_count"):
        score_evidence_retention_endpoint(replace(runtime_score, evaluated_layer_count=3))


def test_endpoint_rejects_inconsistent_or_incomplete_layer_fact_grids() -> None:
    with pytest.raises(EvidenceEndpointError, match="same fact IDs"):
        score_layer_fact_retention(
            (
                _layers()[0],
                LayerFactRecall(17, (_layers()[1].facts[0],)),
            )
        )

    changed_count = replace(_layers()[1].facts[0], required_token_count=3)
    with pytest.raises(EvidenceEndpointError, match="changes between layers"):
        score_layer_fact_retention(
            (_layers()[0], LayerFactRecall(17, (changed_count, _layers()[1].facts[1])))
        )

    with pytest.raises(EvidenceEndpointError, match="strict evidence diagnostic disagrees"):
        score_layer_fact_retention(_layers(), strict_diagnostic=True)


def test_serialized_projection_detects_tampered_derived_fraction() -> None:
    payload = {
        "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer": True,
        "layers": [
            {
                "layer_index": 0,
                "facts": [
                    {
                        "fact_id": "fact-a",
                        "retained_token_count": 2,
                        "required_token_count": 2,
                        "token_retention_fraction": 0.5,
                        "all_fact_tokens_retained": True,
                    }
                ],
            }
        ],
    }
    with pytest.raises(EvidenceEndpointError, match="fraction disagrees"):
        score_serialized_evidence_retention(payload)
