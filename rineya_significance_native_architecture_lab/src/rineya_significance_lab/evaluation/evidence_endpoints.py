"""Outcome-independent endpoints for exact evidence-retention traces.

The runtime evidence scorer intentionally exposes a very strong case Boolean: every
token of every required fact must be available in every evaluated layer.  That remains
an important diagnostic, but a conjunction over all layer/fact cells becomes
uninformative when even one cell fails in every case.

This module defines a less brittle *retention* endpoint without weakening the evidence
annotation or consulting model answers.  The primary endpoint is the macro-average of
exact atomic-fact completion over evaluated layers and required facts.  It therefore
does not reward partial boilerplate, does not let longer facts dominate, and has no
tuned threshold.  Partial-token summaries are explicitly secondary sensitivity
diagnostics; none of these metrics imply that retained evidence caused an answer.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, SupportsIndex, cast

from rineya_significance_lab.benchmarks.evidence import EvidenceRetentionScore

EVIDENCE_RETENTION_ENDPOINT_VERSION: Final = "layer-fact-retention-v1"
PRIMARY_ENDPOINT_NAME: Final = "layer_fact_exact_completion_rate"


class EvidenceEndpointError(ValueError):
    """A stored evidence trace cannot support the frozen endpoint definition."""


@dataclass(frozen=True, slots=True)
class FactTokenRecall:
    """Exact token recall for one required fact at one evaluated layer."""

    fact_id: str
    retained_token_count: int
    required_token_count: int

    def __post_init__(self) -> None:
        if not self.fact_id or self.fact_id != self.fact_id.strip():
            raise EvidenceEndpointError("fact_id must be non-empty with no outer whitespace")
        if self.required_token_count <= 0:
            raise EvidenceEndpointError("required_token_count must be positive")
        if not 0 <= self.retained_token_count <= self.required_token_count:
            raise EvidenceEndpointError(
                "retained_token_count must be between zero and required_token_count"
            )

    @property
    def token_recall(self) -> float:
        return self.retained_token_count / self.required_token_count

    @property
    def exactly_complete(self) -> bool:
        return self.retained_token_count == self.required_token_count


@dataclass(frozen=True, slots=True)
class LayerFactRecall:
    """Required-fact recall after unioning resident positions across KV heads."""

    layer_index: int
    facts: tuple[FactTokenRecall, ...]

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise EvidenceEndpointError("layer_index must be non-negative")
        if not self.facts:
            raise EvidenceEndpointError("each evaluated layer must contain required facts")
        fact_ids = tuple(fact.fact_id for fact in self.facts)
        if len(fact_ids) != len(set(fact_ids)):
            raise EvidenceEndpointError("fact IDs must be unique within each layer")


@dataclass(frozen=True, slots=True)
class EvidenceRetentionEndpointScore:
    """Frozen primary endpoint and transparent sensitivity diagnostics."""

    endpoint_version: str
    evaluated_layer_count: int
    required_fact_count: int
    layer_fact_cell_count: int
    layer_fact_exact_completion_rate: float
    layer_fact_macro_token_recall: float
    layer_balanced_minimum_fact_token_recall: float
    minimum_layer_fact_token_recall: float
    strict_all_required_fact_tokens_available_in_every_evaluated_layer: bool
    layer_exact_fact_completion_rates: tuple[float, ...]
    layer_minimum_fact_token_recalls: tuple[float, ...]

    @property
    def primary_value(self) -> float:
        return self.layer_fact_exact_completion_rate

    def to_dict(self) -> dict[str, object]:
        return {
            "endpoint_version": self.endpoint_version,
            "primary_endpoint": PRIMARY_ENDPOINT_NAME,
            "primary_definition": (
                "macro-average over evaluated layers and required facts of the indicator "
                "that every exact annotated token of that fact is available in the union "
                "of resident KV heads at that layer"
            ),
            "layer_fact_exact_completion_rate": self.layer_fact_exact_completion_rate,
            "layer_fact_macro_token_recall": self.layer_fact_macro_token_recall,
            "layer_balanced_minimum_fact_token_recall": (
                self.layer_balanced_minimum_fact_token_recall
            ),
            "minimum_layer_fact_token_recall": self.minimum_layer_fact_token_recall,
            "strict_all_required_fact_tokens_available_in_every_evaluated_layer": (
                self.strict_all_required_fact_tokens_available_in_every_evaluated_layer
            ),
            "evaluated_layer_count": self.evaluated_layer_count,
            "required_fact_count": self.required_fact_count,
            "layer_fact_cell_count": self.layer_fact_cell_count,
            "layer_exact_fact_completion_rates": list(
                self.layer_exact_fact_completion_rates
            ),
            "layer_minimum_fact_token_recalls": list(
                self.layer_minimum_fact_token_recalls
            ),
            "interpretation_limit": (
                "mechanistic post-policy retention only; partial-token diagnostics do not "
                "establish semantic sufficiency or causal mediation of answer correctness"
            ),
        }


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(cast(SupportsIndex, value)))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _rows(value: object, *, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of records")
    try:
        return tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable") from exc


def _finite_fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    result = float(cast(float | int, value))
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise EvidenceEndpointError(f"{name} must be a finite fraction in [0, 1]")
    return result


def _validate_layer_grid(layers: tuple[LayerFactRecall, ...]) -> tuple[str, ...]:
    if not layers:
        raise EvidenceEndpointError("at least one evaluated layer is required")
    layer_indices = tuple(layer.layer_index for layer in layers)
    if len(layer_indices) != len(set(layer_indices)):
        raise EvidenceEndpointError("evaluated layer indices must be unique")
    canonical_fact_ids = tuple(sorted(fact.fact_id for fact in layers[0].facts))
    canonical_required = {
        fact.fact_id: fact.required_token_count for fact in layers[0].facts
    }
    for layer in layers:
        observed_ids = tuple(sorted(fact.fact_id for fact in layer.facts))
        if observed_ids != canonical_fact_ids:
            raise EvidenceEndpointError("every evaluated layer must contain the same fact IDs")
        for fact in layer.facts:
            if fact.required_token_count != canonical_required[fact.fact_id]:
                raise EvidenceEndpointError(
                    f"required token count for fact {fact.fact_id!r} changes between layers"
                )
    return canonical_fact_ids


def score_layer_fact_retention(
    layers: Iterable[LayerFactRecall],
    *,
    strict_diagnostic: bool | None = None,
) -> EvidenceRetentionEndpointScore:
    """Score a validated layer-by-fact grid using the frozen v1 definitions."""

    layer_rows = tuple(layers)
    fact_ids = _validate_layer_grid(layer_rows)
    exact_cells = 0
    token_recalls: list[float] = []
    layer_completion_rates: list[float] = []
    layer_minimum_recalls: list[float] = []
    for layer in layer_rows:
        by_fact = {fact.fact_id: fact for fact in layer.facts}
        ordered = tuple(by_fact[fact_id] for fact_id in fact_ids)
        recalls = [fact.token_recall for fact in ordered]
        exact_count = sum(fact.exactly_complete for fact in ordered)
        exact_cells += exact_count
        token_recalls.extend(recalls)
        layer_completion_rates.append(exact_count / len(fact_ids))
        layer_minimum_recalls.append(min(recalls))

    computed_strict = exact_cells == len(layer_rows) * len(fact_ids)
    if strict_diagnostic is not None and strict_diagnostic != computed_strict:
        raise EvidenceEndpointError(
            "stored strict evidence diagnostic disagrees with the layer/fact grid"
        )
    cell_count = len(layer_rows) * len(fact_ids)
    return EvidenceRetentionEndpointScore(
        endpoint_version=EVIDENCE_RETENTION_ENDPOINT_VERSION,
        evaluated_layer_count=len(layer_rows),
        required_fact_count=len(fact_ids),
        layer_fact_cell_count=cell_count,
        layer_fact_exact_completion_rate=exact_cells / cell_count,
        layer_fact_macro_token_recall=sum(token_recalls) / cell_count,
        layer_balanced_minimum_fact_token_recall=(
            sum(layer_minimum_recalls) / len(layer_minimum_recalls)
        ),
        minimum_layer_fact_token_recall=min(layer_minimum_recalls),
        strict_all_required_fact_tokens_available_in_every_evaluated_layer=computed_strict,
        layer_exact_fact_completion_rates=tuple(layer_completion_rates),
        layer_minimum_fact_token_recalls=tuple(layer_minimum_recalls),
    )


def score_evidence_retention_endpoint(
    retention: EvidenceRetentionScore,
) -> EvidenceRetentionEndpointScore:
    """Score the typed runtime evidence result without consulting model outputs."""

    layers = tuple(
        LayerFactRecall(
            layer_index=layer.layer_index,
            facts=tuple(
                FactTokenRecall(
                    fact_id=fact.fact_id,
                    retained_token_count=fact.retained_token_count,
                    required_token_count=fact.required_token_count,
                )
                for fact in layer.facts
            ),
        )
        for layer in retention.layers
    )
    if retention.evaluated_layer_count != len(layers):
        raise EvidenceEndpointError(
            "typed evidence evaluated_layer_count disagrees with its layer records"
        )
    if not layers or retention.required_fact_count != len(layers[0].facts):
        raise EvidenceEndpointError(
            "typed evidence required_fact_count disagrees with its fact records"
        )
    expected_required_tokens = sum(fact.required_token_count for fact in layers[0].facts)
    if retention.required_token_count != expected_required_tokens:
        raise EvidenceEndpointError(
            "typed evidence required_token_count disagrees with its fact records"
        )
    return score_layer_fact_retention(
        layers,
        strict_diagnostic=(
            retention.strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer
        ),
    )


def score_serialized_evidence_retention(
    payload: Mapping[str, object],
) -> EvidenceRetentionEndpointScore:
    """Score the minimal evidence-only projection of a stored pilot case.

    The parser deliberately accepts only the evidence mapping.  Callers need not and
    should not pass answer correctness, predictions, accepted answers, or condition
    labels while calibrating the endpoint.
    """

    raw_layers = _rows(payload.get("layers"), name="layers")
    layers: list[LayerFactRecall] = []
    for layer_offset, raw_layer in enumerate(raw_layers):
        layer = _mapping(raw_layer, name=f"layers[{layer_offset}]")
        raw_facts = _rows(layer.get("facts"), name=f"layers[{layer_offset}].facts")
        facts: list[FactTokenRecall] = []
        for fact_offset, raw_fact in enumerate(raw_facts):
            fact = _mapping(
                raw_fact,
                name=f"layers[{layer_offset}].facts[{fact_offset}]",
            )
            retained = _integer(
                fact.get("retained_token_count"),
                name="retained_token_count",
            )
            required = _integer(
                fact.get("required_token_count"),
                name="required_token_count",
            )
            record = FactTokenRecall(
                fact_id=str(fact.get("fact_id", "")),
                retained_token_count=retained,
                required_token_count=required,
            )
            if "token_retention_fraction" in fact:
                stored_fraction = _finite_fraction(
                    fact["token_retention_fraction"],
                    name="token_retention_fraction",
                )
                if not math.isclose(stored_fraction, record.token_recall, abs_tol=1e-12):
                    raise EvidenceEndpointError(
                        "stored token_retention_fraction disagrees with exact token counts"
                    )
            if "all_fact_tokens_retained" in fact:
                stored_complete = fact["all_fact_tokens_retained"]
                if not isinstance(stored_complete, bool):
                    raise TypeError("all_fact_tokens_retained must be bool")
                if stored_complete != record.exactly_complete:
                    raise EvidenceEndpointError(
                        "stored all_fact_tokens_retained disagrees with exact token counts"
                    )
            facts.append(record)
        expected_required_tokens = sum(fact.required_token_count for fact in facts)
        expected_retained_tokens = sum(fact.retained_token_count for fact in facts)
        expected_complete_facts = sum(fact.exactly_complete for fact in facts)
        if "required_fact_count" in layer and _integer(
            layer["required_fact_count"], name="required_fact_count"
        ) != len(facts):
            raise EvidenceEndpointError(
                "stored layer required_fact_count disagrees with its fact records"
            )
        if "required_token_count" in layer and _integer(
            layer["required_token_count"], name="required_token_count"
        ) != expected_required_tokens:
            raise EvidenceEndpointError(
                "stored layer required_token_count disagrees with its fact records"
            )
        if "available_required_token_count" in layer and _integer(
            layer["available_required_token_count"],
            name="available_required_token_count",
        ) != expected_retained_tokens:
            raise EvidenceEndpointError(
                "stored available_required_token_count disagrees with its fact records"
            )
        if "complete_fact_count" in layer and _integer(
            layer["complete_fact_count"], name="complete_fact_count"
        ) != expected_complete_facts:
            raise EvidenceEndpointError(
                "stored complete_fact_count disagrees with its fact records"
            )
        if "required_token_availability_fraction" in layer:
            stored_token_fraction = _finite_fraction(
                layer["required_token_availability_fraction"],
                name="required_token_availability_fraction",
            )
            expected_token_fraction = expected_retained_tokens / expected_required_tokens
            if not math.isclose(stored_token_fraction, expected_token_fraction, abs_tol=1e-12):
                raise EvidenceEndpointError(
                    "stored layer token availability fraction disagrees with its fact records"
                )
        if "required_fact_completion_fraction" in layer:
            stored_fact_fraction = _finite_fraction(
                layer["required_fact_completion_fraction"],
                name="required_fact_completion_fraction",
            )
            expected_fact_fraction = expected_complete_facts / len(facts)
            if not math.isclose(stored_fact_fraction, expected_fact_fraction, abs_tol=1e-12):
                raise EvidenceEndpointError(
                    "stored layer fact completion fraction disagrees with its fact records"
                )
        if "all_required_fact_tokens_available_in_at_least_one_kv_head" in layer:
            stored_layer_complete = layer[
                "all_required_fact_tokens_available_in_at_least_one_kv_head"
            ]
            if not isinstance(stored_layer_complete, bool):
                raise TypeError(
                    "all_required_fact_tokens_available_in_at_least_one_kv_head must be bool"
                )
            if stored_layer_complete != (expected_complete_facts == len(facts)):
                raise EvidenceEndpointError(
                    "stored layer completion Boolean disagrees with its fact records"
                )
        layers.append(
            LayerFactRecall(
                layer_index=_integer(layer.get("layer_index"), name="layer_index"),
                facts=tuple(facts),
            )
        )

    strict_value = payload.get(
        "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer"
    )
    if not isinstance(strict_value, bool):
        raise TypeError("stored strict evidence diagnostic must be bool")
    if "evaluated_layer_count" in payload and _integer(
        payload["evaluated_layer_count"], name="evaluated_layer_count"
    ) != len(layers):
        raise EvidenceEndpointError(
            "stored evaluated_layer_count disagrees with its layer records"
        )
    if layers and "required_fact_count" in payload and _integer(
        payload["required_fact_count"], name="required_fact_count"
    ) != len(layers[0].facts):
        raise EvidenceEndpointError(
            "stored required_fact_count disagrees with its fact records"
        )
    if layers and "required_token_count" in payload and _integer(
        payload["required_token_count"], name="required_token_count"
    ) != sum(fact.required_token_count for fact in layers[0].facts):
        raise EvidenceEndpointError(
            "stored required_token_count disagrees with its fact records"
        )
    result = score_layer_fact_retention(tuple(layers), strict_diagnostic=strict_value)
    if "mean_per_layer_required_fact_completion_fraction" in payload:
        stored_primary = _finite_fraction(
            payload["mean_per_layer_required_fact_completion_fraction"],
            name="mean_per_layer_required_fact_completion_fraction",
        )
        if not math.isclose(
            stored_primary,
            result.layer_fact_exact_completion_rate,
            abs_tol=1e-12,
        ):
            raise EvidenceEndpointError(
                "stored mean layer fact completion disagrees with the exact endpoint"
            )
    return result
