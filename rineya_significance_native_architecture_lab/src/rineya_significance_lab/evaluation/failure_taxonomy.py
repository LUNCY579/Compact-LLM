"""Conservative, descriptive failure labels for compressed-cache outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def compressed_retention_failure_labels(
    *,
    correct: bool,
    evidence: Mapping[str, Any],
) -> list[str]:
    """Label final retained state without claiming it caused the answer.

    A failed strict conjunction does not prove that evidence was wholly evicted
    or that missing final-state tokens caused an incorrect answer.  We reserve
    ``EVIDENCE_EVICTED`` for measured zero availability across all evaluated
    layer-token cells and otherwise use ``EVIDENCE_PARTIALLY_RETAINED``.
    ``UNKNOWN`` covers an incorrect answer despite strict final availability.
    """

    if correct:
        return []
    strict = bool(
        evidence.get(
            "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer",
            evidence.get("all_required_survived", False),
        )
    )
    if strict:
        return ["UNKNOWN"]
    mean_availability = float(
        evidence.get("mean_per_layer_required_token_availability_fraction", 0.0)
    )
    if mean_availability <= 0.0:
        return ["EVIDENCE_EVICTED"]
    return ["EVIDENCE_PARTIALLY_RETAINED"]
