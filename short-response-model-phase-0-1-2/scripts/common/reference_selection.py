from __future__ import annotations

"""Shared helpers for the model selected after Phase 4B.

Phase 4B can either keep the accepted Phase 4 dense model or choose a smaller
structural candidate. Later phases must use exactly that selected checkpoint;
otherwise activation traces, routers, runtime measurements, and release hashes
would refer to different models.

This module intentionally contains no Transformers dependency. It only reads
and validates the small JSON manifest written by Phase 4B.
"""

from pathlib import Path
from typing import Any

from scripts.common.experiment_utils import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_MANIFEST = (
    ROOT / "artifacts" / "references" / "selected_inference_reference.json"
)


class ReferenceSelectionError(RuntimeError):
    """Raised when the selected-reference manifest is missing or malformed."""


def resolve_selected_model(
    explicit_model: str | None = None,
    manifest_path: Path = DEFAULT_REFERENCE_MANIFEST,
) -> str:
    """Return an explicit model or the model recorded by Phase 4B.

    An explicit command-line model always wins. This is useful for controlled
    ablations. Normal Phase 5-8 execution should omit it and consume the Phase
    4B manifest so every downstream result uses the same checkpoint.
    """

    if explicit_model:
        return explicit_model

    if not manifest_path.exists():
        raise ReferenceSelectionError(
            "Missing Phase 4B selected-reference manifest: "
            f"{manifest_path}. Run and approve Phase 4B first, then finalize "
            "the selected reference."
        )

    payload = read_json(manifest_path)
    selected = payload.get("selected_model")
    if not isinstance(selected, str) or not selected.strip():
        raise ReferenceSelectionError(
            f"{manifest_path} does not contain a non-empty 'selected_model'."
        )
    if payload.get("human_gate") != "PASS":
        raise ReferenceSelectionError(
            f"{manifest_path} is not tied to an explicit Phase 4B human PASS."
        )
    return selected


def write_selected_reference(
    *,
    output: Path,
    source_phase_4_model: str,
    selected_name: str,
    selected_model: str,
    registry: Path,
    decision_report: Path,
    selection_reason: str,
    candidate_metrics: dict[str, Any],
) -> None:
    """Write the canonical manifest consumed by later phases.

    The source Phase 4 checkpoint is preserved even when a reduced candidate is
    selected, which keeps the dense control available for rollback and Phase 8
    ablations.
    """

    write_json(
        output,
        {
            "manifest_version": 1,
            "human_gate": "PASS",
            "source_phase_4_model": source_phase_4_model,
            "selected_candidate": selected_name,
            "selected_model": selected_model,
            "selection_reason": selection_reason,
            "candidate_registry": str(registry),
            "decision_report": str(decision_report),
            "candidate_metrics": candidate_metrics,
            "downstream_rule": (
                "Phases 5-8 must use selected_model unless running an explicitly "
                "labeled control or ablation. Any later weight change invalidates "
                "Phase 6 router calibration."
            ),
        },
    )
