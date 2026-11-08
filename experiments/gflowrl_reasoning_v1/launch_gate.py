"""Hard launch gate for the GFlowRL native R0/R1 experiment.

The current branch intentionally cannot pass this gate: the historical reasoning lane has
not established a canonical reward-maximizing RL baseline, the R0/R1 dataset is not bound,
and the paper-reported runtime details are not fully qualified. Keeping the refusal in code
prevents a standalone objective test from being mistaken for training authorization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class NativeLaunchBlocked(RuntimeError):
    """Raised when a required scientific or runtime binding is absent."""


def load_current_contracts(root: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    directory = Path(root) if root is not None else Path(__file__).resolve().parent
    binding = json.loads((directory / "CANONICAL_R0_BINDING.json").read_text(encoding="utf-8"))
    fidelity = json.loads((directory / "PAPER_FIDELITY_MATRIX.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    return binding, fidelity, manifest


def assess_native_launch(
    binding: dict[str, Any],
    fidelity: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []

    if binding.get("requested_r0", {}).get("status") != "QUALIFIED_RL_BASELINE":
        reasons.append("canonical R0 is not a qualified reward-maximizing RL baseline")
    if fidelity.get("full_runtime_status") != "QUALIFIED":
        reasons.append("full GFlowRL runtime fidelity is not qualified")
    if manifest.get("data", {}).get("dataset_status") != "BOUND_TRAIN_DEV_ONLY":
        reasons.append("train/dev dataset and split hashes are not bound")
    if manifest.get("sealed_access", {}).get("denied") is not True:
        reasons.append("sealed/confirmation denial is not asserted")

    return {
        "allowed": not reasons,
        "reasons": reasons,
        "training_authorized": not reasons,
        "sealed_access": "DENIED" if manifest.get("sealed_access", {}).get("denied") is True else "UNSAFE",
    }


def assert_native_launch_allowed(
    binding: dict[str, Any],
    fidelity: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    result = assess_native_launch(binding, fidelity, manifest)
    if not result["allowed"]:
        raise NativeLaunchBlocked("; ".join(result["reasons"]))
    return result


def assert_current_branch_launch_allowed(root: str | Path | None = None) -> dict[str, Any]:
    return assert_native_launch_allowed(*load_current_contracts(root))
