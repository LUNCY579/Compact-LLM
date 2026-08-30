"""Hardened public entry point for the paired TGSA × MTP factorial study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .phase2_integrity import attach_phase2_integrity, validate_phase2_contract
from .phase2_runner import run_phase2_study as _run_native_phase2_study


def run_phase2_study(
    *,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    force_data: bool = False,
) -> dict[str, object]:
    """Run the native factorial study behind the integrity/competence contract."""

    config_path = Path(config_path).resolve()
    study: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    # Fail before materializing a dataset or creating an evidence root.
    validate_phase2_contract(study, config_path=config_path)
    _run_native_phase2_study(
        config_path=config_path,
        dataset_dir=Path(dataset_dir).resolve(),
        output_dir=Path(output_dir).resolve(),
        force_data=force_data,
    )
    return attach_phase2_integrity(
        config_path=config_path,
        dataset_dir=Path(dataset_dir).resolve(),
        output_dir=Path(output_dir).resolve(),
        implementation="paired-factorial-reference",
    )


__all__ = ["run_phase2_study"]
