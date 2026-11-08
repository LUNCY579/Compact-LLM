"""Hardened public entry point for the reference TGSA × MTP Phase-2 study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .phase2_integrity import attach_phase2_integrity, validate_phase2_contract
from .phase2_mtp import run_phase2_study as _run_native_phase2_study


def run_phase2_study(
    *,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    repo_root: Path,
    overwrite: bool = False,
    force_data: bool = False,
) -> dict[str, object]:
    """Run the native study behind the fail-closed Phase-2 integrity contract."""

    config_path = Path(config_path).resolve()
    study: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    # This happens before either dataset or output creation.
    validate_phase2_contract(study, config_path=config_path)
    _run_native_phase2_study(
        config_path=config_path,
        dataset_dir=Path(dataset_dir).resolve(),
        output_dir=Path(output_dir).resolve(),
        repo_root=Path(repo_root).resolve(),
        overwrite=overwrite,
        force_data=force_data,
    )
    return attach_phase2_integrity(
        config_path=config_path,
        dataset_dir=Path(dataset_dir).resolve(),
        output_dir=Path(output_dir).resolve(),
        implementation="trajectory-conditioned-reference",
    )


__all__ = ["run_phase2_study"]
