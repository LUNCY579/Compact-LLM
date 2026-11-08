from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_runner_script() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_model_study.py"
    spec = importlib.util.spec_from_file_location("tgsa_phase1_runner_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(
    output_root: Path,
    *,
    arm: str = "learned_delta",
    seed: int = 7,
    config_hash: str = "a" * 64,
    dataset_hash: str = "b" * 64,
    git_head: str = "c" * 40,
) -> None:
    run_dir = output_root / "runs" / arm / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "arm": arm,
                "seed": seed,
                "study_config_sha256": config_hash,
                "dataset_manifest_sha256": dataset_hash,
                "git": {"head": git_head},
            }
        ),
        encoding="utf-8",
    )


def test_completed_run_reuse_accepts_exact_identity(tmp_path: Path) -> None:
    module = _load_runner_script()
    _write_manifest(tmp_path)

    module._validate_completed_run_reuse(
        output_root=tmp_path,
        arms=["learned_delta"],
        seeds=[7],
        study_config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        current_git_head="c" * 40,
    )


def test_completed_run_reuse_rejects_config_dataset_or_source_drift(tmp_path: Path) -> None:
    module = _load_runner_script()
    _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="stale completed run"):
        module._validate_completed_run_reuse(
            output_root=tmp_path,
            arms=["learned_delta"],
            seeds=[7],
            study_config_sha256="d" * 64,
            dataset_manifest_sha256="b" * 64,
            current_git_head="c" * 40,
        )

    with pytest.raises(ValueError, match="dataset_manifest_sha256"):
        module._validate_completed_run_reuse(
            output_root=tmp_path,
            arms=["learned_delta"],
            seeds=[7],
            study_config_sha256="a" * 64,
            dataset_manifest_sha256="e" * 64,
            current_git_head="c" * 40,
        )

    with pytest.raises(ValueError, match="git.head"):
        module._validate_completed_run_reuse(
            output_root=tmp_path,
            arms=["learned_delta"],
            seeds=[7],
            study_config_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            current_git_head="f" * 40,
        )
