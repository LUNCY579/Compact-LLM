from __future__ import annotations

from pathlib import Path

import pytest

from consensus_lab.paths import LabPaths
from consensus_lab.models.committee import ExpertSpec, validate_expert_compatibility


def test_output_guard_blocks_official_and_other_labs(tmp_path: Path) -> None:
    official = tmp_path / "short-response-model-phase-0-1-2"
    lab = tmp_path / "short-response-adapter-consensus-lab"
    official.mkdir()
    lab.mkdir()
    paths = LabPaths(tmp_path, lab, official)
    with pytest.raises(ValueError, match="protected"):
        paths.assert_output_path(official / "results.json")
    with pytest.raises(ValueError, match="protected"):
        paths.assert_output_path(tmp_path / "v22-model-soup-micro-experiment" / "result.json")
    with pytest.raises(ValueError, match="inside the lab"):
        paths.assert_output_path(tmp_path / "elsewhere.json")
    assert paths.assert_output_path(lab / "results/result.json") == (lab / "results/result.json").resolve()


def test_layout_requires_sibling_roots(tmp_path: Path) -> None:
    official = tmp_path / "short-response-model-phase-0-1-2"
    official.mkdir()
    nested = official / "short-response-adapter-consensus-lab"
    nested.mkdir()
    with pytest.raises(RuntimeError, match="sibling"):
        LabPaths(tmp_path, nested, official).validate_layout()


def test_unsupported_mixed_adapter_type_fails_explicitly(tmp_path: Path) -> None:
    official = tmp_path / "short-response-model-phase-0-1-2"
    lab = tmp_path / "short-response-adapter-consensus-lab"
    backbone = official / "backbone"
    adapter = official / "adapter"
    backbone.mkdir(parents=True)
    adapter.mkdir()
    lab.mkdir()
    (backbone / "config.json").write_text('{"model_type":"fake"}', encoding="utf-8")
    (backbone / "tokenizer.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_config.json").write_text(
        '{"peft_type":"ADALORA","base_model_name_or_path":"backbone"}', encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    paths = LabPaths(tmp_path, lab, official)
    with pytest.raises(ValueError, match="Unsupported PEFT type"):
        validate_expert_compatibility(
            paths=paths,
            backbone=backbone,
            experts=[ExpertSpec("unsupported", adapter)],
        )
