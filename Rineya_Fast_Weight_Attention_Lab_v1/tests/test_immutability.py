"""Published checkpoint trees must not be written by this lab."""

from __future__ import annotations

from pathlib import Path

import pytest

from rineya_fwa.paths import ARTIFACTS, SUPA_PATH, assert_readonly_target, assert_unsealed_source


def test_refuses_write_inside_supa_devfreeze():
    with pytest.raises(PermissionError):
        assert_readonly_target(SUPA_PATH / "model.safetensors")


def test_allows_write_under_artifacts():
    target = ARTIFACTS / "unit" / "ok.json"
    assert_readonly_target(target)


def test_refuses_sealed_sources():
    with pytest.raises(PermissionError):
        assert_unsealed_source(Path("rineya_long_context_cc0_v1_main/sealed_eval/prompts.jsonl"))
    with pytest.raises(PermissionError):
        assert_unsealed_source(Path("showcase_prompts.jsonl"))
