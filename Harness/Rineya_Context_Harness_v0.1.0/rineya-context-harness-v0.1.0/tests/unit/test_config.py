from __future__ import annotations

from pathlib import Path

import pytest

from rineya_harness.config import HarnessConfig, load_config, redact_config
from rineya_harness.factory import resolve_config_relative_paths


def test_load_config_and_env_override(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  model: test-model\n", encoding="utf-8")
    monkeypatch.setenv("RINEYA_DATABASE_PATH", str(tmp_path / "env.sqlite3"))
    config = load_config(path)
    assert config.model.model == "test-model"
    assert config.database.path == tmp_path / "env.sqlite3"


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_redacts_secrets() -> None:
    config = HarnessConfig.model_validate(
        {"model": {"api_key": "secret"}, "memory": {"purge_confirmation_token": "purge"}}
    )
    redacted = redact_config(config)
    assert redacted["model"]["api_key"] == "***redacted***"
    assert redacted["memory"]["purge_confirmation_token"] == "***redacted***"


def test_relative_paths_resolve_from_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    resolved = resolve_config_relative_paths(HarnessConfig(), path)
    assert resolved.database.path.is_absolute()
    assert str(resolved.database.path).startswith(str(tmp_path))
