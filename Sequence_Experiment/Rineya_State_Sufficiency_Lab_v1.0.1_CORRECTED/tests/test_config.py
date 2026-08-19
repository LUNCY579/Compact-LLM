from __future__ import annotations

from pathlib import Path

from rineya_state_lab.config import load_config


def test_config_include_env_and_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "C:/models/base")
    (tmp_path / "base.yaml").write_text("model:\n  path: ${MODEL_PATH}\n  rank: 8\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("include: base.yaml\nmodel:\n  alpha: 16\n", encoding="utf-8")
    config = load_config(tmp_path / "child.yaml", ["model.rank=12", "run.enabled=true"])
    assert config["model"] == {"path": "C:/models/base", "rank": 12, "alpha": 16}
    assert config["run"]["enabled"] is True
