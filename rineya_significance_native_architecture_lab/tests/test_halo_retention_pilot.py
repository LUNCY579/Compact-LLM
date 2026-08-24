from __future__ import annotations

from pathlib import Path

import yaml


def test_retention_pilot_config_is_halo_and_matched() -> None:
    lab = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((lab / "configs" / "halo_retention_pilot.yaml").read_text())
    assert [row["id"] for row in config["conditions"]] == ["B2", "B3", "B4"]
    assert config["cache"]["retained_history_budget"] == 128
    assert config["significance"]["score_pooling"]["enabled"] is False
    assert "R2" not in config["selection"]["families"]
