from __future__ import annotations

from pathlib import Path

import pytest

from rineya_significance_lab.cli import main

LAB = Path(__file__).resolve().parents[1]


def test_validate_config_cli(capsys) -> None:
    assert main(["validate-config", "--config", str(LAB / "configs" / "tier_a.yaml")]) == 0
    assert '"status": "PASS"' in capsys.readouterr().out


def test_validate_placeholder_manifest_cli(capsys) -> None:
    assert main(["validate-result", str(LAB / "results" / "manifest.json")]) == 0
    assert '"status": "PASS"' in capsys.readouterr().out


def test_validate_multirecord_case_jsonl_cli(capsys) -> None:
    path = LAB / "results" / "runs" / "tier_a_20260823" / "cases.jsonl"
    assert main(["validate-result", str(path), "--schema", "case.schema.json"]) == 0
    assert '"records": 112' in capsys.readouterr().out


def test_cli_refuses_unimplemented_confirmatory_runner(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(LAB / "configs" / "confirmatory.yaml")])
    assert error.value.code == 2
    assert "no runner is implemented" in capsys.readouterr().err
