import json
import pytest

from experiments.rineya_length_generalization_regularization_v1.__main__ import main


def test_plan_no_checkpoint_access(capsys):
    main(["plan"])
    assert len(json.loads(capsys.readouterr().out)) == 3
    main(["plan", "--factorial-decay", "0.1"])
    assert len(json.loads(capsys.readouterr().out)) == 4


def test_cli_denies_native_before_paths():
    with pytest.raises(SystemExit):
        main(["train", "--checkpoint", "missing", "--plan", "missing", "--bank", "missing",
              "--eligibility", "missing", "--output", "missing"])


def test_fixture_build_and_smoke(tmp_path):
    data = tmp_path / "data.jsonl"
    main(["build-data", "--fixture", "--split", "dev", "--output", str(data), "--lengths", "256", "--parents", "1"])
    assert data.exists()
    main(["audit-data", str(data)])
    output = tmp_path / "smoke"
    main(["smoke", "--output", str(output)])
    report = json.loads((output / "smoke.json").read_text())
    assert report["evidence_level"] == "CPU_FIXTURE_ONLY" and not report["native_checkpoint_loaded"]
    assert report["comparison"]["status"] == "NOT_PROMOTABLE"
    assert len(report["training"]) == 3
