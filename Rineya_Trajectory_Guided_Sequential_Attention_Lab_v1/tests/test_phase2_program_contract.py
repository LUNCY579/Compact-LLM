from __future__ import annotations

import importlib.util
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
SCRIPT = LAB / "scripts" / "verify_phase2_program_contract.py"
SPEC = importlib.util.spec_from_file_location("phase2_program_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_phase2a_smoke_config_has_unambiguous_semantics_and_rejected_regime_boundary() -> None:
    report = MODULE.verify(
        LAB / "configs" / "phase2_program_contract.json",
        LAB / "configs" / "phase2_tgsa_mtp_smoke.json",
    )
    assert report["passed"] is True, report["errors"]
    assert report["study_kind"] == "mechanism_bridge"
    assert report["mtp_semantics"] == "sequential_teacher_forced_recursive_v1"


def test_program_contract_rejects_cross_semantic_relabelling(tmp_path: Path) -> None:
    config = json.loads(
        (LAB / "configs" / "phase2_tgsa_mtp_smoke.json").read_text(encoding="utf-8")
    )
    config["mtp_semantics"] = "parallel_independent_offsets_v1"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    report = MODULE.verify(LAB / "configs" / "phase2_program_contract.json", path)
    assert report["passed"] is False
    assert any("does not match" in error for error in report["errors"])


def test_program_contract_blocks_confirmatory_revival_of_rejected_v4_regime(tmp_path: Path) -> None:
    config = json.loads(
        (LAB / "configs" / "phase2_tgsa_mtp_smoke.json").read_text(encoding="utf-8")
    )
    config["confirmatory"] = True
    config["development_only"] = False
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    report = MODULE.verify(LAB / "configs" / "phase2_program_contract.json", path)
    assert report["passed"] is False
    assert any("may not be confirmatory" in error for error in report["errors"])
