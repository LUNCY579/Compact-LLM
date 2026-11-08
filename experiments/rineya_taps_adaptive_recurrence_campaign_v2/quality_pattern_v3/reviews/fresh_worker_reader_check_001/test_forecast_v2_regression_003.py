"""Bounded synthetic forecast and V2 path regressions; no native dispatch."""
import ast
import copy
import inspect
import json
from pathlib import Path

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_analysis as a
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_scope as scope
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_readiness as readiness
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.tests.test_fresh_analysis import raw_fixture


@pytest.mark.parametrize("field", ["remaining_forward_seconds", "remaining_nonforward_seconds", "remaining_forecast_seconds", "elapsed_seconds"])
def test_negative_forecast_component_refuses_complete_synthetic_matrix(field):
    fixture = raw_fixture()
    assert a.raw_gate(*fixture)[0] == []
    broken = copy.deepcopy(fixture); broken[5][field] = -1.
    assert a.raw_gate(*broken)[0]


def test_combined_negative_forward_nonforward_witness_now_refuses():
    fixture = raw_fixture(); fixture[5]["remaining_forward_seconds"] = fixture[5]["remaining_nonforward_seconds"] = -1.
    assert a.raw_gate(*fixture)[0]


def test_scope_exact_v2_mock_dispatch_positive_and_v1_refusal(tmp_path, monkeypatch):
    root = tmp_path / "MOCK_ROOT"; v3 = root / "campaign/v3"; v3.mkdir(parents=True)
    monkeypatch.setattr(scope, "ROOT", root); monkeypatch.setattr(scope, "V3", v3); monkeypatch.setattr(scope, "CAMPAIGN", root / "campaign")
    monkeypatch.setattr(scope, "validate_scope_files", lambda admission: None)
    required = ("fresh_scope.py", "fresh_entry.py", "fresh_supervisor.py", "fresh_native.py", "opposed_gate_generation.py", "opposed_gate_policy.py", "fresh_open_bank.py", "fresh_open_verifier.py")
    hashes = {}
    for name in required:
        path = v3 / name; path.write_text("SYNTHETIC ONLY", encoding="utf-8"); hashes[path.relative_to(root).as_posix()] = scope.sha(path)
    admission = {"dispatch_hashes": hashes, "review_receipt": "reviews/fresh_dispatch_review_002.json"}
    v2 = v3 / "fresh_admission_v2.json"; v2.write_text(json.dumps(admission), encoding="utf-8")
    review = v3 / admission["review_receipt"]; review.parent.mkdir(); review.write_text(json.dumps({"status": "PASS_FRESH_K01_FP32_DISPATCH", "admission_sha256": scope.sha(v2)}), encoding="utf-8")
    assert scope.verify_dispatch(v2) == admission
    v1 = v3 / "fresh_admission_v1.json"; v1.write_bytes(v2.read_bytes())
    with pytest.raises(PermissionError, match="exact fresh admission path"): scope.verify_dispatch(v1)


def test_readiness_consumer_requests_v2_only_with_cpu_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "V3", tmp_path)
    seen = []
    def parent(source, out, admission):
        seen.append(admission.name)
        admission.write_text("SYNTHETIC_METADATA_ONLY", encoding="utf-8")
        return {"study_id": "MOCK"}, {"fixture": True}
    monkeypatch.setattr(readiness, "check_parent", parent)
    assert readiness.main("FRESH_READINESS_FIXTURE_003.json") == 0
    assert seen == ["fresh_admission_v2.json"]
    assert a.ADMISSION.name == "fresh_admission_v2.json"


def test_builder_v2_producer_and_new_source_receipt_membership_are_explicit():
    root = Path(a.__file__).resolve().parent
    tree = ast.parse((root / "prepare_fresh_admission.py").read_text(encoding="utf-8"))
    constants = {x.value for x in ast.walk(tree) if isinstance(x, ast.Constant) and isinstance(x.value, str)}
    assert "fresh_admission_v2.json" in constants and "reviews/fresh_dispatch_review_002.json" in constants
    assert "reviews/fresh_worker_reader_source_review_002.json" in constants
    assert "fresh_admission_v1.json" not in constants and "reviews/fresh_dispatch_review_001.json" not in constants
