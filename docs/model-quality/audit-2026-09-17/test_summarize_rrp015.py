"""CPU fixtures: reporting must not turn incomplete evidence into a pass."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("rrp015_summary", Path(__file__).with_name("summarize_rrp015.py"))
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def process():
    rows = [{"id": f"{family}:{i}", "family": family, "reference": "{}",
             "prediction": "{}", "canonical": True, "strict": True, "terminated": True}
            for family in summary.FLOORS for i in range(128)]
    return {"instrument": "select-views-001", "views_sha256": summary.SELECT_SHA,
            "instrument_admits_selection": True, "distinct_parents": 384,
            "predictions": rows, "runtime": {"fixture": True}, "numerics": {"tf32": False},
            "implementation_digest": "fixture", "max_new_tokens": 96,
            "per_family": {f: {"binding_floor_count": n, "n": 128, "canonical_correct": 128}
                           for f, n in summary.FLOORS.items()}}


def retention():
    return {"probe_rows": 150, "probe_sha256": summary.PANEL_SHA, "by_bare_policy": {
        p: {"source_correct_total": n, "arm_correct_total": n+10-5,
            "arm_correct_outside_source": 10, "lost_total": 5, "reference_size": n+10,
            "per_class": {"all": {"rows": 150, "retained": n+10-5, "terminated": 149}}}
        for p, n in (("permissive", 47), ("strict", 41))}}


def test_exact_cp_endpoint_and_complete_process():
    assert summary.cp(128, 128)[0] == pytest.approx(.025**(1/128))
    got = summary.process_summary(process())
    assert got["status"] == "VALID"
    assert got["all_families_clear"] is True
    assert got["terminated"] == 384


@pytest.mark.parametrize("fault", ["partial", "duplicate", "wrong_bank", "wrong_score"])
def test_process_refuses_incomplete_or_incompatible_evidence(fault):
    value = process()
    if fault == "partial":
        value["predictions"].pop()
    elif fault == "duplicate":
        value["predictions"][1]["id"] = value["predictions"][0]["id"]
    elif fault == "wrong_bank":
        value["views_sha256"] = "wrong"
    else:
        value["predictions"][0]["canonical"] = False
    assert summary.process_summary(value)["status"] == "INCOMPATIBLE"


def test_retention_uses_source_paired_counts_not_union_retention():
    got = summary.retention_summary(retention())
    assert got["status"] == "VALID"
    assert got["by_policy"]["permissive"]["source_retained"] == 42
    assert got["by_policy"]["permissive"]["gained_vs_source"] == 10
    assert got["by_policy"]["strict"]["source_retained"] == 36


def test_long_context_literal_unknown_does_not_count_punctuation_or_substrings():
    predictions = {f"RLC-{i:03}-{variant}": {"prediction": "UNKNOWN." if variant == "U" else "answer",
                                             "terminated": True}
                   for i in range(1, 65) for variant in "BCPU"}
    predictions["RLC-001-U"]["prediction"] = "UNKNOWN"
    predictions["RLC-001-B"]["prediction"] = "UNKNOWN"
    value = {"predictions": predictions, "per_row_correct": {i: False for i in predictions},
             "rows_total": 256, "rows_skipped_context_overflow": 0, "all256_hits": 0}
    got = summary.long_summary(value)
    assert got["literal_UNKNOWN_unanswerable"] == 1
    assert got["literal_false_abstention_answerable"] == 1
    assert "REPORT_ONLY" in got["admissible_use"]


def test_missing_outputs_remain_unavailable_and_writer_is_create_only(tmp_path):
    evidence = tmp_path / "clone/experiments/architecture_benchmark_001/evidence/retention-design-v1"
    evidence.mkdir(parents=True)
    records = tmp_path / "records"
    records.mkdir()
    out = tmp_path / "summary"
    command = [sys.executable, str(Path(summary.__file__)), "--evidence-dir", str(evidence),
               "--records", str(records), "--out-dir", str(out)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    value = json.loads((out / "SUMMARY.json").read_text())
    assert value["models"]["supa"]["process"] is None
    assert value["models"]["control"]["gate_v2"]["validated_exploratory_result"] is None
    assert value["model_calls"] == 0
    old = (out / "SUMMARY.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert (out / "SUMMARY.json").read_bytes() == old


def test_partial_source_blocks_compatible_acquisition_claim(tmp_path, monkeypatch):
    evidence = tmp_path / "clone/experiments/architecture_benchmark_001/evidence/retention-design-v1"
    evidence.mkdir(parents=True)
    records = tmp_path / "records"
    records.mkdir()
    complete, partial = process(), process()
    partial["predictions"] = partial["predictions"][:100]
    for name, value in (("supa", partial), ("control", complete)):
        (evidence / f"PROCESS_rrp015_{name}.json").write_text(json.dumps(value))
    monkeypatch.setattr(summary, "model_identity", lambda *a: {"status": "VERIFIED"})
    got = summary.collect(evidence, records)
    assert got["models"]["control"]["process"]["all_families_clear"] is True
    assert got["models"]["control"]["compatible_with_complete_source_process"] is False
    assert got["models"]["control"]["gate_v2"]["validated_exploratory_result"] is None


def test_module_never_imports_model_runtime():
    code = (f"import runpy,sys; runpy.run_path({summary.__file__!r},run_name='fixture'); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", [None, "source_runtime", "wrong_checkpoint", "failed_evaluation",
                                   "successful_retry", "wrong_evaluation_kind", "wrong_result_path"])
def test_gate_needs_complete_compatible_source_checkpoint_and_supervision(tmp_path, monkeypatch, fault):
    evidence = tmp_path / "clone/experiments/architecture_benchmark_001/evidence/retention-design-v1"
    evidence.mkdir(parents=True)
    records = tmp_path / "records"
    records.mkdir()
    instrument = tmp_path / "select-views-001/SELECT_VIEWS.jsonl"
    instrument.parent.mkdir()
    instrument.write_text("".join(json.dumps(r) + "\n" for r in process()["predictions"]))
    monkeypatch.setattr(summary, "SELECT_SHA", summary.sha(instrument))
    for name in ("supa", "control"):
        value = process()
        if fault == "source_runtime" and name == "control":
            value["runtime"] = {"different": True}
        (evidence / f"PROCESS_rrp015_{name}.json").write_text(json.dumps(value))
    panel = retention()
    panel["target"] = str(records / "control-research")
    for value in panel["by_bare_policy"].values():
        value["arm_correct_total"] += 20
        value["arm_correct_outside_source"] += 20
    (evidence / "RETENTION_PROBE_rrp015_control.json").write_text(json.dumps(panel))
    monkeypatch.setattr(summary, "model_identity", lambda *a: {
        "status": "VERIFIED", "checkpoint_binding": {"source": "fixture"},
        "checkpoint_progress": {"updates": 2 if fault == "wrong_checkpoint" else 28}})
    monkeypatch.setattr(summary, "training_summary", lambda *a: {
        "status": "COMPLETE_VERIFIED", "attempts": [{"status": "COMPLETE_VERIFIED",
        "binding": {"source": "fixture"}, "progress": {"updates": 28}}]})
    for filename in ("PROCESS_rrp015_supa.json", "PROCESS_rrp015_control.json",
                     "RETENTION_PROBE_rrp015_control.json"):
        directory = records / f"eval-{filename}"
        directory.mkdir()
        kind = "retention" if filename.startswith("RETENTION") else "process"
        result_path = str(evidence / filename)
        if fault == "wrong_evaluation_kind":
            kind = "long_context"
        if fault == "wrong_result_path":
            result_path = str(tmp_path / filename)
        values = {"probe_binding": {"kind": kind, "result": result_path, "script_sha256": "script"},
                  "probe_terminal": {"complete": fault != "failed_evaluation", "script_unchanged": True,
                                     "script_sha256_after": "script", "result": result_path},
                  "supervisor": {"returncode": 0, "wall_seconds": 1},
                  "retention_predictions": {"rows": 150}}
        for key, value in values.items():
            (directory / f"{key}.json").write_text(json.dumps(value))
    if fault == "successful_retry":
        failed = records / "eval-rrp015_supa-process"
        failed.mkdir()
        # The pre-model failure has a terminal/supervisor but no binding or output.
        (failed / "probe_terminal.json").write_text(json.dumps({"complete": False, "failure": {
            "type": "ModuleNotFoundError", "message": "No module named psutil", "phase": "resource_guard"}}))
        (failed / "supervisor.json").write_text(json.dumps({"returncode": 1, "wall_seconds": 2}))
        successful = records / "eval-PROCESS_rrp015_supa.json"
        successful.rename(records / "eval-rrp015_supa-process-retry001")
    collected = summary.collect(evidence, records)
    gate = collected["models"]["control"]["gate_v2"]
    assert gate["observed_conjunction"] is True
    assert gate["validated_exploratory_result"] is (True if fault in (None, "successful_retry") else None)
    assert gate["broad_preservation_or_confirmation"] is False
    if fault == "successful_retry":
        attempts = {a["attempt"]: a for a in collected["evaluation_attempts"]}
        assert attempts["eval-rrp015_supa-process"]["complete_supervised"] is False
        assert attempts["eval-rrp015_supa-process"]["terminal"]["failure"]["type"] == "ModuleNotFoundError"
        assert attempts["eval-rrp015_supa-process-retry001"]["complete_supervised"] is True
        assert collected["session_resources"]["observed_supervisor_wall_seconds"] == 5
