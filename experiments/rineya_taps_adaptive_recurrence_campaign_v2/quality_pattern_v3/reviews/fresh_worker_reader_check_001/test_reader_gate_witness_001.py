"""Reviewer synthetic witnesses only; no corpus/assets/tokenizer/model access."""
from pathlib import Path
import json
import math

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_analysis as a
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.tests.test_fresh_analysis import raw_fixture


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "mockroot"; v3 = root / "v3"; run = v3 / "run"
    run.mkdir(parents=True)
    names = ("fresh_analysis.py", "plan.md", "fresh_native.py", "opposed_gate_generation.py",
             "opposed_gate_policy.py", "fresh_open_bank.py", "fresh_open_verifier.py", "short_response_panel.py")
    for name in names: (v3 / name).write_text("SYNTHETIC SOURCE ONLY", encoding="utf-8")
    review = v3 / "review.json"; review.write_text("SYNTHETIC REVIEW ONLY", encoding="utf-8")
    monkeypatch.setattr(a, "ROOT", root); monkeypatch.setattr(a, "V3", v3)
    monkeypatch.setattr(a, "RUN", run); monkeypatch.setattr(a, "PLAN", v3 / "plan.md")
    monkeypatch.setattr(a, "__file__", str(v3 / "fresh_analysis.py")); monkeypatch.setattr(a, "CAMPAIGN", root)
    monkeypatch.setattr(a, "ADMISSION", v3 / "admission.json")
    h = "1" * 64; monkeypatch.setattr(a, "sha", lambda _: h)
    population, outcomes, events, receipt, qualifier, forecast, routes = raw_fixture()
    dispatch = {"v3/" + x: h for x in names}
    assets = {"SYNTHETIC_ASSET_" + str(i): h for i in range(10)}
    admission = {"status": "READY_FOR_FRESH_K01_FP32_DISPATCH", "primary_policy": a.GUARD,
        "dispatch_hashes": dispatch, "checkpoint_files": assets, "source": str(v3),
        "review_receipt": "review.json", "gpu_uuid": "fixture", "gpu_driver_version": "fixture", "study_id": "fixture"}
    runtime = {"python": "3.11.15", "torch": "2.12.1+cu126", "transformers": "5.14.1", "CPU_threads": 4,
        "dtype": "float32", "attention": "eager", "TF32": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "qwen_implementation_sha256": "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"}
    receipt.update(dispatch_hashes=dispatch, checkpoint_files_before=assets, checkpoint_files_after=assets,
        loaded_custom_code_sha256=a.CODE, runtime=runtime, parent_elapsed_seconds=100.)
    qualifier["checkpoint_files"] = assets
    terminal = {"status": "WORKER_TERMINAL", "exit_code": 0, "admission_sha256": h, "worker_pid": 11,
        "native_worker_seconds": 100., "gpu_initial_inventory": {"uuid": "fixture", "driver_version": "fixture", "free_bytes": 6 * a.GIB}}
    samples = [{"rss_bytes": a.GIB, "host_available_bytes": 2 * a.GIB, "worker_seconds": 50.}]
    ledger = [{"study_id": "fixture", "event": "native_started"},
        {"study_id": "fixture", "event": "native_completed", "worker_pid": 11, "native_worker_seconds": 100., "admission_sha256": h}]
    packets = {str(run / "terminal_receipt.json"): terminal, str(run / "admission.json"): admission,
        str(run / "source_manifest.json"): {"sha256": dispatch},
        str(run / "review.json"): {"status": "PASS_FRESH_K01_FP32_DISPATCH", "admission_sha256": h},
        str(run / "result/receipt.json"): receipt, str(run / "result/qualification.json"): qualifier,
        str(run / "result/matrix_forecast.json"): forecast, str(run / "result/random_routes.json"): routes,
        str(run / "result/population.json"): population,
        str(run / "result/registration.json"): {"runtime": runtime, "admission_sha256": h,
            "initial_resources": {"gpu_free_bytes": 6 * a.GIB, "host_available_bytes": 6 * a.GIB}}}
    opened = []
    def read(path):
        opened.append(str(path)); return packets[str(path)]
    monkeypatch.setattr(a, "read", read)
    tables = {str(run / "resources.jsonl"): samples, str(run / "result/outcomes.jsonl"): outcomes,
        str(run / "result/forwards.jsonl"): events, str(root / "BUDGET_LEDGER.jsonl"): ledger}
    monkeypatch.setattr(a, "rows", lambda path: tables[str(path)])
    return run, admission, terminal, receipt, samples, ledger, opened


def test_current_reader_accepts_nan_parent_resource_and_negative_charged_clock(tmp_path, monkeypatch):
    run, admission, terminal, receipt, samples, ledger, opened = fixture(tmp_path, monkeypatch)
    assert a.load_native(run)["counts"]["failed_forwards"] == 2
    samples[0]["rss_bytes"] = float("nan"); samples[0]["host_available_bytes"] = float("nan")
    samples[0]["worker_seconds"] = float("nan")
    terminal["native_worker_seconds"] = ledger[1]["native_worker_seconds"] = -1.
    receipt["parent_elapsed_seconds"] = -1.
    assert a.load_native(run)["terminal"]["native_worker_seconds"] == -1.


def test_current_reader_does_not_open_admitted_prompt_token_exports(tmp_path, monkeypatch):
    run, admission, terminal, receipt, samples, ledger, opened = fixture(tmp_path, monkeypatch)
    for name in ("scientific_bank", "preservation_bank", "scientific_token_binding", "preservation_token_binding", "prompt_export_manifest"):
        admission[name + "_path"] = "ABSENT_SYNTHETIC_FROZEN_EXPORT_" + name
        admission[name + "_sha256"] = "0" * 64
    assert len(a.load_native(run)["population"]) == 456
    assert all("ABSENT_SYNTHETIC" not in path for path in opened)
