from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import open_native
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.open_native import (
    GRID, SEEDS, adapt_open_case, admit_validation, freeze_development, open_diagnostics,
    read_open_packet, strict_score,
)
from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import open_bank


def freeze_fixture():
    rows = [{"split": "DEVELOPMENT", "case_id": f"dev-{index}", "seed": seed, "arm": "A2",
             "fixed_alpha": alpha, "score": int(alpha == open_native.STATIC_PREFIX_MEAN_ALPHA)}
            for index in range(16) for seed in SEEDS for alpha in GRID]
    rows += [{"split": "DEVELOPMENT", "case_id": f"dev-{index}", "seed": seed, "arm": "A3", "token_ids": [9]}
             for index in range(16) for seed in SEEDS]
    traces = [{"split": "DEVELOPMENT", "arm": "A3", "iteration": iteration, "alpha": 1. if iteration == 0 else 1.200000047,
               "case_id": f"dev-{index}", "seed": seed, "configured_seed": seed,
               "forward_index": 1, "evidence_kind": "native_trajectory"}
              for index in range(16) for seed in SEEDS for iteration in (0, 1)]
    return rows, traces


def test_development_freeze_includes_sixth_static_control_and_generation_pool():
    rows, traces = freeze_fixture()
    freeze = freeze_development(rows, traces, "bank", "tokens", "source")
    assert len(GRID) == 6 and freeze["alpha"] == open_native.STATIC_PREFIX_MEAN_ALPHA
    assert freeze["pool"] == [1.2] * 32 and not freeze["OPEN_VALIDATION_native_exposure_at_freeze"]
    admit_validation(freeze, {"id": "validation", "split": "OPEN_VALIDATION"}, "bank", "tokens", "source")
    for field in ("bank", "tokens", "source"):
        args = ["bank", "tokens", "source"]
        args[["bank", "tokens", "source"].index(field)] = "changed"
        with pytest.raises(PermissionError):
            admit_validation(freeze, {"id": "validation", "split": "OPEN_VALIDATION"}, *args)


def test_validation_score_and_alpha_are_rejected_before_selection_access():
    rows, traces = freeze_fixture()
    class Poison(dict):
        def __getitem__(self, key):
            if key in ("score", "alpha"): raise AssertionError("validation label/stat read")
            return super().__getitem__(key)
    with pytest.raises(PermissionError):
        freeze_development(rows + [Poison(split="OPEN_VALIDATION")], traces, "b", "t", "s")
    with pytest.raises(PermissionError):
        freeze_development(rows, traces + [Poison(split="OPEN_VALIDATION", arm="A3")], "b", "t", "s")


def test_incomplete_fixed_grid_and_mutated_freeze_are_refused():
    rows, traces = freeze_fixture()
    with pytest.raises(RuntimeError): freeze_development(rows[:-1], traces, "b", "t", "s")
    freeze = freeze_development(rows, traces, "b", "t", "s")
    freeze["alpha"] = .8
    with pytest.raises(PermissionError): admit_validation(freeze, {"id": "v", "split": "OPEN_VALIDATION"}, "b", "t", "s")


def test_duplicate_calibration_cell_and_missing_or_wrong_trace_provenance_are_rejected():
    rows, traces = freeze_fixture()
    with pytest.raises(RuntimeError): freeze_development(rows + [rows[0]], traces, "b", "t", "s")
    with pytest.raises(RuntimeError): freeze_development(rows, traces[:-1], "b", "t", "s")
    wrong = deepcopy(traces); wrong[0]["case_id"] = "unadmitted"
    with pytest.raises(PermissionError): freeze_development(rows, wrong, "b", "t", "s")
    wrong = deepcopy(traces); wrong[0]["configured_seed"] = 99
    with pytest.raises(PermissionError): freeze_development(rows, wrong, "b", "t", "s")


def test_strict_scorer_diagnostics_separate_format_answerability_and_reasoning():
    raw = open_bank.generate("native-test", 17)[0]
    case = adapt_open_case(raw)
    assert case["max_new_tokens"] == 8
    assert strict_score(raw["gold"], case["scorer"])["score"] == 1
    assert strict_score(raw["gold"] + " explanation", case["scorer"])["score"] == 0
    assert strict_score("UNKNOWN", case["format_scorer"])["score"] == 0
    diagnostics = open_diagnostics("UNKNOWN", raw)
    assert diagnostics["false_abstention_on_answerable"]
    unanswerable = next(row for row in open_bank.generate("native-test", 17) if row["answerability"] == "unanswerable")
    assert open_diagnostics("99", unanswerable)["unsupported_canonical_answer_on_unanswerable"] is True
    assert open_diagnostics("cannot tell", unanswerable)["unsupported_canonical_answer_on_unanswerable"] is None


def make_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(open_native, "PACKAGE", tmp_path)
    packet = tmp_path / "runs/open-bank-test"
    packet.mkdir(parents=True)
    rows = open_bank.generate("test-nonce", 17)
    audit, positive = open_bank.audit(rows)
    (tmp_path / "open_bank.py").write_text("source")
    for name, value in (("bank.json", rows), ("audit.json", audit), ("positive_controls.json", positive)):
        (packet / name).write_text(json.dumps(value), encoding="utf-8")
    manifest = {"generator_version": "OPEN-TAPS-001", "training": False, "nonce": "test-nonce",
                "source_sha256": open_native.sha(tmp_path / "open_bank.py"),
                "files": {"bank.json": open_native.sha(packet / "bank.json")}}
    (packet / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    admission = {"open_bank_path": str(packet / "bank.json"), "open_bank_sha256": open_native.sha(packet / "bank.json"),
                 "open_manifest_sha256": open_native.sha(packet / "manifest.json"),
                 "open_audit_sha256": open_native.sha(packet / "audit.json"),
                 "open_positive_controls_sha256": open_native.sha(packet / "positive_controls.json"),
                 "open_source_sha256": open_native.sha(tmp_path / "open_bank.py")}
    return admission, rows, packet


def test_exact_open_reader_refuses_changed_bytes_and_historical_provenance(tmp_path, monkeypatch):
    admission, rows, packet = make_packet(tmp_path, monkeypatch)
    admitted, _ = read_open_packet(admission)
    assert len(admitted) == 24
    (packet / "bank.json").write_text("[]")
    with pytest.raises(RuntimeError): read_open_packet(admission)
    admission, rows, packet = make_packet(tmp_path / "new", monkeypatch)
    rows[0]["provenance"]["historical_parents"] = ["protected-parent"]
    (packet / "bank.json").write_text(json.dumps(rows))
    admission["open_bank_sha256"] = open_native.sha(packet / "bank.json")
    manifest = json.loads((packet / "manifest.json").read_text())
    manifest["files"]["bank.json"] = admission["open_bank_sha256"]
    (packet / "manifest.json").write_text(json.dumps(manifest))
    admission["open_manifest_sha256"] = open_native.sha(packet / "manifest.json")
    with pytest.raises(PermissionError): read_open_packet(admission)


def test_open_cap_and_strict_scoring_use_qualified_greedy_helper_without_gold_input():
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_gpu_semantics import (
        FakeModel, Tokenizer, flags, metrics, wrapper,
    )
    raw = open_bank.generate("native-fake", 17)[0]
    case = adapt_open_case(raw)
    model = FakeModel(sequence=(3,))
    events = []
    with torch.inference_mode():
        row, _ = open_native.evaluate_case(model, Tokenizer(), case,
            {"max_new_tokens": 8, "max_prompt_seconds": 900, "top_k": 10}, wrapper(),
            strict_score, flags, metrics, enforce_device=False, forward_sink=events.append)
    assert len(row["token_ids"]) == len(model.seen) == 8
    assert row["termination"] == "token_cap" and row["gold_nll"] is None and row["gold_answer"] is None
    assert row["score"] == 0 and row["format_score"] == 0
    assert len([event for event in events if event["event"] == "FORWARD_COMPLETED"]) == 8
