from collections import Counter
from copy import deepcopy
import json
import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_native import (
    GRID, admit_validation, count_forecast, deterministic_order, freeze_development,
)


def calibration():
    cases = [(f"case-{index}", f"parent-{index//2}", f"category-{index//32}") for index in range(192)]
    rows = [{"split": "DEVELOPMENT", "case_id": case, "parent_id": parent, "category": category,
             "seed": 17, "arm": "A2", "fixed_alpha": alpha, "score": int(alpha == .9)}
            for case, parent, category in cases for alpha in GRID]
    rows += [{"split": "DEVELOPMENT", "case_id": case, "parent_id": parent, "category": category,
              "seed": 17, "arm": "A3", "token_ids": [9]} for case, parent, category in cases]
    traces = [{"split": "DEVELOPMENT", "case_id": case, "seed": 17, "configured_seed": 17,
               "arm": "A3", "forward_index": 1, "iteration": iteration,
               "alpha": 1. if iteration == 0 else 1.2, "evidence_kind": "native_trajectory"}
              for case, _, _ in cases for iteration in (0, 1)]
    return cases, rows, traces


def test_once_only_deterministic_calibration_and_exact_full_cost_forecast():
    cases, rows, traces = calibration()
    freeze = freeze_development(rows, traces, "bank", "tokens", "runner", [case for case, _, _ in cases])
    assert freeze["alpha"] == .9 and len(freeze["selection_parent_ids"]) == 96
    assert freeze["pool"] == [1.2] * 192 and freeze["deterministic_seed"] == 17
    forecast = count_forecast()
    assert forecast["bulk_physical_evaluations"] == 288*11 == 3168
    assert forecast["physical_evaluations"] == 3296
    assert forecast["generation_call_cap"] == 3168*8 + 128*24 == 28416
    assert forecast["model_API_attempt_cap"] == 28436


def test_deterministic_seed_duplicates_missing_variants_and_validation_leaks_refuse():
    cases, rows, traces = calibration()
    with pytest.raises(RuntimeError): freeze_development(rows + [rows[0]], traces, "b", "t", "r")
    changed = deepcopy(rows); changed[0]["seed"] = 29
    with pytest.raises(RuntimeError): freeze_development(changed, traces, "b", "t", "r")
    with pytest.raises(RuntimeError): freeze_development(rows, traces[:-1], "b", "t", "r")
    with pytest.raises(PermissionError):
        freeze_development(rows + [{"split": "OPEN_VALIDATION"}], traces, "b", "t", "r")


def test_parent_validation_firewall_not_just_case_id_and_digest():
    _, rows, traces = calibration()
    freeze = freeze_development(rows, traces, "b", "t", "r")
    admit_validation(freeze, {"id": "val", "parent_id": "new-world", "split": "OPEN_VALIDATION"}, "b", "t", "r")
    with pytest.raises(PermissionError):
        admit_validation(freeze, {"id": "val", "parent_id": "parent-0", "split": "OPEN_VALIDATION"}, "b", "t", "r")
    changed = deepcopy(freeze); changed["alpha"] = 1.
    with pytest.raises(PermissionError):
        admit_validation(changed, {"id": "val", "parent_id": "new-world", "split": "OPEN_VALIDATION"}, "b", "t", "r")


def test_output_blind_rotation_reversal_covers_nine_cells_once_and_balances_positions():
    parents = {f"parent-{i}": i for i in range(144)}
    variants = {(parent, variant): index for parent in parents for index, variant in enumerate(("a", "b"))}
    positions = {name: Counter() for name in ("A0", "A1", *(f"A2:{alpha:g}" for alpha in GRID), "A3")}
    for parent in parents:
        for variant in ("a", "b"):
            raw = {"parent_id": parent, "variant_id": variant, "gold": "unread"}
            order = deterministic_order(raw, parents, variants)
            assert len(order) == len(set(order)) == 9
            for index, name in enumerate(order): positions[name][index] += 1
            changed = {**raw, "gold": "different", "prompt": "other wording"}
            assert deterministic_order(changed, parents, variants) == order
    assert all(max(counts.values())-min(counts.values()) <= 2 for counts in positions.values())


def test_reader_matches_real_bulk_provenance_and_binds_reused_normalizer(tmp_path, monkeypatch):
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_bank, bulk_native
    rows = bulk_bank.generate("reader-regression", 17)
    package = tmp_path / "package"; packet = package / "runs/bulk-bank-test"
    packet.mkdir(parents=True)
    (package / "bulk_bank.py").write_text("source")
    (package / "open_bank.py").write_text("normalizer")
    for name, data in (("bank.json", rows), ("audit.json", {"status": "PASS_PREPARATION_ONLY", "failures": [], "positive_control_correct": 288}),
                       ("positive_controls.json", [{"id": row["id"], "correct": True} for row in rows])):
        (packet / name).write_text(json.dumps(data))
    manifest = {"generator_version": bulk_bank.VERSION, "training": False, "status": "PASS_PREPARATION_ONLY", "nonce": "reader-regression",
                "source_sha256": bulk_native.sha(package / "bulk_bank.py"), "normalizer_source_sha256": bulk_native.sha(package / "open_bank.py"),
                "files": {"bank.json": bulk_native.sha(packet / "bank.json")}}
    (packet / "manifest.json").write_text(json.dumps(manifest))
    admission = {"bulk_bank_path": str(packet / "bank.json"), "bulk_bank_sha256": bulk_native.sha(packet / "bank.json"),
                 "bulk_source_sha256": manifest["source_sha256"], "bulk_normalizer_source_sha256": manifest["normalizer_source_sha256"],
                 **{f"bulk_{key}_sha256": bulk_native.sha(packet / file) for key, file in
                    (("manifest", "manifest.json"), ("audit", "audit.json"), ("positive_controls", "positive_controls.json"))}}
    monkeypatch.setattr(bulk_native, "PACKAGE", package)
    admitted, _ = bulk_native.read_bulk_packet(admission)
    assert len(admitted) == 288 and len({row["parent_id"] for row in admitted}) == 144
    swap = [row for row in admitted if row["category"] == "answerability_source_conflict"]
    assert any(len({row["gold"] for row in swap if row["parent_id"] == parent}) == 2 for parent in {row["parent_id"] for row in swap})
    (package / "open_bank.py").write_text("changed normalizer")
    with pytest.raises(RuntimeError, match="normalizer"):
        bulk_native.read_bulk_packet(admission)


def test_actual_export_packet_reader_and_parent_disjoint_contract_without_tokenizer():
    from pathlib import Path
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_native
    package = Path(bulk_native.__file__).resolve().parent
    packet = package / "runs/bulk-bank-apparatus-001"
    admission = {"bulk_bank_path": str(packet / "bank.json"), "bulk_bank_sha256": bulk_native.sha(packet / "bank.json"),
                 "bulk_source_sha256": bulk_native.sha(package / "bulk_bank.py"),
                 "bulk_normalizer_source_sha256": bulk_native.sha(package / "open_bank.py"),
                 **{f"bulk_{key}_sha256": bulk_native.sha(packet / file) for key, file in
                    (("manifest", "manifest.json"), ("audit", "audit.json"), ("positive_controls", "positive_controls.json"))}}
    rows, manifest = bulk_native.read_bulk_packet(admission)
    assert len(rows) == 288 and manifest["nonce"] == "9463250d48f1"
    dev = {row["parent_id"] for row in rows if row["split"] == "DEVELOPMENT"}
    validation = {row["parent_id"] for row in rows if row["split"] == "OPEN_VALIDATION"}
    assert len(dev) == 96 and len(validation) == 48 and not dev & validation
