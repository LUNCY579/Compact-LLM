"""Synthetic scored-JSON tests only. No checkpoints, tokenizers or native collectors."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import screen
from soup import ARMS, ASSETS, EXPERIMENT, PINS, sha256


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def lab(tmp_path):
    f = {"schema": 1, "experiment": EXPERIMENT, "protocol_sha256": screen.PROTOCOL_SHA256,
         "code_sha256": {**screen.INHERITED_CODE, "screen.py": sha256(Path(screen.__file__))},
         "tokenizer_sha256": ASSETS["tokenizer.json"], "asset_bundle_sha256": screen.digest(ASSETS),
         "checkpoint_sha256": dict(PINS), "build_receipts": {}, "banks": {}}
    for k in (*screen.RUNTIME_KEYS, "qualification_receipt_sha256", "ancestry_audit_sha256"):
        f[k] = screen.digest(k)
    for arm in screen.NEW_ARMS:
        f["checkpoint_sha256"][arm] = screen.digest(arm)
        r = {"experiment": EXPERIMENT, "arm": arm, "status": "MATERIALIZED_NOT_EVALUATED",
             "pair": list(ARMS[arm]), "alpha_right": 0.5, "stored_dtype": "float32",
             "weight_sha256": f["checkpoint_sha256"][arm], "sources_rehashed_after_build": True,
             "optimizer_steps": 0, "model_forwards": 0, "tensor_roundtrip_valid": True,
             "files": {k: {"model.safetensors": PINS[k], **ASSETS} for k in {"supa", *ARMS[arm][:2]}}}
        p = write(tmp_path / (arm + "-build.json"), r)
        f["build_receipts"][arm] = {"path": p.name, "sha256": sha256(p)}
    for split in ("CAL", "CHECK"):
        rows = [{"id": f"{split}-{d}-{i}", "parent_id": f"parent-{split}-{d}-{i}",
                 "domain": d, "answerable": d != "unanswerable_abstention"}
                for d in screen.DOMAINS for i in range(40)]
        f["banks"][split] = {"suite_id": "SYNTHETIC-" + split, "dataset_sha256": screen.digest(split),
                             "row_manifest": rows, "minimum_source_correct": {d: 10 for d in screen.DOMAINS},
                             "donor_min_correct": {"titan": 25, "titan_sparse": 25, "opallo": 25},
                             "shortcut_correct": {"transition": 5, "arithmetic": 5}}
    freeze = write(tmp_path / "freeze.json", f)
    for split in ("CAL", "CHECK"):
        folder = tmp_path / split
        folder.mkdir()
        arms = list(PINS) + (list(screen.NEW_ARMS) if split == "CAL" else ["sparse_opallo_50"])
        for arm in arms:
            meta = {"experiment": EXPERIMENT, "arm_id": arm, "split": split,
                    "suite_id": f["banks"][split]["suite_id"], "dataset_sha256": f["banks"][split]["dataset_sha256"],
                    "checkpoint_sha256": f["checkpoint_sha256"][arm], "freeze_sha256": sha256(freeze),
                    "tokenizer_sha256": f["tokenizer_sha256"], "asset_bundle_sha256": f["asset_bundle_sha256"],
                    "evaluation_dtype": "float32", **{k: f[k] for k in screen.CONTRACT_KEYS}}
            rows = copy.deepcopy(f["banks"][split]["row_manifest"])
            for r in rows:
                limit = 20
                if r["domain"] == "transition" and arm in ("titan", "titan_sparse", *screen.CANDIDATES):
                    limit = 30
                if r["domain"] == "arithmetic" and arm in ("opallo", *screen.CANDIDATES):
                    limit = 30
                r["correct"] = int(r["id"].rsplit("-", 1)[1]) < limit
                r["abstained"] = not r["answerable"] and r["correct"]
            write(folder / (arm + ".json"), {"metadata": meta, "rows": rows})
    return tmp_path, f


def mutate(path, change):
    obj = json.loads(path.read_text())
    change(obj)
    write(path, obj)


def run_cal(lab):
    root, _ = lab
    return screen.execute(root / "freeze.json", root / "CAL", "CAL")


def test_exact_inherited_source_and_protocol(lab):
    f, _ = screen.load_freeze(lab[0] / "freeze.json")
    assert f["protocol_sha256"] == screen.PROTOCOL_SHA256


def test_nine_configuration_primary_tiebreak(lab):
    r = run_cal(lab)
    assert r["selected_candidate"] == "sparse_opallo_50"
    assert r["configuration_count"] == 9 and r["scored_row_count"] == 2160
    assert r["assessments"]["sparse_opallo_50"]["eligible"]
    assert {"supa_sparse_50", "supa_opallo_50"} <= set(r["comparisons"]["sparse_opallo_50"])
    assert not r["native_execution_authorized"] and not r["quality_improvement_established"]


@pytest.mark.parametrize("field", ["checkpoint_sha256", "tokenizer_sha256", "runtime_sha256", "scorer_sha256",
                                   "decoding_sha256", "freeze_sha256", "asset_bundle_sha256", "evaluation_dtype", "arm_id", "split",
                                   "qualification_receipt_sha256", "ancestry_audit_sha256"])
def test_run_metadata_must_match_frozen_identity(lab, field):
    root, _ = lab
    mutate(root / "CAL/sparse_opallo_50.json", lambda x: x["metadata"].update({field: "f" * 64}))
    with pytest.raises(ValueError):
        run_cal(lab)


@pytest.mark.parametrize("kind", ["missing", "extra", "local"])
def test_exact_arm_inventory(lab, kind):
    root, _ = lab
    if kind == "missing":
        (root / "CAL/supa_opallo_50.json").unlink()
    else:
        write(root / "CAL" / ("local_parent_50.json" if kind == "local" else "extra.json"), {})
    with pytest.raises(ValueError):
        run_cal(lab)


@pytest.mark.parametrize("kind", ["missing_row", "duplicate_row", "parent", "domain", "correct_type", "missing_abstained", "answerable_type"])
def test_exact_rows_and_boolean_scoring(lab, kind):
    def change(x):
        if kind == "missing_row": x["rows"].pop()
        elif kind == "duplicate_row": x["rows"].append(copy.deepcopy(x["rows"][0]))
        elif kind == "parent": x["rows"][0]["parent_id"] = "wrong"
        elif kind == "domain": x["rows"][0]["domain"] = "arithmetic"
        elif kind == "correct_type": x["rows"][0]["correct"] = 1
        elif kind == "missing_abstained": x["rows"][0].pop("abstained")
        else: x["rows"][0]["answerable"] = 1
    mutate(lab[0] / "CAL/sparse_opallo_50.json", change)
    with pytest.raises(ValueError): run_cal(lab)


@pytest.mark.parametrize("field,value", [("pair", ["titan_sparse", "opallo", 0.25]),
                                          ("alpha_right", 0.25), ("stored_dtype", "bfloat16"),
                                          ("weight_sha256", "f" * 64), ("sources_rehashed_after_build", False),
                                          ("optimizer_steps", 1), ("status", "FAILED")])
def test_build_receipt_contract_not_just_hash_shape(lab, field, value):
    root, f = lab
    p = root / f["build_receipts"]["sparse_opallo_50"]["path"]
    mutate(p, lambda x: x.update({field: value}))
    f["build_receipts"]["sparse_opallo_50"]["sha256"] = sha256(p)
    write(root / "freeze.json", f)
    with pytest.raises(ValueError): screen.load_freeze(root / "freeze.json")


def test_receipt_hash_tamper(lab):
    root, f = lab
    p = root / f["build_receipts"]["sparse_opallo_50"]["path"]
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError): screen.load_freeze(root / "freeze.json")


@pytest.mark.parametrize("kind", ["pin", "code", "protocol", "unbound_runtime", "missing_domain", "duplicate_parent", "cross_split", "same_dataset", "gold_in_manifest", "too_many_rows", "zero_floor", "shortcut_floor"])
def test_freeze_refuses_unsafe_or_unbound_registration(lab, kind):
    root, f = lab
    bank = f["banks"]["CAL"]
    if kind == "pin": f["checkpoint_sha256"]["supa"] = "f" * 64
    elif kind == "code": f["code_sha256"]["score.py"] = "f" * 64
    elif kind == "protocol": f["protocol_sha256"] = "f" * 64
    elif kind == "unbound_runtime": f["runtime_sha256"] = None
    elif kind == "missing_domain": bank["row_manifest"] = [r for r in bank["row_manifest"] if r["domain"] != "arithmetic"]
    elif kind == "duplicate_parent": bank["row_manifest"][1]["parent_id"] = bank["row_manifest"][0]["parent_id"]
    elif kind == "cross_split": f["banks"]["CHECK"]["row_manifest"][0]["parent_id"] = bank["row_manifest"][0]["parent_id"]
    elif kind == "same_dataset": f["banks"]["CHECK"]["dataset_sha256"] = bank["dataset_sha256"]
    elif kind == "gold_in_manifest": bank["row_manifest"][0]["gold"] = "answer"
    elif kind == "too_many_rows": bank["row_manifest"] *= 2
    elif kind == "zero_floor": bank["minimum_source_correct"]["arithmetic"] = 0
    else: bank["donor_min_correct"]["opallo"] = bank["shortcut_correct"]["arithmetic"]
    write(root / "freeze.json", f)
    with pytest.raises(ValueError): screen.load_freeze(root / "freeze.json")


def lose_rows(root, arm, domain, count):
    def change(x):
        affected = [r for r in x["rows"] if r["domain"] == domain][:count]
        for row in affected: row["correct"] = False
    mutate(root / "CAL" / (arm + ".json"), change)


def test_exact_five_percent_source_losses(lab):
    root, _ = lab
    lose_rows(root, "sparse_opallo_50", "short_format", 1)
    r = run_cal(lab)
    assert r["assessments"]["sparse_opallo_50"]["eligible"]  # 1/20 allowed
    assert r["selected_candidate"] == "titan_opallo_50"  # lower losses wins
    lose_rows(root, "sparse_opallo_50", "short_format", 2)
    r = run_cal(lab)
    assert "source_loss_over_5pct:short_format" in r["assessments"]["sparse_opallo_50"]["reasons"]


def test_new_gains_cannot_hide_source_losses(lab):
    root, _ = lab
    lose_rows(root, "sparse_opallo_50", "short_format", 2)
    def gain(x):
        for r in x["rows"]:
            if r["domain"] == "short_format" and int(r["id"].rsplit("-", 1)[1]) >= 20: r["correct"] = True
    mutate(root / "CAL/sparse_opallo_50.json", gain)
    r = run_cal(lab)["assessments"]["sparse_opallo_50"]
    assert r["retention"]["short_format"]["net_correct"] > 0 and not r["eligible"]


@pytest.mark.parametrize("kept,passes", [(29, True), (28, False)])
def test_acquisition_95_percent_is_integer_exact(lab, kept, passes):
    def change(x):
        for row in x["rows"]:
            if row["domain"] == "transition": row["correct"] = int(row["id"].rsplit("-", 1)[1]) < kept
    mutate(lab[0] / "CAL/sparse_opallo_50.json", change)
    r = run_cal(lab)["assessments"]["sparse_opallo_50"]
    assert r["eligible"] is passes


def test_false_abstention_not_hidden_by_correctness(lab):
    def change(x):
        row = next(r for r in x["rows"] if r["answerable"] and not r["correct"])
        row["abstained"] = True
    mutate(lab[0] / "CAL/sparse_opallo_50.json", change)
    assert "increased_false_abstention" in run_cal(lab)["assessments"]["sparse_opallo_50"]["reasons"]


def test_weak_donor_not_automatic_competence(lab):
    def change(x):
        for r in x["rows"]:
            if r["domain"] == "arithmetic": r["correct"] = False
    mutate(lab[0] / "CAL/opallo.json", change)
    r = run_cal(lab)
    assert r["selected_candidate"] is None
    assert all("donor_competence:opallo" in a["reasons"] for a in r["assessments"].values())


def test_zero_source_correct_denominator_refuses(lab):
    def change(x):
        for r in x["rows"]:
            if r["domain"] == "long_context": r["correct"] = False
    mutate(lab[0] / "CAL/supa.json", change)
    with pytest.raises(ValueError, match="denominator"): run_cal(lab)


def test_no_eligible_candidate_no_promotion(lab):
    for arm in screen.CANDIDATES: lose_rows(lab[0], arm, "short_format", 2)
    assert run_cal(lab)["selected_candidate"] is None


def test_cal_never_reads_check_predictions(lab):
    for p in (lab[0] / "CHECK").glob("*.json"): p.write_text("INVALID CHECK OUTPUT")
    assert run_cal(lab)["selected_candidate"] == "sparse_opallo_50"


def test_check_requires_sealed_lock(lab):
    root, _ = lab
    with pytest.raises(ValueError): screen.execute(root / "freeze.json", root / "CHECK", "CHECK")


def test_check_five_configs_no_reselection(lab):
    root, _ = lab
    lock = root / "lock.json"
    screen.write_new(lock, run_cal(lab))
    def damage(x):
        for r in x["rows"]: r["correct"] = False
    mutate(root / "CHECK/sparse_opallo_50.json", damage)
    r = screen.execute(root / "freeze.json", root / "CHECK", "CHECK", lock, sha256(lock))
    assert r["configuration_count"] == 5 and r["selected_candidate"] == "sparse_opallo_50"
    assert not r["assessments"]["sparse_opallo_50"]["eligible"]
    assert not r["attenuation_controls_available_in_this_split"]
    assert set(r["comparisons"]["sparse_opallo_50"]) == set(PINS)


def test_check_lock_edit_rejected(lab):
    root, _ = lab
    lock = write(root / "lock.json", run_cal(lab))
    saved_hash = sha256(lock)
    mutate(lock, lambda x: x.update({"selected_candidate": "titan_opallo_50"}))
    with pytest.raises(ValueError): screen.execute(root / "freeze.json", root / "CHECK", "CHECK", lock, saved_hash)


def test_exclusive_lock_creation(lab):
    path = lab[0] / "lock.json"
    screen.write_new(path, run_cal(lab))
    before = path.read_bytes()
    with pytest.raises(FileExistsError): screen.write_new(path, {})
    assert path.read_bytes() == before


def test_no_native_model_modules_imported():
    assert "transformers" not in sys.modules
