from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import random

import pytest

from experiments.multi_reasoning_scout_v1 import analysis, bank, budget, core, parsing, prompts, runner


class FixtureBackend:
    """Software fixture only. No pretrained efficacy can be inferred from it."""
    def __init__(self):
        self.prompts = []

    def generate(self, messages, max_new_tokens):
        self.prompts.append(messages)
        text = json.dumps({"answer": "0", "sufficient_evidence": True, "candidate_valid": True, "issues": []})
        return {"raw_text": text, "generated_ids": [1, 2, 3], "terminated": True,
                "prompt_tokens": 20, "output_tokens": 3, "total_tokens": 23,
                "generation_calls": 1, "forward_calls": 3, "wall_seconds": .01}


@pytest.fixture(scope="module")
def generated():
    return bank.generate()


@pytest.fixture
def item(generated):
    return core.PublicItem(**generated[0][0])


def record(case_id="x", role="R0", value="0", **extra):
    return {"case_id": case_id, "role": role, "parsed": {"answer": value},
            "format_failure": False, "terminated": True, **extra}


def test_generation_deterministic_and_group_disjoint(generated):
    assert generated == bank.generate()
    public, gold, worlds = generated
    assert len(public) == 120
    assert len({r["id"] for r in public}) == 120
    assert len({r["scenario_id"] for r in public}) == 120
    assert {r["id"] for r in public} == {r["id"] for r in gold} == {r["id"] for r in worlds}
    assert {s: sum(r["split"] == s for r in public) for s in ("SMOKE", "CAL", "CHECK")} == {"SMOKE": 8, "CAL": 16, "CHECK": 96}
    assert set(public[0]) == {"id", "scenario_id", "family", "split", "problem"}
    for key in core.ANSWERS:
        assert sum(g["answer"] == key for g in gold) == 24


def test_independent_solvers_agree_beyond_selected_bank():
    rng = random.Random(81235)
    for family in bank.FAMILIES:
        for _ in range(80):
            w = bank.world(rng, family)
            assert bank.solve_forward(w) == bank.solve_exhaustive(w)


def test_unknown_is_not_always_missing_marker(generated):
    rows, gold, _ = generated
    labels = {g["id"]: g for g in gold}
    assert any(labels[r["id"]]["answerable"] and "not recorded" in r["problem"] for r in rows)
    assert any(not labels[r["id"]]["answerable"] and "not recorded" not in r["problem"] for r in rows)


def test_gold_never_is_a_prompt_argument(item):
    with pytest.raises(TypeError):
        core.PublicItem(**asdict(item), gold="SECRET_GOLD")
    with pytest.raises(TypeError):
        prompts.messages({**asdict(item), "gold": "SECRET_GOLD"}, "R0")
    for role in ("R0", "D1", "D2", "D3", "R1", "R2", "R3"):
        text = json.dumps(prompts.messages(item, role))
        assert item.id not in text and item.scenario_id not in text
        assert item.split not in text
        with pytest.raises(ValueError):
            prompts.messages(item, role, [record(item.id)])


def test_selector_exact_information_projection(item):
    candidate = record(item.id, parsed={"answer": "2", "state": "brief", "issues": ["missing fact"],
                                        "sufficient_evidence": False, "gold": "SECRET_GOLD"},
                       gold="SECRET_GOLD", correctness=True, oracle_result="SECRET_ORACLE", raw_text="SECRET_RAW")
    for role in ("S1", "S2"):
        text = prompts.messages(item, role, [candidate])[1]["content"]
        assert "SECRET" not in text
        seen = json.loads(text.split("CANDIDATE RECORDS\n")[1])
        assert set(seen[0]) == {"role", "output", "format_failure", "terminated"}
        assert seen[0]["output"] == ({"answer": "2"} if role == "S1" else
                                     {"answer": "2", "state": "brief", "issues": ["missing fact"], "sufficient_evidence": False})
    with pytest.raises(ValueError):
        prompts.messages(item, "S2", [record("wrong-case")])


@pytest.mark.parametrize("text", ["nonsense", '{"answer":2}', '{"answer":"2","answer":"1"}',
                                  '```json\n{"answer":"2"}\n```', '{"answer":"2","confidence":.9}',
                                  '{"answer":"2","sufficient_evidence":"yes"}', '{"answer":"2","issues":[3]}'])
def test_malformed_visible_and_never_repaired(text):
    result = parsing.parse(text, "R0")
    assert result["format_failure"] and result["repair_count"] == 0
    assert parsing.answer(result) is None


def test_nontermination_and_checker_schema():
    assert parsing.answer(parsing.parse('{"answer":"2"}', "R0", False)) is None
    assert parsing.parse('{"answer":"2"}', "R2")["format_failure"]
    assert not parsing.parse('{"answer":"UNKNOWN","sufficient_evidence":false,"issues":[]}', "R2")["format_failure"]


def test_majority_tie_and_invalid_policy():
    assert parsing.majority([record(value="1"), record(value="2")]) == "1"
    assert parsing.majority([record(value="1", terminated=False), record(value="2")]) == "2"
    assert parsing.majority([record(terminated=False)]) is None


def test_arms_calls_caps_raw_preservation_and_repeat_identity(item, tmp_path):
    a, b = tmp_path/"a", tmp_path/"b"
    a.mkdir(); b.mkdir()
    x, y = FixtureBackend(), FixtureBackend()
    first = runner.run_item(item, x, a, {"fixture": True})
    second = runner.run_item(item, y, b, {"fixture": True})
    assert first == second
    assert len(x.prompts) == 11
    assert sum(r["max_new_tokens"] for r in first["outputs"].values()) == 960
    assert len(first["logical_calls"]["A2"]) == len(first["logical_calls"]["A3"]) == 5
    assert sum(first["outputs"][k]["max_new_tokens"] for k in first["logical_calls"]["A3"]) == 320
    for key, r in first["outputs"].items():
        assert r["case_id"] == item.id
        assert r["raw_text"] == core.read_json(a/f"{item.id}.{key}.json")["raw_text"]
        assert r["prompt_sha256"] == core.digest(r["prompt"])
    with pytest.raises(FileExistsError):
        runner.run_item(item, x, a, {"fixture": True})


def test_sequential_variant_information_flow(item):
    backend = FixtureBackend()
    result = runner.run_m2_item(item, backend)
    assert len(result) == 5
    assert "CANDIDATE RECORDS" not in backend.prompts[0][1]["content"]
    assert all("CANDIDATE RECORDS" in p[1]["content"] for p in backend.prompts[1:])


def test_oracle_is_analysis_only():
    for module in (runner, prompts):
        text = Path(module.__file__).read_text()
        assert "gold.jsonl" not in text
        assert "from .analysis" not in text
        assert "solve_exhaustive" not in text


def test_receipts_and_file_hashes_reject_mutations(tmp_path):
    f = tmp_path/"tokenizer.json"
    f.write_text("original")
    hashes = {f.name: core.file_hash(f)}
    core.verify_files(tmp_path, hashes)
    f.write_text("changed")
    with pytest.raises(ValueError): core.verify_files(tmp_path, hashes)
    b = {"model": "first", "source": "abc"}
    receipt = {"status": "PASS", "stage": "NATIVE_QUALIFICATION", "binding": b}
    core.validate_receipt(receipt, b, "NATIVE_QUALIFICATION")
    for key in b:
        with pytest.raises(ValueError):
            core.validate_receipt(receipt, {**b, key: "changed"}, "NATIVE_QUALIFICATION")
    with pytest.raises(ValueError): core.validate_receipt({**receipt, "status": "FAIL"}, b, "NATIVE_QUALIFICATION")


def test_changed_checkpoint_rejected(tmp_path):
    weight, tokenizer = tmp_path/"model.safetensors", tmp_path/"tokenizer.json"
    weight.write_bytes(b"fixture"); tokenizer.write_text("fixture")
    model = {"path": str(tmp_path), "assets": {p.name: core.file_hash(p) for p in (weight, tokenizer)}}
    core.model_assets(model)
    weight.write_bytes(b"changed")
    with pytest.raises(ValueError): core.model_assets(model)


def good_resources():
    return {k: 10*budget.GIB for k in ("host_available_bytes", "gpu_free_bytes", "disk_free_bytes")} | {
        "host_rss_bytes": budget.GIB, "gpu_allocated_bytes": budget.GIB, "gpu_peak_bytes": budget.GIB}


@pytest.mark.parametrize("key", ["host_available_bytes", "gpu_free_bytes", "disk_free_bytes"])
def test_resource_floors_fail_closed(key):
    value = good_resources(); budget.check(value, preload=True)
    value[key] = 0
    with pytest.raises(RuntimeError): budget.check(value, preload=True)


@pytest.mark.parametrize("key", ["host_rss_bytes", "gpu_allocated_bytes"])
def test_memory_ceiling(key):
    value = good_resources(); value[key] = 7 * budget.GIB
    with pytest.raises(RuntimeError): budget.check(value)


def test_budget_ceilings(tmp_path):
    m = budget.Meter(tmp_path, sample=lambda **kw: good_resources())
    m.calls = budget.LIMITS["max_calls_per_model"]
    with pytest.raises(RuntimeError): m.reserve(1)
    m.calls = 0; m.allowance = budget.LIMITS["max_generated_tokens_per_model"]
    with pytest.raises(RuntimeError): m.reserve(1)
    m.started -= budget.LIMITS["max_seconds_per_model"] + 1
    with pytest.raises(RuntimeError): m.guard()


def test_evidence_is_create_only(tmp_path):
    path = tmp_path/"evidence"
    core.new_dir(path)
    with pytest.raises(FileExistsError): core.new_dir(path)
    core.write_new(path/"r.json", {"once": True})
    with pytest.raises(FileExistsError): core.write_new(path/"r.json", {"twice": True})


def test_bank_build_never_reads_a_forbidden_bank(tmp_path, monkeypatch):
    root = tmp_path/"records"; root.mkdir()
    protected = tmp_path/"SEALED_ANY_READ.jsonl"; protected.write_text("MUST_NOT_READ")
    core.write_new(root/"ACCESS_MANIFEST.json", {"quarantined_ids": [], "entries": [{"path": str(protected), "bars": ["ANY_READ"]}]})
    monkeypatch.setattr(bank, "RECORDS", root)
    real = Path.open
    reads = []
    def guarded(path, mode="r", *args, **kwargs):
        if "r" in mode:
            reads.append(str(path))
            assert path != protected
        return real(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    bank.build()
    assert str(protected) not in reads


def test_quarantined_id_prevents_bank_build(tmp_path, generated, monkeypatch):
    core.write_new(tmp_path/"ACCESS_MANIFEST.json", {"quarantined_ids": [generated[0][0]["id"]]})
    monkeypatch.setattr(bank, "RECORDS", tmp_path)
    with pytest.raises(ValueError): bank.build()
    assert not (tmp_path/"bank").exists()


def test_paired_counts_and_world_ci():
    p = analysis.paired_counts([False, True, True, False], [True, False, True, True])
    assert (p["repairs"], p["damage"], p["net_gain"], p["delta_accuracy"]) == (2, 1, 1, .25)
    assert analysis.paired_counts([False]*6, [True]*6)["exact_McNemar_two_sided_p"] == .03125
    ci = analysis.cluster_ci([1, 1, -1, -1], ["a", "a", "b", "b"])
    assert ci["world_clusters"] == 2


def factorial_fixture(bits):
    A, B, C, D = bits
    common = {k: "fixed" for k in ("freeze_sha256", "bank_sha256", "prompt_sha256", "protocol_sha256")}
    base = {"id": "id", "scenario_id": "world", "family": "family", "split": "CHECK"}
    return ({**common, "per_item": [{**base, "correct": {"A0": A, "A3": B}}]},
            {**common, "per_item": [{**base, "correct": {"A0": C, "A3": D}}]})


@pytest.mark.parametrize("bits", list(itertools.product((False, True), repeat=4)))
def test_all_16_future_factorial_cells(bits):
    A, B, C, D = bits
    old, new = factorial_fixture(bits)
    out = analysis.factorial(old, new)
    assert out["overall"]["deltas"] == {"delta_training_direct": int(C)-int(A), "delta_multi_old": int(B)-int(A),
        "delta_multi_new": int(D)-int(C), "interaction": int(D)-int(C)-int(B)+int(A)}
    row = out["per_item"][0]
    assert row["training_only_repair"] == (not A and C)
    assert row["multi_only_repair"] == (not A and B)
    assert row["both_available"] == (not A and B and C)
    assert row["posttrain_multi_extra"] == (not C and D)
    assert row["multi_lost_after_training"] == (B and not D)
    assert row["harm_from_training"] == (A and not C)
    assert row["harm_from_multi_pretrain"] == (A and not B)
    assert row["harm_from_multi_posttrain"] == (C and not D)


def test_future_mismatch_not_silently_intersected():
    old, new = factorial_fixture((False, True, False, True))
    new["per_item"][0]["id"] = "different"
    with pytest.raises(ValueError): analysis.factorial(old, new)
    old, new = factorial_fixture((False, True, False, True))
    new["prompt_sha256"] = "changed"
    with pytest.raises(ValueError): analysis.factorial(old, new)


def test_full_fixture_analysis_correct_candidate_but_selector_fails(item, tmp_path):
    value = runner.run_item(item, FixtureBackend(), tmp_path, {"fixture": True})
    value["outputs"]["R1"]["parsed"]["answer"] = "1"
    labels = {item.id: {"answer": "1", "answerable": True}}
    report = analysis.summarize([value], labels)
    assert report["A4_oracle_analysis_only"]["A3_candidate_ceiling"] == 1
    assert report["arms"]["A3"]["accuracy"] == 0
    assert report["selectors"]["S2"]["correct_candidate_existed_selector_wrong"] == 1
    assert report["complementarity"]["R0:R1"]["second_only_correct"] == 1
    assert report["cost_per_net_recovered_case"]["wall_seconds"] is None


def test_partial_failure_retains_raw_calls(item, tmp_path):
    class Failing(FixtureBackend):
        def generate(self, messages, max_new_tokens):
            if len(self.prompts) == 2:
                raise RuntimeError("fixture failure")
            return super().generate(messages, max_new_tokens)
    with pytest.raises(RuntimeError): runner.run_item(item, Failing(), tmp_path, {"fixture": True})
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert not list(tmp_path.glob("*.item.json"))


def test_source_bank_and_prompt_freeze_reject_drift(tmp_path, monkeypatch):
    root = tmp_path/"root"; root.mkdir()
    records = root/"records"; records.mkdir()
    artifact = root/"bank.jsonl"; artifact.write_text("frozen fixture")
    source = {"repository": "fixture", "branch": "fixture", "sha": "frozen-sha", "files": {"prompts.py": "frozen"}}
    freeze = {"files": {"bank.jsonl": core.file_hash(artifact)}, "source_files": {"prompts.py": "frozen"}, "source": source}
    core.write_new(records/"FREEZE.json", freeze)
    monkeypatch.setattr(core, "ROOT", root)
    monkeypatch.setattr(core, "RECORDS", records)
    monkeypatch.setattr(core, "source_identity", lambda: source)
    monkeypatch.setattr(core, "git", lambda *args: "frozen-sha")
    monkeypatch.setattr(core, "source_files", lambda: {"prompts.py": "frozen"})
    assert core.load_frozen() == freeze
    monkeypatch.setattr(core, "source_files", lambda: {"prompts.py": "changed"})
    with pytest.raises(ValueError, match="stale frozen"): core.load_frozen()
    monkeypatch.setattr(core, "source_files", lambda: {"prompts.py": "frozen"})
    artifact.write_text("changed bank")
    with pytest.raises(ValueError, match="frozen artifact changed"): core.load_frozen()


def test_worker_cannot_bypass_supervisor(tmp_path, monkeypatch):
    from argparse import Namespace
    from experiments.model_quality_gpu_v1.supervisor import CONTEXT_ENV
    monkeypatch.delenv(CONTEXT_ENV, raising=False)
    with pytest.raises(ValueError, match="supervisor context absent"):
        runner.worker(Namespace(out=tmp_path, model="supa", future_registration=None))


def test_smoke_failure_blocks_even_when_some_roles_parse(item, tmp_path):
    row = runner.run_item(item, FixtureBackend(), tmp_path, {"fixture": True})
    assert runner.smoke_pass([row]*8)
    row["outputs"]["R2"]["format_failure"] = True
    assert not runner.smoke_pass([row]*8)
