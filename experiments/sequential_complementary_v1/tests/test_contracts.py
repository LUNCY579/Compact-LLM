"""Adversarial CPU fixtures. No pretrained loading or quality measurements."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from safetensors.torch import save_file

from experiments.sequential_complementary_v1 import analysis, data, delta, reasoning, training, workflow
from experiments.sequential_complementary_v1.integrity import (
    ContractError, canonical, code_digest, digest, file_hash, loads, read_json, write_new,
)

H = "a" * 64


def world(i=0, split="TRAIN", unknown=False):
    key = f"{split}_{i}"
    return {"id": key, "world_id": key, "structure_id": "structure_" + key,
            "template_id": "template_" + key, "entity_pool_id": "pool_" + key,
            "split": split, "family": "state", "question": "What is the final state?",
            "prompt": [{"role": "system", "content": "Use stated evidence; end with FINAL: <answer>."},
                       {"role": "user", "content": f"Record {key}. Start with {i}. Add one. What is the final state?"}],
            "reference": "UNKNOWN" if unknown else str(i + 1),
            "task": {"start": i, "missing": unknown}, "output_contract": "FINAL_LINE_V1"}


def solver(case, task):
    assert isinstance(case, data.PublicCase)
    return ("UNKNOWN", ["The initial state is absent."]) if task["missing"] else (
        str(task["start"] + 1), [f"Add one to {task['start']}."])


class Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        p = [1] + [10 + ord(c) for c in messages[0]["content"] + messages[1]["content"]] + [2]
        if add_generation_prompt:
            return p
        return p + [10 + ord(c) for c in messages[-1]["content"]] + [3, 4]

    def encode(self, text, **kwargs):
        return [10 + ord(c) for c in text]


def compiler(prompt, target, tokenizer, **kwargs):
    x = data.render_tokens(tokenizer, prompt, target)
    n = len(x["prompt_ids"])
    labels = [-100] * n + x["ids"][n:-1] + [-100]
    return {**x, "labels": labels, "post_eos_tokens_supervised": False, "truncated": False}


def scored(bits, *, shared=False):
    return [{"id": str(i), "world_id": "same" if shared else str(i), "family": "state", "correct": b}
            for i, b in enumerate(bits)]


class Backend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def check_context(self, messages, cap):
        assert cap in (32, 64, 288)
        assert all(set(m) == {"role", "content"} for m in messages)

    def generate(self, messages, cap):
        self.prompts.append(copy.deepcopy(messages))
        text = next(self.outputs)
        return {"text": text, "terminated": True, "input_tokens": 20, "output_tokens": 5}


def budget(calls=5, tokens=288):
    return reasoning.Budget(calls, tokens, 30, lambda x: None)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_ambiguous_json_rejected(text):
    with pytest.raises(ContractError):
        loads(text)


def test_create_only(tmp_path):
    path = tmp_path / "receipt.json"
    write_new(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_new(path, {"a": 2})
    assert read_json(path) == {"a": 1}


@pytest.mark.parametrize("keep", [-1, 1.1, float("nan"), True])
def test_bad_keep(keep):
    with pytest.raises(ContractError):
        delta.magnitude_mask(torch.ones(4), keep)


def test_mask_is_not_scaling_and_ties_are_stable():
    original = torch.tensor([1., -1., 1., -1., 2., -2., .1, .2])
    kept = delta.magnitude_mask(original)
    assert kept.tolist() == [0, 0, 0, 0, 2, -2, 0, 0]
    assert not torch.equal(kept, original * .25)
    assert delta.magnitude_mask(torch.tensor([1., -1., 1., -1.])).tolist() == [1, 0, 0, 0]
    assert original[0].item() == 1


@pytest.mark.parametrize("keep", [0, 1])
def test_mask_endpoints(keep):
    d = torch.arange(8, dtype=torch.float32)
    assert torch.equal(delta.magnitude_mask(d, keep), d if keep else d * 0)


@pytest.mark.parametrize("value", [torch.ones(4, dtype=torch.bfloat16), torch.tensor([float("nan")]), torch.empty(0)])
def test_delta_bad_tensor(value):
    with pytest.raises(ContractError):
        delta.magnitude_mask(value)


def test_rank_reconstruction_and_norm_control():
    d = torch.diag(torch.tensor([4., 2., 1.]))
    low, left, right, control, stats = delta.truncate(d, 1)
    torch.testing.assert_close(low, left @ right)
    torch.testing.assert_close(torch.linalg.vector_norm(low), torch.linalg.vector_norm(control))
    assert torch.count_nonzero(control).item() == 3
    assert stats["factor_payload_bytes"] == 24
    assert stats["dense_payload_bytes"] == 36


def test_zero_rank_control():
    low, _, _, control, stats = delta.truncate(torch.zeros(3, 3), 2)
    assert torch.equal(low, control) and stats["norm_ratio"] == 0


@pytest.mark.parametrize("rank", [0, -1, True, 4])
def test_invalid_rank(rank):
    with pytest.raises(ContractError):
        delta.truncate(torch.ones(3, 3), rank)


def test_compression_uses_masked_basis():
    d = torch.tensor([[4., 2.], [1., 3.]])
    low, stats, _ = delta.transform(d, "rank", rank=2)
    torch.testing.assert_close(low, delta.magnitude_mask(d))
    assert stats["basis"] == "incumbent"
    full, _, _ = delta.transform(d, "rank", rank=2, basis="full_delta")
    torch.testing.assert_close(full, d)
    assert not torch.equal(full, low)


def test_nonmatrix_policy():
    d = torch.tensor([4., 2., 1., 3.])
    value, stats, factors = delta.transform(d, "rank")
    assert torch.equal(value, delta.magnitude_mask(d)) and factors is None
    assert stats["nonmatrix_policy"] == "preserve_basis_update"


def checkpoint_inputs(tmp_path):
    src, trn = tmp_path / "src.safetensors", tmp_path / "trn.safetensors"
    save_file({"w": torch.zeros(4, 4), "untouched": torch.ones(2)}, str(src))
    save_file({"w": torch.arange(16, dtype=torch.float32).reshape(4, 4)}, str(trn))
    return src, trn, {"source_sha256": file_hash(src), "trained_sha256": file_hash(trn),
                      "trained_names": ["w"], "max_tensor_elements": 32, "arm": "rank", "rank": 2}


def test_checkpoint_construction_create_only(tmp_path):
    src, trn, contract = checkpoint_inputs(tmp_path)
    before = file_hash(src)
    report = delta.build_checkpoint(src, trn, tmp_path / "out", contract, lambda: None)
    assert report["model_calls"] == report["optimizer_steps"] == 0
    assert report["factor_file_bytes"] > 0 and report["native_qualified"] is False
    assert file_hash(src) == before
    with pytest.raises(ContractError):
        delta.build_checkpoint(src, trn, tmp_path / "out", contract, lambda: None)


@pytest.mark.parametrize("field,value", [("source_sha256", H), ("trained_names", ["other"]), ("max_tensor_elements", 4)])
def test_checkpoint_rejects_bad_contract(tmp_path, field, value):
    src, trn, contract = checkpoint_inputs(tmp_path)
    contract[field] = value
    with pytest.raises(ContractError):
        delta.build_checkpoint(src, trn, tmp_path / "out", contract, lambda: None)
    assert not (tmp_path / "out").exists()


def test_public_data_excludes_gold():
    row = world()
    row["reference"] = "SECRET_LABEL"
    assert "SECRET_LABEL" not in json.dumps(data.PublicCase.from_row(row).messages())
    with pytest.raises(ContractError):
        reasoning.run_case(row, "O", Backend(["FINAL: 1"]), budget(), lambda x: None)


def test_native_tokens_and_overflow():
    row = world()
    result = data.render_tokens(Tokenizer(), row["prompt"], row["reference"])
    assert result["rendered_total_tokens"] > result["rendered_prompt_tokens"]
    with pytest.raises(ContractError):
        data.render_tokens(Tokenizer(), row["prompt"], "x" * 3000)
    with pytest.raises(ContractError):
        data.render_tokens(Tokenizer(), row["prompt"], "1", reserve=-1)


def test_prefix_failure():
    class Bad(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            value = super().apply_chat_template(messages, **kwargs)
            if not kwargs["add_generation_prompt"]:
                value[0] = 99
            return value
    with pytest.raises(ContractError):
        data.render_tokens(Bad(), world()["prompt"], "1")


def test_protected_split_rejected_before_oracle():
    row = world(split="CONFIRMATION")
    def forbidden(*args):
        raise AssertionError("should not execute")
    with pytest.raises(ContractError):
        data.audit_rows([row], Tokenizer(), forbidden, tokenizer_identity={})


def test_structural_and_digit_overlap():
    a, b = world(1), world(2, "CAL")
    b["prompt"][1]["content"] = a["prompt"][1]["content"].replace("1", "9")
    report = data.population_overlap([a, b], None)
    assert report["binding_overlap"] and report["blocked_external_coverage"]
    assert any(r["axis"] == "text_digit_blinded" for r in report["cross_split"])


def test_incomplete_quarantine_registry_rejected():
    with pytest.raises(ContractError):
        data.population_overlap([world()], {"schema": "SCP_FINGERPRINT_REGISTRY_V1",
            "authorized_metadata_only": True, "coverage": [{"manifest_sha256": H}], "axes": {}})


def test_fixture_tokenizer_never_accepts_training():
    rows = [world(i, s, i % 2 == 0) for s in ("TRAIN", "CAL", "CHECK") for i in range(4)]
    result = data.audit_rows(rows, Tokenizer(), solver, tokenizer_identity={"kind": "fixture"})
    assert result["training_ready"] is False
    assert result["candidate_ready_for_review"] is False
    assert result["model_calls"] == 0


def test_shortcut_leak_detected_on_heldout():
    train = [world(i, "TRAIN", i % 2 == 0) for i in range(10)]
    check = [world(i, "CAL", i % 2 == 0) for i in range(10)]
    for row in train + check:
        row["question"] = "archival forbidden missing" if row["reference"] == "UNKNOWN" else "current answer"
    report = data.shortcut_audit(train, check)
    assert report["question_only_balanced_accuracy"] == 1


def acceptance_for(rows):
    return {"schema": "SCP_DATA_AUDIT_V1", "candidate_ready_for_review": True,
            "row_bindings": [{"id": r["id"], "world_id": r["world_id"], "split": r["split"],
                              "public_sha256": data.PublicCase.from_row(r).identity(),
                              "reference_sha256": digest(r["reference"])} for r in rows]}


def teacher_record(row):
    return {"public_sha256": data.PublicCase.from_row(row).identity(), "teacher_checkpoint_sha256": H,
            "answer": row["reference"], "steps": ["A verified alternative operation."], "terminated": True}


@pytest.mark.parametrize("field,value", [("answer", "wrong"), ("public_sha256", H),
                                        ("teacher_checkpoint_sha256", "b" * 64), ("terminated", False), ("steps", [])])
def test_teacher_rejects_bad_record(field, value):
    row = world()
    record = teacher_record(row)
    record[field] = value
    with pytest.raises(ContractError):
        data.verify_teacher(row, record, H, lambda *args: True)


def test_teacher_steps_must_verify():
    row = world()
    with pytest.raises(ContractError):
        data.verify_teacher(row, teacher_record(row), H, lambda *args: False)


def test_matched_fpt_and_shared_preservation():
    rows = [world(i) for i in range(2)]
    anchors = [world(30)]
    views = data.matched_views(rows, solver, arms=("F", "P", "T"),
        teacher_records={r["id"]: teacher_record(r) for r in rows}, teacher_sha256=H,
        step_checker=lambda *args: True, preservation_rows=anchors)
    assert [r["id"] for r in views["F"]] == [r["id"] for r in views["T"]]
    assert views["F"][-1] == views["P"][-1] == views["T"][-1]
    assert views["F"][0]["reference"] != views["P"][0]["reference"]


def test_missing_teacher_is_not_population_filter():
    with pytest.raises(ContractError):
        data.matched_views([world()], solver, arms=("F", "P", "T"), teacher_records={}, teacher_sha256=H)


def test_prepare_refuses_nontrain():
    with pytest.raises(ContractError):
        data.matched_views([world(split="CHECK")], solver)


def test_handoff_native_mask_contract_and_costs(tmp_path):
    views = data.matched_views([world()], solver)
    result = training.compile_handoff(views, Tokenizer(), tmp_path / "handoff",
        acceptance=acceptance_for([world()]),
        recipe={"student_sha256": H, "student_dtype": "float32", "seed": 17, "epochs": 2,
                "comparison": "MATCHED_WORLDS_AND_PRESENTATIONS"}, compiler=compiler, fixture=True)
    assert result["arms"]["P"]["processed_tokens"] > result["arms"]["F"]["processed_tokens"]
    assert result["arms"]["F"]["presentations"] == 2
    assert result["equal_compute_claim"] is False and result["native_execution_permitted"] is False


@pytest.mark.parametrize("text", ["1", "FINAL: 1\nFINAL: 2", "FINAL:", "FINAL: 1\nextra"])
def test_strict_parser(text):
    assert analysis.final_answer(text) is None


def test_hidden_damage_not_laundered():
    result = analysis.paired(scored([True, False, False]), scored([False, True, True]))
    assert result["net_gain"] == 1 and result["damage"] == 1 and result["repairs"] == 2
    assert result["preservation_established"] is False and result["no_observed_damage"] is False


@pytest.mark.parametrize("bad", [[], scored([True, False]), [*scored([True]), *scored([True])]])
def test_paired_requires_exact_unique_population(bad):
    with pytest.raises(ContractError):
        analysis.paired(scored([True]), bad)


def test_world_clustering():
    a, b = scored([True, False], shared=True), scored([False, True], shared=True)
    assert analysis.paired(a, b)["exact_mcnemar_two_sided_p"] is None
    assert analysis.cluster_interval(a, b)["status"] == "INCONCLUSIVE"


def test_factorial_all_patterns():
    values = [[bool((i >> bit) & 1) for i in range(16)] for bit in (3, 2, 1, 0)]
    report = analysis.factorial(*(scored(v) for v in values))
    assert len(report["patterns"]) == 16
    assert report["effects"]["interaction"] == 0 and report["synergy_established"] is False
    for row in report["records"]:
        assert row["interaction"] == (row["D"] - row["C"]) - (row["B"] - row["A"])


def test_selector_oracle_analysis():
    report = analysis.complementarity([[True, False, False, False], [False, True, False, False]], [True, False])
    assert report["selector_utilization"] == .5 and report["correct_candidate_but_wrong_selection"] == 1
    assert report["oracle_is_deployable"] is False
    with pytest.raises(ContractError):
        analysis.complementarity([[False] * 4], [True])


@pytest.mark.parametrize("arm,calls,tokens", [("O", 1, 64), ("T", 1, 288), ("REP", 5, 288), ("MR", 5, 288)])
def test_reasoning_budget_and_label_isolation(arm, calls, tokens):
    case = data.PublicCase.from_row(world())
    backend = Backend(["FINAL: 1"] * 4 + ['{"selected":0}'])
    events, b = [], budget(calls, tokens)
    result = reasoning.run_case(case, arm, backend, b, events.append)
    assert result["calls"] == calls and b.reserved_tokens == tokens
    assert result["classification"] == "SOFTWARE_FIXTURE_ONLY"
    if calls == 5:
        assert all("CANDIDATE_DATA_JSON" not in p[-1]["content"] for p in backend.prompts[:4])
        assert "CANDIDATE_DATA_JSON" in backend.prompts[-1][-1]["content"]
    assert any(e["event"] == "RESERVED" for e in events)


@pytest.mark.parametrize("choice", ['{"selected":true}', '{"selected":8}', '{"selected":0,"answer":"1"}', 'FINAL: 1'])
def test_bad_selector_visible(choice):
    result = reasoning.run_case(data.PublicCase.from_row(world()), "MR",
        Backend(["FINAL: 1"] * 4 + [choice]), budget(), lambda x: None)
    assert result["malformed"] is True and result["terminated"] is False


def test_native_flag_does_not_forge_qualification():
    with pytest.raises(ContractError):
        reasoning.run_case(data.PublicCase.from_row(world()), "O", Backend([]), budget(), lambda x: None, native=True)


def test_budget_fails_before_next_call():
    backend = Backend(["FINAL: 1"] * 4)
    b = budget(1, 64)
    with pytest.raises(ContractError):
        reasoning.run_case(data.PublicCase.from_row(world()), "MR", backend, b, lambda x: None)
    assert len(backend.prompts) == 1 and b.calls == 1


def test_failed_call_still_charged():
    b, events = budget(1, 64), []
    with pytest.raises(StopIteration):
        reasoning.run_case(data.PublicCase.from_row(world()), "O", Backend([]), b, events.append)
    assert b.calls == 1 and events[-1]["event"] == "FAILED"


def test_receipt_staleness_and_artifact_integrity(tmp_path):
    artifact = tmp_path / "checks.json"
    write_new(artifact, {"fixture_checks": True})
    binding = {"experiment": "SCP-001", "test": True}
    r = workflow.receipt("mechanics", "PASS", binding, {"checks": artifact})
    assert workflow.validate_receipt(r, "mechanics", binding)
    with pytest.raises(ContractError):
        workflow.validate_receipt(r, "mechanics", {"changed": True})
    artifact.write_text("tampered")
    with pytest.raises(ContractError):
        workflow.validate_receipt(r, "mechanics", binding)


def test_fixture_receipt_cannot_open_native_stage(tmp_path):
    artifact = tmp_path / "fixture.json"
    write_new(artifact, {})
    r = workflow.receipt("student_qualification", "PASS", {"x": 1}, {"fixture": artifact})
    with pytest.raises(ContractError):
        workflow.validate_receipt(r, "student_qualification", {"x": 1})


def test_status_does_not_claim_execution():
    report = workflow.status()
    assert report["native_launch_available"] is False
    assert all(r["status"] == "BLOCKED" for r in report["stages"].values())


def test_evidence_cannot_dirty_checkout(tmp_path):
    with pytest.raises(ContractError):
        workflow.outside_checkout(tmp_path, tmp_path / "evidence")
    assert workflow.outside_checkout(tmp_path / "repo", tmp_path / "records") == tmp_path / "records"


@pytest.mark.parametrize("value", [None, {}, {k: True for k in workflow.BUDGET_FIELDS}])
def test_missing_or_boolean_budget_rejected(value):
    with pytest.raises(ContractError):
        workflow.validate_budget(value)


def test_cli_status():
    result = subprocess.run([sys.executable, "-m", "experiments.sequential_complementary_v1", "status"],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["experiment"] == "SCP-001"


@pytest.mark.parametrize("rank", [0, -1, True, "16"])
def test_bad_transform_rank_even_for_vector(rank):
    with pytest.raises(ContractError):
        delta.transform(torch.ones(4), "rank", rank=rank)


def test_factor_artifact_reconstructs_each_update(tmp_path):
    from safetensors.torch import load_file
    src, trn, contract = checkpoint_inputs(tmp_path)
    out = tmp_path / "out"
    delta.build_checkpoint(src, trn, out, contract, lambda: None)
    index = read_json(out / "delta_factors" / "INDEX.json")
    entry = index["tensors"]["w"]
    path = out / "delta_factors" / entry["file"]
    assert file_hash(path) == entry["sha256"]
    factors = load_file(str(path))
    reconstructed = (factors["left"].double() @ factors["right"].double()).float()
    weight_index = read_json(out / "model.safetensors.index.json")["weight_map"]
    generated = load_file(str(out / weight_index["w"]))["w"]
    torch.testing.assert_close(reconstructed, generated, atol=0, rtol=0)


def test_score_records_refuses_missing_predictions():
    labels = [{"id": "a", "world_id": "a", "family": "state", "reference": "1", "output_contract": "FINAL_LINE_V1"}]
    with pytest.raises(ContractError):
        analysis.score_records(["a"], labels, [])
    rows = analysis.score_records(["a"], labels, [{"id": "a", "text": "FINAL: 1", "terminated": False}])
    assert rows[0]["correct"] is False


def test_freeze_rejects_missing_and_wrong_precision(tmp_path):
    with pytest.raises(ContractError):
        workflow.freeze_binding({"experiment": "SCP-001"}, tmp_path / "out.json")
    base = {"experiment": "SCP-001", "population_role": "OPEN_DEV", "protocol_sha256": file_hash(
        Path(workflow.__file__).with_name("protocol.json")), "dataset_sha256": H,
        "tokenizer_sha256": H, "student_sha256": H, "student_dtype": "bfloat16"}
    with pytest.raises(ContractError):
        workflow.freeze_binding(base, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_mechanics_stage_pass_does_not_open_other_stages(tmp_path):
    artifact = tmp_path / "test.json"
    write_new(artifact, {"real_scoped_test_count": 1})
    binding = {"experiment": "SCP-001", "scope": "fixture"}
    report = workflow.status(binding, {"mechanics": workflow.receipt("mechanics", "PASS", binding, {"tests": artifact})})
    assert report["stages"]["mechanics"]["status"] == "PASS"
    assert report["stages"]["student_training"]["status"] == "BLOCKED"
    assert report["stages"]["compression"]["status"] == "BLOCKED"


def test_handoff_rejects_changed_arm_population(tmp_path):
    views = data.matched_views([world(), world(1)], solver)
    views["P"].reverse()
    with pytest.raises(ContractError):
        training.compile_handoff(views, Tokenizer(), tmp_path / "out",
            acceptance=acceptance_for([world(), world(1)]),
            recipe={"student_sha256": H, "student_dtype": "float32", "seed": 17, "epochs": 1,
                    "comparison": "MATCHED_WORLDS_AND_PRESENTATIONS"}, compiler=compiler, fixture=True)
    assert not (tmp_path / "out").exists()


def test_handoff_rejects_unaccepted_rows(tmp_path):
    views = data.matched_views([world()], solver)
    with pytest.raises(ContractError):
        training.compile_handoff(views, Tokenizer(), tmp_path / "out", acceptance=acceptance_for([world(7)]),
            recipe={"student_sha256": H, "student_dtype": "float32", "seed": 17, "epochs": 1,
                    "comparison": "MATCHED_WORLDS_AND_PRESENTATIONS"}, compiler=compiler, fixture=True)
    assert not (tmp_path / "out").exists()


def test_public_case_constructor_rejects_hidden_objects():
    with pytest.raises(ContractError):
        data.PublicCase("i", "w", "system", {"reference": "hidden"})
