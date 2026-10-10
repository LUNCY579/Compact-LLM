"""CPU fixtures only; these tests provide no Rineya capability evidence."""
import copy
import pytest
import torch

from experiments.rineya_self_distillation_axes_v1.mechanical_qualification import (
    capture_raw_source_parity, capture_resource_stage, capture_unprivileged_parity, check_loaded_source_parity,
    maximum_shape_backward,
    prepare_parity_probes, qualify_teacher_competence, recheck_source_custody,
    snapshot_source_custody, validate_native_evidence,
)
from experiments.rineya_self_distillation_axes_v1.native import fixed_precision
from experiments.rineya_self_distillation_axes_v1.objectives import token_kl
from experiments.rineya_self_distillation_axes_v1.plan import Arm, arms, digest_json
from experiments.rineya_self_distillation_axes_v1.teacher import tensor_digest
from experiments.rineya_self_distillation_axes_v1.engine import Engine
from experiments.rineya_self_distillation_axes_v1.tests.test_engine import make_engine, TinyTokenizer


def gate_rows():
    return [{"id": f"gate-{i}", "parent_id": f"gate-family-{i}", "task": "new",
             "split": "train", "purpose": "acquisition", "admitted_for": "train",
             "source": "fresh CPU fixture only", "prompt": f"Fixture gate question {i}?",
             "reference": "OK", "answer": "OK", "scorer": "exact"} for i in range(8)]


def scripted_gate(engine, monkeypatch, teacher_successes, student_successes):
    rows = gate_rows()
    inputs = []
    seeds = {int(digest_json([engine.config["seed"], row["id"], "teacher-gate"])[:8], 16): i
             for i, row in enumerate(rows)}
    teacher_prefixes = {tuple(engine.ids(row, True)) for row in rows}
    def generate(prefix, seed, greedy=False):
        assert greedy
        inputs.append(prefix)
        successes = teacher_successes if tuple(prefix) in teacher_prefixes else student_successes
        return [3 if seeds[seed] in successes else 4], "eos"
    monkeypatch.setattr(engine, "generate", generate)
    monkeypatch.setattr(engine.tokenizer, "decode", lambda tokens, **kwargs: "OK" if tokens == [3] else "BAD")
    return rows, inputs


def test_parity_exact_full_logits_max_shape_and_no_optimizer(monkeypatch):
    engine = make_engine()
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("Optimizer forbidden"))
    before = tensor_digest(engine.model.named_parameters())
    probes = prepare_parity_probes(engine, gate_rows())
    assert len(probes[-1]["input_ids"]) == engine.config["max_sequence_tokens"]
    assert all(p["privileged"] is False for p in probes)
    with fixed_precision():
        raw = capture_unprivileged_parity(engine, probes)
        result = check_loaded_source_parity(engine, probes, raw)
    assert result["source_inference_parity"] and result["equality"] == "exact_full_logits"
    assert result["training_updates"] == 0 and result["optimizer_objects_created"] == 0
    assert before == tensor_digest(engine.model.named_parameters())
    altered = copy.deepcopy(raw)
    altered["records"][0]["logits_sha256"] = "0" * 64
    with fixed_precision(), pytest.raises(ValueError, match="full logits changed"):
        check_loaded_source_parity(engine, probes, altered)


def test_actual_raw_bf16_source_matches_selected_fp32_cast_exactly():
    engine = make_engine()
    raw_model = copy.deepcopy(engine.model)
    raw_model.q_proj.weight.data = raw_model.q_proj.weight.data.bfloat16()
    raw_model.requires_grad_(False)
    cast_model = copy.deepcopy(raw_model)
    cast_model.q_proj.weight.data = cast_model.q_proj.weight.data.float()
    cast_model.q_proj.weight.requires_grad_(True)
    adapted = Engine(cast_model, TinyTokenizer(), engine.config)
    probes = prepare_parity_probes(adapted, gate_rows())
    before = tensor_digest(raw_model.named_parameters())
    with fixed_precision():
        raw = capture_raw_source_parity(raw_model, engine.config, probes)
        result = check_loaded_source_parity(adapted, probes, raw)
    assert raw["raw_parameters_sha256"] == before == tensor_digest(raw_model.named_parameters())
    assert result["source_inference_parity"] and result["max_shape_no_update"]


def test_maximum_shape_backward_has_no_optimizer_updates_and_clears_gradients(monkeypatch):
    engine = make_engine()
    engine.config["max_new_tokens"] = 32
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("Optimizer forbidden"))
    before = tensor_digest(engine.model.named_parameters())
    probes = prepare_parity_probes(engine, gate_rows())
    with fixed_precision():
        result = maximum_shape_backward(engine, probes)
    assert result["maximum_shape_tokens"] == 512 and result["response_tokens"] == 32
    assert result["max_shape_backward_no_update"] and result["gradients_cleared"]
    assert result["optimizer_objects_created"] == 0 and result["training_updates"] == 0
    assert before == tensor_digest(engine.model.named_parameters())
    assert all(p.grad is None for p in engine.model.parameters())


def test_student_input_and_prediction_invariant_to_reference_sentinel(monkeypatch):
    engine = make_engine()
    original = gate_rows()
    altered = copy.deepcopy(original)
    for row in altered:
        row["reference"] = "SECRET_REFERENCE_PAYLOAD"
        row["answer"] = "SECRET_REFERENCE_PAYLOAD"
    # Reference/answer never participate in clean prompt construction or parity.
    clean = prepare_parity_probes(engine, original)
    changed = prepare_parity_probes(engine, altered)
    assert clean == changed
    with fixed_precision():
        assert capture_unprivileged_parity(engine, clean) == capture_unprivileged_parity(engine, changed)
    messages = []
    original_template = engine.tokenizer.apply_chat_template
    def record_template(message, **kwargs):
        messages.extend(message)
        return original_template(message, **kwargs)
    monkeypatch.setattr(engine.tokenizer, "apply_chat_template", record_template)
    dev = dict(altered[0], split="dev", admitted_for="dev")
    with fixed_precision():
        engine.evaluate([dev])
    assert messages == [{"role": "user", "content": dev["prompt"]}]
    assert "SECRET_REFERENCE_PAYLOAD" not in messages[0]["content"]


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_off_sampled_token_changes_full_vocabulary_kl(direction):
    student = torch.tensor([[[0., 1., 2., 3.]]], requires_grad=True)
    teacher = torch.tensor([[[0., 2., 1., 3.]]], requires_grad=True)
    original = token_kl(student, teacher, direction)
    altered = teacher.detach().clone()
    altered[..., 0] += 4.  # Token 3 is the hypothetical sampled token; token 0 still matters.
    assert not torch.equal(original, token_kl(student, altered, direction))
    original.sum().backward()
    assert teacher.grad is None and student.grad is not None


@pytest.mark.parametrize("source", ["student", "teacher"])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_common_response_prefix_scored_at_each_causal_offset(source, direction, monkeypatch):
    engine = make_engine()
    row = gate_rows()[0]
    response, captures = [3, 5, engine.eos], []
    original_forward = engine.forward
    def forward(tokens):
        logits = original_forward(tokens)
        captures.append((list(tokens), logits.detach().clone(), logits.requires_grad))
        return logits
    monkeypatch.setattr(engine, "forward", forward)
    monkeypatch.setattr(engine, "generate", lambda *args, **kwargs: (list(response), "eos"))
    student_prefix, teacher_prefix = engine.ids(row, False), engine.ids(row, True)
    assert len(student_prefix) != len(teacher_prefix)
    with fixed_precision(), engine.objective(row, Arm("sd", source, direction, 0.), 0) as (loss, details):
        assert captures[0][0] == teacher_prefix + response
        assert captures[1][0] == student_prefix + response
        teacher_logits = captures[0][1][:, len(teacher_prefix) - 1:len(teacher_prefix) + len(response) - 1]
        student_logits = captures[1][1][:, len(student_prefix) - 1:len(student_prefix) + len(response) - 1]
        torch.testing.assert_close(loss, token_kl(student_logits, teacher_logits, direction).mean())
        assert captures[0][2] is False and captures[1][2] is True
        assert details["response_ids"] == response
        loss.backward()


def test_teacher_gate_threshold_paired_clean_control_and_restore(monkeypatch):
    engine = make_engine()
    rows, inputs = scripted_gate(engine, monkeypatch, set(range(7)), set(range(5)))
    with torch.no_grad():
        engine.model.q_proj.weight.add_(1.)  # Gate uses initial source, restoring caller state.
    before = tensor_digest(engine.model.named_parameters())
    result = qualify_teacher_competence(engine, rows)
    assert result["pass"] and result["teacher_correct"] == 7
    assert result["paired_wins"] == 2 and result["paired_harms"] == 0
    assert result["promotion_allowed"] is False and result["training_updates"] == 0
    assert before == tensor_digest(engine.model.named_parameters())
    assert inputs[::2] == [engine.ids(row, False) for row in rows]
    assert inputs[1::2] == [engine.ids(row, True) for row in rows]


@pytest.mark.parametrize("teacher,student", [
    (set(range(6)), set(range(4))),  # insufficient teacher accuracy
    (set(range(7)), set(range(6))),  # only one paired win
    (set(range(7)), {0, 1, 2, 3, 4, 7}),  # two wins, one paired harm
])
def test_teacher_gate_failures_are_not_redefined(teacher, student, monkeypatch):
    engine = make_engine()
    rows, _ = scripted_gate(engine, monkeypatch, teacher, student)
    assert not qualify_teacher_competence(engine, rows)["pass"]


@pytest.mark.parametrize("mutation", ["dev", "retention", "duplicate_parent", "adaptation_overlap", "noninitial_teacher"])
def test_teacher_gate_admission_is_independent_train_only(mutation):
    engine = make_engine()
    rows = gate_rows()
    adaptation = []
    if mutation == "dev": rows[0].update(split="dev", admitted_for="dev")
    if mutation == "retention": rows[0]["purpose"] = "retention"
    if mutation == "duplicate_parent": rows[0]["parent_id"] = rows[1]["parent_id"]
    if mutation == "adaptation_overlap": adaptation = [dict(rows[0], id="new-id")]
    if mutation == "noninitial_teacher": engine.bank.teacher["q_proj.weight"].add_(1.)
    with pytest.raises(ValueError):
        qualify_teacher_competence(engine, rows, adaptation)


def test_teacher_swap_and_buffer_restore_on_exception(monkeypatch):
    engine = make_engine()
    engine.model.register_buffer("fixture_mutable", torch.tensor([1.]))
    before = tensor_digest(engine.model.named_parameters())
    teacher_before = tensor_digest(engine.bank.teacher.items())
    def interrupted(prefix, seed, greedy=False):
        engine.model.q_proj.weight.add_(3.)
        engine.model.fixture_mutable.add_(2.)
        raise RuntimeError("fixture interruption")
    monkeypatch.setattr(engine, "generate", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        qualify_teacher_competence(engine, gate_rows())
    assert before == tensor_digest(engine.model.named_parameters())
    assert teacher_before == tensor_digest(engine.bank.teacher.items())
    assert torch.equal(engine.model.fixture_mutable, torch.tensor([1.]))
    assert not engine.bank._busy


def test_source_custody_recheck_detects_changed_weights_and_data(tmp_path):
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"fixture-only-not-native")
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("fixture-only")
    before = snapshot_source_custody(checkpoint, dataset)
    assert recheck_source_custody(before)["source_custody_unchanged"]
    weights.write_bytes(b"changed-fixture")
    with pytest.raises(RuntimeError, match="custody changed"):
        recheck_source_custody(before)
    before = snapshot_source_custody(checkpoint, dataset)
    dataset.write_text("changed-data")
    with pytest.raises(RuntimeError, match="custody changed"):
        recheck_source_custody(before)


def evidence():
    source = {"source_id": "fixture-ineligible-for-native-test-schema", "eligible": True,
              "checkpoint_sha256": "a" * 64}
    identity = {"checkpoint_sha256": "a" * 64, "runtime": {"device": "cpu"}}
    checks = {k: True for k in ("parameters_unchanged", "buffers_unchanged", "teacher_initial_identity",
                               "full_vocabulary", "teacher_detached", "common_response_prefix",
                               "source_inference_parity", "max_shape_no_update", "max_shape_backward_no_update",
                               "source_custody_unchanged")}
    report = {"source": source, "identity": identity, "initial_parameters_sha256": "b" * 64,
              "training_updates": 0, "optimizer_objects_created": 0, "checks": checks,
              "backend_events": ["aten::_scaled_dot_product_attention_math"],
              "diagnostics": [{"arm_id": a.name, "loss": .1, "gradient_l2": .2, "response_tokens": 3}
                              for a in arms((0.,))]}
    return report, identity, source, "b" * 64


def test_strict_native_evidence_schema_never_authorizes_campaign():
    result = validate_native_evidence(*evidence())
    assert result["mechanical_evidence_validated"]
    assert not result["native_campaign_authorized"] and result["resource_integration"] == "PENDING"


@pytest.mark.parametrize("mutation", ["loaded_hash", "source", "events", "diagnostics", "nonfinite", "update", "max_shape"])
def test_strict_evidence_rejects_incomplete_native_claims(mutation):
    report, identity, source, digest = evidence()
    if mutation == "loaded_hash": report["initial_parameters_sha256"] = "c" * 64
    if mutation == "source": report["source"] = dict(source, source_id="different")
    if mutation == "events": report["backend_events"] = ["math"]
    if mutation == "diagnostics": report["diagnostics"] = report["diagnostics"][:-1]
    if mutation == "nonfinite": report["diagnostics"][0]["gradient_l2"] = float("nan")
    if mutation == "update": report["training_updates"] = 1
    if mutation == "max_shape": report["checks"]["max_shape_no_update"] = False
    with pytest.raises(ValueError):
        validate_native_evidence(report, identity, source, digest)


def test_cpu_resource_helper_reports_pending_host_measurement():
    record = capture_resource_stage("cpu", "max_shape_after")
    assert record["host_peak_accounting"] == "PENDING" and not record["peak_stats_reset_by_helper"]
    with pytest.raises(ValueError):
        capture_resource_stage("cpu", "max_shape_after", min_free_cuda_gib=True)
