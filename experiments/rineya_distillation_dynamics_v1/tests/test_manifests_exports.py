"""Offline provenance/serialization fixtures: no models, forwards or optimizer steps."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.rineya_distillation_dynamics_v1.backend import Meter, TorchBackend, tensor_hash
from experiments.rineya_distillation_dynamics_v1.contracts import ContractError, digest, file_hash, read_json, sealed, unseal, write_new
from experiments.rineya_distillation_dynamics_v1.fixtures import ByteTokenizer, fixture_plan, fixture_tasks
from experiments.rineya_distillation_dynamics_v1.integration import (
    prepare_gd_consumed_provenance, prepare_gd_public_source_inventory,
    measure_loaded_peft_parity, verify_runtime_parity, write_standard_peft_state)
from experiments.rineya_distillation_dynamics_v1.manifests import validate_manifest
from experiments.rineya_distillation_dynamics_v1.metrics import paired


def recorder():
    # Construct only the provenance recorder; no nn.Module or model constructor.
    backend = TorchBackend.__new__(TorchBackend)
    backend.tokenizer = ByteTokenizer()
    backend.evidence_kind = "mechanical_fixture"
    backend.prompt_catalog, backend.prompt_events, backend.prompt_history, backend.prompt_renderings = {}, [], [], {}
    backend.prompt_identity, backend.tokenizer_content_sha256 = None, None
    backend.version, backend.meter = 0, Meter()
    backend.initial, backend.restore = {}, lambda _: None
    backend.bind_prompt_catalog(fixture_tasks(), "fixture-identity", tokenizer_content_sha256="a" * 64)
    return backend


def record_forward(backend, task, purpose, *, gradient=False, role="student"):
    ids = ByteTokenizer().apply_chat_template([{"role": "user", "content": task.prompt}])
    backend.meter.add(role + "_forward_input_tokens", len(ids))
    event = backend.record_prompt_consumption(task.public(), role=role, purpose=purpose, kind="forward",
        input_tokens=len(ids), rendered_prompt_ids=ids, forward_input_ids=ids)
    event.update({"gradient": gradient, "response_tokens": 2, "response_token_sha256": digest([49, 0])})
    return event


def test_complete_manifest_keeps_qualification_teacher_bank_and_replay(tmp_path):
    backend, tasks = recorder(), fixture_tasks()
    screen = next(t for t in tasks if t.role == "teacher_dev")
    backend.meter.phase = "qualification"
    record_forward(backend, screen, "teacher_competence_screen", role="teacher")
    backend.reset()
    backend.meter.phase = "training"
    for role, purpose in (("train", "current_student_rollout"), ("train", "forward_teacher_feedback"),
                          ("retention_train", "independent_retention_replay")):
        record_forward(backend, next(t for t in tasks if t.role == role), purpose,
                       gradient=purpose == "independent_retention_replay")
    ref = backend.write_prompt_manifest(tmp_path / "CONSUMED_PROMPTS.json", complete=True)
    manifest = unseal(read_json(ref["path"]))
    assert manifest["complete"] and manifest["event_count"] == 4 and len(manifest["sessions"]) == 2
    assert manifest["sessions"][0]["relationship"] == "prior_qualification_or_setup"
    for event in [e for s in manifest["sessions"] for e in s["events"]]:
        assert event["parent_id"] and event["revision"] and event["namespace"]
        assert event["prompt_token_sha256"] == digest(event["input_ids"])
        assert event["truncation"]["policy"] == "none"
        assert "solution" not in event and "answer" not in event


def test_manifest_detects_missing_forward_event_and_unbound_prompt(tmp_path):
    backend = recorder()
    backend.meter.add("student_forward_input_tokens", 3)
    with pytest.raises(ContractError, match="missing model consumption"):
        backend.write_prompt_manifest(tmp_path / "missing.json", complete=True)
    backend = recorder()
    task = replace(fixture_tasks()[0], key="unbound:case")
    record_forward(backend, task, "unknown")
    with pytest.raises(ContractError, match="partial/missing"):
        backend.write_prompt_manifest(tmp_path / "partial.json", complete=True)


def test_manifest_normalized_public_hash_and_catalog_drift():
    backend = recorder()
    task = fixture_tasks()[0]
    event = record_forward(backend, task, "public")
    assert event["prompt_hash"] != file_hash(__file__)
    with pytest.raises(ContractError, match="catalog drift"):
        backend.bind_prompt_catalog([replace(task, revision="rewritten")])
    payload = backend.prompt_manifest()
    payload["sessions"][0]["events"][0]["prompt_hash"] = "b" * 64
    payload["sessions"][0]["events_sha256"] = digest(payload["sessions"][0]["events"])
    with pytest.raises(ContractError, match="prompt hash drift"):
        validate_manifest(payload)


@pytest.mark.parametrize("kind", ["forward", "external_forward"])
@pytest.mark.parametrize("missing", ["forward_input_ids", "causal_input_token_sha256"])
def test_complete_manifest_rejects_missing_causal_inputs(kind, missing):
    backend, task = recorder(), fixture_tasks()[0]
    ids = ByteTokenizer().apply_chat_template([{"role": "user", "content": task.prompt}])
    if kind == "forward":
        backend.meter.add("student_forward_input_tokens", len(ids))
    event = backend.record_prompt_consumption(task.public(), role="student", purpose="causal_control",
        kind=kind, input_tokens=len(ids), rendered_prompt_ids=ids, forward_input_ids=ids)
    payload = backend.prompt_manifest()
    del payload["sessions"][0]["events"][0][missing]
    payload["sessions"][0]["events_sha256"] = digest(payload["sessions"][0]["events"])
    with pytest.raises(ContractError, match="complete forward requires actual causal"):
        validate_manifest(payload)


def test_manifest_preserves_and_checks_generated_causal_suffix():
    backend, task = recorder(), fixture_tasks()[0]
    ids = ByteTokenizer().apply_chat_template([{"role": "user", "content": task.prompt}])
    full = list(ids) + [49, 50]
    backend.meter.add("student_forward_input_tokens", len(full))
    event = backend.record_prompt_consumption(task.public(), role="student", purpose="generated_prefix",
        kind="forward", input_tokens=len(full), rendered_prompt_ids=ids, forward_input_ids=full)
    assert event["input_ids"] == list(ids) and event["forward_input_ids"] == full
    assert event["causal_input_token_sha256"] == digest(full)
    payload = backend.prompt_manifest()
    payload["sessions"][0]["events"][0]["forward_input_ids"][-1] = 51
    payload["sessions"][0]["events_sha256"] = digest(payload["sessions"][0]["events"])
    with pytest.raises(ContractError, match="actual causal input token receipt drift"):
        validate_manifest(payload)


def test_gd_projection_repeats_source_complete_and_unused(tmp_path):
    backend, tasks = recorder(), fixture_tasks()
    backend.meter.phase = "training"
    train = next(t for t in tasks if t.role == "train")
    replay = next(t for t in tasks if t.role == "retention_train")
    for step in range(2):
        backend.version = step
        record_forward(backend, train, "verified_solution_sft", gradient=True)
        record_forward(backend, replay, "independent_retention_replay", gradient=True)
    manifest = backend.write_prompt_manifest(tmp_path / "full.json", complete=True)
    source = prepare_gd_public_source_inventory(tasks, tmp_path / "sources.jsonl")
    result = prepare_gd_consumed_provenance(consumed_manifest=manifest, source_inventory=source,
        tokenizer_sha256="a" * 64, final_step=2, output=tmp_path / "gd")
    assert result["source_count"] == 4 and result["unused_accepted_count"] == 2
    assert result["consumed_count"] == 4 and result["consumed_per_step"] == {"1": 2, "2": 2}
    import json
    rows = [json.loads(line) for line in Path(result["provenance"]["events"]["path"]).read_text().splitlines()]
    assert [r["step"] for r in rows if r["event_type"] == "consume"] == [1, 1, 2, 2]
    manifest_rows = [json.loads(line) for line in Path(result["provenance"]["training_manifest"]["path"]).read_text().splitlines()]
    assert len(manifest_rows) == 4 and manifest_rows[0] == manifest_rows[2]
    assert all("answer" not in r["raw_record"] and "solution" not in r["raw_record"]
               for r in rows if r["event_type"] == "source")


def test_optional_read_only_gd_owner_provenance_contract(tmp_path):
    import importlib
    import importlib.util
    import json
    import sys
    gd = Path(__file__).resolve().parents[4] / "ChatBubble-rineya-generalization-dynamics-v1" / "experiments" / "rineya_generalization_dynamics_v1"
    if not (gd / "attachment.py").is_file():
        pytest.skip("GD owner's optional sibling worktree contract unavailable")
    module_name = "rdd_fixture_readonly_gd_contract"
    spec = importlib.util.spec_from_file_location(module_name, gd / "__init__.py", submodule_search_locations=[str(gd)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = package
    spec.loader.exec_module(package)
    try:
        attachment = importlib.import_module(module_name + ".attachment")
        backend, tasks = recorder(), fixture_tasks()
        backend.meter.phase = "training"
        for step in range(2):
            backend.version = step
            for role in ("train", "retention_train"):
                record_forward(backend, next(t for t in tasks if t.role == role), "fixture_gradient_input", gradient=True)
        manifest = backend.write_prompt_manifest(tmp_path / "full.json", complete=True)
        source = prepare_gd_public_source_inventory(tasks, tmp_path / "source.jsonl")
        result = prepare_gd_consumed_provenance(consumed_manifest=manifest, source_inventory=source,
            tokenizer_sha256="a" * 64, final_step=2, output=tmp_path / "projection")
        rows = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines()]
        completion = {"final_step": 2, **{k: v for k, v in result.items() if k != "provenance"}}
        validated = attachment._provenance(rows(result["provenance"]["events"]["path"]),
            rows(result["provenance"]["training_manifest"]["path"]), "a" * 64, completion)
        assert validated["consumed_count"] == 4 and validated["unused_accepted_count"] == 2
        assert validated["consumed_per_step"] == {"1": 2, "2": 2}
    finally:
        for name in list(sys.modules):
            if name == module_name or name.startswith(module_name + "."):
                del sys.modules[name]


def fixture_state():
    return {"model.layers.0.self_attn.q_proj.lora_A": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "model.layers.0.self_attn.q_proj.lora_B": torch.zeros(4, 2)}


def test_peft_serialization_bitwise_is_not_native_parity(tmp_path):
    plan, state = fixture_plan(), fixture_state()
    base = {"path": str((tmp_path / "fixture-base").resolve()), "files": {"config.json": "f" * 64},
            "sha256": digest({"config.json": "f" * 64})}
    receipt = {"tensor_sha256": tensor_hash(state), "rank": plan.rank, "alpha": plan.alpha,
               "step": 0, "evidence_kind": "mechanical_fixture"}
    result = write_standard_peft_state(state, plan, tmp_path / "peft", base, identity="fixture",
        source_receipt=receipt, artifact_role="mechanical_fixture")
    assert result["serialization_bitwise_verified"] and not result["runtime_parity_qualified"]
    assert result["requires_actual_native_peft_parity"] and result["artifact_role"] == "mechanical_fixture"
    from safetensors.torch import load_file
    loaded = load_file(str(tmp_path / "peft" / "adapter_model.safetensors"))
    assert torch.equal(loaded["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
                       state["model.layers.0.self_attn.q_proj.lora_A"])
    write_new(tmp_path / "fake-parity.json", sealed({"status": "METADATA_PARITY", "evidence_kind": "mechanical_fixture"}))
    with pytest.raises(ContractError, match="actual native"):
        verify_runtime_parity(tmp_path / "peft", tmp_path / "fake-parity.json")


def test_peft_refuses_missing_factor_and_wrong_tensor_hash(tmp_path):
    plan, state = fixture_plan(), fixture_state()
    base = {"path": str(tmp_path.resolve()), "files": {"config.json": "f" * 64},
            "sha256": digest({"config.json": "f" * 64})}
    receipt = {"tensor_sha256": "a" * 64, "rank": plan.rank, "alpha": plan.alpha, "evidence_kind": "mechanical_fixture"}
    with pytest.raises(ContractError, match="tensor/configuration drift"):
        write_standard_peft_state(state, plan, tmp_path / "wrong", base, identity="fixture",
            source_receipt=receipt, artifact_role="mechanical_fixture")
    state.pop(next(n for n in state if n.endswith(".lora_B")))
    with pytest.raises(ContractError, match="both LoRA factors"):
        write_standard_peft_state(state, plan, tmp_path / "missing", base, identity="fixture",
            source_receipt=receipt, artifact_role="mechanical_fixture")


def test_native_peft_parity_rejects_fixture_before_loading_or_file_reads(tmp_path):
    backend = recorder()
    with pytest.raises(ContractError, match="explicit supervised native parity"):
        measure_loaded_peft_parity(backend, object(), object(), export_directory=tmp_path / "not-read",
            policy_path=tmp_path / "not-read-policy", native_load_receipt=tmp_path / "not-read-load",
            output=tmp_path / "not-created", reviewed_by="fixture", allow_native_forward=False)
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("changed", ["model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                    "added_tokens.json"])
def test_runtime_parity_rechecks_entire_base_snapshot(tmp_path, changed):
    """Tiny fabricated files exercise refusal only; no native receipt is qualified."""
    from experiments.rineya_distillation_dynamics_v1.registry import inventory
    base_root, export_root = tmp_path / "base", tmp_path / "export"
    base_root.mkdir()
    export_root.mkdir()
    write_new(base_root / "config.json", {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (base_root / name).write_bytes(b"fabricated-layout-only-not-native")
    base = inventory(base_root.resolve())
    (export_root / "adapter_model.safetensors").write_bytes(b"fabricated-export-only")
    files = {"adapter_model.safetensors": file_hash(export_root / "adapter_model.safetensors")}
    export = {"evidence_kind": "qualified_native_hf", "identity": "negative-fixture",
              "base_snapshot": base, "export_files": files}
    write_new(export_root / "RDD_EXPORT.json", sealed(export))
    parity_path = tmp_path / "metadata-negative-control.json"
    write_new(parity_path, sealed({"status": "NATIVE_WRAPPER_HF_PEFT_PARITY_MEASURED",
        "evidence_kind": "qualified_native_hf", "actual_native_model_execution": True,
        "identity": export["identity"], "export_sha256": file_hash(export_root / "RDD_EXPORT.json"),
        "base_snapshot_sha256": base["sha256"], "export_files": files}))
    # With matching files, this intentionally incomplete fixture fails the next
    # authority check. It can never qualify actual native parity.
    with pytest.raises(ContractError, match="prospective parity policy"):
        verify_runtime_parity(export_root, parity_path)
    (base_root / changed).write_bytes(b"changed-or-added-after-parity")
    with pytest.raises(ContractError, match="base/tokenizer snapshot drift"):
        verify_runtime_parity(export_root, parity_path)


def test_source_correct_damage_repair_and_undefined_retention():
    baseline, current = {}, {}
    for key, source, before, after in (("damaged", "old", True, False), ("retained", "old", True, True),
                                       ("rescued", "old", False, True), ("cold", "zero", False, True)):
        meta = {"parent": key, "source": source, "role": "dev_retention"}
        baseline[key] = {**meta, "correct": before}
        current[key] = {**meta, "correct": after, "selected_correct": after, "samples": [after], "generated_tokens": 1}
    result = paired(baseline, current, bootstrap=0)["groups"]
    old, zero = result["dev_retention/old"], result["dev_retention/zero"]
    assert (old["base_correct"], old["retained"], old["damaged"], old["rescued"]) == (2, 1, 1, 1)
    assert old["reference_retention"] == .5
    assert zero["base_correct"] == 0 and zero["reference_retention"] is None and zero["rescued"] == 1
