"""Approval/counter/receipt fixtures only: no native loader or Transformers calls."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.rineya_self_distillation_axes_v1 import reviewed_qualification as reviewed


def packet_and_review():
    packet = {"experiment": "DSD-001", "runtime_recipe": dict(reviewed.EXPECTED_RUNTIME),
              "data": {"bindings": {name: {"sha256": char * 64} for name, char in
                                     (("screen.jsonl", "a"), ("teacher-gate.jsonl", "b"))}}}
    review = {"experiment": "DSD-001", "phase": reviewed.PHASE, "status": "APPROVED",
              "packet_sha256": "c" * 64, "reviewer": "fixture reviewer",
              "data_admission": {name: {"status": "APPROVED", "reviewer": "fixture data reviewer",
                                        "sha256": binding["sha256"]}
                                 for name, binding in packet["data"]["bindings"].items()}}
    return packet, review


def test_opt_in_rejected_before_file_read_import_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(reviewed, "_native_services", lambda: pytest.fail("Native imports forbidden"))
    with pytest.raises(PermissionError, match="execute-qualification"):
        reviewed.execute(tmp_path / "missing-packet", tmp_path / "missing-review", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["experiment", "packet", "phase", "status", "reviewer", "screen", "gate", "runtime"])
def test_review_and_both_exact_data_admissions_are_required(mutation):
    packet, review = packet_and_review()
    if mutation == "experiment":
        review["experiment"] = "another-experiment"
    elif mutation == "packet":
        review["packet_sha256"] = "d" * 64
    elif mutation == "phase":
        review["phase"] = "comparison_training"
    elif mutation == "status":
        review["status"] = "PENDING"
    elif mutation == "reviewer":
        review["reviewer"] = " "
    elif mutation in {"screen", "gate"}:
        review["data_admission"]["screen.jsonl" if mutation == "screen" else "teacher-gate.jsonl"]["status"] = "PENDING"
    else:
        packet["runtime_recipe"]["torch"] = "another-version"
    with pytest.raises((PermissionError, ValueError)):
        reviewed.validate_review(packet, review, "c" * 64)


def limits():
    return {"qualification_wall_seconds": 10, "max_sequence_tokens": 512,
            "qualification_forward_calls": 2, "qualification_processed_input_tokens": 6,
            "qualification_generated_tokens": 1, "minimum_free_host_gib_before": 1,
            "minimum_free_host_gib_during": 1, "minimum_free_cuda_gib_before": 1,
            "minimum_free_cuda_gib_during": 1, "minimum_free_disk_gib": .000001}


def resources():
    return {"available_host_bytes": 8 * 1024**3, "process_peak_working_set_bytes": 1234,
            "cuda_free_bytes": 8 * 1024**3, "cuda_allocated_bytes": 1024,
            "cuda_reserved_bytes": 2048}


def test_forward_guards_before_after_and_charge_failed_attempts(tmp_path):
    samples = []
    def snapshot():
        samples.append(1)
        return resources()
    guard = reviewed.ResourceGuard(limits(), tmp_path, snapshot, clock=lambda: 1.)
    assert guard.forward(3, lambda: "fixture") == "fixture"
    assert len(samples) == 2
    with pytest.raises(RuntimeError, match="fixture failure"):
        guard.forward(3, lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    assert guard.counts["forward_calls"] == 2 and guard.counts["processed_input_tokens"] == 6
    with pytest.raises(RuntimeError, match="ceiling exceeded before forward"):
        guard.forward(1, lambda: pytest.fail("No work after ceiling"))
    assert len(samples) == 4


def test_generation_token_ceiling_is_reserved_before_last_forward(tmp_path):
    guard = reviewed.ResourceGuard(limits(), tmp_path, resources, clock=lambda: 1.)
    guard.in_generation = 1
    guard.forward(2, lambda: None)
    with pytest.raises(RuntimeError, match="generated_token_charges"):
        guard.forward(2, lambda: pytest.fail("Token ceiling must precede work"))
    assert guard.counts["generated_token_charges"] == 1


@pytest.mark.parametrize("violation", ["host", "cuda", "time"])
def test_resource_floors_and_time_stop_before_native_work(tmp_path, violation):
    record = resources()
    if violation == "host":
        record["available_host_bytes"] = 0
    if violation == "cuda":
        record["cuda_free_bytes"] = 0
    times = iter((0, 20)) if violation == "time" else iter((0, 0))
    guard = reviewed.ResourceGuard(limits(), tmp_path, lambda: record, clock=lambda: next(times))
    with pytest.raises(RuntimeError):
        guard.forward(1, lambda: pytest.fail("Resource violation must precede work"))


def test_caches_are_external_offline_and_environment_restored(tmp_path, monkeypatch):
    keys = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME",
            "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "TRITON_CACHE_DIR")
    for key in keys:
        monkeypatch.setenv(key, "previous-fixture-value")
    previous_tempdir = reviewed.tempfile.tempdir
    with reviewed._external_caches(tmp_path):
        assert all(Path(reviewed.os.environ[key]).is_relative_to(tmp_path) for key in keys)
        assert Path(reviewed.tempfile.gettempdir()).is_relative_to(tmp_path)
        assert reviewed.os.environ["HF_HUB_OFFLINE"] == "1"
        assert reviewed.os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert reviewed.sys.dont_write_bytecode is True
    assert all(reviewed.os.environ[key] == "previous-fixture-value" for key in keys)
    assert reviewed.tempfile.tempdir == previous_tempdir


def test_post_creation_validation_failure_preserves_receipt_and_owned_lease_only(tmp_path, monkeypatch):
    # Relocate the module's inferred workspace for this fixture; no real owner lock.
    fake_module = tmp_path / "BusinessProject" / "source" / "experiments" / "dsd" / "reviewed_qualification.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(reviewed, "__file__", str(fake_module))
    packet, approval = packet_and_review()
    packet["source"] = {"checkpoint_sha256": "e" * 64,
                        "checkpoint": str(tmp_path / "BusinessProject" / "immutable-fixture-source")}
    packet_path, review_path = tmp_path / "packet.json", tmp_path / "review.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    approval["packet_sha256"] = reviewed._hash_file(packet_path)
    review_path.write_text(json.dumps(approval), encoding="utf-8")
    def fail(_packet):
        raise RuntimeError("fixture preflight rejection")
    fixture = SimpleNamespace(fixture_only=True, runtime=lambda: dict(reviewed.EXPECTED_RUNTIME),
                              validate_packet=fail, collect=lambda: None,
                              custody=lambda p: {"checkpoint_sha256": "e" * 64,
                                  "data_sha256": {name: binding["sha256"] for name, binding in p["data"]["bindings"].items()}})
    output = tmp_path / "BusinessProject" / "fixture-output"
    with pytest.raises(RuntimeError, match="fixture preflight rejection"):
        reviewed.execute(packet_path, review_path, output, True, _services=fixture)
    failed = json.loads((output / "FAILED.json").read_text())
    assert failed["phase"] == reviewed.PHASE and failed["training_updates"] == 0
    assert failed["native_checkpoint_executed"] is False
    assert (output / "identity.json").is_file() and (output / "counters.json").is_file()
    assert not (tmp_path / "BusinessProject" / "dsd001-native-owner.lock").exists()
    before = (output / "FAILED.json").read_bytes()
    with pytest.raises(FileExistsError):
        reviewed.execute(packet_path, review_path, output, True, _services=fixture)
    assert (output / "FAILED.json").read_bytes() == before
    lock = tmp_path / "BusinessProject" / "dsd001-native-owner.lock"
    lock.write_text("foreign owner remains", encoding="utf-8")
    with pytest.raises(FileExistsError):
        reviewed.execute(packet_path, review_path, output.parent / "competing", True, _services=fixture)
    assert lock.read_text(encoding="utf-8") == "foreign owner remains"
    assert not (output.parent / "competing").exists()


@pytest.mark.parametrize("protected", ["checkpoint", "git_checkout"])
def test_protected_source_destinations_rejected_before_lease_output(tmp_path, monkeypatch, protected):
    fake_module = tmp_path / "BusinessProject" / "source" / "experiments" / "dsd" / "reviewed_qualification.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(reviewed, "__file__", str(fake_module))
    packet, approval = packet_and_review()
    checkpoint = tmp_path / "BusinessProject" / "immutable-source"
    packet["source"] = {"checkpoint": str(checkpoint)}
    packet_path, review_path = tmp_path / "packet.json", tmp_path / "review.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    approval["packet_sha256"] = reviewed._hash_file(packet_path)
    review_path.write_text(json.dumps(approval), encoding="utf-8")
    if protected == "checkpoint":
        output = checkpoint / "new-qualification"
    else:
        checkout = tmp_path / "BusinessProject" / "other-repository"
        checkout.mkdir()
        (checkout / ".git").write_text("gitdir: fixture", encoding="utf-8")
        output = checkout / "new-qualification"
    with pytest.raises(ValueError):
        reviewed.execute(packet_path, review_path, output, True)
    assert not output.exists()
    assert not (tmp_path / "BusinessProject" / "dsd001-native-owner.lock").exists()


def test_receipt_write_failure_still_releases_own_lease(tmp_path, monkeypatch, capsys):
    fake_module = tmp_path / "BusinessProject" / "source" / "experiments" / "dsd" / "reviewed_qualification.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(reviewed, "__file__", str(fake_module))
    packet, approval = packet_and_review()
    packet["source"] = {"checkpoint": str(tmp_path / "BusinessProject" / "immutable-source"),
                        "checkpoint_sha256": "e" * 64}
    packet_path, review_path = tmp_path / "packet.json", tmp_path / "review.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    approval["packet_sha256"] = reviewed._hash_file(packet_path)
    review_path.write_text(json.dumps(approval), encoding="utf-8")
    original_write = reviewed._write
    def write(path, value):
        if Path(path).name == "counters.json":
            raise OSError("fixture receipt disk error")
        original_write(path, value)
    monkeypatch.setattr(reviewed, "_write", write)
    fixture = SimpleNamespace(fixture_only=True, runtime=lambda: {}, collect=lambda: None,
        custody=lambda p: {"checkpoint_sha256": "e" * 64,
                          "data_sha256": {name: b["sha256"] for name, b in p["data"]["bindings"].items()}})
    with pytest.raises(OSError, match="fixture receipt disk error"):
        reviewed.execute(packet_path, review_path, tmp_path / "BusinessProject" / "receipt-failure", True, _services=fixture)
    assert "receipt/cleanup failure" in capsys.readouterr().err
    assert not (tmp_path / "BusinessProject" / "dsd001-native-owner.lock").exists()


def test_full_tiny_fixture_orchestration_raw_parity_both_seeds_no_optimizer(tmp_path, monkeypatch):
    """Actual CPU objectives/parity/backward; mocked local loader and scripted gate."""
    import gc
    import torch
    from experiments.rineya_self_distillation_axes_v1 import engine as objective_module
    from experiments.rineya_self_distillation_axes_v1 import mechanical_qualification as mechanical
    from experiments.rineya_self_distillation_axes_v1.engine import Engine
    from experiments.rineya_self_distillation_axes_v1.native import fixed_precision
    from experiments.rineya_self_distillation_axes_v1.plan import arms
    from experiments.rineya_self_distillation_axes_v1.teacher import tensor_digest
    from experiments.rineya_self_distillation_axes_v1.tests.test_engine import TinyLM, TinyTokenizer, config, rows
    from experiments.rineya_self_distillation_axes_v1.tests.test_mechanical_qualification import gate_rows
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("Optimizer forbidden"))
    fake_module = tmp_path / "BusinessProject" / "source" / "experiments" / "dsd" / "reviewed_qualification.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(reviewed, "__file__", str(fake_module))
    packet, approval = packet_and_review()
    packet["source"] = {"checkpoint": "/fixture-no-load", "checkpoint_sha256": "e" * 64, "core_id": "CPU_FIXTURE_ONLY"}
    packet["limits"] = dict(limits(), qualification_forward_calls=2048,
        qualification_processed_input_tokens=1048576, qualification_generated_tokens=2048,
        qualification_wall_seconds=120)
    packet["code_sha256"] = "f" * 64
    bindings = packet["data"]["bindings"]
    for name, values in (("screen.jsonl", rows()), ("teacher-gate.jsonl", gate_rows())):
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(row) for row in values) + "\n", encoding="utf-8")
        bindings[name] = {"path": str(path), "sha256": reviewed._hash_file(path)}
    packet["design"] = {"config_sha256_by_seed": {}}
    for seed in (17, 29):
        cfg = dict(config(), checkpoint=packet["source"]["checkpoint"], checkpoint_sha256="e" * 64,
                   dataset=bindings["screen.jsonl"]["path"], dataset_sha256=bindings["screen.jsonl"]["sha256"], seed=seed)
        path = tmp_path / f"native-seed{seed}.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        bindings[path.name] = {"path": str(path), "sha256": reviewed._hash_file(path)}
        packet["design"]["config_sha256_by_seed"][str(seed)] = reviewed._digest(cfg)
    approval["data_admission"] = {name: {"status": "APPROVED", "reviewer": "fixture reviewer",
        "sha256": bindings[name]["sha256"]} for name in ("screen.jsonl", "teacher-gate.jsonl")}
    packet_path, review_path = tmp_path / "packet.json", tmp_path / "review.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    approval["packet_sha256"] = reviewed._hash_file(packet_path)
    review_path.write_text(json.dumps(approval), encoding="utf-8")
    torch.manual_seed(17)
    immutable = TinyLM().eval()
    immutable.q_proj.weight.data = immutable.q_proj.weight.data.bfloat16()
    immutable.requires_grad_(False)
    before = tensor_digest(immutable.named_parameters())
    loads = []
    def bind_adapted(_config):
        loads.append("adapted")
        model = copy.deepcopy(immutable)
        model.q_proj.weight.data = model.q_proj.weight.data.float()
        model.q_proj.weight.requires_grad_(True)
        return model, TinyTokenizer()
    def bind_raw(_config):
        loads.append("raw")
        return copy.deepcopy(immutable)
    def gate(engine, gate_data, adaptation_data):
        original_generate, original_decode = engine.generate, engine.tokenizer.decode
        teacher_prefixes = {tuple(engine.ids(row, True)) for row in gate_data}
        clean_successes = {tuple(engine.ids(row, False)) for row in gate_data[:6]}
        def scripted(prefix, seed, greedy=False):
            original_generate(prefix, seed, greedy)  # Exercise actual guarded forwards.
            return [3 if tuple(prefix) in teacher_prefixes | clean_successes else 4], "eos"
        engine.generate = scripted
        engine.tokenizer.decode = lambda tokens, **kwargs: "OK" if tokens == [3] else "BAD"
        try:
            return mechanical.qualify_teacher_competence(engine, gate_data, adaptation_data)
        finally:
            engine.generate, engine.tokenizer.decode = original_generate, original_decode
    methods = SimpleNamespace(**{name: getattr(mechanical, name) for name in (
        "_gate_rows", "prepare_parity_probes", "capture_raw_source_parity", "check_loaded_source_parity",
        "maximum_shape_backward", "validate_native_evidence")}, qualify_teacher_competence=gate)
    fixture = SimpleNamespace(fixture_only=True, torch=torch, objective_module=objective_module,
        runtime=lambda: dict(reviewed.EXPECTED_RUNTIME), snapshot=resources,
        reset_peaks=lambda: loads.append("reset"), fixed_precision=fixed_precision,
        preflight=lambda cfg: (rows(), {"code_sha256": packet["code_sha256"], "checkpoint_sha256": "e" * 64,
                                       "config_sha256": reviewed._digest(cfg)}),
        bind_model=bind_adapted, bind_raw=bind_raw, engine_type=Engine, mechanical=methods,
        arms=arms, tensor_digest=tensor_digest, validate_packet=lambda p: None,
        custody=lambda p: {"checkpoint_sha256": "e" * 64,
                          "data_sha256": {name: b["sha256"] for name, b in p["data"]["bindings"].items()}},
        profiler=lambda: torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]),
        collect=gc.collect, seed=torch.manual_seed)
    output = tmp_path / "BusinessProject" / "happy-fixture"
    result = reviewed.execute(packet_path, review_path, output, True, _services=fixture)
    assert result["status"] == "FIXTURE_ONLY_QUALIFICATION" and result["native_checkpoint_executed"] is False
    assert result["training_updates"] == 0 and result["optimizer_objects_created"] == 0
    assert result["source_custody_unchanged"] and loads == ["reset", "adapted", "raw", "adapted"]
    assert result["partial_checks"]["teacher_gate"]["paired_wins"] == 2
    assert all(result["partial_checks"][str(seed)]["strict_validation"]["mechanical_evidence_validated"] for seed in (17, 29))
    assert all(len(result["partial_checks"][str(seed)]["objective_observations"]) == 4 for seed in (17, 29))
    assert result["counters"]["counts"]["forward_calls"] > 0
    assert tensor_digest(immutable.named_parameters()) == before
    assert (output / "qualification.json").is_file() and not (output / "FAILED.json").exists()
