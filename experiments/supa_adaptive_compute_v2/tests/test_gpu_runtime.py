"""GPU engineering guards; synthetic fixtures cannot license model execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from shared import generation, integrity, model_loading
from shared.guards import require_runtime
from shared.hashing import canonical_sha256, sha256_file, write_json


def cpu_config():
    return dict(device="cpu", dtype="float32", use_cache=False, threads=4,
                seed=17, max_prompt_seconds=900, max_rss_gib=8,
                checkpoint_sha256=model_loading.CANONICAL_SHA256)


def gpu_config():
    return {**cpu_config(), "track": "native_armr", "device": "cuda:0",
            "tf32": False, "runtime_qualification": "native_gpu_fp32_v1",
            "min_available_gib": .5, "min_available_vram_gib": 3.,
            "max_cuda_memory_gib": 6.5, "attention_implementation": "eager"}


def test_gpu_runtime_requires_explicit_supported_precision_and_device():
    require_runtime(cpu_config())
    require_runtime(gpu_config())
    for key, value in (("device", "cuda"), ("device", "cuda:1"),
                       ("device", "auto"), ("dtype", "float16"),
                       ("dtype", "bfloat16"), ("tf32", True),
                       ("use_cache", True), ("threads", 8),
                       ("runtime_qualification", "")):
        with pytest.raises(ValueError):
            require_runtime({**gpu_config(), key: value})


@pytest.mark.parametrize("key", ["min_available_vram_gib", "max_cuda_memory_gib"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_gpu_runtime_rejects_invalid_memory_contract(key, value):
    with pytest.raises(ValueError):
        require_runtime({**gpu_config(), key: value})


def test_cpu_synchronization_does_not_initialize_cuda(monkeypatch):
    from shared import runtime
    synchronize = Mock(side_effect=AssertionError("CPU must not touch CUDA"))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    runtime.synchronize("cpu")
    synchronize.assert_not_called()


def test_cuda_synchronization_uses_explicit_device(monkeypatch):
    from shared import runtime
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    runtime.synchronize("cuda:0")
    assert synchronize.call_count == 1
    args, kwargs = synchronize.call_args
    device = args[0] if args else kwargs["device"]
    assert str(device) == "cuda:0"


def test_cuda_memory_samples_after_sync_and_preserves_peaks(monkeypatch):
    from shared import runtime
    events = []
    monkeypatch.setattr(runtime, "synchronize", lambda device: events.append("sync"))
    values = dict(memory_allocated=11, memory_reserved=13,
                  max_memory_allocated=17, max_memory_reserved=19)
    for name, value in values.items():
        def sample(device, _name=name, _value=value):
            assert str(device) == "cuda:0" and events[0] == "sync"
            events.append(_name)
            return _value
        monkeypatch.setattr(torch.cuda, name, sample)
    observed = runtime.cuda_memory(gpu_config())
    assert observed["peak_cuda_allocated_bytes"] == 17
    assert observed["peak_cuda_reserved_bytes"] == 19
    assert len(events) == 5
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 7 * 2**30)
    with pytest.raises(MemoryError):
        runtime.cuda_memory(gpu_config())


def test_device_validation_includes_nonparameter_buffers():
    from shared.runtime import assert_model_device
    model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    assert assert_model_device(model, "cpu")
    model.register_buffer("wrong_buffer", torch.empty(2, device="meta"))
    with pytest.raises(RuntimeError, match="wrong_buffer"):
        assert_model_device(model, "cpu")


def reference_hash(model):
    """Historical byte contract, independent of the new streaming implementation."""
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def hash_fixture():
    model = torch.nn.Linear(17, 19).eval().requires_grad_(False)
    model.register_buffer("noncontiguous", torch.arange(323).reshape(17, 19).T)
    model.register_buffer("scalar", torch.tensor(11, dtype=torch.int64))
    model.register_buffer("empty", torch.empty(0, dtype=torch.float32))
    return model


def test_streamed_state_hash_preserves_historical_bytes_and_detects_mutation():
    model = hash_fixture()
    expected = reference_hash(model)
    assert integrity.state_hash(model) == expected
    for chunk_bytes in (1, 17, 128):
        assert integrity.state_hash(model, chunk_bytes=chunk_bytes) == expected
    with torch.no_grad():
        model.weight[0, 0] += .125
    assert integrity.state_hash(model) == reference_hash(model)
    assert integrity.state_hash(model) != expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_state_hash_is_device_independent_without_bulk_model_transfer():
    cpu = hash_fixture()
    expected = reference_hash(cpu)
    gpu = copy.deepcopy(cpu).to("cuda:0")
    assert integrity.state_hash(gpu) == expected
    assert all(value.device.type == "cuda" for value in gpu.state_dict().values())


def complete_cpu_receipt():
    files = {"model.safetensors": model_loading.CANONICAL_SHA256}
    return {**cpu_config(), "status": "COMPLETE", "arm": "A0",
            "canonical_unchanged": True, "torch": "2.12.1+cu126",
            "transformers": "5.14.1", "cache_mode": "full_prefix_no_cache",
            "model_class": "Qwen3AbstainForCausalLM", "trainable_parameters": 0,
            "backbone_gradients": 0, "state_sha256_before": "same-state",
            "state_sha256_after": "same-state", "checkpoint_files_before": files,
            "checkpoint_files_after": files}


def test_complete_cpu_baseline_cannot_license_gpu_candidate():
    receipt = complete_cpu_receipt()
    assert integrity.validate_inference_receipt(receipt, cpu_config())
    with pytest.raises(RuntimeError, match="runtime.*device|device"):
        integrity.validate_inference_receipt(receipt, gpu_config())


def test_device_label_alone_cannot_turn_cpu_receipt_into_gpu_evidence():
    forged = {**complete_cpu_receipt(), "device": "cuda:0"}
    with pytest.raises(RuntimeError):
        integrity.validate_inference_receipt(forged, gpu_config())


@pytest.fixture
def qualification_fixture(tmp_path, monkeypatch):
    from shared import qualifications as q
    from shared import datasets
    config, sources = gpu_config(), {"shared/runtime.py": "runtime-source"}
    dataset = tmp_path / "data/native_armr/manifest.json"
    write_json(dataset, {"track": "native_armr", "version": "fixture"})
    registration = dict(config=config, sources=sources, registration_sha256="registration",
                        dataset_manifests={dataset.relative_to(tmp_path).as_posix(): sha256_file(dataset)})
    hardware = {"device": "cuda:0", "device_name": "fixture GPU", "tf32_matmul": False}
    directory = tmp_path / "native_armr/evidence/00_runtime_qualification/fixture"
    report = directory / "report.json"
    write_json(report, {"qualification_pass": True})
    receipt = dict(status="COMPLETE", qualification_pass=True, track="native_armr",
        runtime_qualification=config["runtime_qualification"], config_sha256=canonical_sha256(config),
        registration_sha256="registration", dataset_manifest_sha256=sha256_file(dataset),
        sources=sources, runtime_identity=hardware, canonical_unchanged=True,
        report_sha256=sha256_file(report))
    path = directory / "run_manifest.json"
    write_json(path, receipt)
    monkeypatch.setattr(q, "ROOT", tmp_path)
    monkeypatch.setattr(q, "REPO", tmp_path)
    monkeypatch.setattr(q, "source_files", lambda track: sources)
    monkeypatch.setattr(q, "configure_runtime", Mock())
    monkeypatch.setattr(q, "runtime_identity", lambda _: hardware)
    monkeypatch.setattr(datasets, "load_frozen_dataset", lambda *a, **k: [])
    # These tests isolate provenance; numerical/coverage validation has its own
    # independent report suite using the real frozen calibration role.
    monkeypatch.setattr(q, "validate_native_qualification_report", lambda *a: True)
    committed = Mock()
    monkeypatch.setattr(q, "_require_committed", committed)
    return SimpleNamespace(q=q, config=config, dataset=dataset, registration=registration,
                           receipt=receipt, path=path, report=report, committed=committed)


def test_gpu_qualification_requires_committed_receipt_and_report(qualification_fixture):
    f = qualification_fixture
    result = f.q.verify_runtime_qualification(f.config, f.dataset, f.registration)
    assert result["status"] == "QUALIFIED" and result["receipt_sha256"] == sha256_file(f.path)
    assert {call.args[0] for call in f.committed.call_args_list} == {f.path, f.report}
    f.committed.side_effect = RuntimeError("fixture report is not committed")
    with pytest.raises(RuntimeError, match="not committed"):
        f.q.verify_runtime_qualification(f.config, f.dataset, f.registration)


@pytest.mark.parametrize("field,value", [
    ("qualification_pass", False), ("status", "FAILED"), ("track", "cross_scale"),
    ("registration_sha256", "stale"), ("runtime_qualification", "different"),
    ("config_sha256", "different"), ("dataset_manifest_sha256", "different"),
    ("sources", {"shared/runtime.py": "edited"}), ("runtime_identity", {"device": "cpu"}),
    ("canonical_unchanged", False), ("report_sha256", "edited")])
def test_gpu_qualification_rejects_changed_provenance(qualification_fixture, field, value):
    f = qualification_fixture
    f.path.write_text(json.dumps({**f.receipt, field: value}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        f.q.verify_runtime_qualification(f.config, f.dataset, f.registration)


def test_gpu_qualification_rejects_missing_or_duplicate_ticket(qualification_fixture):
    f = qualification_fixture
    duplicate = f.path.parent.parent / "duplicate"
    write_json(duplicate / "report.json", {"qualification_pass": True})
    write_json(duplicate / "run_manifest.json", f.receipt)
    with pytest.raises(RuntimeError, match="exactly one"):
        f.q.verify_runtime_qualification(f.config, f.dataset, f.registration)
    f.path.unlink()
    (duplicate / "run_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="exactly one"):
        f.q.verify_runtime_qualification(f.config, f.dataset, f.registration)


def test_cpu_reference_qualification_never_opens_gpu_evidence(monkeypatch):
    from shared import qualifications
    monkeypatch.setattr(qualifications, "configure_runtime", Mock(side_effect=AssertionError("CUDA touched")))
    result = qualifications.verify_runtime_qualification(cpu_config(), "never-opened.json")
    assert result["status"] == "CPU_REFERENCE" and result["receipt_sha256"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_generation_places_prefix_tokens_and_nll_labels_and_returns_cpu_logits(monkeypatch):
    # Four-token synthetic vocabulary exercises device handling without loading
    # any checkpoint, dataset, Qwen model, or native/cross-scale candidate.
    torch.set_num_threads(2)
    seen = []
    class DeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
            self.generation_config = SimpleNamespace(eos_token_id=2)
            self.config = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1,
                head_dim=2, hidden_size=2, intermediate_size=4, num_hidden_layers=1, vocab_size=4)

        def forward(self, input_ids, attention_mask, use_cache, **kwargs):
            assert input_ids.device == attention_mask.device == self.anchor.device
            assert use_cache is False
            seen.append(input_ids.detach().cpu().tolist()[0])
            size = input_ids.shape[1]
            logits = torch.zeros(1, size, 4, device=input_ids.device)
            logits[..., 1 if size == 2 else 2] = 9.
            return SimpleNamespace(logits=logits, past_key_values=None)

    model = DeviceModel().to("cuda:0").eval()
    tokenizer = SimpleNamespace(encode=lambda *a, **k: [1], decode=lambda *a, **k: "yes")
    case = dict(example_id="gpu_fixture", scenario_id="gpu_fixture", family="fixture",
                prompt="Fixture only", scorer={"type": "exact", "answers": ["yes"]})
    wrapper = SimpleNamespace(records=[], calls=0, capture=False, min_rms_ratio=1.,
                              max_rms_ratio=1., extra_layer_applications=0)
    config = {**gpu_config(), "max_new_tokens": 3, "top_k": 2}
    monkeypatch.setattr(generation, "encode", lambda *a: torch.tensor([[0, 3]]))
    loss = torch.nn.functional.cross_entropy
    def checked_loss(values, labels, *args, **kwargs):
        assert values.device == labels.device == model.anchor.device
        return loss(values, labels, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", checked_loss)
    with torch.inference_mode():
        row, first = generation.evaluate_case(model, tokenizer, case, config, wrapper,
                                               base=torch.tensor([0., 9., 0., 0.]))
    assert row["token_ids"] == [1, 2]
    assert row["gold_nll"] is not None
    assert seen == [[0, 3], [0, 3, 1], [0, 3, 1]]
    assert first.device.type == "cpu" and not first.requires_grad
