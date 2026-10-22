"""Explicitly reviewed, bounded ZERO-UPDATE qualification; never comparison training.

Preparation or candidate split labels are not permission. No native dependency is
imported, model loaded, lease created or output written before explicit opt-in and
an exact DSD-001 packet/data review. This module has not executed native Rineya.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid


EXPECTED_RUNTIME = {"python": "3.11.15", "torch": "2.10.0+cu126", "transformers": "4.57.1",
                    "safetensors": "0.7.0", "tokenizers": "0.22.2"}
PHASE = "zero_update_qualification"


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _release_owned_lease(path, token):
    if Path(path).exists():
        try:
            ours = json.loads(Path(path).read_text(encoding="utf-8")).get("token") == token
        except (OSError, ValueError):
            ours = False
        if ours:
            Path(path).unlink()


def validate_review(packet, review, packet_sha256):
    if (review.get("experiment") != "DSD-001" or review.get("packet_sha256") != packet_sha256
            or review.get("phase") != PHASE or review.get("status") != "APPROVED"
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
        raise PermissionError("Exact DSD-001 packet and zero-update phase require an explicit APPROVED reviewer")
    if packet.get("experiment") != "DSD-001":
        raise PermissionError("Foreign experiment permits are forbidden")
    admissions = review.get("data_admission", {})
    for name in ("screen.jsonl", "teacher-gate.jsonl"):
        admission = admissions.get(name, {})
        expected = packet["data"]["bindings"][name]["sha256"]
        if (admission.get("status") != "APPROVED" or admission.get("sha256") != expected
                or not isinstance(admission.get("reviewer"), str) or not admission["reviewer"].strip()):
            raise PermissionError("Both exact screen and gate bytes need independent explicit data admission")
    if any(packet.get("runtime_recipe", {}).get(k) != v for k, v in EXPECTED_RUNTIME.items()):
        raise ValueError("Packet must freeze the exact proposed native runtime versions")
    return True


def windows_host_snapshot():
    """Windows available RAM and process peak working set, without psutil."""
    if os.name != "nt":
        raise RuntimeError("Native qualification requires the reviewed Windows host accounting")
    from ctypes import wintypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page", "avail_page",
                                                   "total_virtual", "avail_virtual", "avail_extended")]
    class ProcessCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "quota_peak_paged",
                                                "quota_paged", "quota_peak_nonpaged", "quota_nonpaged",
                                                "pagefile", "peak_pagefile", "private_usage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    process_api = ctypes.WinDLL("psapi", use_last_error=True)
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    counters = ProcessCounters()
    counters.cb = ctypes.sizeof(counters)
    process_api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessCounters), wintypes.DWORD]
    process_api.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not process_api.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"available_host_bytes": memory.avail_phys, "total_host_bytes": memory.total_phys,
            "process_working_set_bytes": counters.working_set,
            "process_peak_working_set_bytes": counters.peak_working_set,
            "process_private_bytes": counters.private_usage}


class ResourceGuard:
    """Conservative counters and resource floors before AND after every forward.

    Generation reserves one response-token charge before each sampling forward,
    so a ceiling cannot be exceeded by the final token before a later check.
    Failed forwards remain charged attempted work. No token/failure is dropped.
    """
    def __init__(self, limits, output, snapshot, clock=time.monotonic, started=None):
        self.limits, self.output, self.snapshot, self.clock = limits, Path(output), snapshot, clock
        self.started = clock() if started is None else started
        self.counts = {"forward_calls": 0, "processed_input_tokens": 0, "generated_token_charges": 0}
        self.stages, self.forwards = [], []
        self.in_generation = 0
        self.peak_host_working_set_bytes = 0

    def _sample(self, stage, before_load=False):
        elapsed = self.clock() - self.started
        if elapsed > self.limits["qualification_wall_seconds"]:
            raise RuntimeError("Qualification wall-time ceiling exceeded")
        sample = dict(self.snapshot())
        sample.update(stage=stage, elapsed_seconds=elapsed)
        floor = self.limits["minimum_free_host_gib_before" if before_load else "minimum_free_host_gib_during"]
        if sample["available_host_bytes"] < floor * 1024**3:
            raise RuntimeError("Host available-memory floor violated")
        cuda_floor = self.limits["minimum_free_cuda_gib_before" if before_load else "minimum_free_cuda_gib_during"]
        if sample["cuda_free_bytes"] < cuda_floor * 1024**3:
            raise RuntimeError("CUDA free-memory floor violated")
        sample["disk_free_bytes"] = shutil.disk_usage(self.output).free
        if sample["disk_free_bytes"] < self.limits["minimum_free_disk_gib"] * 1024**3:
            raise RuntimeError("Disk free-memory floor violated")
        self.peak_host_working_set_bytes = max(self.peak_host_working_set_bytes,
                                               sample["process_peak_working_set_bytes"])
        return sample

    def stage(self, name, before_load=False):
        if len(self.stages) >= 64:
            raise RuntimeError("Qualification stage sample ceiling exceeded")
        sample = self._sample(name, before_load)
        self.stages.append(sample)
        return sample

    def forward(self, token_count, invoke):
        if type(token_count) is not int or not 1 <= token_count <= self.limits["max_sequence_tokens"]:
            raise ValueError("Forward exceeds frozen complete-sequence budget")
        prospective = {"forward_calls": self.counts["forward_calls"] + 1,
                       "processed_input_tokens": self.counts["processed_input_tokens"] + token_count,
                       "generated_token_charges": self.counts["generated_token_charges"] + bool(self.in_generation)}
        for key, ceiling in (("forward_calls", "qualification_forward_calls"),
                             ("processed_input_tokens", "qualification_processed_input_tokens"),
                             ("generated_token_charges", "qualification_generated_tokens")):
            if prospective[key] > self.limits[ceiling]:
                raise RuntimeError(f"Qualification {key} ceiling exceeded before forward")
        if len(self.forwards) >= 2048:
            raise RuntimeError("Bounded forward resource receipt ceiling exceeded")
        before = self._sample("forward_before")
        self.counts = prospective
        record = {"number": prospective["forward_calls"], "input_tokens": token_count,
                  "generation": bool(self.in_generation), "before": before}
        self.forwards.append(record)
        error = None
        try:
            return invoke()
        except BaseException as exc:
            error = exc
            record["forward_error"] = {"type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            try:
                record["after"] = self._sample("forward_after")
            except BaseException as exc:
                record["after_guard_error"] = {"type": type(exc).__name__, "reason": str(exc)}
                if error is not None:
                    error.add_note("Post-forward resource guard: " + str(exc))
                else:
                    raise

    def attach_engine(self, engine):
        forward, generate = engine.forward, engine.generate
        engine.forward = lambda tokens: self.forward(len(tokens), lambda: forward(tokens))
        def guarded_generate(*args, **kwargs):
            self.in_generation += 1
            try:
                return generate(*args, **kwargs)
            finally:
                self.in_generation -= 1
        engine.generate = guarded_generate
        return engine

    def attach_raw(self, model):
        original = model.forward
        def guarded_forward(*args, **kwargs):
            ids = kwargs.get("input_ids")
            if ids is None or list(ids.shape)[0] != 1:
                raise ValueError("Raw parity requires explicit batch-one input_ids")
            return self.forward(int(ids.shape[-1]), lambda: original(*args, **kwargs))
        model.forward = guarded_forward
        return model

    def report(self):
        return {"counts": self.counts, "stage_samples": self.stages, "forward_samples": self.forwards,
                "peak_host_working_set_bytes": self.peak_host_working_set_bytes,
                "elapsed_seconds": self.clock() - self.started,
                "training_updates": 0, "optimizer_objects_created": 0,
                "counter_semantics": "Attempted forwards; generation token reservation precedes each sampling forward"}


@contextmanager
def _external_caches(output):
    cache_root = Path(output) / "caches"
    temp_root = Path(output) / "temp"
    cache_root.mkdir()
    temp_root.mkdir()
    values = {"HF_HOME": str(cache_root / "huggingface"), "TORCH_HOME": str(cache_root / "torch"),
              "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
              "HUGGINGFACE_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
              "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
              "TORCH_EXTENSIONS_DIR": str(cache_root / "torch-extensions"),
              "CUDA_CACHE_PATH": str(cache_root / "cuda"), "XDG_CACHE_HOME": str(cache_root),
              "TRITON_CACHE_DIR": str(cache_root / "triton"),
              "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "inductor"),
              "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
              "TMP": str(temp_root), "TEMP": str(temp_root)}
    prior = {key: os.environ.get(key) for key in values}
    bytecode = sys.dont_write_bytecode
    previous_tempdir = tempfile.tempdir
    try:
        os.environ.update(values)
        sys.dont_write_bytecode = True
        tempfile.tempdir = str(temp_root)
        yield
    finally:
        sys.dont_write_bytecode = bytecode
        tempfile.tempdir = previous_tempdir
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _native_services():
    # This function is called only after both opt-in/review and output creation.
    import torch
    from . import native, mechanical_qualification as mechanical
    from .engine import Engine
    from . import engine as objective_module
    from .plan import arms, checkpoint_manifest
    from .teacher import tensor_digest
    from .proposal import validate_packet
    def snapshot():
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Reviewed BF16 CUDA runtime is unavailable")
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(windows_host_snapshot(), cuda_free_bytes=free, cuda_total_bytes=total,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    def bind_raw(config):
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(config["checkpoint"], local_files_only=True, trust_remote_code=False)
        if (cfg.model_type != "qwen3" or getattr(cfg, "quantization_config", None)
                or getattr(cfg, "use_sliding_window", False) or getattr(cfg, "rope_scaling", None)):
            raise ValueError("Raw source has unsupported quantization/architecture/context state")
        model, info = AutoModelForCausalLM.from_pretrained(
            config["checkpoint"], local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Raw source did not load exactly")
        if (not 450_000_000 <= sum(p.numel() for p in model.parameters()) <= 750_000_000
                or getattr(model, "is_gradient_checkpointing", False)):
            raise ValueError("Raw source is not the eligible uncheckpointed 0.6B model")
        model.requires_grad_(False)
        model.config.use_cache = False
        return model.to(config["device"]).eval()
    def custody(packet):
        return {"checkpoint_sha256": checkpoint_manifest(packet["source"]["checkpoint"])["sha256"],
                "data_sha256": {name: _hash_file(binding["path"])
                                for name, binding in packet["data"]["bindings"].items()}}
    def runtime():
        return dict(python=platform.python_version(), torch=torch.__version__,
                    **{name: importlib.metadata.version(name) for name in
                       ("transformers", "safetensors", "tokenizers")})
    def profiler():
        return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
    return SimpleNamespace(fixture_only=False, torch=torch, objective_module=objective_module,
                           runtime=runtime, snapshot=snapshot,
                           reset_peaks=torch.cuda.reset_peak_memory_stats,
                           fixed_precision=native.fixed_precision, preflight=native.preflight,
                           bind_model=native.bind_model, bind_raw=bind_raw, engine_type=Engine,
                           mechanical=mechanical, arms=arms, tensor_digest=tensor_digest,
                           custody=custody, validate_packet=validate_packet, profiler=profiler,
                           collect=lambda: (gc.collect(), torch.cuda.empty_cache()),
                           seed=torch.manual_seed)


@contextmanager
def _observe_objectives(engine, services):
    """Check actual KL inputs, common responses and each causal offset at runtime.

    Only <=32-response slices are cloned for equality checks; no full-sequence
    tensor or teacher graph is retained by the observer. The numerical objective
    function receives its original arguments unchanged.
    """
    module = services.objective_module
    original_kl, original_forward, original_objective = module.token_kl, engine.forward, engine.objective
    state, observations = {}, []
    def forward(tokens):
        logits = original_forward(tokens)
        if state:
            prefix = state["teacher_prefix"] if tokens[:len(state["teacher_prefix"])] == state["teacher_prefix"] else state["student_prefix"]
            length = len(tokens) - len(prefix)
            expected = (logits[:, len(prefix) - 1:len(prefix) + length - 1, :].detach().clone()
                        if length > 0 else None)
            state["forwards"].append({"tokens": list(tokens), "expected": expected})
            state["forwards"] = state["forwards"][-2:]
        return logits
    def kl(student, teacher, direction):
        vocabulary = engine.model.config.vocab_size
        if (list(student.shape) != list(teacher.shape) or student.shape[-1] != vocabulary
                or student.shape[0] != 1 or student.shape[1] < 1 or teacher.requires_grad):
            raise RuntimeError("Native KL inputs violate full-vocabulary detached teacher mechanics")
        teacher_record, student_record = state["forwards"]
        length = int(student.shape[1])
        qtokens, ptokens = teacher_record["tokens"], student_record["tokens"]
        if (qtokens[:-length] != state["teacher_prefix"] or ptokens[:-length] != state["student_prefix"]
                or qtokens[-length:] != ptokens[-length:]):
            raise RuntimeError("Teacher/student objectives changed the common sampled response prefix")
        if (not services.torch.equal(teacher, teacher_record["expected"])
                or not services.torch.equal(student, student_record["expected"])):
            raise RuntimeError("Teacher/student response logits have an incorrect causal offset")
        observations.append({"full_vocabulary": True, "teacher_detached": True,
                             "common_response_prefix": True, "correct_causal_shift": True,
                             "vocabulary_size": vocabulary, "response_tokens": length,
                             "teacher_prompt_tokens": len(state["teacher_prefix"]),
                             "student_prompt_tokens": len(state["student_prefix"]),
                             "direction": direction, "arm_id": state["arm_id"]})
        return original_kl(student, teacher, direction)
    @contextmanager
    def objective(row, arm, step):
        state.update(student_prefix=engine.ids(row, False), teacher_prefix=engine.ids(row, True),
                     arm_id=arm.name, forwards=[])
        try:
            with original_objective(row, arm, step) as result:
                yield result
        finally:
            state.clear()
    try:
        engine.forward, engine.objective, module.token_kl = forward, objective, kl
        yield observations
    finally:
        engine.forward, engine.objective, module.token_kl = original_forward, original_objective, original_kl


def _verify_bindings(packet):
    from .data import TEACHER_TEMPLATE
    package = Path(__file__).parent
    actual = {str(path.relative_to(package)): _hash_file(path) for path in sorted(package.rglob("*.py"))}
    if actual != packet["code_files"] or _digest(actual) != packet["code_sha256"]:
        raise ValueError("Packet code changed after exact review")
    if _digest(TEACHER_TEMPLATE) != packet["teacher_template_sha256"]:
        raise ValueError("Reviewed privileged-context template changed")
    for binding in packet["data"]["bindings"].values():
        if _hash_file(binding["path"]) != binding["sha256"]:
            raise ValueError("Reviewed data/config binding changed")
    source = packet["source"]
    for kind in ("custody", "inspection"):
        if _hash_file(source[kind + "_path"]) != source[kind + "_file_sha256"]:
            raise ValueError("Reviewed source receipt changed")


def execute(packet_path, review_path, output, execute_qualification=False, *, _services=None):
    if execute_qualification is not True:
        raise PermissionError("Native zero-update qualification requires --execute-qualification and exact review")
    packet_path, review_path = Path(packet_path), Path(review_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    packet_sha = _hash_file(packet_path)
    validate_review(packet, review, packet_sha)
    workspace = Path(__file__).resolve().parents[2]
    business_root = workspace.parent
    output = Path(output).resolve()
    checkpoint = Path(packet["source"]["checkpoint"]).resolve()
    if (output == business_root or not output.is_relative_to(business_root)
            or output.is_relative_to(workspace) or any(part.lower() == ".codex" for part in output.parts)):
        raise ValueError("Qualification outputs/caches must be external source under BusinessProject")
    if output.is_relative_to(checkpoint):
        raise ValueError("Never create qualification artifacts inside the immutable checkpoint")
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("Qualification artifacts cannot be inside any Git checkout")
    lease_path = business_root / "dsd001-native-owner.lock"
    lease_token = str(uuid.uuid4())
    lease = {"experiment": "DSD-001", "phase": PHASE, "pid": os.getpid(), "token": lease_token,
             "packet_sha256": packet_sha, "output": str(output)}
    qualification_started = time.monotonic()
    # Never inspect/adopt/remove a preexisting lease or launch a competing worker.
    _write(lease_path, lease)
    created = False
    services = guard = engine = raw_model = None
    custody_before = custody_after = None
    native_load_returned = False
    identities, partial, failure = {}, {}, None
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        _write(output / "identity.json", {"experiment": "DSD-001", "phase": PHASE,
                                         "packet_sha256": packet_sha, "review_sha256": _hash_file(review_path),
                                         "reviewer": review["reviewer"], "lease": lease,
                                         "training_updates": 0, "optimizer_objects_created": 0})
        with _external_caches(output):
            services = _services or _native_services()
            # Capture custody before runtime/preflight validation can fail; a
            # failed stage must still preserve both admitted data identities.
            custody_before = services.custody(packet)
            _write(output / "custody-before.json", custody_before)
            actual_runtime = services.runtime()
            if actual_runtime != EXPECTED_RUNTIME:
                raise RuntimeError("Actual native dependency versions differ from the reviewed runtime")
            services.validate_packet(packet)
            if not getattr(services, "fixture_only", False):
                _verify_bindings(packet)
                import subprocess
                branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=workspace, text=True).strip()
                head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
                if branch != packet["branch"] or head != packet["start_commit"]:
                    raise ValueError("Reviewed branch or starting Git commit changed")
                partial["observed_git_identity"] = {"branch": branch, "head": head}
            if (custody_before["checkpoint_sha256"] != packet["source"]["checkpoint_sha256"]
                    or custody_before["data_sha256"] != {name: b["sha256"] for name, b in packet["data"]["bindings"].items()}):
                raise ValueError("Checkpoint/data no longer match reviewed custody")
            guard = ResourceGuard(packet["limits"], output, services.snapshot, started=qualification_started)
            services.reset_peaks()  # Exactly once, BEFORE any native model load.
            guard.stage("preflight_before", before_load=True)
            configs, screen = {}, None
            for seed in (17, 29):
                binding = packet["data"]["bindings"][f"native-seed{seed}.json"]
                config = json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
                if _digest(config) != packet["design"]["config_sha256_by_seed"][str(seed)]:
                    raise ValueError("Per-seed canonical config differs from review")
                if (config["seed"] != seed or config["checkpoint_sha256"] != packet["source"]["checkpoint_sha256"]
                        or config["checkpoint"] != packet["source"]["checkpoint"]
                        or config["dataset_sha256"] != packet["data"]["bindings"]["screen.jsonl"]["sha256"]):
                    raise ValueError("Per-seed config changed source/data/order identity")
                rows, identity = services.preflight(config)
                if identity["code_sha256"] != packet["code_sha256"]:
                    raise ValueError("Native preflight code differs from the reviewed packet")
                if screen is not None and screen != rows:
                    raise ValueError("Paired qualification configs do not use identical screen rows")
                configs[seed], identities[str(seed)], screen = config, identity, rows
            gate = [json.loads(line) for line in Path(packet["data"]["bindings"]["teacher-gate.jsonl"]["path"])
                    .read_text(encoding="utf-8").splitlines() if line.strip()]
            with services.fixed_precision():
                services.seed(17)
                guard.stage("probe_adapted_load_before", before_load=True)
                model, tokenizer = services.bind_model(configs[17])
                native_load_returned = not getattr(services, "fixture_only", False)
                engine = guard.attach_engine(services.engine_type(model, tokenizer, configs[17]))
                del model, tokenizer
                guard.stage("probe_adapted_load_after")
                engine.check_rows(screen)
                services.mechanical._gate_rows(gate, engine.config["task"], screen)
                engine.check_rows(gate)
                probes = services.mechanical.prepare_parity_probes(engine, gate)
                # Clean prompt tokens cannot depend on reference/answer sentinels.
                for row in screen + gate:
                    altered = dict(row, reference="DSD_REFERENCE_SENTINEL", answer="DSD_ANSWER_SENTINEL")
                    if engine.ids(row, False) != engine.ids(altered, False):
                        raise RuntimeError("Privileged reference leaked into clean inference tokens")
                partial["token_budget_checked_rows"] = len(screen) + len(gate)
                _write(output / "parity-probes.json", probes)
                engine = None
                services.collect()
                guard.stage("probe_adapted_unloaded")
                guard.stage("raw_source_load_before", before_load=True)
                raw_model = guard.attach_raw(services.bind_raw(configs[17]))
                guard.stage("raw_source_load_after")
                raw_parity = services.mechanical.capture_raw_source_parity(raw_model, configs[17], probes)
                _write(output / "raw-source-parity.json", raw_parity)
                raw_model = None
                services.collect()
                guard.stage("raw_source_unloaded")
                guard.stage("qualification_adapted_load_before", before_load=True)
                model, tokenizer = services.bind_model(configs[17])
                engine = guard.attach_engine(services.engine_type(model, tokenizer, configs[17]))
                del model, tokenizer
                guard.stage("qualification_adapted_load_after")
                initial = services.tensor_digest(engine.model.named_parameters())
                with services.profiler() as profiler:
                    parity = services.mechanical.check_loaded_source_parity(engine, probes, raw_parity)
                    _write(output / "adapted-source-parity.json", parity)
                    for seed in (17, 29):
                        engine.config = configs[seed]
                        buffers = services.tensor_digest(engine.model.named_buffers())
                        with _observe_objectives(engine, services) as observations:
                            checks = engine.qualify(screen, services.arms((0.,)))
                        required_sd_arms = {arm.name for arm in services.arms((0.,)) if arm.method == "sd"}
                        if len(observations) != 4 or {item["arm_id"] for item in observations} != required_sd_arms:
                            raise RuntimeError("Exactly four observed KL qualification paths required")
                        checks["buffers_unchanged"] = buffers == services.tensor_digest(engine.model.named_buffers())
                        if checks["buffers_unchanged"] is not True:
                            raise RuntimeError("Representative objective qualification mutated buffers")
                        partial[str(seed)] = {"identity": identities[str(seed)], "checks": checks,
                                              "initial_parameters_sha256": initial,
                                              "objective_observations": observations,
                                              "scope": "five representative objectives; no update; not maximum-shape KL"}
                        _write(output / f"mechanics-seed{seed}.json", partial[str(seed)])
                    engine.config = configs[17]
                    maximum = services.mechanical.maximum_shape_backward(engine, probes)
                    _write(output / "maximum-shape-sft-backward.json", maximum)
                events = sorted({entry.key for entry in profiler.key_averages() if "scaled_dot_product" in entry.key})
                if not any("attention_math" in event for event in events):
                    raise RuntimeError("Math SDPA dispatch was not observed")
                guard.stage("mechanics_and_maximum_sft_after")
                teacher = services.mechanical.qualify_teacher_competence(engine, gate, screen)
                partial["teacher_gate"] = teacher
                _write(output / "teacher-competence.json", teacher)
                if teacher.get("pass") is not True:
                    raise RuntimeError("Prospective privileged teacher competence gate failed; comparison barred")
                if services.tensor_digest(engine.model.named_parameters()) != initial:
                    raise RuntimeError("Zero-update qualification changed initial parameters")
                guard.stage("teacher_gate_after")
                partial.update(raw_parity=raw_parity, adapted_parity=parity, maximum_shape_sft=maximum,
                               teacher_gate=teacher, backend_events=events, actual_runtime=actual_runtime)
    except BaseException as exc:
        failure = exc
    finally:
        try:
            if engine is not None:
                partial["actual_engine_counters"] = dict(engine.counts)
            engine = raw_model = None
            if services is not None:
                try:
                    services.collect()
                    custody_after = services.custody(packet)
                    if custody_before is not None and custody_after != custody_before:
                        raise RuntimeError("Source or reviewed data custody changed during qualification")
                    if guard is not None:
                        guard.stage("post_custody_final")
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note("Post-stage custody/resource cleanup: " + str(exc))
            if created:
                if failure is None:
                    try:
                        source = {"eligible": True, "source_id": packet["source"]["core_id"],
                                  "checkpoint_sha256": packet["source"]["checkpoint_sha256"]}
                        for seed in ("17", "29"):
                            cell = partial[seed]
                            observed = cell["objective_observations"]
                            required_sd_arms = {arm.name for arm in services.arms((0.,)) if arm.method == "sd"}
                            if len(observed) != 4 or {item["arm_id"] for item in observed} != required_sd_arms:
                                raise ValueError("Strict mechanical evidence needs all four distinct SD objective paths")
                            checks = dict(cell["checks"],
                                **{key: all(item[key] for item in observed) for key in
                                   ("full_vocabulary", "teacher_detached", "common_response_prefix")},
                                source_inference_parity=partial["adapted_parity"]["source_inference_parity"],
                                max_shape_no_update=partial["adapted_parity"]["max_shape_no_update"],
                                max_shape_backward_no_update=partial["maximum_shape_sft"]["max_shape_backward_no_update"],
                                source_custody_unchanged=custody_before == custody_after)
                            evidence = {"identity": cell["identity"], "source": source,
                                        "initial_parameters_sha256": cell["initial_parameters_sha256"],
                                        "training_updates": 0, "optimizer_objects_created": 0,
                                        "checks": checks, "diagnostics": cell["checks"]["diagnostics"],
                                        "backend_events": partial["backend_events"]}
                            validation = services.mechanical.validate_native_evidence(
                                evidence, cell["identity"], source, cell["initial_parameters_sha256"])
                            cell["strict_validation"] = dict(validation,
                                resource_integration="COOPERATIVE_SAMPLED_BOUNDARY_GUARDS",
                                objective_scope="actual representative full-vocabulary/detached/causal paths; maximum SFT only")
                            _write(output / f"strict-mechanics-seed{seed}.json", evidence)
                    except BaseException as exc:
                        failure = exc
                counters = guard.report() if guard is not None else {"training_updates": 0, "optimizer_objects_created": 0}
                _write(output / "counters.json", counters)
                _write(output / "custody-after.json", {"before": custody_before, "after": custody_after,
                    "unchanged": custody_before is not None and custody_before == custody_after})
                report = {"experiment": "DSD-001", "phase": PHASE, "packet_sha256": packet_sha,
                          "identities_by_seed": identities, "partial_checks": partial, "counters": counters,
                          "training_updates": 0, "optimizer_objects_created": 0,
                          "source_custody_unchanged": custody_before is not None and custody_before == custody_after,
                          "native_checkpoint_executed": native_load_returned,
                          "comparison_training_authorized": False, "production_promotion_allowed": False,
                          "scope": "configs17/29 representative five-objective qualification; exact raw parity; maximum-shape SFT only; eight initial teacher-gate rows",
                          "resource_limits": "Cooperative checks at forward/stage boundaries; cannot preempt a stalled kernel; memory floors are sampled, peaks retained."}
                if failure is None:
                    report["status"] = "FIXTURE_ONLY_QUALIFICATION" if getattr(services, "fixture_only", False) else "REVIEWED_ZERO_UPDATE_QUALIFICATION_PASSED"
                    _write(output / "qualification.json", report)
                else:
                    report.update(status="FAILED", error_type=type(failure).__name__, reason=str(failure),
                                  notes=getattr(failure, "__notes__", []))
                    _write(output / "FAILED.json", report)
        except BaseException as receipt_error:
            print("Qualification receipt/cleanup failure: " + str(receipt_error), file=sys.stderr)
            if failure is not None:
                receipt_error.add_note("Original qualification failure: " + str(failure))
            raise
        finally:
            _release_owned_lease(lease_path, lease_token)
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute-qualification", action="store_true")
    args = parser.parse_args(argv)
    execute(args.packet, args.review, args.output, args.execute_qualification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
