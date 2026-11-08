"""Future guarded disposable Titan qualification; importing this module loads no model.

Production requires the live Guardian from native_safety, its bound check method,
an issued attempt/slot grant, reviewed source bytes, retained leases and Windows
Job membership. The private injected protocol is exclusively a mechanical fixture.
Neither qualification nor its eight development readouts establishes model quality.
"""
from __future__ import annotations

import gc
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import time

from . import custody, generator, oracles, registry, schema, token_audit

STAGE = "native_qualification"
EOS_IDS = (151645, 151643)
FORWARD_CAP = 562
PROPOSAL_DIGEST_EXCLUSIONS = frozenset({"proposal_payload_sha256", "proposal_sha256",
    "review_descriptor", "cpu_qualification_descriptor", "operator_grant_descriptor"})
UNPINNED_LOADER_ASSETS = frozenset({"adapter_config.json", "adapter_model.bin", "adapter_model.safetensors",
    "pytorch_model.bin", "pytorch_model.bin.index.json", "model.safetensors.index.json", "tokenizer.model",
    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja", "processor_config.json"})


class _WindowsReadOnlyFile:
    """Retained GENERIC_READ handle, FILE_SHARE_READ only: deny write/delete."""
    def __init__(self, path, *, directory=False):
        if os.name != "nt":
            raise RuntimeError("STOP_SOURCE_READ_CUSTODY_REQUIRES_WINDOWS")
        self.path, self.directory, self.stream, self.descriptor = Path(path), directory, None, None
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        self.kernel.CreateFileW.restype = ctypes.c_void_p
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel.CloseHandle.restype = ctypes.c_int
        flags = 0x02000000 if directory else 0x08000000  # backup semantics / sequential scan
        self.handle = self.kernel.CreateFileW(str(self.path), 0x80000000, 0x00000001, None, 3, flags, None)
        if self.handle in (None, ctypes.c_void_p(-1).value):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not directory:
                import msvcrt
                self.descriptor = msvcrt.open_osfhandle(int(self.handle), os.O_RDONLY)
                self.stream = os.fdopen(self.descriptor, "rb", buffering=0)
            self.initial_identity = self.identity()
        except BaseException:
            self.close()
            raise

    def identity(self):
        class FileTime(ctypes.Structure):
            _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]
        class Info(ctypes.Structure):
            _fields_ = [("attributes", ctypes.c_uint32), ("created", FileTime), ("accessed", FileTime),
                ("written", FileTime), ("volume", ctypes.c_uint32), ("size_high", ctypes.c_uint32),
                ("size_low", ctypes.c_uint32), ("links", ctypes.c_uint32),
                ("index_high", ctypes.c_uint32), ("index_low", ctypes.c_uint32)]
        self.kernel.GetFileInformationByHandle.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
        self.kernel.GetFileInformationByHandle.restype = ctypes.c_int
        value = Info()
        if not self.handle or not self.kernel.GetFileInformationByHandle(self.handle, ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (value.volume, value.index_high, value.index_low, value.size_high, value.size_low,
                value.created.high, value.created.low, value.written.high, value.written.low)

    def verify(self):
        if self.identity() != self.initial_identity:
            raise RuntimeError("STOP_HELD_SOURCE_IDENTITY_CHANGED")
        # Check current path still addresses the protected file, not a different
        # file after a parent/junction change. This reads metadata, not weights.
        current = _WindowsReadOnlyFile(self.path, directory=self.directory)
        try:
            if current.initial_identity != self.initial_identity:
                raise RuntimeError("STOP_SOURCE_PATH_IDENTITY_CHANGED")
        finally:
            current.close()

    def read_chunks(self, size=8 * 1024 ** 2):
        if self.directory or self.stream is None:
            raise RuntimeError("STOP_SOURCE_STREAM_NOT_AVAILABLE")
        self.stream.seek(0)
        while chunk := self.stream.read(size):
            yield chunk

    def close(self):
        if self.stream is not None:
            self.stream.close()  # owns/closes the Win32 handle via the CRT fd
            self.stream = None
            self.descriptor = None
        elif self.descriptor is not None:
            # open_osfhandle transferred ownership even if fdopen then failed.
            # Close that exact CRT fd; raw CloseHandle would leave a stale fd
            # capable of closing a later reused kernel handle.
            os.close(self.descriptor)
            self.descriptor = None
        elif getattr(self, "handle", None):
            if not self.kernel.CloseHandle(self.handle):
                raise RuntimeError("STOP_SOURCE_HANDLE_CLOSE_FAILED:" + str(ctypes.get_last_error()))
        self.handle = None


class _CoreSourceCustody:
    """Read-locked core snapshot; injected readers are private fixture mechanics."""
    def __init__(self, root, pins, *, guardian=None, opener=_WindowsReadOnlyFile, inventory=None, fixture_only=False):
        self.root, self.pins, self.readers = Path(root), dict(pins), {}
        self.fixture_only, self.closed, self.weight_bytes_hashed = fixture_only, False, 0
        self.guardian = guardian
        if fixture_only:
            if opener is _WindowsReadOnlyFile or inventory is None or _same_path(str(self.root), str(registry.SOURCE_CORE)):
                raise ValueError("STOP_FIXTURE_SOURCE_CUSTODY_MUST_USE_PRIVATE_READERS")
        else:
            from . import native_safety
            if (type(guardian) is not native_safety.Guardian or guardian.worker_mode is not True
                    or not schema.typed_equal(pins, guardian.bundle["proposal"].get("source_sidecar_sha256"))):
                raise ValueError("STOP_SOURCE_CUSTODY_NEEDS_ACTUAL_GUARDIAN")
            native_safety.validate_worker_admission(guardian.admission)
            guardian.check(stage="before_core_source_custody")
        self.inventory = sorted(inventory) if inventory is not None else sorted(p.name for p in self.root.iterdir() if p.is_file())
        if not fixture_only and (opener is not _WindowsReadOnlyFile or self.root.resolve() != registry.SOURCE_CORE.resolve()):
            raise ValueError("STOP_INJECTED_SOURCE_CUSTODY_IS_NOT_PRODUCTION")
        if set(pins) != set(custody.SIDECARS) or any(not _digest_string(value) for value in pins.values()):
            raise ValueError("STOP_SOURCE_SIDECAR_PIN_SET")
        if "model.safetensors" not in self.inventory or UNPINNED_LOADER_ASSETS & set(self.inventory):
            raise ValueError("STOP_UNPINNED_CORE_LOADER_ASSET")
        if not fixture_only and any((self.root / name).exists() for name in ("chat_templates", "additional_chat_templates")):
            raise ValueError("STOP_UNPINNED_NESTED_CORE_CHAT_TEMPLATE")
        try:
            self.directory_reader = opener(self.root, directory=True)
            for name in self.inventory:
                self.readers[name] = opener(self.root / name)
            self.verify()
        except BaseException:
            self.close()
            raise

    def _sidecar_bytes(self, name):
        return b"".join(self.readers[name].read_chunks())

    def verify(self):
        if self.closed:
            raise RuntimeError("STOP_CORE_SOURCE_CUSTODY_ALREADY_CLOSED")
        self.directory_reader.verify()
        if not self.fixture_only and sorted(p.name for p in self.root.iterdir() if p.is_file()) != self.inventory:
            raise RuntimeError("STOP_CORE_SOURCE_INVENTORY_CHANGED")
        if not self.fixture_only and any((self.root / name).exists() for name in ("chat_templates", "additional_chat_templates")):
            raise RuntimeError("STOP_UNPINNED_NESTED_CORE_CHAT_TEMPLATE")
        for reader in self.readers.values():
            reader.verify()
        for name, pin in self.pins.items():
            if hashlib.sha256(self._sidecar_bytes(name)).hexdigest() != pin:
                raise ValueError("STOP_PROTECTED_SOURCE_SIDECAR_DIGEST:" + name)
        return {"asset_names": list(self.inventory), "sidecar_sha256": dict(self.pins),
                "read_only_write_delete_denied": not self.fixture_only, "fixture_only": self.fixture_only,
                "fresh_weight_digest_verified": False}

    def weight_digest(self, check, *, expected_bytes=registry.SOURCE_WEIGHT_BYTES, expected_sha256=registry.SOURCE_WEIGHTS_SHA256):
        if not self.fixture_only and (expected_bytes != registry.SOURCE_WEIGHT_BYTES or expected_sha256 != registry.SOURCE_WEIGHTS_SHA256):
            raise ValueError("STOP_PRODUCTION_WEIGHT_PIN_OVERRIDE")
        if not self.fixture_only:
            # A supplied lambda cannot bypass the actual live production gate.
            check = lambda phase: self.guardian.check(stage=phase)
        check("before_source_hash")
        self.verify()
        sha, total = hashlib.sha256(), 0
        for index, chunk in enumerate(self.readers["model.safetensors"].read_chunks(), 1):
            sha.update(chunk)
            total += len(chunk)
            self.weight_bytes_hashed = total
            if index % 8 == 0:
                check("source_hash_progress")
        if total != expected_bytes or sha.hexdigest() != expected_sha256:
            raise ValueError("STOP_QUALIFICATION_FRESH_SOURCE_HASH")
        self.verify()
        check("after_source_hash")
        return sha.hexdigest()

    def close(self):
        if self.closed:
            return
        errors = []
        for reader in reversed(list(self.readers.values())):
            try:
                reader.close()
            except BaseException as error:
                errors.append(repr(error))
        if getattr(self, "directory_reader", None) is not None:
            try:
                self.directory_reader.close()
            except BaseException as error:
                errors.append(repr(error))
        if errors:
            raise RuntimeError("STOP_SOURCE_CUSTODY_CLOSE_ERRORS:" + ";".join(errors))
        self.closed = True


def _digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _same_path(left, right):
    return PureWindowsPath(left).as_posix().casefold() == PureWindowsPath(right).as_posix().casefold()


def proposal_payload_digest(proposal):
    return schema.digest({key: value for key, value in proposal.items() if key not in PROPOSAL_DIGEST_EXCLUSIONS})


def validate_admission(admission, proposal=None):
    """Pure structural check; this cannot issue a permit or replace live admission."""
    if not isinstance(admission, dict) or admission.get("experiment") != schema.EXPERIMENT or admission.get("stage") != STAGE:
        raise ValueError("STOP_QUALIFICATION_EXPERIMENT_OR_STAGE")
    if admission.get("fixture_only") or admission.get("fixture_mode"):
        raise ValueError("STOP_FIXTURE_IS_NOT_NATIVE_ADMISSION")
    if admission.get("source_checkpoint_sha256") != registry.SOURCE_WEIGHTS_SHA256:
        raise ValueError("STOP_QUALIFICATION_SOURCE_BINDING")
    if admission.get("registry_sha256") != registry.build_registry()["registry_sha256"]:
        raise ValueError("STOP_QUALIFICATION_REGISTRY_BINDING")
    for key in ("source_manifest_sha256", "proposal_payload_sha256", "proposal_file_sha256", "permit_file_sha256"):
        if not _digest_string(admission.get(key)):
            raise ValueError("STOP_QUALIFICATION_DIGEST_FIELD:" + key)
    for key in ("attempt_id", "hardware_slot", "runtime_root", "attempt_dir", "proposal_path", "permit_path"):
        if not isinstance(admission.get(key), str) or not admission[key].strip():
            raise ValueError("STOP_QUALIFICATION_EMPTY_FIELD:" + key)
    runtime = custody.ARTIFACT_ROOT / "native-qualification"
    if not _same_path(admission["runtime_root"], str(runtime)):
        raise ValueError("STOP_QUALIFICATION_RUNTIME_ROOT")
    root, attempt = PureWindowsPath(admission["runtime_root"]), PureWindowsPath(admission["attempt_dir"])
    if attempt == root or root not in attempt.parents or ".." in attempt.parts:
        raise ValueError("STOP_QUALIFICATION_ATTEMPT_DIRECTORY")
    for key in ("job_assigned_before_payload", "guardian_admission_passed"):
        if admission.get(key) is not True:
            raise ValueError("STOP_QUALIFICATION_MISSING_LIVE_PROOF:" + key)
    for key in ("worker_pid", "supervisor_pid"):
        if type(admission.get(key)) is not int or admission[key] <= 0:
            raise ValueError("STOP_QUALIFICATION_PID:" + key)
    for key in ("worker_create_time", "supervisor_create_time"):
        value = admission.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("STOP_QUALIFICATION_PROCESS_IDENTITY:" + key)
    caps = {"supervisor_hard_wall_seconds": 600, "worker_wall_seconds": 540,
            "cleanup_wall_seconds": 120, "declared_attempt_cost_seconds": 720}
    if any(type(admission.get(key)) is not int or admission[key] != value for key, value in caps.items()):
        raise ValueError("STOP_QUALIFICATION_TIME_ENVELOPE")
    limits = {"host_admission_gib": 12, "gpu_admission_gib": 6, "disk_admission_gib": 20,
              "host_reserve_gib": 1.5, "gpu_reserve_gib": .5, "tree_rss_max_gib": 10,
              "output_bytes_max": 6 * 2 ** 30}
    actual = admission.get("limits")
    if not isinstance(actual, dict) or any(type(actual.get(k)) is not type(v) or actual[k] != v for k, v in limits.items()):
        raise ValueError("STOP_QUALIFICATION_RESOURCE_ENVELOPE")
    if proposal is not None:
        validate_proposal(proposal)
        if proposal_payload_digest(proposal) != admission["proposal_payload_sha256"]:
            raise ValueError("STOP_QUALIFICATION_PROPOSAL_BINDING")
        if proposal["source_manifest_sha256"] != admission["source_manifest_sha256"]:
            raise ValueError("STOP_QUALIFICATION_MANIFEST_BINDING")
    return dict(admission)


def validate_proposal(proposal):
    if (not isinstance(proposal, dict) or proposal.get("experiment") != schema.EXPERIMENT
            or type(proposal.get("proposal_version")) is not int or proposal["proposal_version"] != 2
            or proposal.get("stage") != STAGE
            or proposal.get("native_worker_implementation_in_this_milestone") is not True
            or proposal.get("source_checkpoint_sha256") != registry.SOURCE_WEIGHTS_SHA256
            or proposal.get("registry_sha256") != registry.build_registry()["registry_sha256"]
            or proposal.get("architecture") != "Qwen3ForCausalLM" or proposal.get("dtype") != "float32"
            or proposal.get("custom_abstention") is not False):
        raise ValueError("STOP_QUALIFICATION_PROPOSAL_IDENTITY")
    pins = proposal.get("source_sidecar_sha256")
    descriptor = proposal.get("source_identity_descriptor")
    if (not isinstance(pins, dict) or set(pins) != set(custody.SIDECARS)
            or any(not _digest_string(value) for value in pins.values())
            or not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str)
            or not descriptor["path"] or not _digest_string(descriptor.get("sha256"))):
        raise ValueError("STOP_QUALIFICATION_ORIGINAL_SOURCE_ASSET_BINDING")
    a = proposal.get("attempt", {})
    ids = a.get("generation_row_ids_in_order")
    if not isinstance(ids, list) or len(ids) != 8 or len(set(ids)) != 8 or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("STOP_QUALIFICATION_EXACT_EIGHT_DEVELOPMENT_IDS")
    scalar = {"generation_rows": 8, "max_output_tokens": 64, "max_input_tokens": 1984,
              "generation_forward_requests_max": 512, "padding_check_rows": 2, "padding_check_steps": 8,
              "padding_check_forwards": 32, "disposable_optimizer_updates": 2, "optimizer_microbatches": 16,
              "teacher_forced_forward_requests": 16, "backward_requests": 16,
              "reload_readout_rows": 1, "reload_teacher_forced_forward_requests": 2,
              "registered_forward_requests_without_checkpoint_recompute": FORWARD_CAP}
    if any(type(a.get(k)) is not int or a[k] != v for k, v in scalar.items()):
        raise ValueError("STOP_QUALIFICATION_EXACT_WORKLOAD")
    if (a.get("eos_ids") != list(EOS_IDS) or any(type(x) is not int for x in a["eos_ids"])
            or a.get("optimizer_row_ids_in_order") != ids + ids
            or a.get("padding_check_row_ids_in_order") != ids[:2]
            or a.get("reload_readout_row_id") not in ids
            or not schema.typed_equal(a.get("training_defaults"), registry.TRAINING)
            or a.get("greedy") is not True or a.get("thinking") is not False
            or a.get("tf32") is not False or a.get("attention") != "eager"):
        raise ValueError("STOP_QUALIFICATION_PIPELINE_OR_TRAINING_DRIFT")
    if a.get("padding_layout", "left_singleton_vs_unpadded") != "left_singleton_vs_unpadded":
        raise ValueError("STOP_QUALIFICATION_PADDING_LAYOUT_DRIFT")
    if type(a.get("padding_input_slots", 1984)) is not int or a.get("padding_input_slots", 1984) != 1984:
        raise ValueError("STOP_QUALIFICATION_PADDING_INPUT_CAP_DRIFT")
    if not _digest_string(a.get("development_sha256")) or not isinstance(a.get("development_path"), str):
        raise ValueError("STOP_QUALIFICATION_DEVELOPMENT_BINDING")
    return proposal


def _checked_rows(rows, proposal):
    ids = proposal["attempt"]["generation_row_ids_in_order"]
    by_id = {}
    for row in rows:
        schema.validate_row(row)
        if row["split"] != "dev" or row["row_id"] in by_id:
            raise ValueError("STOP_QUALIFICATION_DEVELOPMENT_ONLY_UNIQUE_ROWS")
        by_id[row["row_id"]] = row
    if any(row_id not in by_id for row_id in ids):
        raise ValueError("STOP_QUALIFICATION_ROW_MISSING")
    selected = [by_id[row_id] for row_id in ids]
    if [r["family"] for r in selected] != [family for family in schema.REPAIR_FAMILIES for _ in range(2)]:
        raise ValueError("STOP_QUALIFICATION_FIXED_FAMILY_ORDER")
    if any(r["variant_id"] != "canonical" for r in selected):
        raise ValueError("STOP_QUALIFICATION_CANONICAL_ROWS_ONLY")
    for row in selected:
        oracles.verify_gold(row)
    return selected


def _run_protocol(framework, rows, proposal, *, check, mark, fixture_only=True):
    """Dependency-injected mechanics; fixtures cannot yield native qualification."""
    validate_proposal(proposal)
    selected = _checked_rows(rows, proposal)
    started = time.monotonic()
    counts = {"generation": 0, "padding": 0, "teacher_forced": 0, "reload_readout": 0,
              "backward_requests": 0, "optimizer_updates": 0}
    timings, generation = {}, []
    def phase(name, fn):
        check(name)
        framework.stage = name
        framework.verify_source_assets()
        framework.synchronize()
        stamp = time.monotonic()
        value = fn()
        framework.synchronize()
        timings[name] = time.monotonic() - stamp
        framework.verify_source_assets()
        return value
    mark("LOAD_BEGUN")
    phase("load_source", framework.load_source)
    encoded = phase("full_hf_serialization", lambda: framework.serialize(selected))
    if len(encoded) != 8 or [e["row_id"] for e in encoded] != [r["row_id"] for r in selected]:
        raise ValueError("STOP_QUALIFICATION_ENCODED_ROW_MEMBERSHIP")
    for e in encoded:
        if (e.get("full_hf_equivalence") is not True or type(e.get("loss_tokens")) is not int
                or e["loss_tokens"] <= 0 or e.get("final_eos_label_positive") is not True):
            raise ValueError("STOP_QUALIFICATION_FULL_HF_LABEL_MASK")
    for item in encoded:
        result = phase("generate:" + item["row_id"], lambda item=item: framework.generate(item))
        n = result.get("forward_requests")
        if type(n) is not int or not 1 <= n <= 64 or result.get("greedy") is not True or result.get("eos_ids") != list(EOS_IDS):
            raise ValueError("STOP_QUALIFICATION_GENERATION_ACCOUNTING")
        counts["generation"] += n
        generation.append(result)
    for item in encoded[:2]:
        result = phase("padding:" + item["row_id"], lambda item=item: framework.padding_probe(item, steps=8))
        if result.get("forward_requests") != 16 or result.get("layout") != "left_singleton_vs_unpadded" or result.get("parity") is not True:
            raise ValueError("STOP_QUALIFICATION_PADDING_PARITY_OR_ACCOUNTING")
        counts["padding"] += 16
    phase("install_lora", framework.install_lora)
    for index in range(16):
        result = phase("microbatch:" + str(index), lambda index=index: framework.microbatch(encoded[index % 8], accumulation=8))
        if (result.get("forward_requests") != 1 or result.get("backward_requests") != 1
                or result.get("finite_loss") is not True or result.get("positive_labels") is not True):
            raise ValueError("STOP_QUALIFICATION_NONFINITE_LOSS_OR_MICROBATCH_ACCOUNTING")
        counts["teacher_forced"] += 1
        counts["backward_requests"] += 1
        if (index + 1) % 8 == 0:
            step = phase("optimizer_update:" + str(index // 8), framework.optimizer_step)
            if step.get("finite_gradients") is not True or not math.isfinite(step.get("gradient_norm", float("nan"))) or step["gradient_norm"] <= 0:
                raise ValueError("STOP_QUALIFICATION_NONFINITE_OR_ZERO_GRADIENTS")
            counts["optimizer_updates"] += 1
    phase("merge_fp32_dense", framework.merge_dense)
    readout_id = proposal["attempt"]["reload_readout_row_id"]
    readout = next(e for e in encoded if e["row_id"] == readout_id)
    before = phase("readout_before_export", lambda: framework.readout(readout))
    counts["reload_readout"] += 1
    phase("export_dense", framework.export_dense)
    phase("release_dense_before_reload", framework.release_dense_before_reload)
    mark("DENSE_RELOAD_BEGUN")
    phase("reload_dense", framework.reload_dense)
    after = phase("readout_after_reload", lambda: framework.readout(readout))
    counts["reload_readout"] += 1
    parity = framework.readout_equal(before, after)
    if parity is not True:
        raise ValueError("STOP_QUALIFICATION_RELOADED_FP32_READOUT_PARITY")
    actual = sum(counts[key] for key in ("generation", "padding", "teacher_forced", "reload_readout"))
    if actual > FORWARD_CAP or counts["backward_requests"] != 16 or counts["optimizer_updates"] != 2:
        raise ValueError("STOP_QUALIFICATION_REGISTERED_WORK_CAP")
    recompute = framework.recompute_accounting()
    if (any(type(recompute.get(key)) is not int or recompute[key] < 0 for key in
            ("model_forward_requests_during_backward", "decoder_layer_forward_requests_during_backward"))):
        raise ValueError("STOP_QUALIFICATION_UNMEASURED_RECOMPUTATION")
    check("protocol_complete")
    return {"experiment": schema.EXPERIMENT, "stage": STAGE, "fixture_only": fixture_only,
            "mechanics_passed": True, "native_mechanics_passed": not fixture_only,
            "native_qualification_passed": False,
            "capability_evidence": False, "parent_qualification": False, "training_sweep_authorized": False,
            "final_evaluation_authorized": False, "promotion_authorized": False,
            "source_model_loads": 1, "dense_reload_loads": 1, "phase_forward_requests": counts,
            "forward_requests_without_checkpoint_recompute": actual, "registered_forward_cap": FORWARD_CAP,
            "checkpoint_recompute": recompute, "generation_readouts": generation,
            "reload_readout_exact_parity": True, "stage_wall_seconds": timings,
            "worker_elapsed_seconds": time.monotonic() - started,
            "stage_timing": "FIXTURE_STUB_NOT_DEVICE_TIMING" if fixture_only else "CUDA_SYNCHRONIZED_AT_PHASE_BOUNDARIES_ONCE_INITIALIZED",
            "retained_adapter_or_optimizer_for_trials": False,
            "full_sweep_throughput_admission": "SUPERVISOR_SOURCE_BOUND_TIME_BOUND_AND_CLEANUP_REQUIRED",
            "right_padding_qualified": False}


def _original_source_identity(proposal):
    descriptor = proposal["source_identity_descriptor"]
    path = custody.external(descriptor["path"])
    identity = json.loads(custody.read_pinned(path, descriptor["sha256"]))
    if (identity.get("declared_weight_sha256") != registry.SOURCE_WEIGHTS_SHA256
            or identity.get("declared_weight_bytes") != registry.SOURCE_WEIGHT_BYTES
            or identity.get("architecture") != "Qwen3ForCausalLM" or identity.get("storage_dtype") != "float32"
            or not _same_path(identity.get("core_directory", ""), str(registry.SOURCE_CORE))
            or not schema.typed_equal(identity.get("sidecar_sha256"), proposal["source_sidecar_sha256"])):
        raise ValueError("STOP_ORIGINAL_SOURCE_IDENTITY_DESCRIPTOR")
    return identity


def qualify(admission, *, guardian_check, framework_factory=None):
    """Production entry. A fake fixture callback or framework is never admitted."""
    worker_started = time.monotonic()
    if framework_factory is not None:
        raise ValueError("STOP_INJECTED_FRAMEWORK_IS_NOT_PRODUCTION_QUALIFICATION")
    from . import native_safety
    guardian = getattr(guardian_check, "__self__", None)
    if (type(guardian) is not native_safety.Guardian or guardian.worker_mode is not True
            or getattr(guardian_check, "__func__", None) is not native_safety.Guardian.check):
        raise ValueError("STOP_QUALIFICATION_REQUIRES_LIVE_BOUND_GUARDIAN")
    validate_admission(admission)
    native_safety.validate_worker_admission(admission)
    if not schema.typed_equal(admission, guardian.admission):
        raise ValueError("STOP_QUALIFICATION_ACTUAL_START_ADMISSION_BINDING")
    def check(phase):
        proof = guardian_check(stage=phase)
        if not isinstance(proof, dict) or proof.get("passed") is not True:
            raise ValueError("STOP_QUALIFICATION_LIVE_GUARDIAN_REJECTED")
        bindings = {"experiment": schema.EXPERIMENT, "source_weight_sha256": registry.SOURCE_WEIGHTS_SHA256,
                    "proposal_payload_sha256": admission["proposal_payload_sha256"],
                    "source_manifest_sha256": admission["source_manifest_sha256"],
                    "attempt_id": admission["attempt_id"], "hardware_slot": admission["hardware_slot"],
                    "worker_pid": admission["worker_pid"], "worker_create_time": admission["worker_create_time"]}
        if any(not schema.typed_equal(proof.get(key), value) for key, value in bindings.items()):
            raise ValueError("STOP_QUALIFICATION_GUARDIAN_IDENTITY_BINDING")
        if any(proof.get(key) is not True for key in ("job_bound", "leases_owned", "permit_verified")):
            raise ValueError("STOP_QUALIFICATION_GUARDIAN_OWNERSHIP")
    # Only after this positive live proof may the worker write an owned failure
    # receipt or open core assets. Invalid/fake admissions remain the supervisor's
    # blocked dispatch record; they cannot supply an arbitrary output directory.
    check("before_source_and_pipeline_identity")
    framework, setup_stage = None, "proposal_identity"
    try:
        proposal = json.loads(custody.read_pinned(admission["proposal_path"], admission["proposal_file_sha256"]))
        validate_admission(admission, proposal)
        if (schema.digest(custody.source_manifest()) != admission["source_manifest_sha256"]
                or native_safety.proposal_payload_digest(proposal) != proposal_payload_digest(proposal)):
            raise ValueError("STOP_QUALIFICATION_CURRENT_SOURCE_OR_PROPOSAL_BYTES")
        _original_source_identity(proposal)
        setup_stage = "development_identity"
        a = proposal["attempt"]
        development_path = custody.external(a["development_path"])
        data = custody.read_pinned(development_path, a["development_sha256"])
        # This prospectively compiled DEV JSONL is the only bank opened here.
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        _checked_rows(rows, proposal)
        setup_stage = "framework_source_custody_hash_and_import"
        check("before_framework_creation")
        framework = TorchFramework(admission, proposal, guardian)
        fresh = framework.fresh_source_weight_sha256
        guardian.bind_local_resource_check(framework.resource_snapshot)
        def mark(event):
            check("before_" + event.lower())
            framework.verify_source_assets()
            custody.write_json(Path(admission["attempt_dir"]) / (event + ".json"),
                {"experiment": schema.EXPERIMENT, "stage": STAGE, "event": event,
                 "attempt_id": admission["attempt_id"], "hardware_slot": admission["hardware_slot"],
                 "source_checkpoint_sha256": fresh, "source_manifest_sha256": admission["source_manifest_sha256"],
                 "proposal_payload_sha256": admission["proposal_payload_sha256"],
                 "worker_pid": admission["worker_pid"], "worker_create_time": admission["worker_create_time"],
                 "lease_token": admission["lease_token"],
                 "source_model_loads": 1, "dense_reload_loads": int(event == "DENSE_RELOAD_BEGUN")})
        setup_stage = "qualified_protocol"
        result = _run_protocol(framework, rows, proposal, check=check, mark=mark, fixture_only=False)
        result.update(fresh_source_weight_sha256=fresh, attempt_id=admission["attempt_id"],
                      hardware_slot=admission["hardware_slot"], source_manifest_sha256=admission["source_manifest_sha256"],
                      proposal_payload_sha256=admission["proposal_payload_sha256"],
                      source_hash_and_framework_setup_seconds=framework.setup_seconds,
                      actual_request_accounting=framework.accounting_snapshot(),
                      source_asset_custody=framework.verify_source_assets())
        setup_stage = "framework_and_source_custody_cleanup"
        framework.close()
        result["source_asset_read_handles_closed"] = True
        result["worker_elapsed_seconds_including_setup"] = time.monotonic() - worker_started
        custody.write_json(Path(admission["attempt_dir"]) / "NATIVE_QUALIFICATION_RESULT.json", result)
        return result
    except BaseException as error:
        accounting = framework.accounting_snapshot() if framework is not None else {
            "model_forward_requests_without_recompute": 0, "backward_requests": 0,
            "optimizer_requests": 0, "source_model_loader_not_entered": True}
        cleanup_error = getattr(error, "source_custody_cleanup_error", None)
        if framework is not None:
            try:
                framework.close()
            except BaseException as closing_error:
                cleanup_error = repr(closing_error)
        # Includes constructor/source-hash/import failures before LOAD_BEGUN.
        # The parent charged 720 seconds before dispatch, with no retry/refund.
        custody.write_json(Path(admission["attempt_dir"]) / "NATIVE_QUALIFICATION_FAILURE.json",
            {"experiment": schema.EXPERIMENT, "attempt_id": admission["attempt_id"],
             "status": "INCOMPLETE_PRELOAD" if framework is None else "INCOMPLETE_NATIVE_PROTOCOL",
             "setup_stage": setup_stage,
             "proposal_payload_sha256": admission["proposal_payload_sha256"],
             "source_manifest_sha256": admission["source_manifest_sha256"],
             "worker_pid": admission["worker_pid"], "worker_create_time": admission["worker_create_time"],
             "error_type": type(error).__name__, "error": str(error)[:2000],
             "actual_request_accounting": accounting,
             "source_model_load_begun_marker_present": (Path(admission["attempt_dir"]) / "LOAD_BEGUN.json").exists(),
             "source_custody_cleanup_error": cleanup_error,
             "worker_elapsed_seconds_including_setup": time.monotonic() - worker_started,
             "declared_attempt_cost_seconds": 720, "native_qualification_passed": False,
             "capability_evidence": False})
        raise


class TorchFramework:
    """Real implementations, instantiated only after production live admission/hash."""
    def __init__(self, admission, proposal, guardian):
        # Direct construction is guarded too. An arbitrary caller cannot make
        # importing this adapter a substitute for an admitted native entry.
        from . import native_safety
        if (type(guardian) is not native_safety.Guardian or guardian.worker_mode is not True
                or not schema.typed_equal(guardian.admission, admission)):
            raise ValueError("STOP_TORCH_FRAMEWORK_NEEDS_ACTUAL_GUARDIAN")
        validate_admission(admission, proposal)
        native_safety.validate_worker_admission(admission)
        guardian.check(stage="before_framework_source_hash")
        started = time.monotonic()
        self.admission, self.proposal, self.guardian = admission, proposal, guardian
        self.check = lambda phase: guardian.check(stage=phase)
        self.torch, self.core_custody = None, None
        self.model, self.optimizer, self.tokenizer = None, None, None
        self.model_forwards, self.backward_active = 0, False
        self.recompute_model, self.recompute_layers = 0, 0
        self.model_completed, self.recompute_layers_completed = 0, 0
        self.backward_requested, self.backward_completed = 0, 0
        self.optimizer_requested, self.optimizer_completed = 0, 0
        self.stage, self.stage_requests = "framework_created", {}
        self.hooks = []
        self.export = Path(admission["attempt_dir"]) / "disposable-fp32-dense"
        try:
            _original_source_identity(proposal)
            self.core_custody = _CoreSourceCustody(registry.SOURCE_CORE, proposal["source_sidecar_sha256"], guardian=guardian)
            self.fresh_source_weight_sha256 = self.core_custody.weight_digest(self.check)
            environment = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false",
                           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            if any(os.environ.get(key) != value for key, value in environment.items()):
                raise ValueError("STOP_QUALIFICATION_UNBOUND_DETERMINISTIC_OR_OFFLINE_ENVIRONMENT")
            self.verify_source_assets()
            guardian.check(stage="before_framework_import")
            import torch
            from transformers import AutoTokenizer, Qwen3ForCausalLM
            from peft import LoraConfig, get_peft_model
            self.torch, self.AutoTokenizer, self.Model = torch, AutoTokenizer, Qwen3ForCausalLM
            self.LoraConfig, self.get_peft_model = LoraConfig, get_peft_model
            self.verify_source_assets()
            self.setup_seconds = time.monotonic() - started
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.source_custody_cleanup_error = repr(cleanup)
            raise

    def verify_source_assets(self):
        if self.core_custody is None:
            raise RuntimeError("STOP_NATIVE_SOURCE_CUSTODY_MISSING")
        value = self.core_custody.verify()
        value["fresh_weight_digest_verified"] = hasattr(self, "fresh_source_weight_sha256")
        value["weight_bytes_hashed_from_same_held_handle"] = self.core_custody.weight_bytes_hashed
        return value

    def synchronize(self):
        if self.torch is not None and self.torch.cuda.is_initialized():
            self.torch.cuda.synchronize(0)

    def close(self):
        # Source handles outlive all model/optimizer references and queued work.
        # Even a failed CUDA cleanup closes protected source handles; the parent
        # still verifies whole-Job death and charges/retains incomplete outcomes.
        try:
            self.optimizer, self.model, self.tokenizer = None, None, None
            self.trainable = []
            for hook in self.hooks:
                hook.remove()
            self.hooks = []
            gc.collect()
            self.synchronize()
            if self.torch is not None and self.torch.cuda.is_initialized():
                self.torch.cuda.empty_cache()
        finally:
            if self.core_custody is not None:
                self.core_custody.close()

    def _identity(self):
        if (self.model.__class__ is not self.Model
                or any(p.is_floating_point() and p.dtype != self.torch.float32 for p in self.model.state_dict().values())
                or any(p.device.type != "cuda" or p.device.index != 0 for p in self.model.parameters())):
            raise ValueError("STOP_QUALIFICATION_STANDARD_FP32_MODEL_IDENTITY")
        if (self.model.config.model_type != "qwen3" or getattr(self.model.config, "abstain_logit_bias", 0) != 0
                or getattr(self.model.config, "quantization_config", None) is not None
                or any("lora_" in name for name in self.model.state_dict())):
            raise ValueError("STOP_QUALIFICATION_ROUTER_OFFSET_QUANT_OR_ADAPTER_RESIDUE")

    def _hooks(self):
        def model_hook(*args):
            if self.backward_active:
                self.recompute_model += 1
            else:
                self.model_forwards += 1
            self.stage_requests[self.stage] = self.stage_requests.get(self.stage, 0) + 1
            self.check("each_forward")
        def completed_hook(*args):
            self.model_completed += 1
        def layer_hook(*args):
            if self.backward_active:
                self.recompute_layers += 1
                self.check("checkpoint_recompute_layer")
        def layer_completed_hook(*args):
            if self.backward_active:
                self.recompute_layers_completed += 1
        self.hooks.append(self.model.register_forward_pre_hook(model_hook))
        self.hooks.append(self.model.register_forward_hook(completed_hook))
        self.hooks.extend(layer.register_forward_pre_hook(layer_hook) for layer in self.model.model.layers)
        self.hooks.extend(layer.register_forward_hook(layer_completed_hook) for layer in self.model.model.layers)

    def load_source(self):
        t = self.torch
        t.set_num_threads(1)
        t.manual_seed(17)
        t.cuda.manual_seed_all(17)
        t.use_deterministic_algorithms(True)
        t.backends.cuda.matmul.allow_tf32 = False
        t.backends.cudnn.allow_tf32 = False
        t.cuda.set_per_process_memory_fraction(.78, 0)
        self.model = self.Model.from_pretrained(str(registry.SOURCE_CORE), torch_dtype=t.float32,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True, attn_implementation="eager",
            local_files_only=True, trust_remote_code=False)
        self.model.eval()
        self._identity()
        self._hooks()

    def serialize(self, rows):
        self.tokenizer = self.AutoTokenizer.from_pretrained(str(registry.SOURCE_CORE), local_files_only=True, trust_remote_code=False)
        hf = self.tokenizer
        class HFAdapter:
            truncation, padding = None, None
            def encode(self, text, add_special_tokens=False):
                return hf.encode(text, add_special_tokens=add_special_tokens)
            def token_to_id(self, text):
                return hf.convert_tokens_to_ids(text)
        encoded = []
        for row in rows:
            messages = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}]
            prefix = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            expected_prefix = token_audit.render_prompt(row["system"], row["user"])
            target = generator.target_reply(row)
            full = hf.apply_chat_template(messages + [{"role": "assistant", "content": target, "reasoning_content": ""}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False)
            if prefix != expected_prefix or full != prefix + target + token_audit.IM_END + "\n":
                raise ValueError("STOP_QUALIFICATION_FULL_HF_TEMPLATE_BYTE_MISMATCH")
            value = token_audit.encode_training_row(row, HFAdapter())
            # Exclude exactly the HF template's terminal record-separator newline;
            # target and positive final EOS must remain intact and byte-identical.
            if hf.encode(full[:-1], add_special_tokens=False) != value["input_ids"]:
                raise ValueError("STOP_QUALIFICATION_FULL_HF_TOKEN_OR_MASK_MISMATCH")
            value["full_hf_equivalence"] = True
            encoded.append(value)
        return encoded

    def _batch(self, item, *, prompt=False):
        ids = item["input_ids"][:item["prompt_tokens"]] if prompt else item["input_ids"]
        inputs = self.torch.tensor([ids], dtype=self.torch.long, device="cuda:0")
        values = {"input_ids": inputs, "attention_mask": self.torch.ones_like(inputs)}
        if not prompt:
            values["labels"] = self.torch.tensor([item["labels"]], dtype=self.torch.long, device="cuda:0")
        return values

    def generate(self, item):
        before = self.model_forwards
        with self.torch.inference_mode():
            answer = self.model.generate(**self._batch(item, prompt=True), do_sample=False,
                max_new_tokens=64, eos_token_id=list(EOS_IDS), pad_token_id=EOS_IDS[1], use_cache=True)
        tokens = answer[0, item["prompt_tokens"]:].detach().cpu().tolist()
        return {"row_id": item["row_id"], "forward_requests": self.model_forwards - before,
                "generated_tokens": len(tokens), "output_token_sha256": schema.digest(tokens),
                "greedy": True, "eos_ids": list(EOS_IDS), "capability_evidence": False}

    def padding_probe(self, item, steps):
        before = self.model_forwards
        def decode(padding):
            ids = item["input_ids"][:item["prompt_tokens"]]
            inputs = self.torch.tensor([[EOS_IDS[1]] * padding + ids], dtype=self.torch.long, device="cuda:0")
            mask = self.torch.tensor([[0] * padding + [1] * len(ids)], dtype=self.torch.long, device="cuda:0")
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            past, output = None, []
            with self.torch.inference_mode():
                for index in range(steps):
                    result = self.model(input_ids=inputs, attention_mask=mask, position_ids=positions,
                                        past_key_values=past, use_cache=True)
                    token = result.logits[:, -1, :].argmax(-1, keepdim=True)
                    output.append(int(token.item()))
                    past, inputs = result.past_key_values, token
                    mask = self.torch.cat((mask, self.torch.ones((1, 1), dtype=mask.dtype, device=mask.device)), dim=1)
                    positions = self.torch.tensor([[len(ids) + index]], dtype=self.torch.long, device="cuda:0")
            return output
        padding = 1984 - item["prompt_tokens"]
        if padding <= 0:
            raise ValueError("STOP_QUALIFICATION_PADDING_PROBE_NEEDS_NONEMPTY_PADDING")
        plain, padded = decode(0), decode(padding)
        return {"forward_requests": self.model_forwards - before, "layout": "left_singleton_vs_unpadded",
                "parity": plain == padded, "steps": steps, "left_padding_tokens": padding,
                "padding_input_slots": 1984, "long_context_quality_evidence": False}

    def install_lora(self):
        t, defaults = self.torch, registry.TRAINING
        config = self.LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
            target_modules=list(registry.LORA_TARGETS), bias="none", task_type="CAUSAL_LM")
        self.model = self.get_peft_model(self.model, config)
        self.model.train()
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": True})
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not self.trainable or any(p.dtype != t.float32 for p in self.trainable):
            raise ValueError("STOP_QUALIFICATION_TRAINABLE_LORA_FP32")
        if any(p.requires_grad and "lora_" not in name for name, p in self.model.named_parameters()):
            raise ValueError("STOP_QUALIFICATION_BASE_NOT_FROZEN")
        targets = {target: 0 for target in registry.LORA_TARGETS}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                for target in targets:
                    if "." + target + ".lora_" in name:
                        targets[target] += 1
        if any(count == 0 for count in targets.values()):
            raise ValueError("STOP_QUALIFICATION_MISSING_REGISTERED_LORA_PROJECTION")
        self.trainable_projection_counts = targets
        self.gradient_audits = []
        self.optimizer = t.optim.AdamW(self.trainable, lr=defaults["learning_rate"],
            betas=tuple(defaults["adamw_betas"]), eps=defaults["adamw_eps"], weight_decay=0, foreach=False, fused=False)
        self.optimizer.zero_grad(set_to_none=True)

    def microbatch(self, item, accumulation):
        before = self.model_forwards
        result = self.model(**self._batch(item), use_cache=False)
        finite = bool(self.torch.isfinite(result.loss).item())
        if not finite:
            raise ValueError("STOP_QUALIFICATION_NONFINITE_LOSS")
        self.backward_active = True
        self.backward_requested += 1
        try:
            (result.loss / accumulation).backward()
            self.backward_completed += 1
        finally:
            self.backward_active = False
        return {"forward_requests": self.model_forwards - before, "backward_requests": 1,
                "finite_loss": finite, "positive_labels": item["loss_tokens"] > 0}

    def optimizer_step(self):
        grads = [p.grad for p in self.trainable if p.grad is not None]
        finite = bool(grads) and all(bool(self.torch.isfinite(g).all().item()) for g in grads)
        if not finite:
            raise ValueError("STOP_QUALIFICATION_NONFINITE_GRADIENT")
        if len(grads) != len(self.trainable):
            raise ValueError("STOP_QUALIFICATION_MISSING_TRAINABLE_GRADIENT")
        norm = self.torch.nn.utils.clip_grad_norm_(self.trainable, 1.0, error_if_nonfinite=True)
        value = float(norm.item())
        if not math.isfinite(value) or value <= 0:
            raise ValueError("STOP_QUALIFICATION_ZERO_OR_NONFINITE_GRADIENT_NORM")
        self.gradient_audits.append({"trainable_parameter_tensors": len(self.trainable),
            "gradient_parameter_tensors": len(grads), "gradient_none_tensors": len(self.trainable) - len(grads),
            "zero_gradient_tensors": sum(int(not bool(self.torch.count_nonzero(g).item())) for g in grads),
            "all_observed_gradients_finite": finite, "aggregate_gradient_norm": value})
        self.optimizer_requested += 1
        self.optimizer.step()
        self.optimizer_completed += 1
        self.optimizer.zero_grad(set_to_none=True)
        return {"finite_gradients": True, "gradient_norm": value}

    def merge_dense(self):
        self.optimizer = None
        self.trainable = []
        self.model.gradient_checkpointing_disable()
        self.model = self.model.merge_and_unload(safe_merge=True)
        self.model.config.use_cache = True
        self.model.eval()
        self._identity()

    def readout(self, item):
        before = self.model_forwards
        values = self._batch(item)
        values.pop("labels")
        with self.torch.inference_mode():
            output = self.model(**values, use_cache=False).logits.detach().cpu().clone()
        if self.model_forwards - before != 1 or not bool(self.torch.isfinite(output).all().item()):
            raise ValueError("STOP_QUALIFICATION_READOUT_FORWARD_OR_FINITE")
        return output

    def export_dense(self):
        custody.external(self.export, create=True)
        self.model.save_pretrained(str(self.export), safe_serialization=True)
        self.tokenizer.save_pretrained(str(self.export))

    def release_dense_before_reload(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.model = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def reload_dense(self):
        self.model = self.Model.from_pretrained(str(self.export), torch_dtype=self.torch.float32,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True, attn_implementation="eager", local_files_only=True, trust_remote_code=False)
        self.model.eval()
        self._identity()
        self._hooks()

    def readout_equal(self, left, right):
        return bool(self.torch.equal(left, right))

    def recompute_accounting(self):
        return {"model_forward_requests_during_backward": self.recompute_model,
                "decoder_layer_forward_requests_during_backward": self.recompute_layers,
                "recompute_unit": "ACTUAL_DECODER_LAYER_INVOCATIONS_NO_GUESSED_FULL_MODEL_EQUIVALENT"}

    def accounting_snapshot(self):
        return {"model_forward_requests_without_recompute": self.model_forwards,
                "model_forward_requests_during_backward": self.recompute_model,
                "model_forward_requests_completed": self.model_completed,
                "decoder_layer_recompute_requests": self.recompute_layers,
                "decoder_layer_recompute_completed": self.recompute_layers_completed,
                "backward_requests": self.backward_requested, "backward_completed": self.backward_completed,
                "optimizer_requests": self.optimizer_requested, "optimizer_completed": self.optimizer_completed,
                "model_forward_requests_by_stage": dict(self.stage_requests), "last_stage": self.stage,
                "trainable_projection_counts": getattr(self, "trainable_projection_counts", {}),
                "optimizer_gradient_audits": getattr(self, "gradient_audits", [])}

    def resource_snapshot(self):
        # Native Guardian independently refreshes ownership inventories during
        # full unit checks and in its supervisor watchdog. Per-forward reserve
        # checks reuse those diagnostics and make no nvidia-smi subprocess.
        import psutil
        import shutil
        from . import native_safety
        ownership = getattr(self.guardian, "cached_resource_ownership", None)
        if (not isinstance(ownership, dict) or "foreign_model_processes" not in ownership
                or "gpu_compute_pids" not in ownership or "allowed_desktop_gpu_pids" not in ownership):
            raise RuntimeError("STOP_QUALIFICATION_NO_FULL_RESOURCE_OWNERSHIP_CACHE")
        members = native_safety.verify_owner_handles(self.guardian.admission, True)
        rss = 0
        for pid in members:
            try:
                rss += psutil.Process(pid).memory_info().rss
            except psutil.NoSuchProcess:
                # A member exited between exact Job enumeration and sampling;
                # native Guardian re-enumerates membership before admission.
                continue
        output_bytes = sum(path.stat().st_size for path in Path(self.admission["attempt_dir"]).rglob("*") if path.is_file())
        free, total = self.torch.cuda.mem_get_info(0)
        return {"host_available_bytes": psutil.virtual_memory().available,
                "gpu_available_bytes": int(free), "disk_available_bytes": shutil.disk_usage(self.admission["attempt_dir"]).free,
                "tree_rss_bytes": rss, "new_output_bytes": output_bytes,
                "foreign_model_processes": list(ownership["foreign_model_processes"]),
                "gpu_compute_pids": list(ownership["gpu_compute_pids"]),
                "allowed_desktop_gpu_pids": list(ownership["allowed_desktop_gpu_pids"])}


def main(argv=None):
    import argparse
    from . import native_safety
    parser = argparse.ArgumentParser(description="Guarded future disposable native qualification; requires issued operator evidence.")
    parser.add_argument("--bootstrap-gate", required=True)
    parser.add_argument("--bootstrap-gate-sha256", required=True)
    args = parser.parse_args(argv)
    admission, guardian = native_safety.bootstrap_worker(args.bootstrap_gate, args.bootstrap_gate_sha256, timeout=30)
    result = qualify(admission, guardian_check=guardian.check)
    print(json.dumps({"experiment": schema.EXPERIMENT, "attempt_id": result["attempt_id"],
                      "native_mechanics_passed": result["native_mechanics_passed"],
                      "native_qualification_passed": False, "capability_evidence": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
