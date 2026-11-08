"""Supervised, sequential Stage C capture worker; no generation or optimization.

Read-only references: MQ001 native.py's audited FP32 stock Supa loader; XSG
3059e576 contracts.py explicit block_output_raw and prompt/token coordinates;
records/qwenft001-20260918/train_qwen_lora.py exact NF4 + kbit preparation.
Donor SDPA in the current pinned runtime is a new qualification, not an assertion
of historical attention parity. Imports stay model-free until resource gates pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from . import receipts as r

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("student_base", "donor_base", "donor_skilled")
LAYERS = {"student_base": [0, 13, 27], "donor_base": [0, 17, 35], "donor_skilled": [0, 17, 35]}
GIB = 2 ** 30
CALIBRATION_FILE = Path(__file__).parent / "evidence" / "CALIBRATION_BINDING.v2.json"
ARCHITECTURE_FILE = Path(__file__).parent / "evidence" / "ARCHITECTURE_MAP.json"


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            r.require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON " + value)))


def durable_new(path, body):
    data = r.canonical_bytes(body) + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def check_lease(spec):
    lease = spec["lease"]
    path = Path(lease["path"])
    r.require(path.resolve() == (Path(tempfile.gettempdir()) / "rineya-gpu-0.lock").resolve(),
              "wrong shared GPU lease path")
    actual = read_json(path)
    r.require(isinstance(lease.get("token"), str) and bool(lease["token"]), "missing owner nonce")
    r.require(type(lease.get("owner_pid")) is int and lease["owner_pid"] > 0, "invalid lease owner")
    r.require(actual.get("token") == lease["token"] and actual.get("pid") == lease["owner_pid"],
              "shared lease ownership changed")
    deadline = datetime.fromisoformat(spec["deadline_utc"].replace("Z", "+00:00"))
    r.require(deadline.tzinfo is not None and datetime.now(timezone.utc) < deadline, "capture session expired")


def available_host_bytes():
    import psutil
    available = int(psutil.virtual_memory().available)
    for limit, usage in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if Path(limit).is_file() and Path(usage).is_file():
            text = Path(limit).read_text().strip()
            if text != "max":
                available = min(available, max(0, int(text) - int(Path(usage).read_text())))
    return available


def memory_gate(role, phase, backend=None):
    minimum_host = (7 if phase == "before_import" else 6) if role == "student_base" else 12
    if phase == "before_forward":
        minimum_host = 1.5
    available = available_host_bytes()
    r.require(available >= minimum_host * GIB, f"{role} {phase} host RAM requires {minimum_host} GiB")
    result = {"phase": phase, "available_host_bytes": available, "required_host_bytes": int(minimum_host * GIB)}
    if backend is not None:
        free = backend.free_gpu_bytes()
        need = .5 if phase == "before_forward" else (3 if role == "student_base" else 6)
        r.require(free >= need * GIB, f"{role} {phase} free GPU requires {need} GiB")
        result.update(free_gpu_bytes=free, required_gpu_bytes=int(need * GIB))
    return result


def verify_assets(asset):
    directory = Path(asset["directory"]).resolve()
    r.require(directory.is_dir() and isinstance(asset.get("files"), dict) and bool(asset["files"]), "missing asset inventory")
    for name, expected in asset["files"].items():
        relative = Path(name)
        r.require(not relative.is_absolute() and ".." not in relative.parts, "unsafe asset inventory path")
        r.verify_file(directory / relative, expected)
    if "model_sha256" in asset:
        r.require(asset["model_sha256"] == r.digest(asset["files"]), "asset inventory digest differs")
        suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
        present = {path.name for path in directory.iterdir() if path.is_file() and path.suffix in suffixes}
        expected_weights = {name for name in asset["files"] if Path(name).suffix in suffixes}
        r.require(present == expected_weights and bool(present), "missing or unexpected model weight files")
    return r.digest(asset["files"])


def validate_spec(spec, role):
    r.require(role in ROLES and spec.get("schema") == 1 and spec.get("experiment") == "CSC-001"
              and spec.get("stage") == "C", "wrong native capture scope")
    r.require(spec.get("native_capture_authorized") is True, "capture not authorized")
    r.require(spec.get("max_total_forwards") == 96 and spec.get("max_role_forwards") == 32,
              "capture envelope changed")
    r.require(isinstance(spec.get("session_id"), str) and bool(spec["session_id"]), "missing session id")
    r.valid_hash(spec.get("frozen_registration_sha256"), "frozen registration")
    r.verify_source(ROOT, spec["source_commit"], clean=True)
    inventory = r.implementation_digest(ROOT, list(spec["source_files"]))
    r.require(inventory["files"] == spec["source_files"] and inventory["sha256"] == spec["implementation_digest"],
              "implementation bytes changed")
    r.require("experiments/cross_scale_composable_001/native_capture.py" in spec["source_files"], "worker not source-bound")
    runtime = spec["runtime"]["student_base" if role == "student_base" else "donor"]
    r.require(Path(runtime["python_executable"]).resolve() == Path(sys.executable).resolve(), "wrong runtime executable")
    r.require(runtime.get("device") == "cuda:0" and runtime.get("seed") == 1729, "wrong capture device/seed")
    r.require(runtime.get("attention") == ("eager" if role == "student_base" else "sdpa"), "unregistered attention backend")
    required_packages = {"torch", "transformers", "safetensors", "numpy", "psutil"}
    if role != "student_base":
        required_packages |= {"bitsandbytes", "peft", "accelerate"}
        r.require(runtime.get("quantization") == {"type": "nf4", "double_quant": True,
                  "compute_dtype": "bfloat16", "nonquantized_dtype_after_prepare": "float32"}, "donor quantization differs")
    r.require(required_packages <= set(runtime["packages"]), "incomplete runtime pins")
    for package, expected in runtime["packages"].items():
        r.require(importlib.metadata.version(package) == expected, "runtime pin differs: " + package)
    calibration = spec["calibration"]
    rows, source_rows = calibration["rows"], calibration["source_binding_rows"]
    r.verify_file(CALIBRATION_FILE, calibration["source_binding_file_sha256"])
    original_binding = read_json(CALIBRATION_FILE)
    for key in ("calibration_sha256", "dataset_sha256"):
        r.require(calibration[key] == original_binding[key], "committed calibration binding differs: " + key)
    r.require(source_rows == original_binding["rows"], "committed calibration membership differs")
    r.verify_file(ARCHITECTURE_FILE, calibration["architecture_file_sha256"])
    architecture = read_json(ARCHITECTURE_FILE)
    r.require(calibration["layer_map_sha256"] == r.digest(architecture["layer_map"]), "committed layer map differs")
    r.require(architecture["layer_map"]["capture_pairs_student_donor"] == [[0, 0], [13, 17], [27, 35]],
              "registered capture pairs changed")
    nearest = [min(range(28), key=lambda j: (abs(i * 27 - j * 35), j)) for i in range(36)]
    r.require(architecture["layer_map"]["donor_to_student"] == nearest, "wrong normalized depth map")
    r.require(len(rows) == len(source_rows) == 32, "exactly 32 calibration prompts required")
    r.require(calibration["rows_sha256"] == r.digest(rows), "token rows digest differs")
    r.require(calibration["calibration_sha256"] == r.digest(source_rows), "original calibration digest differs")
    for key in ("dataset_sha256", "source_binding_file_sha256", "tokenizer_sha256", "prompt_positions_sha256", "layer_map_sha256"):
        r.valid_hash(calibration.get(key), key)
    r.validate_calibration({"dataset_sha256": calibration["dataset_sha256"], "role": "open_calibration",
                            "calibration_sha256": calibration["calibration_sha256"], "rows": source_rows})
    seen = set()
    for row, original in zip(rows, source_rows):
        r.require(row.get("id") not in seen, "duplicate prompt id")
        seen.add(row["id"])
        for key in ("id", "parent_id", "split", "open"):
            r.require(row.get(key) == original.get(key), "original calibration membership changed")
        r.require(row.get("messages_sha256") == original.get("prompt_sha256"), "original calibration prompt changed")
        ids = row.get("input_ids")
        r.require(isinstance(ids, list) and 0 < len(ids) <= 256
                  and all(type(token) is int and token >= 0 for token in ids), "prompt overflow/invalid tokens; no truncation")
        r.require(isinstance(row.get("prompt"), str) and bool(row["prompt"]), "missing rendered prompt")
        r.require(row.get("prompt_sha256") == r.digest(row["prompt"]) and row.get("input_ids_sha256") == r.digest(ids),
                  "prompt/token identity differs")
    r.require(sum(row["split"] == "fit" for row in rows) == 24, "fit/check allocation differs")
    coordinates = [{"id": row["id"], "input_ids_sha256": row["input_ids_sha256"],
                    "position": len(row["input_ids"]) - 1, "token_id": row["input_ids"][-1]} for row in rows]
    r.require(calibration["prompt_positions_sha256"] == r.digest(coordinates), "capture token coordinates differ")
    r.require(calibration["tokenizer_sha256"] == r.digest(spec["tokenizer"]["files"]), "tokenizer identity differs")
    check_lease(spec)
    return runtime, rows


class CaptureJournal:
    """Exclusive role and session lease, plus durable pre-forward reservations."""
    def __init__(self, spec, spec_sha256, role):
        self.root, self.role, self.spec_sha256 = Path(spec["reservation_dir"]).resolve(), role, spec_sha256
        r.require(self.root.is_dir() and not self.root.is_relative_to(ROOT), "external existing reservation directory required")
        r.valid_hash(spec.get("frozen_registration_sha256"), "frozen registration")
        binding = {"session_id": spec["session_id"], "frozen_registration_sha256": spec["frozen_registration_sha256"]}
        try:
            durable_new(self.root / "SESSION_BINDING.json", binding)
        except FileExistsError:
            r.require(read_json(self.root / "SESSION_BINDING.json") == binding, "reservation session changed")
        self.active = {**binding, "worker_spec_sha256": spec_sha256, "role": role, "pid": os.getpid()}
        durable_new(self.root / "ACTIVE.json", self.active)
        self.directory = self.root / role
        self.calls = 0
        try:
            self.directory.mkdir(exist_ok=False)
            durable_new(self.directory / "CLAIM.json", self.active)
        except BaseException:
            self.release()
            raise

    def reserve(self, row):
        r.require(self.calls < 32, "role forward budget exhausted")
        self.calls += 1
        durable_new(self.directory / f"{self.calls:03d}-RESERVED.json", {
            **self.active, "ordinal": self.calls, "row_id": row["id"], "input_ids_sha256": row["input_ids_sha256"],
            "reserved_unix": time.time(), "forwards_charged": 1})

    def complete_row(self, body):
        durable_new(self.directory / f"{self.calls:03d}-COMPLETED.json", body)

    def finish(self, status, error=None):
        durable_new(self.directory / "TERMINAL.json", {**self.active, "status": status,
                    "reserved_forwards": self.calls, "error": error, "retry_authorized": False})

    def release(self):
        path = self.root / "ACTIVE.json"
        if path.exists() and read_json(path) == self.active:
            path.unlink()


class NativeBackend:
    """Loaded only after pre-import RAM/source/lease gates; one model per process."""
    def __init__(self, runtime, role):
        self.tokenizer_encodings = 0
        import torch
        import transformers
        self.torch, self.transformers, self.runtime, self.role = torch, transformers, runtime, role
        if role != "student_base":
            import peft
            import bitsandbytes
            self.peft, self.bitsandbytes = peft, bitsandbytes
        torch.set_num_threads(1)
        torch.manual_seed(runtime["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    def free_gpu_bytes(self):
        r.require(self.torch.cuda.is_available(), "CUDA unavailable; no CPU fallback")
        return int(self.torch.cuda.mem_get_info(0)[0])

    def load(self, spec, rows):
        torch, hf = self.torch, self.transformers
        tokenizer = hf.AutoTokenizer.from_pretrained(spec["tokenizer"]["directory"], local_files_only=True,
                                                     trust_remote_code=False, use_fast=True)
        for row in rows:
            self.tokenizer_encodings += 1
            r.require(self.tokenizer_encodings <= 32, "tokenizer revalidation budget exhausted")
            r.require(tokenizer.encode(row["prompt"], add_special_tokens=False) == row["input_ids"],
                      "frozen prompt tokenization changed")
        if self.role == "student_base":
            from experiments.model_quality_gpu_v1.native import load_model
            self.model = load_model(Path(spec["assets"]["student"]["directory"]), "cuda:0")
            core_model = self.model
        else:
            quant = hf.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.model = hf.AutoModelForCausalLM.from_pretrained(
                spec["assets"]["donor"]["directory"], local_files_only=True, trust_remote_code=False,
                use_safetensors=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                quantization_config=quant, device_map={"": 0}, attn_implementation="sdpa")
            self.model = self.peft.prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
            self.model.gradient_checkpointing_disable()
            if hasattr(self.model, "_require_grads_hook"):
                self.model.disable_input_require_grads()
            if self.role == "donor_skilled":
                self.model = self.peft.PeftModel.from_pretrained(self.model, spec["assets"]["adapter"]["directory"],
                                                                 local_files_only=True, is_trainable=False)
                core_model = self.model.get_base_model()
            else:
                core_model = self.model
            r.require(getattr(core_model, "is_loaded_in_4bit", False), "donor not loaded in 4-bit")
        self.model.eval().requires_grad_(False)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_disable()
        self.core = core_model.model
        self.layers = self.core.layers
        self.width = 1024 if self.role == "student_base" else 2560
        r.require(len(self.layers) == (28 if self.role == "student_base" else 36), "unexpected decoder depth")
        r.require(core_model.config.hidden_size == self.width, "unexpected decoder width")
        r.require(core_model.config._attn_implementation == self.runtime["attention"], "attention implementation changed")
        r.require(not any(parameter.requires_grad for parameter in self.model.parameters()), "capture has trainable parameters")
        r.require(not any(tensor.is_meta for tensor in list(self.model.parameters()) + list(self.model.buffers())),
                  "unmaterialized native tensor")

    def capture(self, ids, layers):
        torch, captured, handles, observed_calls = self.torch, {}, [], [0]
        def count_call(module, inputs):
            observed_calls[0] += 1
            r.require(observed_calls[0] == 1, "unreserved extra decoder forward")
        def hook(layer):
            def capture_output(module, inputs, output):
                r.require(layer not in captured, "decoder layer ran twice")
                value = output[0] if isinstance(output, tuple) else output
                r.require(tuple(value.shape) == (1, len(ids), self.width), "unexpected block output shape")
                r.require(not value.requires_grad, "capture unexpectedly tracks gradients")
                value = value[0, len(ids) - 1].detach().to(dtype=torch.float32, device="cpu").contiguous()
                r.require(bool(torch.isfinite(value).all()), "nonfinite capture")
                captured[layer] = value.numpy().copy()
            return capture_output
        handles.append(self.core.register_forward_pre_hook(count_call))
        try:
            for layer in layers:
                handles.append(self.layers[layer].register_forward_hook(hook(layer)))
            with torch.inference_mode():
                tokens = torch.tensor([ids], dtype=torch.long, device="cuda:0")
                output = self.core(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                   use_cache=False, return_dict=True)
                del output
            r.require(observed_calls[0] == 1 and set(captured) == set(layers), "incomplete or unreserved capture")
            return captured
        finally:
            for handle in handles:
                handle.remove()


def run_worker(spec_path, spec_sha256, role, output, supervision):
    """Called only after inherited claim_worker; all failures retain reservations."""
    r.verify_file(spec_path, spec_sha256)
    spec = read_json(spec_path)
    runtime, rows = validate_spec(spec, role)
    launch = read_json(Path(supervision) / "launch.json")
    r.require(launch["supervisor_pid"] == spec["lease"]["owner_pid"], "supervisor does not own shared lease")
    output = Path(output).resolve()
    r.require(output.parent == Path(supervision).resolve() and output.name == "capture", "capture must be supervised child output")
    r.require(not output.is_relative_to(ROOT) and not output.exists(), "new external capture output required")
    output.mkdir()
    journal, backend, phase, resources = None, None, "preflight", []
    try:
        journal = CaptureJournal(spec, spec_sha256, role)
        resources.append(memory_gate(role, "before_import"))
        asset_names = ["student"] if role == "student_base" else ["donor"] + (["adapter"] if role == "donor_skilled" else [])
        for key in asset_names:
            verify_assets(spec["assets"][key])
        verify_assets(spec["tokenizer"])
        check_lease(spec)
        phase = "runtime_import"
        backend = NativeBackend(runtime, role)
        resources.append(memory_gate(role, "before_load", backend))
        phase = "model_load"
        backend.load(spec, rows)
        r.require(backend.tokenizer_encodings == 32, "incomplete tokenizer revalidation count")
        for key in asset_names:
            verify_assets(spec["assets"][key])
        phase = "capture"
        import numpy as np
        from .mechanics import tensor_digest
        row_receipts, values = [], {layer: [] for layer in LAYERS[role]}
        for ordinal, row in enumerate(rows):
            check_lease(spec)
            resource = memory_gate(role, "before_forward", backend)
            journal.reserve(row)
            captured = backend.capture(row["input_ids"], LAYERS[role])
            r.require(set(captured) == set(LAYERS[role]), "capture layer set differs")
            artifact_hashes = {}
            for layer, value in captured.items():
                r.require(value.dtype == np.dtype("float32") and value.shape == (backend.width,)
                          and bool(np.isfinite(value).all()), "invalid captured residual schema")
                path = output / f"row-{ordinal:03d}-layer-{layer:02d}.npy"
                with path.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                artifact_hashes[path.name] = r.file_digest(path)
                values[layer].append(value)
            body = {"id": row["id"], "parent_id": row["parent_id"], "split": row["split"],
                    "prompt_sha256": row["prompt_sha256"], "input_ids_sha256": row["input_ids_sha256"],
                    "position": len(row["input_ids"]) - 1, "token_id": row["input_ids"][-1],
                    "artifacts": artifact_hashes, "resources": resource, "forward_ordinal": ordinal + 1}
            durable_new(output / f"row-{ordinal:03d}.json", body)
            journal.complete_row(body)
            row_receipts.append(body)
        check_lease(spec)
        r.verify_source(ROOT, spec["source_commit"], clean=True)
        r.require(journal.calls == 32, "incomplete capture forward count")
        model_id = spec["assets"]["student" if role == "student_base" else "donor"]["model_sha256"]
        anchor = model_id
        if role == "donor_skilled":
            model_id = r.digest({"base": anchor, "adapter": spec["assets"]["adapter"]["model_sha256"]})
        calibration = spec["calibration"]
        capture_receipts = {}
        for layer, vectors in values.items():
            matrix = np.stack(vectors)
            path = output / f"layer-{layer:02d}.npy"
            with path.open("xb") as stream:
                np.save(stream, matrix, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            metadata = {key: calibration[key] for key in ("dataset_sha256", "calibration_sha256", "tokenizer_sha256",
                        "layer_map_sha256", "prompt_positions_sha256")}
            metadata.update(role=role, model_sha256=model_id, anchor_checkpoint_sha256=anchor,
                            implementation_digest=spec["implementation_digest"], runtime_binding_sha256=r.digest(runtime),
                            tensor_sha256=tensor_digest({"residual": matrix}), layer_ids=[layer],
                            pooling="last_nonpad_prompt_token", eval_mode=True, optimization_steps=0,
                            dtype="float32", padding_excluded=True, row_ids=[row["id"] for row in rows],
                            shape=list(matrix.shape), capture_site="block_output_raw")
            capture_receipts[str(layer)] = {"metadata": metadata, "file": path.name, "file_sha256": r.file_digest(path)}
        result = {"schema": 1, "status": "COMPLETE", "completed": True, "role": role,
                  "session_id": spec["session_id"], "spec_sha256": spec_sha256, "source_commit": spec["source_commit"],
                  "native_model_forwards": journal.calls, "generation_calls": 0, "optimizer_steps": 0,
                  "native_tokenizer_encodings": backend.tokenizer_encodings,
                  "captures": capture_receipts, "rows": row_receipts, "resources": resources,
                  "scientific_improvement_claim": False}
        journal.finish("COMPLETE")
        durable_new(output / "RESULT.json", result)
        return result
    except BaseException as exc:
        failure = {"status": "FAILED", "completed": False, "role": role, "phase": phase,
                   "reserved_forwards": journal.calls if journal is not None else 0,
                   "native_tokenizer_encodings": getattr(backend, "tokenizer_encodings", 0),
                   "error_type": type(exc).__name__, "error": str(exc), "spec_sha256": spec_sha256,
                   "scientific_improvement_claim": False, "retry_authorized": False}
        durable_new(output / "FAILURE.json", failure)
        if journal is not None and not (journal.directory / "TERMINAL.json").exists():
            journal.finish("FAILED", failure["error"])
        raise
    finally:
        if journal is not None:
            journal.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--supervision", required=True, type=Path)
    args = parser.parse_args()
    from experiments.model_quality_gpu_v1.supervisor import claim_worker
    claim_worker(args.supervision)
    result = run_worker(args.spec, args.spec_sha256, args.role, args.out, args.supervision)
    print(json.dumps({"status": result["status"], "role": args.role,
                      "native_model_forwards": result["native_model_forwards"]}), flush=True)


if __name__ == "__main__":
    main()
