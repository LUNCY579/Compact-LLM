"""Bounded Titan E0. No benchmark access, production training or source mutation.

Local stock Qwen3 plus an explicit q/v LoRA wrapper; all outcomes are write-once.
Requires a committed source snapshot and frozen config. A failure never grants E1.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

from experiments.post_training_mimo_centralized_v1.scripts import laptop_profile


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def padding_failure_measurement(*, left, right, max_abs, mean_abs, reference_logprobs,
                                observed_logprobs, changed_argmax_positions):
    """Preserve failed likelihood measurements; null denotes a nonfinite value."""
    def finite_or_none(value):
        return value if math.isfinite(value) else None
    return dict(check="native_padding_likelihood", left=left, right=right,
                max_abs=finite_or_none(max_abs), mean_abs=finite_or_none(mean_abs),
                reference_logprobs=[finite_or_none(value) for value in reference_logprobs],
                observed_logprobs=[finite_or_none(value) for value in observed_logprobs],
                changed_argmax_positions=changed_argmax_positions)


ACCOUNTING_VERSION = 2
CALL_OPERATIONS = ("model_construct", "checkpoint_load", "tensor_copy", "forward",
                   "backward", "optimizer_step", "checkpoint_save", "checkpoint_reload")
_POSITION_COUNTERS = ("forward_input_positions_attempted", "forward_input_positions_returned",
                      "training_forward_input_positions_attempted", "training_forward_input_positions_returned")


def _call_metadata(operation, stage, input_positions, training):
    require(operation in CALL_OPERATIONS, "unregistered accounted operation")
    require(type(stage) is str and bool(stage) and stage == stage.strip(), "invalid accounting stage")
    require(type(input_positions) is int and type(training) is bool, "invalid operation metadata types")
    require(input_positions > 0 if operation == "forward" else input_positions == 0 and not training,
            "input positions/training flag apply only to forward calls")
    return {"operation": operation, "stage": stage, "input_positions": input_positions, "training": training}


class CallAccounting:
    """Stdlib call journal; neither RNG nor inputs/results are inspected or changed.

    An attempt marker is emitted before calling the operation. It records intent
    to invoke, not proof that execution reached the callable before a hard kill.
    Returned means the wrapped Python call/span returned, not that asynchronous
    GPU work finished or passed later numerical checks. A raised call is failed;
    unmatched attempts remain pending. The event sink must persist its records.
    If journal I/O itself fails, in-memory accounting survives, but a snapshot that
    cannot be reconciled with durable events is not qualifying evidence.

    on_return is reserved for immediate legacy completed-step accounting; it runs
    after the operation returned and before the returned event is emitted. Errors
    from that hook remain errors after a returned call, never failed model calls.

    tensor_copy covers the 310 initial serialized-checkpoint-to-base copies only.
    Adapter/optimizer/RNG reload work is inside the checkpoint_reload span;
    restore_initial housekeeping is not separately counted as tensor_copy. This
    is an explicit operation ledger, not a census of every internal tensor copy.
    """
    def __init__(self, emit, maximum_optimizer_calls=5):
        require(callable(emit), "accounting event sink required")
        require(type(maximum_optimizer_calls) is int and 0 <= maximum_optimizer_calls <= 5,
                "accounting optimizer ceiling must not exceed registered five calls")
        self.emit = emit
        self.maximum_optimizer_calls = maximum_optimizer_calls
        self.last_call_id = 0
        self.operations = {name: {state: 0 for state in ("attempted", "returned", "failed", "pending")}
                           for name in CALL_OPERATIONS}
        self.positions = dict.fromkeys(_POSITION_COUNTERS, 0)
        self.pending = {}
        self.stack = []

    def begin(self, operation, *, stage, input_positions=0, training=False):
        metadata = _call_metadata(operation, stage, input_positions, training)
        if operation == "optimizer_step":
            require(self.operations[operation]["attempted"] < self.maximum_optimizer_calls,
                    "OPTIMIZER_CALL_BUDGET")
        self.last_call_id += 1
        call = {"call_id": self.last_call_id, **metadata}
        self.operations[operation]["attempted"] += 1
        self.operations[operation]["pending"] += 1
        self.pending[call["call_id"]] = call
        self.stack.append(call["call_id"])
        if operation == "forward":
            self.positions["forward_input_positions_attempted"] += input_positions
            if training:
                self.positions["training_forward_input_positions_attempted"] += input_positions
        self.emit("operation_attempt", accounting_version=ACCOUNTING_VERSION, **call)
        return call["call_id"]

    def _finish(self, call_id, state):
        require(type(call_id) is int and bool(self.stack) and self.stack[-1] == call_id,
                "operation completion must match the innermost pending call")
        call = self.pending.pop(call_id)
        self.stack.pop()
        self.operations[call["operation"]]["pending"] -= 1
        self.operations[call["operation"]][state] += 1
        if state == "returned" and call["operation"] == "forward":
            self.positions["forward_input_positions_returned"] += call["input_positions"]
            if call["training"]:
                self.positions["training_forward_input_positions_returned"] += call["input_positions"]
        return call

    def returned(self, call_id, *, on_return=None):
        call = self._finish(call_id, "returned")
        try:
            if on_return is not None:
                on_return()
        finally:
            self.emit("operation_returned", accounting_version=ACCOUNTING_VERSION, **call)

    def failed(self, call_id, error):
        call = self._finish(call_id, "failed")
        self.emit("operation_failed", accounting_version=ACCOUNTING_VERSION, **call,
                  error_type=type(error).__name__, error_message=str(error)[:1000])

    def call(self, operation, function, *, stage, input_positions=0, training=False, on_return=None):
        require(callable(function) and (on_return is None or callable(on_return)), "accounted callable required")
        call_id = self.begin(operation, stage=stage, input_positions=input_positions, training=training)
        try:
            value = function()
        except BaseException as error:
            # A nested journal-write failure may leave an inner intent pending.
            # Preserve that uncertainty rather than inventing a terminal marker.
            if self.stack and self.stack[-1] == call_id:
                self.failed(call_id, error)
            raise
        self.returned(call_id, on_return=on_return)
        return value

    def snapshot(self):
        return {"accounting_version": ACCOUNTING_VERSION, "max_optimizer_calls": self.maximum_optimizer_calls,
                "last_call_id": self.last_call_id,
                "operations": {name: dict(values) for name, values in self.operations.items()},
                "pending_calls": [dict(self.pending[key]) for key in sorted(self.pending)],
                **self.positions}


def verify_call_journal(events, snapshot=None, maximum_optimizer_calls=5):
    """Pure reconstruction of an ordered operation journal, allowing pending calls.

    Accept a parsed EVENTS list, including unrelated runtime/resource records.
    Unknown operation event kinds, invalid nesting, missing attempt records,
    repeated completion, changed metadata, budget violations or snapshot drift
    raise. Returned records are not efficacy or finite-GPU-work certificates.
    Hard-kill recovery can call this without a final snapshot; pending is explicit.
    """
    require(type(events) is list, "journal must be a list")
    require(type(maximum_optimizer_calls) is int and 0 <= maximum_optimizer_calls <= 5, "invalid journal optimizer ceiling")
    counts = {name: dict.fromkeys(("attempted", "returned", "failed", "pending"), 0) for name in CALL_OPERATIONS}
    positions = dict.fromkeys(_POSITION_COUNTERS, 0)
    pending, stack, last_id = {}, [], 0
    for record in events:
        require(type(record) is dict and type(record.get("event")) is str, "invalid journal record")
        kind = record["event"]
        if not kind.startswith("operation_"):
            continue
        require(kind in {"operation_attempt", "operation_returned", "operation_failed"}, "unknown operation event")
        fields = {"event", "accounting_version", "call_id", "operation", "stage", "input_positions", "training"}
        if kind == "operation_failed":
            fields |= {"error_type", "error_message"}
            require(type(record.get("error_type")) is str and bool(record["error_type"])
                    and type(record.get("error_message")) is str and len(record["error_message"]) <= 1000,
                    "failed call requires bounded exception metadata")
        require(fields <= set(record) <= fields | {"elapsed_seconds"}, "operation event fields differ")
        if "elapsed_seconds" in record:
            require(type(record["elapsed_seconds"]) in (int, float) and math.isfinite(record["elapsed_seconds"])
                    and record["elapsed_seconds"] >= 0, "invalid event time")
        require(type(record["accounting_version"]) is int and record["accounting_version"] == ACCOUNTING_VERSION,
                "operation accounting version drift")
        call_id = record["call_id"]
        require(type(call_id) is int and call_id > 0, "invalid operation call ID")
        call = {"call_id": call_id, **_call_metadata(record["operation"], record["stage"],
                                                   record["input_positions"], record["training"])}
        operation = call["operation"]
        if kind == "operation_attempt":
            require(call_id == last_id + 1, "operation call IDs must be consecutive")
            if operation == "optimizer_step":
                require(counts[operation]["attempted"] < maximum_optimizer_calls, "OPTIMIZER_CALL_BUDGET")
            last_id = call_id
            pending[call_id] = call
            stack.append(call_id)
            counts[operation]["attempted"] += 1
            counts[operation]["pending"] += 1
            if operation == "forward":
                positions["forward_input_positions_attempted"] += call["input_positions"]
                if call["training"]:
                    positions["training_forward_input_positions_attempted"] += call["input_positions"]
        else:
            require(bool(stack) and stack[-1] == call_id and pending[call_id] == call,
                    "operation completion nesting/metadata mismatch")
            stack.pop()
            del pending[call_id]
            state = "returned" if kind == "operation_returned" else "failed"
            counts[operation]["pending"] -= 1
            counts[operation][state] += 1
            if state == "returned" and operation == "forward":
                positions["forward_input_positions_returned"] += call["input_positions"]
                if call["training"]:
                    positions["training_forward_input_positions_returned"] += call["input_positions"]
    reconstructed = {"accounting_version": ACCOUNTING_VERSION, "max_optimizer_calls": maximum_optimizer_calls,
                     "last_call_id": last_id, "operations": counts,
                     "pending_calls": [pending[key] for key in sorted(pending)], **positions}
    if snapshot is not None:
        require(json.dumps(snapshot, sort_keys=True, allow_nan=False) == json.dumps(reconstructed, sort_keys=True, allow_nan=False),
                "operation journal/snapshot mismatch")
    return reconstructed


def record_resource_profile(profiles, profile, emit):
    """Retain an already measured completed profile before journaling it."""
    profiles.append(dict(profile))
    emit("resource_profile", profile=dict(profile))


def install_lora(model, torch, rank=8, alpha=16):
    """FP32 factors, BF16 base; zero B preserves the original module function."""
    sites = [(name,module) for name,module in model.named_modules() if name.endswith((".q_proj", ".v_proj"))]
    require(len(sites) == 56, "Titan must have 28 q/v pairs")
    require(rank == 8 and alpha == 16 and all(isinstance(m,torch.nn.Linear) for _,m in sites), "LoRA contract")
    require(all(not p.requires_grad for p in model.parameters()), "base must already be frozen")
    class LoRA(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.lora_A = torch.nn.Parameter(torch.empty(rank, base.in_features,
                                                       device=base.weight.device, dtype=torch.float32))
            self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank,
                                                       device=base.weight.device, dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** .5)

        def forward(self, x):
            delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.lora_A), self.lora_B)
            return self.base(x) + (delta * (alpha / rank)).to(x.dtype)

    names = []
    for name, module in sites:
        parent_name, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child, LoRA(module))
        names.append(name)
    require(len(names) == 56, "Titan must have 28 q/v pairs")
    return names


def completion_loss(torch, logits, ids, labels):
    # Select supervised positions before FP32 promotion: no full FP32 vocabulary cache.
    require(logits.ndim == 3 and ids.ndim == labels.ndim == 2 and
            logits.shape[:2] == ids.shape == labels.shape, "loss shapes")
    require(ids.dtype == labels.dtype == torch.long and logits.is_floating_point(), "loss dtypes")
    require(logits.device == ids.device == labels.device, "loss devices")
    require(bool(((ids >= 0) & (ids < logits.shape[-1])).all()), "token range")
    mask = labels[:, 1:] != -100
    require(bool(mask.any(dim=1).all()), "empty completion")
    require(bool((labels[:, 0] == -100).all()), "first token cannot be supervised")
    require(bool(((labels == -100) | (labels == ids)).all()), "label mismatch")
    selected = logits[:, :-1][mask].float()
    require(bool(torch.isfinite(selected).all()), "nonfinite supervised logits")
    # Per-completion mean then mean across sequences, regardless of target length.
    losses = torch.nn.functional.cross_entropy(selected, labels[:, 1:][mask], reduction="none")
    return torch.stack([part.mean() for part in losses.split(mask.sum(1).tolist())]).mean()


def validate_config(cfg):
    if laptop_profile.metadata_present(cfg):
        laptop_profile.validate_config(cfg)
        return
    for field in ("host_free_gib","worker_free_gib","gpu_free_gib","gpu_worker_free_gib","disk_free_gib","wall_seconds"):
        require(type(cfg.get(field)) in (int,float) and math.isfinite(cfg[field]), "nonfinite/invalid resource")
    require(cfg.get("gpu_worker_free_gib",0) >= .5, "GPU worker floor")
    for field,value in {"seed":20260928,"maximum_optimizer_calls":5,"eos_ids":[151645,151643],
                        "pad_id":151645,"base_dtype":"bfloat16","adapter_dtype":"float32",
                        "attention":"eager","resume":"bitwise"}.items():
        require(type(cfg.get(field)) is type(value) and cfg[field] == value, "unregistered " + field)
    require(cfg.get("status") == "FROZEN_E0_ONLY", "unfrozen config")
    require(cfg.get("weight_sha256") == "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7",
            "not the exact Titan checkpoint")
    require(set(cfg.get("asset_sha256", {})) == {"config.json", "tokenizer.json", "tokenizer_config.json",
            "generation_config.json", "merges.txt", "vocab.json"}, "incomplete asset binding")
    require(cfg.get("contexts") == [256,512,1024] and cfg.get("parity_prompts") == 32 and
            cfg.get("generation_cap") == 32, "unregistered workload")
    require(cfg.get("host_free_gib",0) >= 7.4 and cfg.get("worker_free_gib",0) >= 6.0 and
            cfg.get("gpu_free_gib",0) >= 4.0 and cfg.get("disk_free_gib",0) >= 4.0 and
            0 < cfg.get("wall_seconds",0) <= 1800, "invalid resource envelope")
    require(cfg.get("zero_adapter_atol") == 0 and cfg.get("logprob_max_atol") == .03125 and
            cfg.get("logprob_mean_atol") == .01 and cfg.get("ce_atol") == .000002,
            "unregistered tolerance")


def tensor_digest(torch, tensor):
    h = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), 1024 * 1024):
        part = flat[start:start + 1024 * 1024].contiguous().view(torch.uint8).cpu().numpy()
        h.update(part.tobytes())
    return h.hexdigest()


def run(args):
    started = time.monotonic()
    while not args.go_path.exists() and time.monotonic()-started < 30:
        time.sleep(.05)
    require(args.go_path.exists(), "supervisor containment GO missing")
    out = args.output.resolve()
    require(not out.exists(), "output must be new")
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    laptop = laptop_profile.metadata_present(cfg)
    streamed = laptop and cfg.get("loader_mode") == laptop_profile.STREAMED_LOADER_MODE
    unpadded_ce = streamed and cfg["qualification_stage"] in ("unpadded_ce", "unpadded_ce512")
    if streamed:
        from experiments.post_training_mimo_centralized_v1.scripts import stream_checkpoint
    require(json.loads(args.go_path.read_text())["config_sha256"] == sha(config_path), "GO config mismatch")
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True)
    require(not dirty.strip(), "commit all source/drafts before native launch")
    out.mkdir(parents=True)
    result = dict(schema_version=1, experiment_id="RINEYA-0P6B-E0", status="RUNNING",
                  evidence_level="MEASURED / REPRODUCED", code_sha=head, dirty_at_launch=False,
                  config_sha256=sha(config_path), pretrained_loads=0, optimizer_steps=0,
                  training_tokens=0, generation_tokens=0, teacher_calls=0, checks={}, resources=[],
                  resource_profiles=[], accounting_version=ACCOUNTING_VERSION)
    if laptop:
        result.update(resource_profile=cfg["resource_profile"], qualification_stage=cfg["qualification_stage"])
    if streamed:
        result["loader_mode"] = cfg["loader_mode"]
    if unpadded_ce:
        result["runtime_scope"] = cfg["runtime_scope"]
    write(out / "CONFIG.json", cfg)
    source_receipt = {"code_sha": head, "dirty": False, "script_sha256": sha(__file__)}
    if laptop:
        source_receipt["dependency_sha256"] = {
            "experiments/post_training_mimo_centralized_v1/scripts/laptop_profile.py": sha(laptop_profile.__file__)}
        if streamed:
            source_receipt["dependency_sha256"]["experiments/post_training_mimo_centralized_v1/scripts/stream_checkpoint.py"] = sha(stream_checkpoint.__file__)
    write(out / "SOURCE_RECEIPT.json", source_receipt)
    def event(kind, **values):
        with (out / "EVENTS.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"elapsed_seconds": time.monotonic()-started, "event": kind, **values}, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    accounting = CallAccounting(event, cfg["maximum_optimizer_calls"])
    result["call_accounting"] = accounting.snapshot()
    try:
        event("runtime_import_start")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import psutil
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        event("runtime_import_complete")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA unavailable")
        runtime = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       packages={k: importlib.metadata.version(k) for k in
                                 ("torch", "transformers", "safetensors", "tokenizers", "psutil")},
                       gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                       total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                       attention="eager", base_dtype="bfloat16", adapter_dtype="float32")
        write(out / "RUNTIME_RECEIPT.json", runtime)
        require(all(runtime["packages"][k] == v for k,v in {
            "torch":"2.10.0+cu126","transformers":"4.57.1","safetensors":"0.7.0","tokenizers":"0.22.2"}.items()),
            "unregistered runtime versions")
        proc = psutil.Process()
        def guard(stage, admission=False):
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            snap = dict(stage=stage, elapsed_seconds=time.monotonic()-started,
                        host_available_gib=psutil.virtual_memory().available / 2**30,
                        rss_gib=proc.memory_info().rss / 2**30, gpu_free_gib=free / 2**30,
                        gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                        gpu_reserved_gib=torch.cuda.memory_reserved() / 2**30,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        disk_free_gib=psutil.disk_usage(str(out)).free / 2**30)
            if laptop:
                snap["admission_required_gib"] = laptop_profile.admission_required_gib(cfg, snap["rss_gib"])
            result["resources"].append(snap)
            event("resource", **snap)
            if laptop:
                require(snap["rss_gib"] <= cfg["worker_rss_limit_gib"], "PROCESS_RSS_GUARD")
            required_host = snap["admission_required_gib"] if laptop and admission else cfg["host_free_gib"]
            require(snap["host_available_gib"] >= required_host if admission else
                    snap["host_available_gib"] >= cfg["worker_free_gib"], "HOST_RAM_GUARD")
            require(snap["disk_free_gib"] >= cfg["disk_free_gib"], "DISK_GUARD")
            if admission:
                require(snap["gpu_free_gib"] >= cfg["gpu_free_gib"], "GPU_ADMISSION_GUARD")
            require(snap["gpu_free_gib"] >= cfg["gpu_worker_free_gib"], "GPU_WORKER_GUARD")
            require(time.monotonic()-started <= cfg["wall_seconds"], "WALL_CEILING")
            return snap

        guard("preload", True)
        model_dir = Path(cfg["model_dir"])
        require(sha(model_dir / "model.safetensors") == cfg["weight_sha256"], "source weight drift")
        static = json.loads(Path(cfg["primary_receipt"]).read_text())
        require(sha(cfg["primary_receipt"]) == cfg["primary_receipt_sha256"], "primary receipt drift")
        require(static["weight_sha256"] == cfg["weight_sha256"] and static["parameter_count"] == 596049920,
                "primary binding mismatch")
        require(Path(static["checkpoint_path"]).resolve() == model_dir.resolve(), "primary path mismatch")
        require(all(static["files"][n]["sha256"] == h for n,h in cfg["asset_sha256"].items()), "primary asset mismatch")
        # All assets used by the runtime are bound directly before use, independently
        # of static receipt layout; exact frozen hashes live in this E0 config.
        for name, expected in cfg["asset_sha256"].items():
            require(sha(model_dir / name) == expected, "asset drift: " + name)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        require(tokenizer.eos_token_id == 151645, "EOS mismatch")
        template = json.loads((model_dir / "tokenizer_config.json").read_text())["chat_template"]
        require(tokenizer.chat_template == template, "template drift")
        raw = "Qualification: 7 + 5 = 12."
        raw_ids = tokenizer.encode(raw, add_special_tokens=False)
        require(tokenizer.decode(raw_ids) == raw, "raw token roundtrip")
        write(out / "TOKENIZER_RECEIPT.json", dict(class_name=type(tokenizer).__name__,
              template_sha256=hashlib.sha256(template.encode()).hexdigest(), eos_id=tokenizer.eos_token_id,
              original_pad_id=tokenizer.pad_token_id, raw_ids=raw_ids, raw_roundtrip=True,
              enable_thinking=False, add_generation_prompt=True))
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model_cfg = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True)
        model_cfg._attn_implementation = "eager"
        require(model_cfg.rope_theta == 1000000 and model_cfg.num_hidden_layers == 28, "architecture drift")
        torch.cuda.reset_peak_memory_stats()
        load_start = time.monotonic()
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda:0"):
                model = accounting.call("model_construct", lambda: Qwen3ForCausalLM(model_cfg), stage="model_construct")
        finally:
            torch.set_default_dtype(previous)
        model.eval().requires_grad_(False)
        states = model.state_dict()
        base_hash = {}
        def load_checkpoint():
            if streamed:
                hashes, diagnostics = stream_checkpoint.load_streamed(
                    torch, model, model_dir / "model.safetensors", accounting, tensor_digest)
                base_hash.update(hashes)
                result["loader_diagnostics"] = diagnostics
            else:
                with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as weights:
                    keys = set(weights.keys())
                    require(set(states) - keys == {"lm_head.weight"} and not keys - set(states), "checkpoint key mismatch")
                    with torch.no_grad():
                        for name in sorted(keys):
                            source = weights.get_tensor(name)
                            require(source.shape == states[name].shape and source.dtype == states[name].dtype, "tensor mismatch")
                            accounting.call("tensor_copy", lambda: states[name].copy_(source), stage="tensor_copy:" + name)
                            base_hash[name] = tensor_digest(torch, source)
                            require(tensor_digest(torch, states[name]) == base_hash[name], "loaded tensor mismatch")
                    del source
            require(model.lm_head.weight is model.model.embed_tokens.weight, "embedding tie lost")
            require(sum(p.numel() for p in model.parameters()) == 596049920, "parameter count mismatch")
        accounting.call("checkpoint_load", load_checkpoint, stage="checkpoint_load")
        del states
        result["pretrained_loads"] = 1
        result["load_seconds"] = time.monotonic()-load_start
        result["checks"]["exact_native_load"] = True
        guard("loaded")
        model_receipt = dict(weight_sha256=cfg["weight_sha256"],
             primary_receipt_sha256=sha(cfg["primary_receipt"]), unique_parameters=596049920,
             tensor_sha256=base_hash, loader="stock Qwen3 allocated BF16 on CUDA; strict streamed safetensors copy",
             tied=True, config=model_cfg.to_dict())
        if streamed:
            model_receipt.update(loader="stock Qwen3 allocated BF16 on CUDA; strict safetensors header and 2MiB bounded read buffer copy",
                                 loader_mode=cfg["loader_mode"], loader_diagnostics=result["loader_diagnostics"])
        write(out / "MODEL_RECEIPT.json", model_receipt)

        def forward(ids, stage, training=False, **kwargs):
            # numel reads shape metadata and includes actual left/right padding.
            if unpadded_ce:
                validated = laptop_profile.prepare_unpadded_forward(ids, kwargs, runtime_scope=cfg["runtime_scope"])
                if stage == "unpad_allones_mask":
                    require(tuple(ids.shape) == (1,80) and training is False, "all-ones diagnostic requires its fixed frozen 80-token fixture")
                    require(kwargs.get("attention_mask") is not None, "all-ones diagnostic mask missing")
                    validated["attention_mask"] = kwargs["attention_mask"]
                kwargs = validated
            return accounting.call("forward", lambda: model(ids, **kwargs), stage=stage,
                                   input_positions=ids.numel(), training=training)

        def prompt_ids(i, length):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content":
                f"Mechanical fixture {i}: return only the integer 12 for 7 + 5."}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            filler = tokenizer.encode(" neutral context", add_special_tokens=False)
            # Right prefix fill with ordinary non-EOS tokens; not an evaluated task.
            padding = (filler * ((length // len(filler)) + 1))[:max(0, length-len(prompt))]
            seq = padding + prompt
            require(len(seq) == length, "fixture length")
            return torch.tensor([seq], device="cuda", dtype=torch.long)

        if laptop and cfg["qualification_stage"] == "load_forward":
            with torch.no_grad():
                ids = prompt_ids(0, 256)
                logits = forward(ids, "load_forward_256", use_cache=False, logits_to_keep=1).logits
                require(bool(torch.isfinite(logits).all()), "nonfinite load/forward probe")
                del logits
            result["checks"]["load_forward_finite"] = True
            guard("load_forward_256")
            result["status"] = laptop_profile.PROBE_PASS_STATUS
            result["unqualified"] = ["zero-adapter parity", "sampled likelihood", "EOS/PAD masks",
                                     "LoRA/backward/optimizer", "deterministic resume", "E1 data/replay",
                                     "held-out capability", "long context beyond256"]
            return 0

        baseline = []
        with torch.no_grad():
            for i in range(cfg["parity_prompts"]):
                length = cfg["contexts"][i % len(cfg["contexts"]) ]
                ids = prompt_ids(i, length)
                logits = forward(ids, f"base_parity_{i}", use_cache=False, logits_to_keep=1).logits.float()
                require(bool(torch.isfinite(logits).all()), "nonfinite base forward")
                baseline.append(logits.cpu())
                del logits
                guard(f"base_parity_{i}")
        modules = install_lora(model, torch)
        allowed = {n: p for n, p in model.named_parameters() if p.requires_grad}
        require(len(allowed) == 112 and sum(p.numel() for p in allowed.values()) == 1146880, "LoRA scope mismatch")
        require(all(n.endswith((".lora_A", ".lora_B")) for n in allowed), "unauthorized trainable")
        write(out / "TRAINABLE_INVENTORY.json", {"modules": modules, "parameters": {
              n: {"shape": list(p.shape), "dtype": str(p.dtype)} for n,p in allowed.items()}, "count": 1146880})
        parity_max = 0.0
        with torch.no_grad():
            for i, original in enumerate(baseline):
                logits = forward(prompt_ids(i, cfg["contexts"][i % len(cfg["contexts"])]),
                                 f"zero_parity_{i}", use_cache=False, logits_to_keep=1).logits.float().cpu()
                diff = (logits-original).abs().max().item()
                parity_max = max(parity_max, diff)
                require(diff <= cfg["zero_adapter_atol"], "zero-adapter parity failed")
                require(torch.equal(logits.argmax(-1), original.argmax(-1)), "zero adapter argmax changed")
                guard(f"zero_parity_{i}")
        result["checks"]["zero_adapter_parity"] = dict(prompts=len(baseline), max_abs=parity_max)
        del baseline

        backward_resume_only = laptop and cfg["qualification_stage"] == "backward_resume"
        if unpadded_ce:
            with torch.no_grad():
                short_ids = prompt_ids(950,80)
                short_logits = forward(short_ids, "unpad_reference", use_cache=False).logits
                require(bool(torch.isfinite(short_logits).all()), "nonfinite unpadded reference")
                perturbed = short_ids.clone()
                perturbed[:,-3:] = torch.tensor([[17,18,19]],device="cuda")
                altered = forward(perturbed, "unpad_causality_perturbed", use_cache=False).logits
                require(bool(torch.isfinite(altered).all()), "nonfinite unpadded causality forward")
                require(torch.equal(short_logits[:,:-3],altered[:,:-3]), "unpadded future-token causality failure")
                result["checks"]["native_unpadded_causality"] = dict(tokens=80, perturbed_tail_tokens=3,
                                                                     prefix_tokens=77, bitwise=True)
                guard("unpad_causality")
                allones_logits = forward(short_ids, "unpad_allones_mask",
                                         attention_mask=torch.ones_like(short_ids),use_cache=False).logits
                require(bool(torch.isfinite(allones_logits).all()), "nonfinite explicit all-ones forward")
                ref = short_logits[:,:-1].float().log_softmax(-1).gather(-1,short_ids[:,1:,None]).squeeze(-1)
                observed = allones_logits[:,:-1].float().log_softmax(-1).gather(-1,short_ids[:,1:,None]).squeeze(-1)
                errors = (ref-observed).abs()
                finite = bool(torch.isfinite(ref).all()) and bool(torch.isfinite(observed).all()) and bool(torch.isfinite(errors).all())
                metrics = dict(tokens=79, finite=finite, max_abs=None, mean_abs=None, qualified=False,
                               comparison="explicit_allones_mask_vs_omitted_same_80_tokens")
                result["checks"]["explicit_allones_mask_equivalence"] = metrics
                require(finite, "nonfinite explicit all-ones likelihood")
                metrics.update(max_abs=float(errors.max()), mean_abs=float(errors.mean()))
                metrics["qualified"] = metrics["max_abs"] <= cfg["logprob_max_atol"] and metrics["mean_abs"] <= cfg["logprob_mean_atol"]
                require(metrics["qualified"], "explicit all-ones likelihood disagreement")
                guard("unpad_allones_equivalence")
                del short_logits, allones_logits, altered, ref, observed, errors
        if not backward_resume_only and not unpadded_ce:
            with torch.no_grad():
                short_ids = prompt_ids(950,80)
                short_logits = forward(short_ids, "padding_reference", use_cache=False).logits
                for left,right in ((2,0),(0,2),(2,2)):
                    padded = torch.cat([torch.full((1,left),151645,device="cuda"),short_ids,
                                        torch.full((1,right),151645,device="cuda")],1)
                    am = torch.zeros_like(padded)
                    am[:,left:left+80] = 1
                    positions = (am.cumsum(-1)-1).clamp(min=0)
                    padded_logits = forward(padded, f"native_padding_{left}_{right}",
                                            attention_mask=am,position_ids=positions,use_cache=False).logits[:,left:left+80]
                    ref = short_logits[:,:-1].float().log_softmax(-1).gather(-1,short_ids[:,1:,None]).squeeze(-1)
                    observed = padded_logits[:,:-1].float().log_softmax(-1).gather(-1,short_ids[:,1:,None]).squeeze(-1)
                    errors = (ref-observed).abs()
                    padding_ok = float(errors.max()) <= cfg["logprob_max_atol"] and float(errors.mean()) <= cfg["logprob_mean_atol"]
                    if not padding_ok:
                        result["failed_check_measurement"] = padding_failure_measurement(
                            left=left, right=right,
                            max_abs=float(errors.max()), mean_abs=float(errors.mean()),
                            reference_logprobs=ref.detach().cpu().tolist()[0],
                            observed_logprobs=observed.detach().cpu().tolist()[0],
                            changed_argmax_positions=(short_logits.argmax(-1) != padded_logits.argmax(-1)).nonzero()[:,1].cpu().tolist())
                    require(padding_ok, "native padding likelihood disagreement")
                    guard(f"native_padding_{left}_{right}")
                perturbed = short_ids.clone()
                perturbed[:,-3:] = torch.tensor([[17,18,19]],device="cuda")
                altered = forward(perturbed, "causality_perturbed", use_cache=False).logits[:,:-3]
                require(torch.equal(short_logits[:,:-3],altered), "future-token causality failure")
                del short_logits, padded_logits, altered, ref, observed
            result["checks"]["native_padding_and_causality"] = True

            # Independently computed sequential full-prefix sampling and teacher forcing;
            # this qualifies this no-cache sampler only, never an untested KV backend.
            seed_ids = prompt_ids(900, 80)
            generated, online = [], []
            gen_start = time.monotonic()
            with torch.no_grad():
                while len(generated) < cfg["generation_cap"]:
                    all_ids = torch.cat([seed_ids, torch.tensor([generated], dtype=torch.long, device="cuda")], 1)
                    scores = forward(all_ids, f"sample_{len(generated)}", use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    probs = scores.softmax(-1)
                    tok = int(torch.multinomial(probs, 1).item())
                    online.append(float(scores.log_softmax(-1)[tok]))
                    generated.append(tok)
                    result["generation_tokens"] += 1
                    guard("sample")
                    if tok in cfg["eos_ids"]:
                        break
                full_ids = torch.cat([seed_ids, torch.tensor([generated], dtype=torch.long, device="cuda")], 1)
                logits = forward(full_ids, "sampled_logprobs_offline", use_cache=False).logits
                pos = seed_ids.shape[1]-1
                lp = logits[0, pos:pos+len(generated)].float().log_softmax(-1)
                target = torch.tensor(generated, device="cuda")
                offline = lp.gather(1, target[:, None]).squeeze(1).cpu()
                differences = (offline-torch.tensor(online)).abs()
                require(float(differences.max()) <= cfg["logprob_max_atol"] and
                        float(differences.mean()) <= cfg["logprob_mean_atol"], "likelihood disagreement")
                del logits, lp, scores, probs
            result["generation_seconds"] = time.monotonic()-gen_start
            result["generation_tokens_per_second"] = len(generated)/result["generation_seconds"]
            result["checks"]["sampled_logprobs"] = dict(max_abs=float(differences.max()), mean_abs=float(differences.mean()),
                  tokens=generated, stop_reason="eos" if generated[-1] in cfg["eos_ids"] else "length",
                  backend="full_prefix_no_cache_temperature1_topk0_topp1")

        # Actual EOS=PAD masking is positional, not token-value masking.
        from experiments.maxrl_comp002_pretraining_v1.mechanics import completion_labels
        x = torch.tensor([[151645, 10, 11, 12, 151645, 151645]], device="cuda")
        attention = torch.tensor([[False, True, True, True, True, False]], device="cuda")
        labels = completion_labels(x, attention, [2], [2], [151645], ["eos"], 8)
        require(labels.tolist() == [[-100,-100,-100,12,151645,-100]], "EOS/PAD mask failed")
        result["checks"]["eos_pad_mask"] = True

        optimizer = torch.optim.AdamW(list(allowed.values()), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0)
        initial = {n: p.detach().cpu().clone() for n,p in allowed.items()}
        def restore_initial():
            with torch.no_grad():
                for n,p in allowed.items():
                    p.copy_(initial[n])
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)

        def optimizer_step(training_tokens, stage):
            def completed():
                result["optimizer_steps"] += 1
                result["training_tokens"] += training_tokens
            return accounting.call("optimizer_step", optimizer.step, stage=stage, on_return=completed)

        profiles = result["resource_profiles"]
        for length in cfg["contexts"]:
            restore_initial()
            guard(f"before_backward_{length}")
            torch.cuda.reset_peak_memory_stats()
            ids = prompt_ids(1000, length-3)
            tail = torch.tensor([[16, 17, 151645]], device="cuda")
            ids = torch.cat([ids,tail], 1)
            labels = torch.full_like(ids,-100)
            labels[:,-3:] = tail
            torch.cuda.synchronize()
            t0 = time.monotonic()
            output = forward(ids, f"training_{length}", training=True, use_cache=False).logits
            loss = completion_loss(torch, output, ids, labels)
            # Independent summed per-position NLL avoids allocating FP32 logits for
            # unsupervised prompt positions solely for this mechanical check.
            ref = sum(-output[0,j-1].float().log_softmax(-1)[ids[0,j]]
                      for j in range(length-3,length)) / 3
            require(abs(float(loss.detach())-float(ref.detach())) <= cfg["ce_atol"], "causal-shift CE disagreement")
            del ref
            require(bool(torch.isfinite(loss)), "nonfinite loss")
            accounting.call("backward", loss.backward, stage=f"training_{length}")
            require(all(p.grad is None for n,p in model.named_parameters() if n not in allowed), "gradient leak")
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in allowed.values()), "missing/nonfinite gradient")
            norm = torch.nn.utils.clip_grad_norm_(list(allowed.values()),1.0,error_if_nonfinite=True)
            guard(f"backward_{length}")
            optimizer_step(length, f"training_{length}")
            require(any(not torch.equal(p.detach().cpu(), initial[n]) for n,p in allowed.items()), "optimizer made no update")
            torch.cuda.synchronize()
            duration = time.monotonic()-t0
            record_resource_profile(profiles,
                dict(context=length, loss=float(loss.detach()), gradient_norm=float(norm), seconds=duration,
                     tokens_per_second=length/duration, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     optimizer_tensor_bytes=sum(v.numel()*v.element_size() for s in optimizer.state.values()
                                                for v in s.values() if torch.is_tensor(v))), event)
            del output, loss
            optimizer.zero_grad(set_to_none=True)
            guard(f"after_step_{length}")

        # Save adapter + optimizer + RNG. Compare uninterrupted vs loaded continuation.
        checkpoint = out / "adapter_resume.pt"
        state = dict(parameters={n:p.detach().cpu().clone() for n,p in allowed.items()}, optimizer=optimizer.state_dict(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        t0 = time.monotonic()
        accounting.call("checkpoint_save", lambda: torch.save(state,checkpoint), stage="adapter_resume_save")
        result["checkpoint_save_seconds"] = time.monotonic()-t0
        del state
        def one_more(stage):
            optimizer.zero_grad(set_to_none=True)
            output = forward(ids, stage, training=True, use_cache=False).logits
            loss = completion_loss(torch, output, ids, labels)
            require(bool(torch.isfinite(loss)), "resume nonfinite loss")
            accounting.call("backward", loss.backward, stage=stage)
            require(all(p.grad is None for n,p in model.named_parameters() if n not in allowed), "resume gradient leak")
            torch.nn.utils.clip_grad_norm_(list(allowed.values()),1.0,error_if_nonfinite=True)
            optimizer_step(ids.shape[1], stage)
            require(all(bool(torch.isfinite(p).all()) for p in allowed.values()), "resume nonfinite parameter")
            return {n:p.detach().cpu().clone() for n,p in allowed.items()}, float(loss.detach())
        expected, expected_loss = one_more("resume_uninterrupted")
        t0 = time.monotonic()
        def reload_checkpoint():
            saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
            with torch.no_grad():
                for n,p in allowed.items():
                    p.copy_(saved["parameters"][n])
            optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
            return saved
        saved = accounting.call("checkpoint_reload", reload_checkpoint, stage="adapter_resume_reload")
        result["checkpoint_reload_seconds"] = time.monotonic()-t0
        del saved
        actual, actual_loss = one_more("resume_reloaded")
        require(expected_loss == actual_loss and all(torch.equal(expected[n],actual[n]) for n in expected),
                "deterministic resume disagreement")
        result["checks"]["save_reload_resume_bitwise"] = True
        restore_initial()
        model.eval()
        for name,p in model.named_parameters():
            if name in allowed:
                continue
            source_name = name.replace(".q_proj.base.",".q_proj.").replace(".v_proj.base.",".v_proj.")
            require(tensor_digest(torch,p) == base_hash[source_name], "frozen base parameter drift")
        for name,expected_hash in cfg["asset_sha256"].items():
            require(sha(model_dir/name) == expected_hash, "source asset mutated")
        require(sha(model_dir/"model.safetensors") == cfg["weight_sha256"], "source weight mutated")
        result["checks"]["frozen_base_unchanged"] = True
        result["checks"]["finite_scoped_backward"] = True
        if unpadded_ce:
            result["status"] = (laptop_profile.UNPADDED512_PASS_STATUS if cfg["qualification_stage"] == "unpadded_ce512"
                                else laptop_profile.UNPADDED_PASS_STATUS)
            result["unqualified"] = ["padding", "packing", "batched inference", "KV-cache sampler",
                                     "sampled online/offline likelihood", "evaluation capture", "preservation",
                                     "E1 data/replay", "held-out capability", f"long context beyond{cfg['contexts'][-1]}"]
        elif backward_resume_only:
            result["status"] = laptop_profile.BACKWARD_PASS_STATUS
            result["unqualified"] = ["native padding/causality", "sampled likelihood", "KV-cache sampler",
                                     "E1 data/replay", "held-out capability", "long context beyond1024"]
        else:
            result["status"] = "PASS_E0_SCOPED_NO_EFFICACY_CLAIM"
            result["unqualified"] = ["KV-cache sampler", "E1 data/replay", "held-out capability", "long context beyond1024"]
        guard("complete")
    except Exception as exc:
        result["status"] = "BLOCKED_E0"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["call_accounting"] = accounting.snapshot()
        write(out/"FAILURE_RECEIPT.json", {k:v for k,v in result.items() if k != "resources"})
        event("failure", error=str(exc))
    finally:
        result["wall_seconds"] = time.monotonic()-started
        result["e1_authorized"] = False
        result["call_accounting"] = accounting.snapshot()
        write(out/"E0_RESULT.json", result)
        write(out/"RESOURCE_BASELINE.json", dict(status=result["status"], samples=result["resources"],
                                                profiles=result.get("resource_profiles")))
        write(out/"COST_LEDGER.json", {k:result[k] for k in ("wall_seconds","training_tokens","generation_tokens",
                                                           "optimizer_steps","pretrained_loads","teacher_calls",
                                                           "accounting_version","call_accounting")})
        print(json.dumps({k:result.get(k) for k in ("status","error","wall_seconds","optimizer_steps")}), flush=True)
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--go-path", required=True, type=Path)
    sys.exit(run(parser.parse_args()))
