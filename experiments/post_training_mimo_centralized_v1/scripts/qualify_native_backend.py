"""Owned two-update Titan backend mechanics qualification; no E1 launch.

Only the registered TRAIN prefix is consumed. Framework imports occur exclusively
inside the contained worker after committed-source checks. No public trainer,
evaluation bank, generation, cross-process resume or efficacy claim is provided.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
CONFIG_RELATIVE = "experiments/post_training_mimo_centralized_v1/configs/NATIVE_BACKEND_QUALIFICATION_V1.json"
CONFIG_SHA256 = "5c48a50b32cb8824d0c01a164b69375c1244a53ea60838abe466b45d882735ae"
PASS_STATUS = "PASS_NATIVE_BACKEND_TWO_UPDATE_MECHANICS_ONLY"


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def safe_numbers(value):
    if type(value) is float and not math.isfinite(value): return None
    if type(value) is dict: return {key: safe_numbers(item) for key, item in value.items()}
    if type(value) in (tuple, list): return [safe_numbers(item) for item in value]
    return value


def pin(path):
    before = path.stat(); h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024): h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), "file changed during hash")
    return {"sha256": h.hexdigest(), "bytes": after.st_size}


def write(path, value):
    raw = canonical(safe_numbers(value)) + b"\n"
    with path.open("xb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def load_config(path):
    require(path.resolve() == (ROOT / CONFIG_RELATIVE).resolve(), "registered config path required")
    raw = path.read_bytes(); require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, "qualification config drift")
    cfg = json.loads(raw); validate_prefix(cfg)
    return cfg


def validate_prefix(cfg):
    rows = cfg["prefix_rows"]
    require(type(rows) is list and len(rows) == 40 and len({row["row_key"] for row in rows}) == 40, "exact40 unique prefix rows required")
    for index, row in enumerate(rows):
        require(all(type(row[name]) is int for name in ("sequence_index", "optimizer_step", "microbatch_index")) and row["sequence_index"] == index
                and row["optimizer_step"] == index // 20 + 1 and row["microbatch_index"] == index % 20 + 1,
                "registered prefix order differs")
        require(row["row_key"].split(":", 1)[0] == row["component"] and row["component"] in
                ("acquisition", "deterministic_replay", "language_replay"), "prefix component differs")
        require(type(row["nonpadding_tokens"]) is int and 1 < row["nonpadding_tokens"] <= 512
                and type(row["supervised_tokens"]) is int and 0 < row["supervised_tokens"] < row["nonpadding_tokens"], "prefix token counts")
    expected = {"native_rows": 40, "native_input_positions": sum(row["nonpadding_tokens"] for row in rows),
        "native_supervised_positions": sum(row["supervised_tokens"] for row in rows), "reference_rows": 20,
        "reference_input_positions": sum(row["nonpadding_tokens"] for row in rows[20:]),
        "reference_supervised_positions": sum(row["supervised_tokens"] for row in rows[20:]),
        "parity_forwards": 2, "parity_input_positions": 2 * rows[0]["nonpadding_tokens"],
        "parity_supervised_positions": 2 * rows[0]["supervised_tokens"], "total_forwards": 62,
        "total_input_positions": sum(row["nonpadding_tokens"] for row in rows) + sum(row["nonpadding_tokens"] for row in rows[20:]) + 2 * rows[0]["nonpadding_tokens"],
        "total_supervised_forward_positions": sum(row["supervised_tokens"] for row in rows) + sum(row["supervised_tokens"] for row in rows[20:]) + 2 * rows[0]["supervised_tokens"]}
    require(canonical(expected) == canonical(cfg["prepared_dose"]), "registered dose differs")
    return expected


def validate_row(row, expected):
    require(type(row) is dict and row["row_id"] == expected["row_key"].split(":", 1)[1]
            and row["nonpadding_tokens"] == expected["nonpadding_tokens"] and row["completion_tokens"] == expected["supervised_tokens"]
            and row["input_token_ids_sha256"] == expected["input_token_ids_sha256"]
            and row["source_row_sha256"] == expected["source_row_sha256"]
            and hashlib.sha256(canonical(row["labels"])).hexdigest() == expected["labels_sha256"], "actual prefix array differs")
    require(row.get("component", expected["component"]) == expected["component"], "array component differs")
    return dict(row, component=expected["component"])


def source_snapshot(cfg):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True, timeout=15).strip(), "commit worker source before qualification")
    files = dict(cfg["source_pins"])
    files.update(cfg["metadata_pins"])
    for field in ("origin_capture_registration", "origin_component_registration", "prefix_trace_relative"):
        files[cfg[field]] = cfg[field.replace("_relative", "") + "_pin"] if field == "prefix_trace_relative" else cfg[field + "_pin"]
    for relative in (cfg["worker_relative"], cfg["test_relative"], CONFIG_RELATIVE): files[relative] = pin(ROOT / relative)
    for relative, expected in files.items():
        require(pin(ROOT / relative) == expected, "dependency drift:" + relative)
        if not relative.startswith("artifacts/"):
            committed = subprocess.run(["git", "show", head + ":" + relative], cwd=ROOT, capture_output=True, timeout=15, check=False)
            require(committed.returncode == 0 and hashlib.sha256(committed.stdout).hexdigest() == expected["sha256"]
                    and len(committed.stdout) == expected["bytes"], "uncommitted dependency:" + relative)
    for path, expected in cfg["installed_runtime_source_files"].items(): require(pin(Path(path)) == expected, "installed runtime drift")
    return {"producer_commit": head, "files": files, "installed_runtime_source_files": cfg["installed_runtime_source_files"],
            "code_sha256": hashlib.sha256(canonical(files)).hexdigest()}


def recheck_sources(snapshot):
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15).strip() == snapshot["producer_commit"], "worker HEAD changed")
    for path, expected in snapshot["files"].items(): require(pin(ROOT / path) == expected, "worker dependency changed")
    for path, expected in snapshot["installed_runtime_source_files"].items(): require(pin(Path(path)) == expected, "installed source changed")


def preworker_wait(*, source_check, available, clock, pause, record):
    started = previous = clock(); observations = consecutive = 0
    while True:
        source_check()
        now = clock(); require(math.isfinite(now) and now >= previous, "preworker monotonic clock required"); previous = now
        if now - started >= 30: break
        free = available(); require(type(free) in (int, float) and math.isfinite(free) and free >= 0, "finite available RAM required")
        observations += 1
        record({"observation": observations, "elapsed_seconds": clock() - started, "available_RAM_GiB": free})
        decision = clock(); require(math.isfinite(decision) and decision >= previous, "preworker clock regression"); previous = decision
        if decision - started >= 30: break
        consecutive = consecutive + 1 if free >= 3.3125 else 0
        if consecutive == 2: return {"status": "ADMITTED_PREWORKER_RESOURCE_ONLY", "observations": observations, "model_calls": 0, "E1_authorized": False}
        require(observations < 16, "preworker sample ceiling")
        pause(min(2., 30 - (decision - started)))
    return {"status": "TIMED_OUT_NO_WORKER", "observations": observations, "model_calls": 0, "E1_authorized": False}


def verify_accounting(outer, primitive, load_parity, reference, faults, native_positions, parity_targets):
    schedules = [(outer["operations"], {"forward": 40, "backward": 40, "gradient_reset": 2, "gradient_clip": 2,
        "optimizer_step": 2, "checkpoint_save": 2, "checkpoint_reload": 1}),
        (primitive["operations"], {"optimizer_construct": 1, "forward": 40, "backward": 40, "gradient_reset": 3,
        "gradient_clip": 2, "optimizer_step": 2, "checkpoint_save": 2, "checkpoint_load": 1}),
        (load_parity["operations"], {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310, "forward": 2}),
        (reference["operations"], {"reference_forward": 20, "reference_autograd_grad": 20,
        "reference_optimizer_construct": 1, "reference_gradient_clip": 1, "reference_optimizer_step": 1}),
        (faults["operations"], {name: 1 for name in ("restore_fault_adapter", "restore_fault_moment",
        "restore_fault_python_seed", "restore_fault_torch_cpu_seed", "restore_fault_cuda_seed")})]
    for observed, expected in schedules:
        for name, count in expected.items():
            require(canonical(observed[name]) == canonical({"attempted": count, "returned": count, "failed": 0, "pending": 0}), "qualification call schedule differs:" + name)
        require(all(counts["attempted"] == 0 for name, counts in observed.items() if name not in expected), "extra qualification operation")
    for positions, inputs, supervised in ((outer["positions"], 5248, 463), (native_positions, 5248, 463), (reference["positions"], 2876, 251)):
        require(canonical(positions) == canonical({"input_positions_attempted": inputs, "input_positions_returned": inputs,
            "supervised_positions_attempted": supervised, "supervised_positions_returned": supervised}), "qualification position schedule differs")
    require(load_parity["forward_input_positions_attempted"] == load_parity["forward_input_positions_returned"] == 202
            and canonical(parity_targets) == canonical({"attempted": 22, "returned": 22}), "parity position schedule differs")
    return {"forwards": 62, "input_positions": 8326, "supervised_positions": 736}


class ReferenceJournal:
    """Only this qualifier's five registered reference primitive operations."""
    def __init__(self, emit, caps):
        self.emit, self.caps, self.last = emit, caps, 0
        self.operations = {name: dict.fromkeys(("attempted", "returned", "failed", "pending"), 0) for name in caps}
        self.positions = dict.fromkeys(("input_positions_attempted", "input_positions_returned", "supervised_positions_attempted", "supervised_positions_returned"), 0)
    def call(self, name, function, *, sequence=None, input_positions=0, supervised_positions=0):
        require(type(input_positions) is int and type(supervised_positions) is int, "typed reference positions required")
        if name == "reference_forward":
            require(type(sequence) is int and 20 <= sequence < 40 and 1 < input_positions <= 512 and 0 < supervised_positions < input_positions,
                    "registered reference row envelope required")
            require(self.positions["input_positions_attempted"] + input_positions <= 2876
                    and self.positions["supervised_positions_attempted"] + supervised_positions <= 251, "reference position ceiling")
        else: require(input_positions == supervised_positions == 0, "positions apply only toreferenceforward")
        counts = self.operations[name]; require(counts["attempted"] < self.caps[name], "reference operation ceiling:" + name)
        self.last += 1; call = {"call_id": self.last, "operation": name, "sequence_index": sequence,
                              "input_positions": input_positions, "supervised_positions": supervised_positions}
        counts["attempted"] += 1; counts["pending"] += 1
        self.positions["input_positions_attempted"] += input_positions
        self.positions["supervised_positions_attempted"] += supervised_positions
        self.emit("reference_attempt", **call)
        try: value = function()
        except BaseException as error:
            counts["failed"] += 1; counts["pending"] -= 1
            self.emit("reference_failed", error_type=type(error).__name__, **call); raise
        counts["returned"] += 1; counts["pending"] -= 1
        self.positions["input_positions_returned"] += input_positions
        self.positions["supervised_positions_returned"] += supervised_positions
        self.emit("reference_returned", **call)
        return value
    def snapshot(self): return deepcopy({"last_call_id": self.last, "operations": self.operations,
                                        "positions": self.positions, "E1_authorized": False})


def run(args, cfg):
    started = time.monotonic()
    while not args.go_path.exists() and time.monotonic() - started < 30: time.sleep(.05)
    require(args.go_path.exists() and json.loads(args.go_path.read_bytes()) == {"config_sha256": CONFIG_SHA256}, "contained GO receipt required")
    source = source_snapshot(cfg)
    output = (ROOT / cfg["approved_run_relative"]).resolve(); require(not output.exists(), "write-once owned output required"); output.mkdir(parents=True)
    result = {"status": "RUNNING_NATIVE_BACKEND_MECHANICS", "config_sha256": CONFIG_SHA256,
        "producer_commit": source["producer_commit"], "native_completed_updates": 0, "completed_native_rows": 0,
        "generation_tokens": 0, "teacher_calls": 0, "E1_authorized": False, "CHECK_FINAL_access": False,
        "native_math_qualified": False, "native_resume_qualified": False, "resources": []}
    active_native_row = None
    native_positions = dict.fromkeys(("input_positions_attempted", "input_positions_returned", "supervised_positions_attempted", "supervised_positions_returned"), 0)
    parity_targets = {"attempted": 0, "returned": 0}
    def emit(kind, **fields):
        if fields.get("operation") == "forward":
            if kind in ("primitive_operation_attempt", "primitive_operation_returned"):
                require(active_native_row is not None, "physical forward mustbindactiveprefix row")
                suffix = "attempted" if kind.endswith("attempt") else "returned"
                native_positions["input_positions_" + suffix] += active_native_row["nonpadding_tokens"]
                native_positions["supervised_positions_" + suffix] += active_native_row["supervised_tokens"]
            elif kind in ("operation_attempt", "operation_returned") and fields.get("stage") in ("pre_adapter_parity", "post_adapter_parity"):
                suffix = "attempted" if kind.endswith("attempt") else "returned"
                parity_targets[suffix] += cfg["prefix_rows"][0]["supervised_tokens"]
        with (output / "EVENTS.jsonl").open("ab") as stream:
            stream.write(canonical(safe_numbers({"event": kind, "elapsed_seconds": time.monotonic() - started, **fields})) + b"\n")
            stream.flush(); os.fsync(stream.fileno())
    from experiments.post_training_mimo_centralized_v1.scripts import qualify_native as h
    from experiments.post_training_mimo_centralized_v1.scripts import stream_checkpoint, post_import_admission, e1_schedule
    from experiments.post_training_mimo_centralized_v1.scripts.e1_training_accounting import E1CallAccounting
    from experiments.post_training_mimo_centralized_v1.scripts.e1_train_reader import BoundTrainReader
    from experiments.post_training_mimo_centralized_v1.scripts.native_training_backend import NativeTrainingBackend
    load_parity = h.CallAccounting(emit, maximum_optimizer_calls=0)
    outer = E1CallAccounting(emit)
    reference = ReferenceJournal(emit, {name: value for name, value in cfg["cost_caps"].items() if name.startswith("reference_")})
    faults = ReferenceJournal(emit, {name: 1 for name in ("restore_fault_adapter", "restore_fault_moment",
        "restore_fault_python_seed", "restore_fault_torch_cpu_seed", "restore_fault_cuda_seed")})
    backend = None; reader = None; failed_ctor = None
    write(output / "CONFIG.json", cfg); write(output / "SOURCE_RECEIPT.json", source)
    try:
        require(Path(sys.executable).resolve() == Path(cfg["runtime"]["executable"]).resolve()
                and sys.version.startswith(cfg["runtime"]["python"] + " "), "pinned runtime executable required")
        require({name: importlib.metadata.version(name) for name in cfg["runtime"]["packages"]} == cfg["runtime"]["packages"], "pinned package versions required")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        emit("runtime_import_start")
        import psutil
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        emit("runtime_import_returned")
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.device_count() == 1
                and torch.version.cuda == cfg["runtime"]["cuda"], "registered CUDA0 BF16 runtime required")
        torch.set_grad_enabled(True)
        rng_shapes = {"torch_cpu": list(torch.get_rng_state().shape), "cuda:0": list(torch.cuda.get_rng_state(0).shape)}
        write(output / "RUNTIME_RECEIPT.json", {"python": sys.version, "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in cfg["runtime"]["packages"]},
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory, "rng_shapes_before_model": rng_shapes,
            "seed": cfg["seed"], "dtype": "bfloat16", "eager": True, "use_cache": False,
            "autograd_enabled": torch.is_grad_enabled(), "deterministic": True, "TF32": False,
            "matmul_flags": {name: {"available": hasattr(torch.backends.cuda.matmul, name),
                "value": getattr(torch.backends.cuda.matmul, name, None)} for name in (
                "allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction",
                "allow_bf16_reduced_precision_reduction_split_k", "allow_fp16_reduced_precision_reduction_split_k",
                "allow_fp16_accumulation")}, "matmul_flags_changed": False})
        process = psutil.Process()
        def guard(stage, admission=False, waiting=False):
            torch.cuda.synchronize(); free, _ = torch.cuda.mem_get_info()
            sample = {"stage": stage, "rss_gib": process.memory_info().rss / 2**30,
                "host_available_gib": psutil.virtual_memory().available / 2**30, "gpu_free_gib": free / 2**30,
                "disk_free_gib": psutil.disk_usage(str(output)).free / 2**30,
                "allocated_gib": torch.cuda.memory_allocated() / 2**30, "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            result["resources"].append(sample); emit("resource", **sample)
            required_host = 1.25 + max(0., 2. - sample["rss_gib"])
            require(sample["rss_gib"] <= 2. and sample["host_available_gib"] >= (required_host if admission and not waiting else 1.25)
                    and sample["gpu_free_gib"] >= (4. if admission and not waiting else .5) and sample["disk_free_gib"] >= 4., "qualification resource guard")
            require(time.monotonic() - started <= 600, "qualification whole-worker wall guard")
            return sample
        def waiting_sample(): recheck_sources(source); return guard("post_import_wait", True, True)
        wait = post_import_admission.wait_for_admission(waiting_sample, clock=time.monotonic, pause=time.sleep,
            record=lambda observation: emit("post_import_observation", observation=observation), policy=cfg["post_import_wait"])
        result["post_import_admission"] = wait; write(output / "POST_IMPORT_ADMISSION.json", wait)
        require(wait["status"] == "ADMITTED_POST_IMPORT_RESOURCE_ONLY", "post-import admission timed out; no model")
        recheck_sources(source)
        model_dir = Path(cfg["model_dir"])
        for name, expected in cfg["asset_sha256"].items(): require(pin(model_dir / name)["sha256"] == expected, "model asset drift")
        require(pin(model_dir / "model.safetensors")["sha256"] == cfg["weight_sha256"]
                and pin(ROOT / cfg["primary_receipt"])["sha256"] == cfg["primary_receipt_sha256"], "primary weight/receipt drift")
        plan_rel = "artifacts/mimo_central/e1_plan_seed101_v2/PLAN.json"; binding_rel = "artifacts/mimo_central/e1_plan_seed101_v2/DATA_BINDING.json"
        receipt_rel = "experiments/post_training_mimo_centralized_v1/receipts/train_scope_v1/TRAIN_SCOPE_RECEIPT.json"
        reader = BoundTrainReader.open_production(ROOT, plan_relative=plan_rel, plan_pin=cfg["metadata_pins"][plan_rel],
            binding_relative=binding_rel, binding_pin=cfg["metadata_pins"][binding_rel], scope_receipt_relative=receipt_rel,
            scope_receipt_pin=cfg["metadata_pins"][receipt_rel])
        plan = json.loads((ROOT / plan_rel).read_bytes()); require(plan["plan_sha256"] == cfg["historical_plan_sha256"], "plan identity drift")
        require([key for step in plan["steps"][:2] for key in step["row_keys"]] == [row["row_key"] for row in cfg["prefix_rows"]], "first40schedule drift")
        guard("preload_admitted", True)
        model_cfg = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True); model_cfg._attn_implementation = "eager"
        previous = torch.get_default_dtype(); torch.cuda.reset_peak_memory_stats()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda:0"):
                model = load_parity.call("model_construct", lambda: Qwen3ForCausalLM(model_cfg), stage="model_construct")
        finally: torch.set_default_dtype(previous)
        model.eval().requires_grad_(False)
        base_hashes, loader = load_parity.call("checkpoint_load", lambda: stream_checkpoint.load_streamed(
            torch, model, model_dir / "model.safetensors", load_parity, h.tensor_digest), stage="base_checkpoint_load")
        require(sum(p.numel() for p in model.parameters()) == 596049920 and model.lm_head.weight is model.model.embed_tokens.weight, "exact base count/tie required")
        write(output / "MODEL_RECEIPT.json", {"weight_sha256": cfg["weight_sha256"], "tensor_sha256": base_hashes, "loader": loader})
        guard("exact_base_loaded")
        def actual(expected): return validate_row(reader.get(expected["row_key"]), expected)
        def tensors(row):
            return (torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda:0"),
                    torch.tensor([row["labels"]], dtype=torch.long, device="cuda:0"))
        def independent_nll(logits, ids, labels):
            positions = (labels[0] != -100).nonzero().flatten().tolist()
            return torch.stack([-logits[0, index - 1].float().log_softmax(-1)[ids[0, index]] for index in positions]).mean()
        first = actual(cfg["prefix_rows"][0]); ids, labels = tensors(first)
        def parity(stage):
            with torch.no_grad():
                output_logits = load_parity.call("forward", lambda: model(ids, use_cache=False, logits_to_keep=0).logits,
                    stage=stage, input_positions=ids.numel())
                loss = h.completion_loss(torch, output_logits, ids, labels)
                require(abs(float(loss) - float(independent_nll(output_logits, ids, labels))) <= 1e-4, "parity causal CE mismatch")
                return {"logits_sha256": h.tensor_digest(torch, output_logits), "loss": float(loss)}
        before = parity("pre_adapter_parity"); h.install_lora(model, torch); model.eval()
        bindings = {"code_sha256": source["code_sha256"], "data_sha256": cfg["historical_data_sha256"],
            "plan_sha256": cfg["historical_plan_sha256"], "weight_sha256": cfg["weight_sha256"], "inventory_sha256": cfg["inventory_sha256"]}
        try:
            backend = NativeTrainingBackend.create(torch, model, helpers=h, guard=guard, emit=emit, repository_root=ROOT,
                approved_run_relative=cfg["approved_run_relative"], bindings=bindings, source_base_hashes=base_hashes, expected_rng_shapes=rng_shapes)
        except BaseException as error:
            failed_ctor = getattr(error, "native_backend_primitive_accounting", None); raise
        after = parity("post_adapter_parity")
        result["zero_adapter_parity"] = {"before": before, "after": after}; write(output / "PARITY_RECEIPT.json", result["zero_adapter_parity"])
        require(before["logits_sha256"] == after["logits_sha256"] and abs(before["loss"] - after["loss"])
                <= cfg["tolerances"]["completion_nll_atol"], "zero-adapter logits/loss mismatch")
        del ids, labels, first
        def save(update):
            cursor = e1_schedule.make_resume_cursor(plan, update); pre = outer.snapshot()
            state = outer.call("checkpoint_save", lambda: backend.save_checkpoint(cursor, pre), stage=f"mechanical_save_{update}")
            envelope = {"schema_version": 1, "kind": "E1_CONTROLLED_BOUNDARY_COMPONENT_CHECKPOINT_V2",
                "plan_sha256": cfg["historical_plan_sha256"], "cursor": cursor, "cost_snapshot": outer.snapshot(),
                "backend_state": state, "save_call_id": outer.last_call_id, "native_resume_qualified": False, "E1_authorized": False}
            publication = backend.publish_checkpoint(envelope)
            emit("mechanical_boundary_published", step=update, publication=publication, production_selection=False)
            return state
        save(0)
        def native_accumulate(update):
            nonlocal active_native_row
            outer.call("gradient_reset", backend.reset_gradients, stage=f"reset_{update}")
            values = []
            for expected in cfg["prefix_rows"][(update - 1) * 20:update * 20]:
                row = actual(expected)
                active_native_row = expected
                loss = outer.call("forward", lambda: backend.forward_loss(row), stage=f"native_{expected['sequence_index']}",
                    input_positions=row["nonpadding_tokens"], supervised_positions=row["completion_tokens"])
                value = backend.loss_value(loss); values.append(value)
                if expected["sequence_index"] == 0: require(abs(value - before["loss"]) <= cfg["tolerances"]["completion_nll_atol"], "firstbackend loss differs fromzero-adapter baseline")
                outer.call("backward", lambda: backend.backward(loss, scale=1 / 20), stage=f"native_{expected['sequence_index']}")
                result["completed_native_rows"] += 1
                del row, loss
                active_native_row = None
            return values
        def native_update(update):
            outer.call("gradient_clip", lambda: backend.clip_gradients(max_norm=1.0), stage=f"clip_{update}")
            outer.call("optimizer_step", lambda: backend.optimizer_step(lr=cfg["optimizer"]["step_learning_rates"][update - 1]), stage=f"update_{update}")
            result["native_completed_updates"] = backend.completed_updates
        initial_adapter_hashes = {name: h.tensor_digest(torch, p) for name, p in backend.adapters.items()}
        losses1 = native_accumulate(1); native_update(1)
        changed_adapters = [name for name, p in backend.adapters.items() if h.tensor_digest(torch, p) != initial_adapter_hashes[name]]
        result["first_update_changed_adapters"] = changed_adapters
        require(changed_adapters, "native update1 changed no adapter")
        parameters = list(backend.adapters.values())
        snapshot_params = [p.detach().clone() for p in parameters]
        snapshot_optimizer = deepcopy(backend.optimizer.state_dict())
        snapshot_rng = {"python": random.getstate(), "cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(0).clone()}
        state1 = save(1)
        def equal_tensor(left, right):
            return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
                left.detach().reshape(-1).contiguous().view(torch.uint8), right.detach().reshape(-1).contiguous().view(torch.uint8))
        def equal_tree(left, right):
            if torch.is_tensor(left): return torch.is_tensor(right) and equal_tensor(left, right)
            if type(left) is dict: return type(right) is dict and set(left) == set(right) and all(equal_tree(left[k], right[k]) for k in left)
            if type(left) in (tuple, list): return type(right) is type(left) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
            return type(left) is type(right) and left == right
        def rng_unchanged():
            return random.getstate() == snapshot_rng["python"] and equal_tensor(torch.get_rng_state(), snapshot_rng["cpu"]) and equal_tensor(torch.cuda.get_rng_state(0), snapshot_rng["cuda"])
        with torch.no_grad():
            faults.call("restore_fault_adapter", lambda: parameters[0].reshape(-1)[0].add_(1.0))
            faults.call("restore_fault_moment", lambda: backend.optimizer.state[parameters[0]]["exp_avg"].reshape(-1)[0].add_(1.0))
        faults.call("restore_fault_python_seed", lambda: random.seed(cfg["seed"] + 1))
        faults.call("restore_fault_torch_cpu_seed", lambda: torch.default_generator.manual_seed(cfg["seed"] + 1))
        faults.call("restore_fault_cuda_seed", lambda: torch.cuda.manual_seed(cfg["seed"] + 1))
        fault_witness = {"adapter_changed": not equal_tensor(snapshot_params[0], parameters[0]),
            "moment_changed": not equal_tensor(snapshot_optimizer["state"][0]["exp_avg"], backend.optimizer.state[parameters[0]]["exp_avg"]),
            "python_rng_changed": random.getstate() != snapshot_rng["python"],
            "torch_cpu_rng_changed": not equal_tensor(torch.get_rng_state(), snapshot_rng["cpu"]),
            "cuda_rng_changed": not equal_tensor(torch.cuda.get_rng_state(0), snapshot_rng["cuda"])}
        result["restore_fault_witnesses"] = fault_witness
        emit("controlled_restore_fault_witness", witnesses=fault_witness, optimizer_boundary=1, forwards=0, updates=0)
        require(all(fault_witness.values()), "restore fault didnotchangeeveryregisteredstatecomponent")
        outer.call("checkpoint_reload", lambda: backend.restore_checkpoint(state1), stage="controlled_restore_1")
        require(all(equal_tensor(a, p) for a, p in zip(snapshot_params, parameters)) and equal_tree(snapshot_optimizer, backend.optimizer.state_dict())
                and rng_unchanged() and all(p.grad is None for p in parameters), "state1 restore differsbitwise")
        result["restored_state1_bitwise"] = True
        reference_grads = None
        for expected in cfg["prefix_rows"][20:]:
            row = actual(expected); ids, labels = tensors(row)
            logits = reference.call("reference_forward", lambda: model(ids, use_cache=False, logits_to_keep=0).logits, sequence=expected["sequence_index"],
                input_positions=expected["nonpadding_tokens"], supervised_positions=expected["supervised_tokens"])
            loss = h.completion_loss(torch, logits, ids, labels)
            require(abs(float(loss.detach()) - float(independent_nll(logits, ids, labels).detach())) <= 1e-4, "reference causalCE mismatch")
            grads = reference.call("reference_autograd_grad", lambda: torch.autograd.grad(loss / 20, parameters,
                retain_graph=False, create_graph=False, allow_unused=False), sequence=expected["sequence_index"])
            require(all(g.dtype == torch.float32 and bool(torch.isfinite(g).all()) for g in grads), "reference gradients invalid")
            if reference_grads is None: reference_grads = [g.detach().clone() for g in grads]
            else:
                for accumulated, gradient in zip(reference_grads, grads): accumulated.add_(gradient)
            del row, ids, labels, logits, loss, grads
            guard("reference_gradient_" + str(expected["sequence_index"]))
        require(rng_unchanged() and all(equal_tensor(a, p) for a, p in zip(snapshot_params, parameters))
                and equal_tree(snapshot_optimizer, backend.optimizer.state_dict()) and all(p.grad is None for p in parameters), "referencepass mutated state/RNG/gradients")
        reference_nonzero = any(bool((gradient != 0).any()) for gradient in reference_grads)
        result["reference_gradients_nonzero"] = reference_nonzero
        require(reference_nonzero, "reference gradients identicallyzero")
        losses2 = native_accumulate(2)
        native_nonzero = any(bool((parameter.grad != 0).any()) for parameter in parameters)
        result["native_gradients_nonzero"] = native_nonzero
        require(native_nonzero, "native accumulated gradients identicallyzero")
        comparisons = []
        for name, p, gradient in zip(backend.adapters, parameters, reference_grads):
            delta = (p.grad - gradient).abs()
            matches = bool(torch.allclose(p.grad, gradient, atol=cfg["tolerances"]["gradient_atol"], rtol=cfg["tolerances"]["gradient_rtol"], equal_nan=False))
            comparisons.append({"name": name, "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                                "finite": bool(torch.isfinite(delta).all()), "passed": matches})
        result["gradient_comparisons"] = comparisons
        require(all(item["passed"] and item["finite"] for item in comparisons), "sequence-mean referencegradient mismatch")
        ref_params = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        ref_optimizer = reference.call("reference_optimizer_construct", lambda: torch.optim.AdamW(ref_params, lr=1e-4,
            betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=None, fused=None, capturable=False, differentiable=False))
        ref_optimizer.load_state_dict(deepcopy(snapshot_optimizer))
        for target, original in zip(ref_params, parameters): target.grad = original.grad.detach().clone()
        reference.call("reference_gradient_clip", lambda: torch.nn.utils.clip_grad_norm_(ref_params, 1.0, error_if_nonfinite=True))
        ref_optimizer.param_groups[0]["lr"] = cfg["optimizer"]["step_learning_rates"][1]
        reference.call("reference_optimizer_step", ref_optimizer.step)
        native_update(2)
        changed_adapters2 = [name for name, prior, current in zip(backend.adapters, snapshot_params, parameters)
                             if not equal_tensor(prior, current)]
        result["changed_adapters_step2"] = changed_adapters2
        require(changed_adapters2, "native update2 changed no adapter")
        require(all(equal_tensor(a, p) for a, p in zip(ref_params, parameters)) and equal_tree(ref_optimizer.state_dict(), backend.optimizer.state_dict())
                and rng_unchanged(), "copied-gradient clip/Adam continuation differsbitwise")
        result["copied_gradient_optimizer_continuation_bitwise"] = True
        result["sequence_mean_losses"] = [sum(losses1) / 20, sum(losses2) / 20]
        del reference_grads, ref_params, ref_optimizer, snapshot_params, snapshot_optimizer, snapshot_rng
        backend._verify_base(); reader.verify_unchanged(full_hash=True)
        for name, expected in cfg["asset_sha256"].items(): require(pin(model_dir / name)["sha256"] == expected, "finalmodelasset changed")
        require(pin(model_dir / "model.safetensors")["sha256"] == cfg["weight_sha256"], "finalprimary changed")
        recheck_sources(source); guard("qualification_complete")
        result["actual_forward_counts"] = verify_accounting(outer.snapshot(), backend.primitive_operation_snapshot(),
            load_parity.snapshot(), reference.snapshot(), faults.snapshot(), native_positions, parity_targets)
        result["status"] = PASS_STATUS
    except BaseException as error:
        result["status"] = "FAILED_NATIVE_BACKEND_MECHANICS_ONLY"
        result["error_type"], result["error"], result["traceback"] = type(error).__name__, str(error), traceback.format_exc()
        result["automatic_retry_authorized"] = False
    finally:
        if reader is not None: reader.close()
        result["outer_method_accounting"] = outer.snapshot(); result["load_parity_accounting"] = load_parity.snapshot()
        result["reference_primitive_accounting"] = reference.snapshot()
        result["restore_fault_accounting"] = faults.snapshot()
        result["native_primitive_accounting"] = backend.primitive_operation_snapshot() if backend is not None else failed_ctor
        result["native_primitive_positions"] = dict(native_positions); result["parity_supervised_positions"] = dict(parity_targets)
        if backend is not None: result["native_completed_updates"] = backend.completed_updates
        result["wall_seconds"] = time.monotonic() - started
        if result["status"] != PASS_STATUS: write(output / "FAILURE_RECEIPT.json", result)
        write(output / "QUALIFICATION_RESULT.json", result)
        print(json.dumps({name: result[name] for name in ("status", "native_completed_updates", "completed_native_rows", "wall_seconds")}), flush=True)
    return 0 if result["status"] == PASS_STATUS else 2


def supervise(args, cfg):
    controller = ROOT / (cfg["approved_run_relative"] + "_controller")
    require(not controller.exists(), "write-once controller required"); controller.mkdir(parents=True)
    lock = None; launched = False
    try:
        source = source_snapshot(cfg)
        import psutil
        def record(sample):
            with (controller / "ADMISSION_SAMPLES.jsonl").open("ab") as stream:
                stream.write(canonical(sample) + b"\n"); stream.flush(); os.fsync(stream.fileno())
        admitted = preworker_wait(source_check=lambda: recheck_sources(source), available=lambda: psutil.virtual_memory().available / 2**30,
            clock=time.monotonic, pause=time.sleep, record=record)
        write(controller / "ADMISSION_RESULT.json", admitted)
        if admitted["status"] != "ADMITTED_PREWORKER_RESOURCE_ONLY": return 2
        from experiments.opallo_native_collector_v1.supervisor import run_bounded
        common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip())
        common = common.resolve() if common.is_absolute() else (ROOT / common).resolve()
        target_lock = common / "rineya-central-gpu0.lock"
        with target_lock.open("x", encoding="utf-8") as stream: stream.write(str(os.getpid()))
        lock = target_lock
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        require(not any(term in line.lower() for line in active.splitlines() for term in ("python", "torch", "llama", "ollama", "vllm", "train", "comfy")), "competing GPU runtime")
        supervision = ROOT / (cfg["approved_run_relative"] + "_supervisor"); require(not supervision.exists(), "supervisor output exists")
        supervision.mkdir(parents=True)
        command = [sys.executable, "-B", "-m", cfg["worker_module"], "--config", str(args.config.resolve()), "--go-path", str(supervision / "GO.json")]
        write(supervision / "LAUNCH.json", {"command": command, "source": source, "config_sha256": CONFIG_SHA256, "limits": cfg["limits"]})
        recheck_sources(source)
        require(psutil.virtual_memory().available / 2**30 >= 3.3125, "fresh prelaunch RAM budget failed")
        launched = True
        run_bounded(command, ROOT, supervision, seconds=600, min_free_gib=1.25, max_rss_gib=2., go={"config_sha256": CONFIG_SHA256})
        terminal = json.loads((supervision / "SUPERVISOR.json").read_bytes())
        require(terminal["status"] == "PASS_PROCESS_ONLY" and terminal["exit_code"] == 0, "contained qualification failed")
        outcome = json.loads((ROOT / cfg["approved_run_relative"] / "QUALIFICATION_RESULT.json").read_bytes())
        require(outcome["status"] == PASS_STATUS, "worker did not qualify mechanics")
        return 0
    except BaseException as error:
        write(controller / "FAILURE_RECEIPT.json", {"status": "FAILED_CONTROLLER_NO_APPROVAL", "error_type": type(error).__name__,
            "error": str(error), "worker_launch_attempted": launched, "model_calls": None if launched else 0, "E1_authorized": False})
        print(str(error), flush=True)
        return 2
    finally:
        if lock is not None: lock.unlink()


def main():
    require(os.name == "nt", "owned Windows containment required")
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True); modes.add_argument("--supervise", action="store_true"); modes.add_argument("--go-path", type=Path)
    args = parser.parse_args(); cfg = load_config(args.config)
    return supervise(args, cfg) if args.supervise else run(args, cfg)


if __name__ == "__main__": sys.exit(main())
