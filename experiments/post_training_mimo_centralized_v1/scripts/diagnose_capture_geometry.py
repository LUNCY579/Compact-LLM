"""Prospective held-hidden geometry measurements, never capture qualification."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback

PACKAGE = Path(__file__).resolve().parents[1]
CONFIG_RELATIVE = "configs/CAPTURE_GEOMETRY_DIAGNOSTIC_V1.json"
CONFIG_SHA256 = "aff3d12417a7952f5fea9119ff1dad58fd1b15dbd6cd0b833cebda08293cc50c"
SCRIPT_RELATIVE = "experiments/post_training_mimo_centralized_v1/scripts/diagnose_capture_geometry.py"
TEST_RELATIVE = "experiments/post_training_mimo_centralized_v1/tests/test_diagnose_capture_geometry.py"

def require(value, message):
    if not value: raise RuntimeError(message)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def write(path, value):
    with Path(path).open("xb") as stream: stream.write(canonical(value) + b"\n")

def safe_numbers(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {name: safe_numbers(item) for name, item in value.items()}
    if isinstance(value, list): return [safe_numbers(item) for item in value]
    return value

def validate_config(cfg):
    raw = (PACKAGE / CONFIG_RELATIVE).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, "diagnostic config drift")
    require(type(cfg) is dict and canonical(cfg) == canonical(json.loads(raw)), "unregistered diagnostic config")

def load_config(path):
    require(Path(path).resolve() == (PACKAGE / CONFIG_RELATIVE).resolve(), "registered diagnostic path required")
    cfg = json.loads(Path(path).read_bytes()); validate_config(cfg)
    return cfg

def causal_plan(prompt, generated):
    require(type(prompt) is list and len(prompt) == 128 and type(generated) is list and generated == [16, 22, 151645], "exact failed three-token trace required")
    require(all(type(token) is int and 0 <= token < 151936 for token in prompt + generated), "invalid trace token IDs")
    return {"prefixes": [prompt + generated[:index] for index in range(3)], "full": prompt + generated,
            "target_positions": [127, 128, 129], "aligned_full_positions": [127, 128, 129, 130],
            "common_layout_position": 130, "generated_target_count": 3}

def projection_plan():
    return [{"hidden_origin": origin, "head_dtype": dtype, "head_rows": rows, "row_index": index}
            for dtype in ("bfloat16", "float32") for origin in ("prefix", "full")
            for rows, index in ((1, 0), (1, 1), (1, 2), (4, None))]

def expected_counts():
    return {"model_or_base_forwards": 8, "forward_input_positions": 1036, "logical_model_calls": 320,
            "direct_head_projection_calls": 16, "implicit_original_LM_head_calls": 4,
            "new_generated_tokens": 0, "optimizer_steps": 0, "teacher_calls": 0}

def endpoint_measurement(reference_logprobs, controlled_logprobs, reference_argmax, controlled_argmax):
    require(all(type(values) is list and len(values) == 3 for values in (reference_logprobs, controlled_logprobs, reference_argmax, controlled_argmax)), "three target endpoints required")
    require(all(type(value) in (int, float) and math.isfinite(value) for value in reference_logprobs + controlled_logprobs), "nonfinite endpoint logprob")
    require(all(type(token) is int and 0 <= token < 151936 for token in reference_argmax + controlled_argmax), "invalid endpoint argmax")
    delta = [controlled - reference for reference, controlled in zip(reference_logprobs, controlled_logprobs)]
    return {"reference_model_logprobs": reference_logprobs, "controlled_model_logprobs": controlled_logprobs,
        "model_logprob_deltas": delta, "max_abs_model_logprob_delta": max(abs(value) for value in delta),
        "mean_abs_model_logprob_delta": sum(abs(value) for value in delta) / 3,
        "exact_logprob_match": reference_logprobs == controlled_logprobs,
        "reference_argmax_ids": reference_argmax, "controlled_argmax_ids": controlled_argmax,
        "argmax_changed_target_rows": [index for index, (left, right) in enumerate(zip(reference_argmax, controlled_argmax)) if left != right]}

class ProjectionLedger:
    def __init__(self, emit, cap=16):
        require(type(cap) is int and cap == 16, "registered projection ceiling required")
        self.emit, self.cap, self.attempted, self.returned, self.failed = emit, cap, 0, 0, 0
        self.pending = {}
    def call(self, function, metadata):
        require(self.attempted < self.cap, "PROJECTION_CALL_CEILING")
        require(type(metadata) is dict and set(metadata) == {"hidden_origin", "head_dtype", "head_rows", "row_index", "held_hidden_sha256"}, "projection metadata fields")
        require(any(canonical({name: metadata[name] for name in ("hidden_origin", "head_dtype", "head_rows", "row_index")}) == canonical(layout)
                    for layout in projection_plan()), "unregistered projection layout")
        require(type(metadata["held_hidden_sha256"]) is str and re.fullmatch("[0-9a-f]{64}", metadata["held_hidden_sha256"]), "held hidden hash required")
        self.attempted += 1; call_id = self.attempted; self.pending[call_id] = metadata.copy()
        self.emit("projection_attempt", projection_call_id=call_id, **metadata)
        try:
            value = function()
        except BaseException as error:
            self.failed += 1; self.pending.pop(call_id)
            self.emit("projection_failed", projection_call_id=call_id, error_type=type(error).__name__, error_message=str(error), **metadata)
            raise
        self.returned += 1; self.pending.pop(call_id)
        self.emit("projection_returned", projection_call_id=call_id, **metadata)
        return value
    def snapshot(self):
        return {"ceiling": 16, "attempted": self.attempted, "returned": self.returned, "failed": self.failed,
                "pending": [{"projection_call_id": index, **values} for index, values in sorted(self.pending.items())]}

def source_snapshot(root, cfg, capture):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True, timeout=15).strip(), "commit diagnostic source first")
    pins = {name: pin["sha256"] for name, pin in capture["reused_source_files"].items()}
    for relative in (SCRIPT_RELATIVE, TEST_RELATIVE, "experiments/post_training_mimo_centralized_v1/" + CONFIG_RELATIVE,
                     cfg["captured_registration"], cfg["failed_trace"]): pins[relative] = sha(root / relative)
    for relative, expected in pins.items():
        require(sha(root / relative) == expected, "reused diagnostic source drift")
        committed = subprocess.run(["git", "show", head + ":" + relative], cwd=root, capture_output=True, timeout=15, check=False)
        require(committed.returncode == 0 and hashlib.sha256(committed.stdout).hexdigest() == expected, "uncommitted diagnostic dependency")
    for path, pin in capture["installed_runtime_source_files"].items(): require(sha(path) == pin["sha256"], "installed diagnostic runtime drift")
    return {"code_sha": head, "dirty": False, "source_files": pins, "installed_runtime_source_files": capture["installed_runtime_source_files"]}

def recheck_sources(root, source):
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15).strip() == source["code_sha"], "diagnostic HEAD changed")
    for relative, expected in source["source_files"].items(): require(sha(root / relative) == expected, "diagnostic source changed")
    for path, pin in source["installed_runtime_source_files"].items(): require(sha(path) == pin["sha256"], "installed diagnostic runtime changed")

def supervise(args, cfg, capture):
    root = PACKAGE.parents[1]; source = source_snapshot(root, cfg, capture)
    from experiments.opallo_native_collector_v1.supervisor import run_bounded
    output = args.output.resolve(); supervision = output.with_name(output.name + "_supervisor")
    require(not output.exists() and not supervision.exists(), "diagnostic run path exists")
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True).strip())
    common = common.resolve() if common.is_absolute() else (root / common).resolve()
    lock = common / "rineya-central-gpu0.lock"
    with lock.open("x", encoding="utf-8") as stream: stream.write(str(os.getpid()))
    try:
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        require(not any(term in line.lower() for line in active.splitlines() for term in ("python", "torch", "llama", "ollama", "vllm", "train", "comfy")), "competing GPU runtime")
        command = [sys.executable, "-B", "-m", "experiments.post_training_mimo_centralized_v1.scripts.diagnose_capture_geometry",
                   "--config", str(args.config.resolve()), "--output", str(output), "--go-path", str(supervision / "GO.json")]
        supervision.mkdir(parents=True)
        write(supervision / "LAUNCH.json", {"command": command, "config_sha256": CONFIG_SHA256, "source_sha": source["code_sha"],
            "diagnostic_script_sha256": sha(__file__), "resource_limits": cfg["limits"], "observed_gpu_processes": active.splitlines()})
        run_bounded(command, root, supervision, seconds=600, min_free_gib=1.25, max_rss_gib=2.0, go={"config_sha256": CONFIG_SHA256})
        return 0
    except Exception as error:
        supervision.mkdir(parents=True, exist_ok=True); write(supervision / "FAILURE_RECEIPT.json", {"status": "BLOCKED", "reason": str(error)})
        print(str(error), flush=True); return 2
    finally:
        lock.unlink()

def run(args, cfg, capture):
    root = PACKAGE.parents[1]; started = time.monotonic()
    while not args.go_path.exists() and time.monotonic() - started < 30: time.sleep(.05)
    require(args.go_path.exists() and json.loads(args.go_path.read_bytes()) == {"config_sha256": CONFIG_SHA256}, "diagnostic containment GO missing/drifted")
    source = source_snapshot(root, cfg, capture)
    failed_path = root / cfg["failed_trace"]
    require(sha(failed_path) == cfg["failed_trace_sha256"], "failed trace bytes changed")
    failed = json.loads(failed_path.read_bytes()); first = failed["fixtures"][0]
    require(failed["status"] == "BLOCKED_CAPTURE_QUALIFICATION" and failed["code_sha"] == cfg["failed_producer_commit"]
            and first["prompt_ids"] == cfg["prompt_ids"] and first["generated_ids"] == cfg["generated_ids"], "exact failed trace required")
    plan = causal_plan(cfg["prompt_ids"], cfg["generated_ids"])
    output = args.output.resolve(); require(not output.exists(), "diagnostic output exists"); output.mkdir(parents=True)
    result = {"status": "RUNNING", "experiment_id": cfg["experiment_id"], "config_sha256": CONFIG_SHA256,
        "code_sha": source["code_sha"], "failed_trace_sha256": cfg["failed_trace_sha256"], "pretrained_loads": 0,
        "new_generated_tokens": 0, "optimizer_steps": 0, "training_positions": 0, "teacher_calls": 0,
        "resources": [], "reproduction": {}, "hidden_measurements": {}, "projection_cells": [], "comparisons": [],
        "qualification_authorized": False, "DEV_collection_authorized": False, "language_grading_authorized": False,
        "E1_authorized": False, "CHECK_FINAL_access": False}
    write(output / "CONFIG.json", cfg); write(output / "SOURCE_RECEIPT.json", source)
    def event(kind, **fields):
        with (output / "EVENTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical(safe_numbers({"event": kind, "elapsed_seconds": time.monotonic() - started, **fields})).decode() + "\n")
            stream.flush(); os.fsync(stream.fileno())
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting, tensor_digest
    from experiments.post_training_mimo_centralized_v1.scripts import stream_checkpoint
    accounting = CallAccounting(event, maximum_optimizer_calls=0); projections = ProjectionLedger(event)
    try:
        event("runtime_import_start")
        require(Path(sys.executable).resolve() == Path(capture["runtime"]["executable"]).resolve()
                and sys.version.startswith(capture["runtime"]["python"] + " "), "diagnostic Python identity")
        require({name: importlib.metadata.version(name) for name in capture["runtime"]["packages"]} == capture["runtime"]["packages"], "diagnostic package identity")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        import psutil
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        event("runtime_import_complete")
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        random.seed(capture["seed"]); torch.manual_seed(capture["seed"])
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.version.cuda == capture["runtime"]["cuda"], "diagnostic BF16 CUDA identity")
        write(output / "RUNTIME_RECEIPT.json", {"python": sys.version, "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in capture["runtime"]["packages"]},
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "base_dtype": "bfloat16", "attention": "eager", "cache": False, "gradient": False})
        process = psutil.Process()
        def guard(stage, admission=False):
            torch.cuda.synchronize(); free, total = torch.cuda.mem_get_info(); limits = cfg["limits"]
            sample = {"stage": stage, "elapsed_seconds": time.monotonic() - started, "rss_gib": process.memory_info().rss / 2**30,
                "host_available_gib": psutil.virtual_memory().available / 2**30, "gpu_free_gib": free / 2**30,
                "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30, "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "disk_free_gib": psutil.disk_usage(str(output)).free / 2**30}
            sample["admission_required_gib"] = 1.25 + max(0.0, 2.0 - sample["rss_gib"])
            result["resources"].append(sample); event("resource", **sample)
            require(sample["rss_gib"] <= 2.0 and sample["host_available_gib"] >= (sample["admission_required_gib"] if admission else 1.25), "DIAGNOSTIC_RAM_GUARD")
            require(sample["gpu_free_gib"] >= (4.0 if admission else .5) and sample["disk_free_gib"] >= 4.0, "DIAGNOSTIC_GPU_DISK_GUARD")
            require(time.monotonic() - started <= 600, "DIAGNOSTIC_WALL_GUARD")
        guard("preload", True)
        model_dir = Path(capture["model_dir"]); asset_pins = {**capture["asset_sha256"], "model.safetensors": capture["weight_sha256"]}
        for name, expected in asset_pins.items(): require(sha(model_dir / name) == expected, "diagnostic model asset drift")
        primary_path = root / capture["primary_receipt"].replace("\\", "/")
        require(sha(primary_path) == capture["primary_receipt_sha256"], "diagnostic primary receipt drift")
        model_cfg = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True); model_cfg._attn_implementation = "eager"
        previous = torch.get_default_dtype(); torch.cuda.reset_peak_memory_stats()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda:0"):
                model = accounting.call("model_construct", lambda: Qwen3ForCausalLM(model_cfg), stage="model_construct")
        finally: torch.set_default_dtype(previous)
        model.eval().requires_grad_(False)
        def load():
            hashes, details = stream_checkpoint.load_streamed(torch, model, model_dir / "model.safetensors", accounting, tensor_digest)
            require(model.lm_head.weight is model.model.embed_tokens.weight and sum(parameter.numel() for parameter in model.parameters()) == 596049920, "diagnostic native tie/count")
            return hashes, details
        base_hash, loader = accounting.call("checkpoint_load", load, stage="checkpoint_load")
        result.update(pretrained_loads=1, loader_diagnostics=loader); guard("loaded")
        write(output / "MODEL_RECEIPT.json", {"weight_sha256": capture["weight_sha256"], "unique_parameters": 596049920,
            "tied": True, "tensor_sha256": base_hash, "loader_diagnostics": loader, "base_dtype": "bfloat16",
            "adapter": None, "trainable_parameters": 0, "config": model_cfg.to_dict()})
        def forward(ids, stage, keep=None, base=False):
            require(accounting.snapshot()["operations"]["forward"]["attempted"] < cfg["ceilings"]["model_or_base_forwards"], "DIAGNOSTIC_FORWARD_CEILING")
            tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
            function = (lambda: model.model(tensor, use_cache=False).last_hidden_state) if base else (lambda: model(tensor, use_cache=False, logits_to_keep=keep).logits)
            value = accounting.call("forward", function, stage=stage, input_positions=len(ids))
            require(bool(torch.isfinite(value).all()), "nonfinite diagnostic forward")
            guard(stage); return value
        def selected_logps(logits):
            targets = torch.tensor(cfg["generated_ids"], device="cuda", dtype=torch.long)
            return logits[:3].float().log_softmax(-1).gather(1, targets[:, None]).squeeze(1).cpu().tolist()
        def matrix_measure(matrix, target_ids=None):
            require(bool(torch.isfinite(matrix).all()), "nonfinite diagnostic head projection")
            target_ids = cfg["generated_ids"] if target_ids is None else target_ids
            wanted = torch.tensor(target_ids, device="cuda", dtype=torch.long); count = len(target_ids)
            return {"raw_logits_sha256": tensor_digest(torch, matrix), "shape": list(matrix.shape), "dtype": str(matrix.dtype),
                "selected_logits": matrix[:count].gather(1, wanted[:, None]).squeeze(1).float().cpu().tolist(),
                "selected_model_logprobs": matrix[:count].float().log_softmax(-1).gather(1, wanted[:, None]).squeeze(1).cpu().tolist(),
                "raw_argmax_ids": matrix[:count].argmax(-1).cpu().tolist()}
        def differences(left, right):
            delta = (left.float() - right.float()).abs()
            logprob_delta = [abs(first - second) for first, second in zip(selected_logps(left), selected_logps(right))]
            return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "l2_norm": float(torch.linalg.vector_norm(delta)),
                    "argmax_changed_rows": (left.argmax(-1) != right.argmax(-1)).nonzero().flatten().cpu().tolist(),
                    "selected_model_logprob_absolute_deltas": logprob_delta, "max_abs_model_logprob_delta": max(logprob_delta),
                    "mean_abs_model_logprob_delta": sum(logprob_delta) / 3}
        with torch.no_grad():
            online, online_logits = [], []
            for index, ids in enumerate(plan["prefixes"]):
                logits = forward(ids, f"reproduce_prefix_{len(ids)}", 1)[0]
                online.append(float(logits[0].float().log_softmax(-1)[cfg["generated_ids"][index]].item()))
                online_logits.append(matrix_measure(logits, [cfg["generated_ids"][index]]))
            bulk = forward(plan["full"], "reproduce_full_131", 4)[0]
            offline = selected_logps(bulk)
            errors = [abs(left - right) for left, right in zip(online, offline)]
            result["reproduction"] = {"online_model_logprobs": online, "bulk_model_logprobs": offline, "absolute_errors": errors,
                "max_abs": max(errors), "mean_abs": sum(errors) / 3, "online_reference_deltas": [left - right for left, right in zip(online, cfg["reference_online_model_logprobs"])],
                "bulk_reference_deltas": [left - right for left, right in zip(offline, cfg["reference_bulk_model_logprobs"])],
                "exact_reference_agreement": online == cfg["reference_online_model_logprobs"] and offline == cfg["reference_bulk_model_logprobs"],
                "original_fixed_tolerances_passed": max(errors) <= .03125 and sum(errors) / 3 <= .01,
                "online_logits_measurements": online_logits, "bulk_logits_measurement": matrix_measure(bulk)}
            event("reproduction_complete", measurement=result["reproduction"])
            prefix_rows = [forward(ids, f"hidden_prefix_{len(ids)}", base=True)[0, -1].detach().clone() for ids in plan["prefixes"]]
            full = forward(plan["full"], "hidden_full_131", base=True)[0, 127:131].detach().clone().contiguous()
            prefix = torch.stack(prefix_rows + [full[3].clone()]).contiguous()
            held = {"prefix": prefix, "full": full}
            held_hash = {name: tensor_digest(torch, values) for name, values in held.items()}
            hidden_delta = (prefix[:3].float() - full[:3].float()).abs()
            result["hidden_measurements"] = {"positions": [127, 128, 129], "common_layout_position": 130,
                "held_hidden_sha256": held_hash, "prefix_row_sha256": [tensor_digest(torch, row) for row in prefix],
                "full_row_sha256": [tensor_digest(torch, row) for row in full], "shape": [4, 1024], "dtype": "torch.bfloat16",
                "upstream_max_abs": float(hidden_delta.max()), "upstream_mean_abs": float(hidden_delta.mean()),
                "upstream_l2_norm": float(torch.linalg.vector_norm(hidden_delta))}
            write(output / "HELD_HIDDEN_MEASUREMENTS.json", result["hidden_measurements"])
            matrices = {}
            for dtype in ("bfloat16", "float32"):
                fp32_weight = model.lm_head.weight.float() if dtype == "float32" else None
                if fp32_weight is not None:
                    result["fp32_comparator"] = {"scope": "GPU-only tied-head weight and held-hidden upcast; BF16 base unchanged",
                        "weight_shape": list(fp32_weight.shape), "weight_dtype": str(fp32_weight.dtype), "device": str(fp32_weight.device),
                        "weight_bytes": fp32_weight.numel() * fp32_weight.element_size(), "weight_raw_sha256": tensor_digest(torch, fp32_weight)}
                for origin in ("prefix", "full"):
                    values = held[origin] if dtype == "bfloat16" else held[origin].float()
                    singles = []
                    for rows, index in ((1, 0), (1, 1), (1, 2), (4, None)):
                        metadata = {"hidden_origin": origin, "head_dtype": dtype, "head_rows": rows, "row_index": index, "held_hidden_sha256": held_hash[origin]}
                        inputs = (values if rows == 4 else values[index:index + 1]).unsqueeze(0)
                        function = (lambda: model.lm_head(inputs)) if dtype == "bfloat16" else (lambda: torch.nn.functional.linear(inputs, fp32_weight))
                        logits = projections.call(function, metadata)
                        require(tuple(logits.shape) == (1, rows, 151936), "direct head 3D output geometry")
                        logits = logits.reshape(rows, 151936)
                        require(bool(torch.isfinite(logits).all()), "nonfinite direct head logits")
                        cell = {**metadata, "projection_call_id": projections.attempted, "raw_logits_sha256": tensor_digest(torch, logits),
                                "projected_input_sha256": tensor_digest(torch, inputs), "projected_input_shape": list(inputs.shape),
                                "projected_input_dtype": str(inputs.dtype), "shape": list(logits.shape), "dtype": str(logits.dtype),
                                "raw_argmax_ids": logits.argmax(-1).cpu().tolist()}
                        if rows == 1:
                            token = cfg["generated_ids"][index]
                            cell.update(selected_logits=[float(logits[0, token])], selected_model_logprobs=[float(logits[0].float().log_softmax(-1)[token])])
                            singles.append(logits)
                        else:
                            cell.update(selected_logits=matrix_measure(logits)["selected_logits"], selected_model_logprobs=selected_logps(logits))
                            matrices[(dtype, origin, 4)] = logits[:3].clone()
                        result["projection_cells"].append(cell); event("projection_measurement", cell=cell)
                        guard("projection_" + str(projections.attempted))
                    matrices[(dtype, origin, 1)] = torch.cat(singles, 0)
                    require(tensor_digest(torch, held[origin]) == held_hash[origin], "held hidden values changed across layout")
                del fp32_weight
            prefix_single = [cell for cell in result["projection_cells"] if cell["hidden_origin"] == "prefix" and cell["head_dtype"] == "bfloat16" and cell["head_rows"] == 1]
            full_four = next(cell for cell in result["projection_cells"] if cell["hidden_origin"] == "full" and cell["head_dtype"] == "bfloat16" and cell["head_rows"] == 4)
            result["endpoint_parity"] = {
                "head_input_layout": "[1,rows,1024] as in original causal-LM; outputs flattened only after projection",
                "prefix_single_vs_original_online": endpoint_measurement(online,
                    [cell["selected_model_logprobs"][0] for cell in prefix_single],
                    [cell["raw_argmax_ids"][0] for cell in online_logits], [cell["raw_argmax_ids"][0] for cell in prefix_single]),
                "full_four_vs_original_bulk": endpoint_measurement(offline, full_four["selected_model_logprobs"],
                    result["reproduction"]["bulk_logits_measurement"]["raw_argmax_ids"], full_four["raw_argmax_ids"][:3])}
            for dtype in ("bfloat16", "float32"):
                for origin in ("prefix", "full"):
                    result["comparisons"].append({"comparison": "head_layout", "head_dtype": dtype, "hidden_origin": origin,
                        **differences(matrices[(dtype, origin, 1)], matrices[(dtype, origin, 4)])})
                for rows in (1, 4):
                    result["comparisons"].append({"comparison": "upstream_hidden", "head_dtype": dtype, "head_rows": rows,
                        **differences(matrices[(dtype, "prefix", rows)], matrices[(dtype, "full", rows)])})
            del matrices, held, prefix_rows, prefix, full
        require(model.lm_head.weight is model.model.embed_tokens.weight, "diagnostic final tie changed")
        for name, parameter in model.named_parameters(): require(parameter.dtype == torch.bfloat16 and not parameter.requires_grad
            and parameter.grad is None and tensor_digest(torch, parameter) == base_hash[name], "diagnostic base changed")
        for name, expected in asset_pins.items(): require(sha(model_dir / name) == expected, "diagnostic asset changed")
        require(sha(primary_path) == capture["primary_receipt_sha256"], "diagnostic primary changed")
        recheck_sources(root, source); guard("complete")
        final_accounting = accounting.snapshot()
        require(final_accounting["operations"]["forward"]["returned"] == 8 and final_accounting["last_call_id"] == 320
            and final_accounting["forward_input_positions_returned"] == 1036 and projections.snapshot()["returned"] == 16, "diagnostic schedule incomplete")
        result["actual_counts"] = expected_counts(); result["base_unchanged"] = True; result["source_assets_unchanged"] = True
        endpoints_match = all(result["endpoint_parity"][name]["exact_logprob_match"] and not result["endpoint_parity"][name]["argmax_changed_target_rows"]
                              for name in ("prefix_single_vs_original_online", "full_four_vs_original_bulk"))
        result["interpretation_limit"] = cfg["diagnosis_claim"] + ("; original failure reproduction matched exactly" if result["reproduction"]["exact_reference_agreement"] else "; original numerical failure was not reproduced exactly; original cause remains unproven")
        if not endpoints_match: result["interpretation_limit"] += "; controlled head grid endpoints differ from original causal-LM, so grid attribution is limited to the controlled projection experiment"
        result["status"] = "COMPLETE_CAPTURE_GEOMETRY_DIAGNOSTIC_ONLY"
    except Exception as error:
        result["status"] = "BLOCKED_CAPTURE_GEOMETRY_DIAGNOSTIC"; result["error"] = str(error); result["traceback"] = traceback.format_exc()
        result["call_accounting"] = accounting.snapshot(); result["projection_accounting"] = projections.snapshot()
        event("failure", error=str(error))
        write(output / "FAILURE_RECEIPT.json", safe_numbers(result))
    finally:
        result["call_accounting"] = accounting.snapshot(); result["projection_accounting"] = projections.snapshot()
        result["wall_seconds"] = time.monotonic() - started
        write(output / "DIAGNOSTIC_RESULT.json", safe_numbers(result))
        write(output / "COST_LEDGER.json", safe_numbers({name: result[name] for name in ("wall_seconds", "pretrained_loads", "new_generated_tokens",
            "optimizer_steps", "training_positions", "teacher_calls", "call_accounting", "projection_accounting")}))
        print(json.dumps({name: result.get(name) for name in ("status", "error", "wall_seconds", "actual_counts")}), flush=True)
    return 0 if result["status"] == "COMPLETE_CAPTURE_GEOMETRY_DIAGNOSTIC_ONLY" else 2

def main():
    require(os.name == "nt", "registered diagnostic containment is Windows-only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--supervise", action="store_true"); mode.add_argument("--go-path", type=Path)
    args = parser.parse_args(); cfg = load_config(args.config); root = PACKAGE.parents[1]
    require(sha(root / cfg["captured_registration"]) == cfg["captured_registration_sha256"], "original capture registration changed")
    capture = json.loads((root / cfg["captured_registration"]).read_bytes())
    return supervise(args, cfg, capture) if args.supervise else run(args, cfg, capture)

if __name__ == "__main__": sys.exit(main())
