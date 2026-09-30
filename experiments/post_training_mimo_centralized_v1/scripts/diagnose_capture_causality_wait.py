"""Fixed-shape future-token causal measurements, never capture qualification."""
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
CONFIG_RELATIVE = "configs/CAPTURE_CAUSAL_DIAGNOSTIC_WAIT_V2.json"
CONFIG_SHA256 = "b254ec5b880e9911d4ca8ba7edc27665c63ca09cbfe66ffafde76cf025fb4c65"
SCRIPT_RELATIVE = "experiments/post_training_mimo_centralized_v1/scripts/diagnose_capture_causality_wait.py"
TEST_RELATIVE = "experiments/post_training_mimo_centralized_v1/tests/test_diagnose_capture_causality_wait.py"

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
    require(type(prompt) is list and len(prompt) == 128 and generated == [16, 22, 151645], "exact failed trace required")
    require(all(type(token) is int and 0 <= token < 151936 for token in prompt + generated), "invalid trace token IDs")
    full = prompt + generated
    trials = [{"stage": "baseline", "ids": full, "mutation_start": None, "protected_positions": [127, 128, 129]}]
    for start in (128, 129, 130):
        ids = full[:start] + [17, 23, 151646][start - 128:]
        require(len(ids) == 131 and ids[:start] == full[:start] and all(ids[i] != full[i] for i in range(start, 131)), "mutation boundary")
        trials.append({"stage": "mutate_suffix_" + str(start), "ids": ids, "mutation_start": start,
                       "protected_positions": list(range(127, start))})
    return trials


def expected_counts():
    return {"model_or_base_forwards": 4, "forward_input_positions": 524, "logical_model_calls": 316,
            "direct_head_projection_calls": 0, "implicit_LM_head_calls": 4,
            "new_generated_tokens": 0, "optimizer_steps": 0, "teacher_calls": 0}


def mask_measurement(values, finite_min):
    require(type(values) is list and len(values) == 131 and all(type(row) is list and len(row) == 131 for row in values), "actual mask dimensions")
    require(type(finite_min) is float and math.isfinite(finite_min) and finite_min < -1e30, "finite BF16 mask sentinel")
    valid = sum(value == (0.0 if column <= row else finite_min)
                for row, vals in enumerate(values) for column, value in enumerate(vals))
    require(valid == 17161, "actual mask differs from triangular causal reference")
    return {"exact_triangular_match": True, "allowed_entries": 8646, "forbidden_entries": 8515,
            "total_entries": 17161, "allowed_value": 0.0, "forbidden_value": finite_min}


def position_measurement(values):
    require(values == [list(range(131))], "actual rotary positions differ")
    return {"shape": [1, 131], "position_ids": values[0], "exact_reference_match": True}


def require_hook_counts(counts):
    require(counts == {"first_decoder_pre": 1, "rotary_pre": 1, "final_norm_post": 1}, "actual witness hook invocation count")


def equality_measurement(left_hash, right_hash, max_abs, mean_abs, logprob_deltas):
    require(all(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) for value in (left_hash, right_hash)), "raw hashes required")
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in [max_abs, mean_abs] + logprob_deltas), "finite nonnegative differences required")
    return {"bitwise_equal": left_hash == right_hash, "baseline_sha256": left_hash, "mutant_sha256": right_hash,
            "max_abs": max_abs, "mean_abs": mean_abs, "selected_model_logprob_absolute_deltas": logprob_deltas,
            "passed": left_hash == right_hash and max_abs == 0 and mean_abs == 0 and all(value == 0 for value in logprob_deltas)}

def source_snapshot(root, cfg, capture):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True, timeout=15).strip(), "commit diagnostic source first")
    pins = {name: pin["sha256"] for name, pin in capture["reused_source_files"].items()}
    for relative in (SCRIPT_RELATIVE, TEST_RELATIVE, "experiments/post_training_mimo_centralized_v1/" + CONFIG_RELATIVE,
                     cfg["captured_registration"], cfg["failed_trace"], cfg["geometry_result"], cfg["derived_diagnostic"],
                     cfg["post_import_helper"], cfg["post_import_test"]): pins[relative] = sha(root / relative)
    for key in ("derived_diagnostic", "post_import_helper", "post_import_test"):
        require({"sha256": sha(root / cfg[key]), "bytes": (root / cfg[key]).stat().st_size} == cfg[key + "_pin"], "registered admission dependency drift")
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
        command = [sys.executable, "-B", "-m", "experiments.post_training_mimo_centralized_v1.scripts.diagnose_capture_causality_wait",
                   "--config", str(args.config.resolve()), "--output", str(output), "--go-path", str(supervision / "GO.json")]
        supervision.mkdir(parents=True)
        write(supervision / "LAUNCH.json", {"command": command, "config_sha256": CONFIG_SHA256, "source_sha": source["code_sha"],
            "diagnostic_script_sha256": sha(__file__), "resource_limits": cfg["limits"], "observed_gpu_processes": active.splitlines()})
        run_bounded(command, root, supervision, seconds=600, min_free_gib=1.25, max_rss_gib=2.0, go={"config_sha256": CONFIG_SHA256})
        supervision_result = json.loads((supervision / "SUPERVISOR.json").read_bytes())
        require(supervision_result["exit_code"] == 0 and supervision_result["status"] == "PASS_PROCESS_ONLY", "contained diagnostic worker failed")
        completed = json.loads((output / "DIAGNOSTIC_RESULT.json").read_bytes())
        require(completed["status"] == "COMPLETE_CAPTURE_CAUSAL_POSTIMPORT_WAIT_DIAGNOSTIC_ONLY", "contained diagnostic incomplete")
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
    require(sha(root / cfg["geometry_result"]) == cfg["geometry_result_sha256"], "geometry result drift")
    geometry = json.loads((root / cfg["geometry_result"]).read_bytes())
    require(geometry["status"] == "COMPLETE_CAPTURE_GEOMETRY_DIAGNOSTIC_ONLY"
            and geometry["reproduction"]["bulk_logits_measurement"]["raw_logits_sha256"] == cfg["reference_bulk_logits_sha256"]
            and geometry["hidden_measurements"]["held_hidden_sha256"]["full"] == cfg["reference_bulk_hidden_sha256"], "geometry endpoint contract drift")
    plan = causal_plan(cfg["prompt_ids"], cfg["generated_ids"])
    output = args.output.resolve(); require(not output.exists(), "diagnostic output exists"); output.mkdir(parents=True)
    result = {"status": "RUNNING", "experiment_id": cfg["experiment_id"], "config_sha256": CONFIG_SHA256,
        "code_sha": source["code_sha"], "failed_trace_sha256": cfg["failed_trace_sha256"], "pretrained_loads": 0,
        "new_generated_tokens": 0, "optimizer_steps": 0, "training_positions": 0, "teacher_calls": 0,
        "resources": [], "baseline": {}, "trials": [], "comparisons": [],
        "qualification_authorized": False, "DEV_collection_authorized": False, "language_grading_authorized": False,
        "E1_authorized": False, "CHECK_FINAL_access": False}
    write(output / "CONFIG.json", cfg); write(output / "SOURCE_RECEIPT.json", source)
    def event(kind, **fields):
        with (output / "EVENTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical(safe_numbers({"event": kind, "elapsed_seconds": time.monotonic() - started, **fields})).decode() + "\n")
            stream.flush(); os.fsync(stream.fileno())
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting, tensor_digest
    from experiments.post_training_mimo_centralized_v1.scripts import stream_checkpoint
    accounting = CallAccounting(event, maximum_optimizer_calls=0)
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
            "base_dtype": "bfloat16", "attention": "eager", "cache": False, "gradient": False,
            "matmul_flags": {name: {"available": hasattr(torch.backends.cuda.matmul, name),
                "value": getattr(torch.backends.cuda.matmul, name, None)} for name in (
                "allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction",
                "allow_bf16_reduced_precision_reduction_split_k", "allow_fp16_reduced_precision_reduction_split_k",
                "allow_fp16_accumulation")},
            "matmul_flags_changed": False})
        process = psutil.Process()
        def guard(stage, admission=False, waiting=False):
            torch.cuda.synchronize(); free, total = torch.cuda.mem_get_info(); limits = cfg["limits"]
            sample = {"stage": stage, "elapsed_seconds": time.monotonic() - started, "rss_gib": process.memory_info().rss / 2**30,
                "host_available_gib": psutil.virtual_memory().available / 2**30, "gpu_free_gib": free / 2**30,
                "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30, "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "disk_free_gib": psutil.disk_usage(str(output)).free / 2**30}
            sample["admission_required_gib"] = 1.25 + max(0.0, 2.0 - sample["rss_gib"])
            result["resources"].append(sample); event("resource", **sample)
            require(sample["rss_gib"] <= 2.0 and sample["host_available_gib"] >= (sample["admission_required_gib"] if admission and not waiting else 1.25), "DIAGNOSTIC_RAM_GUARD")
            require(sample["gpu_free_gib"] >= (4.0 if admission and not waiting else .5) and sample["disk_free_gib"] >= 4.0, "DIAGNOSTIC_GPU_DISK_GUARD")
            require(time.monotonic() - started <= 600, "DIAGNOSTIC_WALL_GUARD")
            return sample
        from experiments.post_training_mimo_centralized_v1.scripts import post_import_admission
        wait_observations = []
        def wait_sample():
            recheck_sources(root, source)
            return guard("post_import_wait_" + str(len(wait_observations) + 1), admission=True, waiting=True)
        def record_wait(observation):
            wait_observations.append(observation)
            event("post_import_admission_observation", observation=observation)
        result["post_import_admission_observations"] = wait_observations
        result["post_import_admission"] = post_import_admission.wait_for_admission(wait_sample,
            clock=time.monotonic, pause=time.sleep, record=record_wait, policy=cfg["post_import_wait"])
        write(output / "POST_IMPORT_ADMISSION.json", result["post_import_admission"])
        require(result["post_import_admission"]["status"] == "ADMITTED_POST_IMPORT_RESOURCE_ONLY", "POST_IMPORT_ADMISSION_TIMEOUT_NO_MODEL")
        recheck_sources(root, source)
        guard("preload_admitted", True)
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
        witness = {}
        def first_decoder_pre(module, inputs, kwargs):
            witness["counts"]["first_decoder_pre"] += 1
            mask = kwargs.get("attention_mask")
            require(mask is not None and tuple(mask.shape) == (1, 1, 131, 131) and mask.dtype == torch.bfloat16, "actual decoder mask shape/dtype")
            require(kwargs.get("past_key_values") is None and kwargs.get("past_key_value") is None, "actual decoder cache present")
            witness["mask"] = {"shape": list(mask.shape), "dtype": str(mask.dtype), "raw_sha256": tensor_digest(torch, mask),
                **mask_measurement(mask[0, 0].float().cpu().tolist(), float(torch.finfo(torch.bfloat16).min))}
        def rotary_pre(module, inputs, kwargs):
            witness["counts"]["rotary_pre"] += 1
            positions = inputs[1] if len(inputs) > 1 else kwargs.get("position_ids")
            require(positions is not None and positions.dtype == torch.long, "actual rotary position dtype")
            witness["positions"] = position_measurement(positions.cpu().tolist())
        def norm_post(module, inputs, output):
            witness["counts"]["final_norm_post"] += 1
            require(tuple(output.shape) == (1, 131, 1024) and output.dtype == torch.bfloat16, "actual final hidden shape/dtype")
            witness["hidden"] = output[0, -4:].detach().clone().contiguous()
        handles = [model.model.layers[0].register_forward_pre_hook(first_decoder_pre, with_kwargs=True),
                   model.model.rotary_emb.register_forward_pre_hook(rotary_pre, with_kwargs=True),
                   model.model.norm.register_forward_hook(norm_post)]
        baseline_logits = baseline_hidden = None
        try:
            with torch.no_grad():
                for trial in plan:
                    witness.clear(); witness["counts"] = {"first_decoder_pre": 0, "rotary_pre": 0, "final_norm_post": 0}
                    require(accounting.snapshot()["operations"]["forward"]["attempted"] < 4, "CAUSAL_FORWARD_CEILING")
                    ids = torch.tensor([trial["ids"]], dtype=torch.long, device="cuda")
                    logits = accounting.call("forward", lambda: model(ids, use_cache=False, logits_to_keep=4).logits,
                                             stage=trial["stage"], input_positions=131)[0]
                    require(tuple(logits.shape) == (4, 151936) and logits.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all()), "causal logits finite/shape/dtype")
                    require_hook_counts(witness["counts"])
                    hidden = witness.pop("hidden"); require(bool(torch.isfinite(hidden).all()), "nonfinite causal hidden")
                    selected = logits[:3].float().log_softmax(-1).gather(1, torch.tensor(cfg["generated_ids"], device="cuda")[:, None]).squeeze(1).cpu().tolist()
                    record = {"stage": trial["stage"], "mutation_start": trial["mutation_start"], "input_ids": trial["ids"],
                        "input_ids_sha256": hashlib.sha256(canonical(trial["ids"])).hexdigest(), "protected_positions": trial["protected_positions"],
                        "logits_shape": list(logits.shape), "logits_dtype": str(logits.dtype), "raw_logits_sha256": tensor_digest(torch, logits),
                        "final_hidden_shape": list(hidden.shape), "raw_hidden_sha256": tensor_digest(torch, hidden),
                        "raw_argmax_ids": logits[:3].argmax(-1).cpu().tolist(), "selected_model_logprobs": selected,
                        "witnesses": dict(witness)}
                    if trial["mutation_start"] is None:
                        require(record["raw_logits_sha256"] == cfg["reference_bulk_logits_sha256"]
                            and record["raw_hidden_sha256"] == cfg["reference_bulk_hidden_sha256"]
                            and selected == cfg["reference_bulk_model_logprobs"], "baseline differs from original bulk endpoint")
                        baseline_logits, baseline_hidden = logits.detach().clone(), hidden.clone()
                        result["baseline"] = record
                    else:
                        count = len(trial["protected_positions"])
                        for label, left, right in (("protected_logits", baseline_logits[:count], logits[:count]),
                                                   ("protected_final_hidden", baseline_hidden[:count], hidden[:count])):
                            delta = (left.float() - right.float()).abs()
                            lp_deltas = [abs(a-b) for a,b in zip(result["baseline"]["selected_model_logprobs"][:count], selected[:count])] if label == "protected_logits" else []
                            m = equality_measurement(tensor_digest(torch, left), tensor_digest(torch, right), float(delta.max()), float(delta.mean()), lp_deltas)
                            require(m["bitwise_equal"] == bool(torch.equal(left, right)), "hash versus tensor equality mismatch")
                            result["comparisons"].append({"stage": trial["stage"], "measurement": label,
                                "protected_positions": trial["protected_positions"], **m})
                    result["trials"].append(record); event("causal_trial_complete", measurement=record)
                    guard(trial["stage"])
                    del ids, logits, hidden
        finally:
            for handle in handles: handle.remove()
        result["fixed_shape_causal_controls_passed"] = len(result["comparisons"]) == 6 and all(m["passed"] for m in result["comparisons"])
        event("causal_comparisons", comparisons=result["comparisons"], passed=result["fixed_shape_causal_controls_passed"])
        del baseline_logits, baseline_hidden, witness
        require(model.lm_head.weight is model.model.embed_tokens.weight, "diagnostic final tie changed")
        for name, parameter in model.named_parameters(): require(parameter.dtype == torch.bfloat16 and not parameter.requires_grad
            and parameter.grad is None and tensor_digest(torch, parameter) == base_hash[name], "diagnostic base changed")
        for name, expected in asset_pins.items(): require(sha(model_dir / name) == expected, "diagnostic asset changed")
        require(sha(primary_path) == capture["primary_receipt_sha256"], "diagnostic primary changed")
        recheck_sources(root, source); guard("complete")
        final_accounting = accounting.snapshot()
        require(final_accounting["operations"]["forward"]["returned"] == 4 and final_accounting["last_call_id"] == 316
            and final_accounting["forward_input_positions_returned"] == 524, "causal schedule incomplete")
        result["actual_counts"] = expected_counts(); result["base_unchanged"] = True; result["source_assets_unchanged"] = True
        result["interpretation_limit"] = cfg["diagnosis_claim"]
        result["status"] = "COMPLETE_CAPTURE_CAUSAL_POSTIMPORT_WAIT_DIAGNOSTIC_ONLY"
    except Exception as error:
        result["status"] = "BLOCKED_CAPTURE_CAUSAL_POSTIMPORT_WAIT_DIAGNOSTIC"; result["error"] = str(error); result["traceback"] = traceback.format_exc()
        result["call_accounting"] = accounting.snapshot()
        event("failure", error=str(error))
        write(output / "FAILURE_RECEIPT.json", safe_numbers(result))
    finally:
        result["call_accounting"] = accounting.snapshot()
        result["wall_seconds"] = time.monotonic() - started
        write(output / "DIAGNOSTIC_RESULT.json", safe_numbers(result))
        write(output / "COST_LEDGER.json", safe_numbers({name: result[name] for name in ("wall_seconds", "pretrained_loads", "new_generated_tokens",
            "optimizer_steps", "training_positions", "teacher_calls", "call_accounting")}))
        print(json.dumps({name: result.get(name) for name in ("status", "error", "wall_seconds", "actual_counts")}), flush=True)
    return 0 if result["status"] == "COMPLETE_CAPTURE_CAUSAL_POSTIMPORT_WAIT_DIAGNOSTIC_ONLY" else 2

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
