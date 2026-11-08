"""One contained six-forward bare-Titan exact-prefix qualification; never DEV/E1."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
import uuid

from . import exact_prefix_capture as control
from . import greedy_capture as old

OUTPUT_RELATIVE = "artifacts/mimo_central/exact_prefix_fixed_trace_v1_20261001"


@contextmanager
def exclusive_lease(path, payload):
    raw = control.canonical(payload) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        yield
    finally:
        # Release only our byte-identical lease; never reclaim or remove a peer's.
        control.require(path.read_bytes() == raw, "GPU lease ownership changed; preserve peer lease")
        path.unlink()


def overlap_snapshot(psutil):
    me = psutil.Process()
    mine = {me.pid, *(p.pid for p in me.parents()), *(p.pid for p in me.children(recursive=True))}
    workers = []
    for process in psutil.process_iter(["pid", "name"]):
        if process.pid in mine or not any(x in (process.info["name"] or "").lower() for x in ("python", "uv", "llama", "ollama", "vllm")):
            continue
        try:
            command = process.cmdline()
            if any(term in " ".join(command).lower() for term in ("chatbubble", "chat-bubble", "supaexperiment", "rineyaarchitecture", "mimo", "comp_rs")):
                workers.append({"pid": process.pid, "name": process.info["name"], "command": command})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            control.require(False, "unreadable possible research worker; no launch")
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    for line in gpu.splitlines():
        pid, _, name = line.partition(",")
        if not pid.strip().isdigit() or int(pid.strip()) in mine:
            continue
        control.require(not any(x in name.lower() for x in ("python", "torch", "llama", "ollama", "vllm", "train", "comfy")), "competing GPU runtime")
        try:
            control.require(psutil.Process(int(pid.strip())).memory_info().rss < .5 * 2**30, "foreign GPU process exceeds conservative RSS floor")
        except psutil.NoSuchProcess:
            continue
    control.require(not workers, "active matching research worker; no overlap")
    return {"matching_research_workers": workers, "observed_gpu_processes": gpu.splitlines()}


def validate_go_binding(go, source):
    fields = {"config_sha256", "source_sha", "source_snapshot_sha256", "supervisor_pid", "supervisor_created",
              "lease_token", "job_handle", "worker_root_pid", "worker_root_created"}
    control.require(type(go) is dict and set(go) == fields, "complete current supervisor GO binding required")
    control.require(go["config_sha256"] == control.CONFIG_SHA256 and go["source_sha"] == source["code_sha"]
                    and go["source_snapshot_sha256"] == hashlib.sha256(control.canonical(source)).hexdigest(), "GO source/config binding mismatch")
    control.require(all(type(go[key]) is int and go[key] > 0 for key in ("supervisor_pid", "worker_root_pid", "job_handle"))
                    and all(type(go[key]) is float and math.isfinite(go[key]) and go[key] > 0 for key in ("supervisor_created", "worker_root_created"))
                    and args_token_valid(go["lease_token"]), "current process/Job/attempt identities required")


def verify_worker_owner(go, source, root, supervision):
    """Verify the actual supervisor-owned Job handle before framework imports."""
    validate_go_binding(go, source)
    import ctypes as c
    from ctypes import wintypes as w
    import psutil
    me = psutil.Process()
    ancestors = {p.pid: p for p in me.parents()}
    control.require(type(go["supervisor_pid"]) is int and go["supervisor_pid"] in ancestors, "owned supervisor is not a live ancestor")
    supervisor = ancestors[go["supervisor_pid"]]
    control.require(supervisor.create_time() == go["supervisor_created"] and supervisor.is_running(), "supervisor PID identity changed")
    control.require(type(go["worker_root_pid"]) is int and go["worker_root_pid"] in {me.pid, *ancestors}, "owned worker root lineage mismatch")
    worker_root = me if me.pid == go["worker_root_pid"] else ancestors[go["worker_root_pid"]]
    control.require(worker_root.create_time() == go["worker_root_created"], "worker root PID identity changed")
    control.require(args_token_valid(go["lease_token"]), "current attempt token required")
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True, timeout=15).strip())
    common = common.resolve() if common.is_absolute() else (root / common).resolve()
    leases = [old.parse(path.read_bytes()) for path in (Path(r"C:\Users\clclc\AppData\Local\Temp\rineya-gpu-0.lock"), common / "rineya-central-gpu0.lock")]
    control.require(leases[0] == leases[1] and leases[0]["pid"] == go["supervisor_pid"]
                    and leases[0]["token"] == go["lease_token"] and leases[0]["supervisor_created"] == go["supervisor_created"], "owned live GPU lease mismatch")
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.DuplicateHandle.argtypes = [w.HANDLE, w.HANDLE, w.HANDLE, c.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD]
    kernel.DuplicateHandle.restype = w.BOOL
    kernel.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
    kernel.IsProcessInJob.restype = w.BOOL
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.CloseHandle.restype = w.BOOL
    control.require(type(go["job_handle"]) is int and go["job_handle"] > 0, "specific owned Job handle required")
    parent = kernel.OpenProcess(0x0040, False, go["supervisor_pid"])  # PROCESS_DUP_HANDLE
    control.require(bool(parent), "cannot open owned supervisor for Job identity")
    copied_job = w.HANDLE()
    try:
        control.require(bool(kernel.DuplicateHandle(parent, w.HANDLE(go["job_handle"]), kernel.GetCurrentProcess(),
                        c.byref(copied_job), 0, False, 0x2)), "owned Job handle duplication failed")
        member = w.BOOL()
        control.require(bool(kernel.IsProcessInJob(kernel.GetCurrentProcess(), copied_job, c.byref(member)))
                        and bool(member.value), "worker is not in the specific supervisor Job")
    finally:
        if copied_job:
            kernel.CloseHandle(copied_job)
        kernel.CloseHandle(parent)
    return {"source_sha": go["source_sha"], "source_snapshot_sha256": go["source_snapshot_sha256"],
            "supervisor_pid": go["supervisor_pid"], "worker_root_pid": go["worker_root_pid"],
            "lease_token": go["lease_token"], "specific_owned_windows_job_verified": True,
            "go_path": str(supervision / "GO.json")}


def validate_returned_run(completed, output, cfg, source, root):
    control.require(completed["status"] == control.PASS_STATUS and completed["code_sha"] == source["code_sha"]
                    and completed["config_sha256"] == control.CONFIG_SHA256, "contained exact-prefix result source/config/status mismatch")
    control.require(control.canonical(old.parse((output / "SOURCE_RECEIPT.json").read_bytes())) == control.canonical(source)
                    and control.canonical(old.parse((output / "CONFIG.json").read_bytes())) == control.canonical(cfg), "returned source/config receipts differ from supervisor binding")
    old.recheck_sources(root, source)


def args_token_valid(token):
    return type(token) is str and len(token) == 32 and all(x in "0123456789abcdef" for x in token)


def supervise(args, cfg, capture, source, root):
    from experiments.opallo_native_collector_v1 import supervisor as watchdog
    import psutil
    output = args.output.resolve()
    supervision = output.with_name(output.name + "_supervisor")
    control.require(not output.exists() and not supervision.exists(), "single registered attempt path already exists")
    supervision.mkdir(parents=True)
    observations = []
    launched = False
    try:
        common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True, timeout=15).strip())
        common = common.resolve() if common.is_absolute() else (root / common).resolve()
        payload = {"pid": os.getpid(), "worker_pids": [], "token": uuid.uuid4().hex, "created": time.time(),
                   "supervisor_created": psutil.Process().create_time(), "label": cfg["experiment_id"]}
        global_lock = Path(r"C:\Users\clclc\AppData\Local\Temp\rineya-gpu-0.lock")
        local_lock = common / "rineya-central-gpu0.lock"
        with exclusive_lease(global_lock, payload), exclusive_lease(local_lock, payload):
            overlap = overlap_snapshot(psutil)
            started = time.monotonic()
            consecutive = 0
            while True:
                old.recheck_sources(root, source)
                gpu_raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
                control.require(len(gpu_raw.splitlines()) == 1, "one GPU card required")
                sample = {"elapsed_seconds": time.monotonic() - started, "host_available_gib": psutil.virtual_memory().available / 2**30,
                          "gpu_free_gib": float(gpu_raw) / 1024, "disk_free_gib": psutil.disk_usage(str(supervision)).free / 2**30}
                observations.append(sample)
                with (supervision / "ADMISSION_SAMPLES.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(control.canonical(sample).decode() + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                admitted = sample["host_available_gib"] >= 3.3125 and sample["gpu_free_gib"] >= 4 and sample["disk_free_gib"] >= 4
                consecutive = consecutive + 1 if admitted else 0
                control.require(time.monotonic() - started < 30, "PREWORKER_ADMISSION_TIMEOUT_NO_MODEL")
                if consecutive == 2:
                    break
                time.sleep(2)
            # Recheck the physical card after both exclusive leases and admission.
            overlap = overlap_snapshot(psutil)
            old.recheck_sources(root, source)
            control.require(psutil.virtual_memory().available / 2**30 >= 3.3125, "fresh preworker RAM admission failed")
            command = [sys.executable, "-B", "-m", "experiments.post_training_mimo_centralized_v1.scripts.qualify_exact_prefix",
                       "--config", str(args.config.resolve()), "--output", str(output), "--go-path", str(supervision / "GO.json")]
            control.write(supervision / "LAUNCH.json", {"source_sha": source["code_sha"], "config_sha256": control.CONFIG_SHA256,
                          "command": command, "leases": [str(global_lock), str(local_lock)], "resource_limits": cfg["limits"],
                          "admission_observations": observations, "overlap_check": overlap, "E1_authorized": False})
            go = {"config_sha256": control.CONFIG_SHA256, "source_sha": source["code_sha"],
                  "source_snapshot_sha256": hashlib.sha256(control.canonical(source)).hexdigest(),
                  "supervisor_pid": os.getpid(), "supervisor_created": payload["supervisor_created"], "lease_token": payload["token"]}
            original_job_factory = watchdog.WindowsJob

            def owned_job_factory(process):
                job = original_job_factory(process)  # Assignment succeeds before publishing GO.
                try:
                    go.update(job_handle=int(job.handle), worker_root_pid=process.pid,
                              worker_root_created=psutil.Process(process.pid).create_time())
                    return job
                except BaseException:
                    job.close()
                    raise

            launched = True
            watchdog.WindowsJob = owned_job_factory
            try:
                watchdog.run_bounded(command, root, supervision, seconds=600, min_free_gib=1.25, max_rss_gib=2.0, go=go)
            finally:
                watchdog.WindowsJob = original_job_factory
            completed = old.parse((output / "CAPTURE_RESULT.json").read_bytes())
            validate_returned_run(completed, output, cfg, source, root)
            print(json.dumps({"status": completed["status"], "output": str(output)}), flush=True)
            return 0
    except Exception as error:
        recovered = control.recover_worker_costs(output, supervision, cfg, source, str(error))
        control.write(supervision / "FAILURE_RECEIPT.json", {"status": "BLOCKED_EXACT_PREFIX_SUPERVISOR", "error": str(error),
                      "supervisor_invoked": launched, "admission_observations": observations,
                      "call_accounting": recovered["call_accounting"], "worker_recovery": "worker_recovery", "E1_authorized": False})
        print(json.dumps({"status": "BLOCKED_EXACT_PREFIX_SUPERVISOR", "error": str(error)}), flush=True)
        return 2


def run(args, cfg, capture, source, root, preserved_bulk):
    started = time.monotonic()
    while not args.go_path.exists() and time.monotonic() - started < 30:
        time.sleep(.05)
    supervision = args.output.resolve().with_name(args.output.name + "_supervisor")
    control.require(args.go_path.resolve() == (supervision / "GO.json").resolve() and args.go_path.exists(), "owned supervisor GO path required")
    owner = verify_worker_owner(old.parse(args.go_path.read_bytes()), source, root, supervision)
    output = args.output.resolve()
    control.require(not output.exists(), "exact-prefix worker output exists")
    output.mkdir(parents=True)
    result = {"schema_version": 1, "status": "RUNNING", "experiment_id": cfg["experiment_id"], "code_sha": source["code_sha"],
              "config_sha256": control.CONFIG_SHA256, "dirty_at_launch": False, "evidence_level": "MEASURED / REPRODUCED",
              "pretrained_loads": 0, "generation_tokens": 0, "optimizer_steps": 0, "training_positions": 0, "teacher_calls": 0,
              "resources": [], "trace": {}, "preserved_bulk_likelihood": preserved_bulk, "general_capture_qualified": False,
              "DEV_collection_authorized": False, "language_grading_authorized": False, "E1_authorized": False,
              "CHECK_FINAL_access": False, "interpretation_limit": cfg["interpretation_limit"]}
    control.write(output / "CONFIG.json", cfg)
    control.write(output / "SOURCE_RECEIPT.json", source)
    control.write(output / "CONTAINMENT_RECEIPT.json", owner)

    def event(kind, **fields):
        with (output / "EVENTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(control.canonical(old.safe_numbers({"event": kind, "elapsed_seconds": time.monotonic() - started, **fields})).decode() + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    from .qualify_native import CallAccounting, tensor_digest, verify_call_journal
    from . import stream_checkpoint
    accounting = CallAccounting(event, maximum_optimizer_calls=0)
    calls = old.BudgetedCalls(accounting, cfg)
    try:
        event("runtime_import_start")
        control.require(Path(sys.executable).resolve() == Path(capture["runtime"]["executable"]).resolve()
                        and sys.version.startswith(capture["runtime"]["python"] + " "), "pinned Python identity")
        packages = {key: importlib.metadata.version(key) for key in capture["runtime"]["packages"]}
        control.require(packages == capture["runtime"]["packages"], "pinned package identity")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        import psutil
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        event("runtime_import_complete")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(capture["seed"])
        torch.manual_seed(capture["seed"])
        control.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                        and torch.version.cuda == capture["runtime"]["cuda"], "pinned BF16 CUDA identity")
        control.write(output / "RUNTIME_RECEIPT.json", {"python": sys.version, "executable": sys.executable, "packages": packages,
                      "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "base_dtype": "bfloat16", "attention": "eager",
                      "use_cache": False, "gradient": False, "adapter": None, "torch_threads": 2, "allow_tf32": False,
                      "matmul_flags": {key: {"available": hasattr(torch.backends.cuda.matmul, key), "value": getattr(torch.backends.cuda.matmul, key, None)}
                          for key in ("allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction", "allow_fp16_accumulation")},
                      "reduction_flags_changed": False})
        process = psutil.Process()

        def guard(stage, admission=False, waiting=False):
            torch.cuda.synchronize()
            free, _ = torch.cuda.mem_get_info()
            sample = {"stage": stage, "elapsed_seconds": time.monotonic() - started, "rss_gib": process.memory_info().rss / 2**30,
                      "host_available_gib": psutil.virtual_memory().available / 2**30, "gpu_free_gib": free / 2**30,
                      "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30, "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "disk_free_gib": psutil.disk_usage(str(output)).free / 2**30}
            sample["admission_required_gib"] = 1.25 + max(0.0, 2.0 - sample["rss_gib"])
            result["resources"].append(sample)
            event("resource", **sample)
            control.require(sample["rss_gib"] <= 2 and sample["host_available_gib"] >= (sample["admission_required_gib"] if admission and not waiting else 1.25), "EXACT_PREFIX_RAM_GUARD")
            control.require(sample["gpu_free_gib"] >= (4 if admission and not waiting else .5) and sample["disk_free_gib"] >= 4, "EXACT_PREFIX_GPU_DISK_GUARD")
            control.require(time.monotonic() - started <= 600, "EXACT_PREFIX_WALL_GUARD")
            return sample

        from . import post_import_admission
        result["post_import_observations"] = []

        def wait_sample():
            old.recheck_sources(root, source)
            return guard("post_import_wait", admission=True, waiting=True)

        def wait_record(value):
            result["post_import_observations"].append(value)
            event("post_import_observation", observation=value)

        admission = post_import_admission.wait_for_admission(wait_sample, clock=time.monotonic, pause=time.sleep,
                      record=wait_record, policy=cfg["post_import_wait"])
        result["post_import_admission"] = admission
        control.write(output / "POST_IMPORT_ADMISSION.json", admission)
        control.require(admission["status"] == "ADMITTED_POST_IMPORT_RESOURCE_ONLY", "POST_IMPORT_ADMISSION_TIMEOUT_NO_MODEL")
        old.recheck_sources(root, source)
        guard("preload", True)
        model_dir = Path(capture["model_dir"])
        asset_pins = {**capture["asset_sha256"], "model.safetensors": capture["weight_sha256"]}
        for name, expected in asset_pins.items():
            control.require(control.file_identity(model_dir / name)["sha256"] == expected, "model asset drift: " + name)
        primary = old.parse((root / capture["primary_receipt"]).read_bytes())
        control.require(control.file_identity(root / capture["primary_receipt"])["sha256"] == capture["primary_receipt_sha256"]
                        and primary["weight_sha256"] == capture["weight_sha256"] and primary["parameter_count"] == 596049920
                        and Path(primary["checkpoint_path"]).resolve() == model_dir.resolve(), "frozen primary identity")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        template = old.parse((model_dir / "tokenizer_config.json").read_bytes())["chat_template"]
        control.require(tokenizer.chat_template == template and tokenizer.eos_token_id == 151645, "tokenizer/template identity")
        suffix = tokenizer.apply_chat_template([{"role": "user", "content": capture["fixtures"]["user_message"]}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        filler = tokenizer.encode(capture["fixtures"]["filler"], add_special_tokens=False)
        control.require(bool(filler) and len(suffix) <= 128, "mechanical tokenizer fixture shape")
        prompt = (filler * (128 // len(filler) + 1))[:128 - len(suffix)] + suffix
        control.require(prompt == cfg["prompt_ids"] and old.token_hash(prompt) == cfg["prompt_ids_sha256"], "complete frozen prompt reconstruction")
        control.write(output / "TOKENIZER_RECEIPT.json", {"template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                      "prompt_ids": prompt, "prompt_ids_sha256": old.token_hash(prompt), "eos_token_id": tokenizer.eos_token_id,
                      "skip_special_tokens": False, "clean_up_tokenization_spaces": False, "enable_thinking": False})
        guard("before_construct", True)
        model_cfg = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True)
        model_cfg._attn_implementation = "eager"
        control.require(model_cfg.rope_theta == 1000000 and model_cfg.num_hidden_layers == 28, "architecture identity")
        previous = torch.get_default_dtype()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda:0"):
                model = calls.call("model_construct", lambda: Qwen3ForCausalLM(model_cfg), stage="model_construct")
        finally:
            torch.set_default_dtype(previous)
        model.eval().requires_grad_(False)

        def load():
            hashes, details = stream_checkpoint.load_streamed(torch, model, model_dir / "model.safetensors", calls, tensor_digest)
            control.require(model.lm_head.weight is model.model.embed_tokens.weight
                            and sum(p.numel() for p in model.parameters()) == 596049920, "native tie/parameter count")
            return hashes, details

        base_hash, loader = calls.call("checkpoint_load", load, stage="checkpoint_load")
        result.update(pretrained_loads=1, loader_diagnostics=loader)
        guard("loaded")
        control.write(output / "MODEL_RECEIPT.json", {"weight_sha256": capture["weight_sha256"], "unique_parameters": 596049920,
                      "serialized_tensors": 310, "tensor_sha256": base_hash, "tied": True, "adapter": None, "trainable_parameters": 0,
                      "loader_diagnostics": loader, "config": model_cfg.to_dict()})

        def forward(ids, target, stage):
            control.require(type(ids) is list and 128 <= len(ids) <= 130, "registered exact-prefix forward shape")
            old.token_hash(ids)
            guard("before_" + stage)
            tensor = torch.tensor([ids], device="cuda", dtype=torch.long)
            logits = calls.call("forward", lambda: model(tensor, use_cache=False, logits_to_keep=1).logits,
                                stage=stage, input_positions=len(ids))
            control.require(tuple(logits.shape) == (1, 1, 151936) and logits.dtype == torch.bfloat16
                            and bool(torch.isfinite(logits).all()), "finite BF16 full-vocabulary logits required")
            scores = logits[0, -1].float()
            maximum = scores.max()
            first = int((scores == maximum).nonzero()[0, 0].item())
            selected = int(scores.argmax(-1).item()) if target is None else target
            logprob = float(scores.log_softmax(-1)[selected].item())
            control.require(math.isfinite(logprob) and logprob <= 0, "finite raw-model selected likelihood required")
            measured = {"input_ids": list(ids), "input_ids_sha256": old.token_hash(ids), "target_id": selected,
                        "model_logprob": logprob, "selected_raw_logit": float(scores[selected].item()),
                        "raw_max_logit": float(maximum.item()), "first_maximum_token_id": first,
                        "raw_logits_sha256": tensor_digest(torch, logits), "logits_shape": list(logits.shape), "logits_to_keep": 1}
            del tensor, logits, scores, maximum
            # The durable primitive journal already retained this returned call.
            event("native_forward_measurement", stage=stage, measurement=measured)
            guard("after_" + stage)
            return measured

        with torch.no_grad():
            control.capture_trace(cfg, forward, event, result["trace"])
        result["trace"].update(old.render_response(tokenizer, result["trace"]["generated_ids"], 3, [151645, 151643]))
        for index, (name, parameter) in enumerate(model.named_parameters()):
            control.require(parameter.dtype == torch.bfloat16 and not parameter.requires_grad and parameter.grad is None
                            and tensor_digest(torch, parameter) == base_hash[name], "frozen bare base changed")
            if index % 32 == 0:
                guard("base_hash_" + str(index))
        control.require(model.lm_head.weight is model.model.embed_tokens.weight, "final native tie changed")
        for name, expected in asset_pins.items():
            control.require(control.file_identity(model_dir / name)["sha256"] == expected, "final source asset changed")
        old.recheck_sources(root, source)
        guard("complete")
        snapshot = accounting.snapshot()
        control.require(snapshot["last_call_id"] == 318 and snapshot["forward_input_positions_returned"] == 774
                        and snapshot["forward_input_positions_attempted"] == 774, "exact-prefix total accounting mismatch")
        for name, count in (("model_construct", 1), ("checkpoint_load", 1), ("tensor_copy", 310), ("forward", 6)):
            control.require(snapshot["operations"][name] == {"attempted": count, "returned": count, "failed": 0, "pending": 0}, "incomplete native operation: " + name)
        control.require(all(snapshot["operations"][name]["attempted"] == 0 for name in ("backward", "optimizer_step", "checkpoint_save", "checkpoint_reload")), "unauthorized operation")
        events = [old.parse(line) for line in (output / "EVENTS.jsonl").read_bytes().splitlines()]
        verified = verify_call_journal(events, snapshot, maximum_optimizer_calls=0)
        control.write(output / "JOURNAL_VALIDATION.json", {"status": "PASS_OPERATION_JOURNAL_ONLY", "reconstructed": verified})
        result.update(base_unchanged=True, source_assets_unchanged=True, journal_verified=True,
                      exact_prefix_fixed_trace_mechanics_qualified=True, status=control.PASS_STATUS,
                      actual_counts={"forward_calls": 6, "forward_input_positions": 774, "logical_calls": 318, "generated_tokens": 3})
    except Exception as error:
        result.update(status="BLOCKED_NATIVE_EXACT_PREFIX_FIXED_TRACE", error=str(error), traceback=traceback.format_exc())
        event("failure", error=str(error))
    finally:
        result = control.persist_outcome(output, result, accounting, time.monotonic() - started)
        print(json.dumps({key: result.get(key) for key in ("status", "error", "wall_seconds", "generation_tokens", "actual_counts")}), flush=True)
    return 0 if result["status"] == control.PASS_STATUS else 2


def main():
    control.require(os.name == "nt", "registered Windows containment required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--go-path", type=Path)
    args = parser.parse_args()
    root = control.PACKAGE.parents[1]
    control.require(args.output.resolve() == (root / OUTPUT_RELATIVE).resolve(), "single registered owned run directory required")
    cfg = control.load_config(args.config)
    capture, preserved_bulk = control.prerequisite_inputs(root, cfg)
    source = control.source_snapshot(root, cfg, capture)
    return supervise(args, cfg, capture, source, root) if args.supervise else run(args, cfg, capture, source, root, preserved_bulk)


if __name__ == "__main__":
    sys.exit(main())
