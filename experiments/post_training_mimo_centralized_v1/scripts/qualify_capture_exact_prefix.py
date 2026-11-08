"""Contained bare-Titan exact-prefix capture qualification; never DEV or E1.

Stdlib controls are importable without a framework. Only the owned worker imports
frameworks, after its GO receipt and clean committed source closure are checked.
The original failed bulk capture source, registration and evidence stay intact.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import random
import re
import subprocess
import sys
import time
import traceback


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
CONFIG_RELATIVE = "experiments/post_training_mimo_centralized_v1/configs/CAPTURE_EXACT_PREFIX_QUAL_V1.json"
CONFIG_SHA256 = "ca74db0926b1409bcfe432dadf554b282b01f917e4d107422e0fa673ad65d819"
SCRIPT_RELATIVE = "experiments/post_training_mimo_centralized_v1/scripts/qualify_capture_exact_prefix.py"
TEST_RELATIVE = "experiments/post_training_mimo_centralized_v1/tests/test_qualify_capture_exact_prefix.py"
PASS_STATUS = "PASS_BARE_GREEDY_EXACT_PREFIX_CE512_CAPTURE_ONLY"
FAIL_STATUS = "BLOCKED_EXACT_PREFIX_CAPTURE_QUALIFICATION"
EOS_IDS = [151645, 151643]
PHASES = ("finite", "online", "replay")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise RuntimeError("nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def safe_numbers(value):
    if type(value) is float and not math.isfinite(value):
        return None
    if type(value) is dict:
        return {key: safe_numbers(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [safe_numbers(item) for item in value]
    return value


def pin(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "file changed during hash")
    return {"sha256": h.hexdigest(), "bytes": after.st_size}


def write(path, value):
    raw = canonical(safe_numbers(value)) + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def token_hash(ids):
    require(type(ids) is list and all(type(token) is int and 0 <= token < 151936 for token in ids), "invalid token IDs")
    return hashlib.sha256(canonical(ids)).hexdigest()


def load_config(path):
    path = Path(path)
    require(path.resolve() == (ROOT / CONFIG_RELATIVE).resolve(), "registered exact-prefix config path required")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, "frozen exact-prefix config drift")
    cfg = parse(raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    raw = (ROOT / CONFIG_RELATIVE).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA256 and canonical(cfg) == canonical(parse(raw)), "unregistered exact-prefix config")
    for length in (128, 256, 384, 512):
        fixture = cfg["frozen_fixture_ids"][str(length)]
        require(len(fixture["ids"]) == length and token_hash(fixture["ids"]) == fixture["sha256"], "frozen fixture identity drift")


def output_path(cfg, requested=None):
    relative = cfg["approved_run_relative"]
    require(type(relative) is str and relative.startswith("artifacts/mimo_central/") and "\\" not in relative
            and not PureWindowsPath(relative).drive and ".." not in Path(relative).parts,
            "approved relative run path required")
    target = (ROOT / relative).resolve()
    require(target.is_relative_to(ROOT) and target != ROOT and
            not any(re.search(r"(^|[^a-z])(check|final|test2|fresh_sealed|published)([^a-z]|$)", part.lower())
                    for part in target.relative_to(ROOT).parts), "closed or escaped output path")
    require(requested is None or Path(requested).resolve() == target, "only registered output path allowed")
    return target


def source_snapshot(cfg):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True, timeout=15).strip(),
            "commit exact-prefix source before qualification")
    pins = deepcopy(cfg["reused_source_files"])
    for relative in (SCRIPT_RELATIVE, TEST_RELATIVE, CONFIG_RELATIVE):
        pins[relative] = pin(ROOT / relative)
    for relative, expected in pins.items():
        require(pin(ROOT / relative) == expected, "source drift: " + relative)
        committed = subprocess.run(["git", "show", head + ":" + relative], cwd=ROOT, capture_output=True, timeout=15, check=False)
        require(committed.returncode == 0 and {"sha256": hashlib.sha256(committed.stdout).hexdigest(), "bytes": len(committed.stdout)} == expected,
                "uncommitted dependency: " + relative)
    for path, expected in cfg["installed_runtime_source_files"].items():
        require(pin(path) == expected, "installed runtime source drift")
    return {"code_sha": head, "dirty": False, "source_files": pins,
            "installed_runtime_source_files": deepcopy(cfg["installed_runtime_source_files"])}


def recheck_sources(snapshot):
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15).strip() == snapshot["code_sha"], "worker HEAD changed")
    for relative, expected in snapshot["source_files"].items():
        require(pin(ROOT / relative) == expected, "worker source changed: " + relative)
    for path, expected in snapshot["installed_runtime_source_files"].items():
        require(pin(path) == expected, "installed runtime changed")


def build_fixture_ids(tokenizer, cfg):
    """Independently reconstruct every frozen fixture before any model creation."""
    seq = tokenizer.apply_chat_template([{"role": "user", "content": cfg["fixtures"]["user_message"]}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    filler = tokenizer.encode(cfg["fixtures"]["filler"], add_special_tokens=False)
    token_hash(seq)
    token_hash(filler)
    require(bool(filler), "fixture filler must be nonempty")
    built = {}
    for length in (128, 256, 384, 512):
        require(len(seq) <= length, "fixture template exceeds envelope")
        ids = (filler * (length // len(filler) + 1))[:length - len(seq)] + seq
        expected = cfg["frozen_fixture_ids"][str(length)]
        require(ids == expected["ids"] and token_hash(ids) == expected["sha256"], "tokenizer-built fixture differs: " + str(length))
        built[length] = ids
    return built


def prefix_for_target(prompt_ids, generated_ids, ordinal):
    token_hash(prompt_ids)
    token_hash(generated_ids)
    require(len(prompt_ids) in (128, 384) and 1 <= len(generated_ids) <= 128, "registered rollout envelope required")
    require(not any(token in EOS_IDS for token in generated_ids[:-1]) and
            (generated_ids[-1] in EOS_IDS or len(generated_ids) == 128), "complete EOS/cap rollout required")
    require(type(ordinal) is int and 0 <= ordinal < len(generated_ids), "valid target ordinal required")
    prefix = prompt_ids + generated_ids[:ordinal]
    require(len(prefix) <= 511, "exact-prefix input ceiling exceeded")
    return {"ordinal": ordinal, "ids": prefix, "target_id": generated_ids[ordinal],
            "input_positions": len(prefix), "prefix_token_ids_sha256": token_hash(prefix)}


def expected_counts(first, second):
    require(all(type(count) is int and 1 <= count <= 128 for count in (first, second)), "actual rollout lengths must be1..128")
    forwards = 2 + 2 * (first + second)
    return {"generated_tokens": first + second, "forward_calls": forwards, "logical_calls": 312 + forwards,
            "forward_input_positions": 768 + 2 * (128 * first + first * (first - 1) // 2 +
                384 * second + second * (second - 1) // 2)}


def prefix_comparison(online, replay):
    """Retain mismatches; never accept matching logprob alone as logit identity."""
    for witness in (online, replay):
        require(type(witness) is dict and type(witness["finite"]) is bool, "typed prefix witness required")
        require(type(witness["ordinal"]) is int and 0 <= witness["ordinal"] < 128, "typed target ordinal required")
        token_hash([witness["target_id"]])
        for name in ("prefix_token_ids_sha256", "raw_logits_sha256"):
            require(type(witness[name]) is str and re.fullmatch(r"[0-9a-f]{64}", witness[name]), "witness SHA256 required")
        require(type(witness["model_logprob"]) in (int, float), "numeric selected model logprob required")
    prefix_equal = online["prefix_token_ids_sha256"] == replay["prefix_token_ids_sha256"]
    target_equal = online["target_id"] == replay["target_id"] and online["ordinal"] == replay["ordinal"]
    logits_equal = online["raw_logits_sha256"] == replay["raw_logits_sha256"]
    shape_dtype_equal = all(witness["logits_shape"] == [1, 1, 151936] and witness["logits_dtype"] == "bfloat16"
                            for witness in (online, replay))
    finite = all(witness["finite"] and math.isfinite(witness["model_logprob"]) and witness["model_logprob"] <= 0
                 for witness in (online, replay))
    error = abs(online["model_logprob"] - replay["model_logprob"]) if finite else None
    return {"prefix_equal": prefix_equal, "target_and_ordinal_equal": target_equal, "raw_logits_bitwise_equal": logits_equal, "shape_dtype_equal": shape_dtype_equal,
            "finite": finite, "model_logprob_abs_error": error,
            "passed": bool(prefix_equal and target_equal and logits_equal and shape_dtype_equal and finite and error <= 1e-4)}


def new_phase_accounting():
    return {phase: {"operations": dict.fromkeys(("attempted", "returned", "failed", "pending"), 0),
                   "input_positions_attempted": 0, "input_positions_returned": 0} for phase in PHASES}


def single_prefix_forward(calls, model, tensor, *, stage, input_positions):
    # Output extraction is a later validation step, not a model call failure.
    output = calls.call("forward", lambda: model(tensor, use_cache=False, logits_to_keep=1),
                        stage=stage, input_positions=input_positions)
    return output.logits


def append_event(path, record):
    with Path(path).open("ab") as stream:
        stream.write(canonical(safe_numbers(record)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def event_sink(path, started, result, phases, *, clock=time.monotonic):
    def emit(kind, **fields):
        if fields.get("operation") == "forward" and kind in ("operation_attempt", "operation_returned", "operation_failed"):
            phase = fields["stage"].split("_", 1)[0]
            require(phase in PHASES, "unregistered physical forward phase")
            item = phases[phase]
            state = {"operation_attempt": "attempted", "operation_returned": "returned", "operation_failed": "failed"}[kind]
            item["operations"][state] += 1
            item["operations"]["pending"] += 1 if state == "attempted" else -1
            if state in ("attempted", "returned"):
                item["input_positions_" + state] += fields["input_positions"]
        record = {"event": kind, "elapsed_seconds": clock() - started, **fields}
        try:
            append_event(path, record)
        except BaseException as error:
            result.setdefault("publication_errors", []).append({"event": kind, "error_type": type(error).__name__, "error": str(error)[:1024]})
            # Preserve a physical operation's original exception when publishing
            # its failed marker also fails. The worker still fails on that cause.
            if kind != "operation_failed":
                raise
    return emit


def verify_complete_accounting(snapshot, phases, first, second):
    counts = expected_counts(first, second)
    expected = {"model_construct": 1, "checkpoint_load": 1, "tensor_copy": 310, "forward": counts["forward_calls"]}
    require(type(snapshot["accounting_version"]) is int and snapshot["accounting_version"] == 2 and
            type(snapshot["max_optimizer_calls"]) is int and snapshot["max_optimizer_calls"] == 0 and snapshot["pending_calls"] == [], "complete journal required")
    for operation, actual in snapshot["operations"].items():
        count = expected.get(operation, 0)
        require(canonical(actual) == canonical({"attempted": count, "returned": count, "failed": 0, "pending": 0}), "call schedule drift: " + operation)
    require(set(expected) <= set(snapshot["operations"]) and snapshot["last_call_id"] == counts["logical_calls"], "logical schedule drift")
    require(snapshot["forward_input_positions_attempted"] == snapshot["forward_input_positions_returned"] == counts["forward_input_positions"]
            and snapshot["training_forward_input_positions_attempted"] == snapshot["training_forward_input_positions_returned"] == 0,
            "position schedule drift")
    half_positions = (counts["forward_input_positions"] - 768) // 2
    require(set(phases) == set(PHASES), "phase inventory drift")
    for phase, calls, positions in (("finite", 2, 768), ("online", first + second, half_positions), ("replay", first + second, half_positions)):
        require(canonical(phases[phase]) == canonical({"operations": {"attempted": calls, "returned": calls, "failed": 0, "pending": 0},
            "input_positions_attempted": positions, "input_positions_returned": positions}), "phase dose drift: " + phase)
    return counts


def gpu_free_gib():
    raw = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=15)
    values = raw.strip().splitlines()
    require(len(values) == 1, "one CUDA0 memory observation required")
    free = float(values[0]) / 1024
    require(math.isfinite(free) and free >= 0, "finite GPU memory observation required")
    return free


def preworker_measurement(sample):
    names = ("host_available_gib", "gpu_free_gib", "disk_free_gib")
    require(type(sample) is dict and all(type(sample[name]) in (int, float) and math.isfinite(sample[name]) and sample[name] >= 0 for name in names),
            "finite preworker resource observations required")
    return {"host_admitted": sample["host_available_gib"] >= 3.3125, "gpu_admitted": sample["gpu_free_gib"] >= 4.,
            "disk_admitted": sample["disk_free_gib"] >= 4.,
            "admitted": sample["host_available_gib"] >= 3.3125 and sample["gpu_free_gib"] >= 4. and sample["disk_free_gib"] >= 4.}


def preworker_wait(*, source_check, sample, clock, pause, record):
    started = previous = clock()
    require(type(started) in (int, float) and math.isfinite(started), "finite monotonic clock required")
    observations = consecutive = 0
    last = None
    while True:
        source_check()
        current = clock()
        require(math.isfinite(current) and current >= previous, "preworker clock moved backward")
        previous = current
        if current - started >= 30:
            break
        require(observations < 16, "preworker observation ceiling")
        last = sample()
        observations += 1
        after = clock()
        require(math.isfinite(after) and after >= previous, "preworker clock moved backward")
        previous = after
        record({"observation": observations, "elapsed_seconds": after - started, "sample": deepcopy(last)})
        measured = preworker_measurement(last)
        decision = clock()
        require(math.isfinite(decision) and decision >= previous, "preworker clock moved backward")
        previous = decision
        if decision - started >= 30:
            break
        consecutive = consecutive + 1 if measured["admitted"] else 0
        if consecutive == 2:
            return {"status": "ADMITTED_PREWORKER_RESOURCE_ONLY", "observations": observations,
                    "last_measurement": measured, "model_calls": 0, "framework_imports": 0, "E1_authorized": False}
        pause(min(2., 30 - (decision - started)))
    return {"status": "TIMED_OUT_PREWORKER_NO_WORKER", "observations": observations,
            "last_measurement": preworker_measurement(last) if last is not None else None,
            "model_calls": 0, "framework_imports": 0, "E1_authorized": False}


def validate_prerequisites(cfg):
    for field in ("original_bulk_registration", "original_bulk_failure", "causal_pass_receipt"):
        require(pin(ROOT / cfg[field]) == cfg[field + "_pin"], "historical prerequisite changed: " + field)
    require(parse((ROOT / cfg["original_bulk_failure"]).read_bytes())["status"] == "BLOCKED_CAPTURE_QUALIFICATION", "original bulk failure must remain failed")
    require(parse((ROOT / cfg["causal_pass_receipt"]).read_bytes())["status"] ==
            "PASS_REVIEW_PRESERVED_FIXED_SHAPE_CAUSAL_DIAGNOSTIC_ONLY", "reviewed causal prerequisite not passed")
    require(pin(ROOT / cfg["qualified_core_config"])["sha256"] == cfg["qualified_core_config_sha256"], "qualified CE512 config drift")
    reference_path = PACKAGE / cfg["qualified_CE512_reference"]
    require(pin(reference_path)["sha256"] == cfg["qualified_CE512_reference_sha256"] and
            parse(reference_path.read_bytes())["status"] == "PASS_NATIVE_UNPADDED_CE512_QUALIFICATION_ONLY", "qualified CE512 reference drift")


def run_cpu_controls(cfg):
    """Computed synthetic prefix controls; no native tensor or EOS observations."""
    prompt = cfg["frozen_fixture_ids"]["128"]["ids"]
    generated = [16, 22, 151645]
    cases = []
    for ordinal in range(3):
        target = prefix_for_target(prompt, generated, ordinal)
        require(target["ids"] == prompt + generated[:ordinal] and target["target_id"] == generated[ordinal], "CPU prefix alignment failed")
        cases.append({"case": "prefix_target_" + str(ordinal), "prefix_sha256": target["prefix_token_ids_sha256"],
                      "input_positions": target["input_positions"], "target_id": target["target_id"], "passed": True})
    from experiments.post_training_mimo_centralized_v1.scripts import greedy_capture as control
    for eos in EOS_IDS:
        for generated in ([eos], [7] * 127 + [eos]):
            target = prefix_for_target(prompt, generated, len(generated) - 1)
            require(control.termination_after_emit(prompt, generated, 128, EOS_IDS) == "eos" and target["target_id"] == eos, "CPU EOS control failed")
            cases.append({"case": f"EOS_{eos}_length_{len(generated)}", "input_positions": target["input_positions"], "target_id": eos, "passed": True})
    require(control.termination_after_emit(prompt, [7] * 128, 128, EOS_IDS) == "length", "CPU cap control failed")
    cases.append({"case": "cap_without_EOS", "input_positions": 255, "target_id": prefix_for_target(prompt, [7] * 128, 127)["target_id"], "passed": True})
    require(expected_counts(128, 128) == {"generated_tokens": 256, "forward_calls": 514, "logical_calls": 826, "forward_input_positions": 164352}, "CPU maximum dose failed")
    return {"status": "PASS_CPU_EXACT_PREFIX_CONTROLS_ONLY", "cases": cases, "native_EOS_observations": [],
            "model_calls": 0, "training_authorized": False, "E1_authorized": False, "CHECK_FINAL_access": False}


def terminal_publish(output, result):
    """Publish terminal accounting before the result; preserve prior error causes."""
    payloads = (("RESOURCE_BASELINE.json", lambda: {"status": result["status"], "samples": result["resources"]}),
        ("COST_LEDGER.json", lambda: {name: result[name] for name in ("status", "wall_seconds", "pretrained_loads", "optimizer_steps",
            "training_tokens", "generation_tokens", "teacher_calls", "accounting_version", "call_accounting", "phase_forward_accounting")}),
        ("CAPTURE_RESULT.json", lambda: result))
    for filename, value in payloads:
        try:
            write(output / filename, value())
        except BaseException as error:
            result["status"] = FAIL_STATUS
            result.setdefault("error_type", type(error).__name__)
            result.setdefault("error", str(error)[:1024])
            result.setdefault("publication_errors", []).append({"file": filename, "error_type": type(error).__name__, "error": str(error)[:1024]})
    if result["status"] != PASS_STATUS:
        try:
            write(output / "FAILURE_RECEIPT.json", result)
        except BaseException as error:
            result.setdefault("publication_errors", []).append({"file": "FAILURE_RECEIPT.json", "error_type": type(error).__name__, "error": str(error)[:1024]})


def run(args, cfg):
    started = time.monotonic()
    while not args.go_path.exists() and time.monotonic() - started < 30:
        time.sleep(.05)
    require(args.go_path.exists() and parse(args.go_path.read_bytes()) == {"config_sha256": CONFIG_SHA256}, "contained exact-prefix GO missing or changed")
    snapshot = source_snapshot(cfg)
    validate_prerequisites(cfg)
    output = output_path(cfg, args.output)
    require(not output.exists(), "write-once worker output required")
    output.mkdir(parents=True)
    result = {"schema_version": 1, "experiment_id": cfg["experiment_id"], "status": "RUNNING_EXACT_PREFIX_CAPTURE",
        "runtime_scope": cfg["runtime_scope"], "resource_profile": cfg["resource_profile"], "config_sha256": CONFIG_SHA256,
        "code_sha": snapshot["code_sha"], "dirty_at_launch": False, "evidence_level": "MEASURED / REPRODUCED",
        "seed": cfg["seed"], "pretrained_loads": 0, "optimizer_steps": 0, "training_tokens": 0, "generation_tokens": 0,
        "teacher_calls": 0, "teacher_call_semantics": "External teacher/grader calls; independent student prefix forwards are counted separately.",
        "checks": {}, "resources": [], "fixtures": [], "raw_logit_observations": [], "accounting_version": 2,
        "native_evaluation_bank_access": False, "language_grading_authorized": False, "DEV_collection_authorized": False,
        "E1_authorized": False, "CHECK_FINAL_access": False, "goal_complete": False, "old_bulk_scope_qualified": False,
        "automatic_retry_authorized": False, "training_authorized": False, "original_bulk_failure_pin": cfg["original_bulk_failure_pin"],
        "causal_pass_receipt_pin": cfg["causal_pass_receipt_pin"]}
    phases = new_phase_accounting()
    event = event_sink(output / "EVENTS.jsonl", started, result, phases)
    # The complete reused source closure was verified before these imports.
    from experiments.post_training_mimo_centralized_v1.scripts import greedy_capture as control
    from experiments.post_training_mimo_centralized_v1.scripts import post_import_admission, stream_checkpoint
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting, tensor_digest, verify_call_journal
    accounting = CallAccounting(event, maximum_optimizer_calls=0)
    calls = control.BudgetedCalls(accounting, cfg)
    try:
        write(output / "CONFIG.json", cfg)
        write(output / "SOURCE_RECEIPT.json", snapshot)
        cpu = run_cpu_controls(cfg)
        cpu.update(worker_sha256=snapshot["source_files"][SCRIPT_RELATIVE]["sha256"], test_sha256=snapshot["source_files"][TEST_RELATIVE]["sha256"])
        write(output / "CPU_CONTROL_RECEIPT.json", cpu)
        require(Path(sys.executable).resolve() == Path(cfg["runtime"]["executable"]).resolve() and
                sys.version.startswith(cfg["runtime"]["python"] + " "), "pinned Python runtime required")
        packages = {name: importlib.metadata.version(name) for name in cfg["runtime"]["packages"]}
        require(packages == cfg["runtime"]["packages"], "pinned package versions required")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        event("runtime_import_start")
        import psutil
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        event("runtime_import_returned")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.device_count() == 1 and
                torch.version.cuda == cfg["runtime"]["cuda"], "registered single CUDA0 BF16 runtime required")
        write(output / "RUNTIME_RECEIPT.json", {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory, "attention": "eager", "base_dtype": "bfloat16",
            "adapter": None, "use_cache": False, "logits_to_keep": 1, "inference_context": "torch.no_grad()", "torch_threads": 2,
            "allow_tf32": False, "deterministic_algorithms": True})
        process = psutil.Process()

        def guard(stage, admission=False, waiting=False):
            torch.cuda.synchronize()
            free, _ = torch.cuda.mem_get_info()
            sample = {"stage": stage, "elapsed_seconds": time.monotonic() - started,
                "host_available_gib": psutil.virtual_memory().available / 2**30, "rss_gib": process.memory_info().rss / 2**30,
                "gpu_free_gib": free / 2**30, "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "disk_free_gib": psutil.disk_usage(str(output)).free / 2**30}
            result["resources"].append(sample)
            event("resource", **sample)
            require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for name, value in sample.items() if name != "stage"), "finite resource observations required")
            required_host = 1.25 + max(0., 2. - sample["rss_gib"])
            require(sample["rss_gib"] <= 2. and sample["host_available_gib"] >= (required_host if admission and not waiting else 1.25)
                    and sample["gpu_free_gib"] >= (4. if admission and not waiting else .5) and sample["disk_free_gib"] >= 4., "exact-prefix resource guard")
            require(time.monotonic() - started <= 600, "whole-worker wall ceiling")
            return sample

        def waiting_sample():
            recheck_sources(snapshot)
            return guard("post_import_wait", admission=True, waiting=True)

        wait = post_import_admission.wait_for_admission(waiting_sample, clock=time.monotonic, pause=time.sleep,
            record=lambda observation: event("post_import_observation", observation=observation), policy=cfg["post_import_wait"])
        result["post_import_admission"] = wait
        write(output / "POST_IMPORT_ADMISSION.json", wait)
        require(wait["status"] == "ADMITTED_POST_IMPORT_RESOURCE_ONLY", "post-import admission timed out; no model retry")
        recheck_sources(snapshot)
        guard("pretokenizer_admitted", admission=True)
        model_dir = Path(cfg["model_dir"])
        asset_pins = {**cfg["asset_sha256"], "model.safetensors": cfg["weight_sha256"]}
        for name, expected in asset_pins.items():
            require(pin(model_dir / name)["sha256"] == expected, "model asset drift: " + name)
        primary_path = ROOT / cfg["primary_receipt"].replace("\\", "/")
        require(pin(primary_path)["sha256"] == cfg["primary_receipt_sha256"], "primary receipt drift")
        primary = parse(primary_path.read_bytes())
        require(primary["weight_sha256"] == cfg["weight_sha256"] and primary["parameter_count"] == 596049920 and
                Path(primary["checkpoint_path"]).resolve() == model_dir.resolve(), "primary checkpoint identity")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        template = parse((model_dir / "tokenizer_config.json").read_bytes())["chat_template"]
        require(tokenizer.chat_template == template and tokenizer.eos_token_id == 151645, "tokenizer/template identity")
        roundtrip = "Qualification: 7 + 5 = 12."
        raw_ids = tokenizer.encode(roundtrip, add_special_tokens=False)
        require(tokenizer.decode(raw_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == roundtrip, "raw tokenizer roundtrip")
        fixtures = build_fixture_ids(tokenizer, cfg)
        result["checks"]["tokenizer_fixture_ids_match_before_model"] = True
        write(output / "TOKENIZER_RECEIPT.json", {"class_name": type(tokenizer).__name__, "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "eos_id": tokenizer.eos_token_id, "original_pad_id": tokenizer.pad_token_id, "raw_ids": raw_ids, "raw_roundtrip": True,
            "enable_thinking": False, "add_generation_prompt": True, "skip_special_tokens": False, "clean_up_tokenization_spaces": False,
            "fixture_ids_verified_before_model": {str(length): {"tokens": length, "sha256": token_hash(ids)} for length, ids in fixtures.items()}})
        model_cfg = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True)
        model_cfg._attn_implementation = "eager"
        require(model_cfg.rope_theta == 1000000 and model_cfg.num_hidden_layers == 28, "architecture drift")
        recheck_sources(snapshot)
        guard("preload_admitted", admission=True)
        torch.cuda.reset_peak_memory_stats()
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda:0"), torch.no_grad():
                model = calls.call("model_construct", lambda: Qwen3ForCausalLM(model_cfg), stage="model_construct")
        finally:
            torch.set_default_dtype(previous)
        model.eval().requires_grad_(False)
        base_hashes, diagnostics = calls.call("checkpoint_load", lambda: stream_checkpoint.load_streamed(
            torch, model, model_dir / "model.safetensors", calls, tensor_digest), stage="checkpoint_load")
        require(model.lm_head.weight is model.model.embed_tokens.weight and sum(parameter.numel() for parameter in model.parameters()) == 596049920,
                "base parameter count/tie differs")
        require(not model.training and not model.is_gradient_checkpointing and all(not parameter.requires_grad for parameter in model.parameters()), "bare frozen eval required")
        result["checks"]["exact_bare_native_load"] = True
        guard("loaded")
        write(output / "MODEL_RECEIPT.json", {"weight_sha256": cfg["weight_sha256"], "primary_receipt_sha256": cfg["primary_receipt_sha256"],
            "unique_parameters": 596049920, "serialized_tensors": 310, "tied": True, "trainable_parameters": 0, "adapter": None,
            "tensor_sha256": base_hashes, "loader_mode": cfg["loader_mode"], "loader_diagnostics": diagnostics, "config": model_cfg.to_dict()})

        def forward(ids, stage):
            token_hash(ids)
            require(1 <= len(ids) <= 512 and not model.training and not torch.is_grad_enabled(), "single no-grad CE512 input required")
            tensor = torch.tensor([ids], device="cuda:0", dtype=torch.long)
            return single_prefix_forward(calls, model, tensor, stage=stage, input_positions=len(ids))

        def raw_witness(logits, prefix, stage):
            require(torch.is_tensor(logits), "tensor logits required")
            witness = {"stage": stage, "prefix_token_ids_sha256": token_hash(prefix), "raw_logits_sha256": tensor_digest(torch, logits),
                "logits_shape": list(logits.shape), "logits_dtype": str(logits.dtype).removeprefix("torch."), "logits_device": str(logits.device),
                "finite": bool(torch.isfinite(logits).all())}
            result["raw_logit_observations"].append(deepcopy(witness))
            event("raw_logits_observed", **witness)
            require(witness["logits_shape"] == [1, 1, 151936] and witness["logits_dtype"] == "bfloat16" and
                    witness["logits_device"] == "cuda:0" and witness["finite"], "finite full-vocabulary BF16 CUDA0 logits required")
            return witness

        with torch.no_grad():
            for length in cfg["fixtures"]["finite_prefix_probe_lengths"]:
                stage = f"finite_{length}"
                logits = forward(fixtures[length], stage)
                raw_witness(logits, fixtures[length], stage)
                del logits
                guard(stage)
            result["checks"]["finite_prefix_probes"] = {"contexts": [256, 512], "logits_to_keep": 1, "finite": True}
            for prompt_length in cfg["fixtures"]["prompt_lengths"]:
                ids = fixtures[prompt_length][:]
                fixture_id = "fixture_" + str(prompt_length)
                record = {"fixture_id": fixture_id, "prompt_tokens": prompt_length, "generation_cap": 128, "prompt_ids": ids,
                    "prompt_token_ids_sha256": token_hash(ids), "generated_ids": [], "online_witnesses": [], "replay_witnesses": [],
                    "online_model_logprobs": [], "offline_model_logprobs": [], "behavior_logprobs": [], "raw_argmax_evidence": []}
                result["fixtures"].append(record)
                event("fixture_start", fixture_id=fixture_id, prompt_tokens=prompt_length, generation_cap=128,
                      prompt_token_ids_sha256=record["prompt_token_ids_sha256"])
                generation_started = time.monotonic()
                while len(record["generated_ids"]) < 128:
                    ordinal = len(record["generated_ids"])
                    prefix = ids + record["generated_ids"]
                    stage = f"online_{prompt_length}_{ordinal}"
                    logits = forward(prefix, stage)
                    witness = raw_witness(logits, prefix, stage)
                    scores = logits[0, -1].float()
                    token = int(scores.argmax(-1).item())
                    maximum = scores.max()
                    first_maximum = int((scores == maximum).nonzero()[0, 0].item())
                    selected_value, maximum_value = float(scores[token].item()), float(maximum.item())
                    behavior = control.validate_choice(token, selected_value, maximum_value, first_maximum)
                    logprob = float(scores.log_softmax(-1)[token].item())
                    require(math.isfinite(logprob) and logprob <= 0, "finite selected model logprob required")
                    require(result["generation_tokens"] < 256, "generation ceiling")
                    witness.update(ordinal=ordinal, target_id=token, model_logprob=logprob)
                    evidence = {"selected_raw_logit": selected_value, "raw_max_logit": maximum_value,
                                "first_maximum_token_id": first_maximum, "raw_argmax_verified": True}
                    record["generated_ids"].append(token)
                    record["online_witnesses"].append(deepcopy(witness))
                    record["online_model_logprobs"].append(logprob)
                    record["behavior_logprobs"].append(behavior)
                    record["raw_argmax_evidence"].append(evidence)
                    result["generation_tokens"] += 1
                    event("generated_token", fixture_id=fixture_id, token_id=token, behavior_logprob=behavior, **witness, **evidence)
                    del logits, scores, maximum
                    guard(stage)
                    if control.termination_after_emit(ids, record["generated_ids"], 128, EOS_IDS) is not None:
                        break
                record["generation_seconds"] = time.monotonic() - generation_started
                record["generation_timing"] = "Instrumented wall including durable logits/token witnesses and resource guards."
                record.update(control.render_response(tokenizer, record["generated_ids"], 128, EOS_IDS))
                record["generated_token_ids_sha256"] = token_hash(record["generated_ids"])
                record["visible_token_ids_sha256"] = token_hash(record["visible_ids"])
                event("rollout_complete", fixture_id=fixture_id, generated_tokens=len(record["generated_ids"]), stop_reason=record["stop_reason"],
                      terminal_eos_id=record["terminal_eos_id"], generated_token_ids_sha256=record["generated_token_ids_sha256"],
                      raw_response_sha256=record["raw_response_sha256"], visible_response_sha256=record["visible_response_sha256"])
                replay_started = time.monotonic()
                for ordinal in range(len(record["generated_ids"])):
                    target = prefix_for_target(ids, record["generated_ids"], ordinal)
                    stage = f"replay_{prompt_length}_{ordinal}"
                    logits = forward(target["ids"], stage)
                    witness = raw_witness(logits, target["ids"], stage)
                    logprob = float(logits[0, -1].float().log_softmax(-1)[target["target_id"]].item())
                    witness.update(ordinal=ordinal, target_id=target["target_id"], model_logprob=logprob)
                    comparison = prefix_comparison(record["online_witnesses"][ordinal], witness)
                    record["offline_model_logprobs"].append(logprob)
                    record["replay_witnesses"].append({**deepcopy(witness), "comparison": comparison})
                    event("replayed_token", fixture_id=fixture_id, **witness, comparison=comparison)
                    del logits
                    require(comparison["passed"], "exact-prefix logits or selected model logprob mismatch")
                    guard(stage)
                record["offline_teacher_seconds"] = time.monotonic() - replay_started
                record["offline_backend"] = cfg["teacher_forcing"]["backend"]
                record["likelihood"] = control.likelihood_measurement(record["online_model_logprobs"], record["offline_model_logprobs"],
                    cfg["teacher_forcing"]["logprob_max_atol"], cfg["teacher_forcing"]["logprob_mean_atol"])
                event("fixture_complete", fixture_id=fixture_id, generated_tokens=len(record["generated_ids"]), likelihood=record["likelihood"],
                      raw_logits_bitwise_equal=all(item["comparison"]["raw_logits_bitwise_equal"] for item in record["replay_witnesses"]))
                require(record["likelihood"]["passed"], "exact-prefix likelihood maximum/mean gate failed")
        terminal = sorted({fixture["terminal_eos_id"] for fixture in result["fixtures"] if fixture["terminal_eos_id"] is not None})
        result["checks"]["native_eos_coverage"] = {"observed": bool(terminal), "terminal_eos_ids": terminal}
        require(bool(terminal), "native natural EOS coverage missing")
        result["actual_counts"] = verify_complete_accounting(accounting.snapshot(), phases, *(len(fixture["generated_ids"]) for fixture in result["fixtures"]))
        for index, (name, parameter) in enumerate(model.named_parameters()):
            require(not parameter.requires_grad and parameter.grad is None and tensor_digest(torch, parameter) == base_hashes[name], "frozen bare base changed")
            if index % 16 == 0:
                guard("frozen_base_hash_" + str(index))
        for name, expected in asset_pins.items():
            require(pin(model_dir / name)["sha256"] == expected, "source model asset changed")
        require(pin(primary_path)["sha256"] == cfg["primary_receipt_sha256"], "primary receipt changed")
        validate_prerequisites(cfg)
        recheck_sources(snapshot)
        guard("complete")
        events_path = output / "EVENTS.jsonl"
        require(events_path.stat().st_size <= 8 * 1024 * 1024, "bounded operation journal required")
        verify_call_journal([parse(line) for line in events_path.read_bytes().splitlines()], accounting.snapshot(), maximum_optimizer_calls=0)
        require(not result.get("publication_errors"), "journal publication errors refuse qualification")
        result["checks"].update(frozen_base_unchanged=True, source_assets_unchanged=True, exact_prefix_model_logprobs=True,
            greedy_selected_raw_argmax=True, eos_stop_accounting=True, response_rendering=True, durable_call_journal_reconciled=True)
        errors = [error for fixture in result["fixtures"] for error in fixture["likelihood"]["absolute_errors"]]
        result["checks"]["selected_model_logprobs"] = {"finite": True, "max_abs": max(errors), "mean_abs": sum(errors) / len(errors), "atol": 1e-4}
        result["unqualified"] = ["old bulk likelihood method", "DEV/source enrollment", "adapter capture", "context beyond512",
            "stochastic RL", "calibration", "E1/training", "human language grading", "published-default generation"]
        result["status"] = PASS_STATUS
    except BaseException as error:
        result["status"] = FAIL_STATUS
        result["error_type"], result["error"], result["traceback"] = type(error).__name__, str(error)[:1024], traceback.format_exc()
        try:
            event("failure", error_type=type(error).__name__, error=str(error)[:1024])
        except BaseException:
            pass  # event_sink retained the secondary publication error.
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["call_accounting"] = accounting.snapshot()
        result["phase_forward_accounting"] = deepcopy(phases)
        result["pretrained_loads"] = result["call_accounting"]["operations"]["checkpoint_load"]["returned"]
        result["cost_semantics"] = "Attempted/returned framework spans are authoritative; completed formulas apply only to a fully qualified result."
        terminal_publish(output, result)
        print(json.dumps(safe_numbers({name: result.get(name) for name in ("status", "error_type", "error", "generation_tokens", "wall_seconds")})), flush=True)
    return 0 if result["status"] == PASS_STATUS else 2


def supervise(args, cfg):
    output = output_path(cfg, args.output)
    controller = output.with_name(output.name + "_controller")
    supervision = output.with_name(output.name + "_supervisor")
    require(not output.exists() and not controller.exists() and not supervision.exists(), "write-once controller/worker/supervisor paths required")
    controller.mkdir(parents=True)
    lock = None
    launched = False
    try:
        source = source_snapshot(cfg)
        validate_prerequisites(cfg)
        import psutil

        def sample():
            return {"host_available_gib": psutil.virtual_memory().available / 2**30, "gpu_free_gib": gpu_free_gib(),
                    "disk_free_gib": psutil.disk_usage(str(controller)).free / 2**30}

        admitted = preworker_wait(source_check=lambda: recheck_sources(source), sample=sample, clock=time.monotonic, pause=time.sleep,
            record=lambda observation: append_event(controller / "ADMISSION_SAMPLES.jsonl", observation))
        write(controller / "ADMISSION_RESULT.json", admitted)
        if admitted["status"] != "ADMITTED_PREWORKER_RESOURCE_ONLY":
            return 2
        # This source is pinned, and the shared watchdog owns only its worker tree.
        from experiments.opallo_native_collector_v1.supervisor import run_bounded
        common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True, timeout=15).strip())
        common = common.resolve() if common.is_absolute() else (ROOT / common).resolve()
        target_lock = common / "rineya-central-gpu0.lock"
        with target_lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        lock = target_lock
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        require(not any(term in line.lower() for line in active.splitlines() for term in ("python", "torch", "llama", "ollama", "vllm", "train", "comfy")), "competing GPU runtime; no worker launched")
        recheck_sources(source)
        fresh = sample()
        write(controller / "FRESH_LAUNCH_ADMISSION.json", {"sample": fresh, "measurement": preworker_measurement(fresh)})
        require(preworker_measurement(fresh)["admitted"], "fresh full-budget prelaunch admission failed")
        supervision.mkdir(parents=True)
        command = [cfg["runtime"]["executable"], "-B", "-m", cfg["worker_module"], "--config", str(args.config.resolve()),
                   "--output", str(output), "--go-path", str(supervision / "GO.json")]
        limits = {name: cfg[name] for name in ("resource_profile", "host_free_gib", "worker_free_gib", "worker_rss_limit_gib",
            "gpu_free_gib", "gpu_worker_free_gib", "disk_free_gib", "wall_seconds", "admission_budget")}
        write(supervision / "LAUNCH.json", {"command": command, "source": source, "config_sha256": CONFIG_SHA256,
            "resource_limits": limits, "observed_gpu_processes": active.splitlines()})
        recheck_sources(source)
        require(preworker_measurement(sample())["admitted"], "final prelaunch RAM/GPU/disk admission failed")
        launched = True
        run_bounded(command, ROOT, supervision, seconds=600, min_free_gib=1.25, max_rss_gib=2., go={"config_sha256": CONFIG_SHA256})
        contained = parse((supervision / "SUPERVISOR.json").read_bytes())
        require(contained["status"] == "PASS_PROCESS_ONLY" and contained["exit_code"] == 0, "contained exact-prefix worker failed")
        completed = parse((output / "CAPTURE_RESULT.json").read_bytes())
        require(completed["status"] == PASS_STATUS, "exact-prefix worker did not qualify scoped mechanics")
        return 0
    except BaseException as error:
        receipt = {"status": "BLOCKED_EXACT_PREFIX_CONTROLLER", "error_type": type(error).__name__, "error": str(error)[:1024],
                   "worker_launch_attempted": launched, "model_calls": None if launched else 0, "E1_authorized": False}
        try:
            write(controller / "FAILURE_RECEIPT.json", receipt)
        except BaseException as publication_error:
            receipt["publication_error"] = str(publication_error)[:1024]
        print(json.dumps(receipt), flush=True)
        return 2
    finally:
        if lock is not None:
            lock.unlink()


def main():
    require(os.name == "nt", "owned Windows containment required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--supervise", action="store_true")
    modes.add_argument("--go-path", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    return supervise(args, cfg) if args.supervise else run(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
