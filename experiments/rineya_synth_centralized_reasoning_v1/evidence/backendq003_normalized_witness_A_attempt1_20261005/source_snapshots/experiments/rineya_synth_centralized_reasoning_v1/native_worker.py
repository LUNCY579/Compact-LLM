"""Owned native mechanics worker. Dispatch only through native_supervisor."""
from __future__ import annotations

import argparse
import ctypes
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_contract as contract
from rs_contract import canonical, require


MAX_OWNED_JOB_PIDS = 256


def diagnostic_json(value):
    """Keep invalid measured floats explicit in standards-compliant evidence."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(item) for item in value]
    return value


def _job_query_api():
    """Open only query-capable Windows APIs; no handle is retained here."""
    require(os.name == "nt", "native qualification requires Windows containment")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "OpenJobObjectW": ([wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
        "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                        wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "GetLastError": ([], wintypes.DWORD),
    }
    for name, (arguments, result) in signatures.items():
        getattr(kernel, name).argtypes = arguments
        getattr(kernel, name).restype = result
    return kernel


def _job_query_error(kernel, operation):
    code = int(kernel.GetLastError())
    if os.name == "nt":
        error = ctypes.WinError(code)
        error.add_note(operation + " failed")
        return error
    return OSError(code, operation + " failed")


def owned_job_pids(job_name):
    """Return a complete bounded Job PID snapshot, closing the query handle.

    Checking this assigned Job avoids the global Windows process enumeration
    performed by psutil.children(). The actual interpreter must be present on
    every sample. Holding a query handle between samples would defeat the
    supervisor's kill-on-close containment, so this helper always closes it.
    """
    require(type(job_name) is str and re.fullmatch(r"Local\\SYNTH_NQ_JOB_[0-9a-f]{32}", job_name) is not None,
            "owned Job name")
    kernel = _job_query_api()
    handle = kernel.OpenJobObjectW(4, False, job_name)  # JOB_OBJECT_QUERY only, non-inheritable
    if not handle:
        raise _job_query_error(kernel, "OpenJobObjectW")
    try:
        capacity = 16
        while capacity <= MAX_OWNED_JOB_PIDS:
            class PidList(ctypes.Structure):
                # Windows DWORD is exactly 32 bits, including in CPU fixtures.
                _fields_ = [("assigned", ctypes.c_uint32), ("count", ctypes.c_uint32),
                            ("pids", ctypes.c_size_t * capacity)]
            value = PidList()
            succeeded = kernel.QueryInformationJobObject(handle, 3, ctypes.byref(value), ctypes.sizeof(value), None)
            if succeeded:
                require(0 < value.count == value.assigned <= capacity <= MAX_OWNED_JOB_PIDS,
                        "owned Job PID list is empty, incomplete or exceeds bound")
                pids = [int(pid) for pid in value.pids[:value.count]]
                require(all(0 < pid <= 0xffffffff for pid in pids) and len(pids) == len(set(pids)),
                        "invalid or duplicate owned Job PID")
                require(os.getpid() in pids, "actual interpreter is absent from owned Job PID list")
                return pids
            if int(kernel.GetLastError()) != 234:  # ERROR_MORE_DATA is the only retryable result.
                raise _job_query_error(kernel, "QueryInformationJobObject")
            require(value.assigned <= MAX_OWNED_JOB_PIDS and value.count <= capacity,
                    "owned Job PID list exceeds containment bound")
            capacity = max(capacity * 2, int(value.assigned))
        raise ValueError("owned Job PID list exceeds containment bound")
    finally:
        if not kernel.CloseHandle(handle):
            raise _job_query_error(kernel, "CloseHandle")


def owned_job_resources(job_name):
    """Sum RSS for exactly the queried Job PIDs, ignoring confirmed death only."""
    import psutil
    pids = owned_job_pids(job_name)
    rss, observed, vanished = 0, [], []
    for pid in pids:
        try:
            amount = psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess:
            # Access denial, a reused live PID, and an unconfirmed identity
            # failure cannot silently reduce the measured owned-tree RSS.
            require(pid != os.getpid() and psutil.pid_exists(pid) is False,
                    "owned Job PID disappearance is not confirmed terminal")
            vanished.append(pid)
            continue
        require(type(amount) is int and amount >= 0, "invalid owned Job process RSS")
        rss += amount
        observed.append(pid)
    require(os.getpid() in observed, "actual interpreter RSS was not measured")
    return {"tree_rss_bytes": rss, "queried_job_pids": pids, "observed_pids": observed,
            "vanished_pids": vanished, "query_handle_closed": True,
            "scope": "Windows_JobObjectBasicProcessIdList_owned_PIDs_only"}


def in_owned_job(job_name):
    """Verify the actual interpreter descendant, then close its query handle."""
    require(os.name == "nt", "native qualification requires Windows containment")
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenJobObjectW(4, False, job_name)  # JOB_OBJECT_QUERY
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        inside = wintypes.BOOL()
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), handle, ctypes.byref(inside)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(inside.value)
    finally:
        # Retaining this handle would defeat the supervisor's kill-on-close.
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def verify_worker_ownership(owner, config_path, *, membership=in_owned_job):
    require(type(owner) is dict and type(owner.get("worker_pid")) is int and owner["worker_pid"] > 0
            and type(owner.get("supervisor_pid")) is int and owner["supervisor_pid"] > 0,
            "typed ownership PIDs")
    require(owner.get("config_sha256") == contract.filehash(config_path), "worker config ownership mismatch")
    name = owner.get("job_name")
    require(type(name) is str and re.fullmatch(r"Local\\SYNTH_NQ_JOB_[0-9a-f]{32}", name) is not None,
            "owned Job name")
    require(membership(name) is True, "interpreter is outside owned Job")
    import psutil
    supervisor = psutil.Process(owner["supervisor_pid"])
    require(supervisor.is_running() and supervisor.create_time() == owner.get("supervisor_create_time"),
            "supervisor identity is not currently live")
    return {"actual_worker_pid": os.getpid(), "actual_parent_pid": os.getppid(),
            "launcher_root_pid": owner["worker_pid"], "supervisor_pid": owner["supervisor_pid"],
            "job_name": name, "owned_job_membership_verified": True,
            "query_job_handle_closed_before_framework_import": True}


def read_heartbeat(path):
    """Read only the latest complete append-only heartbeat, with bounded IO."""
    with Path(path).open("rb", buffering=0) as stream:
        size = os.fstat(stream.fileno()).st_size
        require(0 < size <= 1024 * 1024, "heartbeat journal size bound")
        beginning = max(0, size - 4096)
        stream.seek(beginning)
        raw = stream.read(size - beginning)
    if beginning:
        require(b"\n" in raw, "no complete heartbeat tail")
        raw = raw.split(b"\n", 1)[1]
    lines = [line for line in raw.splitlines(keepends=True) if line.endswith(b"\n")]
    require(bool(lines), "no complete heartbeat record")
    value = json.loads(lines[-1].decode("utf-8"), object_pairs_hook=contract._pairs,
                       parse_constant=contract._bad_constant)
    require(type(value) is dict, "heartbeat object")
    return value

class Journal:
    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "events.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.counts = defaultdict(lambda: dict.fromkeys(("attempted", "returned", "failed", "pending"), 0))
        self.calls = 0
        self.guard = lambda stage: None

    def evidence(self, name, value):
        """Persist a completed measurement before subsequent native work."""
        require(type(name) is str and re.fullmatch(r"[a-z0-9_]+", name) is not None, "evidence name")
        path = self.directory / ("stage_" + name + ".json")
        raw = (canonical(diagnostic_json(value)) + "\n").encode("utf-8")
        require(len(raw) <= 1024 * 1024, "stage evidence size bound")
        with path.open("xb", buffering=0) as stream:
            require(stream.write(raw) == len(raw), "short stage evidence write")
            os.fsync(stream.fileno())
        self.emit({"event": "measurement_preserved", "path": path.name,
                   "sha256": contract.filehash(path), "qualification_established": False})

    def emit(self, value):
        self.stream.write(canonical({"utc": datetime.now(timezone.utc).isoformat(), **value}) + "\n")
        self.stream.flush()

    def call(self, operation, function):
        self.guard("before_" + operation)
        self.calls += 1
        identity = self.calls
        counts = self.counts[operation]
        counts["attempted"] += 1
        counts["pending"] += 1
        self.emit({"event": "attempt", "id": identity, "operation": operation})
        try:
            result = function()
        except BaseException as exc:
            counts["pending"] -= 1
            counts["failed"] += 1
            self.emit({"event": "failed", "id": identity, "operation": operation,
                       "error_type": type(exc).__name__, "error": str(exc)})
            raise
        counts["pending"] -= 1
        counts["returned"] += 1
        self.emit({"event": "returned", "id": identity, "operation": operation})
        if operation.endswith("_completion_loss_check"):
            require(type(result) is tuple and len(result) == 2 and type(result[1]) is dict,
                    "loss measurement boundary")
            self.evidence("loss_%04d" % identity, result[1])
        self.guard("after_" + operation)
        return result

    def snapshot(self):
        return {"scope": "registered_API_boundaries;internal_ATen_operations_not_individually_metered",
                "operations": {k: dict(v) for k, v in self.counts.items()}, "last_call": self.calls}


def capture(torch, model, prompt_ids, call, guard, config, evidence=None, preserve=None):
    """Greedy TRAIN-only capture, two repeats and isolated bulk likelihood."""
    limit = config["dose"]["capture_max_generated_tokens"]
    eos = {151645, 151643}
    all_traces = []
    evidence = {} if evidence is None else evidence
    evidence.update({"scope": "TRAIN_only_generated_continuation_no_gold_decode_input",
                     "traces": all_traces, "capability_correctness_scored": False})
    for repeat in range(2):
        prefix = list(prompt_ids)
        trace = []
        all_traces.append(trace)
        with torch.no_grad():
            for step in range(limit):
                guard("capture_step")
                require(len(prefix) <= 512, "capture window exhausted")
                ids = torch.tensor([prefix], dtype=torch.long, device="cuda:0")
                output = call("capture_forward", lambda: model(input_ids=ids,
                              attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True).logits)
                require(output.dtype == torch.float32 and bool(torch.isfinite(output).all()), "capture logits")
                row = output[0, -1]
                token = int(row.argmax().item())
                logprob = float(torch.log_softmax(row, dim=-1)[token].item())
                trace.append({"token": token, "logprob": logprob, "prefix_length": len(prefix)})
                if preserve is not None:
                    preserve("capture_r%d_t%02d" % (repeat, step), trace[-1])
                prefix.append(token)
                del output, ids, row
                if token in eos:
                    break
        require(trace, "capture has no attempted tokens")
    evidence["repeat_exact"] = all_traces[0] == all_traces[1]
    require(evidence["repeat_exact"], "repeat greedy capture differs")
    trace = all_traces[0]
    # Last generated token does not need to be supplied to predict itself.
    bulk_ids = list(prompt_ids) + [row["token"] for row in trace[:-1]]
    with torch.no_grad():
        ids = torch.tensor([bulk_ids], dtype=torch.long, device="cuda:0")
        logits = call("capture_bulk_forward", lambda: model(input_ids=ids,
                      attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True).logits)
        errors = []
        argmax_differences = []
        evidence["selected_logprob_errors"] = errors
        evidence["argmax_differences"] = argmax_differences
        for step, observed in enumerate(trace):
            row = logits[0, len(prompt_ids) - 1 + step]
            if int(row.argmax().item()) != observed["token"]:
                argmax_differences.append(step)
            bulk = float(torch.log_softmax(row, dim=-1)[observed["token"]].item())
            errors.append(abs(bulk - observed["logprob"]))
        del logits, ids
    maximum, mean = max(errors), sum(errors) / len(errors)
    evidence.update({"generated_tokens_per_repeat": len(trace), "prefix_bulk_argmax_equal": not argmax_differences,
                     "selected_logprob_max_error": maximum, "selected_logprob_mean_error": mean,
                     "trace": trace})
    if preserve is not None:
        preserve("capture_numerical_measurements", evidence)
    require(not argmax_differences, "prefix/bulk argmax differs")
    require(maximum <= config["tolerances"]["selected_logprob_max_error"] and
            mean <= config["tolerances"]["selected_logprob_mean_error"], "prefix/bulk numerical gate failed")
    return evidence


def preserve_terminal(output, report, journal, torch):
    """Write the terminal receipt before any optional CUDA diagnostic."""
    with (output / "worker_result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(diagnostic_json(report), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    journal.stream.close()
    if torch is not None:
        try:
            diagnostics = {"CUDA_initialized": torch.cuda.is_initialized()}
            if diagnostics["CUDA_initialized"]:
                diagnostics.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                    "free_bytes_final": torch.cuda.mem_get_info()[0]})
        except BaseException as exc:
            diagnostics = {"diagnostic_error": {"type": type(exc).__name__, "message": str(exc)}}
        with (output / "worker_GPU_final.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(diagnostic_json(diagnostics), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


def run(config_path: Path, output: Path) -> int:
    require(output.resolve().parent == contract.PACKAGE / "artifacts" and output.is_dir(), "owned output directory")
    require((output / "supervisor_ownership.json").is_file(), "supervisor ownership receipt required")
    owner = contract.read_json(output / "supervisor_ownership.json")
    ownership = verify_worker_ownership(owner, config_path)
    with (output / "worker_ownership_verified.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(ownership, stream, indent=2, sort_keys=True)
        stream.write("\n")
    journal = Journal(output)
    start = time.monotonic()
    torch = model = config = None
    status, error = "NATIVE_QUALIFICATION_FAILED", None
    report = {}
    forward_counts = {"attempted_calls": 0, "input_positions": 0}

    def guard(stage):
        import psutil
        rss = owned_job_resources(owner["job_name"])["tree_rss_bytes"]
        available = psutil.virtual_memory().available
        require(time.monotonic() - start <= 600, "native worker deadline")
        require(rss <= 2 * 1024**3 and available >= 1.25 * 1024**3, "native worker host resource guard")
        if torch is not None and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            require(free >= 0.5 * 1024**3, "native GPU free floor")
        require((output / "supervisor_heartbeat.jsonl").is_file(), "missing supervisor heartbeat")
        heartbeat = read_heartbeat(output / "supervisor_heartbeat.jsonl")
        age = time.time() - heartbeat["unix_seconds"]
        require(-0.2 <= age <= 2.0 and heartbeat["supervisor_pid"] == owner["supervisor_pid"],
                "supervisor heartbeat stale, future or wrong owner")
    journal.guard = guard
    try:
        config = contract.preflight(config_path)
        for key, value in config["offline_environment"].items():
            os.environ[key] = value
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        guard("pre_import")
        import torch as imported_torch
        torch = imported_torch
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        require(str(torch.__version__) == config["runtime_versions"]["torch"], "actual Torch version drift")
        require(importlib.metadata.version("transformers") == config["runtime_versions"]["transformers"], "actual Transformers drift")
        # Full remaining growth budget is checked after imports, before CUDA init/model.
        import psutil
        rss = psutil.Process().memory_info().rss
        require(psutil.virtual_memory().available >= 1.25 * 1024**3 + max(0, 2 * 1024**3 - rss), "post-import full growth budget")
        journal.call("cuda_context_init", torch.cuda.init)
        free, total = torch.cuda.mem_get_info()
        require(free >= 4 * 1024**3, "preload GPU admission")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        from native_loader import load_titan, verify_frozen_base
        from native_precision import install_fp32_compute, install_qv_lora, install_fp32_activation_assertions
        from native_mechanics import qualify_training
        from matching_schedule import _array_check
        arrays = {r["row_id"]: {k: v for k, v in r.items() if k != "row_id"}
                  for r in contract.read_jsonl(Path(config["data_bundle"]) / "token_pool_M1.jsonl")}
        rows = [arrays[rid] for rid in config["dose"]["row_ids"]]
        for row in rows:
            _array_check(row)
        require(len(rows) == 40 and sum(r["total_tokens"] for r in rows) == config["dose"]["input_positions"], "mechanical row dose drift")
        model, load_report, source_hashes = load_titan(torch, Qwen3Config, Qwen3ForCausalLM,
                                                     config["model_dir"], journal.call, guard)
        report["loader"] = load_report
        journal.evidence("loader", load_report)
        report["compute"] = journal.call("install_fp32_compute", lambda: install_fp32_compute(model, torch))
        model.eval()
        hooks = journal.call("install_activation_hooks", lambda: install_fp32_activation_assertions(model, torch))

        def forward_hook(module, args, kwargs):
            ids = kwargs.get("input_ids")
            require(ids is not None and ids.ndim == 2 and ids.shape[0] == 1 and 0 < ids.shape[1] <= 512, "single-sequence input contract")
            forward_counts["attempted_calls"] += 1
            forward_counts["input_positions"] += ids.numel()
            require(forward_counts["attempted_calls"] <= 200 and forward_counts["input_positions"] <= 102400, "registered total forward dose cap")
        model.register_forward_pre_hook(forward_hook, with_kwargs=True)
        ids = torch.tensor([rows[0]["input_ids"]], dtype=torch.long, device="cuda:0")
        with torch.no_grad():
            original = journal.call("zero_parity_base_forward", lambda: model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits)
        report["adapters"] = journal.call("install_qv_lora", lambda: install_qv_lora(model, torch))
        with torch.no_grad():
            adapted = journal.call("zero_parity_adapter_forward", lambda: model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits)
            report["zero_adapter_output_max_error"] = float((original - adapted).abs().max().item())
            report["zero_adapter_output_equal"] = torch.equal(original, adapted)
            require(report["zero_adapter_output_equal"], "zero LoRA output parity failed")
        del original, adapted, ids
        report["zero_adapter_output_max_error"] = 0.0
        journal.evidence("zero_adapter_parity", {k: report[k] for k in
                         ("zero_adapter_output_equal", "zero_adapter_output_max_error")})
        report["capture"] = {}
        capture(torch, model, rows[0]["prompt_ids"], journal.call, guard, config, report["capture"], journal.evidence)
        report["training_mechanics"] = qualify_training(torch, model, rows, output, journal.call, guard,
            {"require_cuda": True, "adapter_tensors": 112, "adapter_parameters": 1146880, "eos_id": 151645,
             "rows_per_update": 20, "CE_absolute_error": 2e-6, "reference_gradient_atol": 1e-6,
             "reference_gradient_rtol": 1e-5}, preserve=journal.evidence)
        journal.evidence("training_mechanics", report["training_mechanics"])
        report["frozen_base_postcheck"] = journal.call("frozen_base_postcheck", lambda:
            verify_frozen_base(torch, model, source_hashes, journal.call, guard))
        report["activation_observations"] = dict(hooks["observations"])
        require(all(n > 0 for n in hooks["observations"].values()), "unobserved FP32 activation site")
        for handle in hooks["handles"]:
            handle.remove()
        contract.preflight(config_path)
        guard("completed")
        status = "PASS_SCOPED_NATIVE_MECHANICS"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        journal.emit({"event": "worker_failed", "error": error})
    finally:
        report.update({"status": status, "error": error, "wall_seconds": time.monotonic() - start,
                       "scope": "SYNTH_NQ001_current_runtime_short_mechanics_only",
                       "forward_dose": forward_counts, "operation_accounting": journal.snapshot(),
                       "production_training_authorized": False, "protected_bank_access": False,
                       "quality_gain_established": False, "full_E0_or_central_E1_qualified": False})
        # Preserve the terminal mechanics receipt before optional CUDA reporting:
        # a poisoned/hung context must not erase the authoritative failure.
        preserve_terminal(output, report, journal, torch)
    return 0 if status == "PASS_SCOPED_NATIVE_MECHANICS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return run(args.config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
