"""Owned native mechanics worker. Dispatch only through native_supervisor."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_contract as contract
from rs_contract import canonical, require


class Journal:
    def __init__(self, directory: Path):
        self.path = directory / "events.jsonl"
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.counts = defaultdict(lambda: dict.fromkeys(("attempted", "returned", "failed", "pending"), 0))
        self.calls = 0
        self.guard = lambda stage: None

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
        self.guard("after_" + operation)
        return result

    def snapshot(self):
        return {"scope": "registered_API_boundaries;internal_ATen_operations_not_individually_metered",
                "operations": {k: dict(v) for k, v in self.counts.items()}, "last_call": self.calls}


def capture(torch, model, prompt_ids, call, guard, config, evidence=None):
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
    require(not argmax_differences, "prefix/bulk argmax differs")
    require(maximum <= config["tolerances"]["selected_logprob_max_error"] and
            mean <= config["tolerances"]["selected_logprob_mean_error"], "prefix/bulk numerical gate failed")
    return evidence


def run(config_path: Path, output: Path) -> int:
    require(output.resolve().parent == contract.PACKAGE / "artifacts" and output.is_dir(), "owned output directory")
    require((output / "supervisor_ownership.json").is_file(), "supervisor ownership receipt required")
    owner = contract.read_json(output / "supervisor_ownership.json")
    require(owner.get("worker_pid") == os.getpid() and owner.get("supervisor_pid") == os.getppid()
            and owner.get("config_sha256") == contract.filehash(config_path), "worker ownership/config mismatch")
    journal = Journal(output)
    start = time.monotonic()
    torch = model = config = None
    status, error = "NATIVE_QUALIFICATION_FAILED", None
    report = {}
    forward_counts = {"attempted_calls": 0, "input_positions": 0}

    def guard(stage):
        import psutil
        process = psutil.Process()
        rss = process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True) if p.is_running())
        available = psutil.virtual_memory().available
        require(time.monotonic() - start <= 600, "native worker deadline")
        require(rss <= 2 * 1024**3 and available >= 1.25 * 1024**3, "native worker host resource guard")
        if torch is not None and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            require(free >= 0.5 * 1024**3, "native GPU free floor")
        require((output / "supervisor_alive.json").is_file(), "missing supervisor heartbeat")
        heartbeat = contract.read_json(output / "supervisor_alive.json", 65536)
        require(time.time() - heartbeat["unix_seconds"] <= 2.0, "supervisor heartbeat stale")
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
        report["capture"] = {}
        capture(torch, model, rows[0]["prompt_ids"], journal.call, guard, config, report["capture"])
        report["training_mechanics"] = qualify_training(torch, model, rows, output, journal.call, guard,
            {"require_cuda": True, "adapter_tensors": 112, "adapter_parameters": 1146880, "eos_id": 151645,
             "rows_per_update": 20, "CE_absolute_error": 2e-6, "reference_gradient_atol": 1e-6,
             "reference_gradient_rtol": 1e-5})
        report["frozen_base_postcheck"] = verify_frozen_base(torch, model, source_hashes, journal.call, guard)
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
        if torch is not None and torch.cuda.is_initialized():
            report["GPU"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                             "free_bytes_final": torch.cuda.mem_get_info()[0]}
        with (output / "worker_result.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        journal.stream.close()
    return 0 if status == "PASS_SCOPED_NATIVE_MECHANICS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return run(args.config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
