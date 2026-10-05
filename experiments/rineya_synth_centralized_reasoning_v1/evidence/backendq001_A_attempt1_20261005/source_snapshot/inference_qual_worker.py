"""Owned four-TRAIN-prompt qualification worker; dispatched by external supervisor.

Nothing is imported from a model framework or launched at module import. This
worker has no scorer, DEV packet, optimizer, backward, checkpoint restore/save,
or training path. A successful worker result still requires supervisor/audit
acceptance and qualifies only this frozen short inference mechanics scope.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference_contract as contract
from inference_component import InferenceComponent, validate_row
from native_loader import load_titan, verify_frozen_base
from native_precision import (install_fp32_compute, install_qv_lora, install_fp32_activation_assertions,
                              zero_lora_status)
from native_worker import (Journal, owned_job_resources, read_heartbeat, verify_worker_ownership,
                           preserve_terminal)
from rs_contract import digest, require

GIB = 1024 ** 3


class ForwardDose:
    """Actual model-prehook attempts including both parity and numerical controls."""
    def __init__(self, torch):
        self.torch, self.current_operation = torch, None
        self.attempted_calls, self.input_positions, self.operations = 0, 0, {}

    def before(self, model, args, kwargs):
        ids = kwargs.get("input_ids")
        require(self.torch.is_tensor(ids) and len(ids.shape) == 2 and ids.shape[0] == 1 and
                0 < ids.shape[1] <= 512 and ids.dtype == self.torch.long, "single unpadded bounded model input")
        require(self.current_operation in ("zero_parity_base_forward", "zero_parity_adapter_forward",
                "inference_qualification_forward", "inference_qualification_bulk_forward"), "unregistered model forward operation")
        positions = int(ids.shape[1])
        require(self.attempted_calls < contract.FORWARD_CAP and self.input_positions + positions <= contract.POSITION_CAP,
                "global forward/position cap exhausted")
        self.attempted_calls += 1
        self.input_positions += positions
        row = self.operations.setdefault(self.current_operation, {"attempted_calls": 0, "input_positions": 0})
        row["attempted_calls"] += 1
        row["input_positions"] += positions

    def snapshot(self):
        return deepcopy({"attempted_calls": self.attempted_calls, "input_positions": self.input_positions,
                         "maximum_forward_calls": contract.FORWARD_CAP,
                         "maximum_forward_input_positions": contract.POSITION_CAP, "operations": self.operations})

    def wrap(self, call):
        def counted(operation, function):
            require(self.current_operation is None, "nested API call boundary")
            self.current_operation = operation
            try:
                return call(operation, function)
            finally:
                self.current_operation = None
        return counted


class NoDecode:
    def decode(self, ids, *, skip_special_tokens):
        raise ValueError("qualification never decodes or scores answers")


def make_guard(output, owner, start, torch_getter, *, owned_resources=owned_job_resources,
               heartbeat_reader=read_heartbeat, available=None, monotonic=time.monotonic, wall=time.time):
    """Unchanged worker-side resource/heartbeat floors; external sampler is mandatory."""
    if available is None:
        def available():
            import psutil
            return psutil.virtual_memory().available
    def guard(stage):
        rss = owned_resources(owner["job_name"])["tree_rss_bytes"]
        require(monotonic() - start <= 600, "inference qualification worker deadline")
        require(rss <= 2 * GIB and available() >= 1.25 * GIB, "worker host resource guard")
        torch = torch_getter()
        if torch is not None and torch.cuda.is_initialized():
            require(torch.cuda.mem_get_info()[0] >= .5 * GIB, "native GPU free floor")
        heartbeat = heartbeat_reader(Path(output) / "supervisor_heartbeat.jsonl")
        age = wall() - heartbeat["unix_seconds"]
        require(-.2 <= age <= 2 and heartbeat["supervisor_pid"] == owner["supervisor_pid"], "supervisor heartbeat stale/future/wrong owner")
    return guard


def _logits(torch, output, length, device):
    require(torch.is_tensor(output) and tuple(output.shape) == (1, length, 151936) and
            output.dtype == torch.float32 and str(output.device) == device and not output.requires_grad and
            bool(torch.isfinite(output).all()), "complete FP32 finite no-grad parity logits")
    return output


def qualify_loaded(torch, model, config, source_hashes, journal, guard, report, dose, *,
                   compute=install_fp32_compute, adapters=install_qv_lora,
                   activation_hooks=install_fp32_activation_assertions, zero=zero_lora_status,
                   base_verify=verify_frozen_base, device="cuda:0"):
    """Injected core: fresh loaded dense M0 -> zero-B parity -> four controls.

    Helper injection exists solely for framework-free component tests. Production
    run() uses the exact source-pinned native helpers and cuda:0.
    """
    rows = [validate_row(row) for row in config["rows"]]
    require(len(rows) == 4 and digest(rows) == config["source_packet_sha256"] and
            config["dose"]["maximum_forward_calls"] == contract.FORWARD_CAP and
            config["dose"]["maximum_forward_input_positions"] == contract.POSITION_CAP,
            "frozen four-row TRAIN dose/source hash")
    require(not any(name.endswith((".lora_A", ".lora_B")) for name, _ in model.named_parameters()),
            "fresh dense M0 required; qualification STATE1/adapters forbidden")
    call = dose.wrap(journal.call)
    report["compute"] = call("install_fp32_compute", lambda: compute(model, torch))
    hooks = call("install_activation_hooks", lambda: activation_hooks(model, torch))
    forward_hook = model.register_forward_pre_hook(dose.before, with_kwargs=True)
    component = None
    try:
        ids = torch.tensor([rows[0]["prompt_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            original = _logits(torch, call("zero_parity_base_forward", lambda: model(input_ids=ids,
                              attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True).logits), len(rows[0]["prompt_ids"]), device)
        report["adapters"] = call("install_qv_lora", lambda: adapters(model, torch))
        report["zero_adapter_before"] = call("zero_lora_query_before", lambda: zero(model, torch))
        require(report["zero_adapter_before"]["all_B_zero"] is True, "new adapters must have exact zero B")
        with torch.no_grad():
            adapted = _logits(torch, call("zero_parity_adapter_forward", lambda: model(input_ids=ids,
                             attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True).logits), len(rows[0]["prompt_ids"]), device)
            equal = bool(torch.equal(original, adapted))
            maximum = 0.0 if equal else float((original - adapted).abs().max().item())
        report["zero_adapter_parity"] = {"zero_adapter_output_equal": equal, "zero_adapter_output_max_error": maximum,
                                        "capability_scored": False, "forward_dose": dose.snapshot()}
        journal.evidence("zero_adapter_parity", report["zero_adapter_parity"])
        require(equal and maximum == 0.0, "zero LoRA output parity failed")
        del original, adapted, ids
        component = InferenceComponent(torch, model, NoDecode(), call, guard, journal.evidence,
                         max_attempted_forwards=260, max_input_positions=contract.POSITION_CAP - dose.input_positions,
                         context_cap=512, generation_cap=32, device=device)
        report["numerical_controls"] = []
        for row in rows:
            report["numerical_controls"].append(component.qualify(row, source_packet_sha256=config["source_packet_sha256"]))
        report["parameter_preservation"] = component.verify_unchanged()
        report["zero_adapter_after"] = call("zero_lora_query_after", lambda: zero(model, torch))
        require(report["zero_adapter_after"]["all_B_zero"] is True, "zero B changed without adaptation authority")
        report["base_postcheck"] = call("verify_frozen_base", lambda: base_verify(torch, model, source_hashes, call, guard))
        require(report["base_postcheck"]["verified_source_tensors"] == 310 and
                report["base_postcheck"]["frozen_base_unchanged"] is True, "all310 frozen base bytes must remain unchanged")
        require(bool(hooks["observations"]) and all(count > 0 for count in hooks["observations"].values()), "unobserved FP32 activation site")
        report["activation_observations"] = deepcopy(hooks["observations"])
        journal.evidence("inference_qualification_completed", {"rows": 4, "dose": dose.snapshot(),
            "base_postcheck": report["base_postcheck"], "full_DEV_collection_qualified": False,
            "production_training_authorized": False, "protected_bank_access": False})
    finally:
        report["forward_dose"] = dose.snapshot()
        if component is not None:
            report["qualification_costs"] = component.costs()
            report["last_partial_receipt"] = deepcopy(component.last_receipt)
        forward_hook.remove()
        for handle in hooks["handles"]:
            handle.remove()
    return report


def run(config_path, output):
    """Native entrypoint; callers must use the owned external supervisor."""
    output, config_path = Path(output).absolute(), Path(config_path).absolute()
    require(output.resolve().parent == (contract.PACKAGE / "artifacts").resolve() and output.is_dir() and
            not output.is_symlink(), "owned output directory")
    journal = Journal(output)
    start = time.monotonic()
    torch = model = dose = guard = config = None
    status, error, report = "NATIVE_INFERENCE_QUALIFICATION_FAILED", None, {}
    try:
        require((output / "supervisor_ownership.json").is_file(), "supervisor ownership receipt required")
        owner = contract.native.read_json(output / "supervisor_ownership.json")
        ownership = verify_worker_ownership(owner, config_path)
        journal.evidence("worker_ownership_verified", ownership)
        guard = make_guard(output, owner, start, lambda: torch)
        journal.guard = guard
        config = contract.preflight(config_path)
        report["source_commit"], report["config_sha256"] = config["source_commit"], config["config_sha256"]
        for key, value in config["offline_environment"].items(): os.environ[key] = value
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        guard("pre_import")
        require(shutil.disk_usage(contract.PACKAGE).free >= 4 * GIB, "disk free admission floor")
        import torch as imported_torch
        torch = imported_torch
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        require(str(torch.__version__) == config["runtime_versions"]["torch"] and
                importlib.metadata.version("transformers") == config["runtime_versions"]["transformers"], "actual framework version drift")
        import psutil
        rss = psutil.Process().memory_info().rss
        require(psutil.virtual_memory().available >= 1.25 * GIB + max(0, 2 * GIB - rss), "post-import full growth budget")
        journal.call("cuda_context_init", torch.cuda.init)
        require(torch.cuda.mem_get_info()[0] >= 4 * GIB, "preload GPU admission")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        model, loader_receipt, source_hashes = load_titan(torch, Qwen3Config, Qwen3ForCausalLM,
                                    config["model_dir"], journal.call, guard)
        report["loader"] = loader_receipt
        journal.evidence("loader", loader_receipt)
        dose = ForwardDose(torch)
        qualify_loaded(torch, model, config, source_hashes, journal, guard, report, dose)
        contract.preflight(config_path)
        guard("completed")
        status = "PASS_SCOPED_NATIVE_INFERENCE_MECHANICS"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        try:
            journal.emit({"event": "worker_failed", "error": error})
        except BaseException as journal_error:
            report["journal_failure"] = {"type": type(journal_error).__name__, "message": str(journal_error)}
    finally:
        report.update({"status": status, "error": error, "wall_seconds": time.monotonic() - start,
            "scope": "SYNTH_INFQ001_four_TRAIN_prompts_inference_numerics_only",
            "operation_accounting": journal.snapshot(), "optimizer_steps": 0, "backwards": 0,
            "production_training_authorized": False, "DEV_collection_authorized": False,
            "scoring_authorized": False, "protected_bank_access": False, "full_DEV_collection_qualified": False,
            "quality_gain_established": False, "requires_independent_supervised_audit": True,
            "base_postcheck_completed": report.get("base_postcheck", {}).get("verified_source_tensors") == 310 and
                                        report.get("base_postcheck", {}).get("frozen_base_unchanged") is True})
        if dose is not None: report["forward_dose"] = dose.snapshot()
        preserve_terminal(output, report, journal, torch)
    return 0 if status == "PASS_SCOPED_NATIVE_INFERENCE_MECHANICS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return run(args.config, args.output_dir)


if __name__ == "__main__": raise SystemExit(main())
