"""Offline canonical first-token state sensitivity; separate frozen GPU contract.

No generation, adaptive alpha, teacher continuation, pool access or weight grad.
Only the coordinator's reviewed contained worker may call this native entrypoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import psutil
import torch

from .binding import ARMR, LAB, REPO, admitted_source, sha
from .control import BlockRecurrence, TAPSConfig
from .gpu_native import (GIB, encode_prefix, final_resource_snapshot, forward_logits,
                         gpu_resources, install_gpu_timing, one_shot_anchors,
                         require_cuda_tensor, resource_guard)
from .sensitivity import DiagnosticBlock, fixed_rollout
from experiments.rineya_taps_adaptive_recurrence_v1.safety import Journal, write_json

PACKAGE = Path(__file__).resolve().parent
GPU_HELPER_SHA256 = "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f"
SENSITIVITY_SHA256 = "5d0af45a8e91f0a00d91f83244105d06f1beba6f422cc3cd6d20c2d43d762f0d"


def bind_population(cases, metadata, tokenizer, canonical_answer):
    """Declare every input/exclusion before native outcomes; no target invention."""
    index = {item["case_id"]: item for item in metadata if item["split"] == "DEV"}
    if len(cases) != 16 or set(index) != {case["case_id"] for case in cases}:
        raise RuntimeError("diagnostic population must bind all16 original allowlisted DEV cases")
    result = []
    for case in cases:
        bound = index[case["case_id"]]
        row = {"case_id": case["case_id"], "family": case["family"], "scenario_id": case["scenario_id"],
               "input_token_ids": bound["input_token_ids"], "input_tokens": bound["input_tokens"],
               "prefix_sha256": bound["rendered_prompt_sha256"], "target_token_id": None,
               "target_convention": "inherited exact(single answer)/numeric/JSON first native token only",
               "gold_enters_forward_input": False, "generation_pool_access": False}
        if row["input_tokens"] > 256:
            row.update(status="EXCLUDED_SHAPE", diagnostic_forward_calls=0)
        else:
            answer = canonical_answer(case)
            if answer is None:
                row.update(status="NOT_APPLICABLE_INHERITED_TARGET", diagnostic_forward_calls=0)
            else:
                targets = tokenizer.encode(answer, add_special_tokens=False)
                if not targets:
                    raise RuntimeError("inherited canonical target encoded empty")
                row.update(status="ELIGIBLE", target_token_id=int(targets[0]), canonical_answer=answer,
                           canonical_target_token_ids=targets, diagnostic_forward_calls=3,
                           diagnostic_backward_calls=1)
        result.append(row)
    if (sum(row["status"] == "ELIGIBLE" for row in result) != 7
            or sum(row["status"] == "NOT_APPLICABLE_INHERITED_TARGET" for row in result) != 7
            or sum(row["status"] == "EXCLUDED_SHAPE" for row in result) != 2):
        raise RuntimeError("prospective diagnostic eligibility counts differ; stop before loading")
    return result


def discrepancy_gate(nominal, plus, minus, step=.01):
    derivative = nominal["sensitivity"]
    fd = (plus["loss"] - minus["loss"]) / (2 * step)
    if not all(isinstance(value, (float, int)) and math.isfinite(value) for value in (derivative, fd)):
        raise FloatingPointError("nonfinite native adjoint/finite difference")
    error = abs(fd - derivative)
    tolerance = 1e-3 + .03 * abs(derivative)
    if error > tolerance:
        raise FloatingPointError(f"native finite-difference discrepancy {error} exceeds registered {tolerance}")
    return {"finite_difference": fd, "absolute_error": error, "registered_tolerance": tolerance,
            "status": "PASS_NATIVE_LOCAL_SURROGATE_CHECK"}


def loss_diagnostic(model, ids, target, *, device, guard, synchronize, sink, label):
    """Three whole forwards, one suffix-state adjoint; target never conditions F."""
    results = []
    for alpha, gradient in ((1., True), (1.01, False), (.99, False)):
        start = time.perf_counter()
        sink({"event": "DIAGNOSTIC_FORWARD_STARTED", "label": label, "alpha": alpha,
              "prefix_tokens": ids.shape[1], "target_enters_forward": False,
              "state_gradient_requested": gradient, "completed": False})
        try:
            result = fixed_rollout(model, ids, target, alpha=alpha, device=device, guard=guard,
                                   synchronize=synchronize, compute_gradient=gradient)
        except Exception as error:
            sink({"event": "DIAGNOSTIC_FORWARD_FAILED", "label": label, "alpha": alpha,
                  "elapsed_seconds": time.perf_counter() - start, "completed": False,
                  "partial_compute": "UNKNOWN", "state_gradient_evaluations": None,
                  "error": f"{type(error).__name__}: {error}"}); raise
        sink({"event": "DIAGNOSTIC_FORWARD_COMPLETED", "label": label, "alpha": alpha,
              "prefix_tokens": ids.shape[1], "completed": True, "elapsed_seconds": time.perf_counter() - start,
              "full_model_calls": 1, "state_gradient_evaluations": int(gradient),
              "initial_block_applications": 1, "extra_block_applications": 1,
              "block_width": 2, "anchor_counts": result["anchor_counts"], "resources": guard()})
        results.append(result)
    nominal, plus, minus = results
    return {"nominal": nominal, "plus": plus, "minus": minus,
            **discrepancy_gate(nominal, plus, minus), "full_model_calls": 3,
            "state_gradient_evaluations": 1, "trajectory_reuse": False,
            "gold_enters_forward_input": False, "generation_pool_access": False,
            "interpretation": "canonical first-token CE surrogate; not semantic answer correctness"}


def run(source, out, admission_path):
    source, out, admission_path = Path(source).resolve(), Path(out).resolve(), Path(admission_path).resolve(strict=True)
    if PACKAGE not in out.parents or PACKAGE not in admission_path.parents:
        raise PermissionError("diagnostic output/admission must be campaign-local")
    out.mkdir(parents=True, exist_ok=False)
    started, model = time.perf_counter(), None
    receipt = {"status": "FAIL", "lane": "OFFLINE_STATE_SENSITIVITY_FP32", "native_model_loads": 0,
               "training": False, "native_test": False, "generation_pool_access": False,
               "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        required = {"status": "READY_FOR_DIAGNOSTIC_FP32_DISPATCH", "device": "cuda:0", "dtype": "float32",
                    "attn_implementation": "eager", "maximum_prefix_tokens": 256, "fd_step": .01,
                    "fd_absolute_tolerance": 1e-3, "fd_relative_tolerance": .03, "training": False,
                    "native_test": False, "generation_pool_access": False,
                    "gpu_allocated_ceiling_bytes": 5*GIB, "gpu_global_free_reserve_bytes": GIB,
                    "gpu_initial_free_floor_bytes": 6*GIB, "worker_host_available_floor_bytes": 6*GIB,
                    "worker_rss_ceiling_bytes": 6*GIB}
        if any(admission.get(key) != value for key, value in required.items()) or admission.get("source") != str(source):
            raise RuntimeError("diagnostic source/gradient/device/resource/FD admission differs")
        if sha(PACKAGE / "gpu_native.py") != GPU_HELPER_SHA256 or sha(PACKAGE / "sensitivity.py") != SENSITIVITY_SHA256:
            raise RuntimeError("qualified GPU/math apparatus bytes changed")
        review_path = (PACKAGE / admission["review_receipt"]).resolve(strict=True)
        if PACKAGE not in review_path.parents:
            raise PermissionError("diagnostic review must be campaign-local")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("status") != "PASS_DIAGNOSTIC_FP32_DISPATCH" or review.get("admission_sha256") != sha(admission_path):
            raise RuntimeError("diagnostic review does not bind this admission")
        for relative, expected in admission["dispatch_hashes"].items():
            path = (REPO / relative).resolve(strict=True)
            allowed = PACKAGE in path.parents or relative.endswith(".py") and relative.startswith((
                "experiments/rineya_taps_adaptive_recurrence_v1/", "experiments/anchor_routed_middle_recurrence_v1/src/",
                "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/")) or relative == "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/mergedev_fast.jsonl"
            if not allowed or sha(path) != expected:
                raise RuntimeError("diagnostic reviewed source differs: " + relative)
        historical, inherited_hashes = admitted_source(source)
        binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if admission.get("binding_sha256") != sha(binding_path) or admission.get("checkpoint_files") != binding["checkpoint_files"]:
            raise RuntimeError("diagnostic model/data metadata binding differs")
        cache = out / "hf_cache"
        os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"), HF_MODULES_CACHE=str(cache / "modules"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        if (platform.python_version() != historical["python"] or torch.__version__ != historical["torch"]
                or transformers.__version__ != historical["transformers"]
                or sha(inspect.getfile(Qwen3DecoderLayer)) != historical["qwen_implementation_sha256"]):
            raise RuntimeError("diagnostic exact frozen runtime differs")
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False, cache_dir=cache / "hub")
        sys.path[:0] = [str(ARMR / "src"), str(LAB / "src")]
        from armr.provenance import cases_from, checkpoint_files, state_hash
        from armr.metrics import canonical_answer
        path = LAB / "data/evaluation/mergedev_fast.jsonl"
        if sha(path) != binding["datasets"]["DEV"]["sha256"]:
            raise RuntimeError("diagnostic allowlisted DEV differs")
        cases = cases_from(path)
        metadata = json.loads((binding_path.parent / "PROMPT_AUDIT.json").read_text(encoding="utf-8"))
        population = bind_population(cases, metadata, tokenizer, canonical_answer)
        for entry in population:
            case = next(case for case in cases if case["case_id"] == entry["case_id"])
            if encode_prefix(tokenizer, case, torch.device("cpu"))[0].tolist() != entry["input_token_ids"]:
                raise RuntimeError("diagnostic rendered prefix differs from native token binding")
        write_json(out / "population.json", population)
        write_json(out / "registration.json", {"lane": "OFFLINE_STATE_SENSITIVITY_FP32", "admission_sha256": sha(admission_path),
                   "binding_sha256": sha(binding_path), "population_sha256": sha(out / "population.json"),
                   "sources": inherited_hashes, "sensitivity_sha256": SENSITIVITY_SHA256, "GPU_helpers_sha256": GPU_HELPER_SHA256,
                   "maximum_prefix_tokens": 256, "fd_step": .01, "fd_tolerance": "1e-3+.03*abs(sensitivity)",
                   "training": False, "generation": False, "generation_pool_access": False, "parameter_gradients": False,
                   "state_gradient": "post-L13 controlled clone leaf, full native suffix+custom readout",
                   "eligible_rows": 7, "not_applicable_rows": 7, "excluded_shape_rows": 2,
                   "forecast": {"forward_attempts": 45, "completed_forwards_expected": 44, "expected_forced_partial": 1,
                                "cache_rejections": 2, "model_API_invocations": 47, "state_gradient_evaluations": 9}})
        if not torch.cuda.is_available() or psutil.virtual_memory().available < 6*GIB:
            raise MemoryError("diagnostic GPU/host floor unavailable")
        torch.cuda.set_device(0); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        if gpu_resources()["gpu_free_bytes"] < 6*GIB:
            raise MemoryError("diagnostic initial GPU free floor unavailable")
        receipt["model_load_attempts"] = 1
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=True,
                    dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True,
                    device_map={"": "cuda:0"}, cache_dir=cache / "hub").eval().requires_grad_(False)
        receipt["native_model_loads"] = 1; receipt["after_load_resources"] = resource_guard()
        code = Path(inspect.getfile(type(model))).resolve()
        if (type(model).__name__ != "Qwen3AbstainForCausalLM" or cache not in code.parents
                or sha(code) != historical["checkpoint_files_before"]["modeling_supa_a11.py"]
                or model.config.abstain_token_id != 51378 or model.config.abstain_logit_bias != 5.156249985098839):
            raise RuntimeError("diagnostic actual custom head/code differs")
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            require_cuda_tensor(tensor, name)
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise RuntimeError("diagnostic floating model tensor notFP32")
        before_files, before_state = checkpoint_files(source), state_hash(model)
        if before_state != historical["state_hash_before"]:
            raise RuntimeError("diagnostic loaded state differs")
        receipt.update(loaded_state_sha256=before_state, loaded_custom_code_path=str(code), loaded_custom_code_sha256=sha(code))
        eligible = [entry for entry in population if entry["status"] == "ELIGIBLE"]
        representative = eligible[0]
        ids = torch.tensor([representative["input_token_ids"]], dtype=torch.long, device="cuda:0")
        filler = tokenizer.encode(".", add_special_tokens=False)[0]
        max_ids = torch.cat((ids, torch.full((1, 256 - ids.shape[1]), filler, dtype=torch.long, device="cuda:0")), dim=1)
        sync = lambda: torch.cuda.synchronize(0)
        guard = lambda: resource_guard(started, admission["hard_deadline_seconds"])
        with Journal(out / "attempts.jsonl") as attempts, Journal(out / "outcomes.jsonl") as outcomes, one_shot_anchors(model):
            def value_forward(prefix, alpha, extra, diagnostic, label):
                begin = time.perf_counter()
                attempts.append({"event": "VALUE_FORWARD_STARTED", "label": label, "prefix_tokens": prefix.shape[1],
                                 "alpha": alpha, "extra_loops": extra, "diagnostic": diagnostic, "completed": False})
                try:
                    if diagnostic:
                        with torch.enable_grad(), DiagnosticBlock(model, alpha=alpha, extra_loops=extra, device="cuda:0", guard=guard):
                            value = forward_logits(model, prefix, synchronize=sync).detach().clone()
                    else:
                        cfg = TAPSConfig(scheduler_mode="fixed", fixed_alpha=alpha)
                        with torch.inference_mode(), BlockRecurrence(model, extra_loops=extra, config=cfg) as wrapper:
                            install_gpu_timing(wrapper, sync); value = forward_logits(model, prefix, synchronize=sync)
                except Exception as error:
                    attempts.append({"event": "VALUE_FORWARD_FAILED", "label": label, "completed": False,
                                     "elapsed_seconds": time.perf_counter() - begin, "partial_compute": "UNKNOWN",
                                     "error": f"{type(error).__name__}: {error}"}); raise
                attempts.append({"event": "VALUE_FORWARD_COMPLETED", "label": label, "completed": True,
                                 "prefix_tokens": prefix.shape[1], "alpha": alpha, "extra_loops": extra,
                                 "full_model_calls": 1, "state_gradient_evaluations": 0,
                                 "elapsed_seconds": time.perf_counter() - begin, "resources": guard()})
                return value
            maximum_intact = None
            for label, prefix in (("representative", ids), ("maximum256", max_ids)):
                for alpha, extra in ((1., 0), (.99, 1), (1., 1), (1.01, 1)):
                    expected = value_forward(prefix, alpha, extra, False, label)
                    actual = value_forward(prefix, alpha, extra, True, label)
                    if label == "maximum256" and extra == 0:
                        maximum_intact = expected
                    if not torch.equal(expected, actual):
                        raise RuntimeError("native diagnostic value identity failed")
                loss_diagnostic(model, prefix, representative["target_token_id"], device="cuda:0", guard=guard,
                                synchronize=sync, sink=attempts.append, label="qualification_" + label)
            original_methods = [layer.forward for layer in model.model.layers]
            original = maximum_intact  # reuse already-charged maximum noextra value
            def forced(_module, _args): raise RuntimeError("EXPECTED_DIAGNOSTIC_SUFFIX_EXCEPTION")
            handle = model.model.layers[14].register_forward_pre_hook(forced)
            try:
                try: value_forward(max_ids, 1., 1, True, "forced_exception")
                except RuntimeError as error:
                    if str(error) != "EXPECTED_DIAGNOSTIC_SUFFIX_EXCEPTION": raise
                else: raise RuntimeError("diagnostic forced exception not exercised")
            finally: handle.remove()
            restored = value_forward(max_ids, 1., 0, False, "restored_after_exception")
            if not torch.equal(original, restored) or any(layer.forward != method for layer, method in zip(model.model.layers, original_methods)):
                raise RuntimeError("native diagnostic methods/values did not restore")
            for kwargs in ({"use_cache": True}, {"use_cache": False, "past_key_values": object()}):
                begin = time.perf_counter(); attempts.append({"event": "CACHE_REJECTION_STARTED", "use_cache": kwargs["use_cache"], "cache_object_present": "past_key_values" in kwargs})
                with torch.enable_grad(), DiagnosticBlock(model, device="cuda:0", guard=guard) as block:
                    try: model(input_ids=max_ids, attention_mask=torch.ones_like(max_ids), logits_to_keep=1, **kwargs)
                    except RuntimeError as error:
                        if "cache" not in str(error): raise
                        attempts.append({"event": "CACHE_REJECTED", "completed": False, "model_compute_started": False,
                                         "wrapper_forward_count": block.forwards, "elapsed_seconds": time.perf_counter() - begin})
                    else: raise RuntimeError("native diagnostic cache rejection failed")
            if state_hash(model) != before_state or checkpoint_files(source) != before_files or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise RuntimeError("diagnostic qualification mutated model/source or produced parameter gradients")
            write_json(out / "qualification.json", {"status": "PASS_NATIVE_DIAGNOSTIC_MECHANICS", "maximum_prefix_tokens": 256,
                       "noextra_unit_and_099_101_value_identity": True, "full_suffix_state_adjoint_FD": True,
                       "exception_cache_restoration": True, "parameter_gradients": "NONE", "source_unchanged": True,
                       "before_state_sha256": before_state, "after_state_sha256": before_state,
                       "resources": guard(), "target_convention": "first inherited canonical token; not semantic correctness"})
            for entry in population:
                if entry["status"] != "ELIGIBLE":
                    outcomes.append({**entry, "physical_run": False, "measured_loss": None,
                                     "state_gradient_evaluations": 0, "forward_calls": 0}); continue
                prefix = torch.tensor([entry["input_token_ids"]], dtype=torch.long, device="cuda:0")
                row = loss_diagnostic(model, prefix, entry["target_token_id"], device="cuda:0", guard=guard,
                                      synchronize=sync, sink=attempts.append, label=entry["case_id"])
                outcomes.append({**entry, **row, "physical_run": True, "status": "NATIVE_LOCAL_SURROGATE_MEASURED"})
        sync()
        if state_hash(model) != before_state or checkpoint_files(source) != before_files or sha(code) != historical["checkpoint_files_before"]["modeling_supa_a11.py"] or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("diagnostic final model/source/gradient preservation failed")
        receipt.update(status="DIAGNOSTIC_COMPLETE_NOT_SEMANTIC_EFFICACY", source_unchanged=True,
                       checkpoint_files_unchanged=True, parameter_gradients="NONE", optimizer_steps=0,
                       measured_rows=7, not_applicable=7, excluded_shape=2, final_resources=guard())
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt.update(wall_seconds=time.perf_counter() - started, ended_utc=datetime.now(timezone.utc).isoformat(),
                       final_resources_on_stop=final_resource_snapshot())
        write_json(out / "receipt.json", receipt)
    return 0 if receipt["status"] == "DIAGNOSTIC_COMPLETE_NOT_SEMANTIC_EFFICACY" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.out, args.admission))
