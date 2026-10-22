"""Prospective frozen GPU FP32 lane; never a fallback through the CPU runner.

Requires independent source-bound admission. Importing this module loads no
assets. Only the coordinator's contained native worker may invoke ``run``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
import torch

from .binding import ARMR, LAB, REPO, admitted_source, sha
from .control import BlockRecurrence, TAPSConfig
from experiments.rineya_taps_adaptive_recurrence_v1.integration import BlockRecurrence as ReferenceBlock
from experiments.rineya_taps_adaptive_recurrence_v1.safety import Journal, write_json

PACKAGE = Path(__file__).resolve().parent
GIB = 1024**3
GRID = (.8, .9, 1., 1.1, 1.2)
SEEDS = (17, 29)


def require_cuda_tensor(tensor, name="tensor"):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda" or tensor.device.index != 0:
        raise RuntimeError(f"{name} must reside on explicitly admitted cuda:0")


def gpu_resources():
    free, total = torch.cuda.mem_get_info(0)
    return {"gpu_free_bytes": free, "gpu_total_bytes": total,
            "gpu_allocated_bytes": torch.cuda.memory_allocated(0),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(0),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "rss_bytes": psutil.Process().memory_info().rss,
            "host_available_bytes": psutil.virtual_memory().available}


def resource_guard(started=None, max_seconds=900.):
    if started is not None and time.perf_counter() - started > max_seconds:
        raise TimeoutError("GPU registered per-case deadline exceeded")
    sample = gpu_resources()
    if sample["rss_bytes"] > 6 * GIB or sample["host_available_bytes"] < GIB:
        raise MemoryError("GPU lane host worker/reserve ceiling violated")
    if sample["gpu_peak_allocated_bytes"] > 5 * GIB or sample["gpu_free_bytes"] < GIB:
        raise MemoryError("GPU allocated ceiling/global free reserve violated")
    return sample


def final_resource_snapshot():
    """Account partial loader allocation even when no model object returned."""
    if not torch.cuda.is_initialized():
        return None
    try:
        return gpu_resources()
    except Exception:
        return None


@contextmanager
def one_shot_anchors(model):
    """Count actual L0-L2 calls on every full-model invocation, including repeats."""
    counts, handles = [0, 0, 0], []
    def start(_module, _args, _kwargs):
        counts[:] = [0, 0, 0]
    def finish(_module, _args, _kwargs, output):
        if counts != [1, 1, 1]:
            raise RuntimeError("L0-L2 one-shot invariant violated")
        return output
    try:
        handles.append(model.register_forward_pre_hook(start, with_kwargs=True))
        for index in range(3):
            def visit(_module, _args, i=index):
                counts[i] += 1
            handles.append(model.model.layers[index].register_forward_pre_hook(visit))
        handles.append(model.register_forward_hook(finish, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def encode_prefix(tokenizer, case, device):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": case["prompt"]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return tokenizer(rendered, return_tensors="pt")["input_ids"].to(device)


def install_gpu_timing(wrapper, synchronize, *, require_cuda=True):
    """Synchronize GPU-only submitted core/moment work without changing math.

    Install after wrapper entry. Hook ownership/restoration stays in its normal
    handles list; class-bound observe avoids nesting proxies across prefix reset.
    Ordinary first-pass local core timing still measures L13 only, as in v1.
    """
    if not hasattr(wrapper, "originals"):
        return
    if getattr(wrapper, "_gpu_timing_ready", False):
        raise RuntimeError("GPU timing instrumentation installed twice")
    wrapper._gpu_timing_ready = True
    wrapper.gpu_conditioning_checks = 0
    wrapper._gpu_conditioning = None
    keys = ("position_ids", "position_embeddings", "attention_mask")
    def capture(_module, args, kwargs):
        wrapper._gpu_conditioning = {key: kwargs.get(key) for key in keys}
        if require_cuda:
            hidden = args[0] if args else kwargs["hidden_states"]
            pos = kwargs.get("position_ids")
            require_cuda_tensor(hidden, "block hidden")
            require_cuda_tensor(pos, "automatic position IDs")
            if tuple(pos.shape) != (1, hidden.shape[1]) or not torch.equal(pos, torch.arange(hidden.shape[1], device=pos.device).unsqueeze(0)):
                raise RuntimeError("GPU full-prefix automatic positions differ")
            mask = kwargs.get("attention_mask")
            require_cuda_tensor(mask, "decoder causal mask")
            if tuple(mask.shape[-2:]) != (hidden.shape[1], hidden.shape[1]):
                raise RuntimeError("GPU decoder causal mask shape differs")
    wrapper.handles.append(wrapper.model.model.layers[wrapper.layers[0]].register_forward_pre_hook(capture, with_kwargs=True))
    # Drain the ordinary upstream queue BEFORE integration._forward starts its
    # L13 timer. Direct extra replay callables bypass this module hook.
    wrapper.handles.append(wrapper.model.model.layers[wrapper.layers[-1]].register_forward_pre_hook(
        lambda _module, _args: synchronize()))
    originals = wrapper.originals
    def timed(original):
        def call(*args, **kwargs):
            if wrapper._gpu_conditioning is None or any(kwargs.get(key) is not wrapper._gpu_conditioning[key] for key in keys):
                raise RuntimeError("GPU repeated block conditioning identity differs")
            wrapper.gpu_conditioning_checks += 1
            synchronize()
            try:
                return original(*args, **kwargs)
            finally:
                synchronize()
        return call
    wrapper.originals = [timed(original) for original in originals]
    def scheduler_timer(_module, _args, _kwargs):
        scheduler = wrapper.scheduler
        original = type(scheduler).observe.__get__(scheduler, type(scheduler))
        def observe(*args, **kwargs):
            synchronize()
            try:
                return original(*args, **kwargs)
            finally:
                synchronize()
        scheduler.observe = observe
    wrapper.handles.append(wrapper.model.register_forward_pre_hook(scheduler_timer, with_kwargs=True))
    class Cleanup:
        def remove(self):
            wrapper._gpu_timing_ready = False
            wrapper._gpu_conditioning = None
            scheduler = getattr(wrapper, "scheduler", None)
            if scheduler is not None:
                scheduler.__dict__.pop("observe", None)
    wrapper.handles.append(Cleanup())


def forward_logits(model, ids, *, enforce_device=True, synchronize=None):
    if enforce_device:
        require_cuda_tensor(ids, "input ids")
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 2248:
        raise RuntimeError("GPU lane requires nonempty batch1 prefix<=2248")
    if synchronize:
        synchronize()
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=1)
    if synchronize:
        synchronize()
    if getattr(result, "past_key_values", None) is not None:
        raise RuntimeError("GPU model returned a forbidden cache")
    logits = result.logits[0, -1].float().clone()
    if enforce_device:
        require_cuda_tensor(logits, "output logits")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("GPU nonfinite logits")
    return logits


def evaluate_case(model, tokenizer, case, config, wrapper, score_response, output_flags,
                  logit_metrics, base=None, *, enforce_device=True, synchronize=None,
                  guard=None, forward_sink=None):
    """GPU device adaptation of the exact inherited greedy generation contract.

    The testing-only enforce_device=False permits an analytic CPU fake host.
    Native run always uses the default True and checks every tensor's device.
    No canonical-answer call, teacher forcing or future target tokens exist.
    """
    started = time.perf_counter()
    device = next(model.parameters()).device
    ids = encode_prefix(tokenizer, case, device)
    if enforce_device:
        require_cuda_tensor(ids, "rendered ids")
    cap = min(config["max_new_tokens"], case.get("max_new_tokens", config["max_new_tokens"]))
    if type(cap) is not int or not 1 <= cap <= 24 or ids.shape[1] + cap - 1 > 2248:
        raise RuntimeError("GPU case exceeds frozen generation/prefix cap")
    wrapper.records.clear()
    wrapper.calls = 0
    wrapper.min_rms_ratio = wrapper.max_rms_ratio = 1.
    wrapper.capture = True
    calls, positions, forward_seconds, forward_logging_seconds = 0, 0, 0., 0.
    def log_forward(event):
        nonlocal forward_logging_seconds
        if forward_sink:
            start = time.perf_counter()
            forward_sink(event)
            forward_logging_seconds += time.perf_counter() - start
    def forward(prefix):
        nonlocal calls, positions, forward_seconds
        if guard:
            guard()
        before = time.perf_counter()
        log_forward({"event": "FORWARD_STARTED", "case_id": case["case_id"], "full_model_call": calls + 1,
                     "prefix_tokens": prefix.shape[1], "forward_seconds": None, "completed": False})
        try:
            logits = forward_logits(model, prefix, enforce_device=enforce_device, synchronize=synchronize)
        except Exception as error:
            log_forward({"event": "FORWARD_FAILED", "case_id": case["case_id"], "full_model_call": calls + 1,
                         "prefix_tokens": prefix.shape[1], "forward_seconds": time.perf_counter() - before,
                         "completed": False, "completed_model_calls": 0, "partial_compute": "UNKNOWN",
                         "error": f"{type(error).__name__}: {error}"})
            raise
        elapsed = time.perf_counter() - before
        calls += 1
        positions += prefix.shape[1]
        forward_seconds += elapsed
        sample = guard() if guard else None
        log_forward({"event": "FORWARD_COMPLETED", "completed": True, "case_id": case["case_id"], "full_model_call": calls,
                          "prefix_tokens": prefix.shape[1], "forward_seconds": elapsed,
                          "resources": sample})
        return logits
    logits = forward(ids)
    first = logits.clone()
    hidden = list(wrapper.records)
    wrapper.capture = False
    prefill = time.perf_counter() - started
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    tokens, finish, text = [], "token_cap", ""
    decode_started = time.perf_counter()
    for step in range(cap):
        token = int(logits.argmax().item())
        tokens.append(token)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        flags = output_flags(text, tokens)
        if token in eos:
            finish = "eos"
            break
        if flags["repetition_loop"]:
            finish = "repetition_loop"
            break
        if flags["script_corruption"] or flags["control_count"] or flags["replacement_count"] >= 3:
            finish = "script_or_unicode_corruption"
            break
        if time.perf_counter() - started > config["max_prompt_seconds"]:
            finish = "timeout"
            break
        if guard:
            guard()
        if step + 1 < cap:
            ids = torch.cat((ids, torch.tensor([[token]], dtype=ids.dtype, device=ids.device)), dim=1)
            logits = forward(ids)
    if synchronize:
        synchronize()
    decode_seconds = time.perf_counter() - decode_started
    row = {"case_id": case["case_id"], "family": case["family"], "scenario_id": case["scenario_id"],
           "response": text, "token_ids": tokens, **output_flags(text, tokens), "termination": finish,
           "eos_emitted": finish == "eos", "truncated": finish == "token_cap",
           "safety_stopped": finish not in ("eos", "token_cap"),
           "score": score_response(text, case["scorer"])["score"],
           "format_score": score_response(text, case["format_scorer"])["score"] if case.get("format_scorer") else None,
           "gold_nll": None, "gold_answer": None, "gold_nll_status": "NOT_COLLECTED_CAUSAL_GUARD",
           "prefill_seconds": prefill, "decode_seconds": decode_seconds,
           "decode_tokens_per_second": max(0, len(tokens) - 1) / max(decode_seconds, 1e-12),
           "wall_seconds": time.perf_counter() - started, "synchronized_forward_seconds": forward_seconds,
           "forward_logging_seconds": forward_logging_seconds,
           "processed_token_positions": positions, "full_model_calls_including_gold_nll": calls,
           "total_token_positions_including_gold_nll": positions, "layer_calls_generation": wrapper.calls,
           "initial_block_applications": calls,
           "extra_layer_applications": calls * wrapper.extra_layer_applications,
           "controller_seconds": getattr(wrapper, "controller_seconds", None),
           "controller_logging_seconds": getattr(wrapper, "logging_seconds", None), "hidden": hidden,
           "local_timing_scope": "synchronized original-layer calls and moment observer; first-pass core coversL13only",
           "native_conditioning_checks": getattr(wrapper, "gpu_conditioning_checks", None),
           "trajectory_min_rms_ratio": wrapper.min_rms_ratio, "trajectory_max_rms_ratio": wrapper.max_rms_ratio,
           **logit_metrics(first if base is None else base, first, config["top_k"], None)}
    c = model.config
    hd, inter = c.hidden_size, c.intermediate_size
    q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    per_layer = 2 * (hd*q + 2*hd*kv + q*hd + 3*hd*inter)
    depth = c.num_hidden_layers + wrapper.extra_layer_applications
    row["generation_dense_matmul_flops_estimate"] = int(positions * depth * per_layer + len(tokens) * 2 * hd * c.vocab_size)
    return row, first


def validate_admission(admission, binding, source):
    required = {"status": "READY_FOR_GPU_FP32_DISPATCH", "device": "cuda:0", "dtype": "float32",
                "attn_implementation": "eager", "gpu_allocated_ceiling_bytes": 5 * GIB,
                "gpu_global_free_reserve_bytes": GIB, "gpu_initial_free_floor_bytes": 6 * GIB,
                "worker_host_available_floor_bytes": 6 * GIB, "worker_rss_ceiling_bytes": 6 * GIB}
    if any(admission.get(key) != value for key, value in required.items()):
        raise RuntimeError("prospective GPU admission/device/resource contract differs")
    if admission.get("source") != str(source) or admission.get("training") is not False or admission.get("native_test") is not False:
        raise RuntimeError("GPU source or zero-training/TEST authority differs")
    if admission.get("maximum_shapes") != binding["shapes"] or binding["shapes"]["max_forward_prefix_tokens"] != 2248:
        raise RuntimeError("GPU maximum shape not bound to exact tokenizer packet")
    if admission.get("checkpoint_files") != binding["checkpoint_files"]:
        raise RuntimeError("GPU canonical asset contract differs")


def run(source, out, admission_path, stage="qualification"):
    source, out, admission_path = Path(source).resolve(), Path(out).resolve(), Path(admission_path).resolve(strict=True)
    if PACKAGE not in out.parents or PACKAGE not in admission_path.parents:
        raise PermissionError("GPU output/admission must be campaign-local")
    out.mkdir(parents=True, exist_ok=False)
    started, model = time.perf_counter(), None
    receipt = {"status": "FAIL", "lane": "GPU_FP32_SUCCESSOR", "native_model_loads": 0,
               "training": False, "native_test": False, "started_utc": datetime.now(timezone.utc).isoformat()}
    historical = None
    try:
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        binding_path = PACKAGE / "runs/binding-001/MODEL_DATA_BINDING.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        validate_admission(admission, binding, source)
        if admission.get("binding_sha256") != sha(binding_path):
            raise RuntimeError("GPU metadata packet changed")
        review_path = (PACKAGE / admission["review_receipt"]).resolve(strict=True)
        if PACKAGE not in review_path.parents:
            raise PermissionError("GPU review must be campaign-local")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("status") != "PASS_GPU_FP32_DISPATCH" or review.get("admission_sha256") != sha(admission_path):
            raise RuntimeError("independent GPU review does not bind this admission")
        for relative, digest in admission["dispatch_hashes"].items():
            path = (REPO / relative).resolve(strict=True)
            allowed_code = relative.endswith(".py") and relative.startswith((
                "experiments/rineya_taps_adaptive_recurrence_v1/",
                "experiments/anchor_routed_middle_recurrence_v1/src/",
                "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/"))
            allowed_campaign = PACKAGE in path.parents
            allowed_data = relative in ("Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/mergedev_fast.jsonl",
                                       "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl")
            if not (allowed_code or allowed_campaign or allowed_data) or REPO not in path.parents or sha(path) != digest:
                raise RuntimeError("GPU source binding changed: " + relative)
        historical, inherited_hashes = admitted_source(source)
        cache = out / "hf_cache"
        os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"),
                          HF_MODULES_CACHE=str(cache / "modules"), HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        if (platform.python_version() != historical["python"] or torch.__version__ != historical["torch"]
                or transformers.__version__ != historical["transformers"]
                or sha(inspect.getfile(Qwen3DecoderLayer)) != historical["qwen_implementation_sha256"]):
            raise RuntimeError("GPU successor exact source/dependency runtime differs")
        if not torch.cuda.is_available() or psutil.virtual_memory().available < 6 * GIB:
            raise MemoryError("GPU/CUDA or separate GPU host6GiB floor unavailable")
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        initial = gpu_resources()
        if initial["gpu_free_bytes"] < 6 * GIB:
            raise MemoryError("GPU initial free floor6GiB unavailable")
        config = {**historical["config"], "device": "cuda:0", "dtype": "float32", "max_rss_gib": 6}
        write_json(out / "registration.json", {"lane": "GPU_FP32_SUCCESSOR", "admission_sha256": sha(admission_path),
                   "binding_sha256": sha(binding_path), "source_hashes": inherited_hashes,
                   "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                               "transformers": transformers.__version__, "cuda": torch.version.cuda,
                               "device_name": torch.cuda.get_device_name(0), "tf32": False,
                               "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]},
                   "config": config, "grid": GRID, "seeds": SEEDS, "native_layers": [12, 13],
                   "extra_loops": [0, 1], "taps": asdict(TAPSConfig()), "initial_resources": initial,
                   "maximum_shapes": binding["shapes"], "training": False, "native_test": False,
                   "gold_nll_status": "NOT_COLLECTED_CAUSAL_GUARD"})
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False, cache_dir=cache / "hub")
        receipt["model_load_attempts"] = 1
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=True,
                    dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True,
                    device_map={"": "cuda:0"}, cache_dir=cache / "hub").eval().requires_grad_(False)
        receipt["native_model_loads"] = 1
        # Loading peaks count toward the prospective ceiling; do not erase them.
        receipt["after_load_resources"] = resource_guard()
        if type(model).__name__ != "Qwen3AbstainForCausalLM":
            raise RuntimeError("GPU custom abstention-aware class required")
        code_path = Path(inspect.getfile(type(model))).resolve()
        if cache not in code_path.parents or sha(code_path) != historical["checkpoint_files_before"]["modeling_supa_a11.py"]:
            raise RuntimeError("actual loaded custom head code not exact campaign-local bytes")
        if any(parameter.dtype != torch.float32 or parameter.device.type != "cuda" or parameter.device.index != 0
               or parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("GPU model parameters are not frozen FP32 cuda:0")
        for name, buffer in model.named_buffers():
            require_cuda_tensor(buffer, "model buffer " + name)
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                raise RuntimeError("GPU floating model buffer is not FP32: " + name)
        if model.config.abstain_token_id != 51378 or model.config.abstain_logit_bias != 5.156249985098839:
            raise RuntimeError("GPU custom head offset identity differs")
        sys.path[:0] = [str(ARMR / "src"), str(LAB / "src")]
        from armr.provenance import cases_from, checkpoint_files, state_hash
        from armr.recurrence import FixedBlock
        from armr.hooks import Reapplication
        from armr.metrics import output_flags, logit_metrics
        from armr.reporting import summarize as smoke_gate
        from rineya_cross_scale.evaluation.scoring import score_response
        before_files, before_state = checkpoint_files(source), state_hash(model)
        if before_state != historical["state_hash_before"]:
            raise RuntimeError("GPU loaded FP32 state identity differs from exact canonical state")
        receipt.update(loaded_state_sha256=before_state, loaded_custom_code_sha256=sha(code_path),
                       loaded_custom_code_path=str(code_path))
        prompt_meta = json.loads((binding_path.parent / "PROMPT_AUDIT.json").read_text(encoding="utf-8"))
        for split, name in (("SMOKE", config["smoke_file"]), ("DEV", config["development_file"])):
            path = LAB / "data/evaluation" / name
            if sha(path) != binding["datasets"][split]["sha256"]:
                raise RuntimeError("GPU data binding changed")
        smoke = cases_from(LAB / "data/evaluation" / config["smoke_file"])
        dev = cases_from(LAB / "data/evaluation" / config["development_file"])
        for case in smoke + dev:
            actual = encode_prefix(tokenizer, case, torch.device("cuda:0"))[0].tolist()
            expected = next(item for item in prompt_meta if item["case_id"] == case["case_id"])
            if actual != expected["input_token_ids"]:
                raise RuntimeError("GPU rendered token IDs differ from exact CPU tokenizer packet")
        resource_guard()
        rows, traces, bases, qualification_forward_times = [], [], {}, []
        taps = TAPSConfig()
        unit = replace(taps, scheduler_mode="unit")
        synchronize = lambda: torch.cuda.synchronize(0)
        with Journal(out / "outcomes.jsonl") as outcomes, Journal(out / "trajectories.jsonl") as trajectory, Journal(out / "forwards.jsonl") as forwards, one_shot_anchors(model):
            def evaluate(case, arm, cfg, extra, seed, split, kind="successor"):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                cfg = replace(cfg, seed=seed)
                case_start = time.perf_counter()
                def sink(event):
                    event = {**event, "case_id": case["case_id"], "split": split, "seed": seed,
                             "arm": arm, "extra_loops": extra, "configured_seed": cfg.seed,
                             "fixed_alpha": cfg.fixed_alpha if cfg.scheduler_mode == "fixed" else None}
                    trajectory.append(event)
                    if arm == "A3" and split == "DEV" and event["iteration"] > 0:
                        traces.append(event)
                guard = lambda: resource_guard(case_start, config["max_prompt_seconds"])
                if kind == "legacy":
                    wrapper = FixedBlock(model, [12, 13], applications=2) if extra else Reapplication(model)
                else:
                    cls = ReferenceBlock if kind == "reference" else BlockRecurrence
                    kwargs = {"arm_id": arm} if kind == "successor" else {}
                    wrapper = cls(model, extra_loops=extra, config=cfg, sample_id=case["case_id"], sink=sink, guard=guard, **kwargs)
                with wrapper, torch.inference_mode():
                    install_gpu_timing(wrapper, synchronize)
                    def record_forward(event):
                        if event.get("completed") is True:
                            qualification_forward_times.append(event["forward_seconds"])
                        forwards.append({**event, "arm": arm, "seed": seed, "split": split, "extra_loops": extra})
                    row, logits = evaluate_case(model, tokenizer, case, config, wrapper, score_response,
                        output_flags, logit_metrics, bases.get((split, seed, case["case_id"])),
                        synchronize=synchronize, guard=guard,
                        forward_sink=record_forward)
                    row.update(arm=arm, split=split, seed=seed, configured_seed=cfg.seed, extra_loops=extra,
                               fixed_alpha=cfg.fixed_alpha if cfg.scheduler_mode == "fixed" else None,
                               physical_run=True, deployable=True, lane="GPU_FP32_SUCCESSOR")
                    outcomes.append(row)
                    if row["safety_stopped"]:
                        raise RuntimeError("GPU safety stop; complete matrix required")
                return row, logits
            qualification = []
            for case in smoke:
                for extra in (0, 1):
                    old, old_logits = evaluate(case, "qualification_legacy", unit, extra, 17, "SMOKE", "legacy")
                    new, new_logits = evaluate(case, "qualification_new", unit, extra, 17, "SMOKE")
                    if old["token_ids"] != new["token_ids"] or not torch.equal(old_logits, new_logits):
                        raise RuntimeError("GPU no-extra/unit legacy equivalence failed")
                old, old_logits = evaluate(case, "qualification_reference_A3", taps, 1, 17, "SMOKE", "reference")
                new, new_logits = evaluate(case, "qualification_successor_A3", taps, 1, 17, "SMOKE")
                if old["token_ids"] != new["token_ids"] or not torch.equal(old_logits, new_logits):
                    raise RuntimeError("GPU original/new A3 equivalence failed")
            max_case = max(prompt_meta, key=lambda item: item["input_tokens"])
            base_ids = torch.tensor([max_case["input_token_ids"]], dtype=torch.long, device="cuda:0")
            filler = tokenizer.encode(".", add_special_tokens=False)[0]
            max_ids = torch.cat((base_ids, torch.full((1, 2248 - base_ids.shape[1]), filler, device="cuda:0", dtype=torch.long)), dim=1)
            maximum_forward_times, maximum_attempt_count = [], 0
            def maximum_forward(ids):
                nonlocal maximum_attempt_count
                maximum_attempt_count += 1
                begin = time.perf_counter()
                forwards.append({"event": "MAXIMUM_FORWARD_STARTED", "attempt": maximum_attempt_count,
                                 "prefix_tokens": ids.shape[1], "completed": False, "forward_seconds": None,
                                 "task_measurement": False})
                try:
                    value = forward_logits(model, ids, synchronize=synchronize)
                except Exception as error:
                    forwards.append({"event": "MAXIMUM_FORWARD_FAILED", "attempt": maximum_attempt_count,
                                     "prefix_tokens": ids.shape[1], "forward_seconds": time.perf_counter() - begin,
                                     "completed": False, "completed_model_calls": 0, "partial_compute": "UNKNOWN",
                                     "error": f"{type(error).__name__}: {error}", "resources": gpu_resources(),
                                     "task_measurement": False})
                    raise
                elapsed = time.perf_counter() - begin
                maximum_forward_times.append(elapsed)
                forwards.append({"event": "MAXIMUM_FORWARD_COMPLETED", "attempt": maximum_attempt_count,
                                 "completed": True, "arm": "MAXIMUM_SHAPE_QUALIFICATION", "prefix_tokens": ids.shape[1],
                                 "forward_seconds": elapsed, "resources": resource_guard(), "task_measurement": False})
                return value
            originals = [block.forward for block in model.model.layers]
            with torch.inference_mode():
                intact = maximum_forward(max_ids)
                with BlockRecurrence(model, extra_loops=0, config=taps, arm_id="maximum_noextra") as wrapper:
                    install_gpu_timing(wrapper, synchronize)
                    noextra = maximum_forward(max_ids)
                with FixedBlock(model, [12, 13], applications=2):
                    legacy = maximum_forward(max_ids)
                with BlockRecurrence(model, config=unit, arm_id="maximum_unit") as wrapper:
                    install_gpu_timing(wrapper, synchronize)
                    new = maximum_forward(max_ids)
                if not torch.equal(intact, noextra) or not torch.equal(legacy, new):
                    raise RuntimeError("GPU maximum-shape legacy identity failed")
                with ReferenceBlock(model, config=taps) as wrapper:
                    install_gpu_timing(wrapper, synchronize)
                    old_adaptive = maximum_forward(max_ids)
                with BlockRecurrence(model, config=taps, arm_id="maximum_A3") as wrapper:
                    install_gpu_timing(wrapper, synchronize)
                    new_adaptive = maximum_forward(max_ids)
                if not torch.equal(old_adaptive, new_adaptive):
                    raise RuntimeError("GPU maximum-shape original/new A3 identity failed")
                try:
                    def failure(_event):
                        raise RuntimeError("EXPECTED_FORCED_RESTORATION_EXCEPTION")
                    with BlockRecurrence(model, config=taps, sink=failure) as wrapper:
                        install_gpu_timing(wrapper, synchronize)
                        maximum_forward(max_ids)
                except RuntimeError as error:
                    if str(error) != "EXPECTED_FORCED_RESTORATION_EXCEPTION":
                        raise
                else:
                    raise RuntimeError("forced native exception was not exercised")
                if any(block.forward != original for block, original in zip(model.model.layers, originals)):
                    raise RuntimeError("GPU wrapper did not restore layer methods")
                if not torch.equal(intact, maximum_forward(max_ids)):
                    raise RuntimeError("GPU restored maximum-shape logits differ")
                for cache_kwargs in ({"use_cache": True}, {"use_cache": False, "past_key_values": object()}):
                    with BlockRecurrence(model, config=unit) as wrapper:
                        install_gpu_timing(wrapper, synchronize)
                        begin = time.perf_counter()
                        forwards.append({"event": "CACHE_REJECTION_STARTED", "use_cache": cache_kwargs["use_cache"],
                                         "cache_object_present": "past_key_values" in cache_kwargs,
                                         "completed": False, "task_measurement": False})
                        try:
                            model(input_ids=max_ids[:, :1], **cache_kwargs)
                        except RuntimeError as error:
                            if "cache" not in str(error):
                                raise
                            forwards.append({"event": "CACHE_REJECTED", "completed": False,
                                             "completed_model_calls": 0, "model_compute_started": False,
                                             "wrapper_forward_count": wrapper.forward_count,
                                             "forward_seconds": time.perf_counter() - begin, "resources": gpu_resources(),
                                             "error": str(error), "task_measurement": False})
                        else:
                            raise RuntimeError("GPU cache rejection failed")
            qualification.append({"maximum_prefix_tokens": 2248, "synthetic_extension": True,
                                  "targets_used": False, "identity": "PASS", "forced_exception_restoration": "PASS",
                                  "cache_rejection": "PASS", "resources": resource_guard()})
            arms = [("A0", unit, 0), ("A1", unit, 1)] + [("A2", replace(taps, scheduler_mode="fixed", fixed_alpha=a), 1) for a in GRID] + [("A3", taps, 1)]
            baseline_smoke = []
            for arm, cfg, extra in arms:
                current = []
                for case in smoke:
                    row, logits = evaluate(case, arm, cfg, extra, 17, "SMOKE")
                    current.append(row)
                    if arm == "A0":
                        bases[("SMOKE", 17, case["case_id"])] = logits
                if arm == "A0":
                    baseline_smoke = current
                elif not smoke_gate(current, baseline_smoke, config)["go_fixed_smoke"]:
                    raise RuntimeError("GPU required smoke gate failed: " + arm + ":" + str(cfg.fixed_alpha))
                with torch.inference_mode(), BlockRecurrence(model, extra_loops=extra, config=cfg, arm_id="maximum_" + arm) as wrapper:
                    install_gpu_timing(wrapper, synchronize)
                    maximum_forward(max_ids)
                resource_guard()
            synchronize()
            qualified_state = state_hash(model)
            qualified_files = checkpoint_files(source)
            if qualified_state != before_state or qualified_files != before_files or sha(code_path) != historical["checkpoint_files_before"]["modeling_supa_a11.py"]:
                raise RuntimeError("GPU qualification changed model/assets/custom code before DEV")
            if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("GPU qualification produced trainable parameters/gradients")
            if any("_taps_owner" in block.__dict__ or "_armr_owner" in block.__dict__ for block in model.model.layers):
                raise RuntimeError("GPU qualification left recurrence ownership active")
            write_json(out / "qualification.json", {"status": "QUALIFIED_GPU_FP32_REFERENCE_OPERATOR", "maximum_shape": qualification,
                       "original_new_A3_identity": "PASS", "L0_L2_one_shot": "PASS_EVERY_COMPLETED_FORWARD",
                       "scaled_arms_smoke": "PASS", "A5_frozen_pool_qualification": "PENDING_DEV_POOL",
                       "before_state_sha256": before_state, "after_state_sha256": qualified_state,
                       "checkpoint_files_unchanged": qualified_files == before_files,
                       "native_task_efficacy": False,
                       "mask_positions": {"input_mask": "all-ones enforced; no padding",
                                          "positions": "nativeL12 exactCUDA arange(0,prefixlen) checked",
                                          "replay_conditioning": "same position/mask/position_embedding objects checked",
                                          "causal_mask": "pinned eager source; runtimeCUDA device/full-prefix shape checked; full triangular comparison not claimed"},
                       "local_timing": "GPU-only original-call and moment-observer synchronization; first-pass coreL13only",
                       "maximum_attempts_before_DEV": maximum_attempt_count,
                       "cache_rejection_attempts": 2})
            if stage == "dev":
                remaining_calls = 18 * sum(item["max_new_tokens"] for item in prompt_meta if item["split"] == "DEV") + sum(item["max_new_tokens"] for item in prompt_meta if item["split"] == "SMOKE") + 1
                per_call = max(qualification_forward_times + maximum_forward_times)
                estimate = remaining_calls * per_call * 1.5
                elapsed = time.perf_counter() - started
                reserve = admission.get("native_finalization_reserve_seconds", 1800)
                deadline = admission.get("hard_deadline_seconds", 7200)
                forecast = {"remaining_generation_and_maximum_calls_cap": remaining_calls,
                            "max_observed_synchronized_forward_seconds": per_call,
                            "forecast_factor": 1.5, "remaining_forecast_seconds": estimate,
                            "already_elapsed_worker_seconds": elapsed, "finalization_reserve_seconds": reserve,
                            "parent_deadline_seconds": deadline,
                            "admitted": elapsed + estimate + reserve <= deadline,
                            "scope": "conservative max-observed diagnostic forward cost; estimate not speed claim"}
                write_json(out / "matrix_forecast.json", forecast)
                if not forecast["admitted"]:
                    raise RuntimeError("GPU qualified mechanics but complete registered DEV matrix forecast does not fit deadline/reserve")
                for seed in SEEDS:
                    ordered = arms[:1] + (arms[1:] if seed == 17 else list(reversed(arms[1:])))
                    for arm, cfg, extra in ordered:
                        for case in dev:
                            row, logits = evaluate(case, arm, cfg, extra, seed, "DEV")
                            rows.append(row)
                            if arm == "A0":
                                bases[("DEV", seed, case["case_id"])] = logits
                scores = {alpha: statistics.mean(row["score"] for row in rows if row["arm"] == "A2" and row["fixed_alpha"] == alpha) for alpha in GRID}
                best = max(GRID, key=lambda alpha: (scores[alpha], -alpha))
                pool = tuple(min(1.2, max(.8, event["alpha"])) for event in traces) or (1.,)
                write_json(out / "frozen_dev_selection.json", {"alpha": best, "pool": pool, "config": asdict(taps),
                    "dataset_sha256": binding["datasets"]["DEV"]["sha256"], "binding_sha256": sha(binding_path),
                    "generation_trajectory_sha256_at_freeze": sha(out / "trajectories.jsonl"),
                    "pool_weighting": "generated prefixes across two nominal deterministic A3 repeats",
                    "native_TEST_authorized": False, "A5_configured_seed": "actual replicate17/29"})
                random_cfg = replace(taps, scheduler_mode="random", random_pool=pool)
                random_smoke = [evaluate(case, "A5", random_cfg, 1, 17, "SMOKE")[0] for case in smoke]
                if not smoke_gate(random_smoke, baseline_smoke, config)["go_fixed_smoke"]:
                    raise RuntimeError("GPU corrected A5 frozen-pool smoke failed")
                with torch.inference_mode(), BlockRecurrence(model, config=random_cfg, arm_id="maximum_A5") as wrapper:
                    install_gpu_timing(wrapper, synchronize)
                    maximum_forward(max_ids)
                resource_guard()
                for seed in SEEDS:
                    for case in dev:
                        row, _ = evaluate(case, "A5", random_cfg, 1, seed, "DEV")
                        rows.append(row)
                for seed in SEEDS:
                    for case in dev:
                        fixed = [row for row in rows if row["arm"] == "A2" and row["seed"] == seed and row["case_id"] == case["case_id"]]
                        selected = max(fixed, key=lambda row: (row["score"], -row["fixed_alpha"]))
                        outcomes.append({**selected, "arm": "A4_example", "physical_run": False, "deployable": False,
                                         "note": "retrospective finite constant-grid reuse; not token-prefix schedule oracle"})
                write_json(out / "A5_qualification.json", {"status": "PASS_FROZEN_POOL_SMOKE_AND_MAXIMUM_SHAPE", "actual_seed_binding": True})
        synchronize()
        if state_hash(model) != before_state or checkpoint_files(source) != before_files or sha(code_path) != historical["checkpoint_files_before"]["modeling_supa_a11.py"]:
            raise RuntimeError("GPU model/source/custom head changed")
        receipt.update(status="GPU_DEV_COMPLETE_NOT_TESTED" if stage == "dev" else "GPU_QUALIFIED_NO_DEV_MEASUREMENT",
                       source_unchanged=True, checkpoint_files_unchanged=True, trainable_parameters=0,
                       final_resources=resource_guard(), original_cpu_reproduction=False)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt.update(wall_seconds=time.perf_counter() - started, ended_utc=datetime.now(timezone.utc).isoformat())
        if model is not None:
            receipt["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        receipt["final_resources_on_stop"] = final_resource_snapshot()
        write_json(out / "receipt.json", receipt)
    return 0 if receipt["status"] in ("GPU_DEV_COMPLETE_NOT_TESTED", "GPU_QUALIFIED_NO_DEV_MEASUREMENT") else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--stage", choices=("qualification", "dev"), default="qualification")
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.out, args.admission, args.stage))
