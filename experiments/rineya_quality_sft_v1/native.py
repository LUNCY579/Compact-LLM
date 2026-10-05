"""Local FP32 source binding and zero-update RQSFT mechanics, without grants.

MQ supplies audited numerical primitives only. Its permits, source locks, native
validators and checkpoint experiment identity do not authorize this package.
Importing this module loads no model and changes no runtime policy.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import random
import math
import os
import sys
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.model_quality_gpu_v1.losses import chunked_sum
from experiments.model_quality_gpu_v1.native import hidden_states, parameter_digest, select_tail
from experiments.model_quality_gpu_v1.optimizer import CpuAdamW
from . import EXPERIMENT, SOURCE_CORE, SOURCE_WEIGHT_SHA256
from .contract import Design, SOURCE_PATH, external_output


PARAMETERS = 596049920
TAIL_PARAMETERS = 314618880
TAIL_TENSORS = 220
TOKENIZER_EOS = 151645
PARITY_RTOL, PARITY_ATOL = 1e-5, 1e-5
LOSS_GRAD_RTOL, LOSS_GRAD_ATOL = 1e-5, 1e-6
RUNTIME_PINS = {"torch": "2.10.0", "transformers": "4.57.1", "safetensors": "0.7.0", "tokenizers": "0.22.2"}
GENERATION_RECIPE = {"sampling": "greedy", "max_new_tokens": 64, "maximum_sequence_tokens": 2048,
                     "use_cache": True, "arithmetic": "fp32_eager_no_tf32", "attention": "math_sdpa"}


def _helpers():
    from . import integrity
    return integrity


def _tensor_hash(named):
    integrity = _helpers()
    digest = hashlib.sha256()
    for name, tensor in sorted(named):
        digest.update(integrity.canonical([name, list(tensor.shape), str(tensor.dtype)]))
        flat = tensor.detach().reshape(-1)
        for index in range(0, flat.numel(), 262144):
            value = flat[index:index + 262144].cpu().contiguous().view(torch.uint8).numpy()
            digest.update(memoryview(value))
    return digest.hexdigest()


def buffer_digest(model):
    return _tensor_hash(model.named_buffers())


def freeze_state_digest(model):
    """Fixed tail complement plus buffers, independent of requires_grad flags."""
    n = len(model.model.layers)
    prefixes = tuple(f"model.layers.{index}." for index in range(max(0, n - 20), n))
    tensors = [("parameter:" + name, p) for name, p in model.named_parameters() if not name.startswith(prefixes)]
    tensors += [("buffer:" + name, value) for name, value in model.named_buffers()]
    return _tensor_hash(tensors)


def source_custody(source):
    """Hash complete local file custody; no source lock or another permit imported."""
    integrity = _helpers()
    source = Path(source)
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Materialized source custody required")
    files = {path.relative_to(source).as_posix(): integrity.file_hash(path)
             for path in sorted(source.rglob("*")) if path.is_file()}
    if files.get("model.safetensors") != SOURCE_WEIGHT_SHA256:
        raise ValueError("Exact ungated Supa 1.7 source weight SHA required")
    if {name for name in files if Path(name).suffix in {".bin", ".pt", ".pth", ".safetensors", ".gguf"}} != {"model.safetensors"}:
        raise ValueError("Partial, alternative or extra source weight inventory rejected")
    return {"source_core": SOURCE_CORE, "source_weight_sha256": SOURCE_WEIGHT_SHA256,
            "source_path": str(source.resolve()), "assets": files, "sha256": integrity.digest(files)}


def runtime():
    libraries = {name: importlib.metadata.version(name) for name in RUNTIME_PINS}
    try:
        libraries["psutil"] = importlib.metadata.version("psutil")
    except importlib.metadata.PackageNotFoundError:
        libraries["psutil"] = None
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu = None
    if count:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {"name": properties.name, "total_memory": properties.total_memory,
               "capability": [properties.major, properties.minor], "device_index": index,
               "uuid": str(getattr(properties, "uuid", "unavailable"))}
    return {"python": platform.python_version(), "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(), "libraries": libraries, "cuda_build": torch.version.cuda,
            "cuda_device_count": count, "gpu0": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "arithmetic": "fp32_eager_no_tf32", "attention": "math_sdpa",
            "optimizer": "cpu_fp32_moments", "deterministic": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "generation": dict(GENERATION_RECIPE)}


def configure_runtime(seed=17):
    """Caller configures the reviewed worker before model load; never at import."""
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


@contextmanager
def math_attention():
    with sdpa_kernel(SDPBackend.MATH):
        yield


def _geometry(config):
    wanted = {"model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
              "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
              "head_dim": 128, "vocab_size": 151936, "tie_word_embeddings": True,
              "use_sliding_window": False, "attention_dropout": 0.0}
    if (any(getattr(config, key, None) != value for key, value in wanted.items())
            or getattr(config, "auto_map", None) or getattr(config, "quantization_config", None)
            or getattr(config, "rope_scaling", None)):
        raise ValueError("Exact dense, unscaled Supa Qwen3 geometry required")


def bind_model(source, training=False):
    """Exact local source, no missing/random weights, download or precision fallback."""
    source = Path(source)
    if source.resolve() != Path(SOURCE_PATH).resolve():
        raise ValueError("Native RQSFT accepts only the registered ungated source path")
    before = source_custody(source)
    versions = runtime()
    if any(versions["libraries"][name].split("+")[0] != version for name, version in RUNTIME_PINS.items()):
        raise ValueError("Reviewed native dependency pins unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("Native CUDA unavailable; no CPU fallback")
    if not torch.are_deterministic_algorithms_enabled() or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("Configure deterministic FP32/no-TF32 worker before loading")
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
    _geometry(config)
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation="sdpa", device_map=None, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Refusing partial/randomly initialized model load")
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETERS:
        raise ValueError("Source parameter inventory differs")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("Native vocabulary tying lost")
    if any(value.is_meta or value.dtype != torch.float32 or not torch.isfinite(value).all()
           for _, value in model.named_parameters()):
        raise ValueError("Source requires finite materialized FP32 parameters")
    if any(value.is_meta or not torch.isfinite(value).all() for _, value in model.named_buffers()):
        raise ValueError("Source has nonfinite/unmaterialized buffers")
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id != TOKENIZER_EOS or not tokenizer.chat_template:
        raise ValueError("Exact pinned fast tokenizer/chat-template/EOS required")
    model.requires_grad_(False)
    if training:
        named = select_tail(model, 20)
        if len(named) != TAIL_TENSORS or sum(p.numel() for _, p in named) != TAIL_PARAMETERS:
            raise ValueError("Exact blocks8..27 trainable inventory required")
    model.config.use_cache = False
    model.to("cuda").eval()
    stop_ids = getattr(model.generation_config, "eos_token_id", None)
    stop_ids = [stop_ids] if type(stop_ids) is int else stop_ids
    if (not isinstance(stop_ids, list) or not stop_ids or TOKENIZER_EOS not in stop_ids
            or len(set(stop_ids)) != len(stop_ids)
            or any(type(value) is not int or not 0 <= value < model.config.vocab_size for value in stop_ids)):
        raise ValueError("Complete official generation EOS inventory required")
    after = source_custody(source)
    if before != after:
        raise RuntimeError("Source custody changed while loading")
    model._rqsft_source = str(source.resolve())
    info = dict(before, parameters=PARAMETERS, initial_parameters_sha256=parameter_digest(model),
                initial_frozen_state_sha256=freeze_state_digest(model), runtime=versions,
                attention_backend="math_sdpa", training=bool(training), eos_ids=list(stop_ids),
                generation_recipe=dict(GENERATION_RECIPE, eos_ids=list(stop_ids)))
    model._rqsft_source_info = info
    return model, tokenizer, info


def _guard(guard, stage, **cost):
    if guard is None:
        return {"scope": "fixture_no_native_authority"}
    return guard(stage=stage, **cost)


def _forward(model, ids, guard=None, **kwargs):
    _guard(guard, "forward_before", input_tokens=int(ids.shape[-1]), forward_calls=1)
    try:
        return model(input_ids=ids, return_dict=True, **kwargs)
    finally:
        _guard(guard, "forward_after")


@torch.no_grad()
def generate(model, row, *, tokenizer, eos_ids, guard=None, use_cache=True):
    """Greedy final-answer generation; gold/target/reference are never inputs."""
    prefix = row.get("prompt_ids")
    if hasattr(model, "_rqsft_source") and guard is None:
        raise PermissionError("Native generation requires the reviewed owned resource guard")
    if not isinstance(prefix, list) or not prefix or len(prefix) + 64 > 2048:
        raise ValueError("Complete generation prefix plus64 must fit2048 without truncation")
    if (not eos_ids or len(set(eos_ids)) != len(eos_ids)
            or any(type(value) is not int or not 0 <= value < model.config.vocab_size for value in eos_ids)):
        raise ValueError("Explicit source-bound EOS stop inventory required")
    device = next(model.parameters()).device
    response, past, counters = [], None, {"forward_calls": 0, "processed_input_tokens": 0}
    model.eval()
    try:
        with math_attention():
            for _ in range(64):
                values = prefix + response if past is None or not use_cache else [response[-1]]
                ids = torch.tensor([values], dtype=torch.long, device=device)
                _guard(guard, "generation_token_before", generated_tokens=1)
                counters["forward_calls"] += 1
                counters["processed_input_tokens"] += len(values)
                out = _forward(model, ids, guard, use_cache=use_cache, past_key_values=past)
                logits = out.logits[0, -1]
                if logits.numel() != model.config.vocab_size or not torch.isfinite(logits).all():
                    raise FloatingPointError("Invalid generation full-vocabulary logits")
                token = int(logits.argmax())
                past = out.past_key_values if use_cache else None
                if use_cache and past is None:
                    raise RuntimeError("Requested native KV cache was not returned; no fallback")
                response.append(token)
                del out, logits
                if token in eos_ids:
                    break
        prediction = tokenizer.decode(response, skip_special_tokens=True)
    except BaseException as error:
        error.partial_ids = list(response)
        error.partial_cost = dict(counters)
        try:
            error.partial_prediction = tokenizer.decode(response, skip_special_tokens=True)
        except BaseException as decode_error:
            error.add_note("Partial token decode failed: " + str(decode_error))
            error.partial_prediction = None
        raise
    integrity = _helpers()
    return {"id": row["id"], "prediction": prediction,
            "response_ids": response, "prompt_ids": list(prefix), "prompt_sha256": integrity.digest(prefix),
            "response_tokens": len(response), "stop_reason": "eos" if response[-1] in eos_ids else "length_limit",
            "eos_ids": list(eos_ids), "cache_policy": "enabled" if use_cache else "disabled",
            "cost": counters, "evidence": "RAW_GENERATION_NOT_ADJUDICATION"}


def _objective_check(model, ids, labels, design, guard):
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)
    _guard(guard, "training_forward_before", input_tokens=len(ids), forward_calls=1)
    try:
        hidden = hidden_states(model, torch.tensor([ids], dtype=torch.long, device=device))
    finally:
        _guard(guard, "training_forward_after")
    targets = torch.tensor([labels], dtype=torch.long, device=device)
    loss, count = chunked_sum(hidden, model.lm_head.weight, targets, design.ce_chunk)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite qualification objective")
    _guard(guard, "qualification_backward_before")
    (loss / count).backward()
    _guard(guard, "qualification_backward_after")
    selected = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in selected):
        raise RuntimeError("Missing/nonfinite selected gradient")
    norm = math.sqrt(sum(float(p.grad.double().square().sum()) for _, p in selected))
    value = float(loss.detach())
    # Independently compare masked dense CE and its hidden-state derivative.
    detached = hidden.detach().requires_grad_(True)
    _guard(guard, "independent_loss_hidden_gradient_before")
    chunked, _ = chunked_sum(detached, model.lm_head.weight, targets, design.ce_chunk, recompute=False)
    chunk_gradient = torch.autograd.grad(chunked, detached)[0]
    valid = targets[:, 1:] != -100
    states, y = detached[:, :-1][valid], targets[:, 1:][valid]
    dense = F.cross_entropy(F.linear(states, model.lm_head.weight).float(), y, reduction="sum")
    dense_gradient = torch.autograd.grad(dense, detached)[0]
    if (not torch.allclose(chunked, dense, rtol=LOSS_GRAD_RTOL, atol=LOSS_GRAD_ATOL)
            or not torch.allclose(chunk_gradient, dense_gradient, rtol=LOSS_GRAD_RTOL, atol=LOSS_GRAD_ATOL)):
        raise RuntimeError("Chunked masked objective differs from independently dense loss/hidden gradient")
    _guard(guard, "independent_loss_hidden_gradient_after")
    model.zero_grad(set_to_none=True)
    return {"loss_sum": value, "supervised_tokens": count, "selected_gradient_l2": norm,
            "input_tokens": len(ids), "vocabulary_size": model.config.vocab_size,
            "loss_hidden_gradient_parity": True, "loss_gradient_rtol": LOSS_GRAD_RTOL,
            "loss_gradient_atol": LOSS_GRAD_ATOL, "causal_shift": "labels shifted exactly once inside chunked_sum"}


@torch.no_grad()
def _cache_parity(model, prefix, eos, guard):
    device = next(model.parameters()).device
    ids = list(prefix)
    if len(ids) > 2048:
        raise ValueError("Frozen maximum evaluation shape exceeded")
    ids += [eos] * (2048 - len(ids))
    full = _forward(model, torch.tensor([ids], dtype=torch.long, device=device), guard, use_cache=False)
    if list(full.logits.shape) != [1, 2048, model.config.vocab_size] or not torch.isfinite(full.logits).all():
        raise RuntimeError("Invalid maximum2048 full-vocabulary evaluation forward")
    expected = full.logits[:, -1].detach().clone()
    del full
    prefill = _forward(model, torch.tensor([ids[:-1]], dtype=torch.long, device=device), guard, use_cache=True)
    past = prefill.past_key_values
    if past is None:
        raise RuntimeError("Cache parity requires actual returned KV state")
    del prefill
    cached = _forward(model, torch.tensor([[ids[-1]]], dtype=torch.long, device=device), guard,
                      use_cache=True, past_key_values=past)
    actual = cached.logits[:, -1]
    if (not torch.isfinite(actual).all() or not torch.allclose(expected, actual, rtol=PARITY_RTOL, atol=PARITY_ATOL)
            or not torch.equal(expected.argmax(-1), actual.argmax(-1))):
        raise RuntimeError("Maximum evaluation cached/uncached logits or greedy token differ")
    return {"max_evaluation_tokens": 2048, "cache_parity": True,
            "maximum_logit_abs_error": float((expected - actual).abs().max()),
            "rtol": PARITY_RTOL, "atol": PARITY_ATOL, "greedy_token_equal": True,
            "forward_calls": 3, "cache_forward_calls": 2, "processed_input_tokens": 4096,
            "evidence": "SYNTHETIC_EOS_PADDING_MECHANICS_NOT_DEV_BEHAVIOR"}


def qualify(model, tokenizer, sealedplan, binding, guard=None, *, directory=None):
    from .compiler import validate_plan, targets
    from .engine import checkpoint_roundtrip_zero, verify_binding, select_trainables
    integrity = _helpers()
    plan = validate_plan(sealedplan)
    if plan["kind"] == "native_plan" and guard is None:
        raise PermissionError("Native qualification requires the reviewed owned resource guard")
    verify_binding(model, sealedplan, plan, binding)
    if directory is None:
        raise ValueError("Qualification requires a fresh output for official0-update checkpoint recovery")
    directory = Path(directory)
    if plan["kind"] == "native_plan":
        directory = external_output(directory)
    if not directory.is_dir():
        raise ValueError("Caller must create reviewed qualification output directory")
    named = select_trainables(model, plan)
    initial, frozen, buffers = parameter_digest(model), freeze_state_digest(model), buffer_digest(model)
    source_before = source_custody(model._rqsft_source) if plan["kind"] == "native_plan" else None
    d = Design(**plan["design"])
    if (plan["tokenizer"]["eos_token_id"] != tokenizer.eos_token_id
            or plan["tokenizer"]["chat_template_sha256"] != integrity.digest(tokenizer.chat_template)):
        raise ValueError("Qualification tokenizer differs from exact compiled plan")
    if plan["kind"] == "native_plan":
        for compiled in plan["train"] + plan["dev"]:
            messages = compiled["metadata"]["messages"]
            full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
            prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if full != compiled["ids"] or prefix != compiled["prompt_ids"]:
                raise ValueError("Native full/history-preserving prefix token IDs changed after compilation")
    model.eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    diagnostics = []
    try:
        with math_attention(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            row = plan["train"][0]
            for objective in ("whole", "response"):
                model.train()
                diagnostics.append(dict(_objective_check(model, row["ids"], targets(row, objective), d, guard),
                                        objective=objective, scope="representative_fixture_row" if plan["kind"] == "fixture" else "representative_native_row"))
            eos = tokenizer.eos_token_id
            if type(eos) is not int:
                raise ValueError("Explicit tokenizer EOS required")
            maximum = list(row["ids"])
            maximum += [eos] * (512 - len(maximum))
            if len(maximum) != 512:
                raise ValueError("Representative train row exceeds512; no truncation")
            for objective in ("whole", "response"):
                labels = list(maximum) if objective == "whole" else [-100] * (512 - 64) + maximum[-64:]
                diagnostics.append(dict(_objective_check(model, maximum, labels, d, guard),
                                        objective=objective, scope="maximum512_synthetic_backward"))
            model.eval()
            cache = _cache_parity(model, plan["dev"][0]["prompt_ids"], eos, guard)
        events = sorted({event.key for event in profiler.key_averages() if "scaled_dot_product" in event.key})
        if not any("attention_math" in event for event in events):
            raise RuntimeError("Math SDPA backend was not observed")
        _guard(guard, "checkpoint_recovery_before", cpu_moment_bytes=sum(p.numel() * 8 for _, p in named),
               checkpoint_bytes=sum(p.numel() * 12 for _, p in named) + 16 * 1024**2)
        recovery = checkpoint_roundtrip_zero(model, named, d, directory / "recovery-step0", binding)
        _guard(guard, "checkpoint_recovery_after")
        if (parameter_digest(model) != initial or freeze_state_digest(model) != frozen or buffer_digest(model) != buffers
                or any(p.grad is not None for p in model.parameters())):
            raise RuntimeError("Zero-update qualification changed parameters/buffers or left gradients")
        source_after = source_custody(model._rqsft_source) if source_before is not None else None
        if source_before != source_after:
            raise RuntimeError("Source custody changed during qualification")
        checks = {"parameters_unchanged": True, "frozen_state_unchanged": True, "buffers_unchanged": True,
                  "gradients_cleared": True, "representative_both_objectives": True,
                  "maximum512_both_backward": True, "maximum2048_eval_forward": True,
                  "loss_hidden_gradient_parity": all(item["loss_hidden_gradient_parity"] for item in diagnostics),
                  "cache_parity": cache["cache_parity"], "math_sdpa_observed": True,
                  "checkpoint_recovery_zero_update": recovery["recovery_passed"],
                  "source_custody_unchanged": source_before == source_after,
                  "native_template_ids_parity": plan["kind"] == "native_plan"}
        report = {"experiment": EXPERIMENT, "phase": "zero_update_qualification", "binding": binding,
                  "compiled_plan_sha256": sealedplan["sha256"], "fixture_only": plan["kind"] == "fixture",
                  "status": "FIXTURE_QUALIFIED" if plan["kind"] == "fixture" else "NATIVE_QUALIFIED",
                  "checks": checks, "initial_parameters_sha256": initial,
                  "initial_frozen_state_sha256": frozen, "diagnostics": diagnostics,
                  "backend_events": events, "cache_check": cache, "checkpoint_recovery": recovery,
                  "training_updates": 0, "optimizer_step_calls": 0, "optimizer_objects_created": 1,
                  "source_before": source_before, "source_after": source_after,
                  "efficacy_established": False, "native_run_authorized": False}
        sealed = integrity.seal(report)
        integrity.write_new(directory / "qualification.json", sealed)
        return sealed
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()


def verify_qualification(report, binding):
    integrity = _helpers()
    body = integrity.unseal(report)
    if type(body.get("fixture_only")) is not bool:
        raise ValueError("Explicit native/fixture qualification scope required")
    required = {"parameters_unchanged", "frozen_state_unchanged", "buffers_unchanged", "gradients_cleared",
                "representative_both_objectives", "maximum512_both_backward", "maximum2048_eval_forward",
                "loss_hidden_gradient_parity", "cache_parity", "math_sdpa_observed",
                "checkpoint_recovery_zero_update", "source_custody_unchanged"}
    if (body.get("experiment") != EXPERIMENT or body.get("phase") != "zero_update_qualification"
            or body.get("binding") != binding or body.get("training_updates") != 0
            or body.get("optimizer_step_calls") != 0 or not all(body.get("checks", {}).get(key) is True for key in required)):
        raise ValueError("Missing, stale, foreign or incomplete RQSFT qualification")
    if body.get("fixture_only") and binding.get("fixture_only") is not True:
        raise ValueError("Fixture mechanics cannot authorize native Rineya")
    if body.get("status") != ("FIXTURE_QUALIFIED" if body.get("fixture_only") else "NATIVE_QUALIFIED"):
        raise ValueError("Qualification status and evidence scope differ")
    if not body.get("fixture_only") and body.get("checks", {}).get("native_template_ids_parity") is not True:
        raise ValueError("Actual native full/prefix token parity is required")
    if body.get("initial_parameters_sha256") != binding.get("initial_parameters_sha256"):
        raise ValueError("Qualification/source loaded-state digest differs")
    if body.get("initial_frozen_state_sha256") != binding.get("initial_frozen_state_sha256"):
        raise ValueError("Qualification/source fixed frozen-state digest differs")
    events = body.get("backend_events", [])
    if not any(isinstance(event, str) and "scaled_dot_product" in event and "attention_math" in event for event in events):
        raise ValueError("Qualification lacks observed math-SDPA dispatch witness")
    representative = "representative_fixture_row" if body["fixture_only"] else "representative_native_row"
    expected = {(objective, scope) for objective in ("whole", "response")
                for scope in (representative, "maximum512_synthetic_backward")}
    diagnostics = body.get("diagnostics", [])
    if (len(diagnostics) != 4 or any(not isinstance(item, dict) for item in diagnostics)
            or {(item.get("objective"), item.get("scope")) for item in diagnostics} != expected):
        raise ValueError("Exactly four objective/shape qualification witnesses required")
    for item in diagnostics:
        if (any(type(item.get(key)) not in {int, float} or not math.isfinite(item[key])
                for key in ("loss_sum", "selected_gradient_l2"))
                or item["loss_sum"] < 0 or item["selected_gradient_l2"] < 0
                or type(item.get("input_tokens")) is not int or not 2 <= item["input_tokens"] <= 512
                or type(item.get("supervised_tokens")) is not int
                or not 1 <= item["supervised_tokens"] < item["input_tokens"]
                or type(item.get("vocabulary_size")) is not int or item["vocabulary_size"] < 2
                or item.get("loss_hidden_gradient_parity") is not True
                or item.get("loss_gradient_rtol") != LOSS_GRAD_RTOL or item.get("loss_gradient_atol") != LOSS_GRAD_ATOL):
            raise ValueError("Nonfinite/incomplete objective loss/gradient/target witness")
        if item["objective"] == "whole" and item["supervised_tokens"] != item["input_tokens"] - 1:
            raise ValueError("Whole objective did not cover every next-token position")
        if item["scope"] == "maximum512_synthetic_backward" and (
                item["input_tokens"] != 512 or item["objective"] == "response" and item["supervised_tokens"] != 64):
            raise ValueError("Maximum training shape/masked response witness changed")
        if not body["fixture_only"] and item["vocabulary_size"] != 151936:
            raise ValueError("Native objective vocabulary inventory changed")
    cache = body.get("cache_check", {})
    if (cache.get("max_evaluation_tokens") != 2048 or cache.get("cache_parity") is not True
            or cache.get("greedy_token_equal") is not True or cache.get("forward_calls") != 3
            or cache.get("cache_forward_calls") != 2 or cache.get("processed_input_tokens") != 4096
            or cache.get("rtol") != PARITY_RTOL or cache.get("atol") != PARITY_ATOL
            or type(cache.get("maximum_logit_abs_error")) not in {int, float}
            or not math.isfinite(cache["maximum_logit_abs_error"]) or cache["maximum_logit_abs_error"] < 0):
        raise ValueError("Maximum2048 actual cache/forward parity witness missing")
    recovery = body.get("checkpoint_recovery", {})
    if (recovery.get("recovery_passed") is not True or recovery.get("sentinel_restore_verified") is not True
            or recovery.get("training_updates") != 0 or recovery.get("optimizer_step_calls") != 0
            or recovery.get("sentinel_scope") != "every_selected_parameter_and_both_moment_tensors"
            or recovery.get("optimizer_cursor_sentinel") != 17 or recovery.get("python_cpu_rng_sentinels") is not True
            or recovery.get("parameters_buffers_rng_cursor_and_zero_moments_restored") is not True
            or not isinstance(recovery.get("selected_official_state_sha256"), str)
            or len(recovery["selected_official_state_sha256"]) != 64):
        raise ValueError("Official0-update sentinel restoration witness missing")
    if not body["fixture_only"] and recovery.get("cuda_rng_sentinel") is not True:
        raise ValueError("Native recovery must challenge CUDA RNG restoration")
    path = Path(recovery["checkpoint"])
    if not body["fixture_only"]:
        path = external_output(path)
    physical = integrity.read_json(path / "manifest.json")
    if physical.get("sha256") != recovery.get("manifest_sha256"):
        raise ValueError("Official checkpoint manifest changed after qualification")
    meta = integrity.unseal(physical)
    entries = recovery.get("entries", [])
    if (not entries or meta.get("entries") != entries
            or len({entry.get("name") for entry in entries}) != len(entries)):
        raise ValueError("Official checkpoint tensor inventory witness missing")
    if recovery.get("parameter_sentinel_writes") != len(entries) or recovery.get("moment_sentinel_writes") != 2 * len(entries):
        raise ValueError("Official restoration did not challenge every selected parameter/moment tensor")
    from types import SimpleNamespace
    from .engine import validate_checkpoint
    descriptors = [(entry["name"], SimpleNamespace(shape=torch.Size(entry["shape"]))) for entry in entries]
    validated = validate_checkpoint(path, descriptors, binding)
    expected_progress = {"updates": 0, "next_group": 0, "consumed_rows": 0, "consumed_row_ids": [],
                         "processed_tokens": 0, "supervised_tokens": 0}
    if validated["progress"] != expected_progress:
        raise ValueError("Official qualification checkpoint cursor is not0")
    if not body["fixture_only"] and (len(entries) != 220 or sum(math.prod(entry["shape"]) for entry in entries) != TAIL_PARAMETERS):
        raise ValueError("Native official checkpoint is not the exact314M/220-tensor state")
    return body
