"""Exact local checkpoint binding and fail-closed, opt-in execution.

No model download, remote code, production default changes, or unattended sweeps.
"""
from __future__ import annotations
from contextlib import contextmanager
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from . import BASE_COMMIT, EXPERIMENT_ID, PAPER
from .data import load_rows
from .engine import Engine
from .plan import Arm, arms, checkpoint_manifest, digest_json, hash_file, validate_config, write_json
from .teacher import tensor_digest


@contextmanager
def fixed_precision():
    old_precision = torch.get_float32_matmul_precision()
    old_reduced = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(old_reduced)
        torch.set_float32_matmul_precision(old_precision)


def memory_check(c, during=False):
    if c["device"] == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested BF16 CUDA runtime is unavailable")
        key = "min_free_cuda_during_gib" if during else "min_free_cuda_before_gib"
        if torch.cuda.mem_get_info()[0] < c[key] * 1024**3:
            raise RuntimeError(f"Free GPU memory below {key}; no fallback or recipe change")


def preflight(c):
    validate_config(c)
    manifest = checkpoint_manifest(c["checkpoint"])
    if manifest["sha256"] != c["checkpoint_sha256"] or hash_file(c["dataset"]) != c["dataset_sha256"]:
        raise ValueError("Checkpoint or data no longer matches frozen hashes")
    rows = load_rows(Path(c["dataset"]), c["task"])
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "transformers": importlib.metadata.version("transformers"), "cuda_runtime": torch.version.cuda,
               "device": c["device"], "gpu": torch.cuda.get_device_name() if c["device"] == "cuda" and torch.cuda.is_available() else None,
               "attention": "forced_math_sdpa", "projection_operands": "bf16", "trainable_masters": "fp32",
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    code = {str(p.relative_to(Path(__file__).parent)): hash_file(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))}
    identity = {"experiment": EXPERIMENT_ID, "source": PAPER, "parent": BASE_COMMIT,
                "config_sha256": digest_json(c), "code_sha256": digest_json(code),
                "checkpoint_sha256": manifest["sha256"], "dataset_sha256": c["dataset_sha256"], "runtime": runtime}
    memory_check(c)
    return rows, identity


def bind_model(c):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(c["checkpoint"], local_files_only=True, trust_remote_code=False)
    if cfg.model_type != "qwen3" or getattr(cfg, "quantization_config", None):
        raise ValueError("Only non-quantized Qwen3-derived Rineya supported by this binding")
    if getattr(cfg, "use_sliding_window", False) or getattr(cfg, "rope_scaling", None):
        raise ValueError("Sliding-window/scaled-RoPE checkpoints require separate qualification")
    model, info = AutoModelForCausalLM.from_pretrained(
        c["checkpoint"], local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
    )
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Checkpoint does not load exactly; refusing partial/random parameter initialization")
    size = sum(p.numel() for p in model.parameters())
    if not 450_000_000 <= size <= 750_000_000:
        raise ValueError("This binding targets the exact 0.6B-class Rineya, not a fixture or larger model")
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("Gradient checkpointing requires separate graph/swap qualification")
    named = dict(model.named_parameters())
    if set(c["trainable_names"]) - named.keys():
        raise ValueError("Trainable parameter name not found in exact checkpoint")
    if sum(named[n].numel() for n in c["trainable_names"]) > c["max_trainable_parameters"]:
        raise ValueError("Trainable parameter budget exceeded")
    model.requires_grad_(False)
    for name in c["trainable_names"]:
        named[name].data = named[name].data.float()
        named[name].requires_grad_(True)
    model.config.use_cache = False
    model.to(c["device"]).eval()
    tokenizer = AutoTokenizer.from_pretrained(c["checkpoint"], local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Pinned tokenizer needs an explicit chat template")
    memory_check(c, during=True)
    return model, tokenizer


def verify_qualification(report, identity):
    if report.get("status") != "EXACT_CHECKPOINT_MECHANICS_QUALIFIED" or report.get("identity") != identity:
        raise ValueError("Native qualification missing, stale, failed or from a different runtime")
    if not report.get("backend_events") or not report.get("checks", {}).get("parameters_unchanged"):
        raise ValueError("Native qualification lacks dispatch or no-update evidence")
    if not report["checks"].get("teacher_initial_identity"):
        raise ValueError("Teacher did not start from student parameters")


def execute(c, output, command="qualify", arm=None, qualification=None, execute_training=False):
    if command not in {"qualify", "train"}:
        raise ValueError("Unknown native command")
    if command == "train" and not execute_training:
        raise PermissionError("Training disabled; explicitly pass --execute-training after qualification")
    if command == "train" and not isinstance(arm, Arm):
        raise ValueError("Select an explicit registered arm")
    rows, identity = preflight(c)
    if command == "train":
        verify_qualification(qualification or {}, identity)
    output = Path(output)
    if output.resolve().is_relative_to(Path(c["checkpoint"]).resolve()):
        raise ValueError("Do not write artifacts into the source checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "identity.json", identity)
    write_json(output / "config.json", c)
    model = engine = None
    try:
        torch.manual_seed(c["seed"])
        with fixed_precision():
            model, tokenizer = bind_model(c)
            engine = Engine(model, tokenizer, c)
            initial = tensor_digest(model.named_parameters())
            if c["device"] == "cuda":
                torch.cuda.reset_peak_memory_stats()
            if command == "qualify":
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    checks = engine.qualify(rows, arms((0.,)))
                events = sorted({e.key for e in prof.key_averages() if "scaled_dot_product" in e.key})
                if not any("attention_math" in event for event in events):
                    raise RuntimeError("Math SDPA not observed; do not infer dispatch from config alone")
                result = {"status": "EXACT_CHECKPOINT_MECHANICS_QUALIFIED", "checks": checks,
                          "backend_events": events, "training_updates": 0, "efficacy_established": False}
            else:
                def on_step(record):
                    write_json(output / "steps" / f'{record["step"]:04d}.json', record)
                result = engine.train(rows, arm, on_step, lambda: memory_check(c, during=True))
                result["status"] = "EXPLORATORY_NATIVE_RUN_COMPLETED"
                result["arm"] = vars(arm)
                # Save only the selected FP32 delta; never merge into source or production.
                delta = {n: p.detach().cpu() - engine.bank.initial[n] for n, p in engine.bank.selected.items()}
                with (output / "selected_delta.pt").open("xb") as f:
                    torch.save(delta, f)
                result["delta_sha256"] = hash_file(output / "selected_delta.pt")
            result.update(identity=identity, initial_parameters_sha256=initial,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if c["device"] == "cuda" else None,
                          native_checkpoint_executed=True, production_promotion_allowed=False)
            write_json(output / ("qualification.json" if command == "qualify" else "result.json"), result)
            return result
    except Exception as exc:
        write_json(output / "FAILED.json", {"status": "FAILED", "error_type": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        del engine, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
