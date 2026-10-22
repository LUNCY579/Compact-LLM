"""Reversible, opt-in native activation-reconstruction screen, NOT fact editing.

A safe bridge: fit native MLP input/output pairs, intervene at a named MLP,
then measure held-out next-token drift. No claim that copying an MLP's outputs
adds facts. Desired factual target vectors require a separate reviewed study.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor, nn
from .artifacts import sha256
from .memory import BuildConfig, FrozenBilinear, build


class MemoryBlend(nn.Module):
    def __init__(self, original: nn.Module, memory: FrozenBilinear, alpha: float):
        super().__init__()
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("blend alpha must be in [0,1]")
        self.original, self.memory, self.alpha = original, memory, float(alpha)

    def forward(self, x: Tensor) -> Tensor:
        # EXACT bypass: do not even evaluate the memory at alpha=0.
        if self.alpha == 0:
            return self.original(x)
        replacement = self.memory(x)
        if replacement.shape != x.shape:
            raise ValueError("native MLP replacement must preserve hidden shape")
        if self.alpha == 1:
            return replacement
        baseline = self.original(x)
        if replacement.shape != baseline.shape:
            raise ValueError("replacement shape differs from native MLP")
        return baseline + self.alpha * (replacement - baseline)


@contextmanager
def intervene(model: nn.Module, module_path: str, memory: FrozenBilinear,
              alpha: float) -> Iterator[MemoryBlend]:
    """No permanent mutation. Scoped replacement restores the exact module object."""
    if not module_path or "." not in module_path or module_path.split(".")[-1] != "mlp":
        raise ValueError("an explicit dotted path ending in .mlp is required")
    parent_path, name = module_path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    original = parent.get_submodule(name)
    if isinstance(original, MemoryBlend):
        raise ValueError("nested intervention is rejected")
    parameters = list(original.parameters())
    if not parameters:
        raise ValueError("native MLP must expose floating-point parameters")
    p = parameters[0]
    if not p.is_floating_point() or any(t.device != p.device for t in parameters):
        raise ValueError("quantized/multi-device MLP is not supported")
    copied = FrozenBilinear(memory.a, memory.g, memory.b).to(device=p.device, dtype=p.dtype)
    wrapper = MemoryBlend(original, copied, alpha)
    parent._modules[name] = wrapper
    try:
        yield wrapper
    finally:
        parent._modules[name] = original


def fingerprint(model: nn.Module) -> str:
    """Content identity of parameters AND buffers, including nonpersistent buffers."""
    h = hashlib.sha256()
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        value = t.detach().cpu().contiguous()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("nonfinite native tensor")
        h.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def validate_rows(rows: list[dict], vocab: int, max_tokens: int = 256,
                  max_rows: int = 128) -> tuple[list[dict], list[dict]]:
    if not isinstance(rows, list) or not 4 <= len(rows) <= max_rows:
        raise ValueError("screen requires 4..max_rows explicit tokenized records")
    ids, prompts = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "split", "input_ids"}:
            raise ValueError("each row must have exactly id, split, input_ids")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("row ids must be unique nonempty strings")
        tokens = row["input_ids"]
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= max_tokens:
            raise ValueError("invalid token length; truncation is not permitted")
        if any(type(t) is not int or not 0 <= t < vocab for t in tokens):
            raise ValueError("token outside vocabulary")
        if tuple(tokens) in prompts or row["split"] not in {"train", "check"}:
            raise ValueError("duplicate prompts or invalid split")
        ids.add(row["id"])
        prompts.add(tuple(tokens))
    train = [r for r in rows if r["split"] == "train"]
    check = [r for r in rows if r["split"] == "check"]
    if len(train) < 2 or len(check) < 2:
        raise ValueError("at least two train and two check rows required")
    return train, check


@torch.no_grad()
def capture(model: nn.Module, module_path: str, rows: list[dict]) -> tuple[Tensor, Tensor]:
    captured: list[tuple[Tensor, Tensor]] = []
    module = model.get_submodule(module_path)

    def hook(_module, args, output):
        if len(args) != 1 or not isinstance(output, Tensor) or args[0].shape != output.shape:
            raise ValueError("requires a same-width single-input native MLP")
        captured.append((args[0][0, -1].detach().cpu().double().clone(),
                         output[0, -1].detach().cpu().double().clone()))

    handle = module.register_forward_hook(hook)
    try:
        device = next(model.parameters()).device
        for row in rows:
            before = len(captured)
            model(input_ids=torch.tensor([row["input_ids"]], device=device), use_cache=False)
            if len(captured) != before + 1:
                raise ValueError("MLP must execute exactly once per model forward")
    finally:
        handle.remove()
    return torch.stack([k for k, _ in captured]), torch.stack([v for _, v in captured])


@torch.no_grad()
def predictions(model: nn.Module, rows: list[dict]) -> Tensor:
    device = next(model.parameters()).device
    result = []
    for row in rows:
        logits = model(input_ids=torch.tensor([row["input_ids"]], device=device), use_cache=False).logits
        result.append(logits[0, -1].detach().cpu().double())
    out = torch.stack(result)
    if not torch.isfinite(out).all():
        raise ValueError("nonfinite native logits")
    return out


@torch.no_grad()
def cache_parity(model: nn.Module, rows: list[dict], atol: float, rtol: float) -> dict:
    device = next(model.parameters()).device
    maximum, equal = 0.0, True
    for row in rows:
        tokens = torch.tensor([row["input_ids"]], device=device)
        first = model(input_ids=tokens, use_cache=True)
        if getattr(first, "past_key_values", None) is None:
            raise ValueError("native model returned no cache")
        token = first.logits[:, -1:].argmax(-1)
        cached = model(input_ids=token, past_key_values=first.past_key_values, use_cache=True).logits[:, -1]
        full = model(input_ids=torch.cat([tokens, token], 1), use_cache=False).logits[:, -1]
        if not torch.isfinite(cached).all() or not torch.isfinite(full).all():
            raise ValueError("nonfinite cache parity outputs")
        maximum = max(maximum, float((cached.double()-full.double()).abs().max()))
        equal = equal and torch.allclose(cached, full, atol=atol, rtol=rtol)
    return {"pass": bool(equal), "maximum_absolute_logit_difference": maximum,
            "atol": atol, "rtol": rtol, "probe": "one greedy continuation per check prompt"}


def screen(model: nn.Module, rows: list[dict], module_path: str, config: BuildConfig,
           alpha: float, max_kl: float = .01, min_agreement: float = .99,
           cache_atol: float = 1e-4, cache_rtol: float = 1e-4) -> tuple[FrozenBilinear, dict]:
    """Caller-provided model; no download or implicit training. No efficacy promotion."""
    if any(m.training for m in model.modules()):
        raise ValueError("screen requires eval mode on every module")
    if not 0 < alpha <= 1 or not math.isfinite(alpha):
        raise ValueError("nonzero finite alpha is required for the intervention screen")
    if not 0 <= min_agreement <= 1 or not math.isfinite(max_kl) or max_kl < 0:
        raise ValueError("invalid retention screen thresholds")
    train, check = validate_rows(rows, model.config.vocab_size)
    before = fingerprint(model)
    k, y = capture(model, module_path, train)
    ck, cy = capture(model, module_path, check)
    memory, construction = build(k, y, config)
    baseline = predictions(model, check)
    with intervene(model, module_path, memory, 0):
        bypass = predictions(model, check)
    zero_parity = torch.equal(baseline, bypass)
    baseline_cache = cache_parity(model, check, cache_atol, cache_rtol)
    with intervene(model, module_path, memory, alpha):
        candidate = predictions(model, check)
        candidate_cache = cache_parity(model, check, cache_atol, cache_rtol)
    restored = predictions(model, check)
    after = fingerprint(model)
    logp = baseline.log_softmax(-1)
    logq = candidate.log_softmax(-1)
    kl = (logp.exp()*(logp-logq)).sum(-1).clamp_min(0)
    agreement = float((baseline.argmax(-1) == candidate.argmax(-1)).double().mean())
    unchanged = before == after and torch.equal(restored, baseline)
    finite = bool(torch.isfinite(kl).all())
    passed = (zero_parity and unchanged and finite and baseline_cache["pass"] and candidate_cache["pass"]
              and float(kl.max()) <= max_kl and agreement >= min_agreement)
    receipt = {"kind": "NATIVE_ACTIVATION_RECONSTRUCTION_SCREEN_NOT_FACT_EFFICACY",
               "native_forward_executed": True, "native_qualified": False,
               "screen_pass": passed, "module_path": module_path, "alpha": alpha,
               "construction": construction, "train_ids": [r["id"] for r in train],
               "check_ids": [r["id"] for r in check], "heldout_activation_mse": float((memory(ck)-cy).square().mean()),
               "next_token_kl_mean": float(kl.mean()), "next_token_kl_max": float(kl.max()),
               "next_token_top1_agreement": agreement, "zero_alpha_exact_parity": zero_parity,
               "restoration_verified": unchanged, "weight_buffer_hash_before": before,
               "weight_buffer_hash_after": after, "baseline_cache": baseline_cache,
               "candidate_cache": candidate_cache, "thresholds": {"max_kl": max_kl, "min_agreement": min_agreement},
               "limitations": ["No new factual targets or language fact edits", "Next-token drift is not broad retention",
                               "One-token cache parity is not long-generation qualification", "Check rows must never tune alpha/width/ridge"]}
    return memory, receipt


def verify_checkpoint(path: Path, manifest: dict) -> dict:
    if path.is_symlink() or not path.is_dir() or not manifest:
        raise ValueError("explicit local checkpoint and nonempty hash manifest required")
    files = {}
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ValueError("symlinks are not allowed in the checkpoint")
        if entry.is_file():
            if entry.suffix in {".py", ".bin", ".pt", ".pth", ".pkl"}:
                raise ValueError("code/pickle checkpoint entries are rejected")
            files[entry.relative_to(path).as_posix()] = sha256(entry)
    if files != manifest or "config.json" not in files or not any(p.endswith(".safetensors") for p in files):
        raise ValueError("checkpoint file inventory/hash mismatch")
    return files


def load_qwen_local(path: Path, manifest: dict, device: str, dtype: str) -> nn.Module:
    """Opt-in CLI loader; use a separately prepared isolated Transformers 4.51.3 env."""
    import importlib.metadata
    import psutil
    verify_checkpoint(path, manifest)
    if device not in {"cpu", "cuda:0"} or dtype not in {"float32", "bfloat16"}:
        raise ValueError("unsupported device/dtype")
    if device == "cpu" and dtype != "float32":
        raise ValueError("CPU native screen requires float32")
    if psutil.virtual_memory().available < 3 * 1024**3:
        raise MemoryError("less than 3 GiB available host RAM")
    if device == "cuda:0":
        if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 3 * 1024**3:
            raise MemoryError("CUDA:0 unavailable or less than 3 GiB free")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is not supported by this GPU")
    if importlib.metadata.version("transformers") != "4.51.3":
        raise RuntimeError("native loader requires isolated transformers==4.51.3; do not upgrade a pinned environment")
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3" or config.hidden_size > 1024 or config.num_hidden_layers > 32:
        raise ValueError("this loader is scoped to compact stock-Qwen3-compatible Rineya")
    model, info = AutoModelForCausalLM.from_pretrained(
        str(path), config=config, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=getattr(torch, dtype), attn_implementation="eager",
        output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"non-exact checkpoint load: {info}")
    verify_checkpoint(path, manifest)
    return model.to(device).eval()
