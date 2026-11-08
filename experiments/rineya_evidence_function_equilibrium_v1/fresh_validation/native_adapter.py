"""Bound EFE stage-one native mechanics; native launch and qualification remain NOT_RUN.

Importing this module imports no Torch/Transformers, resolves no assets and touches no GPU.
The public load entry fails closed. The private CPU loader is preparation code requiring
same-run qualification and supervised orchestration before it can become a launch path.
CPU helper success is never a native receipt. No sibling-worktree imports are used.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from types import MappingProxyType

FROZEN_ARMS = MappingProxyType({"source": (0.0, 0.0), "T": (1.0, 0.0),
                               "H": (0.0, 1.0), "H025": (0.0, 0.25),
                               "independent": (1.0, 0.25)})
RAW_SUM_BUDGET = SHARED_SCALE_BUDGET = 0
QUALIFICATION_STATUS = "NOT_RUN"
SOURCE_ASSETS = MappingProxyType({
    "model.safetensors": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"})
ENDPOINT_SHA = MappingProxyType({
    "T": "07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d",
    "H": "a1103060591a91ee7d9ceb2ad7450621be352b087cd19188f2c383a0dbbc7b90"})
RUNTIME_PINS = MappingProxyType({"python": "3.11.15", "torch": "2.10.0+cu126",
    "transformers": "4.57.1", "safetensors": "0.7.0", "tokenizers": "0.22.2"})
EOS_IDS = (151645, 151643)
SVD_THREADS = 16
ARITHMETIC = "FP32 subtract; FP64 P16; aH*pH+aT*pT; FP64 source+update; FP32 once"
LAYER_SUFFIXES = ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight",
    "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight")
ENDPOINT_KEYS = frozenset(f"model.layers.{layer}.{suffix}"
                         for layer in range(8, 28) for suffix in LAYER_SUFFIXES)
SOURCE_KEYS = frozenset(f"model.layers.{layer}.{suffix}"
                       for layer in range(28) for suffix in LAYER_SUFFIXES) | {
                           "model.embed_tokens.weight", "model.norm.weight"}
REQUIRED_NATIVE_QUALIFICATION = (
    "exact_assets_and_inventory", "original_indep_reconstruction_data_parity",
    "tied_weights_and_untouched_parameters", "manual_vs_hf_greedy_tokens_and_eos",
    "cached_vs_full_prefix_logits", "maximum_registered_context_and_output_shape",
    "repeat_and_row_order_parity", "tokenizer_rendering_and_overflow_contract",
    "target_host_resource_envelope")


class Blocked(RuntimeError):
    """A failed or unresolved execution gate, never a quality outcome."""


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def runtime_identity() -> dict:
    versions = {"python": platform.python_version()}
    for package in RUNTIME_PINS:
        if package != "python":
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
    return {"versions": versions, "matches": versions == dict(RUNTIME_PINS),
            "qualification_status": QUALIFICATION_STATUS,
            "torch_imported_by_this_check": False}


def asset_binding(source_dir: Path, endpoint_t: Path, endpoint_h: Path) -> dict:
    """Hash binding only: failures stop before Torch imports or checkpoint/model loads.

    Schema 1 binds exact source/endpoint paths, byte hashes, runtime versions and frozen
    coefficient/rank/rounding semantics. It does not grant launch authority or qualification.
    """
    source_dir = Path(source_dir).resolve()
    endpoint_paths = {"T": Path(endpoint_t).resolve(), "H": Path(endpoint_h).resolve()}
    actual = {}
    for name, expected in SOURCE_ASSETS.items():
        path = source_dir / name
        if not path.is_file():
            raise Blocked(f"missing exact source asset: {path}")
        actual[name] = sha_file(path)
        if actual[name] != expected:
            raise Blocked(f"exact source SHA-256 mismatch: {name}")
    checkpoint_files = {p.name for p in source_dir.iterdir() if p.suffix.lower() in
                        {".bin", ".pt", ".pth", ".gguf", ".safetensors"}}
    if checkpoint_files != {"model.safetensors"}:
        raise Blocked("unexpected source weight inventory; replacement/export forbidden")
    endpoints = {}
    for skill, path in endpoint_paths.items():
        if not path.is_file() or sha_file(path) != ENDPOINT_SHA[skill]:
            raise Blocked(f"missing or changed exact FP32 {skill} endpoint: {path}")
        endpoints[skill] = {"path": str(path), "sha256": ENDPOINT_SHA[skill]}
    runtime = runtime_identity()
    if not runtime["matches"]:
        raise Blocked(f"runtime differs from registered .venv-mq001 pins: {runtime['versions']}")
    result = {"schema": 1, "experiment": "EFE-FUNC-001-FRESH-STAGE1",
        "source_dir": str(source_dir), "source_assets": actual, "endpoints": endpoints,
        "runtime": runtime["versions"], "arms": {k: list(v) for k, v in FROZEN_ARMS.items()},
        "rank": 16, "svd_cpu_threads": SVD_THREADS,
        "arithmetic": ARITHMETIC,
        "original_untouched_storage": "BF16; exact upcast in all-FP32 loaded function",
        "qualification_status": QUALIFICATION_STATUS, "native_launch_enabled": False}
    return {**result, "binding_sha256": digest(result)}


def validate_binding(binding: dict) -> None:
    body = {k: v for k, v in binding.items() if k != "binding_sha256"}
    if binding.get("binding_sha256") != digest(body):
        raise Blocked("asset binding digest mismatch")
    if (binding.get("schema") != 1 or binding.get("experiment") != "EFE-FUNC-001-FRESH-STAGE1"
            or binding.get("source_assets") != dict(SOURCE_ASSETS)
            or binding.get("runtime") != dict(RUNTIME_PINS)
            or binding.get("arms") != {k: list(v) for k, v in FROZEN_ARMS.items()}
            or binding.get("rank") != 16 or binding.get("svd_cpu_threads") != SVD_THREADS
            or binding.get("arithmetic") != ARITHMETIC
            or binding.get("original_untouched_storage") != "BF16; exact upcast in all-FP32 loaded function"
            or binding.get("qualification_status") != "NOT_RUN"
            or binding.get("native_launch_enabled") is not False):
        raise Blocked("asset binding differs from frozen experiment")
    if set(binding.get("endpoints", {})) != {"T", "H"}:
        raise Blocked("binding requires both original endpoints")
    for skill, expected in ENDPOINT_SHA.items():
        if binding["endpoints"][skill].get("sha256") != expected:
            raise Blocked(f"wrong {skill} endpoint binding")


def load_candidate(binding: dict, arm: str, *, protocol: dict, guard=None):
    """Fail-closed public entry. There is no native execution command in this delivery.

    A caller flag cannot unlock the unimplemented same-run qualification/orchestration
    gates. The current frozen protocol disables native launch. No assets are loaded here.
    """
    if arm not in FROZEN_ARMS:
        raise Blocked("arm outside frozen stage-one budget")
    if protocol.get("native_launch_enabled") is not True:
        raise Blocked("frozen protocol disables native launch; qualification NOT_RUN")
    raise Blocked("native launch remains unimplemented: same-run admissibility, scorer, "
                  "code, protocol and native qualification gates must be earned")


def p16(delta):
    """Historical deterministic full FP64 CPU SVD, rank 16; vector update kept whole."""
    import torch
    if delta.device.type != "cpu" or delta.ndim not in (1, 2):
        raise ValueError("P16 accepts CPU vectors/matrices only")
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("nonfinite delta")
    d = delta.double()
    if d.ndim == 1:
        return d
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    rank = min(16, s.numel())
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


def compose_tensor(source, endpoint_t, endpoint_h, arm: str):
    """Original INDEP arithmetic order, frozen coefficients; no checkpoint export.

    Stored BF16 source upcasts exactly. Subtraction happens before conversion to FP64;
    scaled projected components are summed before the single FP32 output rounding.
    """
    import torch
    if arm not in FROZEN_ARMS:
        raise ValueError("unregistered arm; coefficient sweeps forbidden")
    if source.dtype != torch.bfloat16 or source.device.type != "cpu":
        raise ValueError("original source must be BF16 on CPU")
    if source.ndim not in (1, 2) or not bool(torch.isfinite(source).all()):
        raise ValueError("source is not a finite vector/matrix")
    if arm == "source":
        return source.float()
    for endpoint in (endpoint_t, endpoint_h):
        if (endpoint is None or endpoint.dtype != torch.float32 or endpoint.shape != source.shape
                or endpoint.device.type != "cpu" or not bool(torch.isfinite(endpoint).all())):
            raise ValueError("both finite FP32 endpoints must match original source")
    at, ah = FROZEN_ARMS[arm]
    # Both endpoint inventories are audited independently; avoid an unnecessary SVD when zero.
    pt = p16(endpoint_t - source.float()) if at else 0.0
    ph = p16(endpoint_h - source.float()) if ah else 0.0
    update = ah * ph + at * pt
    result = (source.double() + update).float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite composition")
    return result


def check_inventory(source_keys, t_keys, h_keys) -> None:
    if set(source_keys) != SOURCE_KEYS:
        raise Blocked("source key inventory differs from exact 310-tensor Qwen3 anchor")
    if set(t_keys) != ENDPOINT_KEYS or set(h_keys) != ENDPOINT_KEYS:
        raise Blocked("both partial endpoints must contain all 220 registered last-20-layer keys")


def _load_cpu_candidate(binding: dict, arm: str, guard):
    """Unqualified private mechanics, deliberately not reachable from public launch.

    A future supervised qualifier may call this only after checking both locks, protocol
    and resource admission. The guard callback must enforce the registered envelope.
    """
    if arm not in FROZEN_ARMS:
        raise Blocked("arm outside frozen stage-one budget")
    if not callable(guard):
        raise Blocked("registered admission/resource callback required")
    guard("before_native_import")
    validate_binding(binding)
    rebound = asset_binding(Path(binding["source_dir"]), Path(binding["endpoints"]["T"]["path"]),
                            Path(binding["endpoints"]["H"]["path"]))
    if rebound != binding:
        raise Blocked("bound assets/runtime changed before loading")
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    torch.set_num_threads(SVD_THREADS)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    guard("before_cpu_model_load")
    source = Path(binding["source_dir"])
    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as s, \
            safe_open(binding["endpoints"]["T"]["path"], framework="pt", device="cpu") as t, \
            safe_open(binding["endpoints"]["H"]["path"], framework="pt", device="cpu") as h:
        check_inventory(s.keys(), t.keys(), h.keys())
        model = Qwen3ForCausalLM.from_pretrained(str(source), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
            attn_implementation="eager", low_cpu_mem_usage=True, device_map=None).eval()
        if sum(p.numel() for p in model.parameters()) != 596049920:
            raise Blocked("loaded parameter count differs from source")
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
            raise Blocked("tied vocabulary identity lost")
        with torch.no_grad():
            for key in sorted(SOURCE_KEYS):
                guard("checkpoint_tensor")
                base = s.get_tensor(key)
                parameter = model.get_parameter(key)
                if parameter.is_meta or parameter.dtype != torch.float32:
                    raise Blocked(f"unmaterialized or non-FP32 parameter: {key}")
                if key in ENDPOINT_KEYS:
                    et, eh = t.get_tensor(key), h.get_tensor(key)
                    for endpoint in (et, eh):
                        if (endpoint.dtype != torch.float32 or endpoint.shape != base.shape
                                or not bool(torch.isfinite(endpoint).all())):
                            raise Blocked(f"endpoint dtype/shape/finiteness mismatch: {key}")
                    value = compose_tensor(base, et, eh, arm)
                    del et, eh
                else:
                    value = compose_tensor(base, None, None, "source")
                if parameter.shape != value.shape:
                    raise Blocked(f"source/parameter shape mismatch: {key}")
                parameter.copy_(value)
                del base, value, parameter
        for buffer in model.buffers():
            if buffer.is_meta or not bool(torch.isfinite(buffer).all()):
                raise Blocked("unmaterialized or nonfinite native buffer")
        model.requires_grad_(False)
        model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True,
        trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id != EOS_IDS[0]:
        raise Blocked("tokenizer/EOS mismatch")
    guard("after_cpu_model_load")
    return model, tokenizer


def greedy_decode(model, prompt_ids, *, max_new_tokens: int, max_context: int,
                  guard, device="cpu") -> dict:
    """Unqualified batch-one full-KV greedy mechanics, with explicit EOS accounting.

    Guard before every forward. Overflow is an explicit failed row; input is never cut.
    CUDA use remains disabled until this adapter has a same-run native qualifier.
    """
    if device != "cpu":
        raise Blocked("CUDA entry disabled: target-host native qualification NOT_RUN")
    if (not callable(guard) or type(max_new_tokens) is not int or max_new_tokens <= 0
            or type(max_context) is not int or max_context <= 0):
        raise ValueError("positive context/output limits and registered guard required")
    if not prompt_ids or any(type(token) is not int or token < 0 for token in prompt_ids):
        raise ValueError("nonempty integer prompt IDs required")
    if len(prompt_ids) + max_new_tokens > max_context:
        return {"status": "skipped_context_overflow", "input_tokens": len(prompt_ids),
                "token_ids": [], "terminated": False, "stop_reason": "context_overflow",
                "forward_calls": 0, "qualification_status": QUALIFICATION_STATUS}
    import torch
    tokens, past, calls, ended = [], None, 0, False
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None]
    with torch.inference_mode():
        for step in range(max_new_tokens):
            guard("prefill" if step == 0 else "decode")
            out = model(input_ids=x, past_key_values=past, use_cache=True,
                        return_dict=True, logits_to_keep=1)
            calls += 1
            if not bool(torch.isfinite(out.logits).all()):
                raise Blocked("nonfinite native generation logits")
            token = int(out.logits[0, -1].argmax())
            tokens.append(token)
            if token in EOS_IDS:
                ended = True
                break
            past = out.past_key_values
            x = torch.tensor([[token]], dtype=torch.long, device=device)
            del out
    return {"status": "ok", "input_tokens": len(prompt_ids), "token_ids": tokens,
        "body_token_ids": tokens[:-1] if ended else list(tokens), "terminated": ended,
        "stop_reason": "eos" if ended else "max_new_tokens", "forward_calls": calls,
        "qualification_status": QUALIFICATION_STATUS}
