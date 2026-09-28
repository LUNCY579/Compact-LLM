"""Single-model native forward path and independently replayable signal records.

Torch/Transformers are imported lazily. No trainer, optimizer, generate(), adapters,
accelerate dependency, device_map, downloads, gold labels, or text-dependent gating.
"""
from __future__ import annotations

import array
import hashlib
import importlib
import importlib.metadata as metadata
import math
import os
import platform
import sys
import time
from pathlib import Path

from .contract import (Blocked, CLOSED, COLLECTOR, EFFECTIVE, append_event, asset_manifest,
                       canonical, digest, finite, pins, read, require, safe_child, sha,
                       source_manifest, validate_recipe, write_new)


def package_inventory() -> dict:
    return dict(sorted((d.metadata["Name"].lower().replace("_", "-"), d.version)
                       for d in metadata.distributions() if d.metadata.get("Name")))


def runtime_profile(device: str, dtype: str, threads: int) -> dict:
    """Hash actual installed dependency bytes, not only reported version strings."""
    import torch
    require(device in ("cpu", "cuda:0"), "unsupported device")
    require(dtype in ("float32", "bfloat16"), "unsupported dtype")
    require(type(threads) is int and 1 <= threads <= 8, "invalid threads")
    require(device != "cpu" or dtype == "float32", "CPU requires FP32")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    cuda = None
    if device == "cuda:0":
        require(torch.cuda.is_available(), "CUDA is not available; no fallback")
        prop = torch.cuda.get_device_properties(0)
        require(dtype != "bfloat16" or torch.cuda.is_bf16_supported(), "native BF16 unsupported")
        cuda = {"name": prop.name, "total_memory": prop.total_memory,
                "capability": list(torch.cuda.get_device_capability(0)),
                "uuid": str(getattr(prop, "uuid", "unavailable"))}
    names = ("torch", "transformers", "tokenizers", "safetensors", "numpy", "psutil",
             "jinja2", "huggingface-hub", "regex", "packaging")
    packages, dependencies, loaded = {}, {}, {}
    for name in names:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise Blocked(f"required isolated-runtime package missing: {name}") from exc
        packages[name] = dist.version
        module_name = name.replace("-", "_")
        module = importlib.import_module(module_name)
        imported = Path(module.__file__).resolve()
        expected_import = Path(dist.locate_file(module_name + "/__init__.py")).resolve()
        require(imported == expected_import, f"shadowed dependency import: {name}")
        loaded[name] = {"imported_file": str(imported), "sha256": sha(imported)}
        records = {}
        for entry in sorted(dist.files or [], key=str):
            # Installed scripts outside site-packages are not imported by this path.
            if ".." in entry.parts or entry.suffix == ".pyc" or "__pycache__" in entry.parts:
                continue
            file = Path(dist.locate_file(entry))
            require(file.is_file(), f"dependency file missing: {name}/{entry}")
            records[str(entry)] = sha(file)
        require(bool(records), f"dependency inventory unavailable: {name}")
        dependencies[name] = {"files": len(records), "actual_files_sha256": digest(records)}
    require(packages["transformers"] == "4.51.3", "do not upgrade the pinned host; qualify isolated Transformers 4.51.3")
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(), "machine": platform.machine(), "node": platform.node(),
            "packages": packages, "installed_packages": package_inventory(),
            "dependency_content": dependencies, "loaded_modules": loaded, "torch_build": torch.__config__.show(),
            "torch_version_loaded": str(torch.__version__), "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cuda_device": cuda,
            "compute_dtype": dtype, "device": device, "threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "environment": {k: os.environ.get(k) for k in
                ("CUBLAS_WORKSPACE_CONFIG", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                 "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def make_recipe(device: str, dtype: str, threads: int) -> dict:
    runtime = runtime_profile(device, dtype, threads)
    result = {"schema": 1, "collector": COLLECTOR, "device": device,
              "compute_dtype": dtype, "threads": threads, "runtime": runtime,
              "runtime_sha256": digest(runtime), "source_sha256": source_manifest(),
              "effective": EFFECTIVE, "official_eos_ids": pins()["official_eos_ids"],
              "limits": {"per_capture_seconds": 900, "screen_seconds": 3600,
                         "generation_max": 400, "min_free_host_gib": 3.0,
                         "max_process_tree_rss_gib": 8.0,
                         "min_free_cuda_gib_before_load": 3.0}, **CLOSED}
    validate_recipe(result)
    return result


def unknown_token(tokenizer, vocabulary: int) -> int:
    ids = tokenizer.encode("UNKNOWN", add_special_tokens=False)
    require(isinstance(ids, list) and len(ids) == 1 and type(ids[0]) is int and
            0 <= ids[0] < vocabulary, "UNKNOWN is not one valid token")
    require(tokenizer.decode(ids, skip_special_tokens=False,
                             clean_up_tokenization_spaces=False) == "UNKNOWN", "UNKNOWN exact decode failed")
    require(ids[0] not in tokenizer.all_special_ids, "UNKNOWN cannot be a special token")
    return ids[0]


def render_input(tokenizer, request: dict, vocabulary: int) -> tuple[str, list[int]]:
    require(isinstance(request, dict) and set(request) == {"id", "messages"}, "prompt-only input required")
    require(isinstance(request["id"], str) and bool(request["id"]), "request ID required")
    messages = request["messages"]
    require(isinstance(messages, list) and len(messages) == 2, "exact system/user messages required")
    for message, role in zip(messages, ("system", "user")):
        require(isinstance(message, dict) and set(message) == {"role", "content"} and
                message["role"] == role and isinstance(message["content"], str), "invalid prompt schema")
    options = {"add_generation_prompt": True, "enable_thinking": False}
    text = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    require(isinstance(text, str) and bool(text), "chat template unavailable")
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    direct = tokenizer.apply_chat_template(messages, tokenize=True, **options)
    require(ids == direct, "render/tokenize route parity failed")
    require(isinstance(ids, list) and all(type(i) is int and 0 <= i < vocabulary for i in ids), "invalid input IDs")
    require(0 < len(ids) <= 4072, "input overflow; truncation forbidden")
    return text, ids


def signal_from_logits(logits, unknown_id: int, vocabulary: int) -> tuple[dict, bytes]:
    """Full raw vocabulary -> FP64 log_softmax; no top-k or token/text heuristics."""
    import torch
    require(isinstance(logits, torch.Tensor) and logits.ndim == 1 and logits.numel() == vocabulary,
            "full vocabulary logits required")
    require(type(unknown_id) is int and 0 <= unknown_id < vocabulary, "invalid UNKNOWN ID")
    require(bool(torch.isfinite(logits).all()), "nonfinite raw logits")
    raw = logits.detach().to(device="cpu", dtype=torch.float64).contiguous()
    lp = float(torch.log_softmax(raw, dim=-1)[unknown_id].item())
    norm = float(torch.logsumexp(raw, dim=-1).item())
    values = array.array("d", raw.tolist())
    if sys.byteorder != "little":
        values.byteswap()
    blob = values.tobytes()
    result = {"encoding": "float64-le", "vocabulary_size": vocabulary,
              "logits_sha256": hashlib.sha256(blob).hexdigest(), "unknown_token_id": unknown_id,
              "unknown_logit": float(raw[unknown_id].item()), "log_normalizer": norm,
              "unknown_log_probability": lp, "unknown_probability": math.exp(lp),
              "first_argmax_id": int(raw.argmax().item())}
    require(finite(lp) and lp <= 0, "invalid UNKNOWN log probability")
    return result, blob


def validate_signal(signal: dict, root: Path, unknown_id: int, vocabulary: int) -> None:
    """Independent stdlib replay from the complete retained logits vector."""
    require(signal.get("encoding") == "float64-le" and signal.get("vocabulary_size") == vocabulary and
            signal.get("unknown_token_id") == unknown_id, "signal identity mismatch")
    path = safe_child(root, signal.get("path", ""))
    require(path.stat().st_size == 8 * vocabulary and sha(path) == signal.get("logits_sha256"), "signal bytes changed")
    values = array.array("d")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    require(all(math.isfinite(x) for x in values), "nonfinite retained logits")
    maximum = max(values)
    # Subtract maximum before summation; more accurate than separately subtracting a huge log-normalizer.
    log_sum = math.log(math.fsum(math.exp(x - maximum) for x in values))
    expected_lp = (values[unknown_id] - maximum) - log_sum
    require(signal.get("first_argmax_id") == values.index(maximum), "first greedy token mismatch")
    require(signal.get("unknown_logit") == values[unknown_id], "UNKNOWN raw logit changed")
    for key, wanted in (("log_normalizer", maximum + log_sum),
                        ("unknown_log_probability", expected_lp),
                        ("unknown_probability", math.exp(expected_lp))):
        value = signal.get(key)
        # Independent FP64 reduction can differ by ~1.5e-12 for a 151k
        # vector in the scripted stress case. This 1e-10 audit tolerance is
        # NEVER used by calibration or gating; retained p = exp(raw lp).
        require(finite(value) and math.isclose(value, wanted, rel_tol=1e-12, abs_tol=1e-10), f"incorrect {key}")
    require(signal["unknown_log_probability"] <= 0 and 0 <= signal["unknown_probability"] <= 1, "invalid probability range")
    require(signal["unknown_probability"] == math.exp(signal["unknown_log_probability"]), "probability must be exp of retained log probability")


def tensor_identity(model, model_dir: Path | None = None) -> dict:
    """Stream tensor verification; never instantiate a second model or optimizer."""
    import torch
    from safetensors import safe_open
    state = model.state_dict()
    records = {}
    handle = safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") if model_dir else None
    if handle:
        handle.__enter__()
    try:
        keys = set(handle.keys()) if handle else set()
        require(not handle or keys <= set(state), "unexpected checkpoint tensors")
        absent = set(state) - keys if handle else set()
        if absent:
            require(absent == {"lm_head.weight"} and model.config.tie_word_embeddings and
                    model.lm_head.weight is model.model.embed_tokens.weight, "unverified missing/tied tensors")
        for name, tensor in sorted(state.items()):
            require(tensor.device.type != "meta" and bool(torch.isfinite(tensor).all()), f"invalid loaded tensor: {name}")
            if handle:
                source_name = "model.embed_tokens.weight" if name in absent else name
                raw = handle.get_tensor(source_name)
                require(raw.dtype == torch.bfloat16, "published storage dtype is not BF16")
                require(tuple(raw.shape) == tuple(tensor.shape) and torch.equal(raw.to(tensor.device, tensor.dtype), tensor),
                        f"loaded tensor mismatch: {name}")
                del raw
            raw_bytes = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            records[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                             "sha256": hashlib.sha256(raw_bytes).hexdigest()}
            del raw_bytes
    finally:
        if handle:
            handle.__exit__(None, None, None)
    buffers = {}
    for name, tensor in sorted(model.named_buffers()):
        require(tensor.device.type != "meta" and bool(torch.isfinite(tensor).all()), f"invalid native buffer: {name}")
        raw_bytes = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        buffers[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                         "sha256": hashlib.sha256(raw_bytes).hexdigest()}
    return {"tensors": len(records), "tensor_manifest_sha256": digest(records),
            "buffers": len(buffers), "buffer_manifest_sha256": digest(buffers)}


class NativeEngine:
    """Production-only stock HF Qwen3 loader. Test doubles are not CLI-selectable."""
    def __init__(self, model_dir: Path, recipe: dict):
        import torch
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        validate_recipe(recipe)
        actual = runtime_profile(recipe["device"], recipe["compute_dtype"], recipe["threads"])
        require(actual == recipe["runtime"], "runtime identity differs from frozen recipe")
        self.assets = asset_manifest(model_dir)
        if recipe["device"] == "cuda:0":
            free, _ = torch.cuda.mem_get_info(0)
            require(free >= 3 * 1024**3, "insufficient free device memory")
        self.recipe, self.model_dir = recipe, model_dir
        self.device = recipe["device"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
        require(type(self.tokenizer).__name__ == "Qwen2TokenizerFast", "unexpected tokenizer class")
        template = read(model_dir / "tokenizer_config.json").get("chat_template")
        require(isinstance(template, str) and self.tokenizer.chat_template == template, "unregistered chat template")
        self.model, loading = Qwen3ForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, recipe["compute_dtype"]), attn_implementation="eager",
            low_cpu_mem_usage=False, use_safetensors=True, output_loading_info=True)
        require(type(self.model) is Qwen3ForCausalLM, "unexpected forward class")
        require(not loading.get("unexpected_keys") and not loading.get("mismatched_keys") and
                not loading.get("error_msgs"), "non-exact native load")
        require(set(loading.get("missing_keys", [])) <= {"lm_head.weight"}, "missing loaded tensors")
        self.model.to(self.device).eval().requires_grad_(False)
        require(self.model.config._attn_implementation == "eager", "attention implementation drift")
        self.vocabulary = self.model.config.vocab_size
        self.unknown = unknown_token(self.tokenizer, self.vocabulary)
        self.eos = pins()["official_eos_ids"]
        self.tensor_receipt = tensor_identity(self.model, model_dir)
        require(asset_manifest(model_dir) == self.assets, "assets changed during load")
        self.identity = {"weight_sha256": pins()["assets"]["model.safetensors"]["sha256"],
                         "runtime_sha256": recipe["runtime_sha256"], "recipe_sha256": digest(recipe),
                         "unknown_single_token": True, "unknown_token_id": self.unknown,
                         "unknown_encoded_ids": [self.unknown], "unknown_decoded": "UNKNOWN",
                         "eos_token_ids": self.eos, "vocabulary_size": self.vocabulary,
                         "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                         "assets": self.assets, "forward_class": type(self.model).__module__ + "." + type(self.model).__name__,
                         "loaded_tensors": self.tensor_receipt, "effective": EFFECTIVE,
                         "published_generation_config": read(model_dir / "generation_config.json")}

    def forward(self, ids, past):
        import torch
        require(not self.model.training and not torch.is_grad_enabled(), "inference-only mode required")
        require(all(not p.requires_grad and p.grad is None for p in self.model.parameters()), "trainable/gradient state detected")
        current = torch.tensor([ids], dtype=torch.long, device=self.device)
        output = self.model(input_ids=current, past_key_values=past, use_cache=True,
                            logits_to_keep=1, return_dict=True)
        require(output.past_key_values is not None, "missing native KV cache")
        return output.logits[0, -1], output.past_key_values

    def synchronize(self):
        if self.device == "cuda:0":
            import torch
            torch.cuda.synchronize(0)

    def verify_unchanged(self):
        require(tensor_identity(self.model) == self.tensor_receipt, "weights changed during inference")
        require(asset_manifest(self.model_dir) == self.assets, "assets changed during capture")
        require(runtime_profile(self.recipe["device"], self.recipe["compute_dtype"], self.recipe["threads"]) == self.recipe["runtime"],
                "runtime changed during capture")


def capture_one(engine, request: dict, root: Path, relative: str, confidence: bool) -> dict:
    """Shared greedy mechanics. Every request resets its cache; EOS counts as a step."""
    import torch
    require(type(confidence) is bool, "confidence flag must be boolean")
    text, ids = render_input(engine.tokenizer, request, engine.vocabulary)
    emitted, past, current, signal = [], None, ids, None
    engine.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for step in range(24):
            logits, past = engine.forward(current, past)
            require(logits.ndim == 1 and logits.numel() == engine.vocabulary and
                    bool(torch.isfinite(logits).all()), "invalid raw generation logits")
            token = int(logits.argmax().item())  # Never computed from gated/scaled scores.
            if step == 0 and confidence:
                signal, blob = signal_from_logits(logits, engine.unknown, engine.vocabulary)
                path = safe_child(root, relative + ".f64")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(blob)
                    handle.flush()
                    os.fsync(handle.fileno())
                signal["path"] = path.relative_to(root).as_posix()
                require(token == signal["first_argmax_id"], "probability capture mutated greedy decision")
            emitted.append(token)
            if token in engine.eos:
                break
            current = [token]
    del past
    engine.synchronize()
    terminated = bool(emitted and emitted[-1] in engine.eos)
    content = emitted[:-1] if terminated else emitted
    row = {"id": request["id"], "prompt_sha256": digest(request), "rendered_prompt": text,
           "input_ids": ids, "input_tokens": len(ids), "generated_ids": emitted,
           "content_ids": content, "generation_steps": len(emitted), "generated_tokens": len(content),
           "prediction": engine.tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False),
           "decoded_with_eos": engine.tokenizer.decode(emitted, skip_special_tokens=False, clean_up_tokenization_spaces=False),
           "terminated": terminated, "eos_token_id": emitted[-1] if terminated else None,
           "status": "ok" if terminated else "unterminated", "latency_seconds": time.perf_counter() - started,
           "confidence_enabled": confidence, "signal": signal,
           "unknown_probability": signal["unknown_probability"] if signal else None,
           "unknown_log_probability": signal["unknown_log_probability"] if signal else None}
    return row


def validate_row(row: dict, request: dict, identity: dict, root: Path, confidence: bool) -> None:
    require(isinstance(row, dict) and row.get("id") == request["id"] and row.get("prompt_sha256") == digest(request), "capture prompt/ID mismatch")
    vocab, eos = identity["vocabulary_size"], identity["eos_token_ids"]
    require(type(vocab) is int and vocab > max(pins()["official_eos_ids"]) and
            type(identity.get("unknown_token_id")) is int and 0 <= identity["unknown_token_id"] < vocab,
            "invalid native vocabulary/UNKNOWN identity")
    require(eos == pins()["official_eos_ids"], "full official EOS required")
    require(row.get("status") == "ok" and row.get("terminated") is True, "failed or unterminated generation")
    inputs, generated = row.get("input_ids"), row.get("generated_ids")
    for items in (inputs, generated):
        require(isinstance(items, list) and all(type(i) is int and 0 <= i < vocab for i in items), "invalid token IDs")
    require(type(row.get("input_tokens")) is int and 0 < len(inputs) <= 4072 and row.get("input_tokens") == len(inputs), "input count/context mismatch")
    require(1 <= len(generated) <= 24 and generated[-1] in eos and
            not any(token in eos for token in generated[:-1]), "EOS absent, early, or post-EOS generation")
    require(type(row.get("generation_steps")) is int and type(row.get("generated_tokens")) is int and
            row.get("generation_steps") == len(generated) and row.get("generated_tokens") == len(generated) - 1 and
            row.get("content_ids") == generated[:-1] and row.get("eos_token_id") == generated[-1], "EOS/token accounting mismatch")
    require(all(isinstance(row.get(k), str) for k in ("prediction", "decoded_with_eos", "rendered_prompt")), "decoded text missing")
    require(finite(row.get("latency_seconds")) and row["latency_seconds"] >= 0, "invalid latency")
    require(row.get("confidence_enabled") is confidence, "confidence mode mismatch")
    if confidence:
        signal = row.get("signal")
        require(isinstance(signal, dict), "full first-step signal missing")
        validate_signal(signal, root, identity["unknown_token_id"], vocab)
        require(finite(row.get("unknown_probability")) and finite(row.get("unknown_log_probability")) and
                generated[0] == signal["first_argmax_id"] and
                row.get("unknown_probability") == signal["unknown_probability"] and
                row.get("unknown_log_probability") == signal["unknown_log_probability"], "signal/row mismatch")
    else:
        require(row.get("signal") is None and row.get("unknown_probability") is None and
                row.get("unknown_log_probability") is None, "confidence-off must not manufacture probability")


def parity(on: list[dict], off: list[dict], repeat: list[dict]) -> None:
    require(len(on) == len(off) == len(repeat) == 8, "all eight qualification rows required")
    keys = ("id", "prompt_sha256", "rendered_prompt", "input_ids", "input_tokens", "generated_ids",
            "content_ids", "generation_steps", "generated_tokens", "prediction", "decoded_with_eos",
            "terminated", "eos_token_id", "status")
    for a, b, c in zip(on, off, repeat):
        require(all(a[k] == b[k] == c[k] for k in keys), "confidence on/off/repeat parity failed")
        require(a["signal"]["logits_sha256"] == c["signal"]["logits_sha256"] and
                a["unknown_probability"] == c["unknown_probability"] and
                a["unknown_log_probability"] == c["unknown_log_probability"], "repeat signal differs")
