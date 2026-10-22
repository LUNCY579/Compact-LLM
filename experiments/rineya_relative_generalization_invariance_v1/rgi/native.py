"""Opt-in, local-only FP32 Qwen3 collection and bounded unmerged LoRA training.

Optional imports stay here. Synthetic tests never qualify native model efficacy.
"""
from __future__ import annotations
import math
import os
from pathlib import Path
import random
from .io import (artifact, digest, file_hash, load, require, score_rows, seal,
                 validate_bank, verify, write_new)

TOKEN_FILES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
               "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja"}


def fingerprint(path: str | Path, kind: str) -> str:
    root = Path(path).expanduser().resolve(strict=True)
    require(root.is_dir(), "Local model/tokenizer directory required; hub IDs are forbidden")
    require(kind in {"model", "tokenizer", "adapter"}, "Unknown fingerprint kind")
    files = []
    for p in sorted(root.iterdir()):
        selected = p.name in TOKEN_FILES if kind == "tokenizer" else (
            p.suffix == ".safetensors" or p.name in {"config.json", "model.safetensors.index.json",
                                                     "adapter_config.json"})
        if selected:
            require(p.is_file() and not p.is_symlink(), "Input file must be a regular local file")
            files.append({"name": p.name, "sha256": file_hash(p), "bytes": p.stat().st_size})
    names = {f["name"] for f in files}
    require(bool(files), "No fingerprintable inputs")
    if kind == "tokenizer":
        require("tokenizer.json" in names and "tokenizer_config.json" in names, "Fast tokenizer sidecars required")
    else:
        require(any(n.endswith(".safetensors") for n in names), "Safetensors weights required")
        require(("config.json" if kind == "model" else "adapter_config.json") in names, "Missing model configuration")
    return digest(files)


def inspect_spec(spec: dict) -> dict:
    require(spec.get("evidence_class") in {"native_model", "tiny_random_model"}, "Explicit native evidence class required")
    require(spec.get("precision") == "float32", "Sparse research artifacts require FP32; no silent casting/quantization")
    require(spec.get("device") == "cpu" or str(spec.get("device", "")).startswith("cuda:"), "Explicit CPU/CUDA device required")
    require(type(spec.get("max_length")) is int and 2 <= spec["max_length"] <= 8192, "Invalid context bound")
    require(type(spec.get("max_scored_tokens")) is int and 1 <= spec["max_scored_tokens"] <= 250000, "Invalid token cap")
    host = fingerprint(spec["model_path"], "model")
    tok = fingerprint(spec["tokenizer_path"], "tokenizer")
    require(host == spec["model_sha256"] and tok == spec["tokenizer_sha256"], "Native input fingerprint mismatch")
    config = load(Path(spec["model_path"]) / "config.json")
    require(config.get("model_type") == "qwen3", "Only dense Qwen3 is qualified by this loader contract")
    require(not config.get("quantization_config") and not config.get("auto_map"), "Quantized/custom remote-code models forbidden")
    components = {}
    for adapter in spec.get("adapters", []):
        sha = fingerprint(adapter["path"], "adapter")
        coef = adapter["coefficient"]
        require(sha == adapter["sha256"] and math.isfinite(coef) and coef != 0, "Invalid registered adapter")
        require(sha not in components, "Duplicate adapter input")
        cfg = load(Path(adapter["path"]) / "adapter_config.json")
        require(cfg.get("peft_type") == "LORA" and not cfg.get("use_dora", False)
                and not cfg.get("modules_to_save") and cfg.get("bias", "none") == "none",
                "Only ordinary no-bias LoRA without saved host modules is supported")
        components[sha] = float(coef)
    return {"host_sha256": host, "tokenizer_sha256": tok, "components": components}


def resource_guard(spec: dict) -> None:
    import psutil
    ram = psutil.virtual_memory().available / 2**30
    floor = float(spec.get("min_free_ram_gib", 8))
    require(math.isfinite(floor) and floor >= 0, "Invalid RAM guard")
    if spec["evidence_class"] == "native_model":
        require(floor >= 4, "Native RAM guard may not be disabled")
    require(ram >= floor, f"Host memory gate: {ram:.2f} GiB free < {floor:.2f}")
    if spec["device"].startswith("cuda:"):
        import torch
        require(torch.cuda.is_available(), "CUDA unavailable; no implicit device fallback")
        floor = float(spec.get("min_free_vram_gib", 7))
        require(math.isfinite(floor) and floor >= 0, "Invalid CUDA guard")
        if spec["evidence_class"] == "native_model":
            require(floor >= 4, "Native VRAM guard may not be disabled")
        free, _ = torch.cuda.mem_get_info(spec["device"])
        require(free / 2**30 >= floor, "Insufficient free CUDA memory")


def check_bank_spec(bank: dict, spec: dict) -> None:
    validate_bank(bank)
    require(bank["tokenizer_sha256"] == spec["tokenizer_sha256"], "Tokenizer drift")
    require(bank["evidence_class"] == spec["evidence_class"], "Synthetic/native evidence boundary")
    require(max(len(x["input_ids"]) for x in bank["items"]) <= spec["max_length"], "Context cap exceeded")
    count = sum(sum(r != "prompt" for r in x["roles"]) for x in bank["items"])
    require(count <= spec["max_scored_tokens"], "Scored-token cap exceeded")


def local_tokenizer(spec: dict):
    require(fingerprint(spec["tokenizer_path"], "tokenizer") == spec["tokenizer_sha256"], "Tokenizer fingerprint mismatch")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec["tokenizer_path"], local_files_only=True,
                                         trust_remote_code=False, use_fast=True)


def load_model(spec: dict):
    receipt = inspect_spec(spec)
    resource_guard(spec)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, info = AutoModelForCausalLM.from_pretrained(spec["model_path"], local_files_only=True,
        trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True)
    require(not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            f"Native state-dict mismatch: {info}")
    count = sum(p.numel() for p in model.parameters())
    if spec["evidence_class"] == "tiny_random_model":
        require(count <= 5_000_000, "Tiny-model test mode cannot run a native Rineya core")
    require(all(p.dtype == torch.float32 for p in model.parameters()), "Non-FP32 model parameters")
    adapters = spec.get("adapters", [])
    if adapters:
        from peft import PeftModel
        names = [f"component_{i}" for i in range(len(adapters))]
        model = PeftModel.from_pretrained(model, adapters[0]["path"], adapter_name=names[0], is_trainable=False)
        for name, item in zip(names[1:], adapters[1:]):
            model.load_adapter(item["path"], adapter_name=name, is_trainable=False)
        # Concatenate low-rank factors for an exact weighted sum; NEVER dense-merge.
        model.add_weighted_adapter(names, [x["coefficient"] for x in adapters],
                                   adapter_name="rgi_combined", combination_type="cat")
        model.set_adapter("rgi_combined")
    model.to(spec["device"]).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, receipt


def answer_losses(model, item: dict, device: str, chunk_size: int = 16):
    """One transformer pass, answer-position LM-head chunks, correct next-token shift."""
    import torch
    import torch.nn.functional as F
    require(type(chunk_size) is int and chunk_size > 0, "Invalid projection chunk size")
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
    positions = torch.tensor([i for i, r in enumerate(item["roles"]) if r != "prompt"], device=device)
    require(len(positions) > 0 and int(positions.min()) > 0, "Unscorable target")
    hidden = core.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    losses = []
    for pos in positions.split(chunk_size):
        logits = core.lm_head(hidden[:, pos - 1, :]).float().reshape(len(pos), -1)
        losses.append(F.cross_entropy(logits, ids[0, pos], reduction="none"))
    return torch.cat(losses)


def collect(bank: dict, spec: dict) -> dict:
    check_bank_spec(bank, spec)
    require({x["split"] for x in bank["items"]} <= {"development", "confirmation"}, "Do not collect validation from training rows")
    model, receipt = load_model(spec)
    import torch
    import gc
    import importlib.metadata
    values = []
    execution = {"collector": "qwen3_local_fp32_v1", "device": spec["device"], "precision": "float32",
                 "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                 "peft": importlib.metadata.version("peft") if spec.get("adapters") else "not_used",
                 "attention": "eager", "tf32": False, "projection_chunk": 16}
    # PEFT version is recorded uniformly when installed, even for the adapter-off arm.
    try:
        execution["peft"] = importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        with torch.inference_mode():
            for item in bank["items"]:
                values.append(answer_losses(model, item, spec["device"]).cpu().tolist())
        return artifact(bank, score_rows(bank, values), digest(spec), receipt["host_sha256"], receipt["components"],
                        execution_contract=execution)
    finally:
        del model
        gc.collect()
        if spec["device"].startswith("cuda:"):
            torch.cuda.empty_cache()


def training_plan(bank: dict, spec: dict) -> dict:
    check_bank_spec(bank, spec)
    require(all(x["split"] == "train" for x in bank["items"]), "Training cannot consume evaluation rows")
    require(not spec.get("adapters"), "Train a fresh adapter on the frozen host, not on a hidden preloaded stack")
    t = spec["training"]
    require(t.get("optimizer") == "adamw", "Only AdamW is registered; Muon is not a LoRA drop-in")
    for name, limit in (("updates", 128), ("gradient_accumulation", 16), ("rank", 64)):
        require(type(t.get(name)) is int and 1 <= t[name] <= limit, f"Unbounded training {name}")
    for name in ("learning_rate", "alpha", "max_grad_norm"):
        require(isinstance(t.get(name), (int, float)) and math.isfinite(t[name]) and t[name] > 0, f"Invalid {name}")
    require(t["learning_rate"] <= 0.01, "Learning-rate guard")
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    require(isinstance(t.get("target_modules"), list) and bool(t["target_modules"])
            and set(t["target_modules"]) <= allowed, "Unregistered trainable modules")
    require(type(t.get("seed")) is int, "Explicit training seed required")
    rng, order = random.Random(t["seed"]), []
    count = t["updates"] * t["gradient_accumulation"]
    while len(order) < count:
        epoch = list(range(len(bank["items"])))
        rng.shuffle(epoch)
        order.extend(epoch)
    order = order[:count]
    return seal({"kind": "training_plan", "schema_version": 1, "spec_sha256": digest(spec),
                 "bank_sha256": bank["content_sha256"], "order": order,
                 "processed_tokens": sum(len(bank["items"][i]["input_ids"]) for i in order),
                 "supervised_tokens": sum(sum(r != "prompt" for r in bank["items"][i]["roles"]) for i in order),
                 "updates": t["updates"], "evidence_class": bank["evidence_class"], "native_execution": False})


def optimize(model, bank: dict, order: list[int], settings: dict, device: str) -> dict:
    """Bounded optimizer kernel, testable with an isolated tiny torch host."""
    import torch
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    require(params and all("lora_" in n for n, _ in params), "Optimizer may update only LoRA parameters")
    optimizer = torch.optim.AdamW([p for _, p in params], lr=settings["learning_rate"], weight_decay=0.0)
    accumulation = settings["gradient_accumulation"]
    require(len(order) == settings["updates"] * accumulation, "Training plan/order mismatch")
    model.train()
    trace, processed, supervised = [], 0, 0
    for start in range(0, len(order), accumulation):
        group = [bank["items"][i] for i in order[start:start + accumulation]]
        labels = sum(sum(r != "prompt" for r in x["roles"]) for x in group)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for item in group:
            losses = answer_losses(model, item, device)
            require(bool(torch.isfinite(losses).all()), "Non-finite training loss")
            (losses.sum() / labels).backward()
            loss_sum += float(losses.detach().sum())
            processed += len(item["input_ids"])
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], settings["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        supervised += labels
        trace.append({"step": len(trace) + 1, "mean_nll": loss_sum / labels, "gradient_norm": float(norm)})
    model.eval()
    return {"updates": len(trace), "processed_tokens": processed, "supervised_tokens": supervised, "trace": trace}


def safe_output_directory(out: str | Path, spec: dict) -> Path:
    out = Path(out).resolve()
    roots = [spec["model_path"], spec["tokenizer_path"]] + [a["path"] for a in spec.get("adapters", [])]
    require(all(out != Path(r).resolve() and Path(r).resolve() not in out.parents for r in roots),
            "Output cannot mutate a model, tokenizer or source adapter directory")
    out.mkdir(parents=True, exist_ok=False)
    return out


def train(bank: dict, spec: dict, plan: dict, out: str | Path, execute: bool = False) -> dict:
    require(execute, "Training requires explicit --execute after plan review")
    verify(plan)
    require(plan == training_plan(bank, spec), "Frozen training plan changed")
    model, receipt = load_model(spec)
    import torch
    from peft import LoraConfig, get_peft_model
    t = spec["training"]
    torch.manual_seed(t["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(t["seed"])
    config = LoraConfig(r=t["rank"], lora_alpha=t["alpha"], lora_dropout=0.0, bias="none",
                        target_modules=t["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    destination = safe_output_directory(out, spec)
    try:
        result = optimize(model, bank, plan["order"], t, spec["device"])
        require(all(result[k] == plan[k] for k in ("updates", "processed_tokens", "supervised_tokens")), "Dose mismatch")
        model.save_pretrained(destination / "adapter", safe_serialization=True)
        result.update(kind="training_receipt", schema_version=1, status="COMPLETED",
                      plan_sha256=plan["content_sha256"], host_sha256=receipt["host_sha256"],
                      adapter_sha256=fingerprint(destination / "adapter", "adapter"),
                      evidence_class=bank["evidence_class"], quality_improvement_established=False,
                      release_permitted=False)
        write_new(destination / "receipt.json", seal(result))
        return result
    except Exception as exc:
        write_new(destination / "failure.json", {"status": "FAILED", "error_type": type(exc).__name__,
                   "plan_sha256": plan["content_sha256"], "quality_improvement_established": False})
        raise


def control_plan_matrix(targeted: dict, control: dict, spec: dict, seeds: list[int]) -> dict:
    """Materialize matched independent training-seed plans; never start a job."""
    import copy
    validate_bank(targeted)
    validate_bank(control)
    require(seeds and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds), "Distinct integer seeds required")
    require(len(targeted["items"]) == len(control["items"]), "Control row count mismatch")
    for a, b in zip(targeted["items"], control["items"]):
        for field in ("family", "format_id", "horizon"):
            require(a[field] == b[field], "Control structural matching failed")
        require(len(a["input_ids"]) == len(b["input_ids"]), "Processed-token dose mismatch")
        require(sum(r != "prompt" for r in a["roles"]) == sum(r != "prompt" for r in b["roles"]), "Supervised-token dose mismatch")
    arms = {}
    for seed in seeds:
        current = copy.deepcopy(spec)
        current["training"]["seed"] = seed
        for name, bank in (("targeted", targeted), ("control", control)):
            arms[f"{name}_s{seed}"] = {"spec": copy.deepcopy(current), "plan": training_plan(bank, current)}
        ta, co = arms[f"targeted_s{seed}"]["plan"], arms[f"control_s{seed}"]["plan"]
        require(all(ta[k] == co[k] for k in ("order", "processed_tokens", "supervised_tokens", "updates")), "Paired plan drift")
    return {"kind": "matched_control_plan_matrix", "arms": arms, "training_executed": False,
            "interpretation": "Within-seed matched training dose; repeated evaluation parents are not independent seeds."}
