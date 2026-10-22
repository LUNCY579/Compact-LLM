"""Opt-in local-only FP32 Qwen3-0.6B execution. No downloading or dense merging.

The native paths are implemented but require local custody/host qualification.
Generic in-memory Rineya loaders can instead call moe.inject directly.
"""
from contextlib import nullcontext
from pathlib import Path
import importlib.util
import json
import threading
import time
import torch
from .artifacts import load_json, verify_tree, sha256_file, write_json, validate_rows
from .control import ControlConfig, digest
from .moe import MoEConfig, inject


def preflight(config):
    blockers = []
    if config.get("allow_native") is not True:
        blockers.append("allow_native must be explicitly true in the local run config")
    if config.get("dtype") != "float32":
        blockers.append("only float32 is admitted; no implicit downcast/quantization")
    device = str(config.get("device", "cpu"))
    if device != "cpu" and not device.startswith("cuda"):
        blockers.append("only a single CPU or CUDA device is supported")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            blockers.append("CUDA unavailable")
        else:
            minimum = float(config.get("min_free_gib", 7.0))
            if not 0 < minimum < 1000:
                blockers.append("invalid min_free_gib")
            else:
                for _ in range(2):
                    if torch.cuda.mem_get_info(torch.device(device))[0] / 2**30 < minimum:
                        blockers.append("insufficient free VRAM; do not start")
                        break
    for dependency in ("transformers", "safetensors"):
        if importlib.util.find_spec(dependency) is None:
            blockers.append(f"missing optional dependency: {dependency}")
    specifications = [("base", config.get("base", {}))] + list(config.get("adapters", {}).items())
    for name, spec in specifications:
        try:
            path = Path(spec["path"])
            if not str(spec["path"]).strip():
                raise ValueError("empty path")
            verify_tree(path, spec["manifest"])
            if name == "base":
                c = load_json(path / "config.json")
                if (c.get("model_type") != "qwen3" or c.get("quantization_config")
                        or not 1 <= c.get("hidden_size", 0) <= 1024
                        or not 1 <= c.get("num_hidden_layers", 0) <= 28
                        or not 1 <= c.get("intermediate_size", 0) <= 4096
                        or not 1 <= c.get("vocab_size", 0) <= 160000
                        or not list(path.glob("*.safetensors"))):
                    raise ValueError("requires a dense, small Qwen3 safetensors export")
            else:
                c = load_json(path / "adapter_config.json")
                if (c.get("peft_type") != "LORA" or c.get("bias", "none") != "none"
                        or c.get("modules_to_save") or c.get("use_dora")
                        or not (path / "adapter_model.safetensors").is_file()):
                    raise ValueError("only unmerged ordinary LoRA safetensors admitted")
        except (KeyError, ValueError, OSError, TypeError) as exc:
            blockers.append(f"{name}: {exc}")
    if config.get("adapters") and importlib.util.find_spec("peft") is None:
        blockers.append("missing optional dependency: peft")
    return {"ready": not blockers, "blockers": blockers,
            "native_executed": False, "config_sha256": digest(config),
            "warning": "Preflight is not model, memory-peak, or efficacy qualification."}


def load_base(config):
    report = preflight(config)
    if not report["ready"]:
        raise RuntimeError("native preflight blocked: " + "; ".join(report["blockers"]))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = str(Path(config["base"]["path"]).resolve())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation="eager")
    if sum(p.numel() for p in model.parameters()) > 700_000_000:
        raise ValueError("model exceeds the 0.6B-track parameter guard")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("loader failed FP32 contract")
    model.to(config.get("device", "cpu")).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return model, tokenizer


class LocalAdapterBackend:
    """Serial, whole-call adapter switching. Existing artifact strengths stay 1.0.

    Loading is reported separately from per-call latency. No past_key_values
    cross calls, and no merging, training, target-conditioned routing or sampling.
    """
    def __init__(self, config):
        start = time.perf_counter()
        self.config = config
        self.model, self.tokenizer = load_base(config)
        self.names = set(config.get("adapters", {})) | {"core"}
        if "core" in config.get("adapters", {}):
            raise ValueError("core is reserved for adapter-off")
        for number, (name, spec) in enumerate(config.get("adapters", {}).items()):
            if number == 0:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, spec["path"],
                                                       adapter_name=name, is_trainable=False,
                                                       local_files_only=True)
            else:
                self.model.load_adapter(spec["path"], adapter_name=name, is_trainable=False,
                                        local_files_only=True)
        self.model.eval().requires_grad_(False)
        self.lock = threading.Lock()
        self.load_seconds = time.perf_counter() - start

    def __call__(self, expert, strength, request):
        if expert not in self.names or strength != 1.0:
            raise ValueError("unknown expert or non-unit registered strength; no silent rescaling")
        if set(request) - {"prompt", "max_new_tokens"}:
            raise ValueError("backend accepts prompt and max_new_tokens only; no gold/goal metadata")
        if not isinstance(request.get("prompt"), str) or not request["prompt"]:
            raise ValueError("nonempty prompt required")
        limit = request.get("max_new_tokens", 128)
        if type(limit) is not int or not 1 <= limit <= 512:
            raise ValueError("max_new_tokens must be 1..512")
        with self.lock:
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            start = time.perf_counter()
            if expert != "core":
                self.model.set_adapter(expert)
                # Some PEFT versions make selected adapter weights trainable.
                self.model.requires_grad_(False)
            context = self.model.disable_adapter() if (expert == "core" and
                       self.config.get("adapters")) else nullcontext()
            inputs = self.tokenizer(request["prompt"], return_tensors="pt", add_special_tokens=False)
            length = inputs["input_ids"].shape[-1]
            if length + limit > min(4096, self.model.config.max_position_embeddings):
                raise ValueError("request exceeds pilot context budget; truncation is forbidden")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with context, torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                             use_cache=True,
                                             pad_token_id=self.tokenizer.eos_token_id)
            tokens = output[0, length:]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            return {"text": text, "elapsed_seconds": time.perf_counter() - start,
                    "input_tokens": length, "output_tokens": len(tokens),
                    "evidence_kind": "native_execution", "expert": expert,
                    "config_sha256": digest(self.config)}


def train_native(config, rows, output_dir, moe_config=MoEConfig(), control=ControlConfig(),
                 steps=20, lr=0.0001, seed=0):
    """Completion-only SFT of fresh LoRA banks; frozen base, explicit down_proj paths.

    No auto training sweep, sealed-set access, dense merge, or promotion. One
    process, no gradient checkpointing, no DDP, no optimizer-state resume.
    """
    validate_rows(rows)
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("native training requires only TRAIN rows")
    if config.get("adapters"):
        raise ValueError("Stage B cannot modify registered Stage A adapters")
    if type(steps) is not int or not 1 <= steps <= 10000 or not 0 < lr <= 0.01:
        raise ValueError("invalid bounded training budget")
    paths = config.get("projection_paths", [])
    if not paths or any(not p.startswith("model.layers.") or not p.endswith(".mlp.down_proj")
                        for p in paths):
        raise ValueError("register explicit Qwen3 MLP down_proj paths")
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("use a new output directory")
    torch.manual_seed(seed)
    model, tokenizer = load_base(config)
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("gradient checkpoint recomputation would double-count loads")
    bank = inject(model, paths, moe_config, control, digest(config))
    model.eval()
    for module in bank.values():
        module.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    order = torch.randperm(len(rows)).tolist()
    logs = []
    start = time.perf_counter()
    for step in range(steps):
        row = rows[order[step % len(order)]]
        prompt = tokenizer.encode(row["payload"]["prompt"], add_special_tokens=False)
        target = tokenizer.encode(row["target"], add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            target.append(tokenizer.eos_token_id)
        if not prompt or not target or len(prompt) + len(target) > 2048:
            raise ValueError("invalid/oversized example; no silent truncation")
        ids = torch.tensor([prompt + target], device=model.device)
        labels = ids.clone()
        labels[:, :len(prompt)] = -100
        optimizer.zero_grad(set_to_none=True)
        for module in bank.values():
            module.begin_step()
        try:
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           labels=labels, use_cache=False)
            auxiliary = sum(m.last_auxiliary for m in bank.values())
            loss = output.loss + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            controls = {p: m.end_step() for p, m in bank.items()}
        except BaseException:
            for module in bank.values():
                module.end_step(False)
            raise
        logs.append({"step": step, "id": row["id"], "loss": float(loss.detach()),
                     "completion_loss": float(output.loss.detach()), "tokens": ids.numel(),
                     "gradient_norm": float(norm), "control": controls})
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - start
    out.mkdir(parents=True)
    checkpoint = {"schema": 1, "kind": "native_pilot_unqualified", "config_sha256": digest(config),
                  "moe_config": vars(moe_config), "control_config": vars(control),
                  "projections": {p: m.adapter_state() for p, m in bank.items()}}
    torch.save(checkpoint, out / "adapter_bank.pt")
    report = {"schema": 1, "native_executed": True, "native_qualified": False,
              "status": "TRAINED_PILOT_REQUIRES_PARITY_RETENTION_AND_OOD_EVALUATION",
              "seed": seed, "steps": steps, "elapsed_seconds": elapsed,
              "trainable_parameters": sum(p.numel() for p in parameters),
              "data_sha256": digest(rows), "config": config,
              "moe_config": vars(moe_config), "control_config": vars(control),
              "checkpoint_sha256": sha256_file(out / "adapter_bank.pt"), "logs": logs}
    write_json(out / "report.json", report)
    return report


def load_bank(model, config, path, expected_sha256):
    if sha256_file(path) != expected_sha256:
        raise ValueError("adapter bank hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["schema"] != 1 or saved["config_sha256"] != digest(config):
        raise ValueError("bank base/run configuration mismatch")
    bank = inject(model, list(saved["projections"]), MoEConfig(**saved["moe_config"]),
                  ControlConfig(**saved["control_config"]), digest(config))
    for name, module in bank.items():
        module.load_adapter_state(saved["projections"][name])
        module.eval()
    model.eval().requires_grad_(False)
    return bank
