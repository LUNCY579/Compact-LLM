"""Explicit, local-only Qwen3/Rineya binding; never invoked on import."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import torch

from .backends import environment, precision_guard
from .capture import sdpa_session
from .core import error_metrics
from .protocol import (checkpoint_manifest, file_hash, identity, load_rows,
                       parameter_hash, validate_config)


def preflight(config: dict):
    validate_config(config)
    checkpoint = checkpoint_manifest(Path(config["checkpoint"]))
    if checkpoint["sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match its frozen manifest")
    if file_hash(Path(config["data"])) != config["data_sha256"]:
        raise ValueError("Input data hash mismatch")
    rows = load_rows(Path(config["data"]), config["max_tokens"])
    runtime = environment()
    try:
        runtime["transformers"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError("Native binding needs transformers; synthetic tests do not") from None
    if config["device"] == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA is unavailable")
        free, _ = torch.cuda.mem_get_info()
        if free < config["min_free_cuda_gib"] * 1024**3:
            raise RuntimeError("GPU free-memory floor not met; no model loaded")
    return rows, identity(config, checkpoint, runtime)


def load_model(config: dict):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        config["checkpoint"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    )
    count = sum(p.numel() for p in model.parameters())
    if model.config.model_type != "qwen3" or not 450_000_000 <= count <= 750_000_000:
        raise ValueError("This native binding is only for the exact Qwen3-derived 0.6B-class Rineya")
    if getattr(model.config, "quantization_config", None):
        raise ValueError("Quantized loading is not part of this experiment")
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("Checkpoint recomputation needs separate capture qualification")
    model.config.use_cache = False
    parameters = dict(model.named_parameters())
    names = set(config["trainable_names"])
    if names - parameters.keys():
        raise ValueError(f"Unknown trainable parameters: {sorted(names - parameters.keys())}")
    selected_count = sum(parameters[name].numel() for name in names)
    if selected_count > config["max_trainable_parameters"]:
        raise ValueError("Trainable parameter budget exceeded")
    model.requires_grad_(False)
    # FP32 master trainable tensors, shared across every arm; projection operands
    # remain BF16 under autocast. No BF16 optimizer-state confound is introduced.
    for name in names:
        parameters[name].data = parameters[name].data.float()
        parameters[name].requires_grad_(True)
    model.to(config["device"])
    model.eval()  # deliberate: dropout disabled even during gradient updates
    return model


def forward_loss(model, row: dict, config: dict, input_gradient: bool = False):
    device = config["device"]
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
    vocab = model.config.vocab_size
    if ids.max() >= vocab or (labels[labels != -100] >= vocab).any():
        raise ValueError("Token id exceeds the pinned model vocabulary")
    args = {"labels": labels, "use_cache": False, "return_dict": True}
    if input_gradient:
        args["inputs_embeds"] = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    else:
        args["input_ids"] = ids
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        output = model(**args)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("Nonfinite/missing model loss")
    return output.loss


def qualification_batches(model, rows: list[dict], config: dict):
    """Full-model fixed-weight paired gradients plus exact boundary captures.

    The native and FP32 arms may have different forwards. Only the two dense
    arms are a fixed-forward backward intervention. Nothing here updates weights.
    """
    captures, details, events = [], [], set()
    before = parameter_hash(model.named_parameters())
    selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for purpose in ("acquisition", "retention"):
        row = next(r for r in rows if r["split"] == "dev" and r["purpose"] == purpose)
        gradients, losses = {}, {}
        for arm in ("native", "fp32", "cast_only", "gproj"):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(config["seeds"][0])
            local_captures = [] if arm == "native" else None
            with precision_guard(), sdpa_session(
                arm, config["backend"], local_captures, max_captures=len(config["capture_call_indices"]),
                capture_call_indices=tuple(config["capture_call_indices"]),
                max_tokens=config["max_tokens"],
                metadata={"origin": "native_rineya", "purpose": purpose, "row_id": row["id"],
                          "checkpoint_sha256": config["checkpoint_sha256"]},
            ) as session:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    loss = forward_loss(model, row, config, input_gradient=True)
                    loss.backward()
            if session["calls"] == 0:
                raise RuntimeError("No SDPA calls intercepted; this loader is not qualified")
            if arm in {"native", "fp32"}:
                events.update(e.key for e in prof.key_averages() if "scaled_dot_product" in e.key)
            losses[arm] = float(loss.detach())
            gradients[arm] = {}
            for name, p in selected:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise RuntimeError(f"Missing/nonfinite trainable gradient: {name}")
                gradients[arm][name] = p.grad.detach().cpu().clone()
            if local_captures is not None:
                if len(local_captures) != len(config["capture_call_indices"]):
                    raise RuntimeError("Requested SDPA capture sites did not all execute")
                if len(local_captures) != session["captured"]:
                    raise RuntimeError("Not all recorded forwards received incoming derivatives")
                captures.extend(local_captures)
        if losses["cast_only"] != losses["gproj"]:
            raise RuntimeError("Dense arms failed their shared-forward loss invariant")
        comparisons = {
            arm: {name: error_metrics(gradient, gradients["fp32"][name])
                  for name, gradient in gradients[arm].items()}
            for arm in ("native", "cast_only", "gproj")
        }
        details.append({"purpose": purpose, "losses": losses,
                        "trainable_gradients_vs_fp32": comparisons})
    model.zero_grad(set_to_none=True)
    if parameter_hash(model.named_parameters()) != before:
        raise RuntimeError("Qualification changed model parameters")
    return captures, details, sorted(events)
