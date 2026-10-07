"""Optional local-only Qwen3-compatible 0.6B lane. Never called by import/smoke.

A custom Rineya loader, quantized graph, alternative attention implementation,
Windows telemetry fallback, or changed shape needs separate qualification.
"""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .adapters import adapter_state, inject_lora, load_adapters, merge_adapters
from .data import validate_cases
from .evaluation import collect
from .provenance import (NativeGuard, binding, exclusive_run, runtime_identity,
                         verify_receipt, write_json_new, file_sha256)
from .regularization import regularize, norm_targets
from .training import bounded_prefix, completion_loss, tensors, train_bounded


def _load(checkpoint, plan):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = Path(checkpoint)
    if not path.is_dir():
        raise ValueError("Remote IDs/downloads are forbidden")
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3" or config.num_hidden_layers != plan.expected_layers:
        raise ValueError("Checkpoint is not the explicitly bound Qwen3 layout")
    if (config.hidden_size != 1024 or config.intermediate_size != 3072 or
            not 150000 <= config.vocab_size <= 160000 or config.num_hidden_layers != 28):
        raise ValueError("Native shape is outside the bound 0.6B Qwen3 configuration")
    if plan.train_context > config.max_position_embeddings:
        raise ValueError("Training context exceeds the checkpoint contract")
    if getattr(config, "attention_dropout", 0) != 0 or getattr(config, "quantization_config", None):
        raise ValueError("Uncontrolled attention dropout or quantization")
    if getattr(config, "use_sliding_window", False) or getattr(config, "rope_scaling", None):
        raise ValueError("Changed positional/attention architecture requires a separate lane")
    if plan.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 is not supported")
    model = AutoModelForCausalLM.from_pretrained(str(path), config=config, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, torch_dtype=getattr(torch, plan.dtype),
        attn_implementation=plan.attention).to("cuda")
    count = sum(p.numel() for p in model.parameters())
    if not 500_000_000 <= count <= 800_000_000:
        raise ValueError("This lane is restricted to a 0.6B-class checkpoint")
    norm_targets(model, plan.expected_layers)
    torch.manual_seed(plan.seed)
    inject_lora(model, plan)
    if plan.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False, "preserve_rng_state": True})
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("An EOS/padding contract is required")
    return model, tokenizer


def _qualify(model, row, plan, guard):
    """Forward/backward and serving transformations; no optimizer updates."""
    device = next(model.parameters()).device
    ids, labels = tensors(row, device)
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits.detach()
        with regularize(model, plan):
            actual = model(input_ids=ids, use_cache=False).logits.detach()
        identity = torch.equal(expected, actual)
        with regularize(model, replace(plan, probability=0.0)):
            model.train()
            zero = model(input_ids=ids, use_cache=False).logits.detach()
        identity = identity and torch.equal(expected, zero)
    guard()
    model.train()
    with regularize(model, plan):
        loss = completion_loss(model(input_ids=ids, use_cache=False).logits, labels)
        loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    finite = bool(torch.isfinite(loss)) and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
    model.zero_grad(set_to_none=True)
    original = adapter_state(model)
    # Test a NONZERO adapter, not only its trivial zero-initialized export.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".B") and parameter.requires_grad:
                parameter.normal_(0, 1e-4)
    model.eval()
    state = adapter_state(model)
    import io
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    load_adapters(model, torch.load(stream, weights_only=True))
    serialization = all(torch.equal(v, adapter_state(model)[k]) for k, v in state.items())
    saved = [(name.rsplit(".", 1), mod) for name, mod in model.named_modules()
             if name + ".A" in state]
    try:
        with torch.no_grad():
            before = model(input_ids=ids, use_cache=False).logits
            merge_adapters(model)
            after = model(input_ids=ids, use_cache=False).logits
            error = (before.float() - after.float()).abs().max().item()
            merge = torch.allclose(before, after, atol=0.1 if plan.dtype == "bfloat16" else 1e-5,
                                   rtol=0.01 if plan.dtype == "bfloat16" else 1e-5)
    finally:
        for (parent, attr), mod in saved:
            setattr(model.get_submodule(parent), attr, mod)
        load_adapters(model, original)
    return {"identity_parity": bool(identity), "finite_backward": bool(finite),
            "serialization_parity": bool(serialization), "merge_parity": bool(merge),
            "merge_max_abs_error": error, "qualification_optimizer_steps": 0}


def native_run(operation, *, checkpoint, rows, bank_path, plan, eligibility, output_dir,
               allow_native_load=False, allow_training=False, qualification=None,
               adapter_path=None, allow_confirmation=False):
    if operation not in {"qualify", "train", "evaluate"}:
        raise ValueError("Unknown native operation")
    if not allow_native_load:
        raise PermissionError("Native checkpoint loading requires --allow-native-load")
    if operation == "train" and not allow_training:
        raise PermissionError("Training requires --allow-training and a fresh qualification receipt")
    validate_cases(rows, split="train" if operation in {"qualify", "train"} else None)
    if any(r["fixture"] for r in rows):
        raise ValueError("Fixture data is not native evidence or native training material")
    if operation != "evaluate":
        bounded_prefix(rows, plan)
    elif any(r["split"] == "train" for r in rows):
        raise ValueError("Do not report training-set scores as evaluation")
    elif any(r["split"] == "confirmation" for r in rows) and not allow_confirmation:
        raise PermissionError("Opening confirmation requires separate explicit authorization")
    guard = NativeGuard()
    guard.before_load()
    runtime = runtime_identity(torch)
    # Evaluation eligibility is separate: never request/forge TRAIN eligibility for a held-out bank.
    if operation == "evaluate":
        from .provenance import checkpoint_fingerprint, code_fingerprint
        from .data import digest
        attest = json.loads(Path(eligibility).read_text(encoding="utf-8"))
        if attest.get("bank_sha256") != file_sha256(bank_path) or attest.get("evaluation_eligible") is not True:
            raise ValueError("Missing/mismatched evaluation eligibility")
        bound = {"checkpoint_sha256": checkpoint_fingerprint(checkpoint), "bank_sha256": file_sha256(bank_path),
                 "code_sha256": code_fingerprint(), "plan_sha256": digest(plan.to_dict()), "runtime": runtime,
                 "eligibility_sha256": digest(attest)}
    else:
        bound = binding(checkpoint, bank_path, plan, eligibility, runtime)
    if operation == "train":
        if qualification is None:
            raise ValueError("Training requires a native qualification receipt")
        verify_receipt(json.loads(Path(qualification).read_text(encoding="utf-8")), bound)
    with exclusive_run(output_dir) as output:
        write_json_new(output / "request.json", {"operation": operation, "binding": bound, "plan": plan.to_dict()})
        try:
            model, tokenizer = _load(checkpoint, plan)
            guard()
            validate_cases(rows, vocabulary=model.config.vocab_size)
            if operation == "qualify":
                checks = _qualify(model, rows[0], plan, guard)
                success = all(checks[k] for k in ("identity_parity", "finite_backward", "serialization_parity", "merge_parity"))
                result = {"status": "READY_FOR_EXPLICIT_BOUNDED_TRAINING" if success else "QUALIFICATION_FAILED",
                          "binding": bound, "checks": checks, "created_at": datetime.now(timezone.utc).isoformat(),
                          "note": "No source authenticity/behavioral qualification is inferred from graph parity."}
            elif operation == "train":
                def save(meta, state):
                    name = f"adapter-{int(meta['fraction'] * 100):03d}"
                    torch.save(state, output / (name + ".pt"))
                    write_json_new(output / (name + ".json"), dict(meta, binding=bound,
                        adapter_sha256=file_sha256(output / (name + ".pt"))))
                result = train_bounded(model, rows, plan, guard=guard, on_checkpoint=save)
                result["binding"] = bound
            else:
                if adapter_path is not None:
                    meta = json.loads(Path(adapter_path).with_suffix(".json").read_text(encoding="utf-8"))
                    if (meta.get("adapter_sha256") != file_sha256(adapter_path) or
                            meta.get("binding", {}).get("checkpoint_sha256") != bound["checkpoint_sha256"] or
                            meta["binding"].get("plan_sha256") != bound["plan_sha256"] or
                            meta["binding"].get("code_sha256") != bound["code_sha256"]):
                        raise ValueError("Adapter identity/configuration mismatch")
                    load_adapters(model, torch.load(adapter_path, weights_only=True, map_location="cpu"))
                result = {"status": "COLLECTED_NOT_RELEASED", "binding": bound,
                          "adapter_sha256": file_sha256(adapter_path) if adapter_path else None,
                          "evaluation_contract": digest({"bank": bound["bank_sha256"],
                              "checkpoint": bound["checkpoint_sha256"], "code": bound["code_sha256"],
                              "runtime": bound["runtime"], "max_new_tokens": plan.max_new_tokens,
                              "context_limit": model.config.max_position_embeddings}),
                          "records": collect(model, tokenizer, rows, seed=plan.seed,
                              context_limit=model.config.max_position_embeddings,
                              max_new_tokens=plan.max_new_tokens, guard=guard)}
            write_json_new(output / "result.json", result)
            return result
        except Exception as exc:
            write_json_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                "status": "FAILED_NO_AUTOMATIC_RETRY", "binding": bound})
            raise
