"""Explicit standard-PEFT export for later, separately qualified GD-001 ingestion."""
from pathlib import Path

from .contracts import digest, file_hash, read_json, require, sealed, unseal, write_new


def peft_names(state):
    require(bool(state) and all(n.endswith((".lora_A", ".lora_B")) for n in state), "q/v LoRA state only")
    return {"base_model.model." + n + ".weight": t for n, t in state.items()}


def export_peft(checkpoint, registration, output):
    from safetensors.torch import load_file, save_file
    from .registry import check_registration
    r, plan, _ = check_registration(registration)
    checkpoint, output = Path(checkpoint).resolve(), Path(output)
    result = unseal(read_json(checkpoint.parents[1] / "RESULT.json"))
    receipt = unseal(read_json(checkpoint / "receipt.json"))
    require(result["status"] == "COMPLETE" and result["identity"] == digest(r)
            and result["evidence_kind"] == "qualified_native_hf", "complete registered native run required")
    require(receipt in [c["receipt"] for c in result["checkpoints"]]
            and receipt["adapter_sha256"] == file_hash(checkpoint / "adapter.safetensors"), "checkpoint provenance")
    state = load_file(str(checkpoint / "adapter.safetensors"), device="cpu")
    output.mkdir(parents=True, exist_ok=False)
    save_file(peft_names(state), str(output / "adapter_model.safetensors"))
    write_new(output / "adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM",
        "inference_mode": True, "r": plan.rank, "lora_alpha": plan.alpha, "lora_dropout": 0.0,
        "bias": "none", "target_modules": ["q_proj", "v_proj"],
        "base_model_name_or_path": r["assets"]["student"]["path"]})
    write_new(output / "RDD_EXPORT.json", sealed({"identity": digest(r), "source_receipt": receipt,
        "base_snapshot": r["assets"]["student"]["sha256"], "runtime_parity_qualified": False,
        "note": "Format conversion only. Qualify exact HF/PEFT output parity before GD-001 record_snapshot(kind='peft')."}))
