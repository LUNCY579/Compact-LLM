from __future__ import annotations

import gc
import shutil
from pathlib import Path

from ..utils import (
    environment_manifest,
    require_non_overlapping_path,
    utc_timestamp,
    write_json,
)


ADAPTER_WEIGHT_PREFIXES = ("adapter_model", "adapter_model-")


def _has_full_checkpoint_weights(path: Path) -> bool:
    names = {item.name for item in path.iterdir()} if path.is_dir() else set()
    return bool(
        {"model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"}
        & names
        or any(name.startswith("model-") and name.endswith(".safetensors") for name in names)
        or any(name.startswith("pytorch_model-") and name.endswith(".bin") for name in names)
    )


def is_peft_adapter(path: Path) -> bool:
    if not path.is_dir() or not (path / "adapter_config.json").exists():
        return False
    names = {item.name for item in path.iterdir() if item.is_file()}
    has_adapter_weights = any(
        name.startswith(ADAPTER_WEIGHT_PREFIXES)
        and (name.endswith(".safetensors") or name.endswith(".bin") or name.endswith(".json"))
        for name in names
    )
    # A directory that contains a complete checkpoint and an adapter is ambiguous. Prefer
    # the complete checkpoint rather than accidentally applying the adapter twice.
    return has_adapter_weights and not _has_full_checkpoint_weights(path)


def materialize_peft_adapter(
    adapter_path: Path,
    base_model_path: Path,
    output_path: Path,
    *,
    trust_remote_code: bool = False,
    output_dtype: str = "auto",
) -> Path:
    """Merge a PEFT adapter into a full checkpoint before model souping.

    This function is intentionally lazy-imported so full-checkpoint users do not
    need PEFT. Adapter A/B factors are never averaged directly; each endpoint is
    first reconstructed into its actual full weight delta.
    """

    adapter_path = adapter_path.expanduser().resolve()
    base_model_path = base_model_path.expanduser().resolve()
    output_path = require_non_overlapping_path(
        output_path,
        (adapter_path, base_model_path),
        label="adapter materialization destination",
    )
    temporary = output_path.with_name(f".{output_path.name}.building")
    require_non_overlapping_path(
        temporary,
        (adapter_path, base_model_path),
        label="adapter materialization temporary directory",
    )
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Install the 'adapters' extra to materialize PEFT adapters.") from exc
    if output_path.exists():
        raise FileExistsError(f"Adapter materialization destination already exists: {output_path}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model = None
    merged = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            torch_dtype=output_dtype,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter_path))
        merged = model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(str(temporary), safe_serialization=True, max_shard_size="1GB")
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model_path), trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(str(temporary))
        write_json(
            temporary / "adapter_materialization.json",
            {
                "schema_version": "1.0",
                "created_at": utc_timestamp(),
                "adapter_path": str(adapter_path.expanduser().resolve()),
                "base_model_path": str(base_model_path.expanduser().resolve()),
                "safe_merge": True,
                "output_dtype": output_dtype,
                "environment": environment_manifest(),
            },
        )
        temporary.replace(output_path)
        return output_path
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        del merged, model
        gc.collect()
