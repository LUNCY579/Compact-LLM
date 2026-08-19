#!/usr/bin/env python3
"""Copy or LoRA-merge dense Rineya cores into this vault. Never overwrite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

VAULT = Path(__file__).resolve().parent
REPO = VAULT.parent
PHASE = REPO / "short-response-model-phase-0-1-2" / "artifacts"
TEACHER = PHASE / "phase_4_model_soups_phase2c_balanced_training" / "accepted_model"
V1_ADAPTER = PHASE / "teacher_dual_sft_v1" / "lr_sweep" / "lr5e4" / "sft_run" / "adapter"

CORES = {
    "pulse": {
        "name": "Rineya-Pulse-1.0.0-rc1",
        "core_id": "rineya.core.pulse.v1.0.0-rc1.19L.454M.dense",
        "codename": "Pulse",
        "version": "1.0.0-rc1",
        "kind": "dense_copy",
        "public_claim": "Phase 8 sealed short-response teacher. Local one-line chat.",
        "status": "human_commercial_pass_phase_8",
        "base": TEACHER,
        "adapter": None,
    },
    "aegis": {
        "name": "Rineya-Aegis-1.1.0-devfreeze",
        "core_id": "rineya.core.aegis.v1.1.0-devfreeze.19L.454M.dual-sft-v1-lr5e4",
        "codename": "Aegis",
        "version": "1.1.0-devfreeze",
        "kind": "lora_merge",
        "public_claim": "Dual-skill DEV freeze (chat + grounded RAG). Sealed TEST not run.",
        "status": "dev_freeze_not_sealed",
        "base": TEACHER,
        "adapter": V1_ADAPTER,
    },
}

WEIGHT_NAMES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "soup_manifest.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(path: Path) -> dict[str, object]:
    info: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        stat = path.stat()
        info.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)})
    elif path.is_dir():
        files = []
        for child in sorted(path.rglob("*")):
            if child.is_file():
                stat = child.stat()
                files.append(
                    {
                        "rel": str(child.relative_to(path)),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        info["files"] = files
    return info


def assert_unchanged(before: dict[str, object], after: dict[str, object], label: str) -> None:
    if before != after:
        raise RuntimeError(f"Source changed during materialize ({label}). Aborting without repairing sources.")


def refuse_existing(dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing core: {dest}")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_dense(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=False)
    for name in WEIGHT_NAMES:
        source = src / name
        if source.is_file():
            shutil.copy2(source, dest / name)


def merge_lora(base: Path, adapter: Path, dest: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        local_files_only=True,
    )
    adapter_cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    expected = str(base)
    got = str(adapter_cfg.get("base_model_name_or_path") or "")
    if Path(got).resolve() != base.resolve():
        raise RuntimeError(f"Adapter base mismatch:\n  adapter={got}\n  expected={expected}")
    wrapped = PeftModel.from_pretrained(model, str(adapter))
    merged = wrapped.merge_and_unload(safe_merge=True)
    dest.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(str(dest), safe_serialization=True)
    tokenizer.save_pretrained(str(dest))
    del wrapped, merged, model
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(dest),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        local_files_only=True,
    )
    layers = int(reloaded.config.num_hidden_layers)
    del reloaded
    if layers != 19:
        raise RuntimeError(f"Reloaded layer count {layers} != 19")


def verify_core(dest: Path) -> dict[str, object]:
    config = json.loads((dest / "config.json").read_text(encoding="utf-8"))
    weights = dest / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if (dest / "adapter_config.json").exists() or (dest / "adapter_model.safetensors").exists():
        raise RuntimeError(f"Destination still looks like a PEFT adapter: {dest}")
    layers = int(config["num_hidden_layers"])
    if layers != 19:
        raise RuntimeError(f"{dest} has num_hidden_layers={layers}, expected 19")
    if config.get("model_type") != "qwen3":
        raise RuntimeError(f"Unexpected model_type {config.get('model_type')}")
    return {
        "num_hidden_layers": layers,
        "hidden_size": config.get("hidden_size"),
        "vocab_size": config.get("vocab_size"),
        "model_type": config.get("model_type"),
        "architecture": (config.get("architectures") or [None])[0],
        "weight_bytes": weights.stat().st_size,
        "weight_sha256": sha256_file(weights),
        "has_tokenizer": (dest / "tokenizer.json").is_file(),
    }


def materialize(key: str) -> Path:
    spec = CORES[key]
    dest = VAULT / spec["name"]
    base: Path = spec["base"]
    adapter: Path | None = spec["adapter"]
    refuse_existing(dest)
    if dest.exists():
        raise FileExistsError(dest)
    if not (base / "model.safetensors").is_file():
        raise FileNotFoundError(base / "model.safetensors")
    if adapter is not None and not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter / "adapter_model.safetensors")

    before_base = snapshot(base)
    before_adapter = snapshot(adapter) if adapter is not None else None
    started = utc_now()
    staging = dest.with_name(f"{dest.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        if spec["kind"] == "dense_copy":
            copy_dense(base, staging)
        elif spec["kind"] == "lora_merge":
            assert adapter is not None
            merge_lora(base, adapter, staging)
        else:
            raise ValueError(spec["kind"])

        after_base = snapshot(base)
        after_adapter = snapshot(adapter) if adapter is not None else None
        assert_unchanged(before_base, after_base, "teacher base")
        if before_adapter is not None:
            assert_unchanged(before_adapter, after_adapter, "adapter")

        checks = verify_core(staging)
        payload = {
            "core_id": spec["core_id"],
            "codename": spec["codename"],
            "directory_name": spec["name"],
            "version": spec["version"],
            "kind": spec["kind"],
            "status": spec["status"],
            "public_claim": spec["public_claim"],
            "created_at_utc": started,
            "finished_at_utc": utc_now(),
            "device": "cpu",
            "sources_unmodified": True,
            "source_base": str(base),
            "source_adapter": str(adapter) if adapter else None,
            "source_base_weight_sha256": sha256_file(base / "model.safetensors"),
            "source_adapter_sha256": sha256_file(adapter / "adapter_model.safetensors") if adapter else None,
            "checks": checks,
            "note": "Vault write only. Official phase artifacts were not opened for write.",
        }
        write_json(staging / "CORE.json", payload)
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite existing core: {dest}")
        staging.replace(dest)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return dest


def update_catalog(written: list[str]) -> None:
    catalog_path = VAULT / "CATALOG.json"
    catalog = {"vault": "Rineya_Published_Cores", "updated_at_utc": utc_now(), "cores": []}
    for spec in CORES.values():
        dest = VAULT / spec["name"]
        entry = {
            "core_id": spec["core_id"],
            "codename": spec["codename"],
            "directory": spec["name"],
            "present": dest.is_dir() and (dest / "model.safetensors").is_file(),
        }
        core_json = dest / "CORE.json"
        if core_json.is_file():
            entry["core"] = json.loads(core_json.read_text(encoding="utf-8"))
        catalog["cores"].append(entry)
    catalog["last_written"] = written
    write_json(catalog_path, catalog)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Rineya dense cores without overwriting.")
    parser.add_argument("cores", nargs="+", choices=["pulse", "aegis", "all"])
    args = parser.parse_args()
    keys = list(CORES) if "all" in args.cores else args.cores
    written: list[str] = []
    for key in keys:
        dest = VAULT / CORES[key]["name"]
        if dest.exists():
            print(f"SKIP (already exists, never overwrite): {dest}", flush=True)
            continue
        print(f"WRITE {CORES[key]['core_id']} -> {dest}", flush=True)
        materialize(key)
        written.append(CORES[key]["name"])
        print(f"OK {dest}", flush=True)
    update_catalog(written)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
