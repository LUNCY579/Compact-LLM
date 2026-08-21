#!/usr/bin/env python3
"""CPU shape/suffix preflight: R2 kept blocks vs Pulse/Aegis donors. Read-only on RineyaExperiment."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
READ_ONLY_SOURCE = Path(r"C:\Users\clclc\RineyaExperiment\artifacts\rineya-core-en32k-28L")
SCAFFOLD = LAB / "scaffolds" / "rineya-core-en32k-28L"
PULSE = (
    REPO
    / "short-response-model-phase-0-1-2"
    / "artifacts"
    / "phase_4_model_soups_phase2c_balanced_training"
    / "accepted_model"
)
AEGIS = REPO / "Rineya_Published_Cores" / "Rineya-Aegis-1.1.0-devfreeze"
LAYER_MAP = LAB / "configs" / "layer_map.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def safetensors_path(model_dir: Path) -> Path:
    single = model_dir / "model.safetensors"
    if single.is_file():
        return single
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        raise SystemExit(f"Sharded checkpoint not supported in preflight: {model_dir}")
    raise SystemExit(f"Missing model.safetensors: {model_dir}")


def layer_suffix_shapes(weights: Path, index: int) -> dict[str, list[int]]:
    prefix = f"model.layers.{index}."
    mapping: dict[str, list[int]] = {}
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                mapping[key[len(prefix) :]] = list(handle.get_slice(key).get_shape())
    if not mapping:
        raise SystemExit(f"No tensors for {prefix}* in {weights}")
    return mapping


def compare_layers(
    left_label: str,
    left_shapes: dict[str, list[int]],
    right_label: str,
    right_shapes: dict[str, list[int]],
) -> dict[str, Any]:
    missing = sorted(set(left_shapes) - set(right_shapes))
    extra = sorted(set(right_shapes) - set(left_shapes))
    mismatches = sorted(
        suffix
        for suffix in set(left_shapes) & set(right_shapes)
        if left_shapes[suffix] != right_shapes[suffix]
    )
    return {
        "left": left_label,
        "right": right_label,
        "suffix_count": len(left_shapes),
        "missing_on_right": missing,
        "extra_on_right": extra,
        "shape_mismatches": mismatches,
        "ok": not missing and not extra and not mismatches,
    }


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYTHONUTF8", "1")

    if not READ_ONLY_SOURCE.is_dir():
        raise SystemExit(f"Read-only R2 source missing: {READ_ONLY_SOURCE}")
    if not SCAFFOLD.is_dir():
        raise SystemExit(f"Local R2 copy missing: {SCAFFOLD}")

    layer_map = load_json(LAYER_MAP)
    mapping = {int(src): int(dst) for src, dst in layer_map["rineya_to_qwen3"].items()}
    r2_weights = safetensors_path(SCAFFOLD)
    src_weights = safetensors_path(READ_ONLY_SOURCE)
    pulse_weights = safetensors_path(PULSE)
    aegis_weights = safetensors_path(AEGIS)

    src_rec = file_record(src_weights)
    dst_rec = file_record(r2_weights)
    if src_rec["bytes"] != dst_rec["bytes"]:
        raise SystemExit(
            f"R2 copy size mismatch: source {src_rec['bytes']} vs copy {dst_rec['bytes']}"
        )

    print(f"[{utc_now()}] Hashing source and copy model.safetensors", flush=True)
    src_hash = sha256_file(src_weights)
    dst_hash = sha256_file(r2_weights)
    if src_hash != dst_hash:
        raise SystemExit("R2 copy SHA-256 does not match read-only source")

    scaffold_cfg = load_json(SCAFFOLD / "config.json")
    pulse_cfg = load_json(PULSE / "config.json")
    aegis_cfg = load_json(AEGIS / "config.json")
    if int(scaffold_cfg["vocab_size"]) != 32768:
        raise SystemExit(f"R2 vocab_size {scaffold_cfg['vocab_size']} != 32768")
    if int(scaffold_cfg["num_hidden_layers"]) != 28:
        raise SystemExit("R2 is not 28 layers")
    if int(pulse_cfg["num_hidden_layers"]) != 19 or int(aegis_cfg["num_hidden_layers"]) != 19:
        raise SystemExit("Pulse/Aegis must be 19-layer donors")

    print(f"[{utc_now()}] Comparing kept-layer suffixes (safetensors, no full load)", flush=True)
    checks: list[dict[str, Any]] = []
    pairs = [
        (0, mapping[0], "pulse", pulse_weights),
        (0, mapping[0], "aegis", aegis_weights),
        (18, mapping[18], "pulse", pulse_weights),
        (18, mapping[18], "aegis", aegis_weights),
    ]
    for rineya_i, r2_i, donor_name, donor_path in pairs:
        r2_shapes = layer_suffix_shapes(r2_weights, r2_i)
        donor_shapes = layer_suffix_shapes(donor_path, rineya_i)
        row = compare_layers(
            f"r2.layers.{r2_i}",
            r2_shapes,
            f"{donor_name}.layers.{rineya_i}",
            donor_shapes,
        )
        checks.append(row)
        print(f"[{utc_now()}] {row['left']} vs {row['right']}: ok={row['ok']}", flush=True)
        if not row["ok"]:
            raise SystemExit(f"Shape/suffix mismatch: {json.dumps(row)}")

    payload = {
        "generated_at": utc_now(),
        "read_only_source": str(READ_ONLY_SOURCE),
        "local_copy": str(SCAFFOLD),
        "source_weights": src_rec,
        "copy_weights": dst_rec,
        "sha256_match": True,
        "sha256": src_hash,
        "scaffold_vocab_size": int(scaffold_cfg["vocab_size"]),
        "scaffold_layers": int(scaffold_cfg["num_hidden_layers"]),
        "checks": checks,
        "note": "RineyaExperiment was opened read-only. Embeddings are not compared; stitch does not blend them.",
    }
    out_json = LAB / "outputs" / "preflight.json"
    write_json(out_json, payload)
    write_json(LAB / "copy_manifest.json", {
        "generated_at": utc_now(),
        "read_only_source": str(READ_ONLY_SOURCE),
        "local_copy": str(SCAFFOLD),
        "model_safetensors_bytes": src_rec["bytes"],
        "sha256": src_hash,
        "vocab_size": 32768,
        "wrote_to_rineyaexperiment": False,
    })
    print(f"[{utc_now()}] Preflight OK -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
