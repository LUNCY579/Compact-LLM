"""Create a receipt-bound FP32 scalar arm from a COMPLETE native tail checkpoint.

No model is instantiated. Validation is torch-free; materialization holds the same
lease as model workers and refuses less than 4.6 GiB available host RAM. Failed
outputs are preserved, never retried in place or deleted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.model_quality_gpu_v1.integrity import (  # noqa: E402
    decode, digest, file_hash, new_dir, read_json, regular, unseal, write_new,
)
from experiments.model_quality_gpu_v1.resources import lease  # noqa: E402

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt")
SUFFIXES = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)
EXPECTED_NAMES = {f"model.layers.{layer}.{suffix}"
                  for layer in range(8, 28) for suffix in SUFFIXES}
GIB = 2**30


def require_ram() -> int:
    import psutil
    available = psutil.virtual_memory().available
    if available < 4.6 * GIB:
        raise RuntimeError(f"materialization host RAM guard: need 4.6 GiB, have {available / GIB:.2f}")
    return available


def resource_lock_path() -> Path:
    return Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"


def tensor_header(path: Path) -> dict:
    """Read bounded safetensors metadata without a tensor runtime or allocation."""
    path = regular(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        length, = struct.unpack("<Q", prefix)
        if not 2 <= length <= 32 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError("invalid safetensors header size")
        header = decode(stream.read(length).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    header.pop("__metadata__", None)
    if not header:
        raise ValueError("empty safetensors inventory")
    size = path.stat().st_size - 8 - length
    spans = []
    for name, value in header.items():
        if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"invalid tensor metadata: {name}")
        shape, offsets = value["shape"], value["data_offsets"]
        if (not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] < offsets[1] <= size
                or value["dtype"] not in ("F32", "BF16", "F16")):
            raise ValueError(f"invalid floating tensor: {name}")
        width = 4 if value["dtype"] == "F32" else 2
        if offsets[1] - offsets[0] != math.prod(shape) * width:
            raise ValueError(f"tensor shape/bytes mismatch: {name}")
        spans.append(tuple(offsets))
    spans.sort()
    if spans[0][0] != 0 or spans[-1][1] != size or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("tensor data contains gaps, overlaps, or trailing bytes")
    return header


def inspect_inputs(source: Path, checkpoint: Path) -> dict:
    """Verify sealed provenance and every input byte before creating an output."""
    source, checkpoint = Path(source).absolute(), Path(checkpoint).absolute()
    manifest_path = checkpoint / "manifest.json"
    manifest = unseal(read_json(manifest_path))
    if (set(manifest) != {"schema", "experiment", "status", "binding", "progress", "rng", "entries"}
            or manifest["schema"] != 1 or manifest["experiment"] != "MQ-001"
            or manifest["status"] != "COMPLETE"):
        raise ValueError("expected a COMPLETE native MQ-001 checkpoint")
    if type(manifest["progress"].get("updates")) is not int or manifest["progress"]["updates"] < 1:
        raise ValueError("checkpoint must record completed updates")
    binding = manifest["binding"]
    required = {"experiment", "plan_sha256", "data_contract_sha256", "grant_sha256",
                "code_sha256", "mq_code_sha256", "checkout", "runtime", "source"}
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError("checkpoint lacks exact native training binding")
    assets = binding["source"]["assets"]
    if (not isinstance(assets, dict) or set(assets) != {"model.safetensors", *SIDECARS}
            or binding["source"].get("sha256") != digest(assets)):
        raise ValueError("checkpoint source asset binding differs")
    for name, expected in assets.items():
        if file_hash(source / name) != expected:
            raise ValueError(f"source asset hash mismatch: {name}")
    config = read_json(source / "config.json")
    if config.get("num_hidden_layers") != 28 or config.get("tie_word_embeddings") is not True:
        raise ValueError("source must use the registered 28-layer tied vocabulary geometry")
    source_header = tensor_header(source / "model.safetensors")
    tail = {name for name in source_header if any(name.startswith(f"model.layers.{i}.") for i in range(8, 28))}
    if tail != EXPECTED_NAMES:
        raise ValueError("source tail must contain exactly 220 expected tensors across layers 8 through 27")
    entries = manifest["entries"]
    if (not isinstance(entries, list) or len(entries) != 220
            or {entry.get("name") for entry in entries} != EXPECTED_NAMES):
        raise ValueError("checkpoint must cover exactly the same 220 trainable tensors")
    expected_files = {"manifest.json"} | {f"tensor-{i:04d}.safetensors" for i in range(220)}
    if {p.name for p in checkpoint.iterdir()} != expected_files:
        raise ValueError("checkpoint has missing or unexpected files")
    for i, entry in enumerate(entries):
        if (set(entry) != {"name", "file", "shape", "sha256"}
                or entry["file"] != f"tensor-{i:04d}.safetensors"
                or entry["shape"] != source_header[entry["name"]]["shape"]):
            raise ValueError("checkpoint entry path/name/shape mismatch")
        path = checkpoint / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"checkpoint tensor hash mismatch: {entry['file']}")
        header = tensor_header(path)
        if set(header) != {"parameter", "exp_avg", "exp_avg_sq"} or any(
                meta["dtype"] != "F32" or meta["shape"] != entry["shape"] for meta in header.values()):
            raise ValueError("checkpoint parameter and optimizer states must all be matching FP32 tensors")
    return {"source": str(source), "checkpoint": str(checkpoint),
            "source_assets_sha256": assets, "checkpoint_manifest_sha256": file_hash(manifest_path),
            "checkpoint_binding_sha256": digest(binding), "manifest": manifest,
            "source_header": source_header}


def materialize(source: Path, checkpoint: Path, alpha: float, out: Path) -> dict:
    if alpha not in GRID:
        raise ValueError(f"alpha must belong to the registered grid {GRID}")
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError("refusing to overwrite an existing arm directory")
    with lease(resource_lock_path()):
        require_ram()
        inspected = inspect_inputs(source, checkpoint)
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        torch.set_num_threads(1)
        available_before = require_ram()
        manifest = inspected["manifest"]
        entries = {entry["name"]: entry for entry in manifest["entries"]}
        tensors = {}
        with safe_open(str(Path(source) / "model.safetensors"), framework="pt", device="cpu") as source_file:
            for name in source_file.keys():
                src = source_file.get_tensor(name).to(torch.float32).contiguous()
                if not bool(torch.isfinite(src).all()):
                    raise ValueError(f"nonfinite source tensor: {name}")
                if name not in entries:
                    tensors[name] = src
                    continue
                entry = entries[name]
                path = Path(checkpoint) / entry["file"]
                # Check again immediately before consuming bytes; the final validation below
                # also refuses input mutation before publishing any result.
                if file_hash(path) != entry["sha256"]:
                    raise ValueError("checkpoint tensor changed after validation")
                with safe_open(str(path), framework="pt", device="cpu") as state:
                    trained = state.get_tensor("parameter")
                    if trained.dtype != torch.float32 or not bool(torch.isfinite(trained).all()):
                        raise ValueError(f"invalid trained tensor: {name}")
                    value = src if alpha == 0.0 else (trained.clone() if alpha == 1.0 else src + alpha * (trained - src))
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"nonfinite interpolant: {name}")
                tensors[name] = value.contiguous()
        # Identity verification is intentionally byte-based, including optimizer files.
        if inspect_inputs(source, checkpoint) != inspected:
            raise ValueError("source/checkpoint changed during materialization")
        output_bytes = sum(t.numel() * 4 for t in tensors.values())
        ancestor = out.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < output_bytes + 14 * GIB:
            raise RuntimeError("arm export would violate the native 14 GiB disk floor")
        new_dir(out)
        save_file(tensors, str(out / "model.safetensors"),
                  metadata={"format": "pt", "dose_interpolation_alpha": str(alpha)})
        del tensors
        for name in SIDECARS:
            with regular(Path(source) / name).open("rb") as src, (out / name).open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if file_hash(out / name) != inspected["source_assets_sha256"][name]:
                raise ValueError(f"copied sidecar changed: {name}")
        weight_sha = file_hash(out / "model.safetensors")
        receipt = {
            "schema": 1, "record": "CHECKPOINT_MATERIALIZATION", "layout": "huggingface",
            "experiment": "DOSE-INTERPOLATION-001", "alpha": alpha,
            "formula": "source + alpha * (trained - source); endpoints copied directly",
            "what_this_is": "FP32 scalar interpolation of one verified native training checkpoint; no model execution.",
            "source": inspected["source"], "source_assets_sha256": inspected["source_assets_sha256"],
            "source_weight_sha256": inspected["source_assets_sha256"]["model.safetensors"],
            "checkpoint": inspected["checkpoint"], "checkpoint_experiment": manifest["experiment"],
            "checkpoint_manifest_sha256": inspected["checkpoint_manifest_sha256"],
            "checkpoint_binding": manifest["binding"],
            "checkpoint_binding_sha256": inspected["checkpoint_binding_sha256"],
            "checkpoint_progress": manifest["progress"], "checkpoint_entries": manifest["entries"],
            "out_weight_sha256": weight_sha, "arm_weight_sha256": weight_sha,
            "arm_weight_bytes": (out / "model.safetensors").stat().st_size,
            "stored_dtype": "float32", "interpolation_dtype": "float32",
            "bf16_interpolation_used": False, "tensors_in_checkpoint": 220,
            "tensors_total": len(inspected["source_header"]),
            "tensors_applied": 0 if alpha == 0 else 220,
            "tensors_interpolated": 220 if 0 < alpha < 1 else 0,
            "tensors_copied_from_trained": 220 if alpha == 1 else 0,
            "source_tensors_untouched_count": len(inspected["source_header"]) - (0 if alpha == 0 else 220),
            "trainable_layers": list(range(8, 28)), "sidecars_copied": list(SIDECARS),
            "builder_sha256": file_hash(Path(__file__)), "available_host_bytes_before": available_before,
            "resource_lock": str(resource_lock_path()), "quality_claim": False, "release_permitted": False,
        }
        write_new(out / "MATERIALIZATION.json", receipt)
        write_new(out / "ARM.json", {**receipt, "record": "DOSE_INTERPOLATION_ARM"})
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(args.source, args.checkpoint, args.alpha, args.out)
    print(json.dumps({key: receipt[key] for key in
                      ("alpha", "checkpoint_manifest_sha256", "out_weight_sha256", "stored_dtype")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
