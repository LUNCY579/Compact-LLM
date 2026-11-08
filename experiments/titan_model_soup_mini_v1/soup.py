"""SOUP-MINI-001: offline, CPU-only, FP32, chunked checkpoint interpolation.

No model class, tokenizer loader, generation, training, network, or subprocess runner.
Inputs are immutable full Hugging Face safetensors exports, never raw adapters/deltas.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
from typing import Any

EXPERIMENT = "SOUP-MINI-001"
BASE_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
PINS = {
    "supa": BASE_SHA,
    "titan": "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7",
    "titan_sparse": "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284",
    "opallo": "e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b",
}
ASSETS = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
}
ARMS = {
    "supa_titan_50": ("supa", "titan", 0.5),
    "supa_sparse_50": ("supa", "titan_sparse", 0.5),
    "supa_opallo_50": ("supa", "opallo", 0.5),
    "titan_opallo_50": ("titan", "opallo", 0.5),
    "sparse_opallo_50": ("titan_sparse", "opallo", 0.5),
    "local_parent_50": ("LOCAL_PARENT", "local_ft", 0.5),
    "local_supa_50": ("supa", "local_ft", 0.5),
    "local_opallo_50": ("local_ft", "opallo", 0.5),
}
CHUNK_ELEMENTS = 262144
DTYPES = {"BF16", "F16", "F32"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, f"duplicate JSON key: {key}")
            out[key] = value
        return out
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    require(isinstance(result, dict), "expected a JSON object")
    return result


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    require(config.get("architectures") == ["Qwen3ForCausalLM"], "custom/unknown architecture")
    require(config.get("model_type") == "qwen3", "not qwen3")
    require(config.get("num_hidden_layers") == 28, "not 28 layers")
    require(config.get("hidden_size") == 1024, "wrong hidden width")
    require(config.get("vocab_size") == 151936, "wrong vocabulary size")
    require(config.get("tie_word_embeddings") is True, "untied output head")
    require(not any(k in config for k in ("auto_map", "quantization_config", "abstain_logit_bias")),
            "custom loader, quantization, or abstention sidecar is excluded")
    ignored = {"torch_dtype", "dtype", "_name_or_path", "transformers_version"}
    return {k: v for k, v in config.items() if k not in ignored}


def checked_file(root: Path, name: str, expected: str) -> Path:
    path = root / name
    require(path.is_file() and not path.is_symlink(), f"missing file or symlink: {path}")
    require(sha256(path) == expected, f"SHA-256 mismatch: {path}")
    return path


def inventory(handle) -> dict[str, list[int]]:
    keys = set(handle.keys())
    require("model.embed_tokens.weight" in keys, "missing embedding tensor")
    # HF tied-weight serialization may omit the duplicate head; never drop other keys.
    keys.discard("lm_head.weight")
    inv = {k: handle.get_slice(k).get_shape() for k in sorted(keys)}
    for k, shape in inv.items():
        require(shape and all(type(n) is int and n > 0 for n in shape), f"bad shape: {k}")
        require(handle.get_slice(k).get_dtype() in DTYPES, f"unsupported dtype: {k}")
        require(math.prod(shape[1:]) <= CHUNK_ELEMENTS, f"one row exceeds chunk limit: {k}")
    return inv


def slices(shape: list[int]):
    step = max(1, CHUNK_ELEMENTS // math.prod(shape[1:]))
    for start in range(0, shape[0], step):
        yield start, min(start + step, shape[0])


def check_tied_head(handle) -> None:
    import torch
    if "lm_head.weight" not in handle.keys():
        return
    a = handle.get_slice("model.embed_tokens.weight")
    b = handle.get_slice("lm_head.weight")
    require(a.get_shape() == b.get_shape(), "tied-head shape mismatch")
    for lo, hi in slices(a.get_shape()):
        require(torch.equal(a[lo:hi].float(), b[lo:hi].float()), "tied-head values differ")


def lerp(a, b, alpha: float):
    import torch
    require(type(alpha) in (int, float) and math.isfinite(alpha) and 0 <= alpha <= 1,
            "alpha must be finite and in [0,1]")
    require(a.shape == b.shape, "shape mismatch")
    require(a.device.type == b.device.type == "cpu", "CPU tensors required")
    require(a.is_floating_point() and b.is_floating_point(), "floating tensors required")
    a, b = a.float(), b.float()
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "nonfinite input")
    # Endpoint copies and a + alpha*(b-a) preserve unchanged coordinates exactly.
    out = a.clone() if alpha == 0 else b.clone() if alpha == 1 else a + (b - a) * alpha
    require(bool(torch.isfinite(out).all()), "nonfinite output")
    return out


def inspect_manifest(path: Path, arm: str | None = None) -> dict[str, Any]:
    from safetensors import safe_open
    manifest = read_json(path)
    require(manifest.get("experiment") == EXPERIMENT and manifest.get("schema") == 1,
            "wrong experiment/schema")
    entries = manifest.get("artifacts", {})
    require(isinstance(entries, dict), "artifacts must be an object")
    ids = {"supa"}
    pair = None
    if arm:
        require(arm in ARMS, "unregistered arm")
        left, right, alpha = ARMS[arm]
        if "local_ft" in (left, right):
            local = entries.get("local_ft", {})
            require(local.get("parent_id") in ("titan", "titan_sparse"),
                    "local parent must be exact registered Titan or Titan-Sparse")
            ids.add(local["parent_id"])
            if left == "LOCAL_PARENT":
                left = local["parent_id"]
        ids.update((left, right))
        pair = (left, right, alpha)
    else:
        ids.update(PINS)
    roots, snapshots, configs, inventories = {}, {}, {}, {}
    for model_id in sorted(ids):
        entry = entries.get(model_id, {})
        require(entry.get("representation") == "full_weights", "full weights required; no adapters/partial tensors")
        require(entry.get("shared_base_weight_sha256") == BASE_SHA, "shared base mismatch")
        expected = PINS.get(model_id)
        if model_id == "local_ft":
            expected = entry.get("weight_sha256")
            require(isinstance(expected, str) and bool(re.fullmatch(r"[0-9a-f]{64}", expected)), "unbound local hash")
            proof = entry.get("lineage_receipt", {})
            proof_path = Path(proof.get("path") or "__UNBOUND__")
            require(proof_path.is_file() and sha256(proof_path) == proof.get("sha256"), "missing local lineage receipt")
            receipt = read_json(proof_path)
            require(receipt.get("child_weight_sha256") == expected, "local child hash mismatch")
            require(receipt.get("parent_weight_sha256") == PINS[entry["parent_id"]], "local parent hash mismatch")
            require(receipt.get("shared_base_weight_sha256") == BASE_SHA, "local ancestry mismatch")
            require(receipt.get("coordinate_alignment") == "same_coordinates_no_permutation", "unaligned local coordinates")
            require(receipt.get("reviewed") is True, "local lineage not reviewed")
            # Provenance is operator-attested, not inferred from a directory name or shape.
        require(entry.get("weight_sha256") == expected, f"pin mismatch: {model_id}")
        require(isinstance(entry.get("path"), str) and bool(entry["path"].strip()), f"unbound path: {model_id}")
        raw_root = Path(entry["path"]).expanduser()
        require(raw_root.is_absolute() and raw_root.is_dir() and not raw_root.is_symlink(), "absolute non-symlink model directory required")
        root = raw_root.resolve()
        require(not (root / "adapter_config.json").exists(), "adapter export excluded")
        require(not (root / "model.safetensors.index.json").exists(), "sharded input not supported in v1")
        files = {"model.safetensors": expected, **ASSETS}
        for name, digest in files.items():
            checked_file(root, name, digest)
        config_path = root / "config.json"
        require(config_path.is_file() and not config_path.is_symlink(), "missing config or symlink")
        if model_id == "supa":
            checked_file(root, "config.json", "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd")
        configs[model_id] = canonical_config(read_json(config_path))
        files["config.json"] = sha256(config_path)
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as f:
            inventories[model_id] = inventory(f)
            check_tied_head(f)
        inv = inventories[model_id]
        require(len(inv) == 310 and sum(math.prod(x) for x in inv.values()) == 596049920,
                "wrong full-model inventory; partial trained tensors are not deltas or full checkpoints")
        require(inv["model.embed_tokens.weight"] == [151936, 1024], "embedding axes mismatch")
        roots[model_id], snapshots[model_id] = str(root), files
    require(all(c == configs["supa"] for c in configs.values()), "architecture/config mismatch")
    require(all(i == inventories["supa"] for i in inventories.values()), "tensor key/shape mismatch")
    return {"experiment": EXPERIMENT, "manifest_sha256": sha256(path), "roots": roots,
            "files": snapshots, "inventory": inventories["supa"], "pair": pair,
            "local_lineage": "operator_attested" if "local_ft" in ids else "pinned_records",
            "status": "CHECKPOINT_COMPATIBILITY_ONLY", "native_model_qualified": False,
            "training_authorized": False, "release_permitted": False}


def recheck(receipt: dict[str, Any]) -> None:
    for model_id, files in receipt["files"].items():
        for name, digest in files.items():
            checked_file(Path(receipt["roots"][model_id]), name, digest)


def stream_pair(left: Path, right: Path, target: Path, alpha: float) -> dict[str, Any]:
    """Low-level writer, also exercised on tiny fixtures; not a model qualification API."""
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    require(not target.exists(), "target exists; overwrite refused")
    with ExitStack() as stack:
        a = stack.enter_context(safe_open(left, framework="pt", device="cpu"))
        b = stack.enter_context(safe_open(right, framework="pt", device="cpu"))
        inv = inventory(a)
        require(inv == inventory(b), "key/shape mismatch")
        check_tied_head(a)
        check_tied_head(b)
        header, offset = {"__metadata__": {"format": "pt", "experiment": EXPERIMENT}}, 0
        for key, shape in inv.items():
            nbytes = math.prod(shape) * 4
            header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        changed, erased_bf16, delta_nonzero, delta_sq, round_error_sq = 0, 0, 0, 0.0, 0.0
        with target.open("xb") as out:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            with torch.no_grad():
                for key, shape in inv.items():
                    for lo, hi in slices(shape):
                        x, y = a.get_slice(key)[lo:hi], b.get_slice(key)[lo:hi]
                        z = lerp(x, y, alpha)
                        x32 = x.float()
                        delta = z - x32
                        nonzero = delta != 0
                        rounded = z.bfloat16().float()
                        changed += int(nonzero.sum())
                        delta_nonzero += int(nonzero.sum())
                        erased_bf16 += int((nonzero & (rounded == x32)).sum())
                        delta_sq += float(delta.double().square().sum())
                        round_error_sq += float((rounded - z).double().square().sum())
                        out.write(z.numpy().astype("<f4", copy=False).tobytes(order="C"))
            out.flush()
            os.fsync(out.fileno())
        require(target.stat().st_size == 8 + len(encoded) + offset, "output byte count mismatch")
    with safe_open(target, framework="pt", device="cpu") as check:
        require(inventory(check) == inv, "round-trip inventory mismatch")
    return {"weight_sha256": sha256(target), "weight_bytes": target.stat().st_size,
            "stored_dtype": "float32", "arithmetic": "left + alpha*(right-left)",
            "alpha_right": alpha, "changed_elements_vs_left": changed,
            "delta_l2_vs_left": math.sqrt(delta_sq),
            "bf16_diagnostic_only": {"changed_elements_erased_vs_left": erased_bf16,
                                     "changed_elements": delta_nonzero,
                                     "absolute_rounding_error_l2": math.sqrt(round_error_sq)},
            "tensor_roundtrip_valid": True, "native_model_qualified": False}


def build(manifest: Path, arm: str, output: Path) -> dict[str, Any]:
    receipt = inspect_manifest(manifest, arm)
    output = output.expanduser().absolute()
    require(output.parent.is_dir(), "output parent must already exist")
    output = output.parent.resolve() / output.name
    require(not output.exists() and not output.is_symlink(), "output exists; overwrite refused")
    for root in receipt["roots"].values():
        require(not output.is_relative_to(Path(root)), "output must not be inside an input")
    need = 4 * sum(math.prod(s) for s in receipt["inventory"].values()) + 1024**3
    require(shutil.disk_usage(output.parent).free >= need, "insufficient disk: output plus 1 GiB margin required")
    output.mkdir(exist_ok=False)
    marker = output / "INCOMPLETE.json"
    marker.write_text(json.dumps({"experiment": EXPERIMENT, "arm": arm}), encoding="utf-8")
    try:
        left, right, alpha = receipt["pair"]
        result = stream_pair(Path(receipt["roots"][left]) / "model.safetensors",
                             Path(receipt["roots"][right]) / "model.safetensors",
                             output / "model.safetensors.partial", alpha)
        base = Path(receipt["roots"]["supa"])
        for name in ASSETS:
            shutil.copyfile(base / name, output / name)
            require(sha256(output / name) == ASSETS[name], "copied sidecar mismatch")
        config = read_json(base / "config.json")
        config["torch_dtype"] = "float32"
        if "dtype" in config:
            config["dtype"] = "float32"
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        recheck(receipt)
        require(sha256(manifest) == receipt["manifest_sha256"], "manifest changed during build")
        (output / "model.safetensors.partial").rename(output / "model.safetensors")
        receipt.update(result)
        receipt.update({"arm": arm, "status": "MATERIALIZED_NOT_EVALUATED", "output": str(output),
                        "sources_rehashed_after_build": True, "model_forwards": 0, "optimizer_steps": 0})
        # Source paths can be private; receipts stay local unless deliberately published.
        (output / "BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        marker.unlink()
        return receipt
    except Exception as exc:
        (output / "FAILED.json").write_text(json.dumps({"error": str(exc), "release_permitted": False}), encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS)
    p = sub.add_parser("merge")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = inspect_manifest(args.manifest, args.arm) if args.command == "preflight" else build(args.manifest, args.arm, args.output)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc), "release_permitted": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
