"""Offline FP32 skill-vector composition for Titan-Sparse and Opallo.

This experiment is deliberately separate from SOUP-MINI-001.  It reconstructs
both updates against the same stored Supa parent, upcast once to FP32:

    W0 = FP32(Supa_BF16)
    dS = TitanSparse_FP32 - W0
    dO = Opallo_trained_FP32 - W0       (on the 220 trained tensors)

The Opallo published BF16 checkpoint is used only for an independent cast
round-trip check.  No model class, tokenizer, generation, training, or native
evaluation is used here.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
from collections import defaultdict
from typing import Any

import torch
from safetensors import safe_open

EXPERIMENT = "TITAN-OPALLO-SKILL-VECTOR-001"
CHUNK_ELEMENTS = 262144
ASSETS = ("config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "vocab.json", "merges.txt")

ARMS = {
    "BASE": (0.0, 0.0),
    "SPARSE": (1.0, 0.0),
    "OPALLO": (0.0, 1.0),
    "AVG": (0.5, 0.5),
    "ADD-100": (1.0, 1.0),
    "ADD-S75": (0.75, 1.0),
    "ADD-O75": (1.0, 0.75),
    "ADD-75": (0.75, 0.75),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def config_shape(path: Path) -> dict[str, Any]:
    cfg = read_json(path)
    expected = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        require(cfg.get(key) == value, f"config mismatch at {key}: {path}")
    return cfg


def qwen_to_ced(name: str) -> str:
    """Inverse of the exact native map used by the Opallo training harness."""
    if name == "model.embed_tokens.weight":
        return "embed_tokens.weight"
    if name == "model.norm.weight":
        return "final_norm.weight"
    if not name.startswith("model.layers."):
        raise KeyError(name)
    _, _, rest = name.partition("model.layers.")
    layer, _, tail = rest.partition(".")
    prefix = f"encoder.{layer}."
    mapping = {
        "self_attn.q_proj.weight": "q_proj.weight",
        "self_attn.k_proj.weight": "k_proj.weight",
        "self_attn.v_proj.weight": "v_proj.weight",
        "self_attn.o_proj.weight": "o_proj.weight",
        "self_attn.q_norm.weight": "q_norm.weight",
        "self_attn.k_norm.weight": "k_norm.weight",
        "input_layernorm.weight": "input_norm.weight",
        "post_attention_layernorm.weight": "post_norm.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
    }
    require(tail in mapping, f"unmapped Qwen tensor: {name}")
    return prefix + mapping[tail]


def inventory(handle) -> dict[str, list[int]]:
    keys = set(handle.keys())
    require("model.embed_tokens.weight" in keys, "missing embedding")
    if "lm_head.weight" in keys:
        emb = handle.get_tensor("model.embed_tokens.weight")
        head = handle.get_tensor("lm_head.weight")
        require(torch.equal(emb, head), "tied lm_head is not equal to embeddings")
        keys.remove("lm_head.weight")
    result = {name: list(handle.get_slice(name).get_shape()) for name in sorted(keys)}
    require(len(result) == 310, f"unexpected tensor count: {len(result)}")
    require(sum(math.prod(shape) for shape in result.values()) == 596049920,
            "unexpected serialized parameter count")
    return result


def slices(shape: list[int]):
    row_elems = math.prod(shape[1:]) if len(shape) > 1 else 1
    step = max(1, CHUNK_ELEMENTS // row_elems)
    for start in range(0, shape[0], step):
        yield start, min(start + step, shape[0])


def validate_inputs(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    require(manifest.get("experiment") == EXPERIMENT, "wrong experiment")
    files = manifest.get("files")
    require(isinstance(files, dict), "manifest.files must be an object")
    required = {"base", "sparse", "opallo_published", "titan_trained", "opallo_trained"}
    require(set(files) == required, f"manifest files must be exactly {sorted(required)}")
    checked = {}
    for key, item in files.items():
        require(isinstance(item, dict), f"bad manifest entry: {key}")
        path = Path(item["path"]).expanduser().resolve()
        expected = item["sha256"]
        require(path.is_file() and not path.is_symlink(), f"missing input: {path}")
        actual = sha256(path)
        require(actual == expected, f"hash mismatch for {key}: {actual} != {expected}")
        checked[key] = path

    base_root = checked["base"].parent
    sparse_root = checked["sparse"].parent
    opallo_root = checked["opallo_published"].parent
    for root in (base_root, sparse_root, opallo_root):
        config_shape(root / "config.json")
        for asset in ASSETS:
            require((root / asset).is_file(), f"missing sidecar {asset}: {root}")
    with ExitStack() as stack:
        base = stack.enter_context(safe_open(checked["base"], framework="pt", device="cpu"))
        sparse = stack.enter_context(safe_open(checked["sparse"], framework="pt", device="cpu"))
        opublished = stack.enter_context(safe_open(checked["opallo_published"], framework="pt", device="cpu"))
        inv = inventory(base)
        require(inv == inventory(sparse) == inventory(opublished), "full tensor inventory mismatch")
        trained_t = stack.enter_context(safe_open(checked["titan_trained"], framework="pt", device="cpu"))
        trained_o = stack.enter_context(safe_open(checked["opallo_trained"], framework="pt", device="cpu"))
        require(all(trained_t.get_slice(k).get_dtype() == "F32" for k in trained_t.keys()),
                "Titan trained tensors are not FP32")
        require(all(trained_o.get_slice(k).get_dtype() == "F32" for k in trained_o.keys()),
                "Opallo trained tensors are not FP32")
        expected_o = {qwen_to_ced(k): k for k in inv if qwen_to_ced(k) in trained_o.keys()}
        require(len(expected_o) == len(trained_o.keys()) == 220,
                f"Opallo trained inventory mismatch: {len(expected_o)} / {len(trained_o.keys())}")
        require(len(trained_t.keys()) == 220, "Titan trained inventory mismatch")
        for qwen, shape in inv.items():
            ced = qwen_to_ced(qwen)
            if ced in trained_o.keys():
                require(list(trained_o.get_slice(ced).get_shape()) == shape, f"Opallo shape mismatch: {qwen}")
            if ced in trained_t.keys():
                require(list(trained_t.get_slice(ced).get_shape()) == shape, f"Titan shape mismatch: {qwen}")
    return {"manifest": manifest, "files": checked, "inventory": inv}


def _stats(ds: torch.Tensor, do: torch.Tensor) -> dict[str, Any]:
    ds = ds.float()
    do = do.float()
    nzs = ds != 0
    nzo = do != 0
    both = nzs & nzo
    dot = float((ds.double() * do.double()).sum().item())
    ns2 = float((ds.double() * ds.double()).sum().item())
    no2 = float((do.double() * do.double()).sum().item())
    ns = math.sqrt(ns2)
    no = math.sqrt(no2)
    return {
        "elements": int(ds.numel()),
        "nnz_sparse": int(nzs.sum().item()),
        "nnz_opallo": int(nzo.sum().item()),
        "overlap_nnz": int(both.sum().item()),
        "sign_agreement": (float(((ds[both] * do[both]) > 0).sum().item()) / int(both.sum().item())
                            if int(both.sum().item()) else None),
        "sparse_l2": ns,
        "opallo_l2": no,
        "dot": dot,
        "cosine": dot / (ns * no) if ns and no else None,
        "abs_dot": abs(dot),
    }


def audit(manifest_path: Path, output: Path) -> dict[str, Any]:
    receipt = validate_inputs(manifest_path)
    inv = receipt["inventory"]
    files = receipt["files"]
    per_tensor = []
    layers = defaultdict(lambda: {"sparse_l2_sq": 0.0, "opallo_l2_sq": 0.0,
                                  "dot": 0.0, "abs_dot": 0.0, "nnz_sparse": 0,
                                  "nnz_opallo": 0, "overlap_nnz": 0, "elements": 0})
    opallo_roundtrip_total = 0
    opallo_roundtrip_equal = 0
    with ExitStack() as stack:
        base = stack.enter_context(safe_open(files["base"], framework="pt", device="cpu"))
        sparse = stack.enter_context(safe_open(files["sparse"], framework="pt", device="cpu"))
        opublished = stack.enter_context(safe_open(files["opallo_published"], framework="pt", device="cpu"))
        trained_o = stack.enter_context(safe_open(files["opallo_trained"], framework="pt", device="cpu"))
        for name in sorted(inv):
            base_raw = base.get_tensor(name)
            base32 = base_raw.float()
            sparse32 = sparse.get_tensor(name).float()
            ced = qwen_to_ced(name)
            if ced in trained_o.keys():
                op32 = trained_o.get_tensor(ced).float()
                expected_bf16 = op32.to(torch.bfloat16)
            else:
                op32 = base32
                expected_bf16 = base_raw
            published = opublished.get_tensor(name)
            opallo_roundtrip_total += int(published.numel())
            opallo_roundtrip_equal += int((published == expected_bf16).sum().item())
            ds = sparse32 - base32
            do = op32 - base32
            row = _stats(ds, do)
            row["name"] = name
            row["layer"] = int(name.split(".")[2]) if name.startswith("model.layers.") else None
            row["family"] = name.split(".")[-3] if ".mlp." in name else name.split(".")[-2]
            per_tensor.append(row)
            layer = str(row["layer"] if row["layer"] is not None else "global")
            agg = layers[layer]
            agg["sparse_l2_sq"] += row["sparse_l2"] ** 2
            agg["opallo_l2_sq"] += row["opallo_l2"] ** 2
            agg["dot"] += row["dot"]
            agg["abs_dot"] += row["abs_dot"]
            for key in ("nnz_sparse", "nnz_opallo", "overlap_nnz", "elements"):
                agg[key] += row[key]

    sparse_l2 = math.sqrt(sum(r["sparse_l2"] ** 2 for r in per_tensor))
    opallo_l2 = math.sqrt(sum(r["opallo_l2"] ** 2 for r in per_tensor))
    dot = sum(r["dot"] for r in per_tensor)
    layer_rows = []
    for layer, row in layers.items():
        ns = math.sqrt(row["sparse_l2_sq"])
        no = math.sqrt(row["opallo_l2_sq"])
        layer_rows.append({
            "layer": layer, **row, "sparse_l2": ns, "opallo_l2": no,
            "cosine": row["dot"] / (ns * no) if ns and no else None,
        })
    conflicts = sorted(
        (r["name"] for r in per_tensor if r["cosine"] is not None and r["cosine"] <= -0.20)
    )
    result = {
        "schema": 1,
        "experiment": EXPERIMENT,
        "status": "GEOMETRY_ONLY_NATIVE_EVALUATION_NOT_AUTHORIZED",
        "formula": "W0 + alpha*dSparse + beta*dOpallo",
        "parent": "FP32 view of the exact stored Supa BF16 bytes; no hidden unrounded Supa checkpoint was substituted",
        "manifest_sha256": sha256(manifest_path),
        "input_hashes": {key: sha256(path) for key, path in files.items()},
        "global": {
            "sparse_l2": sparse_l2,
            "opallo_l2": opallo_l2,
            "dot": dot,
            "cosine": dot / (sparse_l2 * opallo_l2),
            "absolute_interference": sum(r["abs_dot"] for r in per_tensor) / (sparse_l2 * opallo_l2),
            "nnz_sparse": sum(r["nnz_sparse"] for r in per_tensor),
            "nnz_opallo": sum(r["nnz_opallo"] for r in per_tensor),
            "overlap_nnz": sum(r["overlap_nnz"] for r in per_tensor),
        },
        "opallo_roundtrip": {
            "published_bf16_elements": opallo_roundtrip_total,
            "bitwise_equal_after_fp32_to_bf16": opallo_roundtrip_equal,
            "mismatches": opallo_roundtrip_total - opallo_roundtrip_equal,
        },
        "conflict_rule": "conflict-local attenuates Opallo to 0.5 only where per-tensor cosine <= -0.20",
        "conflict_tensors": conflicts,
        "per_tensor": per_tensor,
        "per_layer": sorted(layer_rows, key=lambda r: (r["layer"] == "global", str(r["layer"]))),
        "native_boundary": {
            "model_loads": 0, "forwards": 0, "scored_rows": 0,
            "native_authorized": False, "quality_claim": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def tensor_for_arm(name: str, base32: torch.Tensor, sparse32: torch.Tensor,
                   op32: torch.Tensor, alpha: float, beta: float,
                   conflict: bool) -> torch.Tensor:
    if conflict:
        beta = 0.5
    # Preserve exact endpoints.  Recomputing an endpoint as
    # base + 0*(sparse-base) + 1*(opallo-base) adds needless roundoff and
    # makes the named donor arm differ from its source bytes.
    if alpha == 0.0 and beta == 0.0:
        return base32.clone()
    if alpha == 1.0 and beta == 0.0:
        return sparse32.clone()
    if alpha == 0.0 and beta == 1.0:
        return op32.clone()
    return base32 + alpha * (sparse32 - base32) + beta * (op32 - base32)


def build_arm(manifest_path: Path, geometry_path: Path, arm: str, output: Path) -> dict[str, Any]:
    require(arm in ARMS or arm == "CONFLICT-LOCAL", f"unknown arm: {arm}")
    receipt = validate_inputs(manifest_path)
    geometry = read_json(geometry_path)
    conflict_names = set(geometry.get("conflict_tensors", []))
    alpha, beta = ARMS.get(arm, (1.0, 1.0))
    output = output.expanduser().resolve()
    require(not output.exists() and not output.is_symlink(), f"output exists: {output}")
    require(output.parent.is_dir(), "output parent must already exist")
    require(shutil.disk_usage(output.parent).free >= 4 * 596049920 + 1024**3,
            "need output space plus 1 GiB margin")
    output.mkdir()
    marker = output / "INCOMPLETE.json"
    marker.write_text(json.dumps({"experiment": EXPERIMENT, "arm": arm}), encoding="utf-8")
    files = receipt["files"]
    inv = receipt["inventory"]
    try:
        with ExitStack() as stack:
            base = stack.enter_context(safe_open(files["base"], framework="pt", device="cpu"))
            sparse = stack.enter_context(safe_open(files["sparse"], framework="pt", device="cpu"))
            trained_o = stack.enter_context(safe_open(files["opallo_trained"], framework="pt", device="cpu"))
            header = {"__metadata__": {"format": "pt", "experiment": EXPERIMENT, "arm": arm}}
            offset = 0
            for name, shape in inv.items():
                nbytes = math.prod(shape) * 4
                header[name] = {"dtype": "F32", "shape": shape,
                                "data_offsets": [offset, offset + nbytes]}
                offset += nbytes
            encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            encoded += b" " * (-len(encoded) % 8)
            partial = output / "model.safetensors.partial"
            with partial.open("xb") as out:
                out.write(struct.pack("<Q", len(encoded)))
                out.write(encoded)
                for name, shape in inv.items():
                    base32 = base.get_tensor(name).float()
                    sparse32 = sparse.get_tensor(name).float()
                    ced = qwen_to_ced(name)
                    op32 = trained_o.get_tensor(ced).float() if ced in trained_o.keys() else base32
                    value = tensor_for_arm(name, base32, sparse32, op32, alpha, beta,
                                           name in conflict_names and arm == "CONFLICT-LOCAL")
                    require(bool(torch.isfinite(value).all()), f"nonfinite output: {name}")
                    out.write(value.contiguous().numpy().astype("<f4", copy=False).tobytes(order="C"))
                out.flush()
                os.fsync(out.fileno())
            require(partial.stat().st_size == 8 + len(encoded) + offset,
                    "output byte count mismatch")
            for asset in ASSETS:
                shutil.copy2(files["base"].parent / asset, output / asset)
            cfg = read_json(files["base"].parent / "config.json")
            cfg["torch_dtype"] = "float32"
            output.joinpath("config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            partial.rename(output / "model.safetensors")
        build = {
            "schema": 1, "experiment": EXPERIMENT, "arm": arm,
            "formula": "W0 + alpha*dSparse + beta*dOpallo",
            "alpha_sparse": alpha, "beta_opallo": beta,
            "conflict_tensors": sorted(conflict_names) if arm == "CONFLICT-LOCAL" else [],
            "status": "MATERIALIZED_NOT_EVALUATED",
            "native_model_forwards": 0, "scored_rows": 0,
            "weight_sha256": sha256(output / "model.safetensors"),
            "weight_bytes": (output / "model.safetensors").stat().st_size,
            "input_hashes": {key: sha256(path) for key, path in files.items()},
            "quality_claim": False, "release_permitted": False,
        }
        (output / "BUILD.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
        marker.unlink()
        return build
    except Exception as exc:
        (output / "FAILED.json").write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        raise


def verify_outputs(manifest_path: Path, arms_root: Path) -> dict[str, Any]:
    """Reload every materialized arm and verify inventory, dtype, and finiteness."""
    receipt = validate_inputs(manifest_path)
    inv = receipt["inventory"]
    dirs = sorted(p for p in arms_root.iterdir() if p.is_dir())
    require(len(dirs) == len(ARMS) + 1, f"expected {len(ARMS) + 1} arms, found {len(dirs)}")
    rows = []
    exact = {"BASE": 0, "SPARSE": 0, "OPALLO": 0}
    with ExitStack() as stack:
        base = stack.enter_context(safe_open(receipt["files"]["base"], framework="pt", device="cpu"))
        sparse = stack.enter_context(safe_open(receipt["files"]["sparse"], framework="pt", device="cpu"))
        trained_o = stack.enter_context(safe_open(receipt["files"]["opallo_trained"], framework="pt", device="cpu"))
        for directory in dirs:
            build = read_json(directory / "BUILD.json")
            arm = build.get("arm")
            require(arm in ARMS or arm == "CONFLICT-LOCAL", f"unknown output arm: {directory}")
            require(build.get("status") == "MATERIALIZED_NOT_EVALUATED", f"bad status: {directory}")
            model_path = directory / "model.safetensors"
            require(model_path.is_file(), f"missing model: {model_path}")
            nonfinite = 0
            with safe_open(model_path, framework="pt", device="cpu") as out:
                require(inventory(out) == inv, f"inventory mismatch: {arm}")
                for name in inv:
                    require(out.get_slice(name).get_dtype() == "F32", f"non-F32 output: {arm}/{name}")
                    value = out.get_tensor(name)
                    nonfinite += int((~torch.isfinite(value)).sum().item())
                if arm == "BASE":
                    for name in inv:
                        exact["BASE"] += int(torch.equal(out.get_tensor(name), base.get_tensor(name).float()))
                elif arm == "SPARSE":
                    for name in inv:
                        exact["SPARSE"] += int(torch.equal(out.get_tensor(name), sparse.get_tensor(name).float()))
                elif arm == "OPALLO":
                    for name in inv:
                        ced = qwen_to_ced(name)
                        expected = trained_o.get_tensor(ced).float() if ced in trained_o.keys() else base.get_tensor(name).float()
                        exact["OPALLO"] += int(torch.equal(out.get_tensor(name), expected))
            require(nonfinite == 0, f"nonfinite values in {arm}: {nonfinite}")
            rows.append({"arm": arm, "weight_sha256": sha256(model_path),
                         "weight_bytes": model_path.stat().st_size, "nonfinite": nonfinite})
    require(exact["BASE"] == len(inv), f"BASE endpoint mismatch: {exact['BASE']}/{len(inv)} tensors")
    require(exact["SPARSE"] == len(inv), f"SPARSE endpoint mismatch: {exact['SPARSE']}/{len(inv)} tensors")
    require(exact["OPALLO"] == len(inv), f"OPALLO endpoint mismatch: {exact['OPALLO']}/{len(inv)} tensors")
    return {"schema": 1, "experiment": EXPERIMENT, "status": "ALL_ARMS_RELOADED_AND_VERIFIED",
            "arms": rows, "exact_endpoint_tensor_checks": exact,
            "native_model_forwards": 0, "scored_rows": 0, "quality_claim": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("build")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--arms-root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            result = audit(args.manifest.resolve(), args.output.resolve())
        elif args.command == "build":
            result = build_arm(args.manifest.resolve(), args.geometry.resolve(), args.arm,
                               args.output.resolve())
        else:
            result = verify_outputs(args.manifest.resolve(), args.arms_root.resolve())
            if args.output:
                args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in result if k in {
            "status", "experiment", "arm", "weight_sha256", "cosine",
            "absolute_interference", "conflict_tensors", "opallo_roundtrip"}}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc),
                          "quality_claim": False}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
