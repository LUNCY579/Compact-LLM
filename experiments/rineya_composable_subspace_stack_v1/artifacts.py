"""Guarded common-base checkpoint bridge and one-arm streaming builder.

Reuse the pinned mini-soup compatibility auditor, without relaxing its pins or
modifying its experiment registry. Never load an entire Rineya state dictionary.
"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import numpy as np
from . import EXPERIMENT, PARENT_SHA
from .core import Arm, compose, require, merge
from .protocol import (read_json, write_new, digest, file_hash, validate_plan,
                       admit, execution_lease, source_digest, control_recipes)

PARENT_SOUP_BLOB = "23101a32838c2f3defe27c7b2deb18864b1fb391"


def parent_bridge():
    path = Path(__file__).parents[1] / "titan_model_soup_mini_v1" / "soup.py"
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    require(blob == PARENT_SOUP_BLOB, "inherited soup.py differs from audited source; requalification required")
    spec = importlib.util.spec_from_file_location("_css_pinned_soup", path)
    require(spec is not None and spec.loader is not None, "parent import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(plan, manifest, grant):
    admit(plan, grant, "preflight")
    require(file_hash(manifest) == grant.get("manifest_sha256"), "manifest not grant-bound")
    with execution_lease(grant):
        parent = parent_bridge()
        result = parent.inspect_manifest(Path(manifest))
        validate_scope(plan, result)
        return {"experiment": EXPERIMENT, "status": "CHECKPOINT_COMPATIBILITY_ONLY",
                "parent_receipt": result, "plan_sha256": digest(plan),
                "native_model_qualified": False, "optimizer_steps": 0}


def validate_scope(plan, receipt):
    validate_plan(plan)
    require(set(plan["donors"]) <= set(receipt["roots"]), "missing donor endpoint")
    require(set(plan["tensors"]) <= set(receipt["inventory"]), "requested matrix missing from checkpoint")
    for name in plan["tensors"]:
        shape = receipt["inventory"][name]
        require(len(shape) == 2 and math.prod(shape) <= plan["max_tensor_elements"], "selected tensor exceeds budget")
        require(max(plan["settings"][k] for k in ("rank", "shared_rank", "protected_rank")) <= min(shape), "requested rank exceeds matrix shape")


def stream_matrices(paths, inventory, target, selected, transform, *, guard=lambda: None):
    """Low-level tiny-fixture-tested writer. Caller owns admission/provenance.

    transform(name, base, donor_deltas) -> (delta, diagnostics). Unselected
    parameters copy the base, NOT a donor soup. Output is standard FP32.
    """
    from safetensors import safe_open
    import torch
    target = Path(target)
    require(not target.exists(), "output exists")
    require(set(selected) <= set(inventory), "selected tensor missing")
    header, offset = {"__metadata__": {"format": "pt", "experiment": EXPERIMENT}}, 0
    require(inventory, "empty tensor inventory")
    for key, shape in sorted(inventory.items()):
        require(shape and all(type(n) is int and n > 0 for n in shape), "invalid tensor shape")
        size = math.prod(shape)*4
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset+size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    diagnostics = {}
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(path), framework="pt", device="cpu")) for path in paths]
        require(len(handles) >= 3, "base and two or more donors required")
        for h in handles:
            require(set(h.keys()) in (set(inventory), set(inventory) | {"lm_head.weight"}), "tensor inventory differs")
            require(all(h.get_slice(k).get_shape() == list(s) for k, s in inventory.items()), "tensor shape differs")
            require(all(h.get_slice(k).get_dtype() in ("F32", "F16", "BF16") for k in inventory), "unsupported source tensor dtype")
        with target.open("xb") as f:
            f.write(struct.pack("<Q", len(encoded)))
            f.write(encoded)
            for name, shape in sorted(inventory.items()):
                guard()
                if name in selected:
                    require(len(shape) == 2 and math.prod(shape) <= 4_194_304, "matrix size limit")
                    w = handles[0].get_tensor(name).float().numpy().astype(np.float64)
                    ds = [h.get_tensor(name).float().numpy().astype(np.float64)-w for h in handles[1:]]
                    delta, report = transform(name, w, ds)
                    require(delta.shape == w.shape and np.isfinite(delta).all(), "invalid transformed delta")
                    out = np.asarray(w+delta, dtype="<f4")
                    require(np.isfinite(out).all(), "FP32 materialization overflow")
                    actual = out.astype(np.float64)-w
                    report.update({"stored_delta_norm": float(np.linalg.norm(actual)),
                                   "fp32_rounding_error_norm": float(np.linalg.norm(actual-delta))})
                    diagnostics[name] = report
                    f.write(out.tobytes(order="C"))
                    del w, ds, delta, out, actual
                else:
                    row_size = math.prod(shape[1:])
                    step = max(1, 262144 // row_size)
                    for lo in range(0, shape[0], step):
                        x = handles[0].get_slice(name)[lo:min(lo+step, shape[0])].float()
                        require(bool(torch.isfinite(x).all()), "nonfinite base tensor")
                        f.write(x.numpy().astype("<f4", copy=False).tobytes(order="C"))
            f.flush()
            os.fsync(f.fileno())
    require(target.stat().st_size == 8+len(encoded)+offset, "serialized byte count mismatch")
    with safe_open(str(target), framework="pt", device="cpu") as check:
        require(set(check.keys()) == set(inventory), "roundtrip keys mismatch")
        for key, shape in inventory.items():
            require(check.get_slice(key).get_shape() == list(shape) and check.get_slice(key).get_dtype() == "F32", "roundtrip shape/dtype mismatch")
    return {"weight_sha256": file_hash(target), "weight_bytes": target.stat().st_size,
            "selected_matrices": diagnostics, "tensor_roundtrip_valid": True}


def build(plan, manifest, grant, arm_name, output):
    admit(plan, grant, "build", arm=arm_name)
    require(file_hash(manifest) == grant.get("manifest_sha256"), "manifest not grant-bound")
    output = Path(output).absolute()
    require(output.parent.resolve() == Path(grant["output_parent"]).resolve(), "output parent not granted")
    require(output.parent.is_dir() and not output.exists() and not output.is_symlink(), "fresh output required")
    arm = next((a for a in validate_plan(plan) if a.name == arm_name), None)
    control = control_recipes(plan).get(arm_name)
    with execution_lease(grant) as guard:
        parent = parent_bridge()
        receipt = parent.inspect_manifest(Path(manifest))
        validate_scope(plan, receipt)
        for root in receipt["roots"].values():
            require(not output.resolve().is_relative_to(Path(root).resolve()), "output inside source")
        need = 4*sum(math.prod(s) for s in receipt["inventory"].values()) + 1024**3
        require(shutil.disk_usage(output.parent).free >= need, "output plus 1 GiB disk margin required")
        output.mkdir(exist_ok=False)
        write_new(output / "INCOMPLETE.json", {"experiment": EXPERIMENT, "arm": arm_name})
        try:
            roots = receipt["roots"]
            paths = [Path(roots[n]) / "model.safetensors" for n in ["supa"]+plan["donors"]]
            selected = plan["tensors"]
            if control and control["kind"] == "endpoint":
                paths[0] = Path(roots[control["source"]]) / "model.safetensors"
                selected = []
            def transform(name, w, ds):
                if control:
                    return merge(ds, control["coefficients"]), {"control": control}
                return compose(w, ds, plan["coefficients"], arm, **plan["settings"])
            result = stream_matrices(paths, receipt["inventory"], output / "model.safetensors.partial", selected,
                                     transform, guard=guard)
            base = Path(roots["supa"])
            for name, expected in parent.ASSETS.items():
                shutil.copyfile(base / name, output / name)
                require(file_hash(output / name) == expected, "copied sidecar mismatch")
            config = read_json(base / "config.json")
            config["torch_dtype"] = "float32"
            if "dtype" in config:
                config["dtype"] = "float32"
            write_new(output / "config.json", config)
            parent.recheck(receipt)
            require(file_hash(manifest) == grant["manifest_sha256"], "manifest changed during build")
            (output / "model.safetensors.partial").rename(output / "model.safetensors")
            result.update({"schema": 1, "experiment": EXPERIMENT, "arm": arm_name,
                           "parent_sha": PARENT_SHA, "plan_sha256": digest(plan), "code_sha256": source_digest(),
                           "sources": receipt, "unselected_policy": "endpoint_copy" if control and control["kind"] == "endpoint" else "frozen_base",
                           "status": "MATERIALIZED_NOT_EVALUATED", "stored_dtype": "float32",
                           "native_model_qualified": False, "model_forwards": 0, "optimizer_steps": 0,
                           "release_permitted": False})
            write_new(output / "BUILD.json", result)
            (output / "INCOMPLETE.json").unlink()
            return result
        except Exception as exc:
            write_new(output / "FAILED.json", {"error": str(exc), "release_permitted": False})
            raise
