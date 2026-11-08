"""FP32 magnitude masking and bounded tensor-at-a-time subspace experiments.

The incumbent is W0 + M_0.25(Delta), NOT W0 + 0.25*Delta.
Primary compression acts on that MASKED update. Full-delta SVD is a distinct axis.
No model is loaded or called by this module; construction is not qualification.
"""
from __future__ import annotations
import math
from pathlib import Path
from .integrity import ContractError, digest, file_hash, new_directory, require, write_new


def _tensor(value):
    import torch
    require(isinstance(value, torch.Tensor), "expected a tensor")
    require(value.dtype == torch.float32, "FP32 required; BF16 loses part of the measured update")
    require(value.device.type == "cpu", "construction is CPU-only")
    require(value.numel() > 0 and bool(torch.isfinite(value).all()), "empty/nonfinite tensor")
    return value.detach().contiguous()


def magnitude_mask(delta, keep: float = 0.25):
    """Exact SPARSE-001 floor(n*keep), stable flat-index tie breaking."""
    import torch
    d = _tensor(delta)
    require(type(keep) in (int, float) and math.isfinite(keep) and 0 <= keep <= 1,
            "keep must be finite and in [0,1]")
    flat = d.reshape(-1)
    count = math.floor(flat.numel() * keep)
    result = torch.zeros_like(flat)
    if count:
        positions = torch.argsort(flat.abs(), descending=True, stable=True)[:count]
        result[positions] = flat[positions]
    return result.reshape(d.shape)


def truncate(delta, rank: int):
    """Return represented FP32 delta, factors, and a norm-matched full-direction control.

    Dense SVD is a bounded reference implementation, not a fast approximation.
    FP64 analysis avoids introducing an additional low-precision decomposition arm.
    """
    import torch
    d = _tensor(delta)
    require(d.ndim == 2 and type(rank) is int and 1 <= rank <= min(d.shape),
            "rank must fit a nonempty matrix")
    u, s, vh = torch.linalg.svd(d.double(), full_matrices=False)
    left = (u[:, :rank] * s[:rank]).float().contiguous()
    right = vh[:rank].float().contiguous()
    represented = (left.double() @ right.double()).float()
    full_norm = float(torch.linalg.vector_norm(d.double()))
    ratio = float(torch.linalg.vector_norm(represented.double())) / full_norm if full_norm else 0.0
    control = d * ratio
    require(bool(torch.isfinite(represented).all()), "nonfinite reconstructed update")
    return represented, left, right, control, {
        "rank": rank, "norm_ratio": ratio,
        "factor_parameters": left.numel() + right.numel(),
        "factor_payload_bytes": (left.numel() + right.numel()) * 4,
        "dense_payload_bytes": d.numel() * 4,
        "analysis_dtype": "float64", "stored_dtype": "float32",
        "nonzero_full_update": int((d != 0).sum()),
        "nonzero_reconstructed_update": int((represented != 0).sum()),
        "note": "Truncation may destroy coordinate sparsity; no composability claim.",
    }


def transform(delta, arm: str, rank: int = 16, basis: str = "incumbent"):
    d = _tensor(delta)
    require(basis in ("incumbent", "full_delta"), "unknown compression basis")
    require(arm in ("source", "full", "sparse", "rank", "norm_control"), "unknown weight arm")
    if arm in ("rank", "norm_control"):
        require(type(rank) is int and rank > 0, "invalid rank")
    if arm == "source":
        return d * 0, {}, None
    if arm == "full":
        return d.clone(), {}, None
    masked = magnitude_mask(d)
    if arm == "sparse":
        return masked, {"keep_fraction": 0.25}, None
    working = masked if basis == "incumbent" else d
    if working.ndim != 2:
        return working.clone(), {"nonmatrix_policy": "preserve_basis_update"}, None
    effective_rank = min(rank, *working.shape)
    require(type(rank) is int and rank > 0, "invalid rank")
    low, left, right, control, stats = truncate(working, effective_rank)
    stats.update({"requested_rank": rank, "basis": basis})
    return (low if arm == "rank" else control), stats, (left, right)


def build_checkpoint(source: Path, trained: Path, out: Path, contract: dict, guard):
    """Make new one-tensor shards and an HF index from hash-locked safe tensors.

    Sidecars/native loading remain a separate qualification. The guard must be a
    real resource supervisor for nonfixture work; this API is not a launch permit.
    A failed build preserves its partial directory; it cannot be resumed/overwritten.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    require(callable(guard), "resource guard required")
    require(file_hash(source) == contract["source_sha256"], "source hash mismatch")
    require(file_hash(trained) == contract["trained_sha256"], "trained hash mismatch")
    allow = contract["trained_names"]
    require(allow and len(allow) == len(set(allow)), "exact unique trained allowlist required")
    require(type(contract["max_tensor_elements"]) is int and contract["max_tensor_elements"] > 0,
            "finite tensor bound required")
    require(contract["arm"] in ("source", "full", "sparse", "rank", "norm_control"), "invalid arm")
    with safe_open(str(source), framework="pt", device="cpu") as src, \
         safe_open(str(trained), framework="pt", device="cpu") as trn:
        require(set(trn.keys()) == set(allow), "trained artifact must match allowlist exactly")
        require(set(allow) <= set(src.keys()), "unknown trained tensor")
        # Validate all shapes before allocating output; do not silently skip a large tensor.
        for name in src.keys():
            shape = src.get_slice(name).get_shape()
            require(math.prod(shape) <= contract["max_tensor_elements"], "tensor exceeds resource bound")
            if name in allow:
                require(shape == trn.get_slice(name).get_shape(), "source/trained shape mismatch")
                require(trn.get_slice(name).get_dtype() == "F32", "trained update must be FP32")
        guard()
        new_directory(out)
        write_new(out / "CONTRACT.json", contract)
        weights, records, payload, factor_index = {}, [], 0, {}
        factor_dir = new_directory(out / "delta_factors") if contract["arm"] == "rank" else None
        for index, name in enumerate(sorted(src.keys())):
            guard()
            original = src.get_tensor(name).float().contiguous()
            require(bool(torch.isfinite(original).all()), "nonfinite source")
            stats, factors = {}, None
            if name in allow:
                delta = trn.get_tensor(name) - original
                update, stats, factors = transform(delta, contract["arm"], contract.get("rank", 16),
                                                    contract.get("basis", "incumbent"))
                result = original + update
            else:
                update, result = None, original
            require(bool(torch.isfinite(result).all()), "nonfinite constructed weights")
            filename = f"model-{index:05d}.safetensors"
            save_file({name: result.contiguous()}, str(out / filename))
            weights[name] = filename
            payload += result.numel() * 4
            if factor_dir is not None and update is not None:
                data = {"left": factors[0], "right": factors[1]} if factors else {"residual": update}
                factor_file = factor_dir / f"delta-{index:05d}.safetensors"
                save_file(data, str(factor_file))
                factor_index[name] = {"file": factor_file.name, "sha256": file_hash(factor_file),
                                      "representation": "left_right" if factors else "residual"}
            records.append({"name": name, "shape": list(result.shape), "file": filename,
                            "sha256": file_hash(out / filename), **stats})
            del original, result, update, factors
        write_new(out / "model.safetensors.index.json", {"metadata": {"total_size": payload},
                                                        "weight_map": weights})
    if factor_dir is not None:
        write_new(factor_dir / "INDEX.json", {"source_sha256": contract["source_sha256"],
            "contract_sha256": digest(contract), "tensors": factor_index,
            "reconstruction": "FP64 left@right then FP32, added to promoted source; runtime unqualified"})
    receipt = {"kind": "ALGEBRAIC_CONSTRUCTION", "contract_sha256": digest(contract),
               "model_calls": 0, "optimizer_steps": 0, "tensor_records": records,
               "checkpoint_file_bytes": sum(p.stat().st_size for p in out.glob("*.safetensors")),
               "factor_file_bytes": sum(p.stat().st_size for p in (factor_dir.glob("*") if factor_dir else [])),
               "factor_files_including_index": factor_index,
               "native_qualified": False, "quality_claim": False, "release_permitted": False,
               "remaining": ["sidecar identity", "native loader parity", "behavioral evaluation"]}
    write_new(out / "CONSTRUCTION.json", receipt)
    return receipt
