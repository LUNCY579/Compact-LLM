"""Non-pickle, no-overwrite skill-boundary checkpoints with explicit masks."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from . import BRANCH, EXPERIMENT, PARENT_COMMIT
from .allocation import Allocation, checked_mask
from .data import digest_json, file_hash, read_json


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(value.dtype), list(value.shape))).encode())
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")


def validate_identity(identity):
    for key in ("model_id", "tokenizer_id", "model_sha256", "tokenizer_sha256", "git_commit", "runtime"):
        if key not in identity or not identity[key]:
            raise ValueError(f"unbound identity: {key}")
    for key, length in (("model_sha256", 64), ("tokenizer_sha256", 64), ("git_commit", 40)):
        if not isinstance(identity[key], str) or len(identity[key]) != length or any(c not in "0123456789abcdef" for c in identity[key]):
            raise ValueError(f"invalid identity hash: {key}")


def save_checkpoint(runner, target, identity):
    validate_identity(identity)
    if not torch.equal(runner.coords.vector(), runner.current):
        raise ValueError("model drift before checkpoint")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    tensors = {"base": runner.base, "current": runner.current}
    records = []
    for i, rec in enumerate(runner.records):
        skill = rec["skill"]
        tensors[f"mask_{i}"] = rec["mask"]
        tensors[f"claimed_{i}"] = runner.allocation.masks[skill]
        tensors[f"delta_{i}"] = runner.deltas[skill]
        plain = {k: v for k, v in rec.items() if k != "mask"}
        plain.update({"mask_sha256": tensor_hash(rec["mask"]), "delta_sha256": tensor_hash(runner.deltas[skill])})
        records.append(plain)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(target / "state.safetensors"))
    manifest = {"schema": 1, "experiment": EXPERIMENT, "branch": BRANCH, "parent_commit": PARENT_COMMIT,
                "identity": identity, "config": runner.config.payload(), "config_sha256": runner.config.digest,
                "dataset_sha256": runner.data_digest, "coordinate_names": list(runner.coords.names),
                "coordinate_shapes": runner.coords.shapes, "total_eligible": runner.coords.size,
                "state_sha256": file_hash(target / "state.safetensors"), "records": records,
                "checkpoint_scope": "skill_boundary_only", "native_training_qualified": False}
    write_new_json(target / "manifest.json", manifest)
    return manifest


def load_checkpoint(runner, target, expected_identity):
    validate_identity(expected_identity)
    target = Path(target)
    m = read_json(target / "manifest.json")
    if (m.get("schema"), m.get("experiment"), m.get("parent_commit"), m.get("branch")) != (1, EXPERIMENT, PARENT_COMMIT, BRANCH):
        raise ValueError("checkpoint lineage/schema mismatch")
    if m["identity"] != expected_identity or m["config_sha256"] != runner.config.digest or digest_json(m["config"]) != runner.config.digest or m["dataset_sha256"] != runner.data_digest:
        raise ValueError("checkpoint binding mismatch")
    if m["coordinate_names"] != list(runner.coords.names) or m["coordinate_shapes"] != runner.coords.shapes or m["total_eligible"] != runner.coords.size:
        raise ValueError("coordinate layout mismatch")
    if file_hash(target / "state.safetensors") != m["state_sha256"]:
        raise ValueError("checkpoint bytes changed")
    tensors = load_file(str(target / "state.safetensors"), device="cpu")
    count = len(m["records"])
    expected_keys = {"base", "current"} | {f"{p}_{i}" for p in ("mask", "claimed", "delta") for i in range(count)}
    if set(tensors) != expected_keys or count > len(runner.config.order) or not torch.equal(tensors["base"], runner.base):
        raise ValueError("checkpoint inventory/base mismatch")
    allocation = Allocation(runner.coords.size, runner.config.order)
    current, deltas, records = runner.base.clone(), {}, []
    for i, plain in enumerate(m["records"]):
        skill = runner.config.order[i]
        mask, claimed, delta = (tensors[f"{p}_{i}"] for p in ("mask", "claimed", "delta"))
        checked_mask(mask, runner.coords.size)
        checked_mask(claimed, runner.coords.size)
        if plain["skill"] != skill or plain["mask_sha256"] != tensor_hash(mask) or plain["delta_sha256"] != tensor_hash(delta):
            raise ValueError("skill checksum/order mismatch")
        if delta.shape != current.shape or delta.dtype != torch.float32 or not bool(torch.isfinite(delta).all()) or bool(delta[~mask].any()):
            raise ValueError("invalid stored skill delta")
        if runner.config.arm in ("random_disjoint", "learned_disjoint") and not torch.equal(mask, claimed):
            raise ValueError("strict mask differs from allocation ledger")
        from .allocation import budget
        k = budget(int(allocation.free.sum()), runner.config.rho)
        if plain["capacity_budget"] != k or plain["selected_coordinates"] != int(mask.sum()) or plain["active_nonzero"] != int((delta != 0).sum()):
            raise ValueError("checkpoint accounting mismatch")
        allocation.claim(skill, claimed, k)
        current += delta
        deltas[skill] = delta.clone()
        records.append({**plain, "mask": mask.clone()})
    if not torch.equal(current, tensors["current"]):
        raise ValueError("accumulated state mismatch")
    # Mutate runner only after every validation passes.
    runner.coords.restore(current)
    runner.current, runner.deltas, runner.records, runner.allocation = current, deltas, records, allocation
    return m
