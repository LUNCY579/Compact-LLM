"""Create-only, tensor-streamed, source-bound training state; no pickle deserialization."""
from __future__ import annotations

import random
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .integrity import file_hash, new_dir, read_json, regular, seal, unseal, write_new


def rng_state() -> dict:
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state().tolist(),
            "torch_cuda": [x.tolist() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value
    random.setstate(tuples(state["python"]))
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    gpu = state["torch_cuda"]
    if len(gpu) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("checkpoint CUDA RNG inventory differs")
    if gpu:
        torch.cuda.set_rng_state_all([torch.tensor(x, dtype=torch.uint8) for x in gpu])


def save_checkpoint(directory: Path, optimizer, binding: dict, progress: dict) -> dict:
    if any(p.grad is not None for _, p in optimizer.named):
        raise ValueError("checkpoint requires an update boundary with cleared gradients")
    if progress["updates"] != optimizer.step_count:
        raise ValueError("optimizer/progress step mismatch")
    directory = new_dir(directory)
    entries = []
    for i, (name, p) in enumerate(optimizer.named):
        m, v = optimizer.state[name]
        state = {"parameter": p.detach().cpu().contiguous(), "exp_avg": m.contiguous(),
                 "exp_avg_sq": v.contiguous()}
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError("cannot checkpoint nonfinite tensors")
        filename = f"tensor-{i:04d}.safetensors"
        save_file(state, str(directory / filename))
        # Ensure durability before publishing the final COMPLETE manifest.
        with (directory / filename).open("rb+") as stream:
            import os
            os.fsync(stream.fileno())
        entries.append({"name": name, "file": filename, "shape": list(p.shape),
                        "sha256": file_hash(directory / filename)})
        del state
    body = {"schema": 1, "experiment": "MQ-001", "status": "COMPLETE",
            "binding": binding, "progress": progress, "rng": rng_state(), "entries": entries}
    manifest = seal(body)
    write_new(directory / "manifest.json", manifest)
    return manifest


def validate_checkpoint(directory: Path, named, binding: dict) -> dict:
    directory = Path(directory)
    meta = unseal(read_json(directory / "manifest.json"))
    if (set(meta) != {"schema", "experiment", "status", "binding", "progress", "rng", "entries"}
            or meta["schema"] != 1 or meta["experiment"] != "MQ-001"
            or meta["status"] != "COMPLETE" or meta["binding"] != binding):
        raise ValueError("wrong checkpoint scope, source, recipe or runtime")
    entries = meta["entries"]
    if [e["name"] for e in entries] != [n for n, _ in named]:
        raise ValueError("checkpoint tensor-name inventory mismatch")
    expected_files = {"manifest.json"} | {f"tensor-{i:04d}.safetensors" for i in range(len(entries))}
    if {p.name for p in directory.iterdir()} != expected_files:
        raise ValueError("checkpoint has extra or missing files")
    # Validate every tensor before modifying any model parameter or optimizer moment.
    for i, (entry, (_, p)) in enumerate(zip(entries, named, strict=True)):
        if (entry["file"] != f"tensor-{i:04d}.safetensors" or entry["shape"] != list(p.shape)
                or file_hash(directory / entry["file"]) != entry["sha256"]):
            raise ValueError("checkpoint path/shape/hash mismatch")
        state = load_file(str(regular(directory / entry["file"])), device="cpu")
        if set(state) != {"parameter", "exp_avg", "exp_avg_sq"}:
            raise ValueError("checkpoint state key mismatch")
        if any(t.dtype != torch.float32 or t.shape != p.shape or not torch.isfinite(t).all()
               for t in state.values()) or (state["exp_avg_sq"] < 0).any():
            raise ValueError("invalid checkpoint tensor")
    return meta


def restore_checkpoint(directory: Path, optimizer, binding: dict) -> dict:
    meta = validate_checkpoint(directory, optimizer.named, binding)
    progress = meta["progress"]
    if type(progress.get("updates")) is not int or progress["updates"] < 0:
        raise ValueError("invalid checkpoint update count")
    # Check RNG validity without changing the caller's random streams.
    original = rng_state()
    try:
        restore_rng(meta["rng"])
    finally:
        restore_rng(original)
    with torch.no_grad():
        for entry, (name, p) in zip(meta["entries"], optimizer.named, strict=True):
            state = load_file(str(directory / entry["file"]), device="cpu")
            p.copy_(state["parameter"])
            m, v = optimizer.state[name]
            m.copy_(state["exp_avg"]); v.copy_(state["exp_avg_sq"])
    optimizer.step_count = progress["updates"]
    optimizer.zero_grad()
    restore_rng(meta["rng"])
    return progress
