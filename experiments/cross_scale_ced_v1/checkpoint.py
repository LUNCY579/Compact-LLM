"""Create-only, source-bound interface checkpoints; safetensors only, never pickle."""
from __future__ import annotations
from pathlib import Path
import torch
from .guards import new_directory, write_json_new, read_json, sha256_file, regular_file, digest


def interface_state(model):
    return {k: v for k, v in model.state_dict().items() if k.startswith(("memory.", "reads."))}


def save_interface(model, directory: Path, binding: dict) -> dict:
    from safetensors.torch import save_file
    if not model.read_enabled:
        raise ValueError("checkpoints must describe the normal read-enabled architecture")
    state = interface_state(model)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("cannot save nonfinite interface")
    directory = new_directory(directory)
    tensors = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    save_file(tensors, str(directory/"interface.safetensors"))
    metadata = dict(schema=1, experiment="XSC-001", kind="interface_only",
                    config=model.config.to_dict(), binding=binding,
                    binding_sha256=digest(binding), weights_sha256=sha256_file(directory/"interface.safetensors"),
                    inventory={k: [list(v.shape), str(v.dtype)] for k, v in tensors.items()})
    write_json_new(directory/"manifest.json", metadata)
    return metadata


def load_interface(model, directory: Path, binding: dict):
    from safetensors.torch import load_file
    directory = Path(directory)
    for name in ("manifest.json", "interface.safetensors"):
        regular_file(directory/name)
    m = read_json(directory/"manifest.json")
    if set(m) != {"schema", "experiment", "kind", "config", "binding", "binding_sha256", "weights_sha256", "inventory"}:
        raise ValueError("invalid interface manifest schema")
    if (m["schema"] != 1 or m["experiment"] != "XSC-001" or m["kind"] != "interface_only"
            or m["config"] != model.config.to_dict() or m["binding"] != binding
            or m["binding_sha256"] != digest(binding) or not model.read_enabled):
        raise ValueError("wrong interface architecture, source or provenance binding")
    if m["weights_sha256"] != sha256_file(directory/"interface.safetensors"):
        raise ValueError("interface file checksum mismatch")
    state = load_file(str(directory/"interface.safetensors"), device="cpu")
    expected = interface_state(model)
    if set(state) != set(expected) or set(m["inventory"]) != set(expected):
        raise ValueError("interface key inventory mismatch")
    for k, v in state.items():
        if (v.shape != expected[k].shape or v.dtype != expected[k].dtype or not torch.isfinite(v).all()
                or m["inventory"][k] != [list(v.shape), str(v.dtype)]):
            raise ValueError(f"interface tensor mismatch: {k}")
    # Validate the entire file before modifying even one parameter.
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k in state:
                p.copy_(state[k].to(p.device))
    return m
