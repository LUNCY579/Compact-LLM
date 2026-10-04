"""Bounded, restorable SDPA interception. Local single-process research only."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import threading

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel

from .backends import SDPA, BACKENDS, dense_attention
from .core import Case

_LOCK = threading.Lock()


@contextmanager
def sdpa_session(arm: str, backend: str = "math", captures: list[Case] | None = None,
                 max_captures: int = 4, max_tokens: int = 256,
                 max_capture_bytes: int = 64 * 1024 * 1024,
                 metadata: dict | None = None,
                 capture_call_indices: tuple[int, ...] | None = None):
    """Capture exact represented operands and incoming dO without detaching the graph.

    No truncation/head slicing. Every intercepted call must fit the stated budget.
    No dropout, torch.compile, cache decoding, concurrent calls or external FA2/FA3.
    The forced backend is an explicit experiment, NOT proof of historical dispatch.
    """
    if arm not in {"native", "fp32", "cast_only", "gproj_q", "gproj"}:
        raise ValueError("Unknown arm")
    if backend not in BACKENDS or min(max_captures, max_tokens, max_capture_bytes) <= 0:
        raise ValueError("Invalid backend or budget")
    if capture_call_indices is not None and (not capture_call_indices or
            any(type(i) is not int or i < 1 for i in capture_call_indices)):
        raise ValueError("Capture indices must be positive SDPA call numbers")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Nested/concurrent SDPA sessions are not supported")
    original = F.scaled_dot_product_attention
    owner_thread = threading.get_ident()
    selected_backend = "math" if arm == "fp32" else backend
    state = {"calls": 0, "captured": 0, "reserved_bytes": 0, "arm": arm, "backend": selected_backend}
    handles = []

    def wrapped(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, *, scale=None, enable_gqa=False):
        if threading.get_ident() != owner_thread:
            raise RuntimeError("SDPA session called from another thread")
        if dropout_p != 0.0:
            raise ValueError("Dropout must be zero for paired numerical audits")
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("Native session requires cache-free square attention")
        if max(query.shape[-2], key.shape[-2]) > max_tokens:
            raise ValueError("Native sequence budget exceeded")
        if query.shape[1] != key.shape[1] and not enable_gqa:
            raise ValueError("Mismatched heads without enable_gqa")
        if query.shape[0] * query.shape[1] * query.shape[-2] * key.shape[-2] > 2_000_000:
            raise ValueError("Dense score budget exceeded before attention dispatch")
        state["calls"] += 1
        if arm in {"native", "fp32"}:
            args = (query, key, value) if arm == "native" else tuple(x.float() for x in (query, key, value))
            mask = attn_mask
            if mask is not None and mask.dtype != torch.bool and arm == "fp32":
                mask = mask.float()
            with sdpa_kernel(BACKENDS[selected_backend]), torch.autocast(query.device.type, enabled=False):
                out = SDPA(*args, attn_mask=mask, dropout_p=0.0, is_causal=is_causal,
                           scale=scale, enable_gqa=enable_gqa).to(query.dtype)
        else:
            out = dense_attention(query, key, value, attn_mask, is_causal, scale, arm)
        chosen = capture_call_indices is None or state["calls"] in capture_call_indices
        if captures is not None and chosen and state["captured"] < max_captures:
            if not out.requires_grad:
                raise ValueError("Capture needs a gradient-enabled attention output")
            tensors = [query, key, value, out]
            byte_count = sum(t.numel() * t.element_size() for t in tensors)
            byte_count += out.numel() * out.element_size()  # incoming derivative
            if attn_mask is not None:
                byte_count += attn_mask.numel() * attn_mask.element_size()
            if state["reserved_bytes"] + byte_count > max_capture_bytes:
                raise ValueError("Capture byte budget exceeded")
            state["reserved_bytes"] += byte_count
            state["captured"] += 1
            q, k, v, saved = (x.detach().cpu().clone() for x in tensors)
            mask = None if attn_mask is None else attn_mask.detach().cpu().clone()
            meta = dict(metadata or {}, call_index=state["calls"], backend_requested=selected_backend, arm=arm)
            def on_grad(u):
                case = Case(q, k, v, u.detach().cpu().clone(), mask, is_causal, scale, saved, meta)
                case.validate()
                captures.append(case)
                return u
            handles.append(out.register_hook(on_grad))
        return out

    try:
        if original is not SDPA:
            raise RuntimeError("SDPA already wrapped; refusing to overwrite another tool")
        F.scaled_dot_product_attention = wrapped
        yield state
    finally:
        F.scaled_dot_product_attention = original
        for handle in handles:
            handle.remove()
        _LOCK.release()


def save_capture(case: Case, path: Path) -> dict:
    case.validate()
    if path.exists() or path.with_suffix(path.suffix + ".json").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tensor-only payload; metadata has a separate inspectable sidecar.
    tensors = {k: v for k, v in vars(case).items() if isinstance(v, torch.Tensor)}
    with path.open("xb") as stream:
        torch.save(tensors, stream)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {"schema": 1, "sha256": digest, "causal": case.causal,
            "scale": case.scale, "metadata": case.metadata}
    with path.with_suffix(path.suffix + ".json").open("x") as stream:
        stream.write(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    return meta


def load_capture(path: Path, device: str = "cpu", max_bytes: int = 64 * 1024 * 1024) -> Case:
    if path.stat().st_size > max_bytes:
        raise ValueError("Serialized capture budget exceeded")
    meta = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if meta.get("schema") != 1 or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Capture schema/hash mismatch")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    allowed = {"q", "k", "v", "u", "mask", "saved_output"}
    if set(tensors) - allowed or not {"q", "k", "v", "u"} <= set(tensors):
        raise ValueError("Unexpected capture payload")
    if any(not isinstance(x, torch.Tensor) for x in tensors.values()):
        raise ValueError("Capture must contain only tensors")
    if sum(x.numel() * x.element_size() for x in tensors.values()) > max_bytes:
        raise ValueError("Expanded tensor budget exceeded")
    case = Case(**{k: v.to(device) for k, v in tensors.items()}, causal=meta["causal"],
                scale=meta["scale"], metadata=meta["metadata"])
    case.validate()
    return case
