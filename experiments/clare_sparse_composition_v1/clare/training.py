"""Masked delta training on any torch model whose footprint slots are ``nn.Linear`` weights.

``DeltaHost`` patches each footprint Linear so that its effective weight is

    W_eff = W0.to(fp32) + D,     D = frozen + trainable

* ``frozen``  : FP32 accumulated delta of previously learned skills (never receives gradients).
* trainable   : either DENSE (an FP32 tensor over the whole slot plus a boolean gradient mask) or
                SPARSE (FP32 values at an explicit list of coordinates; everything else is
                structurally untouchable). The base weight W0 never changes.

Invariants enforced (and tested):

1. coordinates outside the trainable mask receive exactly-zero gradient (dense: gradient hook);
   with weight_decay = 0 their Adam moments stay 0, so their values stay exactly 0;
2. frozen deltas are bit-identical before and after any stage (checksummed);
3. ``stop_trainable`` restores the pre-stage function exactly (the discovery RESET);
4. a dense-masked run and a sparse run over the same mask are the same ALGORITHM (elementwise
   AdamW, identical gradients) but NOT bit-identical: measured on CPU, grad-norm reductions and
   vectorised-vs-scalar-tail elementwise kernels round differently for the two tensor layouts, so
   the deltas agree only to float32 rounding (test_mask_safety). Arms therefore never mix paths:
   D and E both retrain through the SPARSE path; A and B use the dense path.

The canonical composition W = W0 + (sum of deltas) is computed by the same expression, so the
model evaluated after training is exactly the model that was trained.
"""
from __future__ import annotations

import hashlib
import math
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .deltas import DELTA_FORMAT, Delta, DenseDelta, SparseDelta
from .footprint import Footprint, Slot
from .masks import SkillMask
from .recipe import Recipe, rate_at


class _SlotState:
    __slots__ = ("slot", "module", "frozen", "dense", "grad_mask", "hook", "idx", "val")

    def __init__(self, slot: Slot, module: nn.Linear):
        self.slot = slot
        self.module = module
        self.frozen: torch.Tensor | None = None
        self.dense: nn.Parameter | None = None
        self.grad_mask: torch.Tensor | None = None
        self.hook = None
        self.idx: torch.Tensor | None = None
        self.val: nn.Parameter | None = None


def _delta_forward(host: "DeltaHost", st: _SlotState):
    def forward(self, x):
        c = host._eval_cache.get(st.slot.name) if host._eval_cache is not None else None
        if c is not None and not torch.is_grad_enabled() and x.dtype == torch.float32:
            return F.linear(x, c, self.bias)
        return F.linear(x, host.effective_weight(st, x.dtype), self.bias)
    return forward


class DeltaHost:
    def __init__(self, model: nn.Module, footprint: Footprint):
        self.model = model
        self.footprint = footprint
        for p in model.parameters():
            p.requires_grad_(False)
        self.slots: dict[str, _SlotState] = {}
        for slot in footprint:
            if not slot.name.endswith(".weight"):
                raise ValueError(f"{slot.name}: footprint slots must be Linear weights")
            module = model.get_submodule(slot.name[: -len(".weight")])
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{slot.name}: expected nn.Linear, got {type(module).__name__}")
            if tuple(module.weight.shape) != slot.shape:
                raise ValueError(f"{slot.name}: shape {tuple(module.weight.shape)} != footprint {slot.shape}")
            st = _SlotState(slot, module)
            module.forward = types.MethodType(_delta_forward(self, st), module)
            self.slots[slot.name] = st
        self.mode: str | None = None
        self._trainable_mask: SkillMask | None = None
        self._eval_cache: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def cache_effective(self, on: bool) -> None:
        """Evaluation only: precompute each slot's effective FP32 weight (W0 + frozen) once, instead
        of on every forward. Same expression, same values; used only when autograd is off. Any change
        of the frozen delta or any open stage invalidates it."""
        if on and self.mode is not None:
            raise RuntimeError("the effective-weight cache is for evaluation with no open stage")
        self._eval_cache = ({n: self.effective_weight(st) for n, st in self.slots.items()} if on else None)

    @property
    def device(self) -> torch.device:
        return next(iter(self.slots.values())).module.weight.device

    # -- effective weight ---------------------------------------------------------------
    def delta_of(self, st: _SlotState) -> torch.Tensor | None:
        d = st.frozen
        if st.dense is not None:
            d = st.dense if d is None else d + st.dense
        if st.val is not None:
            base = d.reshape(-1) if d is not None else torch.zeros(st.slot.numel, dtype=torch.float32,
                                                                   device=st.val.device)
            d = base.index_add(0, st.idx, st.val).view(st.slot.shape)
        return d

    def effective_weight(self, st: _SlotState, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        w = st.module.weight.to(torch.float32)
        d = self.delta_of(st)
        w = w if d is None else w + d
        return w if dtype == torch.float32 else w.to(dtype)

    # -- frozen accumulated delta ---------------------------------------------------------
    def set_frozen(self, delta: "Delta | Sequence[Delta] | None") -> None:
        """Install the frozen accumulated delta. A sequence is composed slot by slot with exactly
        the arithmetic of ``deltas.compose`` (FP32 accumulation in the given order), so the full
        composed delta is never materialised in host memory."""
        if self.mode is not None:
            raise RuntimeError("cannot change the frozen delta while a stage is open")
        self._eval_cache = None
        parts = [] if delta is None else ([delta] if isinstance(delta, Delta) else list(delta))
        if any(d.footprint.digest() != self.footprint.digest() for d in parts):
            raise ValueError("delta footprint mismatch")
        for name, st in self.slots.items():
            if not parts:
                st.frozen = None
                continue
            if len(parts) == 1:
                arr = np.array(parts[0].dense(name), dtype=np.float32, copy=True)
            else:
                arr = np.zeros(st.slot.numel, dtype=np.float32)
                for d in parts:
                    arr += d.dense(name)
            st.frozen = torch.from_numpy(arr.reshape(st.slot.shape)).to(self.device, torch.float32)

    def delta_to_device(self, delta: Delta) -> dict[str, torch.Tensor]:
        """One delta as per-slot FP32 device tensors (for fast configuration switching)."""
        return {n: torch.from_numpy(np.array(delta.dense(n), dtype=np.float32, copy=True).reshape(st.slot.shape))
                .to(self.device) for n, st in self.slots.items()}

    def set_frozen_tensors(self, tensors: Mapping[str, torch.Tensor] | None) -> None:
        """Install device tensors as the frozen delta WITHOUT copying (pointer swap). The caller
        must not mutate them while installed."""
        if self.mode is not None:
            raise RuntimeError("cannot change the frozen delta while a stage is open")
        self._eval_cache = None
        for n, st in self.slots.items():
            st.frozen = None if tensors is None else tensors[n]

    def frozen_delta(self) -> DenseDelta:
        return DenseDelta(self.footprint, {n: (st.frozen.detach().cpu().numpy().reshape(-1) if st.frozen is not None
                                               else np.zeros(st.slot.numel, dtype=np.float32))
                                           for n, st in self.slots.items()})

    def frozen_checksum(self) -> str:
        return self.frozen_delta().checksum()

    # -- trainable stage ------------------------------------------------------------------
    def start_dense(self, trainable: SkillMask | None) -> list[nn.Parameter]:
        """Zero-initialised dense FP32 delta; gradients outside ``trainable`` are zeroed."""
        self._open("dense", trainable)
        params = []
        for name, st in self.slots.items():
            st.dense = nn.Parameter(torch.zeros(st.slot.shape, dtype=torch.float32, device=self.device))
            if trainable is not None:
                m = torch.from_numpy(trainable.bool(name).reshape(st.slot.shape)).to(self.device)
                st.grad_mask = m
                st.hook = st.dense.register_hook(lambda g, m=m: g.masked_fill(~m, 0.0))
            params.append(st.dense)
        return params

    def start_sparse(self, mask: SkillMask) -> list[nn.Parameter]:
        """Zero-initialised FP32 values at exactly the coordinates of ``mask``."""
        self._open("sparse", mask)
        params = []
        for name, st in self.slots.items():
            idx = mask.indices(name)
            st.idx = torch.from_numpy(idx).to(self.device)
            st.val = nn.Parameter(torch.zeros(idx.size, dtype=torch.float32, device=self.device))
            params.append(st.val)
        return params

    def _open(self, mode: str, mask: SkillMask | None) -> None:
        if self.mode is not None:
            raise RuntimeError(f"a {self.mode} stage is already open")
        self._eval_cache = None
        if mask is not None and mask.footprint.digest() != self.footprint.digest():
            raise ValueError("mask footprint mismatch")
        self.mode = mode
        self._trainable_mask = mask

    def trainable_parameters(self) -> list[nn.Parameter]:
        out = []
        for st in self.slots.values():
            if st.dense is not None:
                out.append(st.dense)
            if st.val is not None:
                out.append(st.val)
        return out

    def trainable_delta(self) -> Delta:
        if self.mode == "dense":
            return DenseDelta(self.footprint, {n: st.dense.detach().cpu().numpy().reshape(-1)
                                               for n, st in self.slots.items()})
        if self.mode == "sparse":
            return SparseDelta(self._trainable_mask, {n: st.val.detach().cpu().numpy()
                                                      for n, st in self.slots.items()})
        raise RuntimeError("no open stage")

    def save_trainable(self, path_stem: str | Path, meta: Mapping | None = None):
        """Stream the open stage's trainable delta to disk one slot at a time; return a DiskDelta.

        Host memory stays at one slot, so a 188.7M-coordinate dense delta never sits in RAM."""
        from .deltas import DiskDelta, write_dense_stream, write_sparse_stream

        if self.mode == "dense":
            gen = ((n, st.dense.detach().cpu().numpy().reshape(-1)) for n, st in self.slots.items())
            write_dense_stream(path_stem, self.footprint, gen, meta)
        elif self.mode == "sparse":
            gen = ((n, st.val.detach().cpu().numpy()) for n, st in self.slots.items())
            write_sparse_stream(path_stem, self._trainable_mask, gen, meta)
        else:
            raise RuntimeError("no open stage")
        return DiskDelta(path_stem, self.footprint)

    def importance(self, name: str) -> np.ndarray:
        """|trainable delta| of one slot (dense stages only): CLARE's s_t(j) = |alpha_j - alpha0_j|."""
        st = self.slots[name]
        if st.dense is None:
            raise RuntimeError("importance is defined for dense stages")
        return st.dense.detach().abs().cpu().numpy().reshape(-1)

    @torch.no_grad()
    def prox_l1(self, threshold: float) -> None:
        """Soft-threshold the dense trainable delta: the proximal step of lambda*||P_A(delta)||_1."""
        if self.mode != "dense":
            raise RuntimeError("the L1 proximal step applies to dense discovery stages")
        if threshold < 0 or not math.isfinite(threshold):
            raise ValueError("threshold must be finite and >= 0")
        if threshold == 0:
            return
        for st in self.slots.values():
            d = st.dense
            d.copy_(torch.sign(d) * torch.clamp(d.abs() - threshold, min=0.0))

    @torch.no_grad()
    def outside_mask_nonzero(self) -> int:
        """Dense stage: number of trainable values that are non-zero OUTSIDE the gradient mask."""
        n = 0
        for st in self.slots.values():
            if st.dense is not None and st.grad_mask is not None:
                n += int((st.dense[~st.grad_mask] != 0).sum())
        return n

    def stop_trainable(self) -> None:
        """Discard the trainable part (the RESET). Frozen deltas and base weights are untouched."""
        for st in self.slots.values():
            if st.hook is not None:
                st.hook.remove()
            st.dense = st.grad_mask = st.hook = st.idx = st.val = None
        self.mode = None
        self._trainable_mask = None

    # -- identity -------------------------------------------------------------------------
    @torch.no_grad()
    def effective_checksum(self) -> str:
        """sha256 over the effective FP32 weights of every footprint slot (function identity)."""
        h = hashlib.sha256(b"clare-effective-v1")
        h.update(bytes.fromhex(self.footprint.digest()))
        for st in self.slots.values():
            h.update(self.effective_weight(st).detach().cpu().numpy().astype("<f4").tobytes())
        return h.hexdigest()


def base_checksum(model: nn.Module, names: Iterable[str] | None = None) -> str:
    """sha256 over named model parameters (the frozen base), in name order."""
    h = hashlib.sha256(b"clare-base-v1")
    params = dict(model.named_parameters())
    for name in sorted(params if names is None else names):
        t = params[name].detach().cpu()
        h.update(name.encode() + b"\0" + str(t.dtype).encode() + b"\0")
        h.update(t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16
                 else t.numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------------------

LossFn = Callable[[dict], "tuple[torch.Tensor, int]"]
"""row -> (summed CE over supervised tokens, number of supervised tokens)."""


@dataclass
class StageLog:
    updates: int = 0
    processed_tokens: int = 0
    supervised_tokens: int = 0
    losses: list = field(default_factory=list)
    lrs: list = field(default_factory=list)
    grad_norms: list = field(default_factory=list)
    seconds: float = 0.0
    peak_gpu_bytes: int | None = None

    def as_dict(self) -> dict:
        return {"updates": self.updates, "processed_tokens": self.processed_tokens,
                "supervised_tokens": self.supervised_tokens, "loss": self.losses, "lr": self.lrs,
                "grad_norm": self.grad_norms, "seconds": self.seconds, "peak_gpu_bytes": self.peak_gpu_bytes}


def make_optimizer(params: Sequence[nn.Parameter], recipe: Recipe) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=recipe.peak_lr, betas=(recipe.beta1, recipe.beta2),
                             eps=recipe.epsilon, weight_decay=recipe.weight_decay,
                             foreach=False, fused=False)


def run_stage(host: DeltaHost, rows: Sequence[dict], groups: Sequence[Sequence[int]], loss_fn: LossFn,
              recipe: Recipe, *, l1_lambda: float = 0.0, total_tokens: int | None = None,
              start_update: int = 0, optimizer: torch.optim.Optimizer | None = None,
              on_update: Callable[[int, dict], None] | None = None,
              guard: Callable[[], None] | None = None) -> tuple[StageLog, torch.optim.Optimizer]:
    """Train the open stage over ``groups`` (lists of row indices), Titan-style:

    per-row forward/backward (no padding), loss = summed CE / group supervised tokens, global
    grad-norm clip, WSD learning rate by processed-token progress, AdamW step, then (discovery
    only) the L1 proximal step with threshold ``lr * l1_lambda``.
    """
    params = host.trainable_parameters()
    if not params:
        raise RuntimeError("no trainable parameters: open a stage first")
    opt = optimizer or make_optimizer(params, recipe)
    total = total_tokens or sum(len(rows[i]["ids"]) for g in groups for i in g)
    done_tokens = sum(len(rows[i]["ids"]) for g in groups[:start_update] for i in g)
    log = StageLog()
    t0 = time.time()
    cuda = host.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(host.device)
    for u in range(start_update, len(groups)):
        group = groups[u]
        denom = sum(int(rows[i]["supervised_tokens"]) for i in group)
        if denom <= 0:
            raise ValueError(f"update {u}: no supervised tokens")
        loss_total = 0.0
        for i in group:
            loss_sum, n = loss_fn(rows[i])
            if n != rows[i]["supervised_tokens"]:
                raise ValueError(f"row {i}: loss covered {n} tokens, expected {rows[i]['supervised_tokens']}")
            (loss_sum / denom).backward()
            loss_total += float(loss_sum.detach())
        for p in params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"update {u}: non-finite gradient")
        gn = torch.nn.utils.clip_grad_norm_(params, recipe.clip_norm)
        done_tokens += sum(len(rows[i]["ids"]) for i in group)
        lr = rate_at(recipe.peak_lr, done_tokens, total, recipe)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
        if l1_lambda:
            host.prox_l1(lr * l1_lambda)
        log.updates += 1
        log.processed_tokens += sum(len(rows[i]["ids"]) for i in group)
        log.supervised_tokens += denom
        log.losses.append(loss_total / denom)
        log.lrs.append(lr)
        log.grad_norms.append(float(gn))
        if on_update is not None:
            on_update(u, {"loss": loss_total / denom, "lr": lr, "grad_norm": float(gn)})
        if guard is not None:
            guard()
    log.seconds = time.time() - t0
    if cuda:
        log.peak_gpu_bytes = int(torch.cuda.max_memory_allocated(host.device))
    return log, opt


# ---------------------------------------------------------------------------------------
# Mid-stage checkpoints (TDR resilience) — create-only, bound to the stage identity
# ---------------------------------------------------------------------------------------

def save_stage_checkpoint(path: str | Path, host: DeltaHost, opt: torch.optim.Optimizer,
                          next_update: int, binding: Mapping) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state = {"format": "clare-stage-ckpt-v1", "binding": dict(binding), "next_update": next_update,
             "mode": host.mode, "trainable": [p.detach().cpu() for p in host.trainable_parameters()],
             "optimizer": opt.state_dict(), "rng_cpu": torch.get_rng_state(),
             "frozen_checksum": host.frozen_checksum()}
    tmp = path.with_name(path.name + ".partial")
    torch.save(state, tmp)
    tmp.replace(path)


def load_stage_checkpoint(path: str | Path, host: DeltaHost, opt: torch.optim.Optimizer,
                          binding: Mapping) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "clare-stage-ckpt-v1" or state["binding"] != dict(binding):
        raise ValueError("stage checkpoint binding mismatch (different stage, data or code)")
    if state["mode"] != host.mode:
        raise ValueError("stage checkpoint mode mismatch")
    if state["frozen_checksum"] != host.frozen_checksum():
        raise ValueError("frozen delta differs from the checkpointed stage")
    params = host.trainable_parameters()
    if len(params) != len(state["trainable"]):
        raise ValueError("trainable inventory mismatch")
    with torch.no_grad():
        for p, v in zip(params, state["trainable"]):
            if p.shape != v.shape:
                raise ValueError("trainable shape mismatch")
            p.copy_(v.to(p.device))
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng_cpu"])
    return int(state["next_update"])
