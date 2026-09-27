"""Transactional two-stage direct-coordinate training with optimizer-safe masking.

Loss callbacks are trusted host integrations. They receive only a validated training
row. All model parameters outside the explicitly bound FP32 pool are frozen.
"""
import hashlib
import random
from contextlib import contextmanager
import torch
from .allocation import Allocation, budget, checked_mask, select


class Coordinates:
    def __init__(self, model, names, max_coordinates):
        all_params = dict(model.named_parameters())
        names = tuple(sorted(names))
        if not names or len(names) != len(set(names)) or any(n not in all_params for n in names):
            raise ValueError("missing/duplicate eligible parameter names")
        self.model, self.names = model, names
        self.params = [all_params[n] for n in names]
        self.shapes = [list(p.shape) for p in self.params]
        self.sizes = [p.numel() for p in self.params]
        self.size = sum(self.sizes)
        if self.size > max_coordinates:
            raise ValueError("eligible pool exceeds registered memory ceiling")
        if any(p.dtype != torch.float32 or p.layout != torch.strided or not p.is_contiguous() for p in self.params):
            raise ValueError("v1 requires contiguous direct FP32 coordinates; no quantized/LoRA-factor claim")
        pointers = [p.untyped_storage().data_ptr() for p in all_params.values()]
        if len(set(pointers)) != len(pointers):
            raise ValueError("aliased parameter storages require a separate qualification")
        if len({str(p.device) for p in self.params}) != 1:
            raise ValueError("one device required")
        if not bool(torch.isfinite(self.vector()).all()):
            raise ValueError("nonfinite initial coordinates")

    def vector(self):
        return torch.cat([p.detach().cpu().reshape(-1) for p in self.params]).clone()

    def pieces(self, vector):
        return [v.reshape(shape) for v, shape in zip(vector.split(self.sizes), self.shapes)]

    def restore(self, vector):
        if vector.shape != (self.size,) or vector.dtype != torch.float32 or not bool(torch.isfinite(vector).all()):
            raise ValueError("invalid restoration vector")
        with torch.no_grad():
            for p, v in zip(self.params, self.pieces(vector)):
                p.copy_(v.to(p.device))

    @contextmanager
    def trainable(self):
        flags = [(p, p.requires_grad) for p in self.model.parameters()]
        modes = [(m, m.training) for m in self.model.modules()]
        try:
            for p, _ in flags:
                p.requires_grad_(False)
                p.grad = None
            for p in self.params:
                p.requires_grad_(True)
            # Fixed evaluation-mode forwards avoid unregistered dropout/buffer drift.
            self.model.eval()
            yield
        finally:
            for p, flag in flags:
                p.grad = None
                p.requires_grad_(flag)
            for m, mode in modes:
                m.training = mode


def stage(coords, start, mask, rows, steps, loss_fn, config, *, l1=0.0, seed=0, guard=lambda: None):
    checked_mask(mask, coords.size)
    if not bool(mask.any()) or not rows:
        raise ValueError("empty mask or training bank")
    coords.restore(start)
    masks = [m.to(p.device) for m, p in zip(coords.pieces(mask), coords.params)]
    starts = [v.to(p.device) for v, p in zip(coords.pieces(start), coords.params)]
    py_state = random.getstate()
    devices = sorted({p.device.index for p in coords.params if p.device.type == "cuda"})
    trace = []
    try:
        with torch.random.fork_rng(devices=devices), coords.trainable():
            torch.manual_seed(seed)
            random.seed(seed)
            opt = torch.optim.AdamW(coords.params, lr=config.lr, weight_decay=config.weight_decay, foreach=False)
            for step in range(steps):
                guard()
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(coords.model, rows[step % len(rows)])
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("loss must be a finite scalar tensor")
                penalty = sum(((p - a).abs() * m).sum() for p, a, m in zip(coords.params, starts, masks))
                objective = loss + l1 * penalty
                if not bool(torch.isfinite(objective)):
                    raise ValueError("nonfinite regularized objective")
                objective.backward()
                for p, m in zip(coords.params, masks):
                    if p.grad is not None:
                        if not bool(torch.isfinite(p.grad).all()):
                            raise ValueError("nonfinite gradient")
                        p.grad.masked_fill_(~m, 0)
                opt.step()
                # Gradient masking alone does NOT stop AdamW decay or stale moments.
                with torch.no_grad():
                    for p, m, a in zip(coords.params, masks, starts):
                        p[~m] = a[~m]
                        for value in opt.state[p].values():
                            if isinstance(value, torch.Tensor) and value.shape == p.shape:
                                value.masked_fill_(~m, 0)
                now = coords.vector()
                if not bool(torch.isfinite(now).all()) or not torch.equal(now[~mask], start[~mask]):
                    raise ValueError("frozen-coordinate invariant failed")
                trace.append({"loss": float(loss.detach()), "l1": float(penalty.detach())})
            result = coords.vector()
    except BaseException:
        coords.restore(start)
        raise
    finally:
        random.setstate(py_state)
    return result, trace


class SkillRunner:
    def __init__(self, model, names, config, *, data_digest):
        if not isinstance(data_digest, str) or len(data_digest) != 64 or any(c not in "0123456789abcdef" for c in data_digest):
            raise ValueError("training manifest must be hash-bound")
        self.config, self.data_digest = config, data_digest
        self.coords = Coordinates(model, names, config.max_coordinates)
        self.base = self.coords.vector()
        self.current = self.base.clone()
        self.allocation = Allocation(self.coords.size, config.order)
        self.deltas, self.records = {}, []

    def train_skill(self, skill, rows, loss_fn, *, guard=lambda: None):
        from .data import validate_training_rows
        if len(self.records) >= len(self.config.order) or skill != self.config.order[len(self.records)]:
            raise ValueError("out-of-order skill")
        validate_training_rows(rows, skill, self.data_digest)
        if not torch.equal(self.coords.vector(), self.current):
            raise ValueError("model state changed outside the runner")
        cfg = self.config
        strict = cfg.arm in ("random_disjoint", "learned_disjoint")
        start = self.current.clone() if strict or cfg.arm == "dense_sequential" else self.base.clone()
        free = self.allocation.free
        k = budget(int(free.sum()), cfg.rho)
        if not k:
            raise ValueError("capacity exhausted; no silent zero-update skill")
        all_mask = torch.ones_like(free)
        eligible = free if strict else all_mask
        seed = int(hashlib.sha256(f"{cfg.seed}:{skill}".encode()).hexdigest()[:8], 16)
        try:
            # Cost-matched probe for A/B/C/D; only E uses its movement for selection.
            discovered, probe_trace = stage(self.coords, start, eligible, rows, cfg.discovery_steps,
                                            loss_fn, cfg, l1=cfg.discovery_l1, seed=seed, guard=guard)
            scores = (discovered - start).abs()
            if strict:
                mask = select(free, k, scores=scores if cfg.arm == "learned_disjoint" else None, seed=seed)
            else:
                mask = all_mask
            self.coords.restore(start)
            reset_exact = torch.equal(self.coords.vector(), start)
            if not reset_exact:
                raise ValueError("discovery reset failed")
            trained, train_trace = stage(self.coords, start, mask, rows, cfg.train_steps,
                                         loss_fn, cfg, seed=seed, guard=guard)
            delta = trained - start
            if cfg.arm == "magnitude_sparse":
                mask = select(all_mask, k, scores=delta.abs())
                delta = delta * mask
            if strict and bool(delta[~mask].any()):
                raise ValueError("support escaped allocation")
            # Dense A increments and independent B/C deltas both add to accumulated state.
            new_current = self.current + delta
            if not bool(torch.isfinite(new_current).all()):
                raise ValueError("nonfinite composition")
            # Shadow capacity ledger for dense/independent controls; not their actual support.
            ledger_mask = mask if strict else select(free, k, seed=seed)
            self.coords.restore(new_current)
            self.allocation.claim(skill, ledger_mask, k)
        except BaseException:
            self.coords.restore(self.current)
            raise
        self.current = new_current
        self.deltas[skill] = delta.clone()
        rec = {"skill": skill, "order": list(cfg.order), "seed": cfg.seed, "stage_seed": seed,
               "arm": cfg.arm, "available_before": int(free.sum()), "capacity_budget": k,
               "selected_coordinates": int(mask.sum()), "selected_fraction_total": int(mask.sum()) / self.coords.size,
               "active_nonzero": int((delta != 0).sum()),
               "remaining_capacity": int(self.allocation.free.sum()), "reset_exact": reset_exact,
               "delta_norm": float(delta.double().norm()), "mask": mask.clone(),
               "fp32_recomposition_max_abs": float((start + (trained-start) - trained).abs().max()),
               "probe_trace": probe_trace, "train_trace": train_trace,
               "probe_used_for_selection": cfg.arm == "learned_disjoint"}
        self.records.append(rec)
        return rec
