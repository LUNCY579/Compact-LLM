"""Parameter-space geometry of skill deltas.

These are GEOMETRIC diagnostics only. For strictly disjoint supports the intersection is 0 and
the dot product is exactly 0 by construction; that is reported as a mechanical check, never as
an empirical finding. Whether the deltas interact FUNCTIONALLY is measured in ``functional``.

Accumulation is float64 per slot (one slot's values in memory at a time). Supports are the
NON-ZERO values of a delta, which can be smaller than its allocated mask (a trained coordinate
may end exactly at 0); allocated capacity is reported separately from active non-zeros.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .deltas import Delta
from .masks import SkillMask


def pair_geometry(a: Delta, b: Delta) -> dict:
    if a.footprint.digest() != b.footprint.digest():
        raise ValueError("deltas live in different footprints")
    dot = na = nb = 0.0
    inter = union = agree = 0
    nnz_a = nnz_b = 0
    for slot in a.footprint:
        x = a.dense(slot.name).astype(np.float64).reshape(-1)
        y = b.dense(slot.name).astype(np.float64).reshape(-1)
        dot += float(x @ y)
        na += float(x @ x)
        nb += float(y @ y)
        sx, sy = x != 0, y != 0
        both = sx & sy
        nnz_a += int(sx.sum())
        nnz_b += int(sy.sum())
        inter += int(both.sum())
        union += int((sx | sy).sum())
        agree += int((np.sign(x[both]) == np.sign(y[both])).sum())
    cos = dot / math.sqrt(na * nb) if na > 0 and nb > 0 else float("nan")
    return {"dot": dot, "cosine": cos, "norm_a": math.sqrt(na), "norm_b": math.sqrt(nb),
            "nonzero_a": nnz_a, "nonzero_b": nnz_b, "support_intersection": inter,
            "support_union": union, "jaccard": inter / union if union else 0.0,
            "sign_agreement_on_overlap": agree / inter if inter else None,
            "overlap_coordinates_for_sign_agreement": inter}


def pairwise(deltas: Mapping[str, Delta]) -> list[dict]:
    names = list(deltas)
    rows = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            rows.append({"a": x, "b": y, **pair_geometry(deltas[x], deltas[y])})
    return rows


def mask_pair(a: SkillMask, b: SkillMask) -> dict:
    inter = a.overlap(b)
    union = a.count() + b.count() - inter
    return {"allocated_a": a.count(), "allocated_b": b.count(), "allocated_intersection": inter,
            "allocated_union": union, "allocated_jaccard": inter / union if union else 0.0}


def layer_energy(delta: Delta, mask: SkillMask | None = None) -> list[dict]:
    """Per slot: squared norm, share of total energy, non-zeros, allocated coordinates."""
    rows = []
    total = 0.0
    for slot in delta.footprint:
        v = delta.dense(slot.name).astype(np.float64).reshape(-1)
        e = float(v @ v)
        total += e
        rows.append({"slot": slot.name, "layer": slot.layer, "module": slot.module,
                     "coordinates": slot.numel, "energy": e, "l2_norm": math.sqrt(e),
                     "nonzero": int(np.count_nonzero(v)),
                     "allocated": mask.count(slot.name) if mask is not None else None})
    for r in rows:
        r["energy_share"] = r["energy"] / total if total > 0 else 0.0
    return rows


def by_layer(rows: Sequence[dict], keys: Sequence[str] = ("energy", "nonzero", "allocated")) -> list[dict]:
    """Aggregate slot-level rows to transformer layers."""
    agg: dict = {}
    for r in rows:
        a = agg.setdefault(r["layer"], {"layer": r["layer"], **{k: 0 for k in keys}, "coordinates": 0})
        a["coordinates"] += r["coordinates"]
        for k in keys:
            if r.get(k) is not None:
                a[k] += r[k]
    total_e = sum(a["energy"] for a in agg.values()) if "energy" in keys else 0.0
    out = []
    for layer in sorted(agg, key=lambda l: (-1 if l is None else l)):
        a = agg[layer]
        if "energy" in keys:
            a["energy_share"] = a["energy"] / total_e if total_e > 0 else 0.0
            a["l2_norm"] = math.sqrt(a["energy"])
        out.append(a)
    return out


def rank_energy(delta: Delta, ks: Sequence[int] = (1, 4, 16, 64), max_slots: int | None = None) -> list[dict]:
    """OFFLINE diagnostic: fraction of each 2-D slot's squared Frobenius norm in its top-k
    singular directions (exact SVD, float64). Never combined with masking in CLARE-RINEYA-001.

    For a SPARSE delta this measures the rank structure of the sparse matrix itself; a sparse
    random-support matrix is typically high-rank, so low concentration there is expected.
    """
    rows = []
    for i, slot in enumerate(delta.footprint):
        if max_slots is not None and i >= max_slots:
            break
        if len(slot.shape) != 2:
            continue
        m = delta.dense(slot.name).astype(np.float64).reshape(slot.shape)
        fro2 = float((m * m).sum())
        if fro2 == 0.0:
            rows.append({"slot": slot.name, "layer": slot.layer, "module": slot.module,
                         "frobenius": 0.0, **{f"top{k}_energy_fraction": None for k in ks},
                         "effective_rank": None})
            continue
        s = np.linalg.svd(m, compute_uv=False)
        s2 = s * s
        p = s2 / s2.sum()
        nz = p[p > 0]
        row = {"slot": slot.name, "layer": slot.layer, "module": slot.module, "frobenius": math.sqrt(fro2),
               "effective_rank": float(math.exp(-(nz * np.log(nz)).sum()))}
        for k in ks:
            row[f"top{k}_energy_fraction"] = float(s2[:k].sum() / s2.sum())
        rows.append(row)
    return rows


def free_capacity_trace(total: int, budgets: Sequence[int]) -> list[dict]:
    free = total
    out = []
    for i, b in enumerate(budgets):
        out.append({"position": i, "free_before": free, "claimed": b, "free_after": free - b,
                    "free_fraction_after": (free - b) / total})
        free -= b
    return out
