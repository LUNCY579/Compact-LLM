"""Geometry is mechanical. Functional effects and acquisition are separate endpoints."""
import math
import torch


def pair_geometry(a, b):
    if a.shape != b.shape or not a.is_floating_point() or not b.is_floating_point() or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("finite same-shaped floating deltas required")
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    sa, sb = a != 0, b != 0
    overlap, union = int((sa & sb).sum()), int((sa | sb).sum())
    na, nb, dot = float(a.norm()), float(b.norm()), float((a * b).sum())
    return {"norm_a": na, "norm_b": nb, "dot": dot,
            "cosine": dot / (na * nb) if na and nb else None,
            "overlap": overlap, "jaccard": overlap / union if union else None,
            "sign_agreement": float((a[sa & sb].sign() == b[sa & sb].sign()).double().mean()) if overlap else None}


def layer_allocation(coords, before_free, mask, delta):
    from .allocation import checked_mask
    checked_mask(before_free, coords.size)
    checked_mask(mask, coords.size)
    out, pos = [], 0
    for name, n in zip(coords.names, coords.sizes):
        free, chosen, d = before_free[pos:pos+n], mask[pos:pos+n], delta[pos:pos+n].double()
        out.append({"parameter": name, "available_before": int(free.sum()), "selected": int(chosen.sum()),
                    "selected_fraction_of_available": int(chosen.sum()) / int(free.sum()) if bool(free.any()) else None,
                    "norm": float(d.norm()), "energy": float(d.square().sum())})
        pos += n
    return out


def interaction(base, a, b, combined):
    values = (base, a, b, combined)
    if base.numel() == 0 or any(x.shape != base.shape or x.ndim < 1 or not x.is_floating_point() or not bool(torch.isfinite(x).all()) for x in values):
        raise ValueError("aligned finite logit tensors required")
    z = combined.double() - a.double() - b.double() + base.double()
    centered = z - z.mean(dim=-1, keepdim=True)
    return {"raw_rms": float(z.square().mean().sqrt()), "raw_max_abs": float(z.abs().max()),
            "vocab_centered_rms": float(centered.square().mean().sqrt()),
            "interpretation": "logit nonadditivity at identical selected positions; not accuracy or causal independence"}


def rank_energy(matrix, rank=16, max_elements=1_000_000):
    if matrix.ndim != 2 or matrix.numel() > max_elements or type(rank) is not int or rank < 1 or not bool(torch.isfinite(matrix).all()):
        raise ValueError("offline rank diagnostic requires a bounded finite matrix")
    energy = torch.linalg.svdvals(matrix.detach().cpu().double()).square()
    total = float(energy.sum())
    return {"rank": rank, "energy_fraction": float(energy[:rank].sum()) / total if total else None}


def compose(deltas, coefficients):
    if not deltas or len(deltas) != len(coefficients) or any(type(c) not in (int, float) or not math.isfinite(c) for c in coefficients):
        raise ValueError("finite explicit coefficients required")
    out = torch.zeros_like(deltas[0])
    for delta, coefficient in zip(deltas, coefficients):
        if not delta.is_floating_point() or delta.shape != out.shape or delta.dtype != out.dtype or delta.device != out.device or not bool(torch.isfinite(delta).all()):
            raise ValueError("unaligned deltas")
        out.add_(delta, alpha=coefficient)
    if not bool(torch.isfinite(out).all()):
        raise ValueError("nonfinite merge")
    return out


def shared_private(shared, private, coefficients):
    # Shared state is added ONCE, not once per private skill. Design-extension helper only.
    delta = compose(private, coefficients)
    if shared.shape != delta.shape or shared.dtype != delta.dtype or shared.device != delta.device or not bool(torch.isfinite(shared).all()):
        raise ValueError("unaligned shared state")
    result = shared + delta
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite shared/private merge")
    return result
