"""Weight-space operations. Never average independently trained LoRA A/B factors."""
from __future__ import annotations

import hashlib
import math

import torch

from .config import canonical, require


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        t = value.detach().cpu().contiguous()
        h.update(canonical([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def check_states(states):
    require(bool(states) and bool(states[0]), "empty weight collection")
    keys = set(states[0])
    require(not any("lora_A" in k or "lora_B" in k for k in keys), "materialize effective LoRA deltas; factor averaging is not weight souping")
    for state in states:
        require(set(state) == keys, "weight keys mismatch")
        for name, value in state.items():
            require(value.shape == states[0][name].shape and value.dtype == states[0][name].dtype, "weight shape/dtype mismatch")
            require(value.is_floating_point() and bool(torch.isfinite(value).all()), "only finite floating weights may be mixed")


def weighted_soup(states, weights, identities):
    check_states(states)
    require(len(states) == len(weights) == len(identities), "soup cardinality mismatch")
    require(bool(identities[0]) and len(set(identities)) == 1, "common base/tokenizer/capacity identity required")
    require(all(math.isfinite(w) and w >= 0 for w in weights) and math.isclose(sum(weights), 1., abs_tol=1e-8), "weights must be a convex combination")
    output = {}
    for name in states[0]:
        accumulator = torch.zeros_like(states[0][name], dtype=torch.float64, device="cpu")
        for state, weight in zip(states, weights):
            accumulator.add_(state[name].detach().cpu().double(), alpha=weight)
        value = accumulator.to(states[0][name].dtype)
        require(bool(torch.isfinite(value).all()), "nonfinite merged weights")
        output[name] = value
    return output


def compose_deltas(base, deltas, coefficients, identities):
    """Independent coefficients, not a convex soup. Requires common-base deltas."""
    check_states([base, *deltas])
    require(len(deltas) == len(coefficients) and len(identities) == len(deltas) + 1, "delta cardinality mismatch")
    require(bool(identities[0]) and len(set(identities)) == 1, "delta base identity mismatch")
    require(all(math.isfinite(c) and abs(c) <= 2 for c in coefficients), "unqualified delta coefficient")
    result = {k: v.detach().cpu().double().clone() for k, v in base.items()}
    for delta, coefficient in zip(deltas, coefficients):
        for name in result:
            result[name].add_(delta[name].detach().cpu().double(), alpha=coefficient)
    result = {k: v.to(base[k].dtype) for k, v in result.items()}
    check_states([base, result])
    return result


def exact_lora_sum(adapters, coefficients):
    """Return A,B with B@A == sum_i c_i * scale_i * B_i@A_i.

    Concatenation increases rank. This utility is NOT enabled in native RLP arms:
    switching ranks would confound their fixed-capacity comparison.
    """
    require(bool(adapters) and len(adapters) == len(coefficients), "invalid adapter list")
    ins, outs = adapters[0][0].shape[1], adapters[0][1].shape[0]
    aa, bb = [], []
    for (a, b, scale), coefficient in zip(adapters, coefficients):
        require(a.ndim == b.ndim == 2 and a.shape[0] == b.shape[1] and a.shape[1] == ins and b.shape[0] == outs, "LoRA shapes mismatch")
        require(math.isfinite(scale) and math.isfinite(coefficient) and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "nonfinite adapter")
        aa.append(a.detach().cpu().double())
        bb.append(b.detach().cpu().double() * (scale * coefficient))
    return torch.cat(aa, dim=0), torch.cat(bb, dim=1)
