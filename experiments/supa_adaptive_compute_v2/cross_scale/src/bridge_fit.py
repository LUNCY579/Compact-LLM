"""Prespecified B0/B1/B2/B3 objectives and mandatory null controls."""
from __future__ import annotations

import time

import torch
from torch import nn
from torch.nn import functional as F

from shared.legacy import legacy_graft
from .delta_alignment import deadline, donor_delta, finite, frozen, normalized_loss


def weighted_ridge(x, y, relative):
    """Inverse target token energy: every token has equal relative-error weight."""
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2 or relative <= 0:
        raise ValueError("invalid weighted ridge inputs")
    x, y = finite(x.double()), finite(y.double())
    weights = y.square().mean(-1).clamp_min(1e-8).reciprocal()
    weights = weights / weights.mean()
    xm, ym = (x*weights[:, None]).sum(0)/weights.sum(), (y*weights[:, None]).sum(0)/weights.sum()
    xc, yc = x-xm, y-ym
    gram = xc.T @ (xc*weights[:, None])
    gram.diagonal().add_(relative*gram.diag().mean().clamp_min(1e-12))
    w = torch.linalg.solve(gram, xc.T @ (yc*weights[:, None]))
    return finite(w.float()), finite((ym-xm@w).float())


def cross_scenario_permutation(scenario_ids, seed):
    """Permutation of all training token targets, with no within-scenario match."""
    if len(set(scenario_ids)) < 2:
        raise ValueError("shuffled control requires at least two training scenarios")
    # Cyclic rotation of scenario-contiguous rows preserves all target marginals.
    groups = {}
    for index, scenario in enumerate(scenario_ids):
        groups.setdefault(scenario, []).append(index)
    if max(map(len, groups.values())) * 2 > len(scenario_ids):
        raise ValueError("one scenario dominates tokens; deranged pairing is impossible")
    generator = torch.Generator().manual_seed(seed)
    keys = sorted(groups)
    keys = [keys[i] for i in torch.randperm(len(keys), generator=generator).tolist()]
    ordered = [i for key in keys for i in groups[key]]
    shift = max(map(len, groups.values()))
    targets = ordered[shift:] + ordered[:shift]
    permutation = torch.empty(len(ordered), dtype=torch.long)
    for source, target in zip(ordered, targets):
        permutation[source] = target
    if any(scenario_ids[i] == scenario_ids[j] for i, j in enumerate(permutation.tolist())):
        raise AssertionError("shuffled control retained a scenario correspondence")
    return permutation


def fit_initial(small, big, objective, config, scenario_ids):
    if (small.ndim != 2 or big.ndim != 2 or len(small) != len(big) or len(small) < 2
            or len(scenario_ids) != len(small) or not small.shape[1] or not big.shape[1]):
        raise ValueError("bridge dimensions or scenario count mismatch")
    finite(small); finite(big)
    legacy = legacy_graft()
    if objective == "BR":
        return legacy.Bridges.random(small, big, config["seed"])
    if objective == "BS":
        permutation = cross_scenario_permutation(scenario_ids, config["seed"])
        return legacy.Bridges.fit(small, big[permutation], config["ridge_relative"])
    if objective in ("B0", "B2"):
        return legacy.Bridges.fit(small, big, config["ridge_relative"])
    if objective not in ("B1", "B3"):
        raise ValueError("unregistered bridge objective")
    return legacy.Bridges(*weighted_ridge(legacy.normalize(small), big, config["ridge_relative"]),
                          *weighted_ridge(big, small, config["ridge_relative"]))


def fit_delta_bridge(initial, layer, rope, prompts, config, *, token_normalized, check_budget=None):
    """Train up-map only through frozen donor; down-map remains calibration ridge."""
    frozen(layer)
    frozen(rope)
    if not prompts:
        raise ValueError("no training prompts")
    legacy = legacy_graft()
    up, bias = nn.Parameter(initial.up.clone()), nn.Parameter(initial.up_bias.clone())
    optimizer = torch.optim.Adam([up, bias], lr=config["delta_learning_rate"])
    order = torch.randperm(len(prompts), generator=torch.Generator().manual_seed(config["seed"])).tolist()
    history = []
    for step in range(config["delta_steps"]):
        if check_budget:
            check_budget()
        started = time.perf_counter()
        legacy.memory_guard(config["max_rss_gib"])
        prompt = prompts[order[step % len(order)]]
        small, big, target_delta = (prompt[k].detach().unsqueeze(0) for k in ("small", "big", "delta"))
        optimizer.zero_grad(set_to_none=True)
        projected = finite(legacy.normalize(small) @ up + bias)
        delta = donor_delta(layer, rope, projected, config=config)
        delta_loss = normalized_loss(delta, target_delta, token_normalized=token_normalized)
        state_loss = normalized_loss(projected, big, token_normalized=token_normalized)
        cosine_loss = (1-F.cosine_similarity(delta, target_delta, dim=-1)).mean()
        loss = finite(delta_loss + config["state_anchor_weight"]*state_loss +
                      config["delta_cosine_weight"]*cosine_loss)
        loss.backward()
        if any(p.grad is not None for p in layer.parameters()):
            raise RuntimeError("unexpected donor gradient")
        finite(up.grad); finite(bias.grad)
        nn.utils.clip_grad_norm_([up, bias], config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        finite(up); finite(bias)
        deadline(started, config["max_prompt_seconds"])
        legacy.memory_guard(config["max_rss_gib"])
        if check_budget:
            check_budget()
        history.append({"step": step, "training_prompt_index": order[step % len(order)],
                        "loss": float(loss.detach()), "delta_loss": float(delta_loss.detach())})
    frozen(layer)
    frozen(rope)
    return legacy.Bridges(up.detach(), bias.detach(), initial.down, initial.down_bias), history
