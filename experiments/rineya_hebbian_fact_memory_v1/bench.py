"""CPU mechanism studies. These are NOT pretrained Rineya efficacy results."""
from __future__ import annotations

import time
import torch
from torch import Tensor, nn
from .memory import BuildConfig, FrozenBilinear, build, ridge_readout, swap_readout, effective_kernel
from .metrics import recall, unit, geometry, noisy_queries, paired


def synthetic_facts(n: int, d: int, seed: int, anisotropy: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
    if n < 2 or d < 4 or not 0 <= anisotropy <= 0.95:
        raise ValueError("invalid synthetic geometry")
    rng = torch.Generator().manual_seed(seed)
    k = torch.randn(n, d, generator=rng, dtype=torch.float64)
    v = torch.randn(n, d, generator=rng, dtype=torch.float64)
    # Shared direction produces controllable key/value crowding, normalized after.
    for x in (k, v):
        x[:, 0] += anisotropy * d**0.5 / max(1-anisotropy, .05)
    return unit(k), unit(v), torch.randperm(n, generator=rng)


def controls(k: Tensor, v: Tensor, labels: Tensor, width: int, seed: int,
             gd_steps: int = 0) -> dict:
    targets = v[labels]
    linear, solve = ridge_readout(k, targets, 1e-6)
    rng = torch.Generator().manual_seed(seed)
    random_model = FrozenBilinear(torch.randn(width, k.shape[1], generator=rng, dtype=k.dtype),
                                  torch.randn(width, k.shape[1], generator=rng, dtype=k.dtype),
                                  torch.randn(v.shape[1], width, generator=rng, dtype=k.dtype))
    shuffled = labels.roll(1)  # A derangement relative to the mapping, not chance fixed points.
    wrong, _ = build(k, v[shuffled], BuildConfig(width=width, seed=seed))
    out = {
        "linear_ridge": {"metrics": recall(k @ linear.T, v, labels), "stored_scalars": linear.numel(), "solve": solve},
        "exact_quadratic_lookup": {"metrics": recall((k @ k.T).square() @ targets/len(k), v, labels),
                                    "stored_scalars": k.numel()+targets.numel(), "external_key_table": True},
        "cosine_nearest_neighbor": {"metrics": recall(targets[(unit(k)@unit(k).T).argmax(1)], v, labels),
                                     "stored_scalars": k.numel()+targets.numel(), "external_key_table": True},
        "random_unfitted": {"metrics": recall(random_model(k), v, labels), **random_model.storage()},
        "shuffled_mapping": {"metrics": recall(wrong(k), v, labels), **wrong.storage()},
    }
    if gd_steps:
        # Matched architecture and scalar budget; only this control uses gradients.
        a, g, b = [nn.Parameter(t.clone()) for t in (random_model.a, random_model.g, random_model.b * .01)]
        optimizer = torch.optim.Adam([a, g, b], lr=.01)
        for _ in range(gd_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (((k@a.T)*(k@g.T))@b.T-targets).square().mean()
            loss.backward()
            optimizer.step()
        out["gd_bilinear"] = {"metrics": recall(((k@a.T)*(k@g.T))@b.T, v, labels),
                              "optimizer_steps": gd_steps,
                              "stored_scalars": sum(t.numel() for t in (a, g, b))}
    return out


def study(n: int = 32, d: int = 24, width: int = 64, seed: int = 17,
          anisotropy: float = 0.0, gd_steps: int = 0) -> dict:
    k, v, labels = synthetic_facts(n, d, seed, anisotropy)
    result = {"kind": "SYNTHETIC_MECHANISM_ONLY", "native_qualified": False,
              "n": n, "dimension": d, "width": width, "seed": seed,
              "anisotropy": anisotropy, "arms": {}, "controls": controls(k,v,labels,width,seed,gd_steps)}
    for variant in ("unwhitened", "diagonal", "whitened", "data_dependent"):
        cfg = BuildConfig(width=width, seed=seed, variant=variant)
        started = time.perf_counter()
        try:
            model, construction = build(k, v[labels], cfg)
        except RuntimeError as exc:
            result["arms"][variant] = {"status": "BLOCKED_NUMERICS", "reason": str(exc)}
            continue
        arm = {"status": "MEASURED", "construction_seconds": time.perf_counter()-started,
               "construction": construction, "noise": {}, "precision": {}}
        for eps in (0.0, .1, .3):
            q = noisy_queries(k, eps, seed+1000)
            arm["noise"][str(eps)] = recall(model(q), v, labels)
        arm["geometry"] = geometry(effective_kernel(model.features(k), cfg), v, labels)
        for dtype in (torch.float32, torch.bfloat16):
            # Storage-rounding study with FP64 arithmetic; NOT native BF16 kernels.
            rounded = FrozenBilinear(*(t.to(dtype).double() for t in (model.a, model.g, model.b)))
            arm["precision"][str(dtype)] = {"protocol": "weight_roundtrip_FP64_compute",
                                             "metrics": recall(rounded(k), v, labels)}
        # Change 10% of facts, rebuild only B; old predictions remain immutable.
        changed = max(1, n//10)
        new_labels = labels.clone()
        new_labels[:changed] = (labels[:changed]+1) % len(v)
        edited, edit_receipt = swap_readout(model, k, v[new_labels], cfg)
        before = recall(model(k), v, labels)
        after = recall(edited(k), v, new_labels)
        arm["edit"] = {"changed_facts": changed, "solve": edit_receipt,
                       "edited_success": recall(edited(k[:changed]), v, new_labels[:changed]),
                       "untouched": paired([p == int(t) for p,t in zip(before["predictions"][changed:],labels[changed:])],
                                           [p == int(t) for p,t in zip(after["predictions"][changed:],labels[changed:])])}
        # Sequential addition on fixed features; full previous bank is replayed.
        half = n//2
        try:
            first, _ = build(k[:half], v[labels[:half]], cfg)
            second, _ = swap_readout(first, k, v[labels], cfg)
            arm["sequential"] = {"first_half_before": recall(first(k[:half]), v, labels[:half]),
                                  "first_half_after": recall(second(k[:half]), v, labels[:half]),
                                  "new_half": recall(second(k[half:]), v, labels[half:]),
                                  "full_key_bank_rebuild": True}
        except RuntimeError as exc:
            arm["sequential"] = {"status": "BLOCKED_NUMERICS", "reason": str(exc),
                                 "full_key_bank_rebuild": True}
        result["arms"][variant] = arm
    return result


class TinyAttention(nn.Module):
    """One causal query-position attention; identity V/O and no residual bypass."""
    def __init__(self, d: int, seed: int):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.q = nn.Parameter(torch.randn(d,d,generator=rng,dtype=torch.float64)/d**.5)
        self.k = nn.Parameter(torch.randn(d,d,generator=rng,dtype=torch.float64)/d**.5)

    def forward(self, sequence: Tensor) -> Tensor:
        query = sequence[:,-1] @ self.q.T
        keys = sequence @ self.k.T
        weights = torch.softmax(torch.einsum("bd,btd->bt",query,keys)/query.shape[-1]**.5, dim=1)
        return torch.einsum("bt,btd->bd",weights,sequence)


def ssfr(seed: int = 17, steps: int = 400, junk: int = 6,
         n: int = 16, d: int = 16, width: int = 64) -> dict:
    """Train the reader on mapping A; freeze and swap to unseen mapping B.

    Synthetic tagged key/query channels simplify routing. Not the exact paper
    architecture, a GPT replacement, or a language-understanding benchmark.
    """
    if steps < 1 or junk < 0:
        raise ValueError("invalid training budget")
    k, v, labels_a = synthetic_facts(n,d,seed)
    k[:,-2:] = 0
    k = unit(k)
    k[:,-2] = 1
    k = unit(k)
    rng = torch.Generator().manual_seed(seed+313)
    config = BuildConfig(width=width,seed=seed)
    memory_a, _ = build(k,v[labels_a],config)
    attention = TinyAttention(d,seed)
    optimizer = torch.optim.Adam(attention.parameters(),lr=.03)

    def batch(indices: Tensor, generator: torch.Generator) -> Tensor:
        s = torch.randn(len(indices),junk+2,d,generator=generator,dtype=k.dtype)/d**.5
        s[:,:,-2:] = 0
        positions = torch.randint(junk+1,(len(indices),),generator=generator)
        s[torch.arange(len(indices)),positions] = k[indices]
        s[:,-1] = 0
        s[:,-1,-1] = 1
        return s

    for _ in range(steps):
        ids = torch.randint(n,(32,),generator=rng)
        prediction = memory_a(attention(batch(ids,rng)))
        loss = (prediction-v[labels_a[ids]]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    attention.requires_grad_(False)
    labels_b = labels_a.roll(1)
    memory_b, _ = swap_readout(memory_a,k,v[labels_b],config)
    eval_rng = torch.Generator().manual_seed(seed+9999)
    ids = torch.arange(n).repeat(8)
    with torch.no_grad():
        queries = attention(batch(ids,eval_rng))
        a = recall(memory_a(queries),v,labels_a[ids])
        b = recall(memory_b(queries),v,labels_b[ids])
        stale = recall(memory_a(queries),v,labels_b[ids])
    return {"kind":"SYNTHETIC_FACT_ADAPTIVE_READER", "native_qualified":False,
            "seed":seed,"steps":steps,"junk":junk,"n":n,"d":d,"width":width,
            "train_mapping_a_heldout_context":a,"swapped_mapping_b":b,
            "stale_memory_negative_control":stale,
            "maximum_query_l2_error":float((queries-k[ids]).norm(dim=1).max()),
            "trainable_attention_scalars":sum(p.numel() for p in attention.parameters()),
            "memory_scalars":memory_a.storage()["stored_scalars"],
            "residual_bypass":False,"value_output_identity":True,
            "eval_rng_disjoint":True, "key_query_tag_channels":True}
