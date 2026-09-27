"""Functional interaction between skill deltas: the 2x2 design f(theta), f(theta+dA),
f(theta+dB), f(theta+dA+dB).

What each quantity measures (and what it does not):

* ``I_lp(x) = lp_AB - lp_A - lp_B + lp_0`` is the second-order (interaction) term of the
  TEACHER-FORCED log-probability of ONE reference answer, summed over its answer tokens. It is 0
  when the two deltas' effects on that log-probability add exactly. It says nothing about other
  valid answers (the tasks accept any valid certificate) and nothing about free generation.
* ``kappa = mean I_lp / mean(lp_A - lp_0)`` normalises the interaction by the first delta's own
  effect (FA-001's estimator, vendored). It is undefined when that effect is ~0.
* ``vocab_interaction`` is the same second difference applied to the full log-softmax VECTOR at
  each answer position. Using log-softmax (not raw logits) removes the softmax's per-position
  additive gauge; a raw-logit interaction would include a meaningless constant shift. Its L2 norm
  over the whole vocabulary is dominated by tail tokens, so it is also reported restricted to the
  union of each configuration's top-k tokens.

None of these is a measure of composition quality. Composition quality is exact-verification
accuracy on held-out combination signatures (``metrics``). Log-probabilities are unclipped; FA-001
showed that a clip can flip the verdict, so no clip is applied here.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

CONFIGS = ("0", "A", "B", "AB")


def _rows(Z: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
    if set(Z) != set(CONFIGS):
        raise ValueError(f"need exactly configs {CONFIGS}, got {sorted(Z)}")
    out = {k: np.asarray(Z[k], dtype=np.float64).reshape(-1) for k in CONFIGS}
    n = {v.size for v in out.values()}
    if len(n) != 1 or 0 in n:
        raise ValueError("configs must score the same non-empty row set")
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("non-finite log-probabilities")
    return out


def interaction_rows(Z: Mapping[str, Sequence[float]]) -> np.ndarray:
    z = _rows(Z)
    return z["AB"] - z["A"] - z["B"] + z["0"]


def label(point: float, lo: float, hi: float, margin: float = 0.10) -> str:
    """FA-001 registered outcome rule (amend_analysis.label, commit 6903e583), margin exposed."""
    if lo > -margin and hi < margin:
        return "ADDITIVE"
    if hi < 0 and point <= -margin:
        return "SUB-ADDITIVE"
    if lo > 0 and point >= margin:
        return "SUPER-ADDITIVE"
    return "UNRESOLVED"


def kappa(Z: Mapping[str, Sequence[float]], seed: int = 20260927, n_boot: int = 10000,
          normaliser: str = "A") -> dict:
    """FA-001 kappa (amend_analysis.kappa, commit 6903e583) with a fresh RNG per call.

    ``normaliser`` = "A" (FA-001's choice, the first delta's own gain) or "AB_sum" (the sum of
    both single-delta gains, symmetric in A and B).
    """
    z = _rows(Z)
    I = z["AB"] - z["A"] - z["B"] + z["0"]
    if normaliser == "A":
        g = z["A"] - z["0"]
    elif normaliser == "AB_sum":
        g = (z["A"] - z["0"]) + (z["B"] - z["0"])
    else:
        raise ValueError(normaliser)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, I.size, size=(n_boot, I.size))
    gm = g[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = I[idx].mean(axis=1) / gm
    point = I.mean() / g.mean() if g.mean() != 0 else float("nan")
    finite = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(finite, [2.5, 97.5]) if finite.size else (float("nan"), float("nan")))
    return {"kappa": float(point), "ci95": [float(lo), float(hi)],
            "label_under_fa001_rule": label(point, lo, hi) if math.isfinite(point) else "UNDEFINED",
            "normaliser": normaliser, "mean_I": float(I.mean()), "mean_gain": float(g.mean()),
            "rows": int(I.size), "rows_I_negative": int((I < 0).sum()),
            "bootstrap_draws_nonfinite": int(boot.size - finite.size), "seed": seed}


def mean_ci(values: Sequence[float], seed: int = 20260927, n_boot: int = 10000) -> dict:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("empty")
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci95": [float(lo), float(hi)], "n": int(v.size), "seed": seed}


def vocab_interaction(logsm: Mapping[str, "object"], topk: int = 32) -> dict:
    """Per-position interaction of full log-softmax vectors.

    ``logsm[c]`` is a torch or numpy array [positions, vocab] of log-softmax values for config c,
    over the SAME positions. Returns per-position L2 norms of the interaction, of each single
    delta's effect, and the same restricted to the union of each config's top-``topk`` tokens.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if set(logsm) != set(CONFIGS):
        raise ValueError(f"need exactly configs {CONFIGS}")
    if torch is not None and all(isinstance(v, torch.Tensor) for v in logsm.values()):
        t = {k: v.double() for k, v in logsm.items()}
        I = t["AB"] - t["A"] - t["B"] + t["0"]
        eA, eB = t["A"] - t["0"], t["B"] - t["0"]
        union = torch.zeros_like(I, dtype=torch.bool)
        for v in t.values():
            union.scatter_(1, v.topk(topk, dim=1).indices, True)
        full = lambda x: x.norm(dim=1)
        restricted = lambda x: (x * union).norm(dim=1)
        out = {"I_full": full(I), "A_full": full(eA), "B_full": full(eB),
               "I_topk": restricted(I), "A_topk": restricted(eA), "B_topk": restricted(eB),
               "union_size": union.sum(dim=1).double()}
        return {k: v.cpu().numpy() for k, v in out.items()} | {"topk": topk}
    a = {k: np.asarray(v, dtype=np.float64) for k, v in logsm.items()}
    I = a["AB"] - a["A"] - a["B"] + a["0"]
    eA, eB = a["A"] - a["0"], a["B"] - a["0"]
    union = np.zeros(I.shape, dtype=bool)
    for v in a.values():
        top = np.argpartition(-v, topk - 1, axis=1)[:, :topk]
        np.put_along_axis(union, top, True, axis=1)
    n = lambda x: np.linalg.norm(x, axis=1)
    return {"I_full": n(I), "A_full": n(eA), "B_full": n(eB), "I_topk": n(I * union),
            "A_topk": n(eA * union), "B_topk": n(eB * union),
            "union_size": union.sum(axis=1).astype(np.float64), "topk": topk}


def summarise_vocab(res: Mapping[str, np.ndarray]) -> dict:
    """Relative interaction = ||I|| / (||eA|| + ||eB||), per position, then mean/median."""
    out = {"topk": int(res["topk"]), "positions": int(np.asarray(res["I_full"]).size)}
    for scope in ("full", "topk"):
        I = np.asarray(res[f"I_{scope}"], dtype=np.float64)
        den = np.asarray(res[f"A_{scope}"]) + np.asarray(res[f"B_{scope}"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(den > 0, I / den, np.nan)
        out[scope] = {"mean_norm_I": float(I.mean()), "median_norm_I": float(np.median(I)),
                      "mean_relative": float(np.nanmean(rel)) if np.isfinite(rel).any() else None,
                      "median_relative": float(np.nanmedian(rel)) if np.isfinite(rel).any() else None}
    return out


def accuracy_effects(acc: Mapping[str, float], acc_rows_a: Mapping[str, float] | None = None,
                     acc_rows_b: Mapping[str, float] | None = None) -> dict:
    """Accuracy-level 2x2 on one signature's rows (e.g. AB rows).

    composition_gain   = acc_AB - max(acc_A, acc_B)       (> 0: the pair beats either alone)
    additive_residual  = acc_AB - acc_A - acc_B + acc_0   (accuracy-scale interaction)
    If single-skill row accuracies are supplied, also:
    retention_A = acc_AB(on A rows) - acc_A(on A rows); acquisition_B = acc_AB(on B rows) - acc_B(on B rows).
    """
    for k in CONFIGS:
        if k not in acc:
            raise ValueError(f"missing config {k}")
    out = {"composition_gain": acc["AB"] - max(acc["A"], acc["B"]),
           "composition_damage": max(0.0, max(acc["A"], acc["B"]) - acc["AB"]),
           "additive_residual": acc["AB"] - acc["A"] - acc["B"] + acc["0"]}
    if acc_rows_a is not None:
        out["retention_A_under_B"] = acc_rows_a["AB"] - acc_rows_a["A"]
    if acc_rows_b is not None:
        out["retention_B_under_A"] = acc_rows_b["AB"] - acc_rows_b["B"]
    return out
