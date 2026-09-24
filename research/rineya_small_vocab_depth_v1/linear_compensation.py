"""SMALLVD-001 Amendment 2 (exploratory arm H-LC) -- training-free LINEAR COMPENSATION of dropped blocks.

Plain depth pruning bypasses a run of blocks: the residual after the last kept block p goes straight
into the next kept block. For each maximal run S = [s..e] of dropped blocks whose predecessor
p = s - 1 is kept, fit one d x d matrix T by ridge least squares so that block p's output, with its
MLP output m_p replaced by m_p T, approximates the TEACHER's residual after block e:

      y_p + m_p T  ~=  h_e(teacher)        y_p = the residual entering p's MLP add

and FOLD it into p's down_proj (row-vector convention: m_p = z W_down^T, so m_p T = z (T^T W_down)^T,
i.e. W_down <- T^T W_down). Nothing is added: the geometry, the parameter count and the file layout
are identical to plain pruning, and T = I is plain pruning exactly. (Closed-form, gradient-free;
after the "ReplaceMe" family of methods.)

Fitting order: runs are fitted first to last, each on STUDENT activations with every earlier fold
applied, against the teacher's own residual -- so a later run also corrects error accumulated
upstream. A run starting at block 0 has no predecessor and stays uncompensated (recorded).

Registered choices (fixed before any fit): fit set = calibration SELECT only (never SELECT2 or
REPORT); every real position except position 0 (the first token carries the attention-sink massive
activations, which would dominate an L2 fit).

RIDGE (Amendment 4, before any fit): min ||R - X T||^2 + lambda ||T - I||^2, i.e. shrink toward T = I
(plain pruning, the free null) -- NOT the textbook ||T||^2, which shrinks toward T = 0 and would
pull the fit toward DELETING block p's MLP output. lambda = 1e-3 x the MEDIAN eigenvalue of X^T X:
the first rule (1e-3 x the MEAN, i.e. tr/d) was coupled to the heavy tail of the activation
spectrum (a few massive dimensions inflate the mean), which can make the penalty a strong prior
over most of the space rather than numerical stabilisation (review COLD_REVIEW_HLC.md). Every fit
records the spectrum, lambda/median, lambda/max and the participation ratio, so the size of the
prior is visible rather than assumed.
"""
from __future__ import annotations

import contextlib

import torch

from layer_ablation import batches_by_length, keep_layers, pad_batch

RIDGE_REL = 1e-3          # x the MEDIAN eigenvalue of X^T X (Amendment 4)


def runs(kept: list[int], n_layers: int) -> list[dict]:
    """Maximal runs of dropped blocks, each with its kept predecessor (None if the run starts at 0)."""
    ks = set(kept)
    out, l = [], 0
    while l < n_layers:
        if l in ks:
            l += 1
            continue
        s = l
        while l < n_layers and l not in ks:
            l += 1
        out.append({"p": s - 1 if s > 0 else None, "s": s, "e": l - 1})
    return out


@contextlib.contextmanager
def folded(model, folds: dict[int, torch.Tensor]):
    """Temporarily fold T into block p's down_proj for every (p, T); restores the exact originals.
    p is an ORIGINAL block index, so this must be entered OUTSIDE keep_layers (inside it, index p
    would address a renumbered block -- the wrong one, silently). Refuses rather than guess."""
    blocks = model.model.layers
    if len(blocks) != model.config.num_hidden_layers:
        raise RuntimeError("folded() entered inside keep_layers: block indices would be renumbered")
    for p in folds:
        if blocks[p].mlp.down_proj.weight.dtype != torch.float32:
            # a small T - I survives fp32 but can round away ENTIRELY in bf16 (8 mantissa bits), which
            # would make H-LC measure exactly as plain for a reason unconnected to the hypothesis
            raise RuntimeError("folding requires fp32 weights")
    saved = {}
    try:
        with torch.no_grad():
            for p, T in folds.items():
                W = blocks[p].mlp.down_proj.weight
                saved[p] = W.detach().clone()
                W.copy_((T.to(W.dtype).T @ W).contiguous())
        yield model
    finally:
        with torch.no_grad():
            for p, W0 in saved.items():
                blocks[p].mlp.down_proj.weight.copy_(W0)


def _first(x):
    return x[0] if isinstance(x, tuple) else x


@torch.no_grad()
def fit_compensation(model, kept: list[int], items: list[dict], pad_id: int, device,
                     ridge_rel: float = RIDGE_REL, max_tokens: int = 8192) -> tuple[dict, list[dict]]:
    """Returns ({p: T}, per-run fit records). Must be called with the FULL layer list in place (not
    inside keep_layers): block objects are addressed by their original index."""
    blocks = list(model.model.layers)
    L = len(blocks)
    d = model.config.hidden_size
    folds, records = {}, []
    for r in runs(kept, L):
        if r["p"] is None:
            records.append({**r, "compensated": False, "reason": "run starts at block 0"})
            continue
        p, e = r["p"], r["e"]
        XtX = torch.zeros(d, d, dtype=torch.float64, device=device)
        XtR = torch.zeros(d, d, dtype=torch.float64, device=device)
        RtR, n = 0.0, 0
        cap = {}
        h_t = blocks[e].register_forward_hook(lambda m, i, o: cap.__setitem__("h_e", _first(o)))
        for b in batches_by_length(items, max_tokens):
            ids, mask = pad_batch([items[i]["ids"] for i in b], pad_id, device)
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)       # teacher, no folds
            h_e = cap.pop("h_e")
            h1 = blocks[p].post_attention_layernorm.register_forward_pre_hook(
                lambda m, i: cap.__setitem__("y", i[0]))
            h2 = blocks[p].mlp.register_forward_hook(lambda m, i, o: cap.__setitem__("m", o))
            try:
                with folded(model, folds), keep_layers(model, kept):
                    model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            finally:
                h1.remove()
                h2.remove()
            y, m = cap.pop("y"), cap.pop("m")
            sel = mask.bool().clone()
            sel[:, 0] = False                                   # registered: drop position 0
            X = m[sel].double()
            R = (h_e[sel] - y[sel]).double()
            XtX += X.T @ X
            XtR += X.T @ R
            RtR += float((R * R).sum())
            n += int(sel.sum())
        h_t.remove()
        ev = torch.linalg.eigvalsh(XtX)                     # ascending
        med = float(ev[d // 2])
        lam = ridge_rel * med
        eye = torch.eye(d, dtype=torch.float64, device=device)
        T = torch.linalg.solve(XtX + lam * eye, XtR + lam * eye)       # shrink toward I, not 0
        spectrum = {"eig_min": float(ev[0]), "eig_p10": float(ev[d // 10]), "eig_median": med,
                    "eig_mean": float(ev.mean()), "eig_max": float(ev[-1]),
                    "lambda_over_median": lam / med if med > 0 else None,
                    "lambda_over_max": lam / float(ev[-1]),
                    "participation_ratio": float(torch.trace(XtX)) ** 2 / float((XtX * XtX).sum())}

        def resid(M):              # ||R - X M||^2 from the accumulated moments
            return RtR - 2 * float(torch.trace(M.T @ XtR)) + float(torch.trace(M.T @ XtX @ M))
        r_id, r_fit = resid(eye), resid(T)
        folds[p] = T.float()
        records.append({**r, "compensated": True, "positions": n, "ridge_lambda": lam, "spectrum": spectrum,
                        "fit_residual_rel_identity": r_id / RtR, "fit_residual_rel_fitted": r_fit / RtR,
                        "T_minus_I_fro_rel": float(torch.linalg.norm(T - eye) / torch.linalg.norm(eye))})
    return folds, records
