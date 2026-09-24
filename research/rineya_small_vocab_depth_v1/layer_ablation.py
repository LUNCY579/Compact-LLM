"""SMALLVD-001 Phase 7 -- causal layer-ablation evaluator.

A block is bypassed as h_(l+1) = h_l: the decoder layer is removed from the ModuleList for the
duration of a measurement and restored afterwards. That is EXACTLY what depth surgery does, so a
configuration measured here and the corresponding surgery-built student must produce identical
logits. `test_surgery.py` asserts this on disk, so the evaluator cannot quietly measure a different
model from the one that gets built.

Why the renumbering. HF's DynamicCache indexes KV by `self_attn.layer_idx` and reads the sequence
length from layer 0. With layer 0 bypassed and the old indices kept, cached generation would silently
use position 0 for every step. Kept layers are renumbered 0..L-1 while a configuration is active,
which is what the built student has too.

MEASUREMENTS per configuration, on teacher-forced calibration rows (reference answers appended):
  * KL(teacher || ablated) at every ANSWER position, and at every position of plain-text rows
  * top-1 agreement with the teacher, top-5 overlap
  * NLL of the reference answer, and its change from the teacher's
  * per-sequence mean KL -> mean, P95, max (a block can be harmless on average and decisive on
    rare rows, and the tail is where that shows)
  * all of the above PER CATEGORY, so a skill-specific dependency cannot hide in the average
and, on the generation panel, greedy decoding:
  * answer flip rate against the TEACHER's own output (no reference needed)
  * accuracy against the reference with the row's declared scorer
  * refusal/abstention rate, mean output length, and rate of hitting the token cap

Teacher distributions are never stored (151,936 floats per position). The teacher's final-norm
hidden state at each measured position IS cached (4 KB/position), and the teacher's log-softmax is
recomputed from it through the tied head, chunk by chunk.

Diagnostics that are reported but NEVER used alone to call a block redundant:
  ||F_l(h_l)|| / ||h_l||, cos(h_l, h_(l+1)), linear CKA(h_l, h_(l+1)), angular distance
  d(h_l, h_(l+n)) for consecutive-block baselines.
"""
from __future__ import annotations

import contextlib
import json
import math
import re
from collections import defaultdict

import torch

ABSTAIN_RE = re.compile(r"\b(unknown|cannot be determined|not (?:stated|mentioned|specified|given|provided)|"
                        r"no information|insufficient|unanswerable|can't (?:tell|determine)|"
                        r"cannot (?:tell|answer))\b", re.I)


def configure_runtime(seed: int = 17) -> dict:
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"tf32": False, "matmul_precision": "highest", "seed": seed}


@contextlib.contextmanager
def keep_layers(model, kept: list[int]):
    """Temporarily keep only `kept` decoder layers (original order), renumbered 0..L-1."""
    base = model.model.layers
    orig = list(base)
    old_idx = [layer.self_attn.layer_idx for layer in orig]
    try:
        chosen = [orig[i] for i in kept]
        for new, layer in enumerate(chosen):
            layer.self_attn.layer_idx = new
        model.model.layers = torch.nn.ModuleList(chosen)
        yield model
    finally:
        model.model.layers = base
        for layer, i in zip(orig, old_idx):
            layer.self_attn.layer_idx = i


def pad_batch(seqs: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """RIGHT padding for teacher forcing: real positions keep their absolute position ids."""
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def final_hidden(model, ids, mask):
    out = model.model(input_ids=ids, attention_mask=mask, use_cache=False)
    return out.last_hidden_state


def batches_by_length(items: list[dict], max_tokens: int):
    """Length-sorted batches under a token budget (padding waste stays small)."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    batch, longest = [], 0
    for i in order:
        n = len(items[i]["ids"])
        if batch and max(longest, n) * (len(batch) + 1) > max_tokens:
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, n)
    if batch:
        yield batch


@torch.no_grad()
def collect_positions(model, items: list[dict], pad_id: int, device, max_tokens: int = 8192):
    """Final-norm hidden states at every measured position. item['score_pos'] are indices t whose
    hidden state predicts token t+1; item['targets'] the reference next tokens."""
    H = [None] * len(items)
    for b in batches_by_length(items, max_tokens):
        ids, mask = pad_batch([items[i]["ids"] for i in b], pad_id, device)
        h = final_hidden(model, ids, mask)
        for j, i in enumerate(b):
            pos = torch.tensor(items[i]["score_pos"], device=device)
            H[i] = h[j].index_select(0, pos).float()
    return H


@torch.no_grad()
def compare(model_head, H_teacher: list, H_student: list, items: list[dict], chunk: int = 2048,
            vocab_slice: torch.Tensor | None = None) -> dict:
    """Per-item KL, top-1/top-5 agreement and reference NLL, from cached final hidden states.

    `vocab_slice` restricts BOTH distributions to a set of teacher ids (used only when comparing
    across a vocabulary change; depth-only comparisons use the full row set)."""
    W = model_head.weight
    per_item = []
    for i, it in enumerate(items):
        ht, hs = H_teacher[i], H_student[i]
        tgt = torch.tensor(it["targets"], device=W.device)
        kl_sum = top1 = top5 = 0.0
        nll_s = nll_t = 0.0
        for s in range(0, ht.shape[0], chunk):
            lt = ht[s:s + chunk] @ W.T.float()
            ls = hs[s:s + chunk] @ W.T.float()
            if vocab_slice is not None:
                lt, ls = lt[:, vocab_slice], ls[:, vocab_slice]
            lpt, lps = torch.log_softmax(lt, -1), torch.log_softmax(ls, -1)
            kl_sum += (lpt.exp() * (lpt - lps)).sum().item()
            a_t, a_s = lpt.argmax(-1), lps.argmax(-1)
            top1 += (a_t == a_s).sum().item()
            t5 = lps.topk(5, -1).indices
            top5 += (t5 == a_t[:, None]).any(-1).sum().item()
            g = tgt[s:s + chunk]
            if vocab_slice is None:
                nll_s += -lps.gather(1, g[:, None]).sum().item()
                nll_t += -lpt.gather(1, g[:, None]).sum().item()
        n = ht.shape[0]
        per_item.append({"id": it["id"], "category": it["category"], "n": n,
                         "kl": kl_sum / n, "top1": top1 / n, "top5": top5 / n,
                         "nll": nll_s / n if vocab_slice is None else None,
                         "nll_teacher": nll_t / n if vocab_slice is None else None})
    return summarise(per_item)


def pctl(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarise(per_item: list[dict]) -> dict:
    def agg(rows):
        n = sum(r["n"] for r in rows)
        out = {"rows": len(rows), "positions": n,
               "kl_mean_pos": sum(r["kl"] * r["n"] for r in rows) / n,
               "kl_mean_row": sum(r["kl"] for r in rows) / len(rows),
               "kl_p95_row": pctl([r["kl"] for r in rows], 0.95),
               "kl_max_row": max(r["kl"] for r in rows),
               "top1": sum(r["top1"] * r["n"] for r in rows) / n,
               "top5": sum(r["top5"] * r["n"] for r in rows) / n}
        if rows[0]["nll"] is not None:
            out["nll"] = sum(r["nll"] * r["n"] for r in rows) / n
            out["nll_teacher"] = sum(r["nll_teacher"] * r["n"] for r in rows) / n
            out["delta_nll"] = out["nll"] - out["nll_teacher"]
        return out
    cats = defaultdict(list)
    for r in per_item:
        cats[r["category"]].append(r)
    return {"all": agg(per_item), "by_category": {c: agg(v) for c, v in sorted(cats.items())},
            "per_item": per_item}


@torch.no_grad()
def greedy_generate(model, tokenizer, prompts: list[list[int]], eos_ids: list[int], pad_id: int,
                    device, max_new_tokens: int = 64, batch: int = 16) -> list[list[int]]:
    """Greedy, left-padded, KV-cached. Stops per row at the first EOS id."""
    outs = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        T = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), T), dtype=torch.long)
        for j, i in enumerate(idx):
            p = prompts[i]
            ids[j, T - len(p):] = torch.tensor(p)
            mask[j, T - len(p):] = 1
        g = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                           do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                           eos_token_id=eos_ids, pad_token_id=pad_id, temperature=None,
                           top_p=None, top_k=None)
        for j, i in enumerate(idx):
            new = g[j, T:].tolist()
            cut = next((k for k, t in enumerate(new) if t in eos_ids), None)
            outs[i] = new[:cut] if cut is not None else new
    return outs


def normalise_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    if s.endswith(".") and not s.endswith(".."):
        s = s[:-1]
    return s.strip().lower()


def score_row(output: str, row: dict) -> bool | None:
    ref = row.get("reference")
    if ref is None:
        return None
    kind = row.get("scorer", "exact")
    o, r = normalise_answer(output), normalise_answer(ref)
    if kind == "exact":
        return o == r
    if kind == "contains":
        return r in o
    if kind == "numeric":
        nums = re.findall(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
        return bool(nums) and nums[-1] == r.replace(",", "")
    if kind == "abstain":
        return bool(ABSTAIN_RE.search(output))
    raise ValueError(f"unknown scorer {kind}")


def generation_metrics(texts: list[str], teacher_texts: list[str], rows: list[dict],
                       new_token_counts: list[int], cap: int) -> dict:
    def agg(ix):
        n = len(ix)
        flips = sum(normalise_answer(texts[i]) != normalise_answer(teacher_texts[i]) for i in ix)
        scored = [score_row(texts[i], rows[i]) for i in ix]
        scored = [x for x in scored if x is not None]
        return {"rows": n, "flip_rate_vs_teacher": flips / n,
                "accuracy": (sum(scored) / len(scored)) if scored else None,
                "scored_rows": len(scored),
                "abstain_rate": sum(bool(ABSTAIN_RE.search(texts[i])) for i in ix) / n,
                "mean_new_tokens": sum(new_token_counts[i] for i in ix) / n,
                "cap_hit_rate": sum(new_token_counts[i] >= cap for i in ix) / n}
    cats = defaultdict(list)
    for i, r in enumerate(rows):
        cats[r["category"]].append(i)
    return {"all": agg(list(range(len(rows)))),
            "by_category": {c: agg(ix) for c, ix in sorted(cats.items())}}


@torch.no_grad()
def residual_diagnostics(model, items: list[dict], pad_id: int, device, max_tokens: int = 4096,
                         max_positions: int = 8192) -> dict:
    """Per-block input/output statistics on the TEACHER. Reported, never decisive alone."""
    layers = list(model.model.layers)
    L = len(layers)
    feats = [[] for _ in range(L + 1)]   # h_0 .. h_L (h_L = output of last block, PRE final norm)
    taken = 0
    hooks = []
    store = {}

    def mk(i):
        def hook(mod, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            y = out[0] if isinstance(out, tuple) else out
            store[i] = (x.detach(), y.detach())
        return hook
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(mk(i), with_kwargs=True))
    try:
        for b in batches_by_length(items, max_tokens):
            if taken >= max_positions:
                break
            ids, mask = pad_batch([items[i]["ids"] for i in b], pad_id, device)
            store.clear()
            model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            sel = mask.bool()
            sel[:, 0] = False            # the first position carries massive activations
            for i in range(L):
                x, y = store[i]
                if i == 0:
                    feats[0].append(x[sel].float().cpu())
                feats[i + 1].append(y[sel].float().cpu())
            taken += int(sel.sum())
    finally:
        for h in hooks:
            h.remove()
    Hs = [torch.cat(f)[:max_positions] for f in feats]

    def cos_rows(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    def cka(a, b):
        a = a - a.mean(0, keepdim=True)
        b = b - b.mean(0, keepdim=True)
        num = (b.T @ a).pow(2).sum()
        return (num / ((a.T @ a).pow(2).sum().sqrt() * (b.T @ b).pow(2).sum().sqrt())).item()

    per_block = []
    for i in range(L):
        x, y = Hs[i], Hs[i + 1]
        per_block.append({"layer": i,
                          "rel_update_norm": ((y - x).norm(dim=-1) / x.norm(dim=-1)).mean().item(),
                          "cos_in_out": cos_rows(x, y).mean().item(),
                          "cka_in_out": cka(x, y),
                          "block_influence": 1.0 - cos_rows(x, y).mean().item()})
    angular = {}
    for n in range(1, L):
        angular[n] = [math.acos(max(-1.0, min(1.0, cos_rows(Hs[l], Hs[l + n]).mean().item()))) / math.pi
                      for l in range(0, L - n + 1)]
    return {"positions": int(Hs[0].shape[0]), "per_block": per_block, "angular_distance": angular}


def dump(path, obj):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
