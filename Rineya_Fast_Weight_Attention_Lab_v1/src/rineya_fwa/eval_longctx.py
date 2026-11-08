"""Scoring for the P2b long-context suite.

One forward pass per item. The answer position is scored by argmax over the
item's candidate token ids, which are all single tokens in the bare
answer-position form. Compared with the P1/P2 free-generation scorer this:

* removes the malformed-output confound (P2's only arm-A "miss" was the string
  "color slate" rather than a wrong colour);
* removes the abstention offset from the comparison, because UNKNOWN is not a
  candidate -- abstention is measured separately and reported, not silently
  folded into the error rate;
* costs one forward instead of ``max_new_tokens`` forwards. The FWA wrapper
  refuses KV-cache mixing, so free generation costs O(new_tokens * T) and is
  what made long-context evaluation unaffordable before.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from rineya_fwa.data_longctx import INTERFERENCE_CATEGORIES


def encode_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> torch.Tensor:
    try:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", enable_thinking=False,
        )
    except TypeError:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
    ids = enc.input_ids if hasattr(enc, "input_ids") else enc
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids


def answer_logits(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> tuple[torch.Tensor, int]:
    device = next(model.parameters()).device
    ids = encode_prompt(tokenizer, messages).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1)
    return out.logits[0, -1].float().cpu(), int(ids.shape[1])


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _mean(xs: list[bool]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def score_position_suite(
    model: Any, tokenizer: Any, items: list[dict[str, Any]], *, abstain_token_id: int | None = None
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        logits, n_prompt = answer_logits(model, tokenizer, item["messages"])
        cand = torch.tensor(item["candidate_token_ids"], dtype=torch.long)
        cand_logits = logits[cand]
        best = int(cand[int(cand_logits.argmax())])
        gold = int(item["gold_token_id"])
        # margin between gold and the best non-gold candidate
        mask = cand != gold
        best_other = float(cand_logits[mask].max()) if bool(mask.any()) else float("-inf")
        row = {
            "id": item["id"],
            "position": item["position"],
            "target_length": item["target_length"],
            "n_context_tokens": item["n_context_tokens"],
            "n_prompt_tokens": n_prompt,
            "gold": item["answer"],
            "pred_token_id": best,
            "exact": best == gold,
            "gold_logit": float(logits[gold]),
            "best_other_logit": best_other,
            "margin": float(logits[gold]) - best_other,
            "unconstrained_argmax": int(logits.argmax()),
            "unconstrained_is_gold": int(logits.argmax()) == gold,
        }
        if abstain_token_id is not None:
            row["abstain_logit"] = float(logits[abstain_token_id])
            row["abstain_is_argmax"] = int(logits.argmax()) == abstain_token_id
        rows.append(row)

    positions = sorted({r["position"] for r in rows})
    lengths = sorted({r["target_length"] for r in rows})
    by_position = {}
    for p in positions:
        cell = [r["exact"] for r in rows if r["position"] == p]
        k, n = sum(cell), len(cell)
        by_position[f"{p:.2f}"] = {"exact": _mean(cell), "k": k, "n": n, "ci95": wilson(k, n)}
    by_length = {}
    for L in lengths:
        cell = [r["exact"] for r in rows if r["target_length"] == L]
        k, n = sum(cell), len(cell)
        by_length[str(L)] = {"exact": _mean(cell), "k": k, "n": n, "ci95": wilson(k, n)}
    by_cell = {}
    for L in lengths:
        for p in positions:
            cell = [r["exact"] for r in rows if r["target_length"] == L and r["position"] == p]
            if cell:
                by_cell[f"{L}@{p:.2f}"] = {"exact": _mean(cell), "k": sum(cell), "n": len(cell)}

    middle = [r["exact"] for r in rows if r["position"] in (0.25, 0.50, 0.75)]
    edge = [r["exact"] for r in rows if r["position"] in (0.05, 0.10, 0.90, 0.95)]
    k, n = sum(r["exact"] for r in rows), len(rows)
    return {
        "n": n,
        "exact": _mean([r["exact"] for r in rows]),
        "exact_ci95": wilson(k, n),
        "unconstrained_exact": _mean([r["unconstrained_is_gold"] for r in rows]),
        "abstain_argmax_rate": _mean([r.get("abstain_is_argmax", False) for r in rows]),
        "mean_margin": sum(r["margin"] for r in rows) / n if n else float("nan"),
        "by_position": by_position,
        "by_length": by_length,
        "by_cell": by_cell,
        "middle_exact": _mean(middle),
        "edge_exact": _mean(edge),
        "middle_edge_gap": _mean(middle) - _mean(edge),
        "rows": rows,
    }


def score_interference(
    model: Any, tokenizer: Any, items: list[dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        logits, n_prompt = answer_logits(model, tokenizer, item["messages"])
        cand = torch.tensor(item["candidate_token_ids"], dtype=torch.long)
        best = int(cand[int(logits[cand].argmax())])
        gold = int(item["gold_token_id"])
        sup = item.get("superseded_token_id")
        if best == gold:
            label = "retained"
        elif sup is not None and best == sup:
            label = "overwritten"  # kept the stale value through a correction
        else:
            label = "wrong"
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "n_context_tokens": item["n_context_tokens"],
            "n_prompt_tokens": n_prompt,
            "gold": item["answer"],
            "pred_token_id": best,
            "exact": best == gold,
            "label": label,
            "gold_logit": float(logits[gold]),
            "superseded_logit": float(logits[sup]) if sup is not None else None,
        })
    by_cat = {}
    for c in INTERFERENCE_CATEGORIES:
        cell = [r for r in rows if r["category"] == c]
        k, n = sum(r["exact"] for r in cell), len(cell)
        by_cat[c] = {
            "exact": _mean([r["exact"] for r in cell]),
            "k": k,
            "n": n,
            "ci95": wilson(k, n),
            "labels": {lab: sum(1 for r in cell if r["label"] == lab) for lab in ("retained", "overwritten", "wrong")},
        }
    k, n = sum(r["exact"] for r in rows), len(rows)
    return {
        "n": n,
        "exact": _mean([r["exact"] for r in rows]),
        "exact_ci95": wilson(k, n),
        "by_category": by_cat,
        "rows": rows,
    }


def mcnemar_exact(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired exact test on matched item ids. b01/b10 are the discordant counts."""
    by_a = {r["id"]: bool(r["exact"]) for r in rows_a}
    by_b = {r["id"]: bool(r["exact"]) for r in rows_b}
    shared = sorted(set(by_a) & set(by_b))
    b01 = sum(1 for i in shared if not by_a[i] and by_b[i])   # b better
    b10 = sum(1 for i in shared if by_a[i] and not by_b[i])   # a better
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        # two-sided exact binomial at q=0.5
        k = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"n_paired": len(shared), "b_better": b01, "a_better": b10, "discordant": n, "p_two_sided": p}
