"""ABST-001 -- per-row teacher-forced measurement, one row per forward pass (B = 1, no padding).

Batch composition changes fp32 results in the last bits, and the pool-conditional H0 needs each row's
value to be a function of that row alone (cold-design statistics critic), so every row is its own forward
pass. One forward pass per (depth config, INPUT vocabulary) is read out under every OUTPUT support
(the full teacher head V0, and the V4 / V2 retained sets): the matched cells (input == support) and the
mixed readouts (input != support) come from the same pass -- HLCC stored only one vocabulary per set and
could not attribute its reversal.

Per row and support: total nats of reference + <|im_end|>; per-token nats; at the FIRST answer position
the log-probs of UNKNOWN, 0, 1, 2, 3 and <|im_end|> under the support, the log retained mass
logsumexp(support) - logsumexp(all), and (matched supports only) the top-5 ids.
"""
from __future__ import annotations

import torch

from layer_ablation import keep_layers

ANSWER_TOKENS = ("UNKNOWN", "0", "1", "2", "3")


def answer_token_ids(teacher_tok, im_end: int) -> dict[str, int]:
    """Old (teacher) ids of the single-token answers; refuses if any is not exactly one token."""
    out = {}
    for s in ANSWER_TOKENS:
        ids = teacher_tok.encode(s, add_special_tokens=False).ids
        if len(ids) != 1:
            raise SystemExit(f"answer {s!r} is {len(ids)} teacher tokens, not 1")
        out[s] = ids[0]
    out["im_end"] = im_end
    return out


def support_index(retained: torch.Tensor | None, vocab_rows: int, device):
    """(retained old ids or None, old id -> position in the support or -1)."""
    if retained is None:
        return None, None
    ret = retained.to(device)
    pos = torch.full((vocab_rows,), -1, dtype=torch.long, device=device)
    pos[ret] = torch.arange(ret.numel(), device=device)
    return ret, pos


@torch.no_grad()
def measure(model, rows: list[dict], kept: list[int], vocab_in, supports: dict, im_end: int,
            answer_ids: dict[str, int], device, top5_for: str | None = None, sink: dict | None = None) -> dict[str, dict]:
    """rows: dicts with id, prompt_text, reference. supports: name -> (retained, pos) from support_index.
    Returns {row_id: {support_name: record}}; if `sink` is given, each row is also written into it AS IT IS
    MEASURED (so a failure mid-cell loses no finished row). The caller applies folds (lc.folded) around this call."""
    W = model.lm_head.weight
    out = sink if sink is not None else {}
    with keep_layers(model, kept):
        for r in rows:
            p = vocab_in.encode_old_ids(r["prompt_text"])
            a = vocab_in.encode_old_ids(r["reference"])
            ids = torch.tensor([p + a + [im_end]], device=device)
            start, n = len(p), len(a) + 1
            h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            logits = h[start - 1: start - 1 + n].float() @ W.T.float()          # [n, V]
            tgt = ids[0, start: start + n]
            full_lse0 = torch.logsumexp(logits[0], -1)
            rec = {}
            for name, (ret, pos) in supports.items():
                if ret is None:
                    lp = torch.log_softmax(logits, -1)
                    g = tgt
                    id_at = lambda i: i
                else:
                    lp = torch.log_softmax(logits[:, ret], -1)
                    g = pos[tgt]
                    if (g < 0).any():
                        raise SystemExit(f"row {r['id']}: a reference token is outside support {name}")
                    id_at = lambda i, pos=pos: int(pos[i])
                tok_nats = (-lp.gather(1, g[:, None]).squeeze(1)).tolist()
                first = {}
                for k, i in answer_ids.items():
                    j = id_at(i)
                    first[k] = float(lp[0, j]) if j >= 0 else None
                logmass = 0.0 if ret is None else float(torch.logsumexp(logits[0, ret], -1) - full_lse0)
                item = {"nats": float(sum(tok_nats)), "per_token": tok_nats, "first": first, "first_logmass": logmass}
                if top5_for == name:
                    t5 = lp[0].topk(5).indices
                    item["top5_old_ids"] = (t5 if ret is None else ret[t5]).tolist()
                rec[name] = item
            out[r["id"]] = rec
    return out
