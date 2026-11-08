"""A virtual student = the teacher with (a) some blocks bypassed and/or (b) its vocabulary restricted
to a candidate tokenizer's retained ids. Mathematically identical to the surgery-built student:

  * blocks: `keep_layers` swaps the ModuleList and renumbers layer_idx, exactly as surgery does;
  * vocabulary: input text is tokenised with the CANDIDATE tokenizer and each new id is mapped back
    to its old id, so the SAME embedding row is read; the output softmax runs over the retained
    old ids only, which is what a head with only those rows computes.

`tests/test_surgery.py` asserts the built checkpoints match the teacher-with-bypass bitwise on
hidden states, which is what licenses measuring the frontier here without writing checkpoints.

Tokenizer-independent comparison. KL is undefined across two tokenisations of the same text, so
cross-vocabulary quantities are BITS PER BYTE of the reference answer (answer tokens plus the
closing <|im_end|>, divided by the answer's UTF-8 bytes) and generation behaviour.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from layer_ablation import batches_by_length, keep_layers, pad_batch


class Vocab:
    """Candidate vocabulary as seen from the teacher's id space."""

    def __init__(self, tok_dir: Path | None, teacher_tokenizer_path: Path):
        import tokenizers
        self.teacher_tok = tokenizers.Tokenizer.from_file(str(teacher_tokenizer_path))
        if tok_dir is None:
            self.name, self.tok, self.new_to_old, self.retained = "V0", self.teacher_tok, None, None
            return
        self.name = Path(tok_dir).name
        self.tok = tokenizers.Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))
        raw = json.loads((Path(tok_dir) / "token_id_map.json").read_text(encoding="utf-8"))["map"]
        m = {int(k): int(v) for k, v in raw.items()}
        self.new_to_old = torch.tensor([o for o, _ in sorted(m.items(), key=lambda kv: kv[1])])
        self.retained = self.new_to_old.clone()          # old ids, in new-id order

    def encode_old_ids(self, text: str) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        return ids if self.new_to_old is None else self.new_to_old[ids].tolist()


class RestrictLogits:
    """LogitsProcessor: -inf on every id outside the candidate vocabulary."""

    def __init__(self, retained: torch.Tensor | None, vocab_rows: int, device):
        self.mask = None
        if retained is not None:
            self.mask = torch.full((vocab_rows,), float("-inf"), device=device)
            self.mask[retained.to(device)] = 0.0

    def __call__(self, input_ids, scores):
        return scores if self.mask is None else scores + self.mask[: scores.shape[-1]]


@torch.no_grad()
def answer_bpb(model, vocab: Vocab, rows: list[dict], kept: list[int], im_end_old: int, device,
               max_tokens: int = 8192, chunk: int = 256) -> dict:
    """Bits per byte of each reference answer under the virtual student (teacher forcing)."""
    items = []
    for r in rows:
        p = vocab.encode_old_ids(r["prompt_text"])
        a = vocab.encode_old_ids(r["reference"])
        ids = p + a + [im_end_old]
        items.append({"ids": ids, "start": len(p), "n": len(a) + 1,
                      "bytes": len(r["reference"].encode("utf-8")), "row": r})
    W = model.lm_head.weight
    ret = vocab.retained.to(device) if vocab.retained is not None else None
    pos_in_ret = None
    if ret is not None:
        pos_in_ret = torch.full((W.shape[0],), -1, dtype=torch.long, device=device)
        pos_in_ret[ret] = torch.arange(ret.numel(), device=device)
    out = [None] * len(items)
    with keep_layers(model, kept):
        for b in batches_by_length(items, max_tokens):
            ids, mask = pad_batch([items[i]["ids"] for i in b], 0, device)
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            for j, i in enumerate(b):
                it = items[i]
                pos = torch.arange(it["start"] - 1, it["start"] - 1 + it["n"], device=device)
                tgt = ids[j, it["start"]: it["start"] + it["n"]]
                nll = 0.0
                for s in range(0, pos.numel(), chunk):
                    logits = h[j].index_select(0, pos[s:s + chunk]).float() @ W.T.float()
                    g = tgt[s:s + chunk]
                    if ret is not None:
                        logits = logits[:, ret]
                        g = pos_in_ret[g]
                        assert (g >= 0).all(), "a target token is outside the candidate vocabulary"
                    nll += -torch.log_softmax(logits, -1).gather(1, g[:, None]).sum().item()
                out[i] = {"id": it["row"]["id"], "category": it["row"]["category"],
                          "nats": nll, "tokens": it["n"], "bytes": it["bytes"]}
    return bpb_summary(out)


def bpb_summary(out: list[dict]) -> dict:
    """Aggregate per-row answer nats into bits per byte, overall and by category. Module-level so a
    subset of rows (the REPORT split) is summarised by exactly the same code as the whole."""
    def agg(xs):
        nats, byts, toks = sum(x["nats"] for x in xs), sum(x["bytes"] for x in xs), sum(x["tokens"] for x in xs)
        return {"rows": len(xs), "bpb": nats / math.log(2) / byts, "tokens": toks, "bytes": byts,
                "nats_per_token": nats / toks}
    cats = {}
    for x in out:
        cats.setdefault(x["category"], []).append(x)
    return {"all": agg(out), "by_category": {c: agg(v) for c, v in sorted(cats.items())}, "per_item": out}


@torch.no_grad()
def generate_rows(model, vocab: Vocab, rows: list[dict], kept: list[int], eos_old: list[int],
                  pad_old: int, device, max_new_tokens: int, batch: int = 16) -> list[dict]:
    """Greedy generation under the virtual student; returns decoded text and token counts."""
    from transformers import LogitsProcessorList
    proc = LogitsProcessorList([RestrictLogits(vocab.retained, model.lm_head.weight.shape[0], device)])
    prompts = [vocab.encode_old_ids(r["prompt_text"]) for r in rows]
    outs = [None] * len(rows)
    # budget on prompt + generated length so prefill AND the last decode step stay inside it
    sized = [{"ids": [0] * (len(p) + max_new_tokens)} for p in prompts]
    with keep_layers(model, kept):
        for idx in batches_by_length(sized, max_tokens=1 << 30, max_rows=batch):
            T = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), T), pad_old, dtype=torch.long)
            mask = torch.zeros((len(idx), T), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, T - len(prompts[i]):] = torch.tensor(prompts[i])
                mask[j, T - len(prompts[i]):] = 1
            g = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), do_sample=False,
                               num_beams=1, max_new_tokens=max_new_tokens, eos_token_id=eos_old,
                               pad_token_id=pad_old, logits_processor=proc,
                               temperature=None, top_p=None, top_k=None)
            for j, i in enumerate(idx):
                new = g[j, T:].tolist()
                cut = next((k for k, t in enumerate(new) if t in eos_old), None)
                body = new[:cut] if cut is not None else new
                outs[i] = {"id": rows[i]["id"], "text": vocab.teacher_tok.decode(body, skip_special_tokens=True),
                           "new_tokens": len(body), "terminated": cut is not None}
    return outs
