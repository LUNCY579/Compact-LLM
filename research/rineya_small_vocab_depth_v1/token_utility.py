"""SMALLVD-001 Phase 3 -- per-token utility profile on the calibration corpus.

Utility is NOT raw frequency. It is the MIXTURE-WEIGHTED count: each category's counts are
normalised by that category's token total and re-weighted by the declared workload weight
(CATEGORY_WEIGHTS), then expressed per million tokens of the target mixture. A token that is rare
overall but frequent in code, maths or structured data therefore keeps the utility that its own
category gives it, and one large source cannot drown the rest.

Per-token statistics written for every vocabulary entry:
  count, document frequency, per-category counts and shares, byte class, protected category,
  whether it is a chat-template token, merge depth, number of retained children at each
  candidate size, its standalone re-encoding length under each candidate tokenizer, and the
  modelled marginal cost of removal (effective count x (re-encoding length - 1)).

Only the FIT split is ever counted. The held-out split is reserved for tokenizer_validation.py,
so the candidate tokenizers are never scored on the text that shaped them.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

# Declared BEFORE any count is seen (registered in SMALLVD001_PREREG.md section 3). The mixture is
# English-centric and weighted toward what the programme's instruments exercise: chat, retrieval,
# grounded answering and abstention. Code/maths/structured data are included because the brief
# protects them, not because the teacher is a coding model.
CATEGORY_WEIGHTS = {
    "english_prose": 0.14, "instruction": 0.12, "conversation": 0.10, "retrieval_context": 0.12,
    "grounded_answering": 0.08, "concise_reasoning": 0.08, "python": 0.07, "javascript": 0.05,
    "json": 0.05, "markdown": 0.05, "math": 0.04, "numbers_dates_urls_paths": 0.04,
    "technical_names_accents": 0.04, "punctuation_whitespace": 0.02,
}
assert abs(sum(CATEGORY_WEIGHTS.values()) - 1.0) < 1e-12


def load_corpus(path: Path, split: str | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if split is None or r["split"] == split:
                rows.append(r)
    return rows


def count_tokens(tokenizer, rows: list[dict], batch: int = 256) -> dict:
    """Teacher-tokenizer counts per category on the given rows. `tokenizer` is a
    tokenizers.Tokenizer; add_special_tokens=False because the rendered chat rows already
    contain their special tokens as text and the tokenizer matches them as added tokens."""
    by_cat = defaultdict(Counter)
    df_by_cat = defaultdict(Counter)
    ntok, nrows = Counter(), Counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        encs = tokenizer.encode_batch([r["text"] for r in chunk], add_special_tokens=False)
        for r, e in zip(chunk, encs):
            c = r["category"]
            ids = e.ids
            by_cat[c].update(ids)
            df_by_cat[c].update(set(ids))
            ntok[c] += len(ids)
            nrows[c] += 1
    return {"counts": by_cat, "doc_freq": df_by_cat, "tokens": ntok, "rows": nrows}


def mixture_utility(counts: dict, weights: dict = CATEGORY_WEIGHTS, per: float = 1e6) -> dict:
    """token_id -> expected occurrences per `per` tokens of the declared mixture."""
    missing = set(weights) - set(counts["tokens"])
    if missing:
        raise ValueError(f"categories with weight but no calibration text: {sorted(missing)}")
    u = defaultdict(float)
    for cat, w in weights.items():
        n = counts["tokens"][cat]
        for tid, c in counts["counts"][cat].items():
            u[tid] += w * c / n * per
    return dict(u)


def merge_depth(tok: str, parents: dict, memo: dict) -> int:
    if tok not in parents:
        return 0
    if tok in memo:
        return memo[tok]
    a, b = parents[tok]
    d = 1 + max(merge_depth(a, parents, memo), merge_depth(b, parents, memo))
    memo[tok] = d
    return d
