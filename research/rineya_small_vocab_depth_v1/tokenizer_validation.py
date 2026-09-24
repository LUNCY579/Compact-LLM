"""SMALLVD-001 Phases 5 and 14 -- measured consequences of each candidate tokenizer.

Every number here is computed on the calibration HELD-OUT split, which never contributed a count to
the utility that selected the tokens. All comparisons are old tokenizer vs new on IDENTICAL input
strings.

Per candidate:
  * vocabulary rows, embedding parameters, total parameters (via param_accounting)
  * tokens per example (mean, median) for old and new
  * per-example expansion ratio new/old: mean, P50, P90, P95, P99, max; overall token ratio
  * the same broken down by category (English, code, maths, structured data, ...)
  * exact round-trip rate (decode(encode(s)) == NFC(s)) and tokenizer failures
  * special-token behaviour on adversarial strings
  * pathology search: P99 worst examples; words that went from 1 token to >= 4; whitespace runs,
    operators and number strings that fragment; per-class fragmentation rates
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict

from layer_ablation import pctl

WORD_RE = re.compile(r" ?[A-Za-z]+(?:'[a-z]+)?")
PROBES = {
    "operators": ["==", "!=", "<=", ">=", "->", "=>", "::", "**", "//", "&&", "||", "+=", "===", "!==", "?.", "??", ":="],
    "whitespace": ["    ", "        ", "\t\t", "\n\n", "\n    ", "\n        ", "  ", " " * 16],
    "numbers": ["2026", "3.14159", "1,234,567.89", "-0.5", "1e-6", "0x1F", "10^9", "99.9%"],
    "urls": ["https://www.example.com/path/to/page.html?q=1&r=2#frag", "http://localhost:8080/api/v1/users",
             "github.com/LUNCY579/ChatBubble", "user@example.org"],
    "paths": ["C:\\Users\\clclc\\file.txt", "/usr/local/bin/python3", "~/projects/src/main.ts", "./a/../b.json"],
    "markdown": ["## Heading", "**bold**", "- [ ] task", "```python", "| a | b |", "> quote", "[link](https://x.y)"],
    "math": ["x^2 + y^2 = r^2", "∑_{i=1}^n i = n(n+1)/2", "a ≤ b ≥ c ≠ d", "∀ε>0 ∃δ>0", "√(x² + y²)", "f′(x) → ∞"],
    "proper_names": [" Shakespeare", " Wittgenstein", " Kubernetes", " PyTorch", " Tchaikovsky", " Nguyen", " Okonkwo"],
    "accented": [" José", " François", " Müller", " Zoë", " São Paulo", " naïve", " Dvořák", " Łódź"],
}
ADVERSARIAL_SPECIAL = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<|im_start|>user",
    "text<|im_end|>more", "<|im_start", "im_start|>", "< |im_start| >", "<|IM_START|>", "<<|im_end|>>",
]


def encode_ids(tok, texts):
    return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]


def roundtrip_ok(tok, text: str, ids: list[int]) -> bool:
    return tok.decode(ids, skip_special_tokens=False) == unicodedata.normalize("NFC", text)


def compare_on_rows(old, new, rows: list[dict]) -> dict:
    texts = [r["text"] for r in rows]
    oi, ni = encode_ids(old, texts), encode_ids(new, texts)
    per = []
    fails = []
    for r, a, b in zip(rows, oi, ni):
        ok = roundtrip_ok(new, r["text"], b)
        if not ok:
            fails.append(r["id"])
        per.append({"id": r["id"], "category": r["category"], "old": len(a), "new": len(b),
                    "ratio": len(b) / max(1, len(a)), "roundtrip": ok})

    def agg(xs):
        ratios = [x["ratio"] for x in xs]
        o, n = sum(x["old"] for x in xs), sum(x["new"] for x in xs)
        return {"rows": len(xs), "old_tokens": o, "new_tokens": n, "token_ratio": n / o,
                "old_mean": o / len(xs), "new_mean": n / len(xs),
                "old_median": pctl([x["old"] for x in xs], 0.5), "new_median": pctl([x["new"] for x in xs], 0.5),
                "ratio_mean": sum(ratios) / len(ratios), "ratio_p50": pctl(ratios, 0.5),
                "ratio_p90": pctl(ratios, 0.9), "ratio_p95": pctl(ratios, 0.95),
                "ratio_p99": pctl(ratios, 0.99), "ratio_max": max(ratios),
                "roundtrip_exact_rate": sum(x["roundtrip"] for x in xs) / len(xs)}
    cats = defaultdict(list)
    for x in per:
        cats[x["category"]].append(x)
    worst = sorted(per, key=lambda x: -x["ratio"])[:max(5, len(per) // 100)]
    return {"all": agg(per), "by_category": {c: agg(v) for c, v in sorted(cats.items())},
            "roundtrip_failures": fails, "worst_p99": worst}


def pretokenized_words(tok, text: str) -> list[str]:
    """The pre-tokenizer's chunks of `text` after the normaliser, as substrings. BPE merges never
    cross a chunk boundary, so the invariance theorem holds chunk by chunk -- a far finer (and far
    better covered) check than whole lines, whose single pruned token disqualifies the line."""
    norm = tok.normalizer.normalize_str(text) if tok.normalizer is not None else text
    words = [norm[a:b] for _, (a, b) in tok.pre_tokenizer.pre_tokenize_str(norm)]
    if "".join(words) != norm:
        raise ValueError("pre-tokenizer offsets do not tile the normalised text")
    return words


def word_invariance(old, new, id_map: dict, rows: list[dict], examples: int = 20) -> dict:
    """Amendment 1 (REJECT 5): every distinct pre-tokenized word whose TEACHER encoding uses only
    retained ids must encode, under the candidate, to exactly those ids mapped. The qualifying
    FRACTION is recorded (distinct and occurrence-weighted), so coverage is visible, not assumed."""
    freq = defaultdict(int)
    for r in rows:
        for w in pretokenized_words(old, r["text"]):
            if w:
                freq[w] += 1
    words = list(freq)
    oe, ne = encode_ids(old, words), encode_ids(new, words)
    qual = qual_occ = 0
    bad = []
    for w, a, b in zip(words, oe, ne):
        if a and all(i in id_map for i in a):
            qual += 1
            qual_occ += freq[w]
            if b != [id_map[i] for i in a]:
                bad.append({"word": w, "count": freq[w], "teacher_ids": a, "candidate_ids": b})
    total = sum(freq.values())
    bad.sort(key=lambda x: -x["count"])
    return {"distinct_words": len(words), "word_occurrences": total, "qualifying_distinct": qual,
            "qualifying_fraction_distinct": qual / len(words), "qualifying_fraction_occurrences": qual_occ / total,
            "violations": len(bad), "violation_examples": bad[:examples]}


def word_fragmentation(old, new, rows: list[dict], top: int = 60) -> dict:
    """Words that were ONE teacher token and became >= 4 new tokens, by frequency."""
    freq = defaultdict(int)
    for r in rows:
        for w in WORD_RE.findall(r["text"]):
            freq[w] += 1
    words = list(freq)
    oe, ne = encode_ids(old, words), encode_ids(new, words)
    exploded, split_any = [], 0
    for w, a, b in zip(words, oe, ne):
        if len(a) == 1 and len(b) > 1:
            split_any += freq[w]
        if len(a) == 1 and len(b) >= 4:
            exploded.append({"word": w, "count": freq[w], "old": len(a), "new": len(b)})
    exploded.sort(key=lambda x: -x["count"])
    total = sum(freq.values())
    return {"distinct_words": len(words), "word_occurrences": total,
            "single_token_words_now_split_occurrence_rate": split_any / total,
            "exploded_ge4_distinct": len(exploded), "exploded_ge4_top": exploded[:top]}


def probe_fragmentation(old, new) -> dict:
    out = {}
    for cls, items in PROBES.items():
        oe, ne = encode_ids(old, items), encode_ids(new, items)
        out[cls] = {"old_tokens": sum(map(len, oe)), "new_tokens": sum(map(len, ne)),
                    "items": [{"s": s, "old": len(a), "new": len(b)} for s, a, b in zip(items, oe, ne)]}
    return out


def special_behaviour(old, new, id_map: dict) -> dict:
    res, bad = [], 0
    for s in ADVERSARIAL_SPECIAL:
        a = old.encode(s, add_special_tokens=False).ids
        b = new.encode(s, add_special_tokens=False).ids
        mapped = [id_map.get(i) for i in a]
        same = None not in mapped and mapped == b
        # the SPECIAL-TOKEN SEGMENTATION must match exactly even if regular pieces re-segment
        sp_old = [i for i in a if i >= 151643]
        sp_new = [i for i in b if i >= len(id_map) - 26]
        ok = [id_map[i] for i in sp_old] == sp_new and new.decode(b, skip_special_tokens=False) == s
        bad += not ok
        res.append({"s": s, "old": a, "new": b, "identical_after_map": same, "special_ok": ok})
    return {"cases": res, "violations": bad}
