"""Systematic answer-agnostic rule sweep across a manipulated variable.

**Why this exists.** MPOS-002 was built with a hand-picked list of 13 answer-agnostic rules. The
two cells that list declared "clean" — band swing exactly 0.000 on `argmin`/`argmax` of mean
mention position — turned out to carry **sentence-index rules swinging up to 0.771**, found only
because the sweep was re-run systematically *during generation*. The registered directional
asymmetry had to be withdrawn four minutes before the first arm finished.

> **A hand-picked rule list is the pooled-floor failure one layer up.** `method/floors.py` learned
> to run per stratum; this module exists because a manipulated variable needs the same treatment,
> over a rule space nobody curated by hand.

## What it sweeps

For every candidate in a row's declared pool, a dozen scalar features (first / last / mean /
median occurrence position, mention count, spread, name length, token count, first / last / mean
sentence index, number of sentences mentioned in), each turned into a rule by `argmin` and
`argmax`; plus a per-sentence-index rule for every sentence position. That is ~40 rules for a
16-sentence bank, generated rather than listed.

## Two things it does that the hand list did not

  * **Both directions.** A rule favouring the level you predict will *manufacture* your result; a
    rule favouring the other level only adds conservatism. The two are not interchangeable and
    the sweep reports the direction of every hit, because MPOS-002 registered an asymmetry that
    was false precisely because only one direction had been looked for.
  * **Explicit tie-breaking.** `max()` over a set resolves ties by arbitrary iteration order.
    Two sessions measuring "the same" `most_frequent` rule got differentials of +0.033 and
    -0.054 from that alone. Every rule here breaks ties by earliest occurrence then string
    order, and the tied-row count is reported rather than hidden.

**This module reports; it does not by itself refuse.** Whether a swing is fatal depends on the
design: for a comparison protected by pairing it may cancel, and for a between-row comparison it
usually does not. `worst_swing()` gives the number a caller needs to make that judgement.
"""
from __future__ import annotations

import re
import statistics as _st
from collections.abc import Sequence
from math import comb

SCALARS = ("first", "last", "mean", "median", "count", "spread",
           "namelen", "ntok", "sent_first", "sent_last", "sent_mean", "nsent")


def _fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, c1 = a + b + c + d, a + b, a + c
    if not (r1 and c1 and n):
        return 1.0

    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    p0 = pr(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12)


def _sentences(t: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def _occ(ctx: str, c: str) -> list[int]:
    out, i = [], ctx.find(c)
    while i >= 0:
        out.append(i)
        i = ctx.find(c, i + 1)
    return out


def _features(ctx: str, pool: Sequence[str]) -> dict:
    L = max(1, len(ctx))
    ss = _sentences(ctx)
    out: dict = {}
    for c in pool:
        o = _occ(ctx, c)
        if not o:
            continue
        si = [k for k, s in enumerate(ss) if c in s]
        out[c] = {
            "first": o[0] / L, "last": o[-1] / L,
            "mean": sum(o) / len(o) / L, "median": _st.median(o) / L,
            "count": len(o), "spread": (o[-1] - o[0]) / L,
            "namelen": len(c), "ntok": len(c.split()),
            "sent_first": si[0] if si else 10 ** 6,
            "sent_last": si[-1] if si else -1,
            "sent_mean": (sum(si) / len(si)) if si else 10 ** 6,
            "nsent": len(si),
            "_first_raw": o[0],
        }
    return out


def _pick(F: dict, feat: str, reverse: bool):
    """argmin/argmax with EXPLICIT tie-breaking. Returns (candidate, was_tied)."""
    if not F:
        return None, False
    vals = {c: F[c][feat] for c in F}
    best = max(vals.values()) if reverse else min(vals.values())
    tied = [c for c in vals if vals[c] == best]
    return sorted(tied, key=lambda c: (F[c]["_first_raw"], c))[0], len(tied) > 1


def sweep(rows: Sequence[dict], pool: Sequence[str], split_field: str,
          levels: tuple[str, str], ref_field: str = "reference",
          ctx_field: str = "context", max_sentences: int = 24) -> dict:
    """Sweep every generated rule and report its hit rate at each level of `split_field`."""
    by_level = {lv: [r for r in rows if r.get(split_field) == lv] for lv in levels}
    ns = {lv: len(v) for lv, v in by_level.items()}
    if min(ns.values()) == 0:
        return {"applicable": False, "why": f"a level of {split_field} is empty: {ns}"}

    hits: dict[str, dict] = {}
    ties: dict[str, int] = {}
    for lv in levels:
        for r in by_level[lv]:
            ctx, ref = str(r.get(ctx_field, "")), str(r.get(ref_field, ""))
            F = _features(ctx, pool)
            ss = _sentences(ctx)
            for feat in SCALARS:
                for lab, rev in (("argmin", False), ("argmax", True)):
                    name = f"{lab}_{feat}"
                    pick, tied = _pick(F, feat, rev)
                    hits.setdefault(name, {lv2: 0 for lv2 in levels})
                    hits[name][lv] += int(pick == ref)
                    ties[name] = ties.get(name, 0) + int(tied)
            for si in range(min(len(ss), max_sentences)):
                name = f"in_sentence_{si}"
                here = sorted(c for c in F if c in ss[si])
                hits.setdefault(name, {lv2: 0 for lv2 in levels})
                hits[name][lv] += int(len(here) == 1 and here[0] == ref)
                ties[name] = ties.get(name, 0) + int(len(here) > 1)

    a, b = levels
    out = []
    for name, h in hits.items():
        ra, rb = h[a] / ns[a], h[b] / ns[b]
        out.append({
            "rule": name, f"rate_{a}": round(ra, 4), f"rate_{b}": round(rb, 4),
            "swing": round(ra - rb, 4), "abs_swing": round(abs(ra - rb), 4),
            "favours": a if ra > rb else (b if rb > ra else "neither"),
            "fisher_p": _fisher(h[a], ns[a] - h[a], h[b], ns[b] - h[b]),
            "tied_rows": ties.get(name, 0),
        })
    out.sort(key=lambda d: -d["abs_swing"])
    return {
        "applicable": True, "n_rules": len(out), "n_per_level": ns,
        "levels": list(levels), "rules": out,
        "note": ("a rule favouring the level you PREDICT will manufacture your result; one "
                 "favouring the other level only adds conservatism. direction is reported for "
                 "every rule because MPOS-002 registered an asymmetry that was false precisely "
                 "because only one direction had been swept"),
    }


def can_manufacture(result: dict, min_swing: float = 0.02) -> list[dict]:
    """The ONLY rules worth measuring a differential for: those that swing across the levels.

    **Run this BEFORE measuring any cue differential.** Manufacturable interaction is
    `differential x swing`, so a rule whose swing is zero cannot manufacture anything **at any
    differential**, and measuring its differential is wasted work that invites a false conclusion.

    MPOS-002 learned this the expensive way. A pattern in cue-exploitation differentials was
    published as an observation and looked like clean support -- four arms splitting by mask
    status. It was computed entirely on **candidate-identity** cues (earliest person, longest
    name, alphabetical), whose measured band swing on that bank was **exactly 0.0000**: the
    manipulation moved the answer's POSITION, not which candidate was which. The differentials
    were real and completely irrelevant.

    The filter that failed was *"is this a deterministic answer-agnostic rule?"* It never asked
    *"can this rule distinguish the conditions at all?"* -- which is this function.
    """
    if not result.get("applicable"):
        return []
    return [r for r in result["rules"] if r["abs_swing"] >= min_swing]


def worst_swing(result: dict, favouring: str | None = None) -> float:
    """Largest absolute band swing, optionally restricted to rules favouring one level."""
    if not result.get("applicable"):
        return 0.0
    rules = result["rules"]
    if favouring is not None:
        rules = [r for r in rules if r["favours"] == favouring]
    return max((r["abs_swing"] for r in rules), default=0.0)
