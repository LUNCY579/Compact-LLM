"""The instrument gate: answer-agnostic floors, a ceiling, and a three-way verdict.

Every check here exists because its absence cost this programme something specific. The war
stories are in the docstrings rather than in a separate document, because a check whose reason
lives elsewhere gets deleted by the next person who finds it inconvenient.

    MRS spent a scout whose checkpoints sat at or below the constant-answer baseline. Every
    arm contrast on it was uninterpretable and the compute was gone.
        -> FLOOR check.

    DMRF-SCOUT-001's long-context stratum came back 32/32 on both cores. It cleared its floor
    and could not show a regression, which is half of what an instrument is for.
        -> CEILING check.

    RCI-001 published four floors -- always_abstain, constant_majority, per_stratum_majority,
    copy_entity_from_question -- and every one of them was DISTRIBUTIONAL. Not one read the
    context. A recency rule that does (`last_candidate_in_context`) scored 0.3711 against the
    published 0.2773, and 100% on two of five task types, because the generator emitted every
    correction after the thing it corrected. The defect was found by an outside audit AFTER the
    instrument was spent.
        -> EXTRACTION predictors are a required class, not an optional extra.

    CAL at n=6 per stratum cannot clear a 0.0526 floor below 3/6, so a stratum with a real 17%
    signal was reported FAILS when the honest answer was "this split cannot certify it".
        -> three-way verdict, and `min_k_to_clear` printed alongside.

The two predictor classes answer different questions and you need both:

    DISTRIBUTIONAL  what does the ANSWER DISTRIBUTION give away? (always abstain, guess the
                    modal answer, guess the modal answer for this stratum)
    EXTRACTION      what does the ITEM give away to something that reads it but does not
                    reason? (copy the last/first/most-frequent candidate, copy from the
                    question)

A bank can look hard distributionally and be trivial to an extractor. RCI-001 did.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

#: A score at or above this cannot show a regression, so the instrument cannot confirm one.
DEFAULT_CEILING = 0.90


# --------------------------------------------------------------------------------------- #
# Exact binomial machinery, matching the acquisition instrument's published rule
# --------------------------------------------------------------------------------------- #

def binom_at_least(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact two-sided 95% lower bound."""
    if k <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_at_least(k, n, mid) > alpha / 2:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def min_k_to_clear(n: int, floor: float, alpha: float = 0.05) -> int:
    """Smallest correct count at this n whose lower bound clears `floor`.

    Printed with every verdict so "we could not certify it at this n" is never mistaken for
    "it failed". At n=6 against a 0.0526 floor this is 3: a stratum scoring 1/6 or 2/6 cannot
    be certified however real its signal.
    """
    for k in range(n + 1):
        if cp_lower(k, n, alpha) > floor:
            return k
    return n + 1


# --------------------------------------------------------------------------------------- #
# Predictors. `rows` is a sequence of mappings with at least `reference`; optionally
# `stratum`, `context`, `question`, and `copy_source` (e.g. the input state to echo).
# --------------------------------------------------------------------------------------- #

def _vocab(rows: Sequence[Mapping]) -> dict:
    """Label vocabulary per stratum, built from the bank's own references."""
    v: dict = {}
    for r in rows:
        v.setdefault(r.get("stratum", "_all"), set()).add(str(r["reference"]))
    return v


def distributional(rows: Sequence[Mapping], abstain: str | None = None) -> dict[str, int]:
    """What the answer distribution alone gives away. No item is read."""
    refs = [str(r["reference"]) for r in rows]
    out = {"constant_majority": max(Counter(refs).values()) if refs else 0}
    if abstain is not None:
        out["always_abstain"] = sum(1 for x in refs if x == abstain)
    per: dict = {}
    for r in rows:
        per.setdefault(r.get("stratum", "_all"), []).append(str(r["reference"]))
    out["per_stratum_majority"] = sum(Counter(v).most_common(1)[0][1] for v in per.values())
    if any("copy_source" in r for r in rows):
        out["copy_source_unchanged"] = sum(
            1 for r in rows if "copy_source" in r and str(r["copy_source"]) == str(r["reference"]))
    return out


def extraction(rows: Sequence[Mapping]) -> dict[str, int]:
    """What the ITEM gives away to something that reads but does not reason.

    THE CLASS RCI-001 OMITTED. Each rule picks a member of the stratum's label vocabulary by
    a positional or frequency property of the context -- no labels at inference, one uniform
    rule, deployable by anyone.
    """
    vocab = _vocab(rows)
    scores = {"last_candidate_in_context": 0, "first_candidate_in_context": 0,
              "most_frequent_candidate_in_context": 0, "copy_entity_from_question": 0}
    for r in rows:
        ctx = str(r.get("context") or _ctx_of(r))
        q = str(r.get("question") or "")
        ref = str(r["reference"])
        cands = vocab.get(r.get("stratum", "_all"), set())
        if ctx and cands:
            last = max(cands, key=lambda c: ctx.rfind(c))
            if ctx.rfind(last) >= 0 and last == ref:
                scores["last_candidate_in_context"] += 1
            present = [c for c in cands if c in ctx]
            if present:
                first = min(present, key=lambda c: ctx.find(c))
                if first == ref:
                    scores["first_candidate_in_context"] += 1
                freq = max(present, key=lambda c: ctx.count(c))
                if freq == ref:
                    scores["most_frequent_candidate_in_context"] += 1
        if q:
            ents = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", q)
            if any(e.strip() == ref for e in ents):
                scores["copy_entity_from_question"] += 1
    return scores


def _ctx_of(r: Mapping) -> str:
    """Best-effort context text from a chat-style prompt."""
    p = r.get("prompt")
    if isinstance(p, list):
        return "\n".join(str(m.get("content", "")) for m in p if m.get("role") == "user")
    return str(p or "")


def floors(rows: Sequence[Mapping], abstain: str | None = None,
           extra: Mapping[str, Callable[[Sequence[Mapping]], int]] | None = None) -> dict:
    """Binding floor over BOTH classes. Refuses to run with only one class present."""
    if not rows:
        raise ValueError("no rows")
    d = distributional(rows, abstain=abstain)
    e = extraction(rows)
    if extra:
        e.update({k: fn(rows) for k, fn in extra.items()})
    allp = {**d, **e}
    n = len(rows)
    binding = max(allp.values())
    pred = max(allp, key=allp.get)
    return {
        "n": n,
        "distributional": d,
        "extraction": e,
        "binding_floor_count": binding,
        "binding_floor_rate": round(binding / n, 4),
        "binding_predictor": pred,
        "binding_class": "extraction" if pred in e else "distributional",
        "distinct_references": len(set(str(r["reference"]) for r in rows)),
        "note": ("Both predictor classes are required. A bank can look hard distributionally "
                 "and be trivial to an extractor -- RCI-001 published four distributional "
                 "floors at 0.2773 and a recency rule scored 0.3711."),
    }


# --------------------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------------------- #

def gate(correct: int, n: int, floor_rate: float, ceiling: float = DEFAULT_CEILING,
         alpha: float = 0.05) -> dict:
    """CLEARS / INCONCLUSIVE_UNDERPOWERED / BELOW_FLOOR / ABOVE_CEILING.

    Underpowered is not failure. A point estimate above the floor whose interval cannot exclude
    it means the split cannot certify the stratum -- report that, do not tune the task until a
    small test says yes.
    """
    lb = cp_lower(correct, n, alpha)
    rate = correct / n if n else 0.0
    need = min_k_to_clear(n, floor_rate, alpha)
    if rate >= ceiling:
        verdict = "ABOVE_CEILING"
    elif lb > floor_rate:
        verdict = "CLEARS"
    elif rate > floor_rate:
        verdict = "INCONCLUSIVE_UNDERPOWERED"
    else:
        verdict = "BELOW_FLOOR"
    return {"n": n, "correct": correct, "rate": round(rate, 4),
            "ci_lower_95": round(lb, 4), "floor_rate": floor_rate, "ceiling": ceiling,
            "min_correct_to_clear_at_this_n": need, "verdict": verdict,
            "why": {
                "ABOVE_CEILING": "cannot show a regression, so cannot confirm one",
                "CLEARS": "exact lower bound exceeds the strongest answer-agnostic predictor",
                "INCONCLUSIVE_UNDERPOWERED": (f"point estimate exceeds the floor but n={n} "
                                              f"needs {need} correct to certify it"),
                "BELOW_FLOOR": "an answer-agnostic predictor is competitive; contrasts here "
                               "are uninterpretable",
            }[verdict]}


def immunity(rows: Sequence[Mapping], subset: Callable[[Mapping], bool],
             abstain: str | None = None) -> dict:
    """Which shortcut CLASSES can reach a candidate primary endpoint?

    Immunity is per-class, and conflating the classes is a mistake this function was written
    with and then corrected by its own fixture. Two distinct things:

      EXTRACTION immunity is structural and is what saved RCI-001. An extractor answers
      something from the context, so it can never emit the sentinel and scores 0 on an
      unanswerable stratum -- which is why a recency defect worth 0.3711 on the aggregate
      could not touch the registered primary endpoint.

      DISTRIBUTIONAL immunity is NOT obtained by choosing an abstention stratum. An
      all-`UNKNOWN` subset is trivially won by `always_abstain`. What neutralises that is not
      the endpoint but a CONTROL: false abstention on answerable rows. RCI-001 read 0 for both
      systems, which is what rules out "it just says UNKNOWN a lot" as the explanation.

    So this reports both, and refuses to collapse them into one boolean. Predictors are
    calibrated on the FULL bank and scored on the subset -- calibrating on the subset would
    ask whether the subset is self-consistent, which is a different and useless question.
    """
    sub = [r for r in rows if subset(r)]
    if not sub:
        return {"n": 0, "extraction_immune": None}
    ids = {id(r) for r in sub}
    full_vocab = _vocab(rows)

    # Extraction predictors, vocabulary from the FULL bank, scored on the subset only.
    ex = {"last_candidate_in_context": 0, "first_candidate_in_context": 0,
          "most_frequent_candidate_in_context": 0}
    for r in rows:
        if id(r) not in ids:
            continue
        ctx = str(r.get("context") or _ctx_of(r))
        ref = str(r["reference"])
        cands = full_vocab.get(r.get("stratum", "_all"), set())
        if not (ctx and cands):
            continue
        if max(cands, key=lambda c: ctx.rfind(c)) == ref and ctx.rfind(ref) >= 0:
            ex["last_candidate_in_context"] += 1
        present = [c for c in cands if c in ctx]
        if present:
            if min(present, key=lambda c: ctx.find(c)) == ref:
                ex["first_candidate_in_context"] += 1
            if max(present, key=lambda c: ctx.count(c)) == ref:
                ex["most_frequent_candidate_in_context"] += 1

    dist = {}
    if abstain is not None:
        dist["always_abstain"] = sum(1 for r in sub if str(r["reference"]) == abstain)
    refs = [str(r["reference"]) for r in sub]
    dist["constant_majority"] = max(Counter(refs).values()) if refs else 0

    ex_max = max(ex.values()) if ex else 0
    return {
        "n": len(sub),
        "extraction": ex, "extraction_max": ex_max,
        "extraction_immune": ex_max == 0,
        "distributional": dist, "distributional_max": max(dist.values()) if dist else 0,
        "distributional_immune": (max(dist.values()) if dist else 0) == 0,
        "required_control_if_not_distributionally_immune": (
            "false abstention on answerable rows must be reported and near zero; without it, "
            "'the system just emits the sentinel often' is a live alternative explanation"),
        "note": ("Choose a primary endpoint that is EXTRACTION-immune, and pair it with the "
                 "false-abstention control. Extraction immunity is structural; distributional "
                 "immunity is not obtainable by endpoint choice alone."),
    }
