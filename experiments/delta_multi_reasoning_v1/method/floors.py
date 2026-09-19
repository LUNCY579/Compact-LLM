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

    RCI-002 cleared BOTH published classes on every stratum -- 0.20 to 0.29 -- and a red-team
    pass then found a decoy sentence that listed three of four roles, always omitting the
    exempt one. Set difference over the two listings scored 1.000, and 120/120 on the sealed
    bank. It was found before the seal was spent, which is the only reason this file has a
    third class instead of a third erratum.
        -> STRUCTURAL predictors, and an adversarial pass BEFORE sealing.

The three predictor classes answer different questions and you need all of them:

    DISTRIBUTIONAL  what does the ANSWER DISTRIBUTION give away? (always abstain, guess the
                    modal answer, guess the modal answer for this stratum)
    EXTRACTION      what does the ITEM give away to something that reads it but does not
                    reason? (copy the last/first/most-frequent candidate, copy from the
                    question)
    STRUCTURAL      what do two spans of the item give away RELATIVE TO EACH OTHER? (set
                    complement between listings, arity of a list, candidate-set collapse,
                    length tells for an omission)

A bank can look hard distributionally and be trivial to an extractor. RCI-001 was. A bank can
clear both and still fall to one line of set arithmetic. RCI-002's first draft did.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

#: A score at or above this cannot show a regression, so the instrument cannot confirm one.
DEFAULT_CEILING = 0.90


# --------------------------------------------------------------------------------------- #
# Build-time structural requirements
# --------------------------------------------------------------------------------------- #

def require_pack_structure(rows: Sequence[Mapping], pack_key: str = "scenario_id",
                           variant_key: str = "variant", min_variants: int = 3) -> dict:
    """MANDATORY, and not a matter of taste.

    Score several variants of one scenario together and require all of them. RCI-001 had this
    only because it was copied from the Demo suite before anyone had a reason to want it, and
    it turned out load-bearing: the recency rule worth 0.3711 on the aggregate scored **0/128**
    per pack, because any pack containing an unanswerable row is unwinnable by an extractor.
    The instrument survived a defect nobody anticipated on the strength of a habit.

    A habit is not a safeguard. Make it a gate.
    """
    packs: dict = {}
    for r in rows:
        packs.setdefault(r.get(pack_key), set()).add(r.get(variant_key))
    sizes = [len(v) for v in packs.values()]
    return {"packs": len(packs), "min_variants_per_pack": min(sizes) if sizes else 0,
            "required": min_variants, "satisfied": bool(packs) and min(sizes) >= min_variants,
            "why": "a pack containing an unanswerable row cannot be won by any extractor, so "
                   "per-pack all-correct is shortcut-immune by construction",
            "action_if_unsatisfied": "do not ship the instrument; add variants per scenario"}


_ARTICLE = re.compile(r"^(the|a|an)\s+")


def convention_sensitivity(rows: Sequence[Mapping], predictions: Mapping[str, str]) -> dict:
    """How many rows turn on a scoring convention rather than on being right?

    On RCI-001, 52 of 384 answerable rows hinged on a leading definite article and one on a
    trailing period -- the model answers `acquisitions secretary` where the reference reads
    `the acquisitions secretary`. Two sessions scoring the same bank under their own
    conventions differed by ~50 rows and spent real effort reconciling numbers that were never
    in disagreement about the model.

    Report it at build time, and prefer a primary endpoint that is convention-FREE. A literal
    sentinel admits no article and no punctuation ambiguity. That was a third reason RCI-001's
    endpoint was the right one, and it was not among the reasons given when it was chosen.
    """
    def _n(s):
        return " ".join(str(s).strip().lower().split())

    def _canon(s):
        return " ".join(str(s).strip().lower().replace(".", " ").split())

    strict = canon = tolerant = 0
    for r in rows:
        raw, ref = predictions.get(r["id"], ""), r["reference"]
        strict += _n(raw) == _n(ref)
        canon += _canon(raw) == _canon(ref)
        bare = _ARTICLE.sub("", _canon(raw))
        tolerant += bool(bare) and bare == _ARTICLE.sub("", _canon(ref))
    return {"n": len(rows), "strict": strict, "punctuation_insensitive": canon,
            "article_tolerant": tolerant,
            "rows_turning_on_punctuation": canon - strict,
            "rows_turning_on_article": tolerant - canon,
            "share_turning_on_convention": (round((tolerant - strict) / len(rows), 4)
                                            if rows else 0.0),
            "note": "prefer a convention-free primary endpoint; a literal sentinel is one"}


def gate_strata(scores: Mapping[str, tuple], floors_by_stratum: Mapping[str, float],
                ceiling: float = DEFAULT_CEILING,
                candidate_scores: Mapping[str, tuple] | None = None) -> dict:
    """Apply the gate PER STRATUM, not only to the aggregate.

    RCI-001's aggregate cleared its floor and sat well below the ceiling, so the instrument
    looked sound. Per task type it was not: `temporal_correction` answerable was 78/78 for both
    systems *and* 78/78 for the recency rule -- simultaneously above the ceiling and at the
    extraction floor, therefore carrying exactly zero information about either model. A fifth
    of the bank was decoration, and nobody saw it until the read was spent.

    A HIGH FLOOR THREATENS AN INSTRUMENT BY TWO DIFFERENT ROUTES, and they need separating
    (the distinction is the peer session's, adopted here):

      SOLVABILITY   a rule with no task competence reaches the score, so an ABSOLUTE claim --
                    "this system can do X" -- is not supported. This is what a floor is
                    normally understood to bound.
      COMPRESSION   BOTH systems adopt the shortcut, cluster at the same score, and the
                    discordant pairs vanish. Nothing is faked; the ability to SEE a difference
                    is destroyed. This is `temporal_correction` in RCI-001 -- 78/78 both arms,
                    zero discordant pairs, p = 1.00.

    A paired contrast is immune to the first and not the second. A deterministic rule emits the
    same string for both systems, so it can never manufacture a difference -- but it can erase
    one by pushing both arms to the same place. Pass `candidate_scores` to detect that.

    `scores` maps stratum -> (correct, n) for the baseline system; `candidate_scores` the same
    for the candidate, when a paired design is being checked.
    """
    per = {name: gate(c, n, floors_by_stratum.get(name, 0.0), ceiling)
           for name, (c, n) in scores.items()}
    informative = [k for k, v in per.items() if v["verdict"] == "CLEARS"]

    compression = {}
    if candidate_scores:
        for name, (c, n) in scores.items():
            if name not in candidate_scores:
                continue
            c2, n2 = candidate_scores[name]
            fl = floors_by_stratum.get(name, 0.0)
            both_high = n and n2 and min(c / n, c2 / n2) >= ceiling
            both_at_floor = n and n2 and fl > 0 and max(c / n, c2 / n2) <= fl + 1e-9
            if both_high or both_at_floor:
                compression[name] = {
                    "baseline": f"{c}/{n}", "candidate": f"{c2}/{n2}", "floor": fl,
                    "mode": "both above ceiling" if both_high else "both at the floor",
                    "consequence": "discordant pairs collapse; the stratum cannot show a "
                                   "paired difference even if one exists"}

    return {"per_stratum": per, "informative": sorted(informative),
            "uninformative": sorted(set(per) - set(informative)),
            "informative_share": (round(len(informative) / len(per), 4) if per else 0.0),
            "power_compression": compression,
            "why": "a stratum above its ceiling cannot show a regression and a stratum at its "
                   "extraction floor was never being read; either way it is not measuring",
            "why_compression_is_separate": (
                "a floor that both systems reach does not fake a difference -- a deterministic "
                "rule gives both the same answer -- it removes the ability to see one")}


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


def structural(rows: Sequence[Mapping], abstain: str | None = None) -> dict[str, int]:
    """THE CLASS A RED TEAM FOUND AND NEITHER PUBLISHED CLASS COULD SEE.

    Distributional predictors read the answer distribution. Extraction predictors read positions
    and frequencies. Neither can see *relational* structure between two spans of the context,
    and that is where the worst defect in RCI-002's first draft lived.

    The generator wrote a decoy sentence listing three of the four roles -- and the one it left
    out was always the exempt one. Set difference over the two listings scored **1.000** on both
    carve-out strata and **120/120 on the sealed bank**, solving the primary endpoint outright.
    Every published floor read 0.20-0.29 on those same strata and saw nothing, because no
    positional or distributional rule can express "the member of list A missing from list B".
    The block had been added to CLOSE a positional leak; it opened a worse one.

    These are deliberately generic -- they do not know this instrument's phrasing:

      set_complement_singleton  two same-class listings, one nested in the other, differing by
                                exactly one element -> answer that element
      unique_candidate          exactly one answer-vocabulary member appears -> answer it
      list_arity_minus_k        count the longest comma list, answer count - k (k = 0..3), the
                                attack on any "how many after exclusions" task
      abstain_if_short          omission shortens the context, so answer the sentinel whenever
                                the row is shorter than its stratum median
    """
    vocab = _vocab(rows)
    scores = {"set_complement_singleton": 0, "unique_candidate": 0, "abstain_if_short": 0}
    for k in range(4):
        scores[f"list_arity_minus_{k}"] = 0

    med: dict = {}
    for r in rows:
        med.setdefault(r.get("stratum", "_all"), []).append(len(_ctx_of(r)))
    med = {k: sorted(v)[len(v) // 2] for k, v in med.items()}

    for r in rows:
        ctx = str(r.get("context") or _ctx_of(r))
        ref = str(r["reference"])
        cands = vocab.get(r.get("stratum", "_all"), set())
        if not ctx:
            continue

        # -- set complement over sentence-level candidate sets
        # Split on FULL STOPS only. Semicolons are how this generator writes its listings, so
        # splitting on them shattered the very sets this predictor has to compare, and the
        # fixture for the leak scored 0 until that was fixed.
        sents = [s for s in re.split(r"(?<=\.)\s+", ctx) if s]
        sets = [frozenset(c for c in cands if c in s) for s in sents]
        sets = [s for s in sets if len(s) >= 3]
        found = None
        for i, a in enumerate(sets):
            for b in sets[i + 1:]:
                big, small = (a, b) if len(a) > len(b) else (b, a)
                if small < big and len(big - small) == 1:
                    found = next(iter(big - small))
                    break
            if found:
                break
        if found is not None and found == ref:
            scores["set_complement_singleton"] += 1

        # -- candidate-set collapse
        present = [c for c in cands if c in ctx]
        if len(present) == 1 and present[0] == ref:
            scores["unique_candidate"] += 1

        # -- arity of the longest comma list
        best = max((len(re.split(r",| and ", seg)) for seg in sents), default=0)
        for k in range(4):
            if str(best - k) == ref:
                scores[f"list_arity_minus_{k}"] += 1

        # -- length tell for omission
        if abstain is not None and len(ctx) < med.get(r.get("stratum", "_all"), 0) \
                and ref == abstain:
            scores["abstain_if_short"] += 1

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
    st = structural(rows, abstain=abstain)
    if extra:
        e.update({k: fn(rows) for k, fn in extra.items()})
    allp = {**d, **e, **st}
    n = len(rows)
    binding = max(allp.values())
    pred = max(allp, key=allp.get)
    return {
        "n": n,
        "distributional": d,
        "extraction": e,
        "structural": st,
        "binding_floor_count": binding,
        "binding_floor_rate": round(binding / n, 4),
        "binding_predictor": pred,
        "binding_class": ("extraction" if pred in e else
                          "structural" if pred in st else "distributional"),
        "distinct_references": len(set(str(r["reference"]) for r in rows)),
        "note": ("All THREE predictor classes are required. A bank can look hard "
                 "distributionally and be trivial to an extractor -- RCI-001 published four "
                 "distributional floors at 0.2773 and a recency rule scored 0.3711 -- and it "
                 "can clear both and still fall to a relational rule: RCI-002's first draft "
                 "read 0.20-0.29 on every published floor while set-complement scored 1.000."),
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

    # The structural class, scored on the subset with the same full-bank calibration. An
    # abstention endpoint is only extraction-immune if NOTHING that reads the context can emit
    # the sentinel -- and `abstain_if_short` can, which is exactly why it belongs here.
    st = structural([r for r in rows if id(r) in ids], abstain=abstain)
    ex.update({f"structural::{k}": v for k, v in st.items() if not k.startswith("abstain_")})
    ex_max = max(ex.values()) if ex else 0
    return {
        "n": len(sub),
        "extraction": ex, "extraction_max": ex_max,
        "extraction_immune": ex_max == 0,
        "structural_abstainers": {k: v for k, v in st.items() if k.startswith("abstain_")},
        "distributional": dist, "distributional_max": max(dist.values()) if dist else 0,
        "distributional_immune": (max(dist.values()) if dist else 0) == 0,
        "required_control_if_not_distributionally_immune": (
            "false abstention on answerable rows must be reported and near zero; without it, "
            "'the system just emits the sentinel often' is a live alternative explanation"),
        "note": ("Choose a primary endpoint that is EXTRACTION-immune, and pair it with the "
                 "false-abstention control. Extraction immunity is structural; distributional "
                 "immunity is not obtainable by endpoint choice alone."),
    }
