"""Prose defects in generated bank text — the class that survived every structural check.

`The the provisioning steward that week was Annika Oyelowo, which is not named again...`

That appeared on **360 of 360 rows** of PRES-004, landing on the binding sentence — the one
sentence containing the answer — and it survived:

  * all three floor classes (distributional, extraction, structural) at 0.167-0.175;
  * `gate()` on the frozen source;
  * `convention_sensitivity()`;
  * per-pack mention-count assertions;
  * a verbatim-bridge occurrence assertion.

**Five checks, none of which operate on the text as text.** It was visible to the naked eye in
the first rendered context anyone printed, and it was printed only after the result was in hand.
The peer session found a mislabelled condition in its own published table the same way, in four
minutes, the day it started printing rows.

## What this module does and does not claim

It catches the **mechanical** half — a doubled word, a doubled space, a lowercase sentence start,
a dangling article. Those are automatable and today's defect was one of them.

It does **not** replace reading. A mislabelled condition, a question that asks for the wrong
thing, an answer that is true but for the wrong reason — none of that is detectable here.
`render_sample()` exists so the rendered rows land **in the build manifest**, where a reviewer
can read them without re-running anything, because the failure mode is not "nobody could read
it" but "nobody did".
"""
from __future__ import annotations

import re
from collections.abc import Sequence

#: A word repeated immediately. Catches `The the`, `was was`, `to to`. Case-insensitive on the
#: pair but requires the same word, so "That that" is caught and "Had had" is too -- both are
#: far more likely to be generator defects than intended English in a synthetic bank.
_DOUBLED_WORD = re.compile(r"\b(\w+)\s+\1\b", re.I)
_DOUBLED_SPACE = re.compile(r"  +")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+[,.;:]")
#: A sentence beginning lowercase, e.g. from capitalising a bridge that already had a determiner
_LOWER_SENTENCE_START = re.compile(r"(?:^|(?<=[.!?])\s+)([a-z])")
#: An article with nothing after it, or two stacked determiners
_STACKED_DETERMINER = re.compile(r"\b(the|a|an)\s+(the|a|an)\b", re.I)


def prose_defects(text: str) -> dict[str, list[str]]:
    """Mechanical defects in one rendered context. Empty dict means nothing mechanical found."""
    out: dict[str, list[str]] = {}

    def add(k, m):
        if m:
            out[k] = m

    add("doubled_word", [m.group(0) for m in _DOUBLED_WORD.finditer(text)])
    add("stacked_determiner", [m.group(0) for m in _STACKED_DETERMINER.finditer(text)])
    add("doubled_space", _DOUBLED_SPACE.findall(text))
    add("space_before_punctuation", _SPACE_BEFORE_PUNCT.findall(text))
    add("lowercase_sentence_start",
        [m.group(1) for m in _LOWER_SENTENCE_START.finditer(text)])
    return out


def scan(rows: Sequence[dict], field: str = "context") -> dict:
    """Scan a whole bank. Returns a summary a build can REFUSE on."""
    hits: dict[str, int] = {}
    examples: dict[str, str] = {}
    affected = 0
    for r in rows:
        d = prose_defects(str(r.get(field, "")))
        if d:
            affected += 1
        for k, v in d.items():
            hits[k] = hits.get(k, 0) + len(v)
            examples.setdefault(k, f"{r.get('id', '?')}: ...{v[0]}...")
    return {
        "rows_scanned": len(rows),
        "rows_with_defects": affected,
        "share": round(affected / len(rows), 4) if rows else 0.0,
        "by_kind": hits,
        "examples": examples,
        "clean": affected == 0,
        "note": ("mechanical defects only -- this cannot see a mislabelled condition, a question "
                 "that asks for the wrong thing, or an answer right for the wrong reason. "
                 "render_sample() puts rows in the manifest so a reviewer can read them."),
    }


def render_sample(rows: Sequence[dict], n: int = 3, field: str = "context",
                  seed_field: str = "stratum") -> dict:
    """N rendered rows per stratum, verbatim, for the build manifest.

    Not a check. The point is that the text ends up in the artifact, so reading it costs a
    reviewer nothing and skipping it is visible.
    """
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get(seed_field, "_all")), []).append(r)
    return {k: [{"id": r.get("id"), "question": r.get("question"),
                 "reference": r.get("reference"), field: str(r.get(field, ""))}
                for r in v[:n]]
            for k, v in by.items()}


def stratum_surface_outliers(rows: Sequence[dict], field: str = "context",
                             stratum_field: str = "stratum", z: float = 3.0) -> dict:
    """Is any stratum a SURFACE-STATISTICAL outlier against the others?

    The lowercase-sentence-start defect was one instance of a general class: **a stratum whose
    prose differs from its siblings in a way no one thought to check.** `prose_defects` catches
    the instance because it was found; this catches the class.

    The generalisation matters because the leak is a **TYPE** difference, and type differences
    leak through whatever representation the type is expressed in:

      * linguistic: common noun vs proper noun -> sentence-initial case separates the strata,
        and capitalisation became a perfect distractor/answer predicate in one stratum;
      * structural: `"amount": -4` vs `"amount": "between -6 and -2"` -> a JSON type check
        separates matched members with no false positives.

    Same failure, different substrate. Nobody predicts which surface statistic will carry it, so
    the check should not require a prediction: compute a handful per stratum and refuse on any
    outlier.

    **Only meaningful with 3+ strata** -- with two, "outlier" is undefined and this returns
    `applicable: False` rather than a misleading pass.
    """
    import statistics as _st

    def feats(txt: str) -> dict:
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", txt) if x.strip()]
        toks = txt.split()
        return {
            "lower_sentence_start_share": (
                sum(1 for x in sents if x[:1].islower()) / len(sents)) if sents else 0.0,
            "determiner_share": (
                sum(1 for t in toks if t.lower().strip(".,;:") in ("the", "a", "an"))
                / len(toks)) if toks else 0.0,
            "mean_token_len": (sum(len(t) for t in toks) / len(toks)) if toks else 0.0,
            "punct_density": (sum(txt.count(c) for c in ",;:") / len(toks)) if toks else 0.0,
            "capitalised_token_share": (
                sum(1 for t in toks[1:] if t[:1].isupper()) / len(toks)) if toks else 0.0,
        }

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get(stratum_field, "_all")), []).append(feats(str(r.get(field, ""))))
    if len(by) < 3:
        return {"applicable": False, "n_strata": len(by),
                "why": "outlier detection needs 3+ strata; with 2 there is no 'others'"}

    means = {k: {f: _st.mean(x[f] for x in v) for f in v[0]} for k, v in by.items()}
    flags = []
    for f in next(iter(means.values())):
        for k in means:
            others = [means[o][f] for o in means if o != k]
            mu = _st.mean(others)
            sd = _st.pstdev(others) if len(others) > 1 else 0.0
            if sd == 0.0:
                if abs(means[k][f] - mu) > 1e-9:
                    flags.append({"stratum": k, "feature": f, "value": round(means[k][f], 4),
                                  "others_mean": round(mu, 4), "z": None,
                                  "why": "others identical, this one differs"})
            elif abs(means[k][f] - mu) / sd > z:
                flags.append({"stratum": k, "feature": f, "value": round(means[k][f], 4),
                              "others_mean": round(mu, 4),
                              "z": round((means[k][f] - mu) / sd, 2)})
    return {"applicable": True, "n_strata": len(by), "z_threshold": z,
            "per_stratum_means": {k: {f: round(v, 4) for f, v in m.items()}
                                  for k, m in means.items()},
            "outliers": flags, "clean": not flags}


def answer_sentence_locator(rows: Sequence[dict], field: str = "context",
                            ref_field: str = "reference", margin: float = 0.10,
                            candidates: Sequence[str] | None = None) -> dict:
    """Does any SURFACE feature identify the answer-bearing sentence above chance?

    **This is the check both of this bank's defects actually needed, and neither of the other
    two performs it.**

      * `The the` was a unique bigram **on the binding sentence** -- the one containing the
        answer.
      * lowercase-vs-capital separated distractor lines from **answer-bearing** lines.

    `prose_defects` caught the first because a doubled word is mechanically wrong.
    `stratum_surface_outliers` caught the second **only incidentally**, because that cue
    happened to be confined to one stratum. **A cue applied uniformly across every stratum would
    pass both with a perfect score and still hand the model the answer.**

    So test it directly and stratum-internally: split each context into sentences, find the one
    containing the reference, and ask whether a one-line rule picks it above the 1/n_sentences
    floor. This never compares strata, so it is immune to the objection that strata legitimately
    differ in surface form.

    It is `structural()`'s logic moved from **answer selection** to **sentence selection** --
    the same predicate class that found a JSON type tell at 16/64 with zero false positives in
    the peer session's bank.
    """
    import statistics as _st

    def sentences(t):
        return [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]

    picks: dict[str, int] = {}
    floors, n = [], 0
    for r in rows:
        ss = sentences(str(r.get(field, "")))
        ref = str(r.get(ref_field, ""))
        if len(ss) < 2 or not ref:
            continue
        tgt = [i for i, x in enumerate(ss) if ref in x]
        if len(tgt) != 1:          # ambiguous or absent: cannot score this row
            continue
        t = tgt[0]
        n += 1
        floors.append(1.0 / len(ss))
        cand = {
            "first_sentence": 0,
            "last_sentence": len(ss) - 1,
            "longest_sentence": max(range(len(ss)), key=lambda i: len(ss[i])),
            "shortest_sentence": min(range(len(ss)), key=lambda i: len(ss[i])),
            "most_capitals": max(range(len(ss)),
                                 key=lambda i: sum(1 for c in ss[i] if c.isupper())),
            "most_determiners": max(range(len(ss)), key=lambda i: sum(
                1 for w in ss[i].split() if w.lower().strip(".,;:") in ("the", "a", "an"))),
            "has_doubled_word": next((i for i, x in enumerate(ss)
                                      if _DOUBLED_WORD.search(x)), None),
            "lowercase_start": next((i for i, x in enumerate(ss) if x[:1].islower()), None),
            "has_digit": next((i for i, x in enumerate(ss) if any(c.isdigit() for c in x)), None),
            "most_commas": max(range(len(ss)), key=lambda i: ss[i].count(",")),
        }
        for k, v in cand.items():
            if v is not None and v == t:
                picks[k] = picks.get(k, 0) + 1

    if n == 0:
        return {"applicable": False, "why": "no row had exactly one sentence containing its "
                                            "reference"}
    floor = _st.mean(floors)
    rates = {k: v / n for k, v in picks.items()}
    leaks = {k: round(v, 4) for k, v in rates.items() if v > floor + margin}

    # CANDIDATE-SENTENCE floor, added 2026-09-19. `floor` is 1/n_sentences, which assumes every
    # sentence is an equally likely pick. A rule that selects a SENTENCE can only land on the
    # answer via a sentence that contains a candidate at all -- so where only 4 of 16 sentences
    # name anyone, the real chance level is 1/4, not 1/16.
    #
    # Measured instance: on a bank with 4.00 candidate-bearing sentences per row and identical
    # sentence templates, `longest_sentence` scored 0.2569 against a 1/16 = 0.0625 floor and was
    # reported as a leak. It is chance -- it selects whichever candidate has the LONGEST NAME
    # (0.2986 of rows), which is 1-in-4, not a cue.
    #
    # Reported ALONGSIDE the original, never replacing it, so receipts written before this date
    # remain exactly reproducible.
    cand_floors = []
    for r in rows:
        ss = sentences(str(r.get(field, "")))
        if len(ss) < 2:
            continue
        ncand = sum(1 for s in ss if any(str(c) and str(c) in s for c in candidates))             if candidates else 0
        cand_floors.append(1.0 / ncand if ncand else 1.0 / max(1, len(ss)))
    cfloor = _st.mean(cand_floors) if cand_floors else floor
    cleaks = {k: round(v, 4) for k, v in rates.items() if v > cfloor + margin}
    return {
        "applicable": True, "rows_scored": n,
        "chance_floor": round(floor, 4), "margin": margin,
        "chance_floor_candidate_sentences": round(cfloor, 4),
        "leaks_vs_candidate_floor": cleaks,
        "clean_vs_candidate_floor": not cleaks,
        "hit_rate_by_feature": {k: round(v, 4) for k, v in sorted(
            rates.items(), key=lambda kv: -kv[1])},
        "leaks": leaks, "clean": not leaks,
        "note": ("a feature beating 1/n_sentences by the margin LOCATES the answer's sentence; "
                 "this is stratum-internal, so it also catches a cue applied uniformly across "
                 "all strata, which a cross-stratum outlier test structurally cannot see"),
    }
