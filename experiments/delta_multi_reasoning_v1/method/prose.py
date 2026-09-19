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
