"""MRAT-001 — mention COUNT separated from salience RATIO, at FIXED answer position.

    python probe/build_mrat.py --out <records>/mrat001-<date>/MRAT_PROBE.jsonl

MPOS-002 established that the position effect is **causal** but could not separate mention count
from salience ratio: every isolating contrast ran through its most cue-contaminated cell, and its
two clean cells differed in **both** factors at once — the ALIAS-001 confound it was built to
break, reappearing. That decomposition is what this bank is for.

## Why a separate bank, and why it is cleaner

The contamination in MPOS-002 was created **by manipulating position**: any rule reading position
tracks a position manipulation by definition, and sentence-index rules swung 0.79 even in cells
declared clean. **Here position is held FIXED**, so there is no band, no band-dependent rule, and
that entire class of cue cannot exist.

## The lattice

`k` = times the answer is named, `d` = times EACH of three distractors is named.

| cell | k | d | ratio | role |
|---|---|---|---|---|
| `k1d1` | 1 | 1 | 1:1 | count axis |
| `k2d2` | 2 | 2 | 1:1 | count axis |
| `k3d3` | 3 | 3 | 1:1 | count axis |
| `k3d1` | 3 | 1 | 3:1 | ratio axis |
| `k3d2` | 3 | 2 | 3:2 | ratio axis |

* **count at fixed ratio:** `k1d1` vs `k2d2` vs `k3d3` — all 1:1, count 1/2/3.
* **ratio at fixed count:** `k3d1` vs `k3d2` vs `k3d3` — all count 3, ratio 3:1 / 3:2 / 1:1.

`k3d3` is the hinge of both axes and is **frequency-neutral**, which the MPOS `k3d3` was not.

## The one cue that cannot be closed, named in advance

**Every COUNT-DERIVED feature necessarily tracks these axes.** `most_frequent` is the obvious
one, but so are mention `spread` and sentences-mentioned-in: all are functions of how many times
a candidate is named, and the manipulation IS how many times a candidate is named. The build
reports their swings in full and exempts them from refusal; the gate stays strict for
position-derived, name-derived and alphabetical rules, where a swing would be a real defect. Where k > d the
answer is the most frequent candidate by construction; where k = d it is tied. That is the ratio
manipulation restated, exactly as sentence-index rules were the position manipulation restated in
MPOS-002 — *you cannot manipulate salience ratio and leave frequency rules uninformative.*

**Consequence, registered before any run:** the **count axis is clean** — all three of its cells
are frequency-neutral, so no frequency rule can distinguish them. **The ratio axis is not**, and
its cells must be read with the same care MPOS-002's contaminated cells required. This asymmetry
is a property of the question, not a defect of the build, and it is why the count axis is the
primary.

## Matched scenarios, which is what makes the contrast paired

Every scenario is built **once** and rendered in **all five cells** from the same names, the same
question, the same answer and the same binding sentence — only the number of mentions differs. So
a scenario is a matched set, and the primary can be an arm x cell **interaction**, in which any
cue available to both arms cancels. That is the structure that worked in MPOS-002.

## One sentence per candidate — why mentions repeat WITHIN a sentence

The first build of this bank was **refused by its own gate**: `in_sentence_7` swung
**0.000 in `k1d1` to 1.000 in `k3d3`**, because placing k mentions in k consecutive slots makes
the answer occupy MORE SENTENCES as k rises. Sentence-index rules then track the count axis
exactly as they tracked position in MPOS-002.

That is closable here, and by construction rather than by relaxing the gate: **every candidate
occupies exactly ONE sentence in every cell**, and `k` (or `d`) is the number of times that
candidate is named *inside its own sentence*. The body therefore has the same sentence structure
in all five cells — one answer sentence at a fixed anchor, three distractor sentences, one action
sentence, and neutral padding — and no sentence-index rule can distinguish the cells at all.

## Fixed position is an assertion, not an intention

Varying k while placing extra mentions anywhere would move the answer's MEAN position with count
— re-introducing exactly the variable MPOS-002 showed is causal, as a confound with the count
axis. So the k mentions occupy **consecutive slots centred on a fixed anchor**, and the build
**refuses** unless mean answer position is matched across every cell to a tight tolerance.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))

from build_mpos import BODY_N, NEUTRAL  # noqa: E402
from build_pres4 import (ACTIVITY, BRIDGE, DAYS, FILLER, ITEMS, PEOPLE, PLACES,  # noqa: E402
                         SENTINEL, SITES, SYSTEM, VARIANTS, _ART, h)
from floors import floors as gate_floors  # noqa: E402
from prose import (answer_sentence_locator, render_sample, scan as prose_scan)  # noqa: E402
from rule_sweep import sweep  # noqa: E402

CELLS = (("k1d1", 1, 1), ("k2d2", 2, 2), ("k3d3", 3, 3), ("k3d1", 3, 1), ("k3d2", 3, 2))
COUNT_AXIS = ("k1d1", "k2d2", "k3d3")
RATIO_AXIS = ("k3d1", "k3d2", "k3d3")
N_DISTRACTORS = 3
STRATUM = "person_binding"
#: middle of the body; the answer's mentions are centred here in EVERY cell
ANCHOR = BODY_N // 2
#: mean answer position must match across cells to within this, or the build refuses
POSITION_TOLERANCE = 0.06
#: a non-frequency rule may not track the cell manipulation by more than this
RULE_SWING_TOLERANCE = 0.25
#: COUNT-DERIVED features. `count`, `spread` and `nsent` are all functions of how many times a
#: candidate is named, and this bank's manipulation IS how many times a candidate is named -- so
#: they track it by construction, on BOTH axes. The same tautology as sentence-index rules
#: tracking a position manipulation in MPOS-002: you cannot manipulate mention count and leave
#: count-derived features uninformative.
#:
#: They are EXEMPT FROM THE REFUSAL and REPORTED IN FULL. The gate stays strict for everything
#: that COULD have been closed -- position-derived, name-derived and alphabetical rules -- which
#: is the class where a swing means a build defect rather than a restatement of the question.
COUNT_DERIVED = ("count", "spread", "nsent")


def _is_count_derived(rule: str) -> bool:
    return any(rule.endswith("_" + f) for f in COUNT_DERIVED)


def place(rng, answer_lines, distractor_lines, other, anchor):
    """Answer mentions in CONSECUTIVE slots centred on `anchor`; ends pinned to distractors."""
    if len(distractor_lines) < 2:
        raise SystemExit("need >= 2 distractor lines to pin both ends")
    head, tail = distractor_lines[0], distractor_lines[1]
    rest = list(other) + list(distractor_lines[2:])
    k = len(answer_lines)
    n = k + len(rest) + 2
    start = anchor - (k - 1) // 2
    start = max(1, min(start, n - 1 - k))          # never slot 0 or the final slot
    slots = list(range(start, start + k))
    rng.shuffle(rest)
    out, ai, oi = [], 0, 0
    for i in range(n):
        if i == 0:
            out.append(head)
        elif i == n - 1:
            out.append(tail)
        elif ai < k and i == slots[ai]:
            out.append(answer_lines[ai])
            ai += 1
        else:
            out.append(rest[oi])
            oi += 1
    return out


def build(cell, k, d, packs, seed):
    bridge, acts = BRIDGE[STRATUM], ACTIVITY[STRATUM]
    rows = []
    for i in range(packs):
        sid = f"MR-{i:04d}"
        for variant in VARIANTS:
            # content RNG omits the CELL, so every cell draws the same names, items, days and
            # templates -- the scenario is a matched set across cells, differing only in counts.
            r = random.Random(h(seed, sid, variant))
            rp = random.Random(h(seed, cell, sid, variant))
            cands = r.sample(PEOPLE, N_DISTRACTORS + 1)
            answer, distractors = cands[0], cands[1:]
            other_item, other_place = r.choice(ITEMS), r.choice(PLACES)

            # ONE sentence, naming the answer exactly k times. See the module docstring: this
            # is what stops sentence-index rules from tracking the count axis.
            # PARALLEL templates: the answer's sentence and every distractor's sentence
            # have the same shape and comparable length, so the WITHIN-SENTENCE spacing of
            # repeated names matches. Without that, `argmin_spread` picked the answer 100% of
            # the time in k3d3 and 0% in k1d1 -- a second rule family tracking the count axis,
            # caught by the build gate after the sentence-index family was closed.
            # SPAN-MATCHED templates. Parallel SHAPE was not enough: the text BETWEEN
            # repeated names differed, so the answer's first-to-last span differed from the
            # distractors' -- and the bias REVERSED with k (answer longer at k=2, shorter at
            # k=3), making `argmin_spread` prefer the answer 0.396 of the time in k3d3 against
            # a 0.250 chance level. Here the connecting text is character-matched segment by
            # segment: "the provisioning steward" / "present at the stocktake" are both 24
            # chars, "duty"/"gate" both 4, "entry"/"sheet" both 5 -- so the span between a
            # candidate's first and last mention is identical for the answer and every
            # distractor, and spread carries no answer preference at any k.
            binding = {
                1: f"{answer} was {bridge} that week, according to the book.",
                2: f"{answer} was {bridge} that week, and {answer} signed the duty book.",
                3: (f"{answer} was {bridge} that week, {answer} signed the duty book, and "
                    f"{answer} countersigned the entry."),
            }[k]
            action = (f"It was {bridge} who carried {other_item} to {other_place}, "
                      f"unaccompanied.")
            q = (f"Who personally carried {other_item} to {other_place}?"
                 if variant != "paraphrase"
                 else f"By name, who took {other_item} to {other_place} alone?")

            answer_lines = [binding]
            # one sentence per distractor too, naming it exactly d times
            distractor_lines = []
            for x in distractors:
                day = r.choice(DAYS)
                distractor_lines.append({
                    1: f"{x} was present at the stocktake that week, according to the book.",
                    2: (f"{x} was present at the stocktake that week, and {x} signed the "
                        f"gate book."),
                    3: (f"{x} was present at the stocktake that week, {x} signed the gate "
                        f"book, and {x} countersigned the sheet."),
                }[d])
            rp.shuffle(distractor_lines)
            other_lines = [action]
            pad = BODY_N - len(answer_lines) - len(distractor_lines) - len(other_lines)
            if pad < 0:
                raise SystemExit(f"{cell}: needs {-pad} sentences more than BODY_N")
            other_lines += [ln.format(d=r.choice(DAYS))
                            for ln in r.sample(NEUTRAL + FILLER, pad)]

            body = place(rp, answer_lines, distractor_lines, other_lines, ANCHOR)
            head = (f"This is a consolidated account of the audit at {r.choice(SITES)}, "
                    f"covering six consecutive days, transcribed in the order the entries "
                    f"were found.")
            ctx = " ".join([head] + body)

            counts = {c: ctx.count(c) for c in cands}
            if counts[answer] != k or any(counts[x] != d for x in distractors):
                raise SystemExit(f"{sid}/{cell}: counts {counts} != k={k}, d={d}")
            if ctx.lower().count(bridge.lower()) != 2:
                raise SystemExit(f"{sid}/{cell}: bridge must appear exactly twice")
            if "the the " in ctx.lower() or SENTINEL.lower() in (ctx + q).lower():
                raise SystemExit(f"{sid}/{cell}: prose defect or sentinel leak")
            if _ART.match(answer):
                raise SystemExit(f"{sid}: reference carries a leading article")

            occ, j = [], ctx.find(answer)
            while j >= 0:
                occ.append(j / len(ctx))
                j = ctx.find(answer, j + 1)

            rows.append({
                "id": f"{sid}-{cell.upper()}-{variant[:3].upper()}",
                "scenario_id": sid, "match_id": f"{sid}-{variant[:3].upper()}",
                "stratum": STRATUM, "cell": cell, "k": k, "d": d,
                "ratio": f"{k}:{d}", "frequency_neutral": k == d,
                "on_count_axis": cell in COUNT_AXIS, "on_ratio_axis": cell in RATIO_AXIS,
                "variant": variant, "partition": "PROBE", "task_type": STRATUM,
                "answerable": True, "answer_mentions": k, "distractor_mentions": d,
                "prompt": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": f"CONTEXT\n{ctx}\n\nQUESTION\n{q}"}],
                "reference": answer, "reference_normalised": answer.lower(),
                "context": ctx, "question": q,
                "answer_mean_pos": round(sum(occ) / len(occ), 6),
                "answer_first_pos": round(occ[0], 6),
                "context_words": len(ctx.split()), "context_sha256": h(ctx),
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--packs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args(argv)

    rows = []
    for cell, k, d in CELLS:
        rows += build(cell, k, d, a.packs, a.seed)

    # --- POSITION must be matched across cells, or count confounds with position -------
    mp = {c: sum(r["answer_mean_pos"] for r in rows if r["cell"] == c)
          / sum(1 for r in rows if r["cell"] == c) for c, _, _ in CELLS}
    spread = max(mp.values()) - min(mp.values())
    if spread > POSITION_TOLERANCE:
        raise SystemExit(f"mean answer position differs across cells by {spread:.4f} "
                         f"(> {POSITION_TOLERANCE}): {mp}. Count would confound with POSITION, "
                         f"which MPOS-002 established is causal.")

    # --- matched scenarios across all five cells --------------------------------------
    match: dict = {}
    for r in rows:
        match.setdefault(r["match_id"], {})[r["cell"]] = r
    bad = [m for m, v in match.items() if len(v) != len(CELLS)]
    if bad:
        raise SystemExit(f"{len(bad)} scenarios missing a cell, e.g. {bad[:3]}")
    for m, v in match.items():
        refs = {v[c]["reference"] for c in v}
        qs = {v[c]["question"] for c in v}
        if len(refs) != 1 or len(qs) != 1:
            raise SystemExit(f"{m}: matched scenario disagrees on answer or question")

    # --- RULE SWEEP AS A BUILD GATE, on both axes -------------------------------------
    # MPOS-002 ran this DURING generation and had to withdraw a registered claim four minutes
    # before the first arm finished. After the first arm lands, an honest withdrawal and moving
    # the goalposts are externally indistinguishable -- so this runs here, before any token.
    pool = sorted({r["reference"] for r in rows})
    sweeps, violations = {}, []
    for axis, cells in (("count", COUNT_AXIS), ("ratio", RATIO_AXIS)):
        lo, hi = cells[0], cells[-1]
        sub = [r for r in rows if r["cell"] in (lo, hi)]
        res = sweep(sub, pool, "cell", (lo, hi))
        sweeps[axis] = res
        for rule in res["rules"]:
            if rule["abs_swing"] <= RULE_SWING_TOLERANCE:
                continue
            if _is_count_derived(rule["rule"]):
                continue          # tautological on both axes; declared and reported above
            violations.append({"axis": axis, **rule})
    if violations:
        print(json.dumps(violations[:8], indent=2))
        raise SystemExit(f"REFUSING to write: {len(violations)} rule(s) track the cell "
                         f"manipulation by more than {RULE_SWING_TOLERANCE} outside the declared "
                         f"frequency exemption.")

    pd = prose_scan(rows)
    if not pd["clean"]:
        print(json.dumps(pd, indent=2))
        raise SystemExit("REFUSING to write: mechanical prose defects")
    # The locator runs PER CELL, not pooled. In a frequency-NEUTRAL cell (k == d) every
    # candidate's sentence names it equally often, so a capital-count or length rule that
    # locates the answer's sentence there is a real defect. In a k != d cell it is the ratio
    # manipulation restated -- the answer's sentence necessarily carries more names -- and is
    # reported rather than refused, exactly like the count-derived rule exemption above.
    loc = {c: answer_sentence_locator([r for r in rows if r["cell"] == c],
                                      candidates=pool) for c, _, _ in CELLS}
    dirty = [c for c, k, d in CELLS
             if k == d and loc[c].get("applicable")
             and not loc[c]["clean_vs_candidate_floor"]]
    if dirty:
        for c in dirty:
            print(f"--- {c} (frequency-neutral, must be clean) ---")
            print(json.dumps(loc[c]["leaks_vs_candidate_floor"], indent=2))
        raise SystemExit(f"REFUSING to write: a surface feature LOCATES the answer sentence in "
                         f"frequency-neutral cell(s) {dirty} -- there every candidate's sentence "
                         f"names it equally often, so this is a build defect, not the "
                         f"manipulation restated.")
    leaky_ratio = {c: loc[c].get("leaks_vs_candidate_floor") for c, k, d in CELLS
                   if k != d and loc[c].get("applicable")
                   and not loc[c]["clean_vs_candidate_floor"]}

    fl = {c: gate_floors([r for r in rows if r["cell"] == c]) for c, _, _ in CELLS}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
    man = a.out.with_name("MRAT_BUILD.json")
    man.write_text(json.dumps({
        "record": "MRAT_BUILD", "class": "BUILD -- reads no predictions, registers no criterion",
        "n_rows": len(rows), "packs": a.packs, "seed": a.seed,
        "cells": [{"cell": c, "k": k, "d": d} for c, k, d in CELLS],
        "count_axis": list(COUNT_AXIS), "ratio_axis": list(RATIO_AXIS),
        "matched_scenarios": len(match),
        "mean_answer_position_by_cell": {c: round(v, 4) for c, v in mp.items()},
        "position_spread_across_cells": round(spread, 4),
        "position_tolerance": POSITION_TOLERANCE,
        "rule_sweep": sweeps, "rule_swing_tolerance": RULE_SWING_TOLERANCE,
        "count_derived_exemption": list(COUNT_DERIVED),
        "exempted_rule_swings": {ax: [r for r in sw["rules"] if _is_count_derived(r["rule"])]
                                 for ax, sw in sweeps.items()},
        "floors_per_cell": fl, "prose_scan": pd, "answer_sentence_locator_per_cell": loc,
        "expected_ratio_axis_locator_leaks": leaky_ratio,
        "rendered_sample": render_sample(rows, n=1, seed_field="cell"),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(rows)} rows -> {a.out}")
    print(f"matched scenarios across all {len(CELLS)} cells: {len(match)}")
    print(f"\nmean answer position by cell (spread {spread:.4f}, tolerance "
          f"{POSITION_TOLERANCE}):")
    for c, _, _ in CELLS:
        print(f"  {c:<7}{mp[c]:.4f}")
    print("\nrule sweep at the axis endpoints (build gate):")
    for axis, res in sweeps.items():
        top = res["rules"][:3]
        print(f"  {axis} axis ({res['levels'][0]} vs {res['levels'][1]}), "
              f"{res['n_rules']} rules:")
        for r in top:
            print(f"    {r['rule']:<20}{r['rate_' + res['levels'][0]]:.3f} / "
                  f"{r['rate_' + res['levels'][1]]:.3f}  swing {r['swing']:+.3f}  "
                  f"ties {r['tied_rows']}")
    print(f"\nprose: {'CLEAN' if pd['clean'] else pd['by_kind']}")
    print("locator, per cell (frequency-neutral cells must be clean):")
    for c, k, d in CELLS:
        st = ("CLEAN" if loc[c].get("clean_vs_candidate_floor")
              else str(loc[c].get("leaks_vs_candidate_floor")))
        tag = "neutral" if k == d else "ratio-axis"
        print(f"  {c:<7}{tag:<12}{st}")
    print(f"manifest -> {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
