"""MPOS-001 — answer position MANIPULATED, and mention count decoupled from salience ratio.

    python probe/build_mpos.py --out <records>/mpos001-<date>/MPOS_PROBE.jsonl

Every position result in this programme so far is **observational**. `build_pres4.py` and
ALIAS-001 both `shuffle(body)`, so where the answer lands is an accident of the shuffle; we then
correlated that accident with damage. The correlation replicated across two independently trained
deltas (Amendments 18-20), which is why it is worth spending a bank to turn it into an
intervention.

Two things this bank does that no existing one can.

## 1. Position is MANIPULATED, not observed

The answer's mentions are placed into a target band — `early` (first third of the body) or
`late` (final third) — instead of being shuffled. Distractors and fillers fill the rest. So
position is assigned, and a position effect becomes a claim about what happens *when you move the
answer*, not about which rows happened to have late answers.

## 2. Mention count is CROSSED with distractor count

ALIAS-001's three conditions were `low` (answer 1x, distractors 3x), `equal` (2/2), `high` (3/1).
**Answer count and salience ratio move together across all three**, so "position sensitivity is
strongest where the answer is mentioned most" and "...where the answer out-mentions its
distractors" are the same sentence about that bank. They are different claims and this bank
separates them: answer count k in {1, 3} crossed with distractor count d in {1, 3}.

| cell | k | d | ratio | note |
|---|---|---|---|---|
| `k1d1` | 1 | 1 | 1:1 | **frequency-neutral** |
| `k1d3` | 1 | 3 | 1:3 | ALIAS-001 `low` |
| `k3d1` | 3 | 1 | 3:1 | ALIAS-001 `high` |
| `k3d3` | 3 | 3 | 1:1 | **frequency-neutral** |

The two frequency-neutral cells are the payoff. In ALIAS-001 only `low` had a readable absolute
rate, because in `high` the answer is the most frequent candidate by construction and
`most_frequent_candidate_in_context` scores 1.000. Here `k1d1` and `k3d3` have the answer and
each distractor mentioned **the same number of times**, so a frequency rule cannot pick the
answer, and they differ from each other *only* in how many mentions exist. That is the cell pair
that isolates mention count from salience.

## What is NOT claimed here

This is a **BUILD**. It registers nothing about the delta and reads no predictions. The scoring
criterion goes in `MPOS001_PREREG.md` before any arm runs, per the rule that scorers are written
before predictions are read. Absolute rates are readable in the frequency-neutral cells only;
paired contrasts are readable everywhere, since an answer-agnostic deterministic rule emits the
same string for both arms and cannot create a discordant pair.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))

from build_pres4 import (ACTIVITY, BRIDGE, DAYS, FILLER, ITEMS, PEOPLE, PLACES,  # noqa: E402
                         SENTINEL, SITES, SYSTEM, VARIANTS, _ART, h)
from floors import floors as gate_floors  # noqa: E402
from prose import (answer_sentence_locator, render_sample, scan as prose_scan,  # noqa: E402
                   stratum_surface_outliers)

#: answer mentions x distractor mentions. k and d are crossed so count and ratio separate.
CELLS = (("k1d1", 1, 1), ("k1d3", 1, 3), ("k3d1", 3, 1), ("k3d3", 3, 3))
BANDS = ("early", "late")
N_DISTRACTORS = 3
BAND_FRACTION = 1.0 / 3.0
STRATUM = "person_binding"          # one filler type: this bank varies position, not filler type

#: Body length is held CONSTANT at this many sentences in every cell, padded with neutral lines.
#: Without it the body runs 7 sentences in `k1d1` and 15 in `k3d3`, so the cell contrast would be
#: confounded with context length -- a known difficulty driver, and the kind of uncontrolled
#: difference that made two same-spec banks land 20 points apart.
BODY_N = 15

#: An extraction rule may not score DIFFERENTLY across the bands. Tested two ways, because a
#: raw gap is the wrong instrument on a noisy estimate: `most_frequent_candidate_in_context` is
#: a 4-way TIE in the frequency-neutral cells, `max()` breaks it by arbitrary set order, and the
#: per-band rate then has binomial spread around 0.25 (sd 0.0625 at n=48). A flat 0.10 gap
#: refused this bank on a p = 0.107 difference. Fisher still catches the real thing by thirty
#: orders of magnitude: the first build's `first_candidate` cue was 34/48 vs 0/48, p = 9.9e-13.
BAND_FLOOR_ALPHA = 0.05 / 16          # 4 extraction rules x 4 cells
BAND_FLOOR_HARD_GAP = 0.25            # refuse a large gap even when underpowered

#: Neutral lines naming NO candidate. `build_pres4.FILLER` has only four, and `k1d1` needs ten.
#: Deliberately free of proper nouns except weekday names, so padding cannot add a candidate or
#: shift the capitalised-token share in a cell-dependent way beyond what the distractors do.
NEUTRAL = [
    "The ledger for {d} was transcribed twice and both copies agree.",
    "Entries recorded on {d} were countersigned before the ledger closed.",
    "A margin note on {d} records that the weather delayed the morning round.",
    "The inventory column for {d} was left blank and later ruled through.",
    "Receipts filed on {d} were numbered in sequence without omission.",
    "The duty roster for {d} survives only as a fragment.",
    "An addendum dated {d} corrects an arithmetic slip in the running total.",
    "Stock counted on {d} matched the previous week to within one crate.",
    "The gate log for {d} shows two entries and no corresponding exit.",
    "A loose leaf dated {d} lists tolls paid at the river crossing.",
    "Candle stubs issued on {d} were charged against the general account.",
    "The copyist notes that the page for {d} was rebound out of order.",
    "Measurements taken on {d} were recorded in the older unit throughout.",
    "A tally scratched on {d} appears in the margin without explanation.",
]


def _fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact. Used to test whether a floor rule tracks the manipulation."""
    from math import comb
    n, r1, c1 = a + b + c + d, a + b, a + c
    if not (r1 and c1 and n):
        return 1.0

    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    p0 = pr(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12)


def place(rng: random.Random, answer_lines: list[str], distractor_lines: list[str],
          other: list[str], band: str) -> list[str]:
    """Put every answer line inside the target third; fill the rest.

    The band is defined over SENTENCE SLOTS in the body, so it does not depend on sentence
    lengths, and `late` means late for every row rather than late on average.

    **The first and last body slots are both reserved for DISTRACTOR lines, in both bands.**
    Without that, moving the answer moves which candidate is first or last, and the first build
    of this bank measured the result: `first_candidate_in_context` scored **0.708 in `early` and
    0.000 in `late`**, `last_candidate_in_context` 0.000 and 0.604. Deterministic answer-agnostic
    rules that track the manipulation exactly. Pinning both ends to distractors forces both rules
    to 0.000 in every band, so neither boundary rule can express the position contrast.
    """
    if len(distractor_lines) < 2:
        raise SystemExit("need >= 2 distractor lines to pin both ends of the body")
    head_line, tail_line = distractor_lines[0], distractor_lines[1]
    other = list(other) + list(distractor_lines[2:])
    n = len(answer_lines) + len(other) + 2
    width = max(len(answer_lines), int(round(n * BAND_FRACTION)))
    lo = 0 if band == "early" else n - width
    # The FINAL body slot is reserved for a non-answer line in BOTH bands. Without this the
    # `late` band puts the answer in the last sentence often enough that "emit the name in the
    # last sentence" locates it at 0.214 against a 0.098 chance floor -- and that cue would sit
    # in exactly ONE arm of the early-vs-late contrast this bank exists to make. A shared-cue
    # immunity argument does not protect a comparison where only one arm carries the cue; that
    # is what forced the 1.81x person-vs-object ratio to be withdrawn. `answer_sentence_locator`
    # REFUSED the first build of this bank on precisely that feature.
    allowed = [i for i in range(lo, lo + width) if 0 < i < n - 1]
    if len(allowed) < len(answer_lines):
        raise SystemExit(f"band {band}: {len(allowed)} usable slots for "
                         f"{len(answer_lines)} answer lines -- widen the band")
    slots = sorted(rng.sample(allowed, len(answer_lines)))
    rest = list(other)
    rng.shuffle(rest)
    out, ai, oi = [], 0, 0
    for i in range(n):
        if i == 0:
            out.append(head_line)
        elif i == n - 1:
            out.append(tail_line)
        elif ai < len(slots) and i == slots[ai]:
            out.append(answer_lines[ai])
            ai += 1
        else:
            out.append(rest[oi])
            oi += 1
    return out


def build(cell: str, k: int, d: int, band: str, packs: int, seed: int) -> list[dict]:
    bridge, acts = BRIDGE[STRATUM], ACTIVITY[STRATUM]
    rows = []
    for i in range(packs):
        sid = f"MP-{cell.upper()}-{band[:1].upper()}{i:04d}"
        for variant in VARIANTS:
            r = random.Random(h(seed, cell, band, sid, variant))
            cands = r.sample(PEOPLE, N_DISTRACTORS + 1)
            answer, distractors = cands[0], cands[1:]
            other_item = r.choice(ITEMS)
            other_place = r.choice(PLACES)

            # NO relative clause. `build_pres4` writes "..., who is not named again in this
            # account", which is TRUE there because K_ANSWER is always 1. Inherited unchanged it
            # is FALSE in every k=3 row -- the context states the answer is not named again and
            # then names it twice more -- and, worse, it is true in the k=1 cells and false in
            # the k=3 cells, so a self-contradiction would be perfectly confounded with the
            # mention-count manipulation. Found by READING a rendered row: prose_scan,
            # answer_sentence_locator, all three floor classes and the band-gap check all
            # passed it. A mechanical check cannot see a sentence that is merely untrue.
            binding = (f"{bridge.capitalize()} that week was {answer}, as recorded in the "
                       f"duty book for the period.")
            action = (f"It was {bridge} who carried {other_item} to {other_place}, "
                      f"unaccompanied.")
            q = (f"Who personally carried {other_item} to {other_place}?"
                 if variant != "paraphrase"
                 else f"By name, who took {other_item} to {other_place} alone?")

            # The answer's EXTRA mentions use the same activity templates as the distractors'.
            # Same surface form, so count is the only thing that differs between them.
            answer_lines = [binding] + [r.choice(acts).format(x=answer, d=r.choice(DAYS))
                                        for _ in range(k - 1)]
            distractor_lines = []
            for x in distractors:
                distractor_lines += [r.choice(acts).format(x=x, d=r.choice(DAYS))
                                     for _ in range(d)]
            r.shuffle(distractor_lines)     # vary WHICH distractor pins each end
            other_lines = [action]
            pad = BODY_N - len(answer_lines) - len(distractor_lines) - len(other_lines)
            if pad < 0:
                raise SystemExit(f"{sid}: cell {cell} needs {-pad} sentences more than BODY_N")
            other_lines += [ln.format(d=r.choice(DAYS))
                            for ln in r.sample(NEUTRAL + FILLER, pad)]

            body = place(r, answer_lines, distractor_lines, other_lines, band)
            head = (f"This is a consolidated account of the audit at {r.choice(SITES)}, "
                    f"covering six consecutive days, transcribed in the order the entries "
                    f"were found.")
            ctx = " ".join([head] + body)

            counts = {c: ctx.count(c) for c in cands}
            if counts[answer] != k or any(counts[x] != d for x in distractors):
                raise SystemExit(f"{sid}/{variant}: counts {counts} != k={k}, d={d}")
            if ctx.lower().count(bridge.lower()) != 2:
                raise SystemExit(f"{sid}/{variant}: bridge must appear exactly twice")
            if "the the " in ctx.lower():
                raise SystemExit(f"{sid}/{variant}: DOUBLE ARTICLE -- the PRES-004 defect")
            if SENTINEL.lower() in (ctx + q).lower():
                raise SystemExit(f"{sid}/{variant}: sentinel leaked")
            if _ART.match(answer):
                raise SystemExit(f"{sid}: reference carries a leading article")

            # position receipts, computed the same way the analyses compute them
            occ, j = [], ctx.find(answer)
            while j >= 0:
                occ.append(j / len(ctx))
                j = ctx.find(answer, j + 1)
            if len(occ) != k:
                raise SystemExit(f"{sid}/{variant}: {len(occ)} occurrences, expected {k}")

            rows.append({
                "id": f"{sid}-{variant[:3].upper()}", "scenario_id": sid,
                "stratum": STRATUM, "cell": cell, "band": band, "variant": variant,
                "partition": "PROBE", "task_type": STRATUM, "answerable": True,
                "answer_mentions": k, "distractor_mentions": d,
                "frequency_neutral": k == d,
                "bridge": bridge,
                "prompt": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": f"CONTEXT\n{ctx}\n\nQUESTION\n{q}"}],
                "reference": answer, "reference_normalised": answer.lower(),
                "context": ctx, "question": q,
                "answer_first_pos": round(occ[0], 6),
                "answer_mean_pos": round(sum(occ) / len(occ), 6),
                "answer_last_pos": round(occ[-1], 6),
                "context_words": len(ctx.split()), "context_sha256": h(ctx),
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--packs-per-cell", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args(argv)

    rows = []
    for cell, k, d in CELLS:
        for band in BANDS:
            rows += build(cell, k, d, band, a.packs_per_cell, a.seed)

    # --- the band must actually separate, in every cell, or the manipulation did not happen ---
    sep = {}
    for cell, _, _ in CELLS:
        by = {b: [r["answer_mean_pos"] for r in rows if r["cell"] == cell and r["band"] == b]
              for b in BANDS}
        lo_hi = max(by["early"]), min(by["late"])
        sep[cell] = {"early_max": round(lo_hi[0], 4), "late_min": round(lo_hi[1], 4),
                     "disjoint": lo_hi[0] < lo_hi[1],
                     "early_mean": round(sum(by["early"]) / len(by["early"]), 4),
                     "late_mean": round(sum(by["late"]) / len(by["late"]), 4)}
        if not sep[cell]["disjoint"]:
            raise SystemExit(f"{cell}: early and late mean-position ranges OVERLAP "
                             f"({lo_hi[0]:.4f} >= {lo_hi[1]:.4f}) -- the manipulation is not "
                             f"clean and a position contrast here would be confounded")

    # --- prose: the defect class that survived every structural check on PRES-004 ---
    pd = prose_scan(rows)
    if not pd["clean"]:
        print(json.dumps(pd, indent=2))
        raise SystemExit("REFUSING to write: mechanical prose defects")

    loc = answer_sentence_locator(rows)
    if loc.get("applicable") and not loc["clean"]:
        print(json.dumps(loc, indent=2))
        raise SystemExit("REFUSING to write: a surface feature LOCATES the answer sentence")

    surf = stratum_surface_outliers(rows, stratum_field="cell")

    fl = {c: gate_floors([r for r in rows if r["cell"] == c]) for c, _, _ in CELLS}
    overall = gate_floors(rows)

    # --- floors must not track the MANIPULATION -------------------------------------------
    # The first build of this bank passed every per-cell floor and was still broken: pooling
    # the two bands, `first_candidate_in_context` read 0.354 in `k1d1`, which is the average of
    # 0.708 in `early` and 0.000 in `late`. A rule that scores differently across the bands is a
    # cue in ONE ARM of the contrast the bank exists to make, and paired-arm immunity does not
    # protect it -- that is what forced the 1.81x person-vs-object ratio to be withdrawn.
    # Per-stratum floors cannot see this: the manipulation is orthogonal to the stratum.
    band_floors, band_gaps = {}, []
    for c, _, _ in CELLS:
        per_band = {b: gate_floors([r for r in rows if r["cell"] == c and r["band"] == b])
                    for b in BANDS}
        band_floors[c] = {b: v["extraction"] for b, v in per_band.items()}
        n_b = {b: per_band[b]["n"] for b in BANDS}
        for rule in per_band["early"]["extraction"]:
            rates = {b: per_band[b]["extraction"][rule] / n_b[b] for b in BANDS}
            gap = abs(rates["early"] - rates["late"])
            ke, kl = (per_band[b]["extraction"][rule] for b in BANDS)
            pv = _fisher(ke, n_b["early"] - ke, kl, n_b["late"] - kl)
            if pv < BAND_FLOOR_ALPHA or gap > BAND_FLOOR_HARD_GAP:
                band_gaps.append({"cell": c, "rule": rule, "early": f"{ke}/{n_b['early']}",
                                  "late": f"{kl}/{n_b['late']}", "gap": round(gap, 4),
                                  "fisher_p": pv})
    if band_gaps:
        print(json.dumps(band_gaps, indent=2))
        raise SystemExit(
            f"REFUSING to write: {len(band_gaps)} extraction rule(s) score differently across "
            f"the bands (Fisher p < {BAND_FLOOR_ALPHA:.4f} or gap > {BAND_FLOOR_HARD_GAP}). A "
            f"deterministic answer-agnostic rule that tracks the manipulation is a cue in one "
            f"arm of the primary contrast.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
    man = a.out.with_name("MPOS_BUILD.json")
    man.write_text(json.dumps({
        "record": "MPOS_BUILD", "class": "BUILD -- reads no predictions, registers no criterion",
        "n_rows": len(rows), "cells": [{"cell": c, "k": k, "d": d} for c, k, d in CELLS],
        "bands": list(BANDS), "packs_per_cell": a.packs_per_cell, "seed": a.seed,
        "band_separation": sep, "floors_per_band": band_floors,
        "band_floor_alpha": BAND_FLOOR_ALPHA, "band_floor_hard_gap": BAND_FLOOR_HARD_GAP, "floors_per_cell": fl, "floors_overall": overall,
        "prose_scan": pd, "answer_sentence_locator": loc, "surface_outliers": surf,
        "rendered_sample": render_sample(rows, n=1, seed_field="cell"),
        "why": ("position is MANIPULATED here, not shuffled; and answer count is crossed with "
                "distractor count so mention count separates from salience ratio"),
        "readable": ("absolute rates in the frequency-neutral cells k1d1 and k3d3; paired "
                     "contrasts in every cell"),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(rows)} rows -> {a.out}")
    print(f"\n{'cell':<7}{'k':>3}{'d':>3}  {'early mean':>11}{'late mean':>11}  separation")
    for c, k, d in CELLS:
        s = sep[c]
        print(f"{c:<7}{k:>3}{d:>3}  {s['early_mean']:>11.4f}{s['late_mean']:>11.4f}  "
              f"{'disjoint' if s['disjoint'] else 'OVERLAP'}"
              f"  (max early {s['early_max']:.3f} < min late {s['late_min']:.3f})")
    print(f"\nfloors (overall): {overall}")
    for c, _, _ in CELLS:
        print(f"  {c:<7}{fl[c]}")
    print(f"\nprose: {'CLEAN' if pd['clean'] else pd['by_kind']}")
    print(f"answer-sentence locator: {'CLEAN' if loc.get('clean') else loc.get('leaks')}")
    print(f"surface outliers across cells: "
          f"{'CLEAN' if surf.get('clean') else surf.get('outliers')}")
    print(f"\nmanifest -> {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
