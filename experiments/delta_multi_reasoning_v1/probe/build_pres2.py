"""PRES-002 — a second preservation instrument, on skills the corpus does not exercise.

    python probe/build_pres2.py --out <records>/pres002-20260919

**MECHANISM SCOUT.** No seal, no read budget, openly re-runnable.

## Why this is the blocking item

`ALIAS_RESOLUTION_INVESTIGATION.md` Amendment 6 claimed direction is "a real axis of the
frontier". Amendment 7 retracted that to **one skill**, because the only second preservation
endpoints available were unusable:

  * **arithmetic /150 is the wrong endpoint CLASS.** It *rises* with dose — INT-001's frontier
    has Supa 47 → 71 at α = 1 — so it is partially **acquired**, not damaged. Nothing is being
    forgotten there for direction to spare, and its null is near-uninformative whatever its n.
  * **`references_retained` is the right class but underpowered ~3×.** Supa 47 → λ=0 36, so 11
    rows damaged; the proportional equivalent of the alias effect is ~4 rows, and exact McNemar
    on 4 discordant pairs bottoms out at p = 0.125.

So "we tested it and it did not transfer" is **false**; "we have not been able to test it" is
true. This bank is built to make the test possible.

## Design constraints, each inherited from a specific past failure

1. **No person names and no role phrases anywhere.** RRP-015's 384-row corpus contains zero of
   both, which is what makes `alias_resolution` an *unexercised* skill. Keeping this bank free of
   them holds that property constant, so a null here cannot be explained by the corpus having
   exercised the skill.
2. **Three structurally distinct strata**, so a null is not one task's quirk:
   * `object_coreference` — a definite description resolved to an inanimate antecedent.
   * `temporal_predecessor` — events on shuffled weekdays; report the immediate predecessor.
   * `container_nesting` — two hops: locate a container by where it stands, then its contents.
3. **All three floor predictor classes**, and the build **REFUSES** above the bound rather than
   publishing it. RCI-001 published its floors and shipped anyway; RCI-002's first draft cleared
   every distributional and extraction floor at 0.20-0.29 and fell to one line of set arithmetic.
4. **Every candidate is named in full.** No stratum requires "the member of list A missing from
   list B", because `structural()` scores 1.000 on that by construction.
5. **Textual order is shuffled in every pack**, so positional predictors cannot stand in for the
   relation. `temporal_predecessor` in particular is only a test of ordering if the textual
   predecessor is not the temporal one — that is asserted per row, not assumed.
6. **The answer is always a short noun phrase from a pool**, so exact match is not a proxy for
   answer form. RCI-002's accuracy endpoint scored 0/24 on both systems purely because the
   frozen source answered substantively in a different shape.

## What it cannot settle

Whether these skills are *damaged* at all. That is measured, not designed: the first read scores
the frozen source against the delta, and only strata showing real damage with adequate power can
serve as preservation endpoints for the direction test. A stratum the delta does not damage is
reported and then dropped, exactly as the arithmetic panel should have been.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
from floors import floors as gate_floors  # noqa: E402

SENTINEL = "UNKNOWN"
SYSTEM = ("You are answering a closed-context reading test. Use only the supplied context. "
          "Reply with the shortest exact answer and no explanation. If the context does not "
          f"determine the answer, reply {SENTINEL}.")

#: A FOURTH entity partition. Disjoint from RCI-001, RCI-002 and ALIAS-001, and deliberately
#: free of anything person-shaped.
ITEMS = ["brass astrolabe", "walnut tally-board", "slate counting-frame", "copper alembic",
         "linen sail-hook", "iron sounding-line", "glass spirit-level", "horn inkwell",
         "leather drum-case", "zinc plumb-bob", "oak cradle-block", "tin lantern-hood"]
CONTAINERS = ["banded chest", "wicker hamper", "elm crate", "canvas holdall",
              "iron strongbox", "cedar trunk"]
PLACES = ["the North Arcade", "the Dye House", "Cellar Nine", "the Rope Walk",
          "the Gauging Room", "the Slip Shed"]
SITES = ["Haverlock Wharf", "the Tolbooth Yard", "Pennant Quay"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
STRATA = ("object_coreference", "temporal_predecessor", "container_nesting")
VARIANTS = ("base", "paraphrase", "reorder")

#: The build refuses above this. Chosen before any floor was computed.
MAX_FLOOR = 0.45

#: Rows dropped because the TEXTUAL predecessor coincided with the TEMPORAL one. Those rows
#: would be answerable by reading order, so the stratum would not measure ordering on them.
#: Recorded rather than silently absorbed -- the per-stratum row counts differ because of it.
SKIPPED: list[str] = []

FILLER = [
    "The inventory was copied from the tally books and carries no editorial comment.",
    "Several entries bear marginal notes in a second hand, reproduced where legible.",
    "Weather for the period was unremarkable and is noted only where it affected access.",
    "Page numbering restarts partway through and the original sequence is not recoverable.",
]


def h(*parts) -> str:
    return hashlib.sha256("|".join(str(x) for x in parts).encode("utf-8")).hexdigest()


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def _pack(rng, stratum, sid, variant):
    """Return (lines, question, reference, assertions) for one pack variant."""
    r = random.Random(h(sid, variant, stratum))
    if stratum == "object_coreference":
        items = r.sample(ITEMS, 4)
        marked, place = items[0], r.choice(PLACES)
        lines = [
            # every candidate named in full, so no set-complement shortcut exists
            f"The note lists {items[0]}, {items[1]}, {items[2]} and {items[3]}.",
            f"Of the items listed, only the {marked} bears a wax seal.",
            f"The wax-sealed item was moved to {place} during the night watch.",
        ]
        # Each item must appear EXACTLY twice. The marked one already has a second mention
        # from the wax-seal line, so it gets no tally-book line; the other three do.
        #
        # The first draft gave every item a tally line, so the marked one appeared three times
        # and the rest twice -- `most_frequent_candidate_in_context` then scored **1.000** and
        # the build refused. That is the same salience leak ALIAS-001 was designed around,
        # arriving from the opposite direction, and it is why frequency is asserted below
        # rather than assumed.
        lines += [f"The {it} was entered in the tally book without comment."
                  for it in items if it != marked]
        q = (f"Which item was moved to {place} during the night watch?" if variant != "paraphrase"
             else f"Name the item taken to {place} on the night watch.")
        counts = {it: sum(ln.count(it) for ln in lines) for it in items}
        if len(set(counts.values())) != 1:
            raise SystemExit(f"{sid}: mention counts not equal across candidates: {counts}")
        return lines, q, marked, {"candidates": 4, "mentions_each": counts[marked]}

    if stratum == "temporal_predecessor":
        items = r.sample(ITEMS, 6)
        days = DAYS[:]                      # all six days used, so every day has an item
        pairs = list(zip(items, days))
        r.shuffle(pairs)
        by_day = {d: it for it, d in pairs}
        ti = r.randrange(1, len(DAYS))      # never the first day: a predecessor must exist
        target_day, prev_day = DAYS[ti], DAYS[ti - 1]
        target, answer = by_day[target_day], by_day[prev_day]
        lines = [f"The {it} was inspected on {d}." for it, d in pairs]
        q = (f"Which item was inspected on the day immediately before the {target} was "
             f"inspected?" if variant != "paraphrase" else
             f"The {target} was inspected on a certain day. Which item was inspected the day "
             f"before that?")
        return lines, q, answer, {"target": target, "target_day": target_day,
                                  "prev_day": prev_day}

    # container_nesting -- two hops: place -> container -> contents
    conts = r.sample(CONTAINERS, 4)
    places = r.sample(PLACES, 4)
    contents = r.sample(ITEMS, 4)
    tgt = r.randrange(4)
    lines = [f"The {conts[i]} stands in {places[i]}." for i in range(4)]
    lines += [f"The {contents[i]} is kept inside the {conts[i]}." for i in range(4)]
    q = (f"What is kept inside the container that stands in {places[tgt]}?"
         if variant != "paraphrase" else
         f"One container stands in {places[tgt]}. Name the item kept inside it.")
    return lines, q, contents[tgt], {"hops": 2, "container": conts[tgt]}


def build(stratum: str, packs: int, seed: int) -> list[dict]:
    rows = []
    for i in range(packs):
        sid = f"P2-{stratum[:3].upper()}{i:04d}"
        for variant in VARIANTS:
            rng = random.Random(h(seed, stratum, sid, variant))
            lines, q, ref, extra = _pack(rng, stratum, sid, variant)
            body = lines + rng.sample(FILLER, 2)
            rng.shuffle(body)                       # EVERY line shuffled, including the relation
            head = (f"This is a consolidated inventory note from {rng.choice(SITES)}, "
                    f"transcribed in the order the entries were found.")
            ctx = " ".join([head] + body)

            # ---- assertions, not assumptions ------------------------------------------
            if ctx.count(ref) < 1:
                raise SystemExit(f"{sid}/{variant}: reference absent from context")
            if SENTINEL.lower() in (ctx + q).lower():
                raise SystemExit(f"{sid}/{variant}: sentinel leaked into item content")
            if stratum == "temporal_predecessor":
                # the whole point: the TEXTUAL predecessor must not be the TEMPORAL one, or
                # the stratum measures reading order rather than ordering
                idx = [k for k, ln in enumerate(body) if extra["target"] in ln]
                ans_idx = [k for k, ln in enumerate(body) if ref in ln]
                if idx and ans_idx and ans_idx[0] == idx[0] - 1:
                    SKIPPED.append(f"{sid}/{variant}")
                    continue
            rows.append({
                "id": f"{sid}-{variant[:3].upper()}", "scenario_id": sid,
                "stratum": stratum, "variant": variant, "partition": "PROBE",
                "task_type": stratum, "answerable": True,
                "prompt": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": f"CONTEXT\n{ctx}\n\nQUESTION\n{q}"}],
                "reference": ref, "reference_normalised": ref.lower(),
                "context": ctx, "question": q, "extra": extra,
                "context_words": len(ctx.split()),
                "context_sha256": h(ctx),
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--packs-per-stratum", type=int, default=44)
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite: {a.out}")

    rows = []
    for s in STRATA:
        rows += build(s, a.packs_per_stratum, a.seed)
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise SystemExit("duplicate contexts across the probe")

    fl = {s: gate_floors([r for r in rows if r["stratum"] == s]) for s in STRATA}
    overall = gate_floors(rows)

    # ---- the build REFUSES; it does not report ------------------------------------------
    bad = {s: v["binding_floor_rate"] for s, v in fl.items()
           if v["binding_floor_rate"] > MAX_FLOOR}
    if bad:
        for s, v in fl.items():
            print(f"  {s:<22} floor {v['binding_floor_rate']:.4f} "
                  f"via {v['binding_predictor']} ({v['binding_class']})")
        raise SystemExit(f"REFUSING to write: strata above the {MAX_FLOOR} bound: {bad}")

    a.out.mkdir(parents=True)
    (a.out / "PRES2_PROBE.jsonl").write_text(
        "".join(canon(r) + "\n" for r in rows), encoding="utf-8")
    man = {
        "schema": 1, "record": "PRES_002_BUILD", "instrument": "PRES-002",
        "class": "MECHANISM SCOUT -- openly spendable, no seal, no read budget",
        "purpose": ("a SECOND preservation instrument on skills the RRP-015 corpus does not "
                    "exercise, to test whether direction sparing generalises beyond "
                    "alias_resolution"),
        "why_the_existing_endpoints_could_not_do_this": {
            "arithmetic_150": "WRONG CLASS -- rises with dose (Supa 47 -> 71), so it is "
                              "partially acquired rather than damaged",
            "references_retained_47": "RIGHT CLASS but underpowered ~3x; the proportional "
                                      "equivalent of the alias effect is ~4 rows and exact "
                                      "McNemar on 4 discordant pairs floors at p = 0.125",
        },
        "no_person_names_or_role_phrases": True,
        "entity_partition": "FOURTH -- disjoint from RCI-001, RCI-002 and ALIAS-001",
        "strata": list(STRATA), "variants": list(VARIANTS),
        "packs_per_stratum": a.packs_per_stratum, "rows": len(rows),
        "rows_per_stratum": {s: sum(1 for r in rows if r["stratum"] == s) for s in STRATA},
        "rows_dropped_textual_equals_temporal": len(SKIPPED),
        "rows_dropped_ids": SKIPPED,
        "why_dropped": ("a row whose TEXTUAL predecessor is also its TEMPORAL predecessor is "
                        "answerable by reading order, so it would not test ordering"),
        "seed": a.seed,
        "max_floor_bound": MAX_FLOOR,
        "bound_chosen_before_floors_were_computed": True,
        "floors_per_stratum": fl, "floors_overall": overall,
        "build_refuses_above_bound": True,
        "quality_claim": False, "release_permitted": False,
        "what_this_cannot_settle": ("whether these skills are DAMAGED at all -- that is measured "
                                    "by the first read, and a stratum the delta does not damage "
                                    "is reported and dropped, as the arithmetic panel should "
                                    "have been"),
    }
    (a.out / "PRES2_BUILD.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

    print(f"{'stratum':<24}{'rows':>6}{'floor':>8}  predictor")
    for s in STRATA:
        v = fl[s]
        n = sum(1 for r in rows if r["stratum"] == s)
        print(f"{s:<24}{n:>6}{v['binding_floor_rate']:>8.4f}  "
              f"{v['binding_predictor']} ({v['binding_class']})")
    print(f"{'OVERALL':<24}{len(rows):>6}{overall['binding_floor_rate']:>8.4f}  "
          f"{overall['binding_predictor']} ({overall['binding_class']})")
    print(f"\nwrote {a.out}  ({len(rows)} rows, bound {MAX_FLOOR})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
