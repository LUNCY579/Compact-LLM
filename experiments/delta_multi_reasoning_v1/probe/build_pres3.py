"""PRES-003 — ALIAS-001's exact shape, with non-person answers. Supersedes PRES-002.

    python probe/build_pres3.py --out <records>/pres003-20260919

**MECHANISM SCOUT.** No seal, no read budget.

## Why PRES-002 was withdrawn and what changed

PRES-002 failed calibration: the frozen source was **below floor on all three strata and scored
zero on two**. Diagnosed from its predictions — the failures returned the **wrong entity**, not
the right one in the wrong shape, so it was not an answer-form artifact. `container_nesting`
returned the container, `object_coreference` echoed *"wax-sealed item"* unresolved,
`temporal_predecessor` returned a weekday.

**The error is located by ALIAS-001, where Supa scores 0.750 on the same two-hop shape.** The
structures are identical; the bridge is not:

  * alias bridges **verbatim** — *"It was the provisions marshal who carried X"* → *"The
    provisions marshal that week was <name>"*.
  * PRES-002 bridged by **paraphrase** — *"the wax-sealed item"* → *"bears a wax seal"*.

**I had optimised for the floors and forgotten the gate.** Floors ask whether the bank is
gameable; the gate asks whether the source can do the task. They are independent.

## What this bank is

**ALIAS-001's construction, copied deliberately, with the answer's TYPE changed.** Same binding
sentence, same "not named again" clause, same unaccompanied-action sentence, same shuffle, same
filler. The bridge phrase is **verbatim in both sentences**, as alias's is.

| stratum | bridge phrase (verbatim in both sentences) | answer type |
|---|---|---|
| `object_binding` | *the item bearing the customs stamp* | an **object** |
| `place_binding` | *the store used for that week's transfers* | a **place** |

**Still zero person names and zero role phrases**, so the property that makes `alias_resolution`
an *unexercised* skill is held constant, and the corpus audit (Addendum 1 of
`PRES002_PREREG.md`) applies unchanged.

## The question this asks, which is better than PRES-002's

Because these are structural twins of alias differing only in the answer's type, a contrast is
interpretable in a way three unrelated tasks never were:

> **Is the alias regression about PERSONS, or about role→filler binding in general?**

  * **damaged here too** → the delta weakens a binding step, and `alias_resolution` was one
    instance of it. The direction test then has a properly-classed second endpoint.
  * **not damaged here** → the regression is specific to person-valued bindings, which would be a
    sharp and surprising constraint on every account of it, including capacity.

Either answer is worth having, which is the property PRES-002 lacked.

## Discipline carried over

Build **REFUSES** above a 0.45 floor bound fixed before any floor was computed. Mention counts
are **asserted** equal across candidates per pack — PRES-002's first draft reintroduced the
ALIAS-001 salience leak as a side effect of adding filler and was caught at floor 1.000. Every
line is shuffled, including the binding and the action.
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

ITEMS = ["brass astrolabe", "walnut tally-board", "slate counting-frame", "copper alembic",
         "linen sail-hook", "iron sounding-line", "glass spirit-level", "horn inkwell",
         "leather drum-case", "zinc plumb-bob", "oak cradle-block", "tin lantern-hood"]
PLACES = ["the North Arcade", "the Dye House", "Cellar Nine", "the Rope Walk",
          "the Gauging Room", "the Slip Shed", "the Lime Yard", "the Cooperage",
          "the Sail Loft", "the Winch House", "the Palm Store", "the Bell Passage"]
SITES = ["Haverlock Wharf", "the Tolbooth Yard", "Pennant Quay"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
STRATA = ("object_binding", "place_binding")
VARIANTS = ("base", "paraphrase", "reorder")
MAX_FLOOR = 0.45

#: The bridge is VERBATIM in both the binding sentence and the action sentence. That is the one
#: thing PRES-002 got wrong and ALIAS-001 got right.
BRIDGE = {"object_binding": "the item bearing the customs stamp",
          "place_binding": "the store used for that week's transfers"}

FILLER = [
    "The account was compiled from duty books and carries no editorial comment.",
    "Several pages bear marginal notes in a second hand, reproduced where legible.",
    "Weather for the period was unremarkable and is noted only where it affected access.",
    "Page numbering restarts partway through and the original sequence is not recoverable.",
]


def h(*parts) -> str:
    return hashlib.sha256("|".join(str(x) for x in parts).encode("utf-8")).hexdigest()


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def build(stratum: str, packs: int, seed: int) -> list[dict]:
    rows = []
    bridge = BRIDGE[stratum]
    for i in range(packs):
        sid = f"P3-{stratum[:3].upper()}{i:04d}"
        for variant in VARIANTS:
            r = random.Random(h(seed, stratum, sid, variant))
            if stratum == "object_binding":
                cands = r.sample(ITEMS, 4)
                answer, other = cands[0], r.choice(PLACES)
                binding = (f"The {bridge} that week was the {answer}, which is not named again "
                           f"in this account.")
                action = f"It was {bridge} that was carried to {other}, unaccompanied."
                q = (f"Which item was carried to {other}?" if variant != "paraphrase"
                     else f"Name the item taken to {other}.")
                neutral = [f"The {c} was entered in the tally book on {r.choice(DAYS)}."
                           for c in cands[1:]]
            else:
                cands = r.sample(PLACES, 4)
                answer, other = cands[0], r.choice(ITEMS)
                binding = (f"The {bridge} that week was {answer}, which is not named again in "
                           f"this account.")
                action = f"It was {bridge} that received the {other} on {r.choice(DAYS)}."
                q = (f"Which store received the {other}?" if variant != "paraphrase"
                     else f"Name the store that took in the {other}.")
                neutral = [f"{c} was swept and aired on {r.choice(DAYS)}." for c in cands[1:]]

            lines = [binding, action] + neutral
            # every candidate named EXACTLY once -- asserted, not assumed
            counts = {c: sum(ln.count(c) for ln in lines) for c in cands}
            if set(counts.values()) != {1}:
                raise SystemExit(f"{sid}/{variant}: mention counts not all 1: {counts}")

            body = lines + r.sample(FILLER, 2)
            r.shuffle(body)                     # EVERY line, including binding and action
            head = (f"This is a consolidated account of the audit at {r.choice(SITES)}, "
                    f"covering six consecutive days, transcribed in the order the entries "
                    f"were found.")
            ctx = " ".join([head] + body)

            if ctx.count(bridge) != 2:
                raise SystemExit(f"{sid}/{variant}: bridge must appear exactly twice verbatim")
            if SENTINEL.lower() in (ctx + q).lower():
                raise SystemExit(f"{sid}/{variant}: sentinel leaked into item content")
            rows.append({
                "id": f"{sid}-{variant[:3].upper()}", "scenario_id": sid,
                "stratum": stratum, "variant": variant, "partition": "PROBE",
                "task_type": stratum, "answerable": True,
                "bridge": bridge, "bridge_is_verbatim": True,
                "prompt": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": f"CONTEXT\n{ctx}\n\nQUESTION\n{q}"}],
                "reference": answer, "reference_normalised": answer.lower(),
                "context": ctx, "question": q,
                "context_words": len(ctx.split()), "context_sha256": h(ctx),
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--packs-per-stratum", type=int, default=60)
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
    bad = {s: v["binding_floor_rate"] for s, v in fl.items()
           if v["binding_floor_rate"] > MAX_FLOOR}
    if bad:
        for s, v in fl.items():
            print(f"  {s:<18} floor {v['binding_floor_rate']:.4f} via "
                  f"{v['binding_predictor']} ({v['binding_class']})")
        raise SystemExit(f"REFUSING to write: strata above {MAX_FLOOR}: {bad}")

    a.out.mkdir(parents=True)
    (a.out / "PRES3_PROBE.jsonl").write_text(
        "".join(canon(r) + "\n" for r in rows), encoding="utf-8")
    (a.out / "PRES3_BUILD.json").write_text(json.dumps({
        "schema": 1, "record": "PRES_003_BUILD", "instrument": "PRES-003",
        "class": "MECHANISM SCOUT -- openly spendable, no seal, no read budget",
        "supersedes": "PRES-002, which FAILED calibration (source below floor on all three "
                      "strata, zero on two) because its bridge was a paraphrase where "
                      "ALIAS-001's is verbatim",
        "question": ("is the alias regression about PERSONS, or about role->filler binding in "
                     "general?"),
        "construction": "ALIAS-001's exact shape with the answer's TYPE changed",
        "bridges_verbatim": BRIDGE,
        "no_person_names_or_role_phrases": True,
        "strata": list(STRATA), "variants": list(VARIANTS),
        "packs_per_stratum": a.packs_per_stratum, "rows": len(rows),
        "rows_per_stratum": {s: sum(1 for r in rows if r["stratum"] == s) for s in STRATA},
        "seed": a.seed, "max_floor_bound": MAX_FLOOR,
        "floors_per_stratum": fl, "floors_overall": overall,
        "mention_counts_asserted_equal": True,
        "quality_claim": False, "release_permitted": False,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{'stratum':<18}{'rows':>6}{'floor':>8}  predictor")
    for s in STRATA:
        v = fl[s]
        print(f"{s:<18}{sum(1 for r in rows if r['stratum']==s):>6}"
              f"{v['binding_floor_rate']:>8.4f}  {v['binding_predictor']} ({v['binding_class']})")
    print(f"{'OVERALL':<18}{len(rows):>6}{overall['binding_floor_rate']:>8.4f}  "
          f"{overall['binding_predictor']}")
    print(f"\nwrote {a.out}  ({len(rows)} rows, bound {MAX_FLOOR})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
