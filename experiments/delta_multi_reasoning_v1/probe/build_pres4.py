"""PRES-004 — ALIAS-001's generator with the answer's TYPE crossed in. Supersedes PRES-003.

    python probe/build_pres4.py --out <records>/pres004-20260919

**MECHANISM SCOUT.** No seal, no read budget.

## Two calibration failures got us here, and both are recorded rather than quietly fixed

**PRES-002** cleared its floors at 0.21-0.33 and the source then scored **below floor on all
three strata, zero on two**. Its bridge was a *paraphrase* where ALIAS-001's is *verbatim*.

**PRES-003** fixed the bridge and failed from the other side. On the frozen source:

| scoring convention | `object_binding` | `place_binding` | verdict |
|---|---:|---:|---|
| strict | 85/180 = 0.472 | **16/180 = 0.089** | BELOW_FLOOR |
| **article-tolerant** | **168/180 = 0.933** | **174/180 = 0.967** | **ABOVE_CEILING** |

**241 of 360 rows — 67% — turned on a leading definite article.** The source answers
*"Dye House"* where the reference reads *"the Dye House"*. `convention_sensitivity()` has existed
in this tree since RCI-001 recorded 52 of 384 rows hinging on exactly this, and the AGENDA's
advice is *"prefer a primary endpoint that is convention-FREE"*. I did not run it at build time.

And with the convention fixed the bank is **too easy** — no headroom to show damage, which is the
ceiling failure DMRF-SCOUT-001 already hit at 32/32.

**So the gate has two sides and this bank failed one each time:** too hard for the source, then
too easy. Floors alone see neither.

## What PRES-004 does differently

1. **Copies ALIAS-001's salience profile**, not just its sentence shape. The answer is named
   **once**; each distractor is named **three times**, in neutral activity lines — ALIAS-001's
   `low` condition, where the frozen source scores **0.633**. PRES-003 named every candidate once
   and landed at 0.95.
2. **Scoring is article-tolerant and the references are article-free where natural.** Places are
   proper-noun-shaped (*Cellar Nine*, *Pennant Quay*) so no article is grammatical in the first
   place. Convention cannot bite from either side.
3. **`person_binding` is carried as a within-run calibration stratum.** It is ALIAS-001's task,
   rebuilt here — so if it does **not** land near 0.633 the bank is mis-built and that is visible
   *inside the same run* rather than inferred across instruments. The peer session asked for the
   person contrast to be within-run and this is also how it validates itself.
4. **`convention_sensitivity()` runs at BUILD time** and the build **REFUSES** if more than 10%
   of rows turn on a convention — measured against the references, so it catches an
   article-bearing reference pool before any GPU is spent.

## The question, unchanged and now askable

> **Is the alias regression about PERSONS, or about role→filler binding in general?**

Three strata differing *only* in the answer's type, same construction, same salience, same run.
`person_binding` doubles as the known-positive scale the peer suggested reporting alongside.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
from floors import floors as gate_floors  # noqa: E402

SENTINEL = "UNKNOWN"
SYSTEM = ("You are answering a closed-context reading test. Use only the supplied context. "
          "Reply with the shortest exact answer and no explanation. If the context does not "
          f"determine the answer, reply {SENTINEL}.")
_ART = re.compile(r"^(the|a|an)\s+", re.I)

#: FIFTH entity partition. Person names are disjoint from RCI-001/002 and ALIAS-001.
PEOPLE = ["Aurelio Banning", "Sunniva Okoro", "Caspian Delacroix-Hale", "Mirembe Sandoval",
          "Leocadio Thistlewood", "Annika Oyelowo", "Perpetua Vanterpool", "Rurik Maharaj"]
#: Article-free by construction -- no leading "the" is grammatical on any of these.
ITEMS = ["brass astrolabe", "walnut tally-board", "slate counting-frame", "copper alembic",
         "linen sail-hook", "iron sounding-line", "glass spirit-level", "horn inkwell"]
PLACES = ["Cellar Nine", "Pennant Quay", "Study Alcove Six", "Marchmont Stair",
          "Kestrel Landing", "Ordnance Bay", "Fennimore Court", "Saltgrass Reach"]
SITES = ["Haverlock Wharf", "the Tolbooth Yard", "Bellwether House"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
STRATA = ("person_binding", "object_binding", "place_binding")
VARIANTS = ("base", "paraphrase", "reorder")
MAX_FLOOR = 0.45
MAX_CONVENTION_SHARE = 0.10

#: ALIAS-001's `low` condition: the answer is named ONCE, each distractor THREE times.
K_ANSWER, K_DISTRACTOR = 1, 3
BRIDGE = {"person_binding": "the provisioning steward",
          "object_binding": "the item bearing the customs stamp",
          "place_binding": "the store used for that week's transfers"}
ACTIVITY = {
    "person_binding": ["{x} signed the attendance book on {d}.",
                       "{x} was recorded as present for the {d} stocktake.",
                       "{x} countersigned the {d} duty sheet."],
    "object_binding": ["{x} was entered in the tally book on {d}.",
                       "{x} appears in the {d} condition report.",
                       "{x} was re-weighed during the {d} audit."],
    "place_binding": ["{x} was swept and aired on {d}.",
                      "{x} appears in the {d} access log.",
                      "{x} was inspected for damp on {d}."],
}
FILLER = [
    "The account was compiled from duty books and carries no editorial comment.",
    "Several pages bear marginal notes in a second hand, reproduced where legible.",
    "Weather for the period was unremarkable and is noted only where it affected access.",
    "Page numbering restarts partway through and the original sequence is not recoverable.",
]


def h(*p) -> str:
    return hashlib.sha256("|".join(str(x) for x in p).encode("utf-8")).hexdigest()


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def build(stratum: str, packs: int, seed: int) -> list[dict]:
    pool = {"person_binding": PEOPLE, "object_binding": ITEMS, "place_binding": PLACES}[stratum]
    bridge, acts = BRIDGE[stratum], ACTIVITY[stratum]
    rows = []
    for i in range(packs):
        sid = f"P4-{stratum[:3].upper()}{i:04d}"
        for variant in VARIANTS:
            r = random.Random(h(seed, stratum, sid, variant))
            cands = r.sample(pool, 4)
            answer, distractors = cands[0], cands[1:]
            other_item = r.choice([x for x in ITEMS if x != answer])
            other_place = r.choice([x for x in PLACES if x != answer])

            # THE BINDING, stated exactly once, bridge VERBATIM in both sentences
            binding = (f"The {bridge} that week was {answer}, which is not named again in this "
                       f"account.")
            if stratum == "place_binding":
                action = f"It was {bridge} that received {other_item} on {r.choice(DAYS)}."
                q = (f"Which store received {other_item}?" if variant != "paraphrase"
                     else f"Name the store that took in {other_item}.")
            elif stratum == "object_binding":
                action = f"It was {bridge} that was carried to {other_place}, unaccompanied."
                q = (f"Which item was carried to {other_place}?" if variant != "paraphrase"
                     else f"Name the item taken to {other_place}.")
            else:
                action = (f"It was {bridge} who carried {other_item} to {other_place}, "
                          f"unaccompanied.")
                q = (f"Who personally carried {other_item} to {other_place}?"
                     if variant != "paraphrase"
                     else f"By name, who took {other_item} to {other_place} alone?")

            lines = [binding, action]
            for d in distractors:                      # each distractor named K_DISTRACTOR times
                for _ in range(K_DISTRACTOR):
                    lines.append(r.choice(acts).format(x=d, d=r.choice(DAYS)))

            counts = {c: sum(ln.count(c) for ln in lines) for c in cands}
            if counts[answer] != K_ANSWER or any(counts[d] != K_DISTRACTOR for d in distractors):
                raise SystemExit(f"{sid}/{variant}: salience not as specified: {counts}")

            body = lines + r.sample(FILLER, 2)
            r.shuffle(body)
            head = (f"This is a consolidated account of the audit at {r.choice(SITES)}, "
                    f"covering six consecutive days, transcribed in the order the entries "
                    f"were found.")
            ctx = " ".join([head] + body)
            if ctx.count(bridge) != 2:
                raise SystemExit(f"{sid}/{variant}: bridge must appear exactly twice verbatim")
            if SENTINEL.lower() in (ctx + q).lower():
                raise SystemExit(f"{sid}/{variant}: sentinel leaked")
            if _ART.match(answer):
                raise SystemExit(f"{sid}: reference {answer!r} carries a leading article")

            rows.append({
                "id": f"{sid}-{variant[:3].upper()}", "scenario_id": sid,
                "stratum": stratum, "variant": variant, "partition": "PROBE",
                "task_type": stratum, "answerable": True,
                "bridge": bridge, "answer_mentions": K_ANSWER,
                "distractor_mentions": K_DISTRACTOR,
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
    ap.add_argument("--packs-per-stratum", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260919)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite: {a.out}")

    rows = []
    for s in STRATA:
        rows += build(s, a.packs_per_stratum, a.seed)
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise SystemExit("duplicate contexts")

    # ---- convention check at BUILD time, against the references themselves ----------------
    art = sum(1 for r in rows if _ART.match(str(r["reference"])))
    share = art / len(rows)
    if share > MAX_CONVENTION_SHARE:
        raise SystemExit(f"REFUSING: {art}/{len(rows)} references carry a leading article "
                         f"({share:.3f} > {MAX_CONVENTION_SHARE}) -- PRES-003 lost 241/360 rows "
                         f"to exactly this")

    fl = {s: gate_floors([r for r in rows if r["stratum"] == s]) for s in STRATA}
    overall = gate_floors(rows)
    bad = {s: v["binding_floor_rate"] for s, v in fl.items()
           if v["binding_floor_rate"] > MAX_FLOOR}
    if bad:
        for s, v in fl.items():
            print(f"  {s:<18} floor {v['binding_floor_rate']:.4f} via {v['binding_predictor']}")
        raise SystemExit(f"REFUSING: strata above {MAX_FLOOR}: {bad}")

    a.out.mkdir(parents=True)
    (a.out / "PRES4_PROBE.jsonl").write_text(
        "".join(canon(r) + "\n" for r in rows), encoding="utf-8")
    (a.out / "PRES4_BUILD.json").write_text(json.dumps({
        "schema": 1, "record": "PRES_004_BUILD", "instrument": "PRES-004",
        "class": "MECHANISM SCOUT -- openly spendable, no seal, no read budget",
        "supersedes": ["PRES-002 (below gate: bridge was a paraphrase)",
                       "PRES-003 (above ceiling once the article convention was fixed; "
                       "241/360 rows turned on a leading definite article)"],
        "question": "is the alias regression about PERSONS, or about role->filler binding?",
        "salience": {"answer_mentions": K_ANSWER, "distractor_mentions": K_DISTRACTOR,
                     "copied_from": "ALIAS-001 `low` condition, where the frozen source "
                                    "scores 0.633"},
        "within_run_calibration": ("person_binding rebuilds ALIAS-001's task, so a score far "
                                   "from ~0.633 means the bank is mis-built and says so inside "
                                   "the same run"),
        "bridges_verbatim": BRIDGE,
        "references_article_free": True,
        "references_with_leading_article": art,
        "scoring_convention": "article-tolerant; references are article-free by construction",
        "strata": list(STRATA), "variants": list(VARIANTS),
        "packs_per_stratum": a.packs_per_stratum, "rows": len(rows),
        "rows_per_stratum": {s: sum(1 for r in rows if r["stratum"] == s) for s in STRATA},
        "seed": a.seed, "max_floor_bound": MAX_FLOOR,
        "max_convention_share": MAX_CONVENTION_SHARE,
        "floors_per_stratum": fl, "floors_overall": overall,
        "quality_claim": False, "release_permitted": False,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{'stratum':<18}{'rows':>6}{'floor':>8}  predictor")
    for s in STRATA:
        v = fl[s]
        print(f"{s:<18}{sum(1 for r in rows if r['stratum']==s):>6}{v['binding_floor_rate']:>8.4f}"
              f"  {v['binding_predictor']} ({v['binding_class']})")
    print(f"{'OVERALL':<18}{len(rows):>6}{overall['binding_floor_rate']:>8.4f}")
    print(f"\nreferences carrying a leading article: {art}/{len(rows)}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
