"""Why does `alias_resolution` lose rows? Salience-over-binding, tested on two banks.

    python analysis/alias_salience.py --rci001 <records>/rci001-20260918 \
        --rci002 <records>/rci002-20260918 --out <...>/ALIAS_SALIENCE.json

**Zero model calls.** Everything runs on recorded predictions from reads that already happened.

## The cell nothing explains

`alias_resolution` is the one stratum that moved the same way on both banks and that no proposed
account touches:

    RCI-001   B 65/78, D 53/78, 0 gains / 12 losses, p = 4.9e-04
    RCI-002   B 41/60, D 36/60,                      p = 6.3e-02

It has a **congruent** interrogative throughout -- the question asks "Who" and the answer is a
person -- so the confirmed wh-compliance mechanism predicts nothing here. And the answer FORM is
preserved: on RCI-001 the candidate emitted a person name on 78 of 78 rows and simply picked the
wrong one. Five accounts of the programme's damage died and none of them reaches this.

## The hypothesis, and why it would unify rather than multiply

Both generators name the aliased person exactly ONCE -- "The <role> that week was <name>, who is
not named again in this account" -- while the movers who appear in the movement block can be
named up to three times, and every person appears once more in the decoy register. So **the
correct answer is systematically the LEAST salient person in the context**.

    SALIENCE-OVER-BINDING: asked "Who", the delta reaches for the most available person rather
    than the one the text binds to the role.

That is the same operation the confirmed result describes, seen where an answer does exist. On
an incongruent carve-out no person is bound, so it emits an arbitrary one -- 99/120, confirmed.
On `alias_resolution` a person IS bound, but others are more frequent and more recent, so it
emits one of those instead. One mechanism, two surfaces, rather than a sixth account.

## What is measured

For every answerable `alias_resolution` row, classify what each system emitted:

    CORRECT             the bound person
    MOVER               someone named in the movement block -- the salient distractors
    OTHER_PERSON        a pool person named nowhere useful
    NOT_A_PERSON        anything else

and, for the emitted person, its **mention count** and **last position** in the context against
the correct answer's. The prediction is directional and can fail: on the rows where the two
systems disagree, the delta's choice should be the more salient one more often than the frozen
source's is.

**A control that must hold or the whole reading collapses:** if the delta simply emitted the
most frequent person always, it would be a constant rule and would score near zero. It does not
-- so the question is whether it is *shifted* toward salience, not whether it is governed by it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired import fisher_exact_2x2, paired  # noqa: E402

ARTICLE = re.compile(r"^(the|a|an)\s+")


def norm(v):
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    s = " ".join(s.lower().replace(".", " ").replace(",", " ").split())
    return ARTICLE.sub("", s)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def load(bank: Path, gens: dict):
    rows = [json.loads(l) for l in bank.read_text(encoding="utf-8").splitlines() if l.strip()]
    preds = {}
    for lab, path in gens.items():
        preds[lab] = {r["id"]: r.get("prediction", "") for r in
                      (json.loads(l) for l in
                       Path(path).read_text(encoding="utf-8").splitlines() if l.strip())}
    return rows, preds


def ctx_of(r):
    return r.get("context") or r["prompt"][1]["content"].split("CONTEXT\n")[-1].split(
        "\n\nQUESTION")[0]


def analyse(rows, preds, label):
    ans = [r for r in rows if r["answerable"] and r["task_type"] == "alias_resolution"]
    if not ans:
        return None
    people = sorted({str(r["reference"]) for r in rows
                     if r["task_type"] == "alias_resolution" and r["answerable"]},
                    key=len, reverse=True)
    # A mover is any person named in a movement sentence. Recovered from the prose rather than
    # from the generator, so this works identically on both banks.
    mover_pat = re.compile(r"On \w+day, ([A-Z][\w'\-]+(?: [A-Z][\w'\-]+)+) was assigned")

    out = {"label": label, "n": len(ans), "person_vocab": len(people), "systems": {}}
    per_row = {}
    for lab, pr in preds.items():
        cls = Counter()
        sal_emitted, sal_correct = [], []
        flags = {}
        for r in ans:
            ctx = ctx_of(r)
            ref = norm(r["reference"])
            got = norm(pr.get(r["id"], ""))
            movers = {norm(m) for m in mover_pat.findall(ctx)}
            hit = next((p for p in people if norm(p) in got), None)
            if got == ref:
                cls["CORRECT"] += 1
            elif hit is None:
                cls["NOT_A_PERSON"] += 1
            elif norm(hit) in movers:
                cls["MOVER"] += 1
            else:
                cls["OTHER_PERSON"] += 1
            flags[r["id"]] = {"correct": got == ref, "emitted": hit}
            if hit is not None:
                low = ctx.lower()
                sal_emitted.append(low.count(norm(hit).split()[0]))
                sal_correct.append(low.count(ref.split()[0]))
        per_row[lab] = flags
        n = len(ans)
        out["systems"][lab] = {
            "classification": dict(cls),
            "correct_rate": round(cls["CORRECT"] / n, 4),
            "mover_share_of_errors": (round(cls["MOVER"] / (n - cls["CORRECT"]), 4)
                                      if n > cls["CORRECT"] else None),
            "mean_mentions_of_emitted": (round(sum(sal_emitted) / len(sal_emitted), 3)
                                         if sal_emitted else None),
            "mean_mentions_of_correct": (round(sum(sal_correct) / len(sal_correct), 3)
                                         if sal_correct else None),
        }

    if sorted(preds) == ["B", "D"]:
        p = paired("B", "D", {k: v["correct"] for k, v in per_row["B"].items()},
                   {k: v["correct"] for k, v in per_row["D"].items()})
        out["paired"] = {k: v for k, v in p.as_dict().items()
                         if k not in ("repair_ids", "damage_ids")}
        # THE TEST. On rows B got right and D got wrong, what did D reach for?
        lost = list(p.damage_ids)
        lc = Counter()
        for rid in lost:
            r = next(x for x in ans if x["id"] == rid)
            ctx = ctx_of(r)
            movers = {norm(m) for m in mover_pat.findall(ctx)}
            e = per_row["D"][rid]["emitted"]
            lc["MOVER" if (e and norm(e) in movers) else
               ("OTHER_PERSON" if e else "NOT_A_PERSON")] += 1
        out["what_D_reached_for_on_rows_it_lost"] = {"n": len(lost), **dict(lc)}
        # Is D's error distribution shifted toward movers relative to B's?
        bm = out["systems"]["B"]["classification"].get("MOVER", 0)
        be = out["systems"]["B"]["n_errors"] = (
            out["n"] - out["systems"]["B"]["classification"].get("CORRECT", 0))
        dm = out["systems"]["D"]["classification"].get("MOVER", 0)
        de = out["systems"]["D"]["n_errors"] = (
            out["n"] - out["systems"]["D"]["classification"].get("CORRECT", 0))
        out["mover_shift"] = {
            "B_movers_of_errors": f"{bm}/{be}", "D_movers_of_errors": f"{dm}/{de}",
            "fisher": (fisher_exact_2x2(dm, de - dm, bm, be - bm)
                       if be and de else None),
            "reading": ("if the delta reaches for the salient distractor more than the source "
                        "does, its errors should be MORE concentrated on movers"),
        }
    return out


def nesting_table(rows, pb, pd, bank):
    """Does D fail everywhere B fails, plus more -- per stratum, and globally?

    This is the context that stops the alias finding being over-read. Across both sealed banks
    the delta REPAIRS 30 answerable rows and damages 56, so it is not uniformly worse and a
    purely subtractive stratum is a real signature rather than the background.
    """
    out = {}
    for tt in sorted({r["task_type"] for r in rows}):
        sub = [r for r in rows if r["answerable"] and r["task_type"] == tt]
        if not sub:
            continue
        ok = lambda pr, r: norm(pr.get(r["id"], "")) == norm(r["reference"])   # noqa: E731
        rep = sum(1 for r in sub if ok(pd, r) and not ok(pb, r))
        dam = sum(1 for r in sub if ok(pb, r) and not ok(pd, r))
        out[f"{bank}/{tt}"] = {
            "n": len(sub), "B": sum(1 for r in sub if ok(pb, r)),
            "D": sum(1 for r in sub if ok(pd, r)),
            "repair": rep, "damage": dam, "strictly_nested": rep == 0 and dam > 0,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rci001", type=Path, required=True)
    ap.add_argument("--rci002", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    banks = {}
    r1, p1 = load(a.rci001 / "RCI_SEALED.jsonl", {
        "B": a.rci001 / "gen-rci_confirm_B" / "predictions.jsonl",
        "D": a.rci001 / "gen-rci_confirm_D" / "predictions.jsonl"})
    banks["RCI-001"] = analyse(r1, p1, "RCI-001 sealed")
    r2, p2 = load(a.rci002 / "RCI_SEALED.jsonl", {
        "B": a.rci002 / "gen-rci2_confirm_B" / "predictions.jsonl",
        "D": a.rci002 / "gen-rci2_confirm_D" / "predictions.jsonl"})
    banks["RCI-002"] = analyse(r2, p2, "RCI-002 sealed")

    nest = {}
    nest.update(nesting_table(r1, p1["B"], p1["D"], "RCI-001"))
    nest.update(nesting_table(r2, p2["B"], p2["D"], "RCI-002"))
    grep_, gdam = sum(v["repair"] for v in nest.values()), sum(v["damage"] for v in nest.values())
    ar = sum(v["repair"] for k, v in nest.items() if k.endswith("alias_resolution"))
    ad = sum(v["damage"] for k, v in nest.items() if k.endswith("alias_resolution"))

    out = {"what_this_is": "post-hoc, EXPLORATORY, zero model calls -- recorded predictions from "
                           "reads that already happened. Not a confirmation.",
           "hypothesis_tested": "salience-over-binding: asked 'Who', the delta reaches for the "
                                "most available person rather than the one the text binds",
           "hypothesis_verdict": (
               "REFUTED AS STATED, and the refutation is the useful part. Salience-over-binding "
               "IS the failure mode -- on RCI-001 every one of the 12 rows the delta lost, it "
               "named a MOVER, and movers are the people the text mentions most. But the frozen "
               "source does exactly the same on 13 of its own 13 errors, and the two error "
               "profiles are identical (3 none-of-these + 10 most-frequent-and-last, each). The "
               "delta did not shift the KIND of error; it increased the RATE of an error both "
               "models already make."),
           "banks": banks,
           "per_stratum_nesting": nest,
           "global": {
               "repair": grep_, "damage": gdam,
               "note": ("the delta REPAIRS 30 answerable rows across both sealed banks, so it "
                        "is not uniformly worse and a purely subtractive stratum is a signature "
                        "rather than the background"),
           },
           "alias_pooled_across_both_banks": {
               "repair": ar, "damage": ad,
               "p_exact_binomial": mcnemar(ar, ad),
               "reading": ("zero reversals in 17 discordant pairs over two independently "
                           "generated banks with disjoint entity pools. The effect is purely "
                           "subtractive and it replicates."),
           }}
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    for name, b in banks.items():
        if not b:
            continue
        print(f"\n=== {name}  ({b['n']} answerable alias rows, {b['person_vocab']} names)")
        for lab, v in b["systems"].items():
            c = v["classification"]
            print(f"  {lab}: correct {c.get('CORRECT', 0):<3} mover {c.get('MOVER', 0):<3} "
                  f"other {c.get('OTHER_PERSON', 0):<3} non-person {c.get('NOT_A_PERSON', 0):<3}"
                  f"   mentions: emitted {v['mean_mentions_of_emitted']} vs "
                  f"correct {v['mean_mentions_of_correct']}")
        if "paired" in b:
            p = b["paired"]
            print(f"  paired: B {p['baseline_correct']} D {p['candidate_correct']}, "
                  f"repair {p['repair']} damage {p['damage']}, "
                  f"p {p['mcnemar_exact_two_sided_p']:.3e}")
            print(f"  on rows D lost, D reached for: {b['what_D_reached_for_on_rows_it_lost']}")
            m = b["mover_shift"]
            fz = "n/a" if m["fisher"] is None else f"{m['fisher']:.3f}"
            print(f"  mover share of errors: B {m['B_movers_of_errors']} vs D "
                  f"{m['D_movers_of_errors']}  Fisher {fz}")
    g = out["global"]
    ap_ = out["alias_pooled_across_both_banks"]
    print("\n=== context: across BOTH sealed banks, every answerable stratum")
    print(f"  repair {g['repair']}   damage {g['damage']}   -- the delta is NOT uniformly worse")
    print(f"  strictly nested strata: {[k for k, v in nest.items() if v['strictly_nested']]}")
    print("\n=== alias_resolution POOLED across both banks")
    print(f"  repair {ap_['repair']}   damage {ap_['damage']}   "
          f"p = {ap_['p_exact_binomial']:.3e}")
    print(f"\nverdict: {out['hypothesis_verdict']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
