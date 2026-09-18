"""Score the CALIBRATION twin and decide whether RCI-001 is usable — without touching the seal.

An unscored instrument might sit at its floor or at its ceiling, and both are uninterpretable:
MRS spent a scout at the constant-answer baseline, and DMRF-SCOUT-001's long-context stratum
came back 32/32 on both cores. Neither was knowable in advance. But scoring the confirmation
bank spends it, so the calibration twin — same generator, same parameters, disjoint entity
partition — is scored instead and the sealed bank inherits the verdict.

THE BAND
--------
A usable confirmation instrument must be clear of BOTH ends:

  FLOOR    the exact two-sided binomial 95% lower bound must exceed the binding shortcut floor,
           or an answer-agnostic predictor is competitive and arm contrasts are uninterpretable
  CEILING  the score must leave detectable headroom. A confirmation instrument at 100% cannot
           confirm a regression, which is half of what confirmation is for. The registered
           ceiling bound is 90%: above that, a real 13-row long-context loss could not show.

Scoring mirrors the Long-Context Demonstration Suite's own rule, because the point is to
confirm results measured on that kind of instrument: first line only, normalised as
`' '.join(v.strip().lower().replace('.',' ').split())`, exact match. Abstention is the literal
sentinel, and is reported inside the pooled bank — correct abstention and false abstention on
answerable rows — never as its own stratum, because scored alone unconditional abstention is
perfect.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SENTINEL = "UNKNOWN"
CEILING_RATE = 0.90


def norm(v: str) -> str:
    return " ".join(str(v).strip().lower().replace(".", " ").split())


def binom_at_least(k, n, p):
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def cp_lower(k, n, alpha=0.05):
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--gen", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = {r["id"]: r for r in (json.loads(l) for l in
            a.bank.read_text(encoding="utf-8").splitlines() if l.strip())}
    if any(r["partition"] == "SEALED" for r in rows.values()):
        raise SystemExit("REFUSING: this scorer must never be pointed at the SEALED bank.")
    build = json.loads(a.build.read_text(encoding="utf-8"))
    floor = build["floors"]["CALIBRATION"]["binding_floor_rate"]
    floor_pred = build["floors"]["CALIBRATION"]["binding_predictor"]

    systems = {}
    for gd in a.gen:
        gm = json.loads((gd / "GENERATE.json").read_text(encoding="utf-8"))
        preds = {json.loads(l)["id"]: json.loads(l)
                 for l in (gd / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()}
        detail = []
        for iid, row in rows.items():
            raw = (preds.get(iid, {}) or {}).get("prediction", "") or ""
            first = raw.strip().splitlines()[0] if raw.strip() else ""
            detail.append({
                "id": iid, "task_type": row["task_type"], "variant": row["variant"],
                "answerable": row["answerable"],
                "correct": norm(first) == norm(row["reference"]),
                "abstained": first.strip() == SENTINEL,
                "present": iid in preds,
            })
        n = len(detail)
        k = sum(d["correct"] for d in detail)
        lb = cp_lower(k, n)
        ans = [d for d in detail if d["answerable"]]
        un = [d for d in detail if not d["answerable"]]
        # per-pack all-four-correct: the Demo suite's anti-luck control
        packs = {}
        for iid, row in rows.items():
            packs.setdefault(row["scenario_id"], []).append(iid)
        by_id = {d["id"]: d for d in detail}
        all4 = sum(1 for sid, mem in packs.items() if all(by_id[i]["correct"] for i in mem))
        systems[gm["label"]] = {
            "model": gm["model"], "dtype": gm["dtype"],
            "transformers": gm["transformers"],
            "rows_skipped_context_overflow": gm["rows_skipped_context_overflow"],
            "n": n, "correct": k, "rate": round(k / n, 4), "ci_lower_95": round(lb, 4),
            "clears_floor": bool(lb > floor),
            "below_ceiling": bool(k / n < CEILING_RATE),
            "answerable_correct": sum(d["correct"] for d in ans), "answerable_n": len(ans),
            "correct_abstention": sum(1 for d in un if d["abstained"]),
            "unanswerable_n": len(un),
            "false_abstention_on_answerable": sum(1 for d in ans if d["abstained"]),
            "packs_all_four_correct": all4, "packs": len(packs),
            "by_task": {t: {"n": sum(1 for d in detail if d["task_type"] == t),
                            "correct": sum(d["correct"] for d in detail if d["task_type"] == t)}
                        for t in sorted({d["task_type"] for d in detail})},
            "by_variant": {v: {"n": sum(1 for d in detail if d["variant"] == v),
                               "correct": sum(d["correct"] for d in detail if d["variant"] == v)}
                           for v in sorted({d["variant"] for d in detail})},
            "per_item": detail,
        }

    clears = [s for s, v in systems.items() if v["clears_floor"]]
    below = [s for s, v in systems.items() if v["below_ceiling"]]
    if clears and len(below) == len(systems):
        verdict = "USABLE"
    elif not clears:
        verdict = "NOT_USABLE_AT_FLOOR"
    else:
        verdict = "NOT_USABLE_AT_CEILING"

    rep = {
        "schema": 1, "record": "RCI_CALIBRATION_REPORT", "instrument": "RCI-001",
        "bank_scored": "CALIBRATION (the twin). The SEALED bank was NOT touched.",
        "binding_floor_rate": floor, "binding_predictor": floor_pred,
        "ceiling_rate": CEILING_RATE,
        "ceiling_rationale": ("A confirmation instrument at or near 100% cannot confirm a "
                              "regression, which is half of what confirmation is for. At 90% a "
                              "13-row long-context loss on 512 rows would still be visible."),
        "verdict": verdict,
        "systems_clearing_floor": clears,
        "what_the_verdict_means": (
            "USABLE: at least one system's exact 95% lower bound exceeds the shortcut floor AND "
            "no system is above the ceiling, so the sealed twin -- same generator, same "
            "parameters, disjoint entities -- can be expected to discriminate. This is inherited "
            "by the sealed bank; it is NOT a measurement of the sealed bank."),
        "systems": systems,
        "seal_untouched": True,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": verdict, "floor": floor, "predictor": floor_pred, "ceiling": CEILING_RATE,
        "systems": {s: {"score": f"{v['correct']}/{v['n']}", "rate": v["rate"],
                        "ci_lo": v["ci_lower_95"], "clears_floor": v["clears_floor"],
                        "below_ceiling": v["below_ceiling"],
                        "correct_abstention": f"{v['correct_abstention']}/{v['unanswerable_n']}",
                        "false_abstention": v["false_abstention_on_answerable"],
                        "packs_all4": f"{v['packs_all_four_correct']}/{v['packs']}",
                        "by_variant": {k2: f"{x['correct']}/{x['n']}"
                                       for k2, x in v["by_variant"].items()}}
                    for s, v in systems.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
