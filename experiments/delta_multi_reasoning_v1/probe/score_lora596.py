"""Independent scoring pass over a LORA596 arm — written BEFORE the arm's endpoints were read.

    python probe/score_lora596.py --label int001_lora596b \
        --evidence <retention-pilot>/experiments/architecture_benchmark_001/evidence/retention-design-v1 \
        --receipt <records>/lora596-20260919/arm-lora-r16-merged/MATERIALIZATION.json \
        --out <records>/lora596-20260919/INDEPENDENT_SCORE_<label>.json

**Why this file exists and why its timestamp matters.** The paired session registered LORA596's
bands, ran its merge-integrity check *prospectively*, and asked for an independent scoring pass.
A scoring script written **after** reading the other session's table is not an independent check —
it is a reconstruction, and nobody (including its author) can tell the difference afterwards.

> **An apparatus check produced in response to a surprising number is indistinguishable from
> looking for a reason to discard the number.**

Same structure as registering a band before the arm exists, and as
`registration-window-closes-at-first-arm`. This module was committed before any arm-2 endpoint
was read by this session.

## The three gates, run BEFORE any endpoint is reported

This programme has been burnt by comparing numbers produced under different conditions, so the
comparability preconditions are **hard refusals**, not warnings:

  1. **Bank identity — `views_sha256`.** The single field that distinguishes two banks which are
     otherwise identical in row count, family split and implementation digest. A near-miss this
     session had two 384-row / 3-family / same-digest banks (`7dd53f2a…` and `45674251…`) one
     sentence from being compared. **Nothing else identifies a bank.**
  2. **Runner identity — `implementation_digest`.** Qualified values reproduce only under
     `a329eb7bb06f…`; a different digest means the number is not readable against the table.
  3. **Did the probe read the MERGED weights?** `hf_candidate_identity.weight_sha256` must equal
     the merge receipt's `out_weight_sha256`. A retention probe that silently scored the base
     model would report perfect retention, which is exactly the shape of the result most likely
     to be believed.

Gate 3 is the one that matters most here. The registered outcome table has a cell —
*retention >= 43 at 1.79x the control's magnitude* — that would **contradict
damage-tracks-magnitude**, one of the better-replicated claims in this programme. A result that
contradicts a well-replicated claim earns an apparatus check before it earns a headline, and the
cheapest apparatus failure that would produce it is scoring the wrong weights.

## Refusal discipline

Like every scorer here, this one **refuses to write on empty or unreadable input**, and never
overwrites an existing record after a failed read. `probe/test_refuse_empty.py` records why: a
lookup failure must never be able to destroy the thing it failed to look up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Registered comparability constants. Changing either invalidates every comparison below.
SELECT_VIEWS_SHA = "7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3"
IMPL_DIGEST = "a329eb7bb06fc003a9220ea41713e721fba575c478008240749c425774b2b58e"

# Reference points, all on SELECT_VIEWS_SHA, published before this arm was scored.
SCALAR_LADDER = {"a025": 145, "a050": 219, "a075": 234, "a100": 235}
LADDER_MAX = 235          # P3': the ladder saturates; below this a scalar matches it more cheaply
CONTROL_RETAINED = 40     # FP32 blocks8-27 control, the magnitude-story reference
CEILING_RETAINED = 47
MAX_SOURCE_LOST = 4       # P4


def _load(p: Path) -> dict | None:
    """Return None rather than raising — a missing input must produce a refusal, not a crash."""
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and d else None


def gate_comparability(proc: dict, ret: dict, receipt: dict) -> list[dict]:
    """The three hard preconditions. Every one is a REFUSAL, not a caveat."""
    ident = ret.get("hf_candidate_identity", {}) or {}
    merged = receipt.get("out_weight_sha256") or receipt.get("weight_sha256")
    probed = ident.get("weight_sha256")
    return [
        {"gate": "bank_identity", "field": "views_sha256",
         "expected": SELECT_VIEWS_SHA, "observed": proc.get("views_sha256"),
         "ok": proc.get("views_sha256") == SELECT_VIEWS_SHA,
         "why": "the only field that identifies a bank; two 384-row 3-family banks with the "
                "same implementation digest exist and are NOT interchangeable"},
        {"gate": "runner_identity", "field": "implementation_digest",
         "expected": IMPL_DIGEST, "observed": proc.get("implementation_digest"),
         "ok": proc.get("implementation_digest") == IMPL_DIGEST,
         "why": "qualified values reproduce only under this digest"},
        {"gate": "probe_read_merged_weights", "field": "hf_candidate_identity.weight_sha256",
         "expected": merged, "observed": probed,
         "ok": bool(merged) and merged == probed,
         "why": "a probe that silently scored the BASE model reports perfect retention -- the "
                "cheapest apparatus failure that would manufacture the headline result"},
    ]


def read_endpoints(proc: dict, ret: dict, lc: dict | None) -> dict:
    agg = proc.get("aggregate", {}) or {}
    pol = (ret.get("by_bare_policy", {}) or {}).get(ret.get("registered_bare_policy", "permissive"), {})
    fams = proc.get("per_family", {}) or {}
    lost = pol.get("lost_total")
    src = pol.get("source_correct_total")
    return {
        "acquisition": agg.get("canonical_correct"),
        "n": agg.get("n"),
        "floors_cleared": sum(1 for v in fams.values() if v.get("clears_floor")),
        "families": len(fams),
        "per_family": {k: v.get("canonical_correct") for k, v in fams.items()},
        "source_correct_total": src,
        "source_answers_lost": lost,
        "retained": (src - lost) if (src is not None and lost is not None) else None,
        "arm_correct_outside_source": pol.get("arm_correct_outside_source"),
        "retention_status": pol.get("status"),
        "long_context": (lc or {}).get("all256_hits"),
    }


def verdicts(e: dict) -> list[dict]:
    """Score against the bands as registered. A band that cannot be evaluated says so."""
    out = []
    acq, ret_, lost = e["acquisition"], e["retained"], e["source_answers_lost"]

    out.append({
        "band": "P3' acquisition >= 235 (scalar ladder maximum at ANY magnitude)",
        "observed": acq,
        "verdict": None if acq is None else ("HELD" if acq >= LADDER_MAX else "FAILED"),
        "why": "the ladder saturates (a075 234, a100 235). Below its maximum, some scalar arm "
               "matches this acquisition more cheaply and the parameterization buys nothing. "
               "Replaces the matched-norm bar, which is UNEXECUTABLE at alpha-equivalent 1.794 "
               "-- outside the ladder's range, and extrapolating past an endpoint is how the "
               "cue-differential bound went wrong.",
    })
    out.append({
        "band": f"P4 source answers lost <= {MAX_SOURCE_LOST}",
        "observed": lost,
        "verdict": None if lost is None else ("HELD" if lost <= MAX_SOURCE_LOST else "FAILED"),
    })

    # The registered three-outcome table. Reported as a READING, never as a test.
    reading, note = "INDETERMINATE", ""
    if acq is not None and ret_ is not None:
        if ret_ >= 43 and acq >= LADDER_MAX:
            reading = "STRONG_RANK_EFFECT"
            note = ("more magnitude AND less forgetting, which no point on the scalar ladder does "
                    "anywhere on its range. This CONTRADICTS damage-tracks-magnitude, so check "
                    "the apparatus before the physics -- gate 3 above is the first place to look.")
        elif ret_ <= CONTROL_RETAINED:
            reading = "PURE_MAGNITUDE"
            note = ("retention at or below the control's 40 at 1.79x its magnitude is exactly "
                    "what damage-tracks-magnitude predicts. Rank is not the lever.")
        elif ret_ >= CEILING_RETAINED and acq < 150:
            reading = "UNDER_CONVERGED"
            note = ("ceiling retention with collapsed acquisition is the small-update story, not "
                    "a rank effect -- the same confound that made arm 1 uninterpretable.")
        else:
            reading = "BETWEEN_CELLS"
            note = "falls between the registered cells; report the numbers, claim nothing."
    out.append({"band": "registered three-outcome reading", "observed":
                {"acquisition": acq, "retained": ret_}, "verdict": reading, "why": note})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    proc = _load(a.evidence / f"PROCESS_{a.label}.json")
    ret = _load(a.evidence / f"RETENTION_PROBE_{a.label}.json")
    lc = _load(a.evidence / f"LONGCTX_{a.label}.json")
    receipt = _load(a.receipt)

    missing = [n for n, v in (("PROCESS", proc), ("RETENTION_PROBE", ret),
                              ("MATERIALIZATION receipt", receipt)) if v is None]
    if missing:
        print(f"REFUSED: missing or empty {', '.join(missing)} for label {a.label!r}.")
        print("Nothing written. A lookup failure must never destroy the thing it failed to read.")
        return 2

    gates = gate_comparability(proc, ret, receipt)
    e = read_endpoints(proc, ret, lc)
    blocked = [g for g in gates if not g["ok"]]

    result = {
        "record": "LORA596_INDEPENDENT_SCORE",
        "label": a.label,
        "scorer_written_before_any_endpoint_of_this_arm_was_read": True,
        "comparability_gates": gates,
        "endpoints": None if blocked else e,
        "verdicts": None if blocked else verdicts(e),
        "status": "BLOCKED_ON_COMPARABILITY" if blocked else "SCORED",
    }

    print(f"LORA596 independent pass — {a.label}\n")
    for g in gates:
        print(f"  [{'ok ' if g['ok'] else 'FAIL'}] {g['gate']:<28} {str(g['observed'])[:24]}")
    if blocked:
        print("\nREFUSED to report endpoints: a comparability gate failed. The numbers may be "
              "fine; they are not READABLE against the reference table until this is resolved.")
        a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 1

    print(f"\n  acquisition {e['acquisition']}/{e['n']}   floors {e['floors_cleared']}/{e['families']}"
          f"   retained {e['retained']}/{e['source_correct_total']}"
          f"   lost {e['source_answers_lost']}   long ctx {e['long_context']}")
    for v in verdicts(e):
        print(f"\n  {v['band']}\n    observed {v['observed']} -> {v['verdict']}")
        if v.get("why"):
            print(f"    {v['why']}")
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
