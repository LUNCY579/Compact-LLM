"""FRESH-DUAL-SKILL-001 v2 — regenerate with recorded-matched input marginals.

v1 failed build gate G2 (donor 63/192 = 32.8%, required >= 40%). Diagnosis, by family:

    family              recorded tsparse   v1 fresh   gap
    ledger_replay              41%           44%      +3pp   matched
    register_machine           65%           38%     -27pp
    prerequisite_plan          75%           17%     -58pp

The v1 generator was off-distribution on two families. Measured from the recorded bank's
INPUTS only (no model output involved):

  prerequisite_plan  recorded `add` is exactly ONE bit in 128/128 rows; `delete` is 0 bits (73)
                     or 1 bit (55); pre_on 0-2 bits; pre_off 0-1 bits; `before` has 1-5 bits set;
                     and ALL 128 rows are applicable. v1 used multi-bit add/delete, denser
                     `before`, and made 20% of rows inapplicable - a case the recorded bank
                     never contains.
  register_machine   recorded `value` spans -8..11, median 1. v1 sampled 0..96, ~50x larger
                     under mod 97, and never negative.

v2 resamples each family's event parameters from the recorded EMPIRICAL MARGINALS. The repair
criterion is the input distribution, which is measurable without running any model; no arm score
entered the choice of parameters. Freshness is still enforced by the same exclusion set, so no
recorded payload can be reproduced.

This is attempt 2. That is stated in RESULTS.md and is not hidden.
"""
from __future__ import annotations

import argparse, hashlib, json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracles import ORACLES, dumps  # noqa: E402
from generate import SYSTEM, harvest_exclusions, payload_key  # noqa: E402

RECORDED = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/process-transition-probe-001/CANDIDATE_VIEWS.jsonl")


def load_marginals():
    rows = [json.loads(l) for l in RECORDED.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {"ledger_replay": [], "prerequisite_plan": [], "register_machine": []}
    for r in rows:
        by[r["family"]].append(json.loads(r["prompt"][1]["content"]))
    bc = lambda x: bin(x).count("1")
    m = {
        "ledger": {
            "kinds": [p["event"]["kind"] for p in by["ledger_replay"]],
            "status": [p["event"]["status"] for p in by["ledger_replay"]],
            "amounts": [p["event"]["amount"] for p in by["ledger_replay"]],
            "balances": [v for p in by["ledger_replay"] for v in p["before"].values()],
        },
        "prereq": {
            "before_bits": [bc(p["before"]) for p in by["prerequisite_plan"]],
            "delete_bits": [bc(p["event"]["delete"]) for p in by["prerequisite_plan"]],
            "pre_on_bits": [bc(p["event"]["pre_on"]) for p in by["prerequisite_plan"]],
            "pre_off_bits": [bc(p["event"]["pre_off"]) for p in by["prerequisite_plan"]],
            "costs": [p["event"]["cost"] for p in by["prerequisite_plan"]],
        },
        "register": {
            "kinds": [p["event"]["kind"] for p in by["register_machine"]],
            "values": [p["event"]["value"] for p in by["register_machine"]],
            "thresholds": [p["event"]["threshold"] for p in by["register_machine"]],
            "befores": [v for p in by["register_machine"] for v in p["before"].values()],
        },
    }
    return m


def pick_bits(rng, n_bits, pool):
    """Choose n_bits distinct bit positions from `pool` (list of positions)."""
    if n_bits <= 0 or not pool:
        return 0
    n_bits = min(n_bits, len(pool))
    mask = 0
    for b in rng.sample(pool, n_bits):
        mask |= 1 << b
    return mask


def gen_ledger(rng, m):
    accts = ["A", "B", "C", "D"]
    before = {a: rng.choice(m["ledger"]["balances"]) for a in accts}
    kind = rng.choice(m["ledger"]["kinds"])
    acct = rng.choice(accts)
    other = rng.choice([a for a in accts if a != acct])
    return {"before": before,
            "event": {"account": acct, "amount": rng.choice(m["ledger"]["amounts"]), "kind": kind,
                      "other": other, "status": rng.choice(m["ledger"]["status"]),
                      "tick": rng.randint(1, 99)}}


def gen_prereq(rng, m):
    nb = rng.choice(m["prereq"]["before_bits"])
    before = pick_bits(rng, nb, list(range(8)))
    on_pos = [b for b in range(8) if (before >> b) & 1]
    off_pos = [b for b in range(8) if not (before >> b) & 1]
    pre_on = pick_bits(rng, rng.choice(m["prereq"]["pre_on_bits"]), on_pos)     # always satisfied
    pre_off = pick_bits(rng, rng.choice(m["prereq"]["pre_off_bits"]), off_pos)  # always satisfied
    add = 1 << rng.randrange(8)                                                # exactly one bit
    delete = pick_bits(rng, rng.choice(m["prereq"]["delete_bits"]), list(range(8)))
    return {"before": before,
            "event": {"add": add, "cost": rng.choice(m["prereq"]["costs"]), "delete": delete,
                      "kind": "action", "name": f"A{rng.randint(1, 40)}",
                      "pre_off": pre_off, "pre_on": pre_on}}


def gen_register(rng, m):
    regs = ["X", "Y", "Z"]
    before = {r: rng.choice(m["register"]["befores"]) for r in regs}
    r = rng.choice(regs)
    other = rng.choice([x for x in regs if x != r])
    return {"before": before,
            "event": {"kind": rng.choice(m["register"]["kinds"]), "other": other, "register": r,
                      "threshold": rng.choice(m["register"]["thresholds"]),
                      "tick": rng.randint(1, 99), "value": rng.choice(m["register"]["values"])}}


GENS = {"ledger_replay": gen_ledger, "prerequisite_plan": gen_prereq, "register_machine": gen_register}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-family", type=int, default=64)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")

    m = load_marginals()
    print("recorded marginals loaded", flush=True)
    print("harvesting exclusion set ...", flush=True)
    excl = harvest_exclusions()
    print(f"  excluded payloads: {len(excl):,}", flush=True)

    rng = random.Random(a.seed)
    rows, emitted, trivial = [], set(), 0
    for fam in ("ledger_replay", "prerequisite_plan", "register_machine"):
        made, guard = 0, 0
        while made < a.per_family:
            guard += 1
            if guard > a.per_family * 800:
                raise RuntimeError(f"could not fill {fam}")
            u = GENS[fam](rng, m)
            text = dumps(u)
            if text in excl or text in emitted:
                continue
            gold = ORACLES[fam](u)
            unchanged = gold == dumps(u["before"])
            if unchanged and rng.random() < 0.75:
                trivial += 1
                continue
            emitted.add(text)
            rid = f"FDS2-{fam}-{hashlib.sha256(text.encode()).hexdigest()[:16]}"
            rows.append({"id": rid, "parent_id": rid, "family": fam, "domain": "transition",
                         "prompt": [{"role": "system", "content": SYSTEM[fam]},
                                    {"role": "user", "content": text}],
                         "reference": gold, "copy_baseline": dumps(u["before"]),
                         "answer_equals_input": unchanged,
                         "gold_verified_by": "oracles.py validated 384/384 on process-transition-probe-001"})
            made += 1
        print(f"  {fam}: {made}", flush=True)

    out = {"schema": 1, "instrument": "FRESH-DUAL-SKILL-001", "version": 2, "split": "T",
           "attempt": 2,
           "attempt_1_outcome": "G2 FAILED (donor 63/192 = 32.8%, required >= 40%)",
           "repair_criterion": ("event parameters resampled from the recorded bank's empirical "
                                "input marginals; no model output entered the choice"),
           "seed": a.seed, "per_family": a.per_family, "n": len(rows),
           "exclusion_set_size": len(excl),
           "scoring": "exact string match on the stripped generation against `reference`",
           "trivial_unchanged_rows_dropped": trivial,
           "copy_baseline_rows": sum(r["answer_equals_input"] for r in rows),
           "rows": rows}
    a.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.output} n={len(rows)} sha256={hashlib.sha256(a.output.read_bytes()).hexdigest()}")
    print(f"copy-baseline rows: {out['copy_baseline_rows']}/{len(rows)}")


if __name__ == "__main__":
    main()
