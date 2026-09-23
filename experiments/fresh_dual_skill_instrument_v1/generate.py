"""FRESH-DUAL-SKILL-001 — generate an uncontaminated transition instrument.

The three transition families are procedural and oracle-verifiable, so a genuinely fresh bank
can be GENERATED rather than hand-authored. That is what makes this admissible where the peer's
hand-authored probe was not: gold comes from an oracle validated 384/384 against the recorded
bank, and every emitted payload is checked against an exclusion set harvested from every
historical file on this machine that contains a transition payload — including all training
corpora.

Construction parameters are matched to the recorded bank's observed ranges. Matched construction
is NOT matched difficulty, so the fresh bank's endpoint scores are measured, never assumed.

Read-only w.r.t. everything outside this directory.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracles import ORACLES, dumps  # noqa: E402

RECORDS = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
SYSTEM = {
    "ledger_replay": (
        "Apply one event to the supplied account balances. If status is not posted, leave all "
        "balances unchanged. add: add amount to account. scale: multiply account by amount. "
        "transfer: subtract amount from account and add it to other, simultaneously. Return only "
        "the complete resulting balances as a JSON object."
    ),
    "prerequisite_plan": (
        "The state is an 8-bit flag mask. A supplied applicable action requires all pre_on bits "
        "set and all pre_off bits clear. Apply delete bits first, then add bits. Return only the "
        "resulting state as a JSON integer. Names and costs do not change the state."
    ),
    "register_machine": (
        "Apply one instruction to the supplied registers. Registers are integers modulo 97. swap "
        "exchanges register and other simultaneously. add adds value to register. mix adds (2 "
        "times other plus value) to register. conditional adds value if other is greater than "
        "threshold, otherwise subtracts value. Reduce updated numeric registers modulo 97; "
        "preserve other registers. Return only the complete resulting registers as a JSON object."
    ),
}


def payload_key(user_json: str) -> str:
    """Order-insensitive semantic key for a payload, so key order cannot hide a duplicate."""
    return dumps(json.loads(user_json))


def harvest_exclusions() -> set[str]:
    """Every transition payload that appears anywhere in the records tree."""
    seen: set[str] = set()
    files = [p for p in RECORDS.rglob("*.jsonl")]
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if 'before\\":' not in raw and '"before":' not in raw:
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            stack = [obj]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)
                elif isinstance(cur, str) and cur.startswith('{"before"'):
                    try:
                        seen.add(payload_key(cur))
                    except json.JSONDecodeError:
                        pass
    return seen


def gen_ledger(rng):
    accts = ["A", "B", "C", "D"]
    before = {a: rng.randint(-120, 120) for a in accts}
    kind = rng.choice(["add", "scale", "transfer"])
    acct = rng.choice(accts)
    other = rng.choice([a for a in accts if a != acct])
    amount = rng.choice([rng.randint(-60, 60), rng.randint(-3, 3), -1, 1, 0, 2, -2])
    status = "posted" if rng.random() < 0.85 else rng.choice(["pending", "void", "draft"])
    return {"before": before,
            "event": {"account": acct, "amount": amount, "kind": kind, "other": other,
                      "status": status, "tick": rng.randint(1, 99)}}


def gen_prereq(rng):
    before = rng.randint(0, 255)
    applicable = rng.random() < 0.8
    if applicable:                                  # force the guard to hold
        pre_on = before & rng.randint(0, 255)
        pre_off = (~before) & rng.randint(0, 255) & 0xFF
    else:                                           # force it to fail
        missing = [b for b in range(8) if not (before >> b) & 1]
        pre_on = (before & rng.randint(0, 255)) | (1 << rng.choice(missing) if missing else 0)
        pre_off = (~before) & rng.randint(0, 255) & 0xFF
    return {"before": before,
            "event": {"add": rng.randint(0, 255) & rng.randint(0, 255),
                      "cost": rng.randint(1, 9),
                      "delete": rng.randint(0, 255) & rng.randint(0, 255),
                      "kind": "action", "name": f"A{rng.randint(1, 40)}",
                      "pre_off": pre_off, "pre_on": pre_on}}


def gen_register(rng):
    regs = ["X", "Y", "Z"]
    before = {r: rng.randint(0, 96) for r in regs}
    kind = rng.choice(["swap", "add", "mix", "conditional"])
    r = rng.choice(regs)
    other = rng.choice([x for x in regs if x != r])
    return {"before": before,
            "event": {"kind": kind, "other": other, "register": r,
                      "threshold": rng.randint(0, 96), "tick": rng.randint(1, 99),
                      "value": rng.randint(0, 96)}}


GENS = {"ledger_replay": gen_ledger, "prerequisite_plan": gen_prereq, "register_machine": gen_register}


def main():
    import random
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-family", type=int, default=64)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    print("harvesting exclusion set from the records tree ...", flush=True)
    excl = harvest_exclusions()
    print(f"  excluded payloads: {len(excl):,}", flush=True)

    rng = random.Random(a.seed)
    rows, emitted = [], set()
    trivial = 0
    for fam in ("ledger_replay", "prerequisite_plan", "register_machine"):
        made = 0
        guard = 0
        while made < a.per_family:
            guard += 1
            if guard > a.per_family * 400:
                raise RuntimeError(f"could not fill {fam}")
            u = GENS[fam](rng)
            text = dumps(u)
            if text in excl or text in emitted:
                continue
            gold = ORACLES[fam](u)
            # reject rows where the answer equals the input state: they are solvable by copying
            if fam == "prerequisite_plan":
                unchanged = gold == dumps(u["before"])
            else:
                unchanged = gold == dumps(u["before"])
            if unchanged and rng.random() < 0.75:
                trivial += 1
                continue
            emitted.add(text)
            rid = f"FDS1-{fam}-{hashlib.sha256(text.encode()).hexdigest()[:16]}"
            rows.append({
                "id": rid, "parent_id": rid, "family": fam, "domain": "transition",
                "prompt": [{"role": "system", "content": SYSTEM[fam]},
                           {"role": "user", "content": text}],
                "reference": gold,
                "copy_baseline": dumps(u["before"]),
                "answer_equals_input": unchanged,
                "gold_verified_by": "oracles.py validated 384/384 on process-transition-probe-001",
            })
            made += 1
        print(f"  {fam}: {made} rows", flush=True)

    out = {
        "schema": 1, "instrument": "FRESH-DUAL-SKILL-001", "split": "T",
        "seed": a.seed, "per_family": a.per_family, "n": len(rows),
        "exclusion_set_size": len(excl),
        "exclusion_sources": "every *.jsonl under rineya-dataset-retention-records containing a transition payload",
        "scoring": "exact string match on the stripped generation against `reference`",
        "trivial_unchanged_rows_dropped": trivial,
        "copy_baseline_rows": sum(r["answer_equals_input"] for r in rows),
        "rows": rows,
    }
    a.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(a.output.read_bytes()).hexdigest()
    print(f"wrote {a.output}  n={len(rows)}  sha256={digest}")
    print(f"copy-baseline (answer == input) rows kept: {out['copy_baseline_rows']}/{len(rows)}")


if __name__ == "__main__":
    main()
