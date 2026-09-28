"""CONFIRM-001 — generate T_BANK_V3, content-disjoint by construction.

Registered in PREREG.md (committed c3a6f8a0) before this file produced any row.

Reuses the validated oracles and the v2 recorded-marginal sampler from FRESH-DUAL-SKILL-001, and
fixes Erratum 2 there: a row is rejected if its OPERATIVE key (answer-determining fields only) or
its (before-state, answer) pair matches any historical payload, or any row of T_BANK / T_BANK_V2,
or any row already emitted into this bank. The collision count is re-checked after generation and
must be zero on both keys.
"""
from __future__ import annotations

import hashlib, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FDS = HERE.parent / "fresh_dual_skill_instrument_v1"
sys.path.insert(0, str(FDS))
from oracles import ORACLES, dumps  # noqa: E402
from generate import SYSTEM, harvest_exclusions, content_keys, operative_key  # noqa: E402
from generate_v2 import load_marginals, GENS  # noqa: E402

SEED = 20260924
PER_FAMILY = 96
FAMILIES = ("ledger_replay", "prerequisite_plan", "register_machine")


def bank_payloads(path: Path) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {dumps(json.loads(r["prompt"][1]["content"])) for r in rows}


def main():
    out = HERE / "T_BANK_V3.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}: the read budget is ONE")

    hist = harvest_exclusions()
    own = bank_payloads(FDS / "T_BANK.json") | bank_payloads(FDS / "T_BANK_V2.json")
    excl_full = hist | own
    excl_op, excl_io = content_keys(excl_full)
    print(f"exclusion: {len(hist):,} historical + {len(own)} own-bank payloads -> "
          f"{len(excl_op):,} operative keys, {len(excl_io):,} (before,answer) pairs", flush=True)

    m = load_marginals()
    rng = random.Random(SEED)
    rows, seen_full, seen_op, seen_io = [], set(), set(), set()
    rejected = {"full": 0, "operative": 0, "before_answer": 0, "internal": 0, "trivial": 0}
    for fam in FAMILIES:
        made = guard = 0
        while made < PER_FAMILY:
            guard += 1
            if guard > PER_FAMILY * 2000:
                raise RuntimeError(f"could not fill {fam} content-disjointly")
            u = GENS[fam](rng, m)
            text = dumps(u)
            gold = ORACLES[fam](u)
            op = (fam, operative_key(fam, u))
            io = (fam, dumps(u["before"]), gold)
            if text in excl_full:
                rejected["full"] += 1; continue
            if op in excl_op:
                rejected["operative"] += 1; continue
            if io in excl_io:
                rejected["before_answer"] += 1; continue
            if text in seen_full or op in seen_op or io in seen_io:
                rejected["internal"] += 1; continue
            unchanged = gold == dumps(u["before"])
            if unchanged and rng.random() < 0.75:
                rejected["trivial"] += 1; continue
            seen_full.add(text); seen_op.add(op); seen_io.add(io)
            rid = f"CF1-{fam}-{hashlib.sha256(text.encode()).hexdigest()[:16]}"
            rows.append({"id": rid, "parent_id": rid, "family": fam, "domain": "transition",
                         "prompt": [{"role": "system", "content": SYSTEM[fam]},
                                    {"role": "user", "content": text}],
                         "reference": gold, "copy_baseline": dumps(u["before"]),
                         "answer_equals_input": unchanged,
                         "gold_verified_by": "oracles.py validated 384/384 on process-transition-probe-001"})
            made += 1
        print(f"  {fam}: {made}", flush=True)

    # independent re-check: recompute both keys from the emitted rows, against the exclusion sets
    post_op = post_io = post_full = 0
    for r in rows:
        u = json.loads(r["prompt"][1]["content"]); fam = r["family"]
        post_full += dumps(u) in excl_full
        post_op += (fam, operative_key(fam, u)) in excl_op
        post_io += (fam, dumps(u["before"]), r["reference"]) in excl_io
    if post_full or post_op or post_io:
        raise SystemExit(f"COLLISION CHECK FAILED: full={post_full} operative={post_op} before+answer={post_io}")

    doc = {"schema": 1, "instrument": "CONFIRM-001", "split": "T", "version": 3,
           "registered_in": "PREREG.md, commit c3a6f8a0, before this bank existed",
           "seed": SEED, "per_family": PER_FAMILY, "n": len(rows),
           "generator": "FRESH-DUAL-SKILL-001 generate_v2 recorded-marginal sampler",
           "exclusion": {"historical_payloads": len(hist), "own_bank_payloads": len(own),
                         "operative_keys": len(excl_op), "before_answer_pairs": len(excl_io)},
           "collisions_after_generation": {"full_payload": 0, "operative": 0, "before_answer": 0},
           "rejected_during_generation": rejected,
           "copy_baseline_rows": sum(r["answer_equals_input"] for r in rows),
           "scoring_primary": "exact match after stripping whitespace and AT MOST ONE trailing '.'",
           "scoring_secondary": "strict exact match",
           "read_budget": "ONE",
           "rows": rows}
    payload = (json.dumps(doc, indent=2) + "\n").encode("utf-8")      # LF, byte mode
    with open(out, "xb") as f:
        f.write(payload)
    print(f"wrote {out.name}: n={len(rows)} sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"rejected: {rejected}")
    print(f"copy-baseline (shortcut floor) rows: {doc['copy_baseline_rows']}/{len(rows)}")


if __name__ == "__main__":
    main()
