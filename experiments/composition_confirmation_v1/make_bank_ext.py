"""CONFIRM-001 Amendment 1 — extend T_BANK_V3 (288 rows) to T_BANK_V3X (864 rows), before any read.

Why: a pre-read power check (AMENDMENT_1.md) found that C3, the interference margin, passes with
probability ~0.52 at n=288 even if the fresh bank behaves exactly like the development bank. A C3
failure at that power would say nothing. At n=864 the same check gives ~0.90.

Construction: T_BANK_V3's 288 rows are kept VERBATIM and in order, followed by 192 new rows per
family from the same recorded-marginal sampler with a new seed. A new row is rejected on exactly
the keys make_bank.py uses (full payload, operative key, (before-state, answer) pair), against the
same exclusion set PLUS every T_BANK_V3 row PLUS every row already emitted. The collision count is
recomputed after generation and must be zero on all three keys.

No arm has been scored on T_BANK_V3 or on this bank when this file runs. Create-only output.
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

V3 = HERE / "T_BANK_V3.json"
V3_SHA = "20597c2247eff91b4197d18e63fc02012673b06fb80ffb694d07fa0adfc7507a"
SEED = 20260925
PER_FAMILY = 192
FAMILIES = ("ledger_replay", "prerequisite_plan", "register_machine")


def bank_payloads(path: Path) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {dumps(json.loads(r["prompt"][1]["content"])) for r in rows}


def main():
    out = HERE / "T_BANK_V3X.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}: the read budget is ONE")
    raw = V3.read_bytes()
    if hashlib.sha256(raw).hexdigest() != V3_SHA:
        raise SystemExit("T_BANK_V3.json does not match its committed sha256")
    v3 = json.loads(raw)

    hist = harvest_exclusions()
    own = bank_payloads(FDS / "T_BANK.json") | bank_payloads(FDS / "T_BANK_V2.json")
    v3_payloads = bank_payloads(V3)
    excl_full = hist | own | v3_payloads
    excl_op, excl_io = content_keys(excl_full)
    print(f"exclusion: {len(hist):,} historical + {len(own)} own-bank + {len(v3_payloads)} V3 payloads -> "
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
            rid = f"CF1X-{fam}-{hashlib.sha256(text.encode()).hexdigest()[:16]}"
            rows.append({"id": rid, "parent_id": rid, "family": fam, "domain": "transition",
                         "prompt": [{"role": "system", "content": SYSTEM[fam]},
                                    {"role": "user", "content": text}],
                         "reference": gold, "copy_baseline": dumps(u["before"]),
                         "answer_equals_input": unchanged,
                         "gold_verified_by": "oracles.py validated 384/384 on process-transition-probe-001"})
            made += 1
        print(f"  {fam}: {made} (attempts {guard})", flush=True)

    # independent re-check against the exclusion sets, which include every V3 row
    post_op = post_io = post_full = 0
    for r in rows:
        u = json.loads(r["prompt"][1]["content"]); fam = r["family"]
        post_full += dumps(u) in excl_full
        post_op += (fam, operative_key(fam, u)) in excl_op
        post_io += (fam, dumps(u["before"]), r["reference"]) in excl_io
    if post_full or post_op or post_io:
        raise SystemExit(f"COLLISION CHECK FAILED: full={post_full} operative={post_op} before+answer={post_io}")
    # positive control: the same check must fire on V3's own rows
    ctl = sum((r["family"], operative_key(r["family"], json.loads(r["prompt"][1]["content"]))) in excl_op
              for r in v3["rows"])
    if ctl != len(v3["rows"]):
        raise SystemExit(f"POSITIVE CONTROL FAILED: only {ctl}/{len(v3['rows'])} V3 rows detected as excluded")
    ids = [r["id"] for r in v3["rows"]] + [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate row id")

    all_rows = v3["rows"] + rows
    doc = {"schema": 1, "instrument": "CONFIRM-001", "split": "T", "version": "3X",
           "registered_in": "PREREG.md (c3a6f8a0) + AMENDMENT_1.md, both before any arm was scored",
           "composition": {"T_BANK_V3_rows_verbatim": len(v3["rows"]), "T_BANK_V3_sha256": V3_SHA,
                           "T_BANK_V3_seed": v3["seed"], "extension_rows": len(rows),
                           "extension_seed": SEED, "extension_per_family": PER_FAMILY},
           "n": len(all_rows),
           "generator": "FRESH-DUAL-SKILL-001 generate_v2 recorded-marginal sampler",
           "exclusion": {"historical_payloads": len(hist), "own_bank_payloads": len(own),
                         "v3_payloads": len(v3_payloads),
                         "operative_keys": len(excl_op), "before_answer_pairs": len(excl_io)},
           "collisions_after_generation": {"full_payload": 0, "operative": 0, "before_answer": 0},
           "positive_control": f"{ctl}/{len(v3['rows'])} V3 rows detected by the operative-key check",
           "rejected_during_generation": rejected,
           "copy_baseline_rows": sum(r["answer_equals_input"] for r in all_rows),
           "scoring_primary": "exact match after stripping whitespace and AT MOST ONE trailing '.'",
           "scoring_secondary": "strict exact match",
           "read_budget": "ONE",
           "rows": all_rows}
    payload = (json.dumps(doc, indent=2) + "\n").encode("utf-8")      # LF, byte mode
    with open(out, "xb") as f:
        f.write(payload)
    print(f"wrote {out.name}: n={len(all_rows)} sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"rejected: {rejected}")
    print(f"copy-baseline rows: {doc['copy_baseline_rows']}/{len(all_rows)} "
          f"(V3 part {v3['copy_baseline_rows']}, extension {sum(r['answer_equals_input'] for r in rows)})")


if __name__ == "__main__":
    main()
