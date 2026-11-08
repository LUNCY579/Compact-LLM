"""Paired analysis of arm scores on the FRESH-DUAL-SKILL-001 transition bank.

Written before the v2 gate outcome was read, so the estimator and the comparisons are fixed
independently of the numbers they will be applied to.

Reports, for the registered decomposition:
    attenuation cost = SPARSE - AVG      (both donors at half strength)
    interference cost = SPARSE - ADD-100 (both donors at full strength)
each as a paired McNemar exact test over the 192 rows, with discordant counts shown, because a
net difference of k rows can hide very different discordance structure.

CORRECTED 2026-09-24: an earlier version of this docstring said it also reports the count of
rows where the generation equals the input state. It never did, and it has no period-tolerant
path. Both — plus the content-clean re-analysis and the gold+"." decomposition — are produced by
`reproduce_results.py`, which also has a `--check` mode asserting every documented number.
This file is kept unchanged in behaviour because it is the estimator registered before the v2
gate outcome was read.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    by_arm: dict[str, dict[str, bool]] = {}
    families: dict[str, str] = {}
    meta = {}
    for p in a.runs:
        d = json.loads(p.read_text(encoding="utf-8"))
        meta.setdefault("bank_sha256", d["bank_sha256"])
        if d["bank_sha256"] != meta["bank_sha256"]:
            raise SystemExit("runs come from different banks; refusing to pool")
        meta["bank_n"] = d["bank_n"]
        meta["shortcut_floor_rows"] = d.get("shortcut_floor_rows")
        for arm in d["arms"]:
            by_arm[arm["arm"]] = {r["id"]: r["correct"] for r in arm["predictions"]}
            for r in arm["predictions"]:
                families[r["id"]] = r["family"]

    ids = sorted(next(iter(by_arm.values())))
    for name, m in by_arm.items():
        if sorted(m) != ids:
            raise SystemExit(f"arm {name} has a different row set; refusing to compare")

    def pair(x: str, y: str):
        if x not in by_arm or y not in by_arm:
            return None
        b = sum(1 for i in ids if by_arm[x][i] and not by_arm[y][i])   # x right, y wrong
        c = sum(1 for i in ids if by_arm[y][i] and not by_arm[x][i])
        return {"reference": x, "candidate": y,
                "reference_correct": sum(by_arm[x].values()),
                "candidate_correct": sum(by_arm[y].values()),
                "net": sum(by_arm[y].values()) - sum(by_arm[x].values()),
                "lost_by_candidate": b, "gained_by_candidate": c,
                "discordant": b + c, "mcnemar_exact_p": mcnemar_exact(b, c)}

    out = {
        "instrument": "FRESH-DUAL-SKILL-001", **meta,
        "per_arm": {name: {
            "correct": sum(m.values()),
            "by_family": {f: sum(1 for i in ids if families[i] == f and m[i])
                          for f in sorted(set(families.values()))},
        } for name, m in sorted(by_arm.items())},
        "registered_decomposition": {
            "attenuation_cost_SPARSE_vs_AVG": pair("SPARSE", "AVG"),
            "interference_cost_SPARSE_vs_ADD-100": pair("SPARSE", "ADD-100"),
        },
        "other_pairs": {f"{x}_vs_{y}": pair(x, y) for x, y in
                        (("BASE", "SPARSE"), ("SPARSE", "ADD-S75"), ("SPARSE", "ADD-75"),
                         ("ADD-100", "ADD-S75"), ("OPALLO", "ADD-100"))
                        if x in by_arm and y in by_arm},
        "caveat": ("Raw p-values, no multiplicity adjustment. One skill, one instrument, "
                   "attempt 2 of 2. Not a quality, retention or release claim."),
    }
    a.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'arm':<18}{'correct':>9}  by family")
    for name, v in out["per_arm"].items():
        print(f"  {name:<16}{v['correct']:>7}/{meta['bank_n']}  {v['by_family']}")
    print()
    for label, r in out["registered_decomposition"].items():
        if r:
            print(f"  {label}")
            print(f"     {r['reference']}={r['reference_correct']} {r['candidate']}={r['candidate_correct']} "
                  f"net={r['net']:+d}  lost={r['lost_by_candidate']} gained={r['gained_by_candidate']} "
                  f"p={r['mcnemar_exact_p']:.4f}")


if __name__ == "__main__":
    main()
