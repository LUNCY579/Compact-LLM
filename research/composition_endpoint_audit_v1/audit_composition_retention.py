"""Can the n=47 retention endpoint resolve ANY composition arm? And does DIRECTION matter on it?

Reads only committed artifacts in the records tree. No model is loaded, no GPU, no build.

TWO QUESTIONS, kept separate because they license different things:

  Q1  At MATCHED TOTAL NORM, does the second component's DIRECTION change any endpoint?
      Answered by a paired row-level contrast, not by comparing summary counts.

  Q2  Could the retention endpoint have detected a composition cost AT ALL?
      Answered EXACTLY, with no model of the effect: an exact two-sided McNemar on n
      discordant rows has p = 2/2^n at its most extreme, so p < 0.05 REQUIRES n >= 6
      discordant rows. Compare that to the discordance the family actually produces.

DENOMINATOR. Retention is scored over the rows the SOURCE got right (`source_correct_total`),
never over the full reference set -- the naive denominator has flipped a conclusion here before.
`retained = source_correct_total - lost_total`, and the union of `per_class.lost_ids` is
asserted equal to `lost_total` on every file so a silently dropped class cannot pass.
"""
import json, math, os
from math import comb

R = os.environ.get("RECORDS", r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")

# (label, directory, materialization, description) -- the delta_T(s17) composition family.
FAM = [
    ("indep001_t100h000",   "indep001-20260921", "MATERIALIZATION_t100h000.json",    "control: delta_T only"),
    ("indep003_s17_h005",   "indep003-20260921", "MATERIALIZATION_h005.json",        "+0.05*delta_H    (different skill)"),
    ("indep003_s17_h010",   "indep003-20260921", "MATERIALIZATION_h010.json",        "+0.10*delta_H    (different skill)"),
    ("indep003_s17_h015",   "indep003-20260921", "MATERIALIZATION_h015.json",        "+0.15*delta_H    (different skill)"),
    ("indep005_s17_h020",   "indep005-20260921", "MATERIALIZATION_h020.json",        "+0.20*delta_H    (different skill)"),
    ("indep003_s17_h025",   "indep003-20260921", "MATERIALIZATION_h025.json",        "+0.25*delta_H    (different skill)"),
    ("indep005_same_a0554", "indep005-20260921", "MATERIALIZATION_same_a0554.json",  "+0.0554*dT(s23)  (SAME skill)"),
    ("indep005_same_a025",  "indep005-20260921", "MATERIALIZATION_same_a025.json",   "+0.25*dT(s23)    (SAME skill)"),
]
CTRL_NORM = 0.674029797822056


def lost_set(d, lab):
    p = json.load(open(f"{R}/{d}/evidence/RETENTION_PROBE_{lab}.json", encoding="utf-8"))["by_bare_policy"]["permissive"]
    s = set()
    for _, x in p["per_class"].items():
        s |= set(x.get("lost_ids", []))
    assert len(s) == p["lost_total"], f"{lab}: union of lost_ids {len(s)} != lost_total {p['lost_total']}"
    return s, p["source_correct_total"]


def probe_id(d, lab):
    return json.load(open(f"{R}/{d}/evidence/RETENTION_PROBE_{lab}.json", encoding="utf-8"))["probe_sha256"][:16]


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return p, min(1.0, 2 / 2 ** n)


def main():
    rows, ids = [], set()
    for lab, d, mp, desc in FAM:
        m = json.load(open(f"{R}/{d}/{mp}", encoding="utf-8"))
        s, den = lost_set(d, lab)
        rows.append(dict(lab=lab, desc=desc, norm=m["built_update_norm"], cos=m["delta_cosine"],
                         lost=s, den=den, ret=den - len(s)))
        ids.add(probe_id(d, lab))
    assert len(ids) == 1, f"more than one retention instrument in the family: {ids}"
    assert len({r['den'] for r in rows}) == 1, "denominators differ across arms"
    print(f"ONE retention instrument across {len(rows)} arms: probe_sha256 {ids.pop()}..., "
          f"denominator {rows[0]['den']}\n")

    print(f"{'arm':22s} {'second component':34s} {'cos':>7s} {'norm':>9s} {'vs ctrl':>8s} {'ret':>4s}")
    for r in rows:
        print(f"{r['lab']:22s} {r['desc']:34s} {r['cos']:7.4f} {r['norm']:9.6f} "
              f"{100*(r['norm']/CTRL_NORM-1):+7.2f}% {r['ret']:4d}")
    rs = [r["ret"] for r in rows]
    print(f"\n  retention spans {min(rs)}-{max(rs)} = {max(rs)-min(rs)} rows across the WHOLE family")

    print("\n=== Q2: COULD THIS ENDPOINT HAVE DETECTED ANYTHING? (exact, no effect model) ===")
    need = next(n for n in range(1, 60) if 2 / 2 ** n < 0.05)
    print(f"  exact two-sided McNemar: most extreme p on n discordant rows is 2/2^n")
    print(f"  => p < 0.05 REQUIRES at least {need} DISCORDANT ROWS, whatever the arms do\n")
    worst = 0
    print(f"  {'pair':46s} {'b':>3s} {'c':>3s} {'disc':>5s} {'p':>9s} {'min_p':>9s}")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            A, B = rows[i], rows[j]
            b, c = len(B["lost"] - A["lost"]), len(A["lost"] - B["lost"])
            p, mp = exact_mcnemar(b, c)
            worst = max(worst, b + c)
            if b + c >= 2:
                print(f"  {A['lab'][7:]+' vs '+B['lab'][7:]:46s} {b:3d} {c:3d} {b+c:5d} {p:9.4f} {mp:9.4f}")
    print(f"\n  MAXIMUM discordance anywhere in the family: {worst} rows. Minimum needed: {need}.")
    print(f"  => NO PAIR IN THIS FAMILY COULD REACH p<0.05. The endpoint is {need/max(worst,1):.1f}x "
          f"short of its own requirement.")

    print("\n=== and the norm law says the null is exactly the size it should be ===")
    print(f"  {'arm':22s} {'law-predicted cost':>19s} {'measured':>9s}")
    for r in rows[1:]:
        print(f"  {r['lab']:22s} {6.28*math.log(r['norm']/CTRL_NORM):19.3f} {rows[0]['ret']-r['ret']:9d}")
    print(f"\n  largest predicted cost in the family: "
          f"{max(6.28*math.log(r['norm']/CTRL_NORM) for r in rows):.3f} rows; endpoint quantisation 1 row")

    print("\n=== Q1: DIRECTION at matched total norm (the only such pair that exists) ===")
    A = next(r for r in rows if r["lab"] == "indep003_s17_h025")
    B = next(r for r in rows if r["lab"] == "indep005_same_a0554")
    C = rows[0]
    print(f"  total norm {A['norm']:.6f} vs {B['norm']:.6f}  ->  {100*(A['norm']/B['norm']-1):+.3f}%")
    print(f"  cosine     {A['cos']:.4f} vs {B['cos']:.4f}      ->  {B['cos']/A['cos']:.0f}x apart")
    for r in (C, A, B):
        print(f"    {r['lab']:22s} lost {len(r['lost'])}  ret {r['ret']}  "
              f"shared with control {len(r['lost'] & C['lost'])}/{len(C['lost'])}")
    print(f"    rows lost by ALL THREE: {len(A['lost'] & B['lost'] & C['lost'])}   "
          f"union of all damage: {len(A['lost'] | B['lost'] | C['lost'])}")


if __name__ == "__main__":
    main()
