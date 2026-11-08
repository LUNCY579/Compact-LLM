"""Which mention feature predicts damage -- and does dispersion add anything to position?

    python analysis/mention_features.py --root <records>/alias001-20260918 \
        --out <records>/alias001-20260918/MENTION_FEATURES.json

Amendment 19 tested the peer session's reframe -- that `answer_first_pos` measures *distance to
the first retrievable mention* and might be the mechanism rather than an artifact. On the
marginals it does not win: in `high`, mean mention position beats first (+0.472 vs +0.429), and
mention **spread** picks up a negative coefficient (-0.237), which reads like redundancy --
widely-spread answers damaged less.

**Both of those marginal readings are unsafe and this script is why.** With exactly three
mentions, `first`, `mean`, `last` and `spread` are algebraically dependent on the same three
numbers, and in `high` spread and mean position are correlated at -0.345. A marginal coefficient
on any one of them can be borrowed entirely from another.

So: rank **partial** correlations, each feature controlling for mean position and vice versa.

## Why `equal` is the load-bearing condition here and `high` is not

The obvious test is to partial within `high`, where spread has the most range. But that is
exactly where spread and mean are entangled (-0.345), so a partial there is asking a question
about a nearly-collinear pair.

**In `equal` the two are orthogonal (+0.022).** The partial is therefore barely different from
the marginal, nothing is being subtracted, and a redundancy effect -- if real -- has a clean
place to show itself. `equal` is the *better* test of dispersion despite having fewer mentions,
and it is the one an argument conducted only in `high` cannot make.

`low` cannot be tested at all: one mention per row, so spread is identically zero.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CONDITIONS = ["equal", "high"]          # `low` has one mention: spread is degenerate
PERM_ITERS = 20000
N_TESTS = 4                             # two lanes x two conditions, per feature family
BONFERRONI = 0.05 / N_TESTS


def norm(v) -> str:
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return " ".join(s.lower().replace(".", " ").replace(",", " ").split())


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[o[k]] = avg
        i = j + 1
    return r


def pearson(x, y) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def spearman(x, y) -> float:
    return pearson(rank(x), rank(y))


def partial(x, y, z) -> float:
    """Rank partial correlation of x with y, controlling z."""
    rx, ry, rz = rank(x), rank(y), rank(z)
    xy, xz, yz = pearson(rx, ry), pearson(rx, rz), pearson(ry, rz)
    d = ((1 - xz ** 2) * (1 - yz ** 2)) ** 0.5
    return (xy - xz * yz) / d if d else 0.0


def perm_p(fn, x, y, z, seed: int, iters: int = PERM_ITERS) -> float:
    """Permute the OUTCOME y, holding x and the control z fixed."""
    obs = abs(fn(x, y, z))
    rnd = random.Random(seed)
    yy = list(y)
    c = 0
    for _ in range(iters):
        rnd.shuffle(yy)
        if abs(fn(x, yy, z)) >= obs - 1e-12:
            c += 1
    return c / iters


def load(root: Path, sub: str, rows: dict) -> dict:
    f = root / sub / "predictions.jsonl"
    if not f.exists():
        raise SystemExit(f"missing predictions: {f}")
    p = {r["id"]: r.get("prediction", "") for r in
         (json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip())}
    return {i: norm(p[i]) == rows[i]["reference_normalised"] for i in rows}


def mentions(r: dict) -> list[float]:
    c, ref = r["context"], r["reference"]
    out, i = [], c.find(ref)
    while i >= 0:
        out.append(i / len(c))
        i = c.find(ref, i + 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = {r["id"]: r for r in
            (json.loads(x) for x in
             (a.root / "ALIAS_PROBE.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    src = load(a.root, "gen-B", rows)
    lanes = {"native_a1": load(a.root, "gen-a10", rows),
             "lwf_l0": load(a.root, "gen-lwf_l0", rows)}

    res: dict = {}
    for lab, arm in lanes.items():
        per_c = {}
        for c in CONDITIONS:
            ids = [i for i in rows if rows[i]["condition"] == c and src[i]]
            m = [mentions(rows[i]) for i in ids]
            mean = [sum(x) / len(x) for x in m]
            spread = [x[-1] - x[0] for x in m]
            first = [x[0] for x in m]
            last = [x[-1] for x in m]
            dmg = [0 if arm[i] else 1 for i in ids]
            entangle = spearman(spread, mean)
            feats = {}
            for fname, fv in (("first", first), ("last", last), ("spread", spread)):
                feats[fname] = {
                    "marginal_rho": round(spearman(fv, dmg), 4),
                    "partial_rho_given_mean": round(partial(fv, dmg, mean), 4),
                    "partial_p": perm_p(partial, fv, dmg, mean, 51),
                }
            feats["mean"] = {
                "marginal_rho": round(spearman(mean, dmg), 4),
                "partial_rho_given_spread": round(partial(mean, dmg, spread), 4),
                "partial_p": perm_p(partial, mean, dmg, spread, 51),
            }
            per_c[c] = {
                "n": len(ids), "damaged": sum(dmg),
                "mentions_per_row": sorted({len(x) for x in m}),
                "spread_vs_mean_rho": round(entangle, 4),
                "orthogonal": abs(entangle) < 0.10,
                "features": feats,
            }
        res[lab] = per_c

    out = {
        "record": "MENTION_FEATURES",
        "class": "MECHANISM SCOUT -- a LOOK, not registered in advance",
        "question": ("does any single-mention or dispersion feature add anything to MEAN mention "
                     "position in predicting damage?"),
        "why_equal_is_load_bearing": (
            "in `high` spread and mean are entangled (rho ~ -0.345) so a partial there asks about "
            "a nearly-collinear pair; in `equal` they are orthogonal (rho ~ +0.022), so the "
            "partial barely differs from the marginal and a real dispersion effect has a clean "
            "place to appear. an argument conducted only in `high` cannot make this point"),
        "low_excluded": "one mention per row -- spread is identically zero",
        "perm_iters": PERM_ITERS, "bonferroni_over": N_TESTS, "bonferroni_bound": BONFERRONI,
        "by_lane": res,
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'lane':<12}{'cond':<7}{'n':>4}{'dmg':>5}{'spread~mean':>13}")
    for lab in lanes:
        for c in CONDITIONS:
            v = res[lab][c]
            tag = "ORTHOGONAL" if v["orthogonal"] else "entangled"
            print(f"{lab:<12}{c:<7}{v['n']:>4}{v['damaged']:>5}{v['spread_vs_mean_rho']:>13.3f}"
                  f"  {tag}")
            for fname in ("mean", "first", "last", "spread"):
                f = v["features"][fname]
                key = ("partial_rho_given_spread" if fname == "mean"
                       else "partial_rho_given_mean")
                ctl = "spread" if fname == "mean" else "mean"
                star = " *" if f["partial_p"] < BONFERRONI else ""
                print(f"    {fname:<8}marginal {f['marginal_rho']:>+.3f}   "
                      f"| {ctl:<7}{f[key]:>+.3f}  p={f['partial_p']:.4f}{star}")
    print(f"\n(* clears Bonferroni {BONFERRONI:.4f} over {N_TESTS} lane x condition cells)")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
