"""Regenerate every headline number in RESULTS.md and the MODULE-DISJOINT RESULTS.md from raw JSON.

Added 2026-09-24 after review found that no committed script produced the period-tolerant column,
its McNemar p-values, the content-clean re-analysis, or the gold+"." decomposition — they had been
computed in ad-hoc shell heredocs. This file is the committed code path for all of them.

Reads only JSON already in the repo. No model, no GPU, no network.

    python reproduce_results.py            # print every table
    python reproduce_results.py --check    # exit 1 if any number differs from the value
                                           #   written in the results documents
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE.parent / "module_disjoint_composition_v1"

# Values as written in the results documents. --check asserts each one.
EXPECTED = {
    ("strict", "SPARSE"): 102, ("strict", "AVG"): 69, ("strict", "ADD-100"): 6, ("strict", "ADD-O75"): 63,
    ("tolerant", "ADD-100"): 97, ("tolerant", "ADD-O75"): 98, ("tolerant", "ADD-75"): 94, ("tolerant", "ADD-S75"): 89,
    ("tolerant", "SPARSE_ATTN"): 68, ("tolerant", "SCALAR_MATCHED"): 88, ("tolerant", "SPARSE_MLP"): 67,
    ("tolerant", "SCALAR_MATCHED_MLP"): 94, ("tolerant", "MODULE_DISJOINT"): 73,
    ("periods", "ADD-100"): 178, ("periods", "ADD-O75"): 63, ("periods", "MODULE_DISJOINT"): 0, ("periods", "SPARSE"): 0,
    ("gold_dot_among_lost", "ADD-100"): 88, ("lost_vs_sparse", "ADD-100"): 99,
    ("clean_n",): 181,
    # copy_wrong, as in the MODULE-DISJOINT P6 table (diagnose_collapse definition)
    ("copy_wrong", "SPARSE"): 4, ("copy_wrong", "SCALAR_MATCHED"): 12, ("copy_wrong", "SCALAR_MATCHED_MLP"): 7,
    ("copy_wrong", "SPARSE_MLP"): 21, ("copy_wrong", "SPARSE_ATTN"): 21, ("copy_wrong", "MODULE_DISJOINT"): 15,
}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load():
    bank = {r["id"]: r for r in json.loads((HERE / "T_BANK_V2.json").read_text(encoding="utf-8"))["rows"]}
    arms = {}
    for f in (HERE / "CALIBRATION_V2.json", HERE / "V2_ARMS.json", MD / "MD_SCORES.json", MD / "P6_SCORES.json"):
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("bank_sha256", "").startswith("fc094136") is False:
            raise SystemExit(f"{f.name} was scored on a different bank ({d.get('bank_sha256')})")
        for a in d["arms"]:
            arms[a["arm"]] = {p["id"]: p["generation"].strip() for p in a["predictions"]}
    for name, g in arms.items():
        if set(g) != set(bank):
            raise SystemExit(f"arm {name} row set differs from the bank")
    dirty_path = HERE / "CONTENT_CONTAMINATED_ROWS.json"
    dirty = set(json.loads(dirty_path.read_text(encoding="utf-8"))) if dirty_path.exists() else set()
    return bank, arms, dirty


def correct(gen: str, ref: str, tolerant: bool) -> bool:
    return (gen.rstrip(".") if tolerant else gen) == ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bank, arms, dirty = load()
    ids_all = sorted(bank)
    ids_clean = sorted(set(bank) - dirty)
    got = {("clean_n",): len(ids_clean)}

    def score(n, ids, tol):
        return sum(correct(arms[n][i], bank[i]["reference"], tol) for i in ids)

    # Two copy definitions exist and the documents used both. `copied_all` counts every verbatim
    # copy of the input state (score_arms.py's copied_input_verbatim). `copied_wrong` counts copies
    # that are also WRONG (diagnose_collapse.py's copied_input_state). They differ on the 13 rows
    # where the correct answer IS the input. For "restriction drives echoing" the failure-relevant
    # measure is copied_wrong.
    print(f"{'arm':<20}{'strict':>8}{'tolerant':>10}{'periods':>9}{'copy_all':>9}{'copy_wrong':>11}{'tol clean':>11}")
    for n in sorted(arms):
        s, t = score(n, ids_all, False), score(n, ids_all, True)
        per = sum(arms[n][i].endswith(".") for i in ids_all)
        cop = sum(arms[n][i] == bank[i]["copy_baseline"] for i in ids_all)
        copw = sum(arms[n][i] == bank[i]["copy_baseline"] and arms[n][i] != bank[i]["reference"] for i in ids_all)
        got[("strict", n)], got[("tolerant", n)], got[("periods", n)] = s, t, per
        got[("copy_wrong", n)] = copw
        print(f"{n:<20}{s:>8}{t:>10}{per:>9}{cop:>9}{copw:>11}{score(n, ids_clean, True):>11}")

    S, A = arms["SPARSE"], arms["ADD-100"]
    lost = [i for i in ids_all if correct(S[i], bank[i]["reference"], False) and not correct(A[i], bank[i]["reference"], False)]
    got[("lost_vs_sparse", "ADD-100")] = len(lost)
    got[("gold_dot_among_lost", "ADD-100")] = sum(A[i] == bank[i]["reference"] + "." for i in lost)
    print(f"\nSPARSE->ADD-100 strict: {len(lost)} lost, of which {got[('gold_dot_among_lost','ADD-100')]} are exactly gold+'.'")

    print("\nContrast                                   rows    strict (net, p)        tolerant (net, p)")
    for lab, x, y in (("attenuation  SPARSE -> AVG", "SPARSE", "AVG"),
                      ("interference SPARSE -> ADD-100", "SPARSE", "ADD-100"),
                      ("P1'  SPARSE_ATTN -> SCALAR_MATCHED", "SPARSE_ATTN", "SCALAR_MATCHED"),
                      ("P6   SPARSE_MLP -> SCALAR_MATCHED_MLP", "SPARSE_MLP", "SCALAR_MATCHED_MLP"),
                      ("P2   SPARSE_ATTN -> MODULE_DISJOINT", "SPARSE_ATTN", "MODULE_DISJOINT"),
                      ("P3   ADD-100 -> MODULE_DISJOINT", "ADD-100", "MODULE_DISJOINT")):
        if x not in arms or y not in arms:
            continue
        for tag, ids in (("all", ids_all), ("clean", ids_clean)):
            cells = []
            for tol in (False, True):
                X = [correct(arms[x][i], bank[i]["reference"], tol) for i in ids]
                Y = [correct(arms[y][i], bank[i]["reference"], tol) for i in ids]
                b = sum(p and not q for p, q in zip(X, Y)); c = sum(q and not p for p, q in zip(X, Y))
                cells.append(f"{c - b:+4d}, p={mcnemar(b, c):.2g}")
            print(f"  {lab:<38} {tag:<6} {cells[0]:<22} {cells[1]}")

    if a.check:
        bad = [(k, v, got.get(k)) for k, v in EXPECTED.items() if got.get(k) != v]
        for k, v, g in bad:
            print(f"MISMATCH {k}: document says {v}, recomputed {g}", file=sys.stderr)
        print(f"\n--check: {len(EXPECTED) - len(bad)}/{len(EXPECTED)} documented numbers reproduce")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
