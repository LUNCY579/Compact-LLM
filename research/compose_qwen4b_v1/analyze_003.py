"""COMPOSE4B-003 analysis: the 3x2 factorial, with Holm m=3 over the registered family.

WRITTEN AND COMMITTED BEFORE ANY 003 SCORE EXISTS. `scored/` is empty as of this file.

STATISTICS ARE IMPORTED, NOT REIMPLEMENTED. exact_mcnemar / mde_at / cp_upper /
equivalence_bound_rows come from analyze.py, which reproduced six recorded values from two prior
experiments exactly. Only the row EXTRACTION is new here, because the HSKILL schema
(`per_row` with row_type/abstains/credited) differs from the acquisition schema
(`detail` with `canonical`) that analyze.py reads.

AND THAT IS EXACTLY WHERE THE RISK IS. Four wrong extractors in one day on this programme, none
caught by reading code, all caught by collision with a known number. So:

    THE CANARY IS MANDATORY AND IT GATES EVERYTHING.

Before reading a single 003 number, this script recomputes COMPOSE4B-002's three published
contrasts from 002's own scored files and refuses to continue unless all three match the
published values exactly:

    P1  th_s - scaled_t  sensitivity  net +19  nd 31  p 0.00088
    P2  th_s - scaled_t  competence   net +15  nd 35  p 0.01667
    P3  h_only - base    sensitivity  net  +2  nd 48  p 0.885

If the extractor is wrong, those will not reproduce, and no 003 number is trustworthy either.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze import exact_mcnemar, mde_at, cp_upper, equivalence_bound_rows  # noqa: E402

RECORDS = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
S002 = RECORDS / "compose4b002-20260921/scored"
S003 = RECORDS / "compose4b003-20260921/scored"
ALPHA = 0.05

# Registered family, fixed in PREREG_003.md section 6. m stays 3 regardless of outcome.
HOLM_M = 3
# Holm's most stringent step is ALPHA/3 = 0.01667. The minimum attainable exact p is 0.0312 at
# nd=6 and 0.0156 at nd=7, so nd<=6 cannot clear step one however the pairs split.
MIN_ND_FOR_HOLM = 7


def load(label):
    for d in (S003, S002):
        p = d / f"{label}.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")), d.parent.name
    raise SystemExit(f"missing scored arm: {label}")


def endpoint_rows(doc, endpoint):
    """{id: bool} over exactly the half the endpoint is defined on."""
    out = {}
    for r in doc["per_row"]:
        if endpoint == "sensitivity":
            if r["row_type"] != "boundary":
                continue
            out[r["id"]] = bool(r["abstains"])
        elif endpoint == "competence":
            if r["row_type"] == "boundary":
                continue
            out[r["id"]] = bool(r["credited"]) and not bool(r["abstains"])
        elif endpoint == "decision":
            if r["row_type"] == "boundary":
                continue
            out[r["id"]] = not bool(r["abstains"])
        else:
            raise SystemExit(f"unknown endpoint {endpoint}")
    return out


def contrast(name, arm_a, arm_b, endpoint):
    """arm_a - arm_b, paired on identical ids. Positive net favours arm_a."""
    da, _ = load(arm_a)
    db, _ = load(arm_b)
    ra, rb = endpoint_rows(da, endpoint), endpoint_rows(db, endpoint)
    if set(ra) != set(rb):
        raise SystemExit(f"{name}: row id sets differ -- BLOCKED_ON_COMPARABILITY")
    ids = sorted(ra)
    b = sum(1 for i in ids if ra[i] and not rb[i])      # gains for arm_a
    c = sum(1 for i in ids if rb[i] and not ra[i])      # losses
    nd = b + c
    ka, kb = sum(ra.values()), sum(rb.values())
    return {
        "contrast": name, "endpoint": endpoint, "n": len(ids),
        "arm_a": arm_a, "arm_b": arm_b, "k_a": ka, "k_b": kb,
        "gains_b": b, "losses_c": c, "nd": nd, "net": b - c,
        "p_exact_two_sided": exact_mcnemar(b, c),
        "min_attainable_p": exact_mcnemar(nd, 0) if nd else None,
        "mde_at_nd": mde_at(nd),
        # a contrast cannot gain a row arm_b already has
        "ceiling_rows": len(ids) - kb,
        "equivalence_bound_rows": equivalence_bound_rows(min(b, c), nd) if nd else None,
    }


def canary():
    """Reproduce COMPOSE4B-002's three PUBLISHED contrasts, or refuse to go on."""
    expect = [
        ("P1 th_s-scaled_t sensitivity", "th_s", "scaled_t", "sensitivity", 19, 31, 0.00088),
        ("P2 th_s-scaled_t competence", "th_s", "scaled_t", "competence", 15, 35, 0.01667),
        ("P3 h_only-base sensitivity", "h_only", "base", "sensitivity", 2, 48, 0.885),
    ]
    out, ok = [], True
    for nm, a, b, ep, net, nd, p in expect:
        r = contrast(nm, a, b, ep)
        agree = (r["net"] == net and r["nd"] == nd and abs(r["p_exact_two_sided"] - p) < 5e-4)
        ok &= agree
        out.append({"check": nm, "published": {"net": net, "nd": nd, "p": p},
                    "recomputed": {"net": r["net"], "nd": r["nd"],
                                   "p": round(r["p_exact_two_sided"], 6)},
                    "agrees": agree})
    if not ok:
        print(json.dumps({"CANARY": "FAILED", "detail": out}, indent=2))
        raise SystemExit("CANARY FAILED -- the extractor disagrees with published numbers. "
                         "No 003 result may be read until this is resolved.")
    return out


def holm(rows_):
    """Holm-Bonferroni at m=3, fixed. Returns the same rows with holm fields added."""
    order = sorted(range(len(rows_)), key=lambda i: rows_[i]["p_exact_two_sided"])
    prev, still = 0.0, True
    for rank, i in enumerate(order):
        thr = ALPHA / (HOLM_M - rank)
        p = rows_[i]["p_exact_two_sided"]
        adj = max(prev, min(1.0, p * (HOLM_M - rank)))
        prev = adj
        rows_[i]["holm_rank"] = rank + 1
        rows_[i]["holm_threshold"] = thr
        rows_[i]["holm_adjusted_p"] = adj
        still = still and (p <= thr)
        rows_[i]["holm_significant"] = bool(still)
    return rows_


def verdict(r):
    if r["nd"] < MIN_ND_FOR_HOLM:
        return "UNINFORMATIVE_BY_DISCORDANCE"
    if r["mde_at_nd"] is None:
        return "UNINFORMATIVE_BY_DISCORDANCE"
    if r["ceiling_rows"] < r["mde_at_nd"]:
        return "UNINFORMATIVE_BY_CEILING"
    if r["net"] >= r["mde_at_nd"] and r["holm_significant"]:
        return "FIRES"
    return "DOES_NOT_FIRE"


def main():
    can = canary()

    registered = [
        contrast("P1 indep-delta_t SENSITIVITY", "indep", "delta_t", "sensitivity"),
        contrast("P2 indep-delta_t COMPETENCE", "indep", "delta_t", "competence"),
        contrast("P3 th_s-scaled_h COMPETENCE", "th_s", "scaled_h", "competence"),
    ]
    registered = holm(registered)
    for r in registered:
        r["verdict"] = verdict(r)

    descriptive = [
        contrast("D1 indep-th_s SENSITIVITY", "indep", "th_s", "sensitivity"),
        contrast("D2 indep-th_s COMPETENCE", "indep", "th_s", "competence"),
        contrast("D3 th_s-scaled_h SENSITIVITY", "th_s", "scaled_h", "sensitivity"),
        contrast("D4 scaled_h-base SENSITIVITY", "scaled_h", "base", "sensitivity"),
        contrast("D5 scaled_h-h_only DECISION", "scaled_h", "h_only", "decision"),
        contrast("D6 indep-delta_t DECISION", "indep", "delta_t", "decision"),
    ]

    arms = {}
    for lbl in ("base", "delta_t", "scaled_t", "th_s", "h_only", "scaled_h", "indep"):
        try:
            d, src = load(lbl)
        except SystemExit:
            continue
        arms[lbl] = {
            "source_root": src,
            "sensitivity": d["sensitivity_abstains_on_boundary"]["k"],
            "decision": d["specificity_decision_attempts_on_answerable"]["k"],
            "competence": d["specificity_competence_correct_on_answerable"]["k"],
            "J_decision": d["J_decision"],
            "boundary_headroom_rows": d["boundary_headroom_rows"],
        }

    out = {
        "experiment": "COMPOSE4B-003",
        "canary": {"status": "PASS", "checks": can,
                   "why": "the extractor is validated by COLLISION with published numbers, "
                          "not by reading it"},
        "holm_m": HOLM_M, "alpha": ALPHA,
        "holm_note": f"nd < {MIN_ND_FOR_HOLM} cannot clear Holm step one; reported "
                     "UNINFORMATIVE_BY_DISCORDANCE, and m stays 3",
        "arms": arms,
        "registered": registered,
        "descriptive": descriptive,
        "quality_claim": False, "release_permitted": False,
    }
    (S003.parent / "ANALYSIS_003.json").write_text(json.dumps(out, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"canary": "PASS", "arms": arms}, indent=2))
    print("\nREGISTERED (Holm m=3):")
    for r in registered:
        print(f"  {r['contrast']:34s} net {r['net']:+4d}  nd {r['nd']:3d}  "
              f"p {r['p_exact_two_sided']:.5f}  holm_adj {r['holm_adjusted_p']:.5f}  "
              f"MDE {r['mde_at_nd']}  ceil {r['ceiling_rows']:3d}  -> {r['verdict']}")
    print("\nDESCRIPTIVE (not in the Holm family, no prediction):")
    for r in descriptive:
        print(f"  {r['contrast']:34s} net {r['net']:+4d}  nd {r['nd']:3d}  "
              f"p {r['p_exact_two_sided']:.5f}  minp {r['min_attainable_p']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
