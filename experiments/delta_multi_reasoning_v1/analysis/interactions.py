"""Compute the predeclared contrast family across systems. Gold-free; reads MATRIX_*.json.

THE FAMILY IS DECLARED HERE AND IN THE CODE, NOT CHOSEN AFTER LOOKING
--------------------------------------------------------------------
Multiplicity correction is meaningless if the family is assembled after seeing which contrasts
look interesting. These five are fixed, in this order, and Holm-Bonferroni runs over exactly
them. Adding a sixth after reading results voids the correction and the module says so.

  1  D: MR vs O     -- does multi-reasoning beat the ordinary solve on the delta candidate?
  2  D: MR vs REP    -- does it beat FOUR TRIES at the same thing? The contrast that decides
                       whether any gain is attributable to differentiated reasoning at all.
  3  B: MR vs O     -- the same on the frozen source, so the interaction has a second arm
  4  Delta x MR      -- [s(D,MR)-s(D,O)] - [s(B,MR)-s(B,O)]
  5  D: T vs O      -- token-allowance control: does the same prompt at MR's TOTAL output
                       allowance already capture the gain, without any roles?

Contrast 5 matters more than it looks. If T alone recovers what MR recovers, then MR bought
nothing but output tokens and the roles are decoration. It is not compute matching -- MR makes
five calls and T makes one -- and that asymmetry is stated wherever the number appears.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.paired import (  # noqa: E402
    REP001_SINGLE_DRAW_SPREAD, cluster_bootstrap_ci, holm_bonferroni, interaction,
    mcnemar_exact, noise_verdict, paired,
)

FAMILY = ("D:MR_vs_O", "D:MR_vs_REP", "B:MR_vs_O", "DeltaxMR", "D:T_vs_O")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    mats = {}
    for p in a.matrix:
        m = json.loads(p.read_text(encoding="utf-8"))
        mats[m["system"]] = m
    if not {"B", "D"} <= set(mats):
        raise SystemExit(f"need systems B and D; have {sorted(mats)}")

    # item -> bool, per system per condition
    out: dict[str, dict[str, dict[str, bool]]] = {}
    worlds: dict[str, str] = {}
    strata: dict[str, str] = {}
    for sysname, m in mats.items():
        out[sysname] = {c: {} for c in ("O", "T", "REP", "MR", "MR_S0")}
        for it in m["per_item"]:
            for c in out[sysname]:
                out[sysname][c][it["id"]] = bool(it[c])
            worlds[it["id"]] = it["world_id"]
            strata[it["id"]] = it["stratum"]

    def pair(sysname, base, cand):
        return paired(f"{sysname}:{base}", f"{sysname}:{cand}",
                      out[sysname][base], out[sysname][cand])

    contrasts: dict[str, dict] = {}
    pvals: dict[str, float] = {}

    for key, sysname, base, cand in (("D:MR_vs_O", "D", "O", "MR"),
                                     ("D:MR_vs_REP", "D", "REP", "MR"),
                                     ("B:MR_vs_O", "B", "O", "MR"),
                                     ("D:T_vs_O", "D", "O", "T")):
        p = pair(sysname, base, cand)
        d = p.as_dict()
        d["noise_verdict"] = noise_verdict("acquisition", p.net_gain)
        # World-clustered interval on the per-item DIFFERENCE, which is the paired quantity.
        diff: dict[str, list[float]] = {}
        for i in out[sysname][base]:
            v = (1.0 if out[sysname][cand][i] else 0.0) - (1.0 if out[sysname][base][i] else 0.0)
            diff.setdefault(worlds[i], []).append(v)
        d["world_clustered_mean_difference"] = cluster_bootstrap_ci(diff)
        contrasts[key] = d
        pvals[key] = d["mcnemar_exact_two_sided_p"]

    def s(sysname, cond):
        return float(sum(out[sysname][cond].values()))

    inter = interaction(s, "D", "B", "MR", "O")
    # An interaction's own interval: per-item (MR-O) difference, D minus B, clustered by world.
    idiff: dict[str, list[float]] = {}
    for i in out["D"]["O"]:
        dd = (1.0 if out["D"]["MR"][i] else 0.0) - (1.0 if out["D"]["O"][i] else 0.0)
        bb = (1.0 if out["B"]["MR"][i] else 0.0) - (1.0 if out["B"]["O"][i] else 0.0)
        idiff.setdefault(worlds[i], []).append(dd - bb)
    inter["world_clustered_mean"] = cluster_bootstrap_ci(idiff)
    # Paired test on the interaction: discordance between the two systems' MR-O improvements.
    d_up = {i for i in out["D"]["O"] if out["D"]["MR"][i] and not out["D"]["O"][i]}
    b_up = {i for i in out["B"]["O"] if out["B"]["MR"][i] and not out["B"]["O"][i]}
    inter["mcnemar_on_improvement_sets"] = mcnemar_exact(len(d_up - b_up), len(b_up - d_up))
    inter["noise_verdict"] = noise_verdict("acquisition", int(round(inter["interaction"])))
    contrasts["DeltaxMR"] = inter
    pvals["DeltaxMR"] = inter["mcnemar_on_improvement_sets"]

    per_stratum = {}
    for st in sorted(set(strata.values())):
        ids = [i for i, v in strata.items() if v == st]
        per_stratum[st] = {
            "n": len(ids),
            **{f"{sysname}_{c}": sum(1 for i in ids if out[sysname][c][i])
               for sysname in sorted(out) for c in ("O", "REP", "MR")},
        }

    report = {
        "schema": 1, "record": "DMRF_INTERACTIONS",
        "split": mats["D"]["split"], "n_items": len(worlds),
        "systems": sorted(mats),
        "predeclared_family": list(FAMILY),
        "family_declared_before_results": True,
        "family_note": ("Holm-Bonferroni runs over exactly these five. Adding a contrast after "
                        "reading results voids the correction."),
        "contrasts": contrasts,
        "multiplicity": holm_bonferroni({k: pvals[k] for k in FAMILY}),
        "per_stratum": per_stratum,
        "noise_floor_used": dict(REP001_SINGLE_DRAW_SPREAD),
        "resampling_unit": "world",
        "interpretation_rules": {
            "mr_vs_rep_is_the_attribution_contrast": (
                "MR beating O is not evidence for differentiated reasoning; MR beating REP is. "
                "REP is four tries at the same operation with the same call and token budget."),
            "t_vs_o_is_the_allowance_control": (
                "If T alone recovers MR's gain, the gain is output tokens rather than roles. T "
                "is token-allowance matched, NOT compute matched: MR makes five calls, T one."),
            "within_noise": ("Any net gain at or below the REP-001 single-draw spread is "
                             "WITHIN_NOISE and may not be reported as a direction."),
            "zero_damage": ("A preservation statement requires the paired damage count to be "
                            "actually zero, not a large p-value."),
        },
        "quality_claim": False, "release_permitted": False,
    }
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "n_items": report["n_items"],
        "scores": {sysname: {c: int(s(sysname, c)) for c in ("O", "T", "REP", "MR", "MR_S0")}
                   for sysname in sorted(out)},
        "contrasts": {k: ({"interaction": contrasts[k]["interaction"],
                           "simple_effects": contrasts[k]["simple_effect_on"],
                           "ci": contrasts[k]["world_clustered_mean"],
                           "noise": contrasts[k]["noise_verdict"]}
                          if k == "DeltaxMR" else
                          {"repair": contrasts[k]["repair"], "damage": contrasts[k]["damage"],
                           "net": contrasts[k]["net_gain"],
                           "p": round(contrasts[k]["mcnemar_exact_two_sided_p"], 4),
                           "noise": contrasts[k]["noise_verdict"]})
                      for k in FAMILY},
        "holm": {k: v["reject"] for k, v in report["multiplicity"]["results"].items()},
        "per_stratum": per_stratum,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
