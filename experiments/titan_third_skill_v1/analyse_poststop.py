"""Descriptive TTR25 versus TT/TTC25 on the exposed Stage A banks only."""
from __future__ import annotations

import json
from pathlib import Path

import analyse_stage_a as A

HERE = Path(__file__).resolve().parent
PN = HERE / "poststop_null"
OUT = HERE / "POSTSTOP_NULL_RESULT.json"


def scores(c: dict, t: dict, h: dict) -> dict:
    return {"C_semantic": sum(r["semantic_correct"] for r in c.values()),
            "C_strict": sum(r["strict_correct"] for r in c.values()),
            "T_primary": sum(r["primary"] for r in t.values()),
            "H_boundary": sum(r["credited"] for r in h.values() if r["row_type"] == "boundary")}


def contrast(tt: tuple, other: tuple) -> dict:
    c0, t0, h0 = tt
    c1, t1, h1 = other
    cb = A.paired(c0, c1, "semantic_correct")
    tb = A.paired(t0, t1, "primary")
    hb0 = {i: r for i, r in h0.items() if r["row_type"] == "boundary"}
    hb1 = {i: r for i, r in h1.items() if r["row_type"] == "boundary"}
    hb = A.paired(hb0, hb1, "credited")
    return {"C_semantic_gain": cb[0] - cb[1], "C_gained": cb[0], "C_lost": cb[1],
            "T_loss": tb[1] - tb[0], "T_gained": tb[0], "T_lost": tb[1],
            "H_loss": hb[1] - hb[0], "H_gained": hb[0], "H_lost": hb[1]}


def main() -> int:
    if OUT.exists():
        raise SystemExit("POSTSTOP_NULL_RESULT.json exists (create-only)")
    stage = json.loads((HERE / "STAGE_A_RESULT.json").read_text(encoding="utf-8"))
    if stage["verdict"] != "STOP_NO_ELIGIBLE_BETA" or (HERE / "STAGE_B_RESULT.json").exists():
        raise SystemExit("post-STOP control refused: Stage A/B state mismatch")
    original_sa = A.SA
    tt = (A.json192(A.c_rows("TT")), A.t_rows("TT"), A.h_rows("TT"))
    c25 = (A.json192(A.c_rows("TTC25")), A.t_rows("TTC25"), A.h_rows("TTC25"))
    A.SA = PN
    try:
        rand = (A.json192(A.c_rows("TTR25")), A.t_rows("TTR25"), A.h_rows("TTR25"))
    finally:
        A.SA = original_sa
    build = json.loads((PN / "arms" / "TTR" / "BUILD.json").read_text(encoding="utf-8"))
    if build["rand_seed"] != 20260926 or abs(build["update_norm_fro"] - 0.21437929157719313) > 2e-6:
        raise SystemExit("TTR25 norm/seed canary failed")
    result = {"experiment": "TRISKILL-POSTSTOP-001", "registered_in": "POSTSTOP_NULL_PREREG.md",
              "stage_a_verdict_unchanged": stage["verdict"], "stage_b_banks_read": False,
              "arm": {"name": "TTR25", "formula": "TT + Gaussian R, per-tensor norm matched to 0.25*delta_C",
                      "seed": build["rand_seed"], "update_norm_fro": build["update_norm_fro"],
                      "weight_sha256": build["weight_sha256"]},
              "scores": {"TT": scores(*tt), "TTC25": scores(*c25), "TTR25": scores(*rand)},
              "contrasts_vs_TT": {"TTC25": contrast(tt, c25), "TTR25": contrast(tt, rand)},
              "status": "EXPLORATORY_ONE_RANDOM_DRAW", "quality_claim": False, "release_permitted": False}
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"scores": result["scores"], "contrasts_vs_TT": result["contrasts_vs_TT"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
