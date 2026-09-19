"""Norm audit of the composition arms — is "matched magnitude" matched in the way it is read?

    python analysis/norm_audit.py --records <records> --out <records>/NORM_AUDIT.json

Run at the peer session's request, on their arms, from their own receipts. Zero model calls and
zero training: every quantity here is recomputed from per-tensor norms the builders already
recorded, so it is an audit of the bookkeeping rather than a new measurement.

## The question that was asked

`TH16s` is described as "both skills' rank-16 directions rescaled per tensor to T16's magnitude",
built with

    s_t = ||P16(delta_T, t)||_F / ||P16(delta_T, t) + P16(delta_H, t)||_F      per 2D tensor

and the peer flagged that this is matched on *measured norm* rather than on a label, but that it
"needs auditing rather than assuming."

## What the audit confirms

The **total** norm bookkeeping is exact: `ratio_arm_over_T16` = 1.0000000004, no fallbacks, all
220 tensors. The builder verifies it and refuses to write on drift. Nothing is wrong with what
the receipt claims.

## What the audit finds anyway, and it bears on the headline

"Rescaled to T16's magnitude" matches the magnitude of the **sum**. It does not preserve the
magnitude of either **component**. Because `s_t < 1` everywhere, TH16s contains

    s_t * P16(delta_T)     and     s_t * P16(delta_H)

so **each skill is present at a fraction of its own solo strength**. The composition arm is
therefore not "both skills at one skill's budget" in the sense of each skill being intact; it is
both skills each attenuated to roughly two thirds.

That matters because the reported composition cost is measured against **T16 at full strength**.
The correct counterfactual for the acquisition loss is `theta_supa + s * P16(delta_T)` with no
delta_H at all — an arm that does not exist. Some unknown share of the loss attributed to skill
conflict is simply the dose reduction the rescaling imposes.

## Why TT-001's same-skill control does not already cover this

TT-001 summed two *same-skill* deltas under the identical construction and scored above T16, and
was read as showing the cost is skill conflict rather than a generic cost of summing. But the
same formula does structurally different things depending on **alignment**:

  * two **aligned** same-skill deltas give ||d1 + d2|| ~ 2||d||, so s ~ 0.5 and the rescaled sum
    is their **average** — each at full effective strength along the shared direction, with seed
    noise reduced. Scoring above T16 is what averaging two seeds should do.
  * two **near-orthogonal** cross-skill deltas give ||dT + dH|| ~ sqrt(||dT||^2 + ||dH||^2), so
    s ~ 0.69 and each component is **attenuated**, with no shared direction to average along.

So TT-001 controls for "summing two deltas" and does **not** control for "each skill attenuated
to s." The two constructions are the same line of code and different operations.

## What this does NOT claim

It does not claim the interference is unreal, and it does not retract anything. It identifies a
confound in the size of the estimate and names the one arm that removes it. The subspace overlap
measurement, the specificity result and the sensitivity numbers are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def fro(vals) -> float:
    return math.sqrt(sum(v * v for v in vals))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    th16s = json.loads((a.records / "mag001-20260918" / "arm-TH16s" / "ARM.json")
                       .read_text(encoding="utf-8"))
    th16 = json.loads((a.records / "compose001-20260918" / "arm-TH16" / "ARM.json")
                      .read_text(encoding="utf-8"))
    t16 = json.loads((a.records / "compose001-20260918" / "arm-T16" / "ARM.json")
                     .read_text(encoding="utf-8"))
    h16 = json.loads((a.records / "compose001-20260918" / "arm-H16" / "ARM.json")
                     .read_text(encoding="utf-8"))

    pts = th16s["norm_bookkeeping"]["per_tensor_scales"]

    # ---- the rank-16 pieces, from the two independent receipt families --------------------
    # compose001 records the FULL delta norms and the fraction each truncation retains.
    full_T = th16["delta_T_fro"]
    full_H = th16["delta_H_fro"]
    p16_T_from_frac = full_T * t16["delta_norm_fraction_retained"]
    p16_H_from_frac = full_H * h16["delta_norm_fraction_retained"]

    # mag001 records per-tensor norms directly. These two routes must agree.
    p16_T_direct = fro([p["norm_T"] for p in pts])
    p16_sum_direct = fro([p["norm_combined"] for p in pts])
    cross_check = abs(p16_T_direct - p16_T_from_frac) / p16_T_from_frac

    # ---- the component doses actually present inside TH16s --------------------------------
    t_content = fro([p["s"] * p["norm_T"] for p in pts])
    s_global = p16_T_direct / p16_sum_direct
    # H's per-tensor norms are not in this receipt; its total is, so the H dose is reported
    # under the explicit approximation that s varies little (min 0.517, max 0.750).
    h_content_approx = s_global * p16_H_from_frac

    # ---- how orthogonal are they really? --------------------------------------------------
    # ||T + H||^2 = ||T||^2 + ||H||^2 + 2<T,H>
    inner = (p16_sum_direct ** 2 - p16_T_direct ** 2 - p16_H_from_frac ** 2) / 2.0
    cosine = inner / (p16_T_direct * p16_H_from_frac)

    st = th16s["norm_bookkeeping"]["s_t"]
    out = {
        "record": "NORM_AUDIT",
        "class": "AUDIT -- zero model calls, zero training; recomputed from existing receipts",
        "requested_by": "peer session (composition lane)",
        "bookkeeping_verified": {
            "T16_total_fro": th16s["norm_bookkeeping"]["T16_total_fro"],
            "arm_update_total_fro": th16s["norm_bookkeeping"]["arm_update_total_fro"],
            "ratio_arm_over_T16": th16s["norm_bookkeeping"]["ratio_arm_over_T16"],
            "n_tensors": st["n_tensors"], "n_fallback_s_eq_1": st["n_fallback_s_eq_1"],
            "verdict": ("EXACT -- the TOTAL norm claim is correct to 10 significant figures "
                        "and the builder refuses to write on drift"),
        },
        "norms": {
            "full_delta_T_fro": full_T,
            "full_delta_H_fro": full_H,
            "p16_delta_T_fro": p16_T_direct,
            "p16_delta_H_fro": p16_H_from_frac,
            "p16_sum_fro": p16_sum_direct,
            "cross_check_two_receipt_routes_rel_err": cross_check,
            "note_full_vs_rank16": ("full ||dH|| exceeds full ||dT|| by "
                                    f"{(full_H / full_T - 1) * 100:.1f}% but after rank-16 "
                                    f"truncation they differ by only "
                                    f"{(p16_H_from_frac / p16_T_direct - 1) * 100:.1f}% -- "
                                    "comparing the FULL norms would overstate the asymmetry"),
        },
        "alignment": {
            "implied_inner_product": inner,
            "implied_cosine_T_H": cosine,
            "recorded_subspace_overlap_mean": th16["subspace_overlap_rank16_T_vs_H"]["mean"],
            "reading": ("cosine near zero means the two rank-16 updates are near-orthogonal AS "
                        "VECTORS, which is what makes s ~ 1/sqrt(2) rather than ~ 1/2 and is "
                        "exactly why rescaling ATTENUATES here where it AVERAGES for same-skill "
                        "pairs"),
        },
        "component_doses_inside_TH16s": {
            "s_t_min": st["min"], "s_t_median": st["median"], "s_t_max": st["max"],
            "s_global_T_over_sum": s_global,
            "T_content_fro": t_content,
            "T_content_as_fraction_of_solo_T16": t_content / p16_T_direct,
            "H_content_fro_approx": h_content_approx,
            "H_content_as_fraction_of_solo_H16_approx": h_content_approx / p16_H_from_frac,
            "reading": ("TH16s carries each skill at roughly two thirds of its solo strength. "
                        "The TOTAL is matched to T16; neither COMPONENT is."),
        },
        "the_confound": {
            "reported_acquisition": {"T16_solo": 235, "TH16s": 219, "difference": -16},
            "counterfactual_used": "T16 at FULL rank-16 strength",
            "counterfactual_required": ("theta_supa + s * P16(delta_T), no delta_H -- T16 at "
                                        "TH16s's own T-content norm"),
            "arm_exists": False,
            "why_TT001_does_not_cover_it": (
                "TT-001 summed two SAME-SKILL deltas under the identical formula. Aligned "
                "deltas give ||d1+d2|| ~ 2||d||, so s ~ 0.5 and the rescaled sum is their "
                "AVERAGE -- full effective strength along the shared direction with seed noise "
                "reduced, which is why it scored ABOVE T16. Near-orthogonal cross-skill deltas "
                "give s ~ 0.69 and ATTENUATE each component with no shared direction to average "
                "along. Same line of code, structurally different operation."),
            "independent_size_estimate_from_the_alpha_ladder": (
                "On INT-001's acquisition ladder the full delta at alpha 0.5 scores 219 and at "
                "0.75 scores 234, so a dose of ~0.68 interpolates to ~230 against 235 at full "
                "dose: roughly a third of the 16-row loss is attenuation rather than conflict. "
                "This is an ESTIMATE ACROSS PARAMETERISATIONS (alpha scales the full delta; "
                "these arms are rank-16) and is offered to size the effect, not to settle it. "
                "The one arm above settles it."),
        },
        "caveats": [
            "H's per-tensor norms are absent from this receipt, so the H dose uses a global s; "
            "s ranges 0.517 to 0.750 so that is an approximation, while the T dose is exact",
            "an audit of bookkeeping is not a measurement -- no model was run",
            "nothing here retracts the interference result, the subspace overlap, the "
            "specificity restoration or the sensitivity numbers; it bounds one estimate's size",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    n = out["norms"]
    c = out["component_doses_inside_TH16s"]
    print("TOTAL NORM BOOKKEEPING")
    print(f"  ratio arm/T16      {th16s['norm_bookkeeping']['ratio_arm_over_T16']:.10f}   EXACT")
    print(f"  tensors {st['n_tensors']}, fallbacks {st['n_fallback_s_eq_1']}")
    print("\nNORMS")
    print(f"  full  ||dT|| {n['full_delta_T_fro']:.4f}   full  ||dH|| {n['full_delta_H_fro']:.4f}"
          f"   (H is +{(full_H/full_T-1)*100:.1f}%)")
    print(f"  P16   ||dT|| {n['p16_delta_T_fro']:.4f}   P16   ||dH|| {n['p16_delta_H_fro']:.4f}"
          f"   (H is +{(p16_H_from_frac/p16_T_direct-1)*100:.1f}%)")
    print(f"  P16   ||dT+dH|| {n['p16_sum_fro']:.4f}")
    print(f"  two-receipt cross-check rel err {cross_check:.2e}")
    print(f"\nALIGNMENT  cosine(dT,dH) = {cosine:.4f}   "
          f"(recorded subspace overlap mean {th16['subspace_overlap_rank16_T_vs_H']['mean']:.4f})")
    print("\nCOMPONENT DOSES INSIDE TH16s")
    print(f"  s_t  min {st['min']:.4f}  median {st['median']:.4f}  max {st['max']:.4f}")
    print(f"  T content {c['T_content_fro']:.4f} = "
          f"{c['T_content_as_fraction_of_solo_T16']*100:.1f}% of solo T16   (exact)")
    print(f"  H content {c['H_content_fro_approx']:.4f} ~ "
          f"{c['H_content_as_fraction_of_solo_H16_approx']*100:.1f}% of solo H16   (global s)")
    print("\nTHE MISSING ARM: theta_supa + s * P16(delta_T), no delta_H")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
