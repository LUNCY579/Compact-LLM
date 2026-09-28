"""INTERFERENCE-MECH-001 step 0 (exploratory, zero forwards): is composition interference a
knife-edge flip of low-margin rows?

Motivation: CONFIRM-001 found real interference (ADD-100 loses 56 / gains 22 of SPARSE's rows),
all content errors, and 39 of the 56 lost rows are also lost by AVG. If the second donor adds a
roughly unstructured perturbation to the content logits, the rows that flip should be the rows
where SPARSE's answer is closest to its decision boundary, in BOTH directions.

Data: FA-001's exact teacher-forced record on the exposed V2 bank (192 rows; TF_EXACT.json), plus
the recorded V2 generations. Margin of a row under an arm = the minimum over the answer's gold
positions of the gold log-odds, made tolerant of the one trailing period the primary rule accepts
(see tolerant_margin). Low margin = the weakest position is nearly lost.

Outputs (MARGIN_FRAGILITY.json):
  1. AUC with which a LOW SPARSE margin predicts that ADD-100 (and AVG) LOSES a SPARSE-correct row,
     with a one-sided permutation p;
  2. the same for GAINED rows (SPARSE wrong, ADD-100 right), which should be SPARSE's near-misses;
  3. the per-row shift ADD-100 minus SPARSE of the weakest-token log-odds, as mean and SD (a noise
     scale), and whether a zero-mean Gaussian perturbation with that SD reproduces the observed
     lost/gained counts from SPARSE's margins alone.
Exploratory: an exposed bank, and nothing here was registered.
"""
from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TF = EXP / "functional_additivity_v1" / "amendment_1" / "TF_EXACT.json"
FDS = EXP / "fresh_dual_skill_instrument_v1"


def prim(g, r):
    s = g.strip()
    return (s[:-1] if s.endswith(".") else s) == r


def auc_low(score, y):
    """P(a positive row's score < a negative row's score); ties count 1/2."""
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(p < neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def perm_p(score, y, obs, rng, n=5000):
    return float(np.mean([auc_low(score, rng.permutation(y)) >= obs for _ in range(n)]))


def logodds_of(lp_sum_terms):
    """log-odds of an event whose probability is the sum of exp(log-probs)."""
    p = min(sum(math.exp(x) for x in lp_sum_terms), 1 - 1e-12)
    return math.log(p) - math.log1p(-p)


def tolerant_margin(r):
    """Weakest-position gold log-odds under the PRIMARY rule's tolerance of one trailing period.

    JSON rows: at the final position the merged token '}.' is as acceptable as '}', so that position
    counts P('}') + P('}.'). Integer rows: at the end position, EOS and '.' are both acceptable, and
    that position is included, because a wrong extra digit there is an error. A first version of this
    script used the raw gold-only log-odds. That charged ADD-100 for the period the primary rule
    accepts: its margin sign agreed with generation on only 121/192 rows.
    """
    z = list(r["z_unclipped"])
    if r["merged_id"] is not None:
        z[-1] = logodds_of([r["logprob"][-1], r["final_pos_logprob_merged"]])
    else:
        z.append(logodds_of([r["eos_logprob"], r["eos_pos_logprob_period"]]))
    return min(z)


def main():
    tf = json.loads(TF.read_text(encoding="utf-8"))
    A = {a["arm"]: {r["id"]: r for r in a["rows"]} for a in tf["arms"]}
    bank = {r["id"]: r for r in json.loads((FDS / "T_BANK_V2.json").read_text(encoding="utf-8"))["rows"]}
    gen = {}
    for f in ("CALIBRATION_V2.json", "V2_ARMS.json"):
        for a in json.loads((FDS / f).read_text(encoding="utf-8"))["arms"]:
            gen[a["arm"]] = {p["id"]: p["generation"] for p in a["predictions"]}
    ids = sorted(A["SPARSE"])
    ok = {a: np.array([prim(gen[a][i], bank[i]["reference"]) for i in ids]) for a in ("SPARSE", "ADD-100", "AVG")}
    mS = np.array([tolerant_margin(A["SPARSE"][i]) for i in ids])
    mA = np.array([tolerant_margin(A["ADD-100"][i]) for i in ids])
    rng = np.random.default_rng(0)
    S, D, V = ok["SPARSE"], ok["ADD-100"], ok["AVG"]
    out = {"bank": "T_BANK_V2 (exposed)", "n": len(ids), "SPARSE_correct": int(S.sum())}

    lost_add, lost_avg = S & ~D, S & ~V
    for name, lost in (("ADD-100", lost_add), ("AVG", lost_avg)):
        a = auc_low(mS[S], lost[S])
        out[f"loss_{name}"] = {"n_lost": int(lost.sum()), "auc_low_margin_predicts_loss": a,
                               "perm_p": perm_p(mS[S], lost[S], a, rng),
                               "median_margin_lost": float(np.median(mS[lost])),
                               "median_margin_kept": float(np.median(mS[S & ~lost]))}
    gained = ~S & D
    # among SPARSE-wrong rows, a HIGH (closer to 0 from below / higher) SPARSE margin should predict a gain
    a_g = auc_low(-mS[~S], gained[~S])
    out["gain_ADD-100"] = {"n_gained": int(gained.sum()), "auc_high_margin_predicts_gain": a_g,
                           "perm_p": perm_p(-mS[~S], gained[~S], a_g, rng),
                           "median_margin_gained": float(np.median(mS[gained])) if gained.any() else None,
                           "median_margin_other_wrong": float(np.median(mS[~S & ~gained]))}
    both = lost_add & lost_avg
    out["overlap_lost"] = {"ADD_lost": int(lost_add.sum()), "AVG_lost": int(lost_avg.sum()), "both": int(both.sum())}

    shift = mA - mS
    out["perturbation"] = {"mean_shift_weakest_token_logodds": float(shift.mean()), "sd_shift": float(shift.std(ddof=1)),
                           "sd_shift_on_SPARSE_correct": float(shift[S].std(ddof=1)),
                           "corr_shift_vs_SPARSE_margin": float(np.corrcoef(shift, mS)[0, 1])}
    # zero-knowledge flip model: ADD margin ~ SPARSE margin + N(mu, sd); correctness ~ margin > 0
    mu, sd = float(shift.mean()), float(shift.std(ddof=1))
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2))
    pred_lost = float(sum(phi(-(m + mu) / sd) for m in mS[S]))
    pred_gain = float(sum(phi((m + mu) / sd) for m in mS[~S]))
    out["gaussian_flip_model"] = {
        "note": "uses the observed shift distribution; a consistency check, not an out-of-sample prediction",
        "predicted_lost": pred_lost, "observed_lost": int(lost_add.sum()),
        "predicted_gained": pred_gain, "observed_gained": int(gained.sum()),
        "margin_sign_agrees_with_generation_SPARSE": int(((mS > 0) == S).sum()),
        "margin_sign_agrees_with_generation_ADD100": int(((mA > 0) == D).sum())}
    (HERE / "MARGIN_FRAGILITY.json").write_bytes((json.dumps(out, indent=1) + "\n").encode())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
