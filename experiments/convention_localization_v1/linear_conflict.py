"""CONVENTION-LOC-001, step 0b (exploratory, zero forwards): is the imported period a LINEAR-LOGIT
conflict between two additive behaviors?

Data: FA-001's exact teacher-forced record (functional_additivity_v1/amendment_1/TF_EXACT.json),
arms BASE, SPARSE, OPALLO, ADD-O75, ADD-100 on the exposed V2 bank (192 rows).

At the convention site the model chooses between the period variant and the gold continuation:
  merged rows   (128, JSON answers): '}.' vs '}' at the final gold position
  integer rows  (64, bare numbers) : '.'  vs EOS at the EOS position
Both are two tokens at ONE position, so  m = log p(period) - log p(gold) = z_period - z_gold  is an
exact LOGIT difference; the softmax normaliser cancels. m > 0 <=> the period wins under greedy.

Tests (all computed here; nothing is fitted to ADD-O75):
 T1 additivity of the margin across donors:  I = m(ADD-100) - m(SPARSE) - m(OPALLO) + m(BASE)
 T2 linearity in beta, OUT OF SAMPLE: predict ADD-O75 (alpha=1, beta=0.75) from SPARSE (beta=0) and
    ADD-100 (beta=1) alone,  m_hat = 0.25*m(SPARSE) + 0.75*m(ADD-100);  compare per row with the
    measured m(ADD-O75), and the predicted win count with the measured one (68) and with the recorded
    GENERATION count of trailing periods (63).
 T3 the curve the linear model implies: predicted period count at beta in [0, 1], and the beta at
    which it falls to <= 2% of rows. A PREDICTION for arms not yet built, stated before they exist.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TF = HERE.parent / "functional_additivity_v1" / "amendment_1" / "TF_EXACT.json"
GEN = HERE.parent / "fresh_dual_skill_instrument_v1" / "V2_ARMS.json"


def margins(rows):
    m, kind = [], []
    for r in rows:
        if r["merged_id"] is not None:
            m.append(r["final_pos_logprob_merged"] - r["logprob"][-1]); kind.append("merged")
        else:
            m.append(r["eos_pos_logprob_period"] - r["eos_logprob"]); kind.append("integer")
    return np.array(m), np.array(kind)


def main():
    d = json.loads(TF.read_text(encoding="utf-8"))
    arms = {a["arm"]: {r["id"]: r for r in a["rows"]} for a in d["arms"]}
    ids = sorted(arms["SPARSE"])
    assert all(sorted(arms[a]) == ids for a in arms)
    M, K = {}, None
    for a in arms:
        M[a], K = margins([arms[a][i] for i in ids])
    gen = {a["arm"]: {p["id"]: p["generation"] for p in a["predictions"]}
           for a in json.loads(GEN.read_text(encoding="utf-8"))["arms"]}
    gen_periods = {a: sum(gen[a][i].strip().endswith(".") for i in ids) for a in ("ADD-O75", "ADD-100")}
    out = {"source": str(TF.name), "n": len(ids), "wins_measured": {a: int((M[a] > 0).sum()) for a in M},
           "generation_trailing_periods": gen_periods}

    I = M["ADD-100"] - M["SPARSE"] - M["OPALLO"] + M["BASE"]
    dO_alone = M["OPALLO"] - M["BASE"]; dO_on_S = M["ADD-100"] - M["SPARSE"]
    rng = np.random.default_rng(0)
    boot = [I[rng.integers(0, len(I), len(I))].mean() for _ in range(10000)]
    out["T1_additivity"] = {"mean_I": float(I.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                            "mean_dOpallo_effect_on_BASE": float(dO_alone.mean()),
                            "mean_dOpallo_effect_on_SPARSE": float(dO_on_S.mean()),
                            "interaction_as_fraction_of_effect": float(I.mean() / dO_alone.mean()),
                            "corr_effect_on_BASE_vs_on_SPARSE": float(np.corrcoef(dO_alone, dO_on_S)[0, 1])}

    hat = 0.25 * M["SPARSE"] + 0.75 * M["ADD-100"]
    obs = M["ADD-O75"]
    agree = int(((hat > 0) == (obs > 0)).sum())
    out["T2_linearity_out_of_sample"] = {
        "predicted_wins_beta_0.75": int((hat > 0).sum()), "measured_TF_wins_ADD-O75": int((obs > 0).sum()),
        "generation_periods_ADD-O75": gen_periods["ADD-O75"],
        "per_row_sign_agreement": agree, "per_row_corr": float(np.corrcoef(hat, obs)[0, 1]),
        "mean_residual_obs_minus_hat": float((obs - hat).mean()),
        "curvature_obs_over_linear": float(((obs - M["SPARSE"]).mean()) / ((hat - M["SPARSE"]).mean())),
        "by_kind": {k: {"predicted": int((hat[K == k] > 0).sum()), "measured": int((obs[K == k] > 0).sum()), "n": int((K == k).sum())}
                    for k in ("merged", "integer")}}

    curve = {}
    for b in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        mb = (1 - b) * M["SPARSE"] + b * M["ADD-100"]
        curve[f"{b:.2f}"] = int((mb > 0).sum())
    out["T3_predicted_period_rows_by_beta"] = {"linear": curve,
        "largest_beta_with_at_most_2pct_period_rows": min((float(b) for b, v in curve.items() if v <= 0.02 * len(ids)), default=None, key=lambda x: -x)}
    (HERE / "LINEAR_CONFLICT.json").write_bytes((json.dumps(out, indent=1) + "\n").encode())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
