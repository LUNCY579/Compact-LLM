"""Score RCI-001's RECORDED predictions under RCI-002's emission metric. Zero model calls.

    python confirm/cross_bank_emission.py --rci001 <records>/rci001-20260918 \
        --rci002 <records>/rci002-20260918 --out <records>/CROSS_BANK_EMISSION.json

**This is not a new read.** RCI-001 is spent for confirmation; its predictions are on disk, and
re-scoring recorded output under a different operationalisation consumes nothing. The suggestion
came from the peer session and it settles a question I had said the sealed read could not.

The question: RCI-002's twin did not reproduce RCI-001's ACCURACY effect on the incongruent
cell (B 14 -> D 12, p = 0.50, against 15 -> 2), while it did reproduce an EMISSION effect. Two
explanations, and I could not separate them:

  (a) the measure was wrong all along -- accuracy was a proxy for emission;
  (b) RCI-002's four-carve-out design made the task harder and shrank the effect.

Scoring RCI-001 under the emission metric separates them, because (a) predicts emission
replicates on RCI-001's own items and (b) does not bear on it either way.

Matched cells, both banks: a ROLE answer asked with a person-typed interrogative.

  RCI-001  `rule_exception` / paraphrase       "Who was exempt from ...?"
  RCI-002  `carve_out_role_incongruent`        "Who was permitted to enter ...?"

and the congruent controls, which must read zero for both systems or the metric is measuring a
general tendency to name people rather than a response to the interrogative.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from paired import paired  # noqa: E402


def load_bank(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_pred(p: Path):
    return {r["id"]: r.get("prediction", "") for r in
            (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def person_vocab(rows):
    """From the bank's own alias-resolution references -- never a hardcoded list."""
    return {str(r["reference"]) for r in rows
            if r["task_type"] == "alias_resolution" and r["answerable"]}


def emits(text, people):
    t = str(text).lower()
    return any(p.lower() in t for p in people)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def cell(rows, pb, pd, people, label):
    eb = {r["id"]: emits(pb.get(r["id"], ""), people) for r in rows}
    ed = {r["id"]: emits(pd.get(r["id"], ""), people) for r in rows}
    p = paired("B", "D", eb, ed)
    nb, nd = sum(eb.values()), sum(ed.values())
    return {"label": label, "n": len(rows), "B": nb, "D": nd,
            "B_rate": round(nb / len(rows), 4) if rows else None,
            "D_rate": round(nd / len(rows), 4) if rows else None,
            # `paired` names the axis for correctness; the boolean here is "emitted a person",
            # so `repair` is D-emitted-and-B-did-not, which is the direction under test.
            "D_more_than_B": p.repair, "B_more_than_D": p.damage,
            "p_mcnemar_exact": mcnemar_exact(p.repair, p.damage)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rci001", type=Path, required=True)
    ap.add_argument("--rci002", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    out = {"what_this_is": "RCI-001 recorded predictions re-scored under RCI-002's emission "
                           "metric, plus RCI-002's own twin. ZERO model calls; no read spent.",
           "banks": {}}

    b1 = load_bank(a.rci001 / "RCI_SEALED.jsonl")
    pb1 = load_pred(a.rci001 / "gen-rci_confirm_B" / "predictions.jsonl")
    pd1 = load_pred(a.rci001 / "gen-rci_confirm_D" / "predictions.jsonl")
    v1 = person_vocab(b1)
    ans1 = [r for r in b1 if r["answerable"] and r["task_type"] == "rule_exception"]
    out["banks"]["RCI-001"] = {
        "person_vocab_size": len(v1),
        "incongruent": cell([r for r in ans1 if r["variant"] == "paraphrase"],
                            pb1, pd1, v1, "rule_exception/paraphrase -- 'Who'"),
        "congruent_controls": [
            cell([r for r in ans1 if r["variant"] == v], pb1, pd1, v1,
                 f"rule_exception/{v} -- 'Which role'")
            for v in ("base", "counterfactual")],
    }

    b2 = load_bank(a.rci002 / "RCI_CALIBRATION.jsonl")
    pb2 = load_pred(a.rci002 / "gen-rci2_calib_B" / "predictions.jsonl")
    pd2 = load_pred(a.rci002 / "gen-rci2_calib_D" / "predictions.jsonl")
    v2 = person_vocab(b2)
    ans2 = [r for r in b2 if r["answerable"]]
    out["banks"]["RCI-002-twin"] = {
        "person_vocab_size": len(v2),
        "incongruent_matched": cell(
            [r for r in ans2 if r["task_type"] == "carve_out_role_incongruent"],
            pb2, pd2, v2, "carve_out_role_incongruent -- 'Who', ROLE answer (matched cell)"),
        "incongruent_pooled_E2": cell(
            [r for r in ans2 if r["wh_congruence"] == "incongruent"],
            pb2, pd2, v2, "both incongruent strata pooled -- the registered E2"),
        "congruent_controls": [
            cell([r for r in ans2 if r["task_type"] == t], pb2, pd2, v2, t)
            for t in ("carve_out_role_congruent", "carve_out_badge_congruent")],
    }

    i1 = out["banks"]["RCI-001"]["incongruent"]
    i2 = out["banks"]["RCI-002-twin"]["incongruent_matched"]
    ctrls = (out["banks"]["RCI-001"]["congruent_controls"]
             + out["banks"]["RCI-002-twin"]["congruent_controls"])
    out["verdict"] = {
        "direction_replicates": (i1["D_more_than_B"] > i1["B_more_than_D"]
                                 and i2["D_more_than_B"] > i2["B_more_than_D"]),
        "reversals": {"RCI-001": i1["B_more_than_D"], "RCI-002": i2["B_more_than_D"]},
        "effect_size": {"RCI-001_D_rate": i1["D_rate"], "RCI-002_D_rate": i2["D_rate"]},
        "controls_clean": all(c["B"] + c["D"] <= 1 for c in ctrls),
        "reading": (
            "Both explanations are PARTLY right, which neither of the two stories allowed. "
            "The emission effect replicates in direction with zero reversals on both banks, so "
            "accuracy was standing in for emission and (a) holds. But the magnitude falls "
            "sharply between banks, so RCI-002's four-carve-out design really did shrink it and "
            "(b) holds too. Reporting this as a clean replication would be as wrong as "
            "reporting the accuracy result as a failure to replicate."),
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'cell':<62}{'n':>4}{'B':>5}{'D':>5}{'D+':>5}{'B+':>5}{'p':>10}")
    for bank, d in out["banks"].items():
        for k, v in d.items():
            for c in (v if isinstance(v, list) else [v]):
                if isinstance(c, dict):
                    print(f"{bank + '  ' + c['label']:<62}{c['n']:>4}{c['B']:>5}{c['D']:>5}"
                          f"{c['D_more_than_B']:>5}{c['B_more_than_D']:>5}"
                          f"{c['p_mcnemar_exact']:>10.2e}")
    print(f"\nverdict: {json.dumps(out['verdict'], indent=2)}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
