"""Phase 3 analysis: positional fingerprints with clustered bootstrap CIs.

Reads the parquet produced by ``phase3_baseline_fingerprint.py`` and answers brief
questions 1-3. Pure CPU; no model is loaded, so this is cheap to re-run.

Two things this script is careful about, because getting either wrong silently
produces a plausible table:

* **The position axis is the DECISIVE evidence sentence.** ``multi_hop`` and
  ``state_constraint`` carry several evidence sentences; the non-decisive ones are
  pinned by construction and only the decisive one is swept. Using
  ``target_positions[0]`` collapses ``state_constraint`` onto the single constant 0.25.
* **Local rhetorical position is held fixed.** The ``local_position_sweep`` grid only
  emits p in {0.02, 0.50, 0.98}, so pooling local positions puts a step at exactly the
  three positions that define the edge-middle gap.

    PYTHONPATH=src python scripts/phase3_analyze.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rineya_psa import paths  # noqa: E402
from rineya_psa.evaluation import metrics  # noqa: E402
from rineya_psa.statistics.bootstrap import (  # noqa: E402
    bootstrap_ci_of_statistic,
    paired_bootstrap,
)
from rineya_psa.statistics.provenance import write_result  # noqa: E402

BASE_ARM = "qwen3"


def load(parquet: Path, bank: Path) -> pd.DataFrame:
    d = pd.read_parquet(parquet)
    if "primary_target" not in d.columns:
        rows = {
            json.loads(line)["item_id"]: json.loads(line)
            for line in bank.open(encoding="utf-8")
        }
        d["primary_target"] = d.item_id.map(
            lambda i: round(float(rows[i]["meta"]["primary_target"]), 2)
        )
    d["primary_target"] = d.primary_target.round(2)
    return d


def group_label(row: pd.Series) -> str:
    if row.family == "multi_hop":
        return "multi_hop_guided" if row.guided else "multi_hop_hard"
    return str(row.family)


def curve_stats(sub: pd.DataFrame, value: str) -> dict:
    """Positional curve plus the brief's section-10 summary statistics, with CIs."""
    by_pos = sub.groupby("primary_target")[value].mean().to_dict()
    n_by_pos = sub.groupby("primary_target").size().to_dict()
    if len(by_pos) < 3:
        return {"by_pos": by_pos, "n_by_pos": n_by_pos, "note": "fewer than 3 positions"}

    positions = sorted(by_pos)
    lo, hi = positions[0], positions[-1]
    mid = min(positions, key=lambda p: abs(p - 0.5))

    def gap_of(rows: Sequence[tuple]) -> float:
        # rows are (position, value); NLL is a loss, so "edge minus middle" is negated
        # to keep the convention "positive gap means the middle is worse".
        acc: dict[float, list[float]] = {}
        for p, v in rows:
            acc.setdefault(p, []).append(v)
        if lo not in acc or hi not in acc or mid not in acc:
            return float("nan")
        edge = (np.mean(acc[lo]) + np.mean(acc[hi])) / 2.0
        return float(np.mean(acc[mid]) - edge)

    rows = list(zip(sub.primary_target.tolist(), sub[value].tolist(), strict=True))
    gap = bootstrap_ci_of_statistic(gap_of, rows, cluster_ids=sub.scenario_id.tolist())

    fit = metrics.ushape_fit(sub.primary_target.tolist(), sub[value].tolist())
    worst_at = max(by_pos, key=by_pos.get)
    return {
        "by_pos": {str(k): round(v, 4) for k, v in by_pos.items()},
        "n_by_pos": {str(k): int(v) for k, v in n_by_pos.items()},
        "middle_minus_edge": round(gap.estimate, 4),
        "middle_minus_edge_ci": [round(gap.ci_low, 4), round(gap.ci_high, 4)],
        "middle_minus_edge_significant": bool(gap.ci_low > 0 or gap.ci_high < 0),
        "worst_position": worst_at,
        "worst_value": round(by_pos[worst_at], 4),
        "best_position": min(by_pos, key=by_pos.get),
        "range": round(max(by_pos.values()) - min(by_pos.values()), 4),
        "ushape": {"beta0": round(fit.beta0, 4), "beta1": round(fit.beta1, 4),
                   "beta2": round(fit.beta2, 4), "se1": round(fit.se1, 4),
                   "se2": round(fit.se2, 4), "r_squared": round(fit.r_squared, 4)},
        "n": int(len(sub)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=str(
        paths.RESULTS / "baseline_position_results_calibration_tf.parquet"))
    ap.add_argument("--bank", default=str(paths.DATA / "calibration" / "position_examples.jsonl"))
    ap.add_argument("--value", default="gold_nll")
    ap.add_argument("--fmt", default="natural",
                    help="surface format to hold fixed; format is a separate factor")
    ap.add_argument("--out", default=str(paths.RESULTS / "phase3_positional_analysis.json"))
    args = ap.parse_args(argv)

    d = load(Path(args.parquet), Path(args.bank))
    # Hold BOTH nuisance factors fixed, not just one. The local_position_sweep and
    # format_sweep grids each emit only p in {0.02, 0.50, 0.98}, so leaving either free
    # loads those three positions with extra conditions -- and those are exactly the three
    # the edge-middle gap is computed from. An earlier run filtered local_position and not
    # fmt, which put 18 non-natural items on each edge/middle cell and none on the others.
    d = d[(d.local_position == "middle") & (d.fmt == args.fmt)].copy()
    d["group"] = d.apply(group_label, axis=1)
    d = d[d[args.value].notna()]

    out: dict = {"value": args.value, "n_rows": int(len(d)),
                 "arms": sorted(d.arm.unique()), "groups": sorted(d.group.unique()),
                 "curves": {}, "vs_base": {}}

    for (arm, grp), sub in d.groupby(["arm", "group"]):
        out["curves"][f"{arm}|{grp}"] = curve_stats(sub, args.value)

    # Paired against the base model, on identical items.
    base = d[d.arm == BASE_ARM].set_index("item_id")
    for arm in sorted(a for a in d.arm.unique() if a != BASE_ARM):
        other = d[d.arm == arm].set_index("item_id")
        shared = base.index.intersection(other.index)
        for grp in sorted(d.group.unique()):
            bm = base.loc[shared]
            om = other.loc[shared]
            sel = bm.group == grp
            if sel.sum() < 8:
                continue
            # paired_bootstrap(a, b) returns b - a. Passing (base, other) therefore
            # gives other - base, which is what "mean_diff_vs_qwen3" means. Passing
            # them the other way round silently inverts every sign and every
            # better/worse verdict.
            res = paired_bootstrap(
                bm.loc[sel, args.value].tolist(),
                om.loc[sel, args.value].tolist(),
                bm.loc[sel, "scenario_id"].tolist(),
            )
            out["vs_base"][f"{arm}|{grp}"] = {
                "mean_diff_vs_qwen3": round(res.estimate, 4),
                "ci": [round(res.ci_low, 4), round(res.ci_high, 4)],
                "cohens_dz": round(res.cohens_dz, 4),
                "n_scenarios": res.n_scenarios, "n_items": res.n_items,
                "better_than_base": bool(res.ci_high < 0),  # NLL: lower is better
            }

    write_result(out, Path(args.out))
    print(json.dumps({k: out[k] for k in ("value", "n_rows", "arms", "groups")}, indent=2))
    for k, v in sorted(out["curves"].items()):
        if "by_pos" in v and "middle_minus_edge" in v:
            sig = "*" if v["middle_minus_edge_significant"] else " "
            lo, hi = v["middle_minus_edge_ci"]
            print(
                f"{k:38s} mid-edge {v['middle_minus_edge']:+8.3f} "
                f"[{lo:+7.3f},{hi:+7.3f}]{sig} "
                f"worst@{v['worst_position']:.2f} range {v['range']:7.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
