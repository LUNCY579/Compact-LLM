"""Phase 3 analysis: positional fingerprints with clustered bootstrap CIs.

Reads the parquet produced by ``phase3_baseline_fingerprint.py`` and answers brief
questions 1-3. Pure CPU; no model is loaded, so this is cheap to re-run.

Two things this script is careful about, because getting either wrong silently
produces a plausible table:

* **The position axis is the DECISIVE evidence sentence.** ``multi_hop`` and
  ``state_constraint`` carry several evidence sentences; the non-decisive ones are
  pinned by construction and only the decisive one is swept. Using
  ``target_positions[0]`` collapses ``state_constraint`` onto the single constant 0.25.
* **The NON-decisive evidence is held fixed too.** "Pinned by construction" is true of
  ``state_constraint`` but *false* of ``multi_hop``: the ``multi_hop_pairs`` grid sweeps
  the six start/middle/end combinations of *both* hops, so grouping on the decisive hop
  alone builds a triangular design -- p=0.02 contains only (0.02, 0.02), p=0.50 contains
  (0.02, 0.50) and (0.50, 0.50), p=0.98 contains all three. The cells then differ in
  *content*, not only in the position of the sentence under study, and the nuisance
  factor is enormous (co-locating the two hops is worth 10-13 nats). See
  ``restrict_position_axis`` and ``reports/multi_hop_axis_investigation.md``.
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


def nondecisive_key(target_positions: Sequence[float], primary_target: float) -> str:
    """The sorted multiset of the NON-decisive evidence positions, as a hashable key.

    One occurrence of the decisive position is removed from ``target_positions``; every
    generator in this lab guarantees ``target_positions[decisive_index] ==
    primary_target``, so which occurrence is removed cannot matter once the remainder is
    sorted. That means the key needs no ``decisive_index`` lookup against the bank and is
    correct even when two evidence sentences sit at the same anchor -- e.g. multi_hop's
    (0.02, 0.02), where naively dropping "the one that is not the primary" is ambiguous.
    """
    rest = [round(float(t), 2) for t in target_positions]
    rest.remove(round(float(primary_target), 2))
    return ",".join(f"{t:.2f}" for t in sorted(rest))


def restrict_position_axis(sub: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep only rows whose non-decisive evidence sits where every position cell has it.

    A positional curve is only interpretable if the cells differ *in the position of the
    decisive sentence and nothing else*. ``multi_hop`` violates that: its grid sweeps
    both hops, so the p=0.98 cell contains configurations (both hops at the end, i.e.
    adjacent) that the p=0.02 cell cannot contain. Averaging those in makes the far edge
    look artificially easy and inflates the middle-minus-edge gap.

    The fix is a within-design restriction, not a reweighting: take the non-decisive
    configuration that is present at *every* level of ``primary_target`` (ties broken by
    row count, then by key for determinism) and drop the rest. For single-evidence
    families the key is the empty string and nothing is dropped; for
    ``state_constraint``, whose non-decisive updates really are pinned to constant
    anchors, the key is constant and nothing is dropped. Only ``multi_hop`` loses rows.

    Returns ``(restricted_frame, info)``; ``info["axis_restricted"]`` says whether it bit.
    """
    keys = [
        nondecisive_key(tp, pt)
        for tp, pt in zip(sub.target_positions, sub.primary_target, strict=True)
    ]
    work = sub.assign(_nd=keys)
    levels = set(work.primary_target)
    shared = [k for k, g in work.groupby("_nd") if set(g.primary_target) == levels]
    info: dict = {
        "nondecisive_keys": sorted(work._nd.unique().tolist()),
        "n_before_restriction": int(len(sub)),
    }
    if not shared or len(work._nd.unique()) <= 1:
        # Nothing varies (single-evidence / genuinely pinned families), or no single
        # configuration spans the axis -- in the latter case there is no comparison to
        # rescue, so report the unrestricted frame and flag it loudly.
        info.update(
            axis_restricted=False,
            nondecisive_pinned_at=info["nondecisive_keys"][0] if not work.empty else "",
            axis_confounded=bool(len(work._nd.unique()) > 1),
            n_dropped=0,
        )
        return sub, info
    best = max(shared, key=lambda k: (int((work._nd == k).sum()), k))
    kept = sub[work._nd.values == best]
    info.update(
        axis_restricted=bool(len(kept) < len(sub)),
        nondecisive_pinned_at=best,
        axis_confounded=bool(len(work._nd.unique()) > 1),
        n_dropped=int(len(sub) - len(kept)),
    )
    return kept, info


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
        # The published curve is computed on the restricted axis: decisive sentence
        # swept, every other evidence sentence held where it is in all three cells.
        # The unrestricted curve is kept beside it under `|all_pairs` so the size of the
        # confound stays auditable instead of vanishing from the record.
        kept, info = restrict_position_axis(sub)
        stats = curve_stats(kept, args.value)
        stats.update(info)
        out["curves"][f"{arm}|{grp}"] = stats
        if info["axis_restricted"]:
            unrestricted = curve_stats(sub, args.value)
            unrestricted.update(axis_restricted=False, axis_confounded=True)
            out["curves"][f"{arm}|{grp}|all_pairs"] = unrestricted

    # Paired against the base model, on identical items. This block deliberately does
    # NOT apply `restrict_position_axis`: it is a between-model contrast at matched
    # items, so the multi_hop design imbalance cancels within each pair. Restricting
    # would only change which items the average is over, at a cost in precision.
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
            tag = " [axis restricted]" if v.get("axis_restricted") else ""
            print(
                f"{k:48s} mid-edge {v['middle_minus_edge']:+8.3f} "
                f"[{lo:+7.3f},{hi:+7.3f}]{sig} "
                f"worst@{v['worst_position']:.2f} range {v['range']:7.3f}{tag}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
