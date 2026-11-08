"""SMALLVD-001 Phase 8 -- candidate layer sets, from five strategies that disagree on purpose.

  evenly_spaced        keep block 0 and the last block, space the rest evenly (naive baseline)
  single_ablation      remove the blocks whose INDIVIDUAL removal costs least (static ranking)
  greedy_backward      remove one block at a time, re-measuring every remaining candidate given
                       the blocks already removed (captures pairwise and group interactions
                       at O(L^2) evaluations instead of C(28, k))
  block_influence      ShortGPT-style: remove the lowest 1 - cos(h_l, h_(l+1)) -- a similarity
                       diagnostic, included precisely so it can be shown to be weaker (or not)
  angular_consecutive  Gromov-style: remove the n CONSECUTIVE blocks whose endpoints are closest
                       in angular distance, keeping the final block

The search objective is fixed in advance (SMALLVD001_PREREG.md section 4) and computed on the
calibration SELECT split only. Every strategy's layer set is then scored on the HOLD-OUT split, so
the greedy search -- which optimises the objective on SELECT -- cannot win by overfitting it.
Retained blocks always keep their original order.

AMENDMENT 1 (before any layer was scored): choosing AMONG the strategies on HOLDOUT makes HOLDOUT a
selection instrument too, and the winner's HOLDOUT figure a best-of-N estimate. `holdout_subsplit`
cuts HOLDOUT into SELECT2 (chooses among strategies and frontier configurations) and REPORT (chooses
nothing; every reported figure for a selected configuration comes from it).
"""
from __future__ import annotations

import hashlib

SUBSPLIT_SALT = "smallvd001-amendment1-report-v1"


def stratum(r: dict) -> tuple:
    """Category x answer type. The answer type separates the abstention contract's UNKNOWN rows from
    its determinate ones: stratifying by category alone put 11 UNKNOWN and only 3 determinate
    abstention rows in REPORT, too few to see over-abstention. Uses the REFERENCE label only."""
    ref = r.get("reference")
    return (r["category"], None if ref is None else ref.strip().upper() == "UNKNOWN")


def holdout_subsplit(rows: list[dict], salt: str = SUBSPLIT_SALT) -> dict[str, str]:
    """id -> 'select2' | 'report' for every HOLDOUT row, stratified by `stratum`: within a stratum,
    rows are ordered by sha256(salt + id) and alternate select2, report, select2, ... Depends on the
    ids and reference labels alone (no model output, no score), so it is fixed before any model is
    run and reproducible from the committed panel file."""
    by_cat: dict[tuple, list[str]] = {}
    for r in rows:
        if r["split"] != "holdout":
            raise ValueError(f"{r['id']} is not a HOLDOUT row")
        by_cat.setdefault(stratum(r), []).append(r["id"])
    out = {}
    for ids in by_cat.values():
        ids = sorted(ids, key=lambda i: hashlib.sha256((salt + i).encode("utf-8")).hexdigest())
        for k, i in enumerate(ids):
            out[i] = "select2" if k % 2 == 0 else "report"
    if len(out) != len(rows):
        raise ValueError("duplicate HOLDOUT ids")
    return out


def evenly_spaced(n_teacher: int, n_keep: int) -> list[int]:
    if n_keep >= n_teacher:
        return list(range(n_teacher))
    if n_keep < 2:
        raise ValueError("need at least 2 blocks")
    step = (n_teacher - 1) / (n_keep - 1)
    kept = sorted({round(i * step) for i in range(n_keep)})
    assert len(kept) == n_keep, kept
    return kept


def remove_lowest(scores: dict[int, float], n_teacher: int, n_keep: int) -> list[int]:
    """Remove the n_teacher - n_keep blocks with the LOWEST score (ties: higher index first)."""
    order = sorted(range(n_teacher), key=lambda l: (scores[l], -l))
    drop = set(order[: n_teacher - n_keep])
    return [l for l in range(n_teacher) if l not in drop]


def angular_consecutive(angular: dict, n_teacher: int, n_keep: int) -> list[int]:
    """angular[n][l] = d(h_l, h_(l+n)); removing blocks l..l+n-1 maps h_l straight to h_(l+n).
    The final block is always kept (l + n <= n_teacher - 1), as in the original method."""
    n = n_teacher - n_keep
    if n == 0:
        return list(range(n_teacher))
    row = angular[n] if n in angular else angular[str(n)]
    cands = [(row[l], l) for l in range(0, n_teacher - n) if l + n <= n_teacher - 1]
    _, l = min(cands)
    return [k for k in range(n_teacher) if not (l <= k < l + n)]


def greedy_backward(evaluate, n_teacher: int, n_min: int, log=print) -> list[dict]:
    """evaluate(kept) -> objective (lower is better). Returns the trajectory, one entry per depth.

    At each step every remaining block is tried for removal given the current set; the block
    whose removal yields the lowest objective goes. Ties break toward the higher index."""
    kept = list(range(n_teacher))
    traj = [{"n_layers": n_teacher, "kept": kept[:], "removed": None, "objective": evaluate(kept),
             "candidates": None}]
    while len(kept) > n_min:
        trials = []
        for l in kept:
            cand = [k for k in kept if k != l]
            trials.append((evaluate(cand), -l, l))
        trials.sort()
        obj, _, l = trials[0]
        kept = [k for k in kept if k != l]
        traj.append({"n_layers": len(kept), "kept": kept[:], "removed": l, "objective": obj,
                     "candidates": {str(t[2]): t[0] for t in trials}})
        log(f"[greedy] {len(kept)} layers: removed {l}, objective {obj:.5f}")
    return traj


def strategy_sets(n_teacher: int, depths, single_scores: dict, bi: dict, angular: dict,
                  greedy_traj: list[dict]) -> dict:
    by_depth = {t["n_layers"]: t["kept"] for t in greedy_traj}
    out = {}
    for n in depths:
        out[n] = {
            "evenly_spaced": evenly_spaced(n_teacher, n),
            "single_ablation": remove_lowest(single_scores, n_teacher, n),
            "greedy_backward": by_depth[n],
            "block_influence": remove_lowest(bi, n_teacher, n),
            "angular_consecutive": angular_consecutive(angular, n_teacher, n),
        }
    return out
