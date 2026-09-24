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

The search objective is fixed in advance (SMALLVD001_PREREG.md section 5) and computed on the
calibration SELECT split only. Every strategy's layer set is then scored on the HOLD-OUT split, so
the greedy search -- which optimises the objective on SELECT -- cannot win by overfitting it.
Retained blocks always keep their original order.
"""
from __future__ import annotations


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
