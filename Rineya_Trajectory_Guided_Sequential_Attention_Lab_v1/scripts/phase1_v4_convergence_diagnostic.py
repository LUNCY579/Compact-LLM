"""Distinguish a truncated training phase transition from a stochastic one.

The v4 pilot found a regime whose seeds were bimodal, and diagnosed it as an
update budget that cut a phase transition short — the two low runs were still
descending at the final step. That diagnosis makes a prediction: **more updates
should convert non-transitioned runs into transitioned ones.**

The competing explanation is that whether a seed transitions at all is
stochastic, settled early, and not a matter of budget. It makes the opposite
prediction: the transitioned fraction stays roughly constant as the budget grows.

The two demand different responses, and only one of them is on the registered
escalation ladder, so telling them apart matters more than either result.

This script classifies each run from its recorded loss curve and reports the
transitioned fraction per budget. It reports every arm because the gate is
arm-blind and needs all of them; **it must not be used to compare arms.** A
difference in transition reliability between arms would be a between-arm
difference on development data, which the registered selection rule forbids
using and which this program has already been burned by once.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

#: A run counts as transitioned if its final loss sits in the low cluster. The
#: split is found in the data rather than fixed, because the loss scale depends
#: on the band and the budget.
MINIMUM_SEPARATION = 0.30


def split_point(losses: list[float]) -> float | None:
    """Largest gap between consecutive sorted losses, if it is a real gap.

    A unimodal population has no large gap and returns None, which is itself the
    answer: there is nothing to classify.
    """

    if len(losses) < 3:
        return None
    ordered = sorted(losses)
    gaps = [(ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1)]
    widest, index = max(gaps)
    if widest < MINIMUM_SEPARATION:
        return None
    return (ordered[index] + ordered[index + 1]) / 2.0


def still_descending(curve: list[list[float]], *, tail_fraction: float = 0.2) -> bool:
    """Was the run still improving when the schedule ended?

    This is the discriminator. A truncated run is cut off mid-descent; a run that
    never transitioned has plateaued.
    """

    if len(curve) < 4:
        return False
    tail = max(2, int(len(curve) * tail_fraction))
    early = statistics.fmean(point[1] for point in curve[-tail:-tail // 2 or -1])
    late = statistics.fmean(point[1] for point in curve[-tail // 2 or -1:])
    total = curve[0][1] - min(point[1] for point in curve)
    if total <= 0:
        return False
    return (early - late) / total > 0.02


def describe(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    runs = report.get("runs") or []
    regime = report.get("regime", {})
    losses = [run.get("final_train_loss") for run in runs]
    if any(value is None for value in losses):
        raise ValueError(f"{path.name} has runs without final_train_loss")

    cut = split_point(losses)
    rows = []
    for run in runs:
        loss = run["final_train_loss"]
        rows.append({
            "arm": run["arm"], "seed": run["seed"],
            "holdout": run["holdout"]["accuracy"],
            "final_loss": loss,
            "transitioned": None if cut is None else loss < cut,
            "still_descending": still_descending(run.get("loss_curve") or []),
        })
    transitioned = [r for r in rows if r["transitioned"]]
    stalled = [r for r in rows if r["transitioned"] is False]
    return {
        "report": path.name,
        "train_examples": regime.get("train_examples"),
        "steps": regime.get("steps"),
        "split_point": cut,
        "runs": rows,
        "transitioned_fraction": None if cut is None else len(transitioned) / len(rows),
        "stalled_still_descending": sum(1 for r in stalled if r["still_descending"]),
        "stalled_plateaued": sum(1 for r in stalled if not r["still_descending"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()

    summaries = [describe(path) for path in args.reports]
    for summary in summaries:
        print(f"\n=== {summary['report']}  "
              f"{summary['train_examples']} ex x {summary['steps']} steps ===")
        if summary["split_point"] is None:
            print("  no bimodal split in final loss: the population is unimodal")
        else:
            print(f"  loss split at {summary['split_point']:.3f}; "
                  f"transitioned {summary['transitioned_fraction']:.0%}")
        for row in sorted(summary["runs"], key=lambda r: (r["arm"], r["seed"])):
            mark = {True: "transitioned", False: "stalled", None: "-"}[row["transitioned"]]
            tail = " (still descending)" if row["still_descending"] else ""
            print(f"    {row['arm']:16s} seed={row['seed']} "
                  f"loss={row['final_loss']:6.3f} holdout={row['holdout']:.4f}  {mark}{tail}")
        if summary["split_point"] is not None:
            print(f"  stalled runs: {summary['stalled_still_descending']} still descending, "
                  f"{summary['stalled_plateaued']} plateaued")

    graded = [s for s in summaries if s["transitioned_fraction"] is not None]
    if len(graded) >= 2:
        print("\n=== budget response ===")
        for summary in sorted(graded, key=lambda s: (s["train_examples"] or 0, s["steps"] or 0)):
            print(f"  {summary['train_examples']:>7} ex x {summary['steps']:>6} steps: "
                  f"{summary['transitioned_fraction']:.0%} transitioned")
        print("\n  A RISING fraction with budget supports truncation, and the registered")
        print("  response is more updates. A FLAT fraction supports a stochastic")
        print("  transition, which more updates will not fix -- and which is not on the")
        print("  escalation ladder, so it rejects the regime family.")
    print("\nReported for every arm because the gate is arm-blind. Do NOT read a")
    print("between-arm difference in transition reliability as a mechanism result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
