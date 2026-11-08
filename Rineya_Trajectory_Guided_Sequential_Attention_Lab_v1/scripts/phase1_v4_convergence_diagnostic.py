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

Report the budget in **epochs**, not steps. `steps x batch_size / train_examples`
is the number of times the model sees each example, and it is what the
transitioned fraction actually tracks. Two comparisons isolate it from its
components: at fixed data, more steps raises the fraction; at fixed steps, more
data lowers it. Both are the same move in epochs, in opposite directions, so a
diagnostic that reported only steps or only data volume would read the second
comparison as a contradiction of the first.

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
    half = -(tail // 2) or -1  # tail >= 2 keeps this negative
    early = statistics.fmean(point[1] for point in curve[-tail:half])
    late = statistics.fmean(point[1] for point in curve[half:])
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
            "per_family": dict(run["holdout"].get("per_family_accuracy") or {}),
            "final_loss": loss,
            "transitioned": None if cut is None else loss < cut,
            "still_descending": still_descending(run.get("loss_curve") or []),
        })
    transitioned = [r for r in rows if r["transitioned"]]
    stalled = [r for r in rows if r["transitioned"] is False]

    # What the transition consists of. If the stalled runs already solve the
    # one-hop guardrail family and only the composed families move, the
    # transition is the discovery of a composition circuit rather than general
    # competence -- which points at depth, not data or budget.
    families = sorted({f for r in rows for f in r["per_family"]})
    by_mode: dict[str, dict[str, float]] = {}
    for label, group in (("transitioned", transitioned), ("stalled", stalled)):
        if group:
            by_mode[label] = {
                family: statistics.fmean(r["per_family"][family] for r in group
                                         if family in r["per_family"])
                for family in families
                if any(family in r["per_family"] for r in group)
            }
    examples = regime.get("train_examples")
    steps = regime.get("steps")
    batch = regime.get("batch_size") or 64
    epochs = (steps * batch / examples) if (steps and examples) else None
    return {
        "per_family_by_mode": by_mode,
        "report": path.name,
        "train_examples": examples,
        "steps": steps,
        "batch_size": batch,
        "epochs": epochs,
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
        epochs = summary.get("epochs")
        print(f"\n=== {summary['report']}  "
              f"{summary['train_examples']} ex x {summary['steps']} steps"
              f"{'' if epochs is None else f'  ({epochs:.1f} epochs)'} ===")
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
        modes = summary.get("per_family_by_mode") or {}
        if len(modes) == 2:
            families = sorted(set(modes["transitioned"]) & set(modes["stalled"]))
            print(f"  {'family':18s} {'stalled':>9s} {'transitioned':>13s} {'delta':>9s}")
            for family in families:
                low, high = modes["stalled"][family], modes["transitioned"][family]
                print(f"  {family:18s} {low:9.4f} {high:13.4f} {high - low:+9.4f}")

    graded = [s for s in summaries if s["transitioned_fraction"] is not None]
    if len(graded) >= 2:
        print("\n=== budget response, ordered by epochs ===")
        for summary in sorted(graded, key=lambda s: s.get("epochs") or 0):
            epochs = summary.get("epochs")
            print(f"  {summary['train_examples']:>7} ex x {summary['steps']:>6} steps"
                  f" = {('  n/a' if epochs is None else format(epochs, '5.1f'))} epochs: "
                  f"{summary['transitioned_fraction']:.0%} transitioned "
                  f"({summary['stalled_still_descending']} of the stalled still descending)")
        print("\n  A fraction RISING WITH EPOCHS means the budget is the lever, and the")
        print("  registered response to a mixture -- more updates at fixed data -- applies.")
        print("  A FLAT fraction means the transition is stochastic, which more updates")
        print("  will not fix, and which no rung of the ladder addresses.")
        print("\n  Read epochs, not steps: raising data at fixed steps LOWERS epochs, so")
        print("  it moves the fraction the opposite way from raising steps at fixed data.")
    print("\nReported for every arm because the gate is arm-blind. Do NOT read a")
    print("between-arm difference in transition reliability as a mechanism result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
