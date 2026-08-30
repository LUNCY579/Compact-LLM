"""The registered competence escalation ladder, made enforceable.

A gate that can be re-run until it passes is not a gate. But a gate that permits
no response at all to a diagnosable failure is not useful either — the pilot's
first `NO_GO` had an obvious mechanistic cause (an update budget that truncated
a training phase transition) and refusing to act on it would have been
superstition rather than rigour.

The line between them is this: the moves are **fixed in advance**, each is tied
to the **specific check that failed**, each may be taken **at most once**, and
the total number of competence measurements is **capped**. When the ladder is
exhausted the regime family is rejected and the band is revisited — which is a
result, not a failure of nerve.

This module reads the competence reports produced so far and answers one
question: what, if anything, is permitted next? It refuses past the cap.

Registered before any 20,000-update result existed. See
`PHASE1_V4_PREREGISTRATION.md` §Escalation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

BASE_EXAMPLES = 50_000
BASE_STEPS = 20_000
ESCALATED_EXAMPLES = 200_000
ESCALATED_STEPS = 30_000

#: At most this many distinct REGIMES, ever. Four is the size of the ladder;
#: a fifth would mean the moves stopped being pre-stated.
#:
#: Counting regimes rather than raw reports is deliberate. Re-measuring the
#: SAME regime after a documented harness defect is a correction, not an
#: escalation -- it does not change what is being tested, so it cannot be a
#: way of searching for a regime that passes. Re-measuring it repeatedly
#: WITHOUT such a fix is exactly the dice-rolling this cap exists to stop, and
#: shows up as a rung already attempted, which is refused below.
MEASUREMENT_CAP = 4

#: The only two failures with a registered mechanistic response.
MIXTURE_CHECK = "competence_is_reliable_across_seeds"
MEMORIZATION_CHECK = "generalization_gap_acceptable"


@dataclass(frozen=True, slots=True)
class Step:
    train_examples: int
    steps: int
    rationale: str

    def command(self, output: str) -> str:
        return (
            f"python scripts/phase1_v4_competence_run.py --output {output} "
            f"--band easy_nontrivial --train-examples {self.train_examples} "
            f"--holdout 3000 --steps {self.steps} --seeds 1,2,3"
        )


BASE = Step(BASE_EXAMPLES, BASE_STEPS, "registered base regime")

# Each response is tied to the arithmetic of the check it answers.
MORE_UPDATES = Step(
    BASE_EXAMPLES, ESCALATED_STEPS,
    "a mixture of converged and unconverged runs means the budget truncates the "
    "training phase transition; the truncated runs were still descending",
)
MORE_DATA = Step(
    ESCALATED_EXAMPLES, BASE_STEPS,
    "a memorization gap at a fixed batch size is a function of epochs: 20k "
    "updates over 50k examples is ~25.6 epochs, over 200k is ~6.4. More updates "
    "would make it worse, so the response is data, not budget",
)
BOTH = Step(
    ESCALATED_EXAMPLES, ESCALATED_STEPS,
    "both failures at once; the two responses are independent and compose",
)


def failed_checks(report: dict) -> set[str]:
    return {name for name, ok in report.get("checks", {}).items() if not ok}


def next_step(reports: list[dict]) -> tuple[Step | None, str]:
    """Return the permitted next measurement, or None with the reason why not."""

    if any(report.get("verdict") == "GO" for report in reports):
        return None, "a GO report already exists; the ladder is complete"
    attempted = {(int(r["regime"]["train_examples"]), int(r["regime"]["steps"]))
                 for r in reports if "regime" in r}
    if len(attempted) >= MEASUREMENT_CAP:
        return None, (
            f"{len(attempted)} distinct regimes have been measured and the "
            f"registered cap is {MEASUREMENT_CAP}; the regime family is REJECTED. "
            f"Revisit the difficulty band under a new preregistration -- do not "
            f"take a fifth"
        )
    if not reports:
        return BASE, "no measurement taken yet"

    latest = failed_checks(reports[-1])

    # Only these two failures have a registered response. Anything else -- a
    # dead family, a saturated ceiling, NLL worse than uniform -- is not a
    # budget or volume problem and no rung of this ladder addresses it.
    unanswerable = latest - {MIXTURE_CHECK, MEMORIZATION_CHECK}
    if unanswerable:
        return None, (
            f"failed checks with no registered response: {sorted(unanswerable)}. "
            f"The ladder only answers a truncated transition or an epoch-driven "
            f"memorization gap. REJECT the regime family and revisit the band"
        )

    mixture = MIXTURE_CHECK in latest
    memorization = MEMORIZATION_CHECK in latest
    if mixture and memorization:
        candidate = BOTH
    elif memorization:
        candidate = MORE_DATA
    elif mixture:
        candidate = MORE_UPDATES
    else:
        return None, "NO_GO with no failed checks recorded; the report is malformed"

    if (candidate.train_examples, candidate.steps) in attempted:
        return None, (
            f"the registered response to {sorted(latest)} is "
            f"{candidate.train_examples} examples x {candidate.steps} updates, "
            f"which has already been measured. Each rung may be taken at most "
            f"once. REJECT the regime family and revisit the band"
        )
    return candidate, candidate.rationale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="*", type=Path,
                        help="competence reports in the order they were produced")
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    for path, report in zip(args.reports, reports):
        regime = report.get("regime", {})
        print(f"{path.name}: {report.get('verdict')} "
              f"({regime.get('train_examples')} ex x {regime.get('steps')} steps)"
              f"{'  failed: ' + str(sorted(failed_checks(report))) if report.get('verdict') != 'GO' else ''}")

    step, reason = next_step(reports)
    print()
    if step is None:
        print(f"NO FURTHER MEASUREMENT PERMITTED: {reason}")
        return 1
    print(f"PERMITTED NEXT: {step.train_examples} examples x {step.steps} updates")
    print(f"  because: {reason}")
    print(f"  {step.command('<evidence>/competence.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
