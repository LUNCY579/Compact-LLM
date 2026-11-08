"""Read a seed count off an existing power surface at the realised variance.

The power audit is expensive to run and cheap to re-read. Its output is a grid
of (seed count x true effect x seed SD) conditions, and the registered
adequacy rule is a pure function of that grid plus the variance the design
should be powered for. So once the surface exists, the realised sigma_a from the
competence report selects a seed count without re-simulating anything.

That ordering matters. Re-running the audit *after* seeing the realised variance
invites choosing a grid that produces a convenient answer. Simulating the whole
surface first and reading from it afterwards does not.

The adequacy rule is v3's, unchanged: at the chosen seed count the lower
confidence bound of detection probability must clear the target at the central
variance, clear a weaker target at the elevated variance, and the upper bound of
the null rejection rate must stay within tolerance. Confidence bounds rather
than point estimates, so Monte Carlo noise cannot select a seed count that
merely looks adequate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
if str(LAB / "scripts") not in sys.path:
    sys.path.insert(0, str(LAB / "scripts"))

from phase1_power_simulation import PRIMARY_GATE_KEYS, smallest_adequate_seed_count


def available(conditions: list[dict], key: str) -> list[float]:
    return sorted({round(float(entry["design"][key]), 6) for entry in conditions})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--competence", type=Path,
                        help="read the realised sigma_a from this competence report")
    parser.add_argument("--central-seed-sd", type=float)
    parser.add_argument("--elevated-multiplier", type=float, default=2.0)
    parser.add_argument("--elevated-seed-sd", type=float)
    parser.add_argument("--target-effect", type=float, default=0.020)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--robust-power", type=float, default=0.60)
    parser.add_argument("--max-null-rate", type=float, default=0.075)
    parser.add_argument("--primary-gate", default="lower_bound_above_zero",
                        choices=tuple(PRIMARY_GATE_KEYS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    surface = json.loads(args.surface.read_text(encoding="utf-8"))
    conditions = surface["conditions"] if "conditions" in surface else surface["evaluations"]

    central = args.central_seed_sd
    source = "supplied on the command line"
    if central is None:
        if args.competence is None:
            parser.error("supply --central-seed-sd or --competence")
        report = json.loads(args.competence.read_text(encoding="utf-8"))
        central = report["detail"]["realised_seed_sd_for_power_audit"]
        if central is None:
            raise ValueError(
                f"{args.competence.name} records no realised seed SD; the competence "
                f"run must complete before the seed count can be fixed"
            )
        source = f"realised in {args.competence.name}"
    elevated = args.elevated_seed_sd or round(central * args.elevated_multiplier, 6)

    grid = available(conditions, "seed_sd")
    for label, value in (("central", central), ("elevated", elevated)):
        if round(value, 6) not in grid:
            raise ValueError(
                f"{label} seed SD {value} is not on the simulated grid {grid}. "
                f"Re-run the audit including it rather than interpolating -- the "
                f"adequacy rule reads exact cells, and a nearby cell is a different "
                f"study"
            )

    recommendation = smallest_adequate_seed_count(
        conditions,
        gate_key=PRIMARY_GATE_KEYS[args.primary_gate],
        target_effect=args.target_effect,
        central_seed_sd=central,
        elevated_seed_sd=elevated,
        target_power=args.target_power,
        robust_power=args.robust_power,
        max_null_rate=args.max_null_rate,
    )

    print(f"central seed SD   {central}  ({source})")
    print(f"elevated seed SD  {elevated}")
    print(f"target effect     {args.target_effect}")
    print()
    print(f"{'seeds':>6} {'power@central':>14} {'power@elevated':>15} {'null rate':>11}  verdict")
    for entry in recommendation["evaluations"]:
        print(f"{entry['seed_count']:>6} "
              f"{entry['central_power']:>14.4f} "
              f"{entry['elevated_power']:>15.4f} "
              f"{entry['null_rate']:>11.4f}  "
              f"{'ADEQUATE' if entry['adequate'] else 'no'}")
    print()
    if recommendation["recommended_seed_count"] is None:
        print("NO ADEQUATE SEED COUNT on the simulated grid. The design is not "
              "powered and must not be registered.")
    else:
        print(f"RECOMMENDED SEED COUNT: {recommendation['recommended_seed_count']}")

    if args.output:
        payload = dict(recommendation)
        payload["central_seed_sd"] = central
        payload["central_seed_sd_source"] = source
        payload["elevated_seed_sd"] = elevated
        payload["surface"] = args.surface.name
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(f"wrote {args.output}")
    return 0 if recommendation["recommended_seed_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
