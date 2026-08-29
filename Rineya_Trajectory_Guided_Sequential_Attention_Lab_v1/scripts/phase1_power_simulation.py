#!/usr/bin/env python3
"""Run and freeze the outcome-blind power simulation for the registered gates."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
from tgsa.phase1_power import default_scenarios, render_markdown, simulate_power
from tgsa.phase1_runner import atomic_write_json, load_study_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--strata-count", type=int, default=8)
    parser.add_argument("--examples-per-stratum", type=int, default=60)
    args = parser.parse_args()
    study = load_study_config(args.config.resolve())
    analysis = study["analysis"]
    threshold = float(analysis.get("minimum_history_delta", 0.02))
    report = simulate_power(
        scenarios=default_scenarios(threshold),
        seeds=[int(seed) for seed in study["seeds"]],
        strata_count=args.strata_count,
        examples_per_stratum=args.examples_per_stratum,
        threshold=threshold,
        bootstrap_samples=args.bootstrap_samples,
        confidence=float(analysis.get("confidence", 0.95)),
        bootstrap_scheme=str(analysis.get("bootstrap_scheme", "crossed_paired")),
        trials=args.trials,
        rng_seed=int(analysis.get("bootstrap_seed", 91273)),
    )
    atomic_write_json(args.output.resolve(), report)
    args.output.resolve().with_suffix(".md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
