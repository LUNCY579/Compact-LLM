"""Exploratory DEV-only paired binary analysis with parent-cluster bootstrap.

Input JSON: {"role":"DEV", "rows":[{"parent_id":"p1", "family":"f",
"seed":101, "base_correct":true, "candidate_correct":false}, ...]}.
Optional paired fields: base_malformed/candidate_malformed and
base_eos_failure/candidate_eos_failure. An unmeasured panel is null in output.
No model, benchmark bank, statistical package, or external service is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys


OPTIONAL_PANELS = ("malformed", "eos_failure")
INTERVAL_METRICS = ("base_accuracy", "candidate_accuracy", "accuracy_delta",
                    "source_correct_loss_rate", "source_incorrect_repair_rate")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _finite_json(value):
    if isinstance(value, dict):
        require(all(type(key) is str for key in value), "JSON object keys must be strings")
        for child in value.values():
            _finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _finite_json(child)
    elif type(value) is float:
        require(math.isfinite(value), "nonfinite JSON number")
    else:
        require(value is None or type(value) in (str, int, bool), "input must contain JSON values only")


def validate_rows(document):
    _finite_json(document)
    require(isinstance(document, dict), "input must be a JSON object")
    require(document.get("role") == "DEV", "only DEV is admitted; CHECK and FINAL remain closed")
    rows = document.get("rows")
    require(isinstance(rows, list) and bool(rows), "nonempty paired rows required")
    parents, seen = {}, set()
    panel_coverage = {name: set() for name in OPTIONAL_PANELS}
    for row in rows:
        require(isinstance(row, dict), "row must be an object")
        required = ("parent_id", "family", "seed", "base_correct", "candidate_correct")
        require(all(key in row for key in required), "missing paired row fields")
        require(row.get("role", "DEV") == "DEV", "row role must be DEV")
        parent_id, family, seed = row["parent_id"], row["family"], row["seed"]
        require(type(parent_id) is str and bool(parent_id.strip()) and parent_id == parent_id.strip(),
                "parent_id must be a nonblank canonical string")
        require(type(family) is str and bool(family.strip()) and family == family.strip(),
                "family must be a nonblank canonical string")
        require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
        require(type(row["base_correct"]) is bool and type(row["candidate_correct"]) is bool,
                "correctness must contain strict paired booleans")
        require((parent_id, seed) not in seen, "duplicate parent+seed")
        seen.add((parent_id, seed))
        parent = parents.setdefault(parent_id, {"parent_id": parent_id, "family": family, "rows": {}})
        require(parent["family"] == family, "parent family changed across seeds")
        parent["rows"][seed] = row
        for panel in OPTIONAL_PANELS:
            left, right = row.get("base_" + panel), row.get("candidate_" + panel)
            require((left is None and right is None) or (type(left) is bool and type(right) is bool),
                    f"{panel}: missing pair or nonboolean observation")
            panel_coverage[panel].add(left is not None)
    seed_sets = {tuple(sorted(parent["rows"])) for parent in parents.values()}
    require(len(seed_sets) == 1, "unequal parent/family seed coverage; missing paired observations")
    for panel, coverage in panel_coverage.items():
        require(len(coverage) == 1, f"{panel}: partial panel coverage is not imputable")
    return [parents[key] for key in sorted(parents)], list(next(iter(seed_sets)))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _point(parents, seed=None):
    """Parents receive equal weight; repeated seeds are averaged inside parents."""
    grouped = [[p["rows"][seed]] if seed is not None else list(p["rows"].values()) for p in parents]
    rows = [row for group in grouped for row in group]
    base = sum(sum(row["base_correct"] for row in group) / len(group) for group in grouped) / len(parents)
    candidate = sum(sum(row["candidate_correct"] for row in group) / len(group) for group in grouped) / len(parents)
    base_correct = sum(row["base_correct"] for row in rows)
    losses = sum(row["base_correct"] and not row["candidate_correct"] for row in rows)
    repairs = sum(not row["base_correct"] and row["candidate_correct"] for row in rows)
    base_incorrect = len(rows) - base_correct
    optional = {}
    for panel in OPTIONAL_PANELS:
        if rows[0].get("base_" + panel) is None:
            optional[panel] = None
        else:
            left = sum(sum(row["base_" + panel] for row in group) / len(group) for group in grouped) / len(parents)
            right = sum(sum(row["candidate_" + panel] for row in group) / len(group) for group in grouped) / len(parents)
            optional[panel] = {"base_rate": left, "candidate_rate": right, "candidate_minus_base": right-left}
    return {"independent_parent_count": len(parents), "paired_seed_observations": len(rows),
            "base_accuracy": base, "candidate_accuracy": candidate, "accuracy_delta": candidate-base,
            "source_correct_losses": losses, "source_correct_denominator": base_correct,
            "source_correct_retained": base_correct-losses,
            "source_correct_loss_rate": _ratio(losses, base_correct),
            "source_correct_retention_rate": _ratio(base_correct-losses, base_correct),
            "source_incorrect_repairs": repairs, "source_incorrect_denominator": base_incorrect,
            "source_incorrect_repair_rate": _ratio(repairs, base_incorrect),
            "discordant_pairs": {"candidate_only_correct": repairs, "base_only_correct": losses},
            "optional_panels": optional}


def _quantile(values, probability):
    ordered = sorted(values)
    location = (len(ordered)-1) * probability
    lower, upper = math.floor(location), math.ceil(location)
    return ordered[lower] + (ordered[upper]-ordered[lower]) * (location-lower)


def _interval(values, confidence):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    tail = (1-confidence)/2
    return {"low": _quantile(valid, tail), "high": _quantile(valid, 1-tail),
            "valid_replicates": len(valid), "undefined_replicates": len(values)-len(valid)}


def analyze(document, *, bootstrap_samples=10000, bootstrap_seed=20260927, confidence=0.95):
    require(type(bootstrap_samples) is int and 1 <= bootstrap_samples <= 100000,
            "bootstrap_samples must be an integer in [1,100000]")
    require(type(bootstrap_seed) is int and bootstrap_seed >= 0, "bootstrap_seed must be a nonnegative integer")
    require(type(confidence) in (int, float) and math.isfinite(confidence) and 0 < confidence < 1,
            "confidence must be finite and between zero and one")
    parents, seeds = validate_rows(document)
    families = sorted({p["family"] for p in parents})
    strata = {family: [p for p in parents if p["family"] == family] for family in families}
    # Every bootstrap replicate carries each drawn parent's complete seed vector.
    # The identical draws feed every endpoint, seed view, and family view below.
    groups = [("all", None, None)]
    groups += [("seed", None, seed) for seed in seeds]
    groups += [("family", family, None) for family in families]
    groups += [("family_seed", family, seed) for family in families for seed in seeds]
    def views(sampled):
        all_parents = [p for family in families for p in sampled[family]]
        return {group: _point(all_parents if family is None else sampled[family], seed)
                for group in groups for _, family, seed in [group]}
    point = views(strata)
    series = {group: {metric: [] for metric in INTERVAL_METRICS} for group in groups}
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_samples):
        draw = {family: [rng.choice(group) for _ in group] for family, group in strata.items()}
        for group, summary in views(draw).items():
            for metric in INTERVAL_METRICS:
                series[group][metric].append(summary[metric])
    for group, summary in point.items():
        summary["descriptive_intervals"] = {metric: _interval(values, confidence)
                                             for metric, values in series[group].items()}
    return {
        "schema_version": 1, "status": "EXPLORATORY_DEV_DESCRIPTIVE_ONLY", "role": "DEV",
        "independence_unit": "parent_id", "seed_aggregation": "equal-weight seed mean inside each parent",
        "paired_observation_count_note": "Loss/repair numerators and denominators count parent-seed observations; those observations are not independent samples.",
        "retention_scope": "Source-correct preservation on these supplied DEV parents only; no claim about unmeasured preservation panels.",
        "bootstrap": {"method": "percentile parent-cluster bootstrap stratified by family",
                      "samples": bootstrap_samples, "seed": bootstrap_seed, "confidence": confidence,
                      "family_weights": "observed parent proportions fixed across replicates",
                      "sampling_unit": "whole parent with all seeds and paired endpoints",
                      "limitation": "Descriptive intervals condition on supplied families and seeds; no seed-population or unseen-family uncertainty is estimated."},
        "aggregate": point[("all", None, None)],
        "per_seed": [{"seed": seed, **point[("seed", None, seed)]} for seed in seeds],
        "per_family": [{"family": family, **point[("family", family, None)],
                        "per_seed": [{"seed": seed, **point[("family_seed", family, seed)]} for seed in seeds]}
                       for family in families],
        "limitations": ["DEV is exploratory and may have informed decisions.",
                        "Few parents or singleton families can yield degenerate bootstrap intervals; these do not establish certainty.",
                        "Missing malformed/EOS panels are null, never imputed as successful.",
                        "Net accuracy gains do not erase source-correct losses or family regressions."],
        "noninferiority_certificate": False, "equivalence_certificate": False,
        "confirmation_claim": False, "automatic_promotion": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260927)
    parser.add_argument("--confidence", type=float, default=.95)
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), "analysis output exists; results are write-once")
        source_bytes = args.input.read_bytes()
        document = json.loads(source_bytes, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        result = analyze(document, bootstrap_samples=args.bootstrap_samples,
                         bootstrap_seed=args.bootstrap_seed, confidence=args.confidence)
        result["input_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        result["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    except (ValueError, OSError, TypeError) as exc:
        print(f"PAIRED_ANALYSIS_FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "independent_parent_count": result["aggregate"]["independent_parent_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
