"""Paired endpoints: acquisition gain and macro-task retention, never pooled seeds."""
from __future__ import annotations
import math
from statistics import mean, stdev


def behavioral_metrics(baseline, final):
    def keyed(records):
        result = {r["id"]: r for r in records}
        if not records or len(result) != len(records):
            raise ValueError("Empty/duplicate evaluation rows")
        for r in records:
            if type(r["correct"]) is not bool:
                raise ValueError("Correctness must be boolean")
        return result
    before, after = keyed(baseline), keyed(final)
    if before.keys() != after.keys():
        raise ValueError("Evaluation coverage changed; missing rows are not dropped")
    by_task = {}
    for key, b in before.items():
        a = after[key]
        if (b["task"], b["purpose"], b["parent_id"]) != (a["task"], a["purpose"], a["parent_id"]):
            raise ValueError("Evaluation identity changed")
        group = by_task.setdefault((b["purpose"], b["task"]), [])
        group.append((b["correct"], a["correct"]))
    tasks = []
    for (purpose, task), pairs in sorted(by_task.items()):
        initial = sum(b for b, _ in pairs)
        final_count = sum(a for _, a in pairs)
        retained = sum(b and a for b, a in pairs)
        tasks.append({"purpose": purpose, "task": task, "n": len(pairs),
                      "initial_correct": initial, "final_correct": final_count,
                      "delta_pp": 100 * (final_count - initial) / len(pairs),
                      "source_correct_retained": retained,
                      "retained_fraction_of_source_correct": retained / initial if initial else None,
                      "damages": sum(b and not a for b, a in pairs),
                      "repairs": sum(not b and a for b, a in pairs)})
    values = {}
    for purpose in ("acquisition", "retention"):
        subset = [r["delta_pp"] for r in tasks if r["purpose"] == purpose]
        if not subset:
            raise ValueError(f"Missing {purpose} evaluation cohort")
        values[purpose + "_pp"] = mean(subset)
    return dict(values, tasks=tasks, unit="percentage_points", evidence="LOCAL_DEV_BEHAVIOR_ONLY")


def last_three_endpoint(baseline, checkpoints):
    if len(checkpoints) < 3 or len({c["step"] for c in checkpoints}) != len(checkpoints):
        raise ValueError("Need at least three distinct saved evaluation checkpoints")
    ordered = sorted(checkpoints, key=lambda c: c["step"])
    metrics = [behavioral_metrics(baseline, c["rows"]) for c in ordered[-3:]]
    return {"acquisition_pp": mean(m["acquisition_pp"] for m in metrics),
            "retention_pp": metrics[-1]["retention_pp"], "final_detail": metrics[-1],
            "acquisition_steps": [c["step"] for c in ordered[-3:]],
            "retention_step": ordered[-1]["step"], "promotion_allowed": False}


def paired_contrast(left, right, metric):
    if not left or left.keys() != right.keys():
        raise ValueError("Paired task-seed keys must match")
    differences = [right[k][metric] - left[k][metric] for k in sorted(left)]
    if not all(math.isfinite(x) for x in differences):
        raise ValueError("Nonfinite contrast")
    n = len(differences)
    se = stdev(differences) / math.sqrt(n) if n > 1 else None
    return {"n_independent_pairs": n, "mean_right_minus_left": mean(differences),
            "normal_95_ci": None if se is None else [mean(differences) - 1.96 * se, mean(differences) + 1.96 * se],
            "warning": "Descriptive normal interval; small n and multiple comparisons limit inference"}


def pareto_indices(records):
    for r in records:
        if any(not math.isfinite(r[k]) for k in ("acquisition_pp", "retention_pp")):
            raise ValueError("Nonfinite Pareto point")
    return [i for i, r in enumerate(records) if not any(
        other["acquisition_pp"] >= r["acquisition_pp"] and other["retention_pp"] >= r["retention_pp"] and
        (other["acquisition_pp"] > r["acquisition_pp"] or other["retention_pp"] > r["retention_pp"])
        for j, other in enumerate(records) if j != i)]


def crossfit_rate(records):
    """Appendix E parity-fold selection; inputs must be a complete one-family grid."""
    cells = {}
    stratum = None
    for r in records:
        signature = {k: r.get(k) for k in ("geometry", "rollout_source", "kl_direction", "steps", "learning_rate", "task")}
        if stratum is not None and signature != stratum:
            raise ValueError("Stratify fixed axes/geometry before cross-fitting")
        stratum = signature
        if type(r["seed"]) is not int or not 0 <= r["ema_rate"] <= 1 or not math.isfinite(r["acquisition_pp"]):
            raise ValueError("Invalid seed/rate/endpoint")
        key = (r["seed"], r["ema_rate"])
        if key in cells:
            raise ValueError("Duplicate seed/rate; stratify other axes before cross-fitting")
        cells[key] = r
    seeds = sorted({k[0] for k in cells})
    rates = sorted({k[1] for k in cells})
    if not seeds or any((s, a) not in cells for s in seeds for a in rates):
        raise ValueError("Incomplete coupling grid")
    output = []
    for fold in (0, 1):
        train, held = [s for s in seeds if s % 2 != fold], [s for s in seeds if s % 2 == fold]
        if not train or not held:
            raise ValueError("Both seed-parity folds required")
        selected = max(rates, key=lambda a: (mean(cells[s, a]["acquisition_pp"] for s in train), -a))
        output += [dict(cells[s, selected], selection_fold=1-fold, heldout_fold=fold) for s in held]
    return output
