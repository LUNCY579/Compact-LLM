"""Paired, parent-clustered analysis. A statistical pass is never model promotion."""
from collections import defaultdict
import numpy as np
from .control import digest


def paired_interval(differences, families, parents, repetitions=4000, seed=0, alpha=0.05):
    d = np.asarray(differences, dtype=float)
    if (d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all()
            or len(families) != len(d) or len(parents) != len(d)
            or (np.abs(d) > 1).any() or repetitions < 100 or not 0 < alpha < 1):
        raise ValueError("invalid paired bootstrap inputs")
    groups = defaultdict(lambda: defaultdict(list))
    for value, family, parent in zip(d, families, parents):
        groups[family][parent].append(float(value))
    if any(len(g) < 2 for g in groups.values()):
        raise ValueError("need >= 2 independent parents per family")
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(repetitions)
    unseen_lower, unseen_upper = 0.0, 0.0
    for clusters in groups.values():
        values = list(clusters.values())
        sums = np.array([sum(v) for v in values])
        counts = np.array([len(v) for v in values])
        means = sums / counts
        if np.ptp(means) < 1e-14:
            # Degenerate bootstrap is not evidence of zero possible harm.
            # Conservative unseen-event mass, with Bonferroni over families.
            mass = 1 - (alpha / (2 * len(groups))) ** (1 / len(values))
            unseen_lower += (means[0] + 1) * mass * counts.sum() / len(d)
            unseen_upper += (1 - means[0]) * mass * counts.sum() / len(d)
        indices = rng.integers(0, len(values), size=(repetitions, len(values)))
        bootstrap += ((sums[indices].sum(1) / counts[indices].sum(1))
                      * counts.sum() / len(d))
    lo, hi = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    lo, hi = max(-1.0, float(lo) - unseen_lower), min(1.0, float(hi) + unseen_upper)
    return {"mean": float(d.mean()), "low": lo, "high": hi,
            "zero_variance_guard": "unseen-event mass bound for constant parent effects",
            "alpha": alpha, "parents": sum(len(v) for v in groups.values()),
            "rows": len(d), "repetitions": repetitions,
            "method": "paired_family_stratified_parent_cluster_bootstrap",
            "seed_uncertainty": "conditional on measured training seeds"}


def compare(candidate, baseline, objective="quality", repetitions=4000, seed=0):
    if objective not in {"quality", "efficiency"}:
        raise ValueError("objective must be preregistered as quality or efficiency")
    for key in ("data_sha256", "pool_sha256", "evaluation_split", "budget"):
        if candidate[key] != baseline[key]:
            raise ValueError(f"unmatched comparison: {key}")
    def index(report):
        indexed = {(r.get("seed", 0), r["id"]): r for r in report["records"]}
        if len(indexed) != len(report["records"]):
            raise ValueError("duplicate paired record")
        return indexed
    a, b = index(candidate), index(baseline)
    if not a or set(a) != set(b):
        raise ValueError("missing/unmatched examples")
    diffs, families, parents, strata = [], [], [], []
    for key in sorted(a):
        ar, br = a[key], b[key]
        if any(ar[f] != br[f] for f in ("family", "parent_id", "payload_sha256")):
            raise ValueError("paired parent/payload metadata changed")
        if ar.get("stratum", "all") != br.get("stratum", "all"):
            raise ValueError("paired challenge stratum changed")
        strata.append(ar["family"] + "::" + ar.get("stratum", "all"))
        for record in (ar, br):
            if not np.isfinite(record["success"]) or not 0 <= record["success"] <= 1:
                raise ValueError("invalid success value")
        diffs.append(ar["success"] - br["success"])
        families.append(ar["family"])
        parents.append(ar["parent_id"])
    overall = paired_interval(diffs, families, parents, repetitions, seed)
    protected = {}
    for family in sorted(set(strata)):
        use = [i for i, f in enumerate(strata) if f == family]
        protected[family] = paired_interval([diffs[i] for i in use],
                                            [families[i] for i in use],
                                            [parents[i] for i in use], repetitions, seed,
                                            alpha=0.05 / len(set(strata)))
    retention = all(v["low"] >= -0.02 for v in protected.values())
    quality = overall["mean"] >= 0.05 and overall["low"] > 0 and retention
    efficiency = {"eligible": False, "reason": "requires actual end-to-end timing, not replay"}
    efficiency_pass = False
    if candidate.get("end_to_end_timing") and baseline.get("end_to_end_timing"):
        ca, cb = [], []
        for key in sorted(a):
            ca.append(a[key]["elapsed_seconds"])
            cb.append(b[key]["elapsed_seconds"])
        if not np.isfinite(ca + cb).all() or min(ca + cb) <= 0:
            raise ValueError("invalid end-to-end timing")
        saving = 1 - sum(ca) / sum(cb)
        efficiency = {"eligible": True, "fractional_time_saving": saving,
                      "timing_uncertainty": "point estimate; repeat hardware measurements"}
        efficiency_pass = saving >= 0.15 and overall["low"] >= -0.02 and retention
    native = all(r.get("evidence_kind") == "native_execution" for r in (candidate, baseline))
    confirmation = candidate["evaluation_split"] == "confirm"
    criterion = quality if objective == "quality" else efficiency_pass
    return {"schema": 1, "objective": objective, "paired_quality": overall,
            "protected_families_bonferroni": protected, "retention_pass": retention,
            "efficiency": efficiency, "numeric_criterion_pass": bool(criterion),
            "status": "STATISTICAL_CANDIDATE_REQUIRES_AUDIT" if
            (criterion and native and confirmation) else "NO_PROMOTION",
            "native_qualified": False,
            "limitations": ["No automatic qualification or release.",
                            "Choose one method on development data before confirmation.",
                            "Evidence labels and equal-budget declarations need external audit.",
                            "Bootstrap conditions on the measured seeds; replicate training."]}
