"""Offset-only fit, paired parent bootstrap and functional interaction diagnostics."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from .io import align, require


def weights(rows: list[dict]) -> np.ndarray:
    """Equal family -> equal parent -> equal observed token; frozen before comparison."""
    families: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        families[r["family"]][r["parent_id"]] += 1
    return np.array([1 / (len(families) * len(families[r["family"]]) *
                         families[r["family"]][r["parent_id"]]) for r in rows], dtype=np.float64)


def profile(x, y, w=None) -> dict:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    require(x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) > 0, "Invalid loss vectors")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Non-finite losses")
    w = np.ones(len(x)) / len(x) if w is None else np.asarray(w, dtype=np.float64)
    require(w.shape == x.shape and np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0, "Invalid weights")
    w = w / w.sum()
    mx, my = float(w @ x), float(w @ y)
    xc, yc = x - mx, y - my
    vx, vy, cov = float(w @ (xc * xc)), float(w @ (yc * yc)), float(w @ (xc * yc))
    residual = (y - x) - (my - mx)
    rv = float(w @ (residual * residual))
    return {"mean_reference_nll": mx, "mean_candidate_nll": my, "offset": my - mx,
            "residual_rms": math.sqrt(rv), "pair_difference_rms": math.sqrt(2 * rv),
            "centered_ccc": 2 * cov / (vx + vy) if vx + vy > 1e-24 else None,
            "pearson": cov / math.sqrt(vx * vy) if min(vx, vy) > 1e-24 else None,
            "free_slope_diagnostic_only": cov / vx if vx > 1e-24 else None,
            "weighted_single_run_rdiff": 0.5 * rv * (1 / vx + 1 / vy) if min(vx, vy) > 1e-24 else None,
            "max_abs_residual": float(np.abs(residual).max()),
            "degenerate_difficulty": min(vx, vy) <= 1e-24}


def parent_bootstrap(rows: list[dict], x, y, repeats: int = 1000, seed: int = 17, alpha: float = 0.05) -> dict:
    require(0 < alpha < 1, "Invalid confidence level")
    require(type(repeats) is int and 100 <= repeats <= 10000, "Bootstrap repeats must be 100..10000")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["family"], r["parent_id"])].append(i)
    family_groups = defaultdict(list)
    for (family, _), idx in groups.items():
        family_groups[family].append(idx)
    counts = {f: len(g) for f, g in family_groups.items()}
    if len(groups) < 8 or min(counts.values()) < 2:
        return {"status": "UNDERPOWERED", "parent_counts": counts, "intervals": None}
    x, y = np.asarray(x, float), np.asarray(y, float)
    # Center before forming moments to reduce cancellation for nearly constant losses.
    x, y = x - x.mean(), y - x.mean()  # same shift preserves candidate-reference offset
    rng = np.random.default_rng(seed)
    moments = np.zeros((repeats, 5), dtype=np.float64)
    for family in sorted(family_groups):
        ms = []
        for idx in family_groups[family]:
            a, b = x[idx], y[idx]
            ms.append([a.mean(), b.mean(), (a*a).mean(), (b*b).mean(), (a*b).mean()])
        ms = np.asarray(ms)
        moments += ms[rng.integers(len(ms), size=(repeats, len(ms)))].mean(axis=1) / len(family_groups)
    mx, my, x2, y2, xy = moments.T
    rv = np.maximum(x2 + y2 - 2 * xy - (my - mx)**2, 0)
    values = {"offset": my - mx, "residual_rms": np.sqrt(rv)}
    return {"status": "EVALUATION_SAMPLING_ONLY", "parent_counts": counts, "repeats": repeats, "seed": seed, "alpha": alpha,
            "intervals": {k: [float(v) for v in np.quantile(a, [alpha / 2, 1 - alpha / 2])] for k, a in values.items()}}


def slices(rows: list[dict], delta: np.ndarray, global_offset: float) -> dict:
    result = {}
    for field in ("family", "operation", "rule_case", "role", "suite", "horizon"):
        by_value = defaultdict(list)
        for i, r in enumerate(rows):
            by_value[str(r[field])].append(i)
        for value, idx in sorted(by_value.items()):
            subset = [rows[i] for i in idx]
            w = weights(subset)
            d = delta[idx]
            result[f"{field}={value}"] = {"tokens": len(idx), "parents": len({r['parent_id'] for r in subset}),
                "mean_delta": float(w @ d), "mean_residual_from_global_offset": float(w @ d - global_offset),
                "rms_residual_from_global_offset": float(np.sqrt(w @ ((d - global_offset)**2)))}
    return result


def compare(reference: dict, candidate: dict, repeats: int = 1000, seed: int = 17) -> dict:
    rows, values = align(reference, candidate)
    require({r["split"] for r in rows} <= {"development", "confirmation"}, "Training rows are not validation")
    require(len({r["split"] for r in rows}) == 1, "Do not pool development and confirmation")
    x, y = np.asarray(values)
    p = profile(x, y, weights(rows))
    subset_reports = {}
    for role in ("content", "format", "mixed"):
        idx = [i for i, r in enumerate(rows) if r["role"] == role]
        if idx:
            rr = [rows[i] for i in idx]
            subset_reports[role] = {"profile": profile(x[idx], y[idx], weights(rr)),
                                   "bootstrap": parent_bootstrap(rr, x[idx], y[idx], repeats, seed)}
    # Whole-answer averages are reported alongside, never instead of, token-level effects.
    by_item = defaultdict(list)
    for i, r in enumerate(rows):
        by_item[r["item_id"]].append(i)
    item_rows = [rows[idx[0]] for idx in by_item.values()]
    item_x = [float(x[idx].mean()) for idx in by_item.values()]
    item_y = [float(y[idx].mean()) for idx in by_item.values()]
    return {"kind": "rgi_diagnostic", "schema_version": 1, "evidence_class": reference["evidence_class"],
            "reference_sha256": reference["content_sha256"], "candidate_sha256": candidate["content_sha256"],
            "weighting": "equal_family_equal_parent_equal_token", "token_profile": p,
            "bootstrap": parent_bootstrap(rows, x, y, repeats, seed), "role_profiles": subset_reports,
            "item_average_profile": profile(item_x, item_y, weights(item_rows)),
            "slices": slices(rows, y - x, p["offset"]), "quality_improvement_established": False,
            "interpretation": "A diagnostic, not a skill, retention, causal or promotion verdict."}


def interaction(base: dict, a: dict, b: dict, ab: dict, repeats: int = 1000, seed: int = 17) -> dict:
    rows, losses = align(base, a, b, ab)
    require(len({m["host_sha256"] for m in (base, a, b, ab)}) == 1, "Composition requires the same exact host")
    require(not base["components"] and a["components"] and b["components"], "Need off, A, B and A+B provenance")
    require(not (set(a["components"]) & set(b["components"])), "A and B must be separately identified components")
    require(ab["components"] == {**a["components"], **b["components"]}, "Combined adapter coefficients differ")
    require(len({r["split"] for r in rows}) == 1 and rows[0]["split"] != "train", "Use one evaluation split")
    z, x, y, both = np.asarray(losses)
    w = weights(rows)
    signal = both - x - y + z
    da, db = x - z, y - z
    ca, cb = da - w @ da, db - w @ db
    denom = float(np.sqrt((w @ (ca * ca)) * (w @ (cb * cb))))
    mean = float(w @ signal)
    return {"kind": "composition_interaction", "schema_version": 1, "evidence_class": base["evidence_class"],
            "input_sha256": [m["content_sha256"] for m in (base, a, b, ab)],
            "mean_interaction": mean, "interaction_rms": float(np.sqrt(w @ (signal**2))),
            "centered_effect_cosine": float(w @ (ca * cb) / denom) if denom > 1e-24 else None,
            "bootstrap": parent_bootstrap(rows, np.zeros(len(rows)), signal, repeats, seed),
            "slices": slices(rows, signal, mean), "quality_improvement_established": False,
            "interpretation": "Nonlinearity is not itself harm or synergy; test component and joint-use tasks."}


def paired_outcomes(reference: dict, candidate: dict, endpoint: str, margin: float,
                    direction: str, repeats: int = 1000, seed: int = 17,
                    mode: str = "superiority", alpha: float = 0.05) -> dict:
    """Independent behavioral gate. Invalid parses must already count as failures."""
    require(direction in {"higher", "lower"} and math.isfinite(margin) and margin >= 0, "Invalid endpoint rule")
    require(mode in {"superiority", "noninferiority"}, "Invalid gate mode")
    threshold = margin if mode == "superiority" else -margin
    for value in (reference, candidate):
        require(value.get("kind") == "behavior_outcomes" and value.get("complete") is True, "Incomplete outcomes")
        require(value.get("invalid_policy") == "count_as_failure", "Do not discard failed parses")
    require(reference["bank_sha256"] == candidate["bank_sha256"], "Different outcome banks")
    require(reference["evidence_class"] == candidate["evidence_class"], "Different evidence classes")
    maps = []
    for value in (reference, candidate):
        rows = [r for r in value["rows"] if r["endpoint"] == endpoint]
        require(rows, "Missing behavioral endpoint")
        m = {r["item_id"]: r for r in rows}
        require(len(m) == len(rows), "Duplicate behavioral item")
        require(all(math.isfinite(r["value"]) for r in rows), "Non-finite behavioral outcome")
        maps.append(m)
    require(maps[0].keys() == maps[1].keys(), "Outcome coverage mismatch")
    keys = sorted(maps[0])
    for k in keys:
        for field in ("parent_id", "family", "split", "endpoint", "payload_sha256"):
            require(maps[0][k][field] == maps[1][k][field], "Behavioral alignment mismatch")
    rows = [maps[0][k] for k in keys]
    require(len({r["split"] for r in rows}) == 1 and rows[0]["split"] != "train", "Use one evaluation split")
    sign = 1 if direction == "higher" else -1
    x = np.array([maps[0][k]["value"] for k in keys]) * sign
    y = np.array([maps[1][k]["value"] for k in keys]) * sign
    ci = parent_bootstrap(rows, x, y, repeats, seed, alpha)
    band = ci["intervals"]["offset"] if ci["intervals"] is not None else None
    group_delta = defaultdict(list)
    for row, d in zip(rows, y - x):
        group_delta[(row["family"], row["parent_id"])].append(float(d))
    degenerate = np.var([np.mean(d) for d in group_delta.values()]) <= 1e-24
    verdict = "UNDERPOWERED" if band is None else "INCONCLUSIVE_DEGENERATE" if degenerate else "PASS" if band[0] >= threshold else "FAIL" if band[1] < threshold else "INCONCLUSIVE"
    return {"endpoint": endpoint, "direction": direction, "mode": mode, "required_improvement": threshold,
            "mean_improvement": float(weights(rows) @ (y - x)), "bootstrap": ci, "verdict": verdict,
            "note": "Constant parent differences need a separately registered exact/bounded test, not a zero-width bootstrap certificate."}


def batch_decomposition(previous_on_previous: float, previous_on_current: float,
                        current_on_current: float) -> dict:
    values = [previous_on_previous, previous_on_current, current_on_current]
    require(all(math.isfinite(x) for x in values), "Non-finite matched batch losses")
    data = previous_on_current - previous_on_previous
    update = current_on_current - previous_on_current
    return {"batch_change": data, "parameter_update": update, "total": current_on_current - previous_on_previous,
            "absolute_data_to_update": abs(data / update) if update != 0 else None}
