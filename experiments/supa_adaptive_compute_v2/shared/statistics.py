"""Validated adapters to audited paired and scenario-cluster statistics."""
from __future__ import annotations

from collections import defaultdict
import importlib.util
import math
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Any, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_paired = _load("_adaptive_paired_statistics", "rineya_significance_native_architecture_lab/src/rineya_significance_lab/evaluation/statistics.py")
_clustered = _load("_adaptive_clustered_statistics", "rineya_significance_native_architecture_lab/src/rineya_significance_lab/evaluation/clustered.py")
_consensus = _load("_adaptive_consensus_statistics", "short-response-adapter-consensus-lab/src/consensus_lab/analysis/statistics.py")
holm_adjust = _consensus.holm_adjust
normal_approximation_mde = _clustered.normal_approximation_mde


def _binary(value: Any) -> bool:
    if isinstance(value, dict):
        value = value["score"]
    if not isinstance(value, (int, float, bool, np.number)) or not math.isfinite(float(value)) or value not in (0, 1):
        raise ValueError("paired correctness must be finite binary 0/1")
    return bool(value)


def exact_mcnemar(baseline: Sequence, candidate: Sequence) -> dict:
    return _paired.exact_mcnemar([_binary(v) for v in baseline], [_binary(v) for v in candidate])


def paired_binary_statistics(baseline: Sequence, candidate: Sequence, *, scenario_ids: Sequence[str] | None = None,
                             seed: int = 20260905, bootstrap_samples: int = 10000,
                             randomization_samples: int = 10000) -> dict:
    left, right = [_binary(v) for v in baseline], [_binary(v) for v in candidate]
    if not left or len(left) != len(right):
        raise ValueError("paired nonempty equal-length vectors required")
    ids = list(scenario_ids) if scenario_ids is not None else [str(i) for i in range(len(left))]
    if any(not isinstance(identity, str) or not identity.strip() for identity in ids):
        raise ValueError("scenario IDs must be nonempty strings")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0 or not isinstance(randomization_samples, int) or randomization_samples <= 0:
        raise ValueError("positive integer resampling counts required")
    # The audited cluster helper validates finiteness, sizes, and nonempty IDs.
    differences = _clustered.cluster_differences(ids, left, right)
    clustered = len(differences) != len(left)
    interval = _clustered.cluster_bootstrap_interval(list(differences.values()), samples=bootstrap_samples, seed=seed)
    mcnemar = exact_mcnemar(left, right)
    base_rate, candidate_rate = sum(left)/len(left), sum(right)/len(right)
    diff = candidate_rate - base_rate
    if clustered:
        p_value, method, samples = _clustered.exact_or_sampled_sign_flip(
            list(differences.values()), monte_carlo_samples=randomization_samples, seed=seed)
    else:
        p_value, method, samples = mcnemar["p_value_two_sided"], "exact_mcnemar_independent_scenarios", 0
    return {"n": len(left), "independent_scenarios": len(differences),
            "baseline_accuracy": base_rate, "candidate_accuracy": candidate_rate,
            "accuracy_difference": diff, "scenario_macro_difference": float(np.mean(list(differences.values()))),
            "absolute_error_reduction": diff, "relative_error_reduction": diff/(1-base_rate) if base_rate < 1 else None,
            "paired_wins": mcnemar["candidate_only_correct"], "paired_losses": mcnemar["baseline_only_correct"],
            "both_correct": sum(a and b for a,b in zip(left,right)), "both_wrong": sum(not a and not b for a,b in zip(left,right)),
            "mcnemar": {**mcnemar, "inference_status": "descriptive_only_correlated_variants" if clustered else "primary_independent_scenarios"},
            "scenario_bootstrap_ci": {"lower": interval[0], "upper": interval[1], "confidence": .95,
                                      "samples": bootstrap_samples, "seed": seed, "resampling_unit": "scenario",
                                      "estimand": "equal-weight scenario mean difference"},
            "primary_p_value": p_value, "primary_test": method, "randomization_samples": samples,
            "ci_supports_positive_difference": interval[0] > 0,
            "degenerate_ci_caution": "No observed discordance does not establish equivalence." if not mcnemar["discordant"] else None}


def _index(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        identity = row.get("example_id", row.get("case_id"))
        if not isinstance(identity, str) or not identity.strip() or identity in result:
            raise ValueError("missing or duplicate output example ID")
        if row.get("safety_stopped") or row.get("termination") in {"timeout", "repetition_loop", "script_or_unicode_corruption", "failure"}:
            raise ValueError("incomplete or safety-stopped evidence cannot support paired efficacy")
        _binary(row["score"])
        result[identity] = row
    return result


def paired_analysis(baseline_rows: list[dict], candidate_rows: list[dict], *, seed: int = 20260905,
                    bootstrap_samples: int = 10000, expected_example_ids: Sequence[str] | None = None,
                    analysis_role: str = "confirmatory") -> dict:
    if analysis_role not in {"exploratory", "candidate_selection", "confirmatory", "diagnostic"}:
        raise ValueError("unknown analysis role")
    if expected_example_ids is not None and (len(set(expected_example_ids)) != len(expected_example_ids) or any(not isinstance(v, str) or not v.strip() for v in expected_example_ids)):
        raise ValueError("expected coverage requires unique nonempty example IDs")
    base, candidate = _index(baseline_rows), _index(candidate_rows)
    if not base or base.keys() != candidate.keys() or expected_example_ids is not None and set(expected_example_ids) != set(base):
        raise ValueError("incomplete baseline/candidate coverage")
    ids = sorted(base)
    for identity in ids:
        for key in ("scenario_id", "family", "track", "split", "variant", "template_id", "template_unseen_in_engineering"):
            if base[identity].get(key) != candidate[identity].get(key):
                raise ValueError(f"paired metadata mismatch: {key}")
        if not base[identity].get("scenario_id") or not base[identity].get("family"):
            raise ValueError("scenario and family identity are required")
        if analysis_role == "confirmatory" and base[identity].get("variant", "original") != "original":
            raise ValueError("linked diagnostics cannot be counted as primary confirmatory cases")

    def summarize(selected: list[str]) -> dict:
        result = paired_binary_statistics([base[i]["score"] for i in selected], [candidate[i]["score"] for i in selected],
                                          scenario_ids=[base[i]["scenario_id"] for i in selected], seed=seed,
                                          bootstrap_samples=bootstrap_samples)
        nll_ids, nll_base, nll_candidate, identity_count = [], [], [], 0
        for identity in selected:
            a,b = base[identity], candidate[identity]
            if "token_ids" in a and "token_ids" in b:
                identity_count += a["token_ids"] == b["token_ids"]
            elif "response" in a and "response" in b:
                identity_count += a["response"] == b["response"]
            for value in (a.get("gold_nll"), b.get("gold_nll")):
                if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                    raise ValueError("NLL must be finite and nonnegative")
            if a.get("gold_nll") is not None and b.get("gold_nll") is not None:
                nll_ids.append(a["scenario_id"]); nll_base.append(a["gold_nll"]); nll_candidate.append(b["gold_nll"])
        output_eligible = sum(("token_ids" in base[i] and "token_ids" in candidate[i]) or ("response" in base[i] and "response" in candidate[i]) for i in selected)
        result["output_identity"] = {"identical": identity_count, "pairs": output_eligible,
                                     "rate": identity_count/output_eligible if output_eligible else None}
        result["nll"] = {"pairs": len(nll_ids), "candidate_minus_baseline": float(np.mean(np.asarray(nll_candidate)-np.asarray(nll_base))) if nll_ids else None,
                         "direction": "negative favors candidate", "available_case_analysis": len(nll_ids) != len(selected)}
        if nll_ids:
            nll_differences = _clustered.cluster_differences(nll_ids, nll_base, nll_candidate)
            result["nll"]["scenario_bootstrap_ci"] = _clustered.cluster_bootstrap_interval(list(nll_differences.values()), samples=bootstrap_samples, seed=seed)
        result["validity"] = {"baseline_invalid": sum(bool(base[i].get("invalid")) for i in selected),
                              "candidate_invalid": sum(bool(candidate[i].get("invalid")) for i in selected)}
        for key in ("wall_seconds", "generation_dense_matmul_flops_estimate"):
            a = [float(base[i][key]) for i in selected if base[i].get(key) is not None]
            b = [float(candidate[i][key]) for i in selected if candidate[i].get(key) is not None]
            if any(not math.isfinite(v) or v < 0 for v in a+b):
                raise ValueError(f"{key} must be finite and nonnegative")
            result[key] = {"baseline_mean": float(np.mean(a)) if a else None, "candidate_mean": float(np.mean(b)) if b else None,
                           "ratio": sum(b)/sum(a) if len(a) == len(b) == len(selected) and sum(a)>0 else None}
        return result

    families, variants, templates = defaultdict(list), defaultdict(list), defaultdict(list)
    for identity in ids:
        families[base[identity]["family"]].append(identity)
        variants[base[identity].get("variant", "original")].append(identity)
        templates["unseen" if base[identity].get("template_unseen_in_engineering") else "seen"].append(identity)
    return {"analysis_role": analysis_role, "overall": summarize(ids),
            "families": {k:summarize(v) for k,v in sorted(families.items())},
            "generalization": {k:summarize(v) for k,v in sorted(templates.items())},
            "variants": {k:summarize(v) for k,v in sorted(variants.items())},
            "multiple_comparisons": "Track preregistration defines the comparison family; exploratory family results are annotated, not automatically selection gates."}


def comparison_annotations(p_values: dict[str, float], *, analysis_role: str) -> dict:
    if analysis_role not in {"exploratory", "candidate_selection", "confirmatory", "diagnostic"}:
        raise ValueError("unknown analysis role")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p_values.values()):
        raise ValueError("p-values must be in [0,1]")
    names = sorted(p_values)
    adjusted = holm_adjust([p_values[k] for k in names])
    return {name: {"raw_p": p_values[name], "holm_p": adjusted[index] if analysis_role == "confirmatory" else None,
                   "analysis_role": analysis_role, "comparison_family_size": len(names)} for index,name in enumerate(names)}


def power_planning(*, baseline_accuracy: float = .75, alpha: float = .05, target_power: float = .80) -> dict:
    """Normal paired-discordance planning, never a measured power claim."""
    if not all(math.isfinite(v) and 0 < v < 1 for v in (baseline_accuracy, alpha, target_power)) or target_power <= .5:
        raise ValueError("finite rates in (0,1), and target power above .5, required")
    normal = NormalDist()
    z = normal.inv_cdf(1-alpha/2)
    effects = {"1pp": .01, "2pp": .02, "3pp": .03, "5pct_relative_error_reduction": .05*(1-baseline_accuracy)}
    rows = []
    for n in (256,512,1024,2048):
        for discordance in (.05,.10,.20):
            for label, delta in effects.items():
                sd = math.sqrt((discordance-delta*delta)/n)
                critical = z*math.sqrt(discordance/n)
                power = 1-normal.cdf((critical-delta)/sd)+normal.cdf((-critical-delta)/sd)
                required = math.ceil((z+normal.inv_cdf(target_power))**2*discordance/delta**2)
                rows.append({"n_independent_scenarios": n, "discordance": discordance, "effect": label,
                             "absolute_difference": delta, "approximate_power": power,
                             "approximate_n_for_target_power": required,
                             "approximate_mde_at_target_power": normal_approximation_mde(independent_clusters=n, cluster_difference_sd=math.sqrt(discordance), alpha=alpha, power=target_power)})
    return {"method": "two-sided normal approximation to paired binary difference with assumed discordance",
            "alpha": alpha, "target_power": target_power, "baseline_accuracy_for_rer": baseline_accuracy,
            "assumptions": "Independent original scenarios; discordance 5/10/20 percent; no multiplicity adjustment in planning. Correlated variants add no independent N. Small discordance counts make exact tests discrete.",
            "interpretation": f"1024 scenarios are not guaranteed to detect 1-2 point gains. Power depends on paired discordance and decreases under multiplicity correction. RER effect assumes a {baseline_accuracy:.1%} baseline; it is not measured baseline performance.",
            "rows": rows}
