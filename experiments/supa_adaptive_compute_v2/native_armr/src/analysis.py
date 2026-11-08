"""Track A decisions from complete, independently rescored frozen evidence.

This module never selects a new primary arm. Architecture and linked-diagnostic
results cannot yield an efficacy GO, even when their scores look favorable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

from .operators import REGISTERED_ARMS
from .protocol import LEARNED_PROTOCOL_VERSION
from shared.statistics import paired_analysis, comparison_annotations, _clustered
from shared.scoring import score_response
from shared.reporting import quality_cost_frontier
from armr.metrics import output_flags


def linked_robustness(original_cases: list[dict], originals: dict[str, list[dict]],
                      variant_cases: list[dict], variants: dict[str, list[dict]], config: dict) -> dict:
    """How paired native advantages change under linked prompt interventions."""
    source_cases = {c["example_id"]: c for c in original_cases}
    source = {arm: {r["example_id"]: r for r in rows} for arm, rows in originals.items()}
    modified = {arm: {r["example_id"]: r for r in rows} for arm, rows in variants.items()}
    groups = defaultdict(list)
    for case in variant_cases:
        original = source_cases.get(case.get("source_example_id"))
        if original is None or original["scenario_id"] != case["scenario_id"] or original["prompt_sha256"] != case["source_prompt_sha256"]:
            raise ValueError("diagnostic does not bind its frozen original scenario")
        groups[case["variant"]].append((original["example_id"], case["example_id"], case["scenario_id"]))
    result = {}
    settings = config["analysis"]
    for variant, pairs in sorted(groups.items()):
        effects = {}
        for arm in REGISTERED_ARMS:
            base = [source[arm][a]["score"] for a, b, s in pairs]
            changed = [modified[arm][b]["score"] for a, b, s in pairs]
            advantage_before = [source[arm][a]["score"]-source["A0"][a]["score"] for a, b, s in pairs]
            advantage_after = [modified[arm][b]["score"]-modified["A0"][b]["score"] for a, b, s in pairs]
            delta = _clustered.cluster_differences([s for a, b, s in pairs], advantage_before, advantage_after)
            lower, upper = _clustered.cluster_bootstrap_interval(list(delta.values()),
                            samples=settings["bootstrap_replicates"], seed=settings["bootstrap_seed"])
            effects[arm] = {"original_accuracy": sum(base)/len(pairs), "variant_accuracy": sum(changed)/len(pairs),
                            "original_correct_variant_wrong": sum(a == 1 and b == 0 for a, b in zip(base, changed)),
                            "original_wrong_variant_correct": sum(a == 0 and b == 1 for a, b in zip(base, changed)),
                            "original_advantage_vs_A0": sum(advantage_before)/len(pairs),
                            "variant_advantage_vs_A0": sum(advantage_after)/len(pairs),
                            "change_in_advantage_scenario_bootstrap_95_ci": [lower, upper]}
        result[variant] = {"pairs": len(pairs), "independent_scenarios": len({s for a, b, s in pairs}), "arms": effects}
    return {"designation": "exploratory linked interventions; no new independent primary examples",
            "variants": result}


def audit_rows(cases: list[dict], by_arm: dict[str, list[dict]], config: dict,
               role: str) -> dict[str, list[dict]]:
    """Reconcile complete immutable IDs, frozen metadata, scores and validity."""
    counts = {"architecture_dev": 128,
              "confirmatory_dev": config["data"]["target_primary_examples"],
              "confirmatory_diagnostics": config["data"]["target_diagnostic_examples"]}
    if role not in counts or len(cases) != counts[role]:
        raise ValueError("full registered bank required; partial efficacy is forbidden")
    if set(by_arm) != set(REGISTERED_ARMS):
        raise ValueError("all five registered arms, including complete A0 and AM, are required")
    expected = {case["example_id"]: case for case in cases}
    if len(expected) != len(cases):
        raise ValueError("duplicate frozen case identity")
    if role != "confirmatory_diagnostics" and len({c["scenario_id"] for c in cases}) != len(cases):
        raise ValueError("primary/mechanical bank requires independent original scenarios")
    expected_split = "confirmatory_dev" if role.startswith("confirmatory") else "architecture_dev"
    for case in cases:
        if case["track"] != "native_armr" or case["split"] != expected_split:
            raise ValueError("case data role or track mismatch")
        if role != "confirmatory_diagnostics" and case["variant"] != "original":
            raise ValueError("diagnostic variants cannot enter primary accuracy")
        if role == "confirmatory_diagnostics" and case["variant"] == "original":
            raise ValueError("diagnostic file cannot contain primary originals")
    registered_families = set(config["analysis"]["preservation_families"])
    if not registered_families <= {c["family"] for c in cases}:
        raise ValueError("registered preservation families missing from bank")
    ordered = {}
    for arm, rows in by_arm.items():
        indexed = {row["example_id"]: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != set(expected):
            raise ValueError("incomplete or duplicate candidate/baseline coverage")
        for identity, row in indexed.items():
            case = expected[identity]
            for key in ("scenario_id", "family", "split", "track", "variant", "template_id", "template_unseen_in_engineering"):
                if row.get(key) != case[key]:
                    raise ValueError(f"frozen result metadata mismatch: {key}")
            if row.get("arm") != arm:
                raise ValueError("row arm mismatch")
            if row.get("safety_stopped") is not False or row.get("termination") not in {"eos", "token_cap"}:
                raise ValueError("safety-stopped/incomplete output cannot support decisions")
            tokens = row.get("token_ids")
            cap = min(config["max_new_tokens"], case.get("max_new_tokens", config["max_new_tokens"]))
            if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("malformed generation token evidence")
            if len(tokens) > cap or row["termination"] == "token_cap" and len(tokens) != cap:
                raise ValueError("generation token cap mismatch")
            if row.get("eos_emitted") != (row["termination"] == "eos") or row.get("truncated") != (row["termination"] == "token_cap"):
                raise ValueError("generation termination flags mismatch")
            rescored = score_response(row["response"], case["scorer"])
            if row.get("score") != rescored["score"] or row.get("correct") != rescored["correct"]:
                raise ValueError("stored score does not match frozen scorer")
            flags = output_flags(row["response"], tokens)
            if any(row.get(key) != value for key, value in flags.items()):
                raise ValueError("stored validity does not match raw response/tokens")
            if any(flags[key] for key in ("repetition_loop", "script_corruption", "replacement_count", "control_count")):
                raise ValueError("hard generation safety failure")
            nll = row.get("gold_nll")
            if not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
                raise ValueError("this exact-answer bank requires finite canonical-answer NLL evidence")
            for key in ("wall_seconds", "generation_dense_matmul_flops_estimate", "peak_rss_bytes"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"missing/nonfinite positive cost evidence: {key}")
            if row["wall_seconds"] > config["max_prompt_seconds"] or row["peak_rss_bytes"] > config["max_rss_gib"] * 2**30:
                raise ValueError("registered execution resource guard exceeded")
            low, high = config["hidden_rms_ratio_bounds"]
            for key in ("trajectory_min_rms_ratio", "trajectory_max_rms_ratio"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                    raise ValueError("missing/out-of-bounds hidden trajectory evidence")
            if row["trajectory_min_rms_ratio"] > row["trajectory_max_rms_ratio"]:
                raise ValueError("reversed hidden trajectory extrema")
        ordered[arm] = [indexed[c["example_id"]] for c in cases]
    return ordered


def analyze_complete_bank(cases: list[dict], by_arm: dict[str, list[dict]], config: dict,
                          role: str) -> dict:
    rows = audit_rows(cases, by_arm, config, role)
    analysis = config["analysis"]
    designation = {"architecture_dev": "candidate_selection", "confirmatory_dev": "confirmatory",
                   "confirmatory_diagnostics": "diagnostic"}[role]
    kwargs = {"seed": analysis["bootstrap_seed"], "bootstrap_samples": analysis["bootstrap_replicates"],
              "expected_example_ids": [c["example_id"] for c in cases], "analysis_role": designation}
    comparisons = {f"{arm}_vs_A0": paired_analysis(rows["A0"], rows[arm], **kwargs)
                   for arm in REGISTERED_ARMS if arm != "A0"}
    comparisons["A3_vs_AM"] = paired_analysis(rows["AM"], rows["A3"], **kwargs)
    invalid = {arm: sum(r["invalid"] for r in values) for arm, values in rows.items()}
    report = {"track": "native_armr", "role": role, "n": len(cases),
              "analysis_role": designation, "comparisons": comparisons,
              "family_counts": dict(sorted(Counter(c["family"] for c in cases).items())),
              "invalid_outputs": invalid, "independently_rescored": True,
              "efficacy_claim_licensed": False,
              "cost": {arm: {"mean_wall_seconds": sum(r["wall_seconds"] for r in values)/len(values),
                              "total_generation_dense_matmul_flops_estimate": sum(r["generation_dense_matmul_flops_estimate"] for r in values),
                              "peak_case_rss_bytes": max(r["peak_rss_bytes"] for r in values)}
                       for arm, values in rows.items()}}
    report["quality_cost_frontier"] = quality_cost_frontier({
        arm: {"accuracy": sum(r["score"] for r in values)/len(values),
              "mean_wall_seconds": report["cost"][arm]["mean_wall_seconds"],
              "flops": report["cost"][arm]["total_generation_dense_matmul_flops_estimate"],
              "added_parameters": (config["control"]["additional_parameters"]
                                   if config.get("protocol_version") == LEARNED_PROTOCOL_VERSION
                                   else config["control"]["additional_matrix_parameters"]) if arm == "AM" else 0,
              "peak_memory_bytes": report["cost"][arm]["peak_case_rss_bytes"]}
        for arm, values in rows.items()})
    if config.get("protocol_version") == LEARNED_PROTOCOL_VERSION:
        report["control_compute_matching"] = {
            "kind": "pretrained_parallel_ffn", "source_layers": [12, 13, 14],
            "added_resident_copied_parameters": config["control"]["additional_parameters"],
            "added_active_parameters": config["control"]["additional_active_parameters"],
            "matrix_parameters": config["control"]["additional_matrix_parameters"],
            "normalization_parameters": config["control"]["additional_normalization_parameters"],
            "newly_learned_parameters": 0,
            "logical_unique_coefficients_beyond_backbone": 0,
            "control_added_matmul_flops_per_token": config["control"]["additional_matmul_flops_per_token"],
            "a3_added_matmul_flops_per_token": 62914560,
            "dominant_matmul_ratio_to_a3": .9,
            "interpretation_limit": "Comparison with this fixed pretrained token-wise alternative uses 10% less added dominant linear work than A3; attention products, normalization and instrumentation are excluded. Measured latency and memory must be considered separately.",
        }
    if role != "confirmatory_dev":
        valid = all(invalid[arm] <= invalid["A0"] for arm in REGISTERED_ARMS)
        report["decision"] = ("MECHANICS_PASS_NO_EFFICACY_CLAIM" if valid else "MECHANICS_REJECT") if role == "architecture_dev" else "DIAGNOSTIC_ONLY_NO_EFFICACY_CLAIM"
        report["interpretation"] = "All registered cases and controls audited; this role cannot establish a capability result."
        return report

    primary = comparisons["A3_vs_A0"]["overall"]
    matched = comparisons["A3_vs_AM"]["overall"]
    alpha = analysis["alpha_two_sided"]
    supported = lambda value: value["mcnemar"]["p_value_two_sided"] <= alpha and value["scenario_bootstrap_ci"]["lower"] > analysis["minimum_confidence_interval_lower_bound_exclusive"]
    first_pass = supported(primary)
    matched_pass = first_pass and supported(matched)
    meaningful = (primary["accuracy_difference"] >= analysis["minimum_absolute_accuracy_gain"] or
                  primary["relative_error_reduction"] is not None and
                  primary["relative_error_reduction"] >= analysis["minimum_relative_error_reduction"])
    report["hierarchy"] = {"A3_vs_A0": {"tested": True, "pass": first_pass},
                           "A3_vs_AM": {"tested": first_pass, "pass": matched_pass,
                                        "p_value_status": "confirmatory" if first_pass else "descriptive_only_hierarchy_closed"}}
    secondary = {f"{arm}_vs_{base}": comparisons[f"{arm}_vs_{base}"]["overall"]["mcnemar"]["p_value_two_sided"]
                 for arm, base in analysis["secondary_comparisons_holm"]}
    report["secondary_holm"] = comparison_annotations(secondary, analysis_role="confirmatory")
    preservation_ids = [i for i, c in enumerate(cases) if c["family"] in analysis["preservation_families"]]
    differences = [rows["A3"][i]["score"]-rows["A0"][i]["score"] for i in preservation_ids]
    # Lower end of a central 90% interval is the preregistered one-sided 95% bound.
    lower, _ = _clustered.cluster_bootstrap_interval(differences, confidence=.90,
                                                    samples=analysis["bootstrap_replicates"], seed=analysis["bootstrap_seed"])
    preservation_pass = lower > -analysis["preservation_noninferiority_margin"]
    validity_pass = invalid["A3"]-invalid["A0"] <= analysis["maximum_additional_invalid_outputs"]
    report["preservation"] = {"families": analysis["preservation_families"], "n": len(preservation_ids),
                              "accuracy_difference": sum(differences)/len(differences),
                              "one_sided_95_lower_bound": lower, "margin": analysis["preservation_noninferiority_margin"],
                              "method": "paired scenario percentile bootstrap; originals only", "pass": preservation_pass}
    report["primary_gates"] = {"meaningful_effect": meaningful, "validity": validity_pass,
                               "preservation": preservation_pass, "safety": True,
                               "cost": "disclosed; no preregistered scalar cost cutoff"}
    if first_pass and matched_pass and meaningful and preservation_pass and validity_pass:
        report.update(decision="GO_NATIVE_RECURRENCE", efficacy_claim_licensed=True,
                      interpretation="A3 beats intact and the registered generic matched-compute control; inference remains limited to this frozen development bank.")
        if config.get("protocol_version") == LEARNED_PROTOCOL_VERSION:
            report["interpretation"] = "A3 beats intact and this registered pretrained parallel FFN control at 90% of A3's added dominant linear work. This supports the particular full block operator over this comparator on this frozen development bank, without establishing superiority over all equal-compute algorithms or isolating attention from block order."
    elif first_pass and meaningful and preservation_pass and validity_pass and not matched_pass:
        report.update(decision="EXTRA_COMPUTE_SIGNAL_RECURRENCE_UNRESOLVED",
                      interpretation="A3 beats A0, but superiority over matched compute was not established.")
    else:
        report.update(decision="NO_GO_OR_INCONCLUSIVE",
                      interpretation="The complete preregistered recurrence GO conjunction was not established. Failure to reject is not equivalence; family patterns remain exploratory.")
    return report
