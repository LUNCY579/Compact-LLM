"""Sibling descriptive DEV diagnostics; the reviewed native package is unchanged.

Inputs are untrusted scored summaries, never authenticated model evidence. This
module has no file/model/bank access and creates no selection or acceptance gates.
"""
from __future__ import annotations

import math
from collections import Counter

from experiments.titan_quality_repair_v1 import registry
from experiments.titan_quality_repair_v1.schema import EXPERIMENT, FAMILIES, REPAIR_FAMILIES, SEEDS, STYLES, digest
from experiments.titan_quality_repair_v1.selection import development_gates
from experiments.titan_quality_repair_v1.statistics import paired_worlds

FLAGS = ("semantic_correct", "strict_correct", "abstained", "unsupported_answer",
         "unnecessary_abstention", "format_valid", "no_answer")


def prospective_spec():
    return {"experiment": EXPERIMENT, "diagnostic_version": 1, "split": "dev",
        "canonical_worlds_per_family": 64, "canonical_worlds": 448,
        "attached_variants_per_repair_family": 16, "attached_variants": 64,
        "total_scored_prompts": 512, "seeds": list(SEEDS), "recipes": 8,
        "primary_projections": ["family x support_expansion tag gain/loss transitions",
            "supported-answer versus correct-abstention gain/loss; adverse-rate transitions",
            "same-world corrected/lost overlap across both fixed seeds",
            "family x attached style compared with its own canonical parents"],
        "expansion_label_boundary": "False means nonexpanded documented operands/no-wrap tags, not established full historical support; ledger balance support is unknown.",
        "macro_definition": "Equal four-repair-family canonical macro only; no pooled seven-family macro.",
        "variant_independent_n": 0, "extra_model_calls": 0, "new_decision_gates": 0,
        "final_access": False, "selection_authority": False, "native_execution_authorized": False}


def _rate(value, n):
    return value / n if n else None


def _flags(item):
    if not isinstance(item, dict):
        raise ValueError("STOP_ATTRIBUTION_SCORE_ITEM_TYPE")
    for name in (*FLAGS, "answerable", "support_expansion"):
        if type(item.get(name)) is not bool:
            raise ValueError("STOP_ATTRIBUTION_SCORE_FLAG_TYPE:" + name)
    if item["strict_correct"] and (not item["semantic_correct"] or not item["format_valid"]):
        raise ValueError("STOP_ATTRIBUTION_INCONSISTENT_STRICT_SCORE")
    if item["no_answer"] and any(item[name] for name in ("semantic_correct", "strict_correct", "abstained", "format_valid")):
        raise ValueError("STOP_ATTRIBUTION_BLANK_SCORE_INCONSISTENT")
    if item["answerable"] and item["abstained"] and item["semantic_correct"]:
        raise ValueError("STOP_ATTRIBUTION_SUPPORTED_ABSTENTION_CORRECTNESS_CONTRADICTION")
    if item["unsupported_answer"] != (not item["answerable"] and not item["abstained"] and not item["no_answer"]):
        raise ValueError("STOP_ATTRIBUTION_UNSUPPORTED_FLAG_INCONSISTENT")
    if item["unnecessary_abstention"] != (item["answerable"] and item["abstained"]):
        raise ValueError("STOP_ATTRIBUTION_ABSTENTION_FLAG_INCONSISTENT")
    if not item["answerable"] and item["semantic_correct"] != item["abstained"]:
        raise ValueError("STOP_ATTRIBUTION_UNANSWERABLE_SEMANTICS_INCONSISTENT")


def _validate_summary(summary):
    if not isinstance(summary, dict) or summary.get("split") != "dev":
        raise ValueError("STOP_ATTRIBUTION_DEVELOPMENT_ONLY")
    if not isinstance(summary.get("primary"), dict) or set(summary["primary"]) != set(FAMILIES):
        raise ValueError("STOP_ATTRIBUTION_EXACT_PRIMARY_FAMILIES")
    worlds, ids, parents = set(), set(), {}
    for family in FAMILIES:
        items = summary.get("primary", {}).get(family, [])
        if not isinstance(items, list) or len(items) != 64:
            raise ValueError("STOP_ATTRIBUTION_EXACT_FAMILY_WORLD_COUNT")
        for item in items:
            _flags(item)
            if item.get("family") != family or item.get("variant_id") != "canonical":
                raise ValueError("STOP_ATTRIBUTION_CANONICAL_MEMBERSHIP")
            world, row_id = item.get("world_id"), item.get("row_id")
            if not isinstance(world, str) or not world or world in worlds or not isinstance(row_id, str) or not row_id or row_id in ids:
                raise ValueError("STOP_ATTRIBUTION_DUPLICATE_OR_EMPTY_ID")
            worlds.add(world); ids.add(row_id); parents[world] = item
    variants = summary.get("variants", {})
    if not isinstance(variants, dict):
        raise ValueError("STOP_ATTRIBUTION_VARIANT_CONTAINER")
    items = variants.get("items", [])
    if not isinstance(items, list) or len(items) != 64 or type(variants.get("independent_n")) is not int or variants["independent_n"] != 0:
        raise ValueError("STOP_ATTRIBUTION_EXACT_VARIANT_COUNT")
    observed, attached = Counter(), set()
    for item in items:
        _flags(item)
        world, style, row_id = item.get("world_id"), item.get("variant_id"), item.get("row_id")
        parent = parents.get(world)
        if parent is None or item.get("family") not in REPAIR_FAMILIES or style not in STYLES[1:]:
            raise ValueError("STOP_ATTRIBUTION_ATTACHED_VARIANT_MEMBERSHIP")
        if world in attached or not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError("STOP_ATTRIBUTION_DUPLICATE_VARIANT_OR_ROW")
        if any(item[name] != parent[name] for name in ("family", "answerable", "support_expansion")):
            raise ValueError("STOP_ATTRIBUTION_VARIANT_PARENT_IDENTITY")
        attached.add(world); ids.add(row_id); observed[item["family"], style] += 1
    expected = Counter((family, STYLES[1 + (index + offset) % 3])
                       for offset, family in enumerate(REPAIR_FAMILIES) for index in range(16))
    if observed != expected or len(worlds) != 448 or len(ids) != 512:
        raise ValueError("STOP_ATTRIBUTION_REGISTERED_VARIANT_PROJECTION")
    return parents


def _metrics(items):
    supported = [item for item in items if item["answerable"]]
    unsupported = [item for item in items if not item["answerable"]]
    return {"n": len(items), "supported_n": len(supported), "unsupported_n": len(unsupported),
        "semantic_accuracy": _rate(sum(item["semantic_correct"] for item in items), len(items)),
        "strict_accuracy": _rate(sum(item["strict_correct"] for item in items), len(items)),
        "supported_answer_accuracy": _rate(sum(item["semantic_correct"] for item in supported), len(supported)),
        "unsupported_answer_rate": _rate(sum(item["unsupported_answer"] for item in unsupported), len(unsupported)),
        "unnecessary_abstention_rate": _rate(sum(item["unnecessary_abstention"] for item in supported), len(supported)),
        "no_answer_rate": _rate(sum(item["no_answer"] for item in items), len(items)),
        "coverage": _rate(sum(not item["abstained"] and not item["no_answer"] for item in items), len(items))}


def _transitions(pairs, flag):
    counts = {"gained": 0, "lost": 0, "true_both": 0, "false_both": 0}
    for before, after in pairs:
        name = "true_both" if before[flag] and after[flag] else "lost" if before[flag] else "gained" if after[flag] else "false_both"
        counts[name] += 1
    return {**counts, "n": len(pairs), "net_true_count_change": counts["gained"] - counts["lost"],
            "net_rate_change": _rate(counts["gained"] - counts["lost"], len(pairs))}


def _adverse(pairs, flag):
    result = _transitions(pairs, flag)
    return {"introduced": result["gained"], "removed": result["lost"],
            "persistent": result["true_both"], "absent_both": result["false_both"],
            "n": result["n"], "adverse_rate_change": result["net_rate_change"]}


def _paired_description(pairs, family=None):
    semantic, strict = _transitions(pairs, "semantic_correct"), _transitions(pairs, "strict_correct")
    supported = [pair for pair in pairs if pair[0]["answerable"]]
    unsupported = [pair for pair in pairs if not pair[0]["answerable"]]
    return {"baseline": _metrics([before for before, _ in pairs]), "candidate": _metrics([after for _, after in pairs]),
        "semantic": semantic, "strict": strict,
        "supported_answer_semantic": _transitions(supported, "semantic_correct"),
        "correct_abstention_on_unanswerable": _transitions(unsupported, "semantic_correct"),
        "unsupported_answer": _adverse(unsupported, "unsupported_answer"),
        "unnecessary_abstention": _adverse(supported, "unnecessary_abstention"),
        "no_answer": _adverse(pairs, "no_answer"),
        "contribution_to_family_semantic_delta": semantic["net_true_count_change"] / 64 if family else None,
        "contribution_to_repair_macro_semantic_delta": semantic["net_true_count_change"] / (64 * 4) if family in REPAIR_FAMILIES else None}


def _overlap(left, right):
    union = left | right
    return {"seed17_n": len(left), "seed29_n": len(right), "common_n": len(left & right),
            "seed17_only_n": len(left - right), "seed29_only_n": len(right - left),
            "union_n": len(union), "jaccard": len(left & right) / len(union) if union else None,
            "common_world_ids": sorted(left & right)}


def analyze_gain_attribution(baseline=None, trial_summaries=None, recipe_registry=None):
    """Describe exact 8x2 DEV cells; never rank, substitute or qualify a candidate."""
    frozen = registry.build_registry() if recipe_registry is None else registry.validate_registry(recipe_registry)
    common = {"experiment": EXPERIMENT, "diagnostic_version": 1, "spec": prospective_spec(),
        "registry_sha256": frozen["registry_sha256"], "input_authentication": "UNTRUSTED_SUMMARIES_NOT_NATIVE_RECEIPT_VERIFICATION",
        "quality_gain_verified": False, "native_evidence": False, "selection_authority": False,
        "new_gates": False, "extra_model_calls": 0, "final_access": False,
        "seed_cherry_picking": False, "final_recipe_fallback_allowed": False,
        "limits": ["No native model observations are authenticated by this descriptive report.",
            "Support tags describe operand/wrap expansion; ledger historical balance support remains unestablished.",
            "Agreement of aggregate signs or corrected worlds across two seeds is not deterministic or generalization proof.",
            "Style contrasts use only matched attached parents; their repeated rows never add independent worlds.",
            "Final hypotheses, recipes, eligibility, thresholds and 48 endpoints remain unchanged."]}
    if baseline is not None:
        baseline_parents = _validate_summary(baseline)
    if trial_summaries is None or trial_summaries == {}:
        return {**common, "status": "UNMEASURED", "analyzed_trials": 0,
                "baseline_summary_sha256": digest(baseline) if baseline is not None else None,
                "recipes": {}}
    if baseline is None:
        raise ValueError("STOP_ATTRIBUTION_MISSING_BASELINE")
    recipe_ids = [recipe["recipe_id"] for recipe in frozen["recipes"]]
    if not isinstance(trial_summaries, dict) or set(trial_summaries) != set(recipe_ids):
        raise ValueError("STOP_ATTRIBUTION_EXACT_RECIPE_GRID")
    result = {}
    for recipe_id in recipe_ids:
        observations = trial_summaries[recipe_id]
        if (not isinstance(observations, dict) or any(type(seed) is not int for seed in observations)
                or set(observations) != set(SEEDS)):
            raise ValueError("STOP_ATTRIBUTION_EXACT_TWO_SEEDS")
        seed_reports, changed = {}, {}
        for seed in SEEDS:
            candidate = observations[seed]
            candidate_parents = _validate_summary(candidate)
            gates = development_gates(baseline, candidate)
            families = {}
            changed[seed] = {}
            for family in FAMILIES:
                pairs = paired_worlds(baseline, candidate, family)
                strata = {"expanded": _paired_description([pair for pair in pairs if pair[0]["support_expansion"]], family),
                          "nonexpanded_operand_tag": _paired_description([pair for pair in pairs if not pair[0]["support_expansion"]], family)}
                families[family] = {"all": _paired_description(pairs, family), "by_support_expansion": strata}
                changed[seed][family] = {
                    "corrected": {left["world_id"] for left, right in pairs if not left["semantic_correct"] and right["semantic_correct"]},
                    "lost": {left["world_id"] for left, right in pairs if left["semantic_correct"] and not right["semantic_correct"]}}
            variant_map = {item["row_id"]: item for item in candidate["variants"]["items"]}
            styles = {}
            for family in REPAIR_FAMILIES:
                styles[family] = {}
                for style in STYLES[1:]:
                    selected = [item for item in baseline["variants"]["items"] if item["family"] == family and item["variant_id"] == style]
                    variants = [(item, variant_map[item["row_id"]]) for item in selected]
                    parents = [(baseline_parents[item["world_id"]], candidate_parents[item["world_id"]]) for item in selected]
                    variant_desc, parent_desc = _paired_description(variants), _paired_description(parents)
                    styles[family][style] = {"attached_row_n": len(variants), "matched_parent_world_n": len(parents),
                        "variant_independent_n": 0, "variant_delta": variant_desc, "same_parent_canonical_delta": parent_desc,
                        "style_minus_parent_semantic_improvement": variant_desc["semantic"]["net_rate_change"] - parent_desc["semantic"]["net_rate_change"],
                        "style_minus_parent_strict_improvement": variant_desc["strict"]["net_rate_change"] - parent_desc["strict"]["net_rate_change"]}
            by_support = {tag: math.fsum(families[family]["by_support_expansion"][tag]["contribution_to_repair_macro_semantic_delta"]
                                        for family in REPAIR_FAMILIES) for tag in ("expanded", "nonexpanded_operand_tag")}
            macro = math.fsum(by_support.values())
            if not math.isclose(macro, gates["repair_macro_gain"], abs_tol=1e-12):
                raise ValueError("STOP_ATTRIBUTION_MACRO_RECONCILIATION")
            seed_reports[str(seed)] = {"summary_sha256": digest(candidate), "families": families,
                "repair_macro_semantic_delta": macro, "repair_macro_contributions_by_support_tag": by_support,
                "attached_style_comparisons": styles, "frozen_development_gate_passed": gates["passed"],
                "frozen_development_gate_failures": gates["failures"]}
        result[recipe_id] = {"seeds": seed_reports,
            "same_world_cross_seed_overlap": {family: {name: _overlap(changed[17][family][name], changed[29][family][name])
                                                       for name in ("corrected", "lost")} for family in FAMILIES}}
    return {**common, "status": "DESCRIPTIVE_DEV_INPUTS_NO_VERIFIED_GAIN", "analyzed_trials": 16,
            "baseline_summary_sha256": digest(baseline), "recipes": result}
