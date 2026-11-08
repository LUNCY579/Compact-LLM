"""Frozen development selection and final acceptance; never launches native work."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from .schema import EXPERIMENT, FAMILIES, REPAIR_FAMILIES, RETENTION_FAMILIES, SEEDS, canonical, digest
from .statistics import (ENDPOINT_ALPHA, ENDPOINT_REGISTRY, damage_bounds, macro_gain_interval,
                         paired_worlds, supported_population_loss)

SOURCE_SHA256 = "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EPSILON = 1e-12  # arithmetic comparison only; never widens a clinical/design margin


@dataclass(frozen=True)
class SelectionReceipt:
    """Canonical immutable decision text binds one recipe, including no candidate."""
    decision_json: str
    sha256: str

    def __post_init__(self):
        decision = json.loads(self.decision_json)
        if canonical(decision) != self.decision_json or digest(decision) != self.sha256:
            raise ValueError("STOP_SELECTION_RECEIPT_INTEGRITY")
        if decision.get("experiment") != EXPERIMENT or decision.get("seeds") != list(SEEDS):
            raise ValueError("STOP_SELECTION_RECEIPT_IDENTITY")
        if (decision.get("deliverable_seed") != 17 or decision.get("replication_seed") != 29
                or decision.get("fallback_permitted") is not False or decision.get("native_authorization") is not False
                or "selected_recipe_id" not in decision or len(decision.get("audits", {})) != 8):
            raise ValueError("STOP_SELECTION_RECEIPT_RULES")
        selected = decision["selected_recipe_id"]
        if selected is not None and not decision["audits"].get(selected, {}).get("eligible"):
            raise ValueError("STOP_SELECTION_RECEIPT_INELIGIBLE_RECIPE")

    @property
    def selected_recipe_id(self):
        return json.loads(self.decision_json)["selected_recipe_id"]

    def as_dict(self):
        return {"decision": json.loads(self.decision_json), "sha256": self.sha256}

    def require_recipe(self, recipe_id):
        if self.selected_recipe_id is None or recipe_id != self.selected_recipe_id:
            raise ValueError("STOP_NO_CANDIDATE_OR_RECIPE_FALLBACK")


def _protocol_summary(summary, split, worlds, variants):
    if summary.get("split") != split:
        raise ValueError("STOP_SELECTION_SPLIT")
    for family in FAMILIES:
        items = summary.get("primary", {}).get(family, [])
        if len(items) != worlds:
            raise ValueError("STOP_INCOMPLETE_FAMILY_WORLD_COUNT")
        paired_worlds(summary, summary, family)
    items = summary.get("variants", {}).get("items", [])
    if len(items) != variants or summary.get("variants", {}).get("independent_n") != 0:
        raise ValueError("STOP_INCOMPLETE_OR_INDEPENDENT_VARIANTS")


def _paired_variants(baseline, candidate):
    fields = ("row_id", "world_id", "family", "variant_id", "answerable", "support_expansion")
    def membership(summary):
        items = [tuple(item.get(field) for field in fields) for item in summary["variants"]["items"]]
        if len(set(items)) != len(items):
            raise ValueError("STOP_DUPLICATE_VARIANT_MEMBERSHIP")
        return set(items)
    if membership(baseline) != membership(candidate):
        raise ValueError("STOP_UNPAIRED_VARIANT_MEMBERSHIP")


def _rates(pairs, side):
    items = [pair[side] for pair in pairs]
    answerable = [item for item in items if item["answerable"]]
    unsupported = [item for item in items if not item["answerable"]]
    return {
        "semantic": sum(item["semantic_correct"] for item in items) / len(items),
        "strict": sum(item["strict_correct"] for item in items) / len(items),
        "supported": sum(item["semantic_correct"] for item in answerable) / len(answerable) if answerable else None,
        "unsupported": sum(item["unsupported_answer"] for item in unsupported) / len(unsupported) if unsupported else None,
        "unnecessary": sum(item["unnecessary_abstention"] for item in answerable) / len(answerable) if answerable else None,
    }


def development_gates(baseline, candidate):
    """Point gates apply separately to each seed; expose every family reversal."""
    _protocol_summary(baseline, "dev", 64, 64)
    _protocol_summary(candidate, "dev", 64, 64)
    _paired_variants(baseline, candidate)
    failures, families, repair_pairs = [], {}, []
    for family in FAMILIES:
        pairs = paired_worlds(baseline, candidate, family)
        left, right = _rates(pairs, 0), _rates(pairs, 1)
        families[family] = {"baseline": left, "candidate": right,
                            "semantic_delta": right["semantic"] - left["semantic"],
                            "strict_delta": right["strict"] - left["strict"]}
        if family in REPAIR_FAMILIES:
            repair_pairs.extend(pairs)
            if left["supported"] is None or right["supported"] is None:
                failures.append(f"{family}:supported_denominator_zero")
            elif right["supported"] + _EPSILON < left["supported"] - .02:
                failures.append(f"{family}:supported_answer_decline")
        if family in RETENTION_FAMILIES:
            for metric in ("semantic", "strict"):
                if right[metric] + _EPSILON < left[metric] - .02:
                    failures.append(f"{family}:retention_{metric}_decline")
    left, right = _rates(repair_pairs, 0), _rates(repair_pairs, 1)
    for metric in ("unsupported", "unnecessary"):
        if left[metric] is None or right[metric] is None:
            failures.append(f"repair:{metric}_denominator_zero")
        elif right[metric] > left[metric] + _EPSILON:
            failures.append(f"repair:{metric}_increase")
    gain = math.fsum(families[family]["semantic_delta"] for family in REPAIR_FAMILIES) / 4
    population = sum(left["semantic_correct"] and not right["semantic_correct"]
                     for family in FAMILIES for left, right in paired_worlds(baseline, candidate, family)) / (7 * 64)
    return {"passed": not failures, "failures": failures, "families": families,
            "repair_baseline": left, "repair_candidate": right,
            "repair_macro_gain": gain, "population_semantic_damage": population}


def select_recipe(baseline, trial_summaries, recipe_ids):
    """All eight recipes and fixed seeds required, without intermediate selection."""
    recipe_ids = tuple(recipe_ids)
    if (len(recipe_ids) != 8 or any(not isinstance(recipe, str) or not recipe for recipe in recipe_ids)
            or len(set(recipe_ids)) != 8 or set(trial_summaries) != set(recipe_ids)):
        raise ValueError("STOP_INCOMPLETE_EIGHT_RECIPE_REGISTRY")
    _protocol_summary(baseline, "dev", 64, 64)
    audits, eligible = {}, []
    for recipe in sorted(recipe_ids):
        seeds = trial_summaries[recipe]
        if set(seeds) != set(SEEDS):
            raise ValueError("STOP_FIXED_SEED_COVERAGE")
        gates = {str(seed): development_gates(baseline, seeds[seed]) for seed in SEEDS}
        gain = math.fsum(gates[str(seed)]["repair_macro_gain"] for seed in SEEDS) / 2
        damage = math.fsum(gates[str(seed)]["population_semantic_damage"] for seed in SEEDS) / 2
        unsupported = math.fsum(gates[str(seed)]["repair_candidate"]["unsupported"]
                                if gates[str(seed)]["repair_candidate"]["unsupported"] is not None else 1.0
                                for seed in SEEDS) / 2
        passed = all(gates[str(seed)]["passed"] for seed in SEEDS)
        audits[recipe] = {"eligible": passed, "seeds": gates,
                          "rank": [-gain, damage, unsupported, recipe]}
        if passed:
            eligible.append(((-gain, damage, unsupported, recipe), recipe))
    selected = min(eligible)[1] if eligible else None
    decision = {
        "schema_version": 1, "experiment": EXPERIMENT, "seeds": list(SEEDS),
        "selected_recipe_id": selected, "deliverable_seed": 17, "replication_seed": 29,
        "status": "SELECTED_PENDING_EXPORT_RECHECK" if selected else "STOP_NO_ELIGIBLE_RECIPE",
        "baseline_summary_sha256": digest(baseline),
        "trial_summary_sha256": {recipe: {str(seed): digest(trial_summaries[recipe][seed]) for seed in SEEDS}
                                 for recipe in sorted(recipe_ids)},
        "audits": audits, "fallback_permitted": False, "native_authorization": False,
    }
    return SelectionReceipt(canonical(decision), digest(decision))


def validate_final_gate(receipt, evidence):
    """Check supplied verified proof fields; this pure predicate is not a permit.

    A native custody/access wrapper must additionally verify receipt file hashes,
    process ownership, resource floors and atomic one-shot final access accounting.
    """
    if not isinstance(receipt, SelectionReceipt) or not isinstance(evidence, dict):
        raise ValueError("STOP_FINAL_GATE_PROOF_TYPE")
    receipt.require_recipe(evidence.get("selected_recipe_id"))
    if evidence.get("experiment") != EXPERIMENT or evidence.get("selection_sha256") != receipt.sha256:
        raise ValueError("STOP_FINAL_SELECTION_BINDING")
    exports = evidence.get("exports", {})
    if set(exports) != {"17", "29"}:
        raise ValueError("STOP_FINAL_EXPORT_SEED_COVERAGE")
    for seed, export in exports.items():
        if (export.get("architecture") != "Qwen3ForCausalLM" or export.get("dtype") != "float32"
                or export.get("custom_abstention") is not False or export.get("dev_passed") is not True
                or export.get("recipe_id") != receipt.selected_recipe_id or export.get("seed") != int(seed)):
            raise ValueError("STOP_FINAL_EXPORT_IDENTITY_OR_DEV_GATE")
        for field in ("artifact_sha256", "dev_receipt_sha256"):
            if not _SHA256.fullmatch(str(export.get(field, ""))):
                raise ValueError("STOP_FINAL_EXPORT_HASH")
    qualification = evidence.get("qualification", {})
    if (qualification.get("passed") is not True or qualification.get("source_sha256") != SOURCE_SHA256
            or not _SHA256.fullmatch(str(qualification.get("receipt_sha256", "")))):
        raise ValueError("STOP_FINAL_QUALIFICATION")
    permit = evidence.get("operator_permit", {})
    if (permit.get("experiment") != EXPERIMENT or permit.get("source_sha256") != SOURCE_SHA256
            or permit.get("stage") != "final_evaluation" or not permit.get("hardware_slot")
            or not _SHA256.fullmatch(str(permit.get("permit_sha256", "")))):
        raise ValueError("STOP_FINAL_OPERATOR_PERMIT")
    ledger = evidence.get("ledger", {})
    if ledger.get("final_unopened") is not True or ledger.get("spent") is not False:
        raise ValueError("STOP_FINAL_ALREADY_EXPOSED_OR_SPENT")
    return {"ready_for_verified_atomic_access_wrapper": True,
            "selected_recipe_id": receipt.selected_recipe_id, "selection_sha256": receipt.sha256,
            "proof_sha256": digest(evidence), "native_execution_authorized_by_this_function": False}


def final_acceptance(baseline, candidates, *, receipt, access_evidence):
    """Compute frozen 48-endpoint decision after authorized final access only."""
    validate_final_gate(receipt, access_evidence)
    if set(candidates) != set(SEEDS):
        raise ValueError("STOP_FINAL_FIXED_SEEDS")
    _protocol_summary(baseline, "final", 512, 512)
    seed_results = {}
    for seed in SEEDS:
        candidate = candidates[seed]
        _protocol_summary(candidate, "final", 512, 512)
        _paired_variants(baseline, candidate)
        failures, family_results, repair_pairs = [], {}, []
        for family in FAMILIES:
            pairs = paired_worlds(baseline, candidate, family)
            left, right = _rates(pairs, 0), _rates(pairs, 1)
            bounds = damage_bounds(pairs)
            for endpoint, result in bounds.items():
                ceiling = .10 if endpoint == "conditional_semantic_damage" else .05
                if result["upper"] is None or result["upper"] > ceiling + _EPSILON:
                    failures.append(f"{family}:{endpoint}")
            if family in REPAIR_FAMILIES:
                repair_pairs.extend(pairs)
                if left["supported"] is None or right["supported"] is None or right["supported"] + _EPSILON < left["supported"] - .02:
                    failures.append(f"{family}:supported_answer_decline_or_missing")
            if family in RETENTION_FAMILIES:
                for metric in ("semantic", "strict"):
                    if right[metric] + _EPSILON < left[metric] - .02:
                        failures.append(f"{family}:retention_{metric}_decline")
            if family in {"answerability", "grounded_qa"}:
                bounds["supported_population_loss"] = supported_population_loss(pairs)
                upper = bounds["supported_population_loss"]["upper"]
                if upper is None or upper > .05 + _EPSILON:
                    failures.append(f"{family}:supported_population_loss")
            family_results[family] = {"baseline": left, "candidate": right, "bounds": bounds}
        left, right = _rates(repair_pairs, 0), _rates(repair_pairs, 1)
        for metric in ("unsupported", "unnecessary"):
            if left[metric] is None or right[metric] is None or right[metric] > left[metric] + _EPSILON:
                failures.append(f"repair:{metric}_increase_or_missing")
        gain = macro_gain_interval(baseline, candidate)
        if gain["point"] + _EPSILON < .03 or gain["lower"] <= 0:
            failures.append("repair:macro_gain_point_or_simultaneous_lower")
        seed_results[str(seed)] = {"passed": not failures, "failures": failures,
                                   "families": family_results, "repair_gain": gain,
                                   "repair_baseline": left, "repair_candidate": right}
    accepted = all(result["passed"] for result in seed_results.values())
    return {"experiment": EXPERIMENT, "selected_recipe_id": receipt.selected_recipe_id,
            "selection_sha256": receipt.sha256, "accepted": accepted,
            "measurement_only": True, "native_capability_claim": False,
            "status": "QUALITY_ACCEPTED_NO_PROMOTION" if accepted else "STOP_FAILED_OR_INCONCLUSIVE_FINAL",
            "deliverable_seed": 17, "replication_seed": 29, "seeds": seed_results,
            "endpoint_alpha": ENDPOINT_ALPHA, "endpoint_registry": list(ENDPOINT_REGISTRY),
            "baseline_summary_sha256": digest(baseline),
            "candidate_summary_sha256": {str(seed): digest(candidates[seed]) for seed in SEEDS},
            "fallback_permitted": False, "promotion_authorized": False}
