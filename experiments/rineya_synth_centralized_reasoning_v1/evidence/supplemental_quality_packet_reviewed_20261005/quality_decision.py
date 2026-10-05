"""Pure, OFFLINE SYNTH decision fixtures; never permission to run or promote.

This module evaluates the prospective engineering screen. Inputs are ordinary
Python objects. It never reads banks, loads a model, trains, or opens files.
Completion receipts supplied to this fixture are assertions, not native proof.

API:
    baseline: {row_key: {correct, format_correct, abstained, capped: bool}}
    candidates: {"M1"/"M2": {"rows": outcome_map, "completion": receipt}}
    cohort: {"rows": [{key, parent_id, source, role, stratum, condition,
                       rendering, prompt_hash}]}
    policy: optional required_retention_sources, long_context_qualified,
            bootstrap_samples, bootstrap_seed, diagnosed_poor_reference_resolution.
            Screen thresholds are fixed. Historical SYNTH policy is untouched.

The actual 384-row DEV schema uses condition 0/1 crossed with rendering
familiar/heldout_ledger. Generic fixtures may omit condition. Variants are
paired across all arms and grouped by parent before averaging.
Retention damage is never offset by repairs. An undefined guard denominator
blocks eligibility. Long-context qualification and broad claims are separate
from the four mandatory source-specific retention screens.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import random
import re
from typing import Any, Mapping


REQUIRED_RETENTION_SOURCES = ("arithmetic", "retrieval", "abstention", "dialogue")
ARMS = ("M1", "M2")
GAIN_THRESHOLD = Fraction(3, 100)
POLICY_ID = "SYNTH-QUALITY-3PP-ZERO-DAMAGE-20261005"
_BOOL_FIELDS = ("correct", "format_correct", "abstained", "capped")
_ROLES = {"R", "S", "RS", "retention"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ratio(value: Fraction) -> dict[str, Any]:
    return {"numerator": value.numerator, "denominator": value.denominator,
            "exact": str(value)}


def _fraction_pair(value: Any, field: str) -> Fraction:
    _require(isinstance(value, (list, tuple)) and len(value) == 2 and
             all(type(part) is int for part in value) and value[1] > 0,
             f"{field} requires an integer numerator and positive denominator")
    return Fraction(*value)


def _validate_frozen_policy(policy: Mapping[str, Any], rows: list[dict[str, Any]]) -> None:
    """Honor root packet schema without discarding its substantive constraints."""
    if "minimum_gain_fraction" in policy:
        _require(_fraction_pair(policy["minimum_gain_fraction"], "minimum_gain_fraction") == GAIN_THRESHOLD,
                 "minimum_gain_fraction must remain exactly 3/100")
    if "atomic_skill_weights" in policy:
        weights = policy["atomic_skill_weights"]
        _require(isinstance(weights, Mapping) and set(weights) == {"R", "S"},
                 "atomic_skill_weights must bind R and S")
        for role in ("R", "S"):
            _require(_fraction_pair(weights[role], f"atomic_skill_weights.{role}") == Fraction(1, 2),
                     "atomic skill weights must remain exactly one half each")
    fixed = {"no_atomic_skill_decline": True, "maximum_damaged_prior_correct_cases": 0,
             "repairs_can_cancel_damage": False, "both_candidates_must_complete": True,
             "checkpoint_selection": "final_only_27", "candidate_order": ["M1", "M2"]}
    for name, value in fixed.items():
        if name in policy:
            _require(type(policy[name]) is type(value) and policy[name] == value,
                     f"{name} cannot weaken the frozen screen")
    for name in ("native_model_execution_authorized", "native_scoring_authorized", "training_authorized",
                 "confirmation_authorized", "promotion_authorized", "native_execution_permitted",
                 "quality_claim_permitted", "promotion_permitted"):
        if name in policy:
            _require(type(policy[name]) is bool and policy[name] is False,
                     f"{name} must remain false for this offline fixture")
    if "replication" in policy:
        replication = policy["replication"]
        _require(isinstance(replication, Mapping) and
                 type(replication.get("additional_independent_training_draws")) is int and
                 replication.get("additional_independent_training_draws") == 2 and
                 type(replication.get("automatic_launch")) is bool and
                 replication.get("automatic_launch") is False,
                 "replication requires two additional draws and no automatic launch")
    dev_rows = [row for row in rows if row["role"] != "retention"]
    dev_parents = {row["parent_id"] for row in dev_rows}
    for name, measured in (("expected_dev_rows", len(dev_rows)), ("expected_dev_parents", len(dev_parents))):
        if name in policy:
            _require(type(policy[name]) is int and policy[name] > 0 and policy[name] == measured,
                     f"{name} does not match complete frozen DEV membership")
    for role in ("R", "S", "RS"):
        groups: dict[str, list[str]] = defaultdict(list)
        for row in dev_rows:
            if row["role"] == role:
                groups[row["parent_id"]].append(row["key"])
        if "expected_parents_per_skill" in policy:
            expected = policy["expected_parents_per_skill"]
            _require(type(expected) is int and expected > 0 and len(groups) == expected,
                     f"expected_parents_per_skill does not match complete frozen {role} membership")
        if "dev_variants_per_parent" in policy:
            expected = policy["dev_variants_per_parent"]
            _require(type(expected) is int and expected > 0 and
                     all(len(keys) == expected for keys in groups.values()),
                     f"dev_variants_per_parent does not match frozen {role} membership")
        if policy.get("schema") == "synth-quality-engineering-screen-v1" or policy.get("expected_dev_rows") == 384:
            actual_pairs = {(0, "familiar"), (0, "heldout_ledger"),
                            (1, "familiar"), (1, "heldout_ledger")}
            for parent_id in groups:
                parent_rows = [row for row in dev_rows if row["parent_id"] == parent_id]
                _require(all(type(row.get("condition")) is int and row["condition"] in (0, 1)
                             for row in parent_rows),
                         f"actual DEV parent {parent_id} requires explicit integer condition 0/1")
                pairs = {(row["condition"], row["rendering"]) for row in parent_rows}
                _require(len(parent_rows) == 4 and pairs == actual_pairs,
                         f"actual DEV parent {parent_id} requires the frozen 2x2 condition/rendering pairs")


def _mean(values: list[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def _bootstrap(values: list[Fraction], samples: int, seed: int) -> dict[str, Any]:
    """Deterministic, paired parent bootstrap, exploratory rather than powered.

    Resample whole parent fractions, including every rendering in its original
    parent. For gains, the input is one paired gain per parent. Bounds are
    nearest-rank percentiles of exact fractions and are not hypothesis tests.
    """
    if not values:
        return {"status": "UNDEFINED", "unit": "parent", "parent_count": 0}
    rng = random.Random(seed)
    boot = sorted(_mean([values[rng.randrange(len(values))]
                         for _ in values]) for _ in range(samples))
    lower_index = max(0, (samples * 25 + 999) // 1000 - 1)
    upper_index = min(samples - 1, (samples * 975 + 999) // 1000 - 1)
    return {"status": "EXPLORATORY", "unit": "parent",
            "parent_count": len(values), "samples": samples, "seed": seed,
            "method": "paired-parent bootstrap; nearest-rank 95% interval",
            "powered_claim": False, "lower": _ratio(boot[lower_index]),
            "upper": _ratio(boot[upper_index])}


def _macro_bootstrap(values_r: list[Fraction], values_s: list[Fraction],
                     samples: int, seed: int) -> dict[str, Any]:
    """Resample within each skill so unequal parent counts never change weights."""
    rng = random.Random(seed)
    boot = sorted((_mean([values_r[rng.randrange(len(values_r))] for _ in values_r]) +
                   _mean([values_s[rng.randrange(len(values_s))] for _ in values_s])) / 2
                  for _ in range(samples))
    lower_index = max(0, (samples * 25 + 999) // 1000 - 1)
    upper_index = min(samples - 1, (samples * 975 + 999) // 1000 - 1)
    return {"status": "EXPLORATORY", "unit": "parent_within_skill",
            "parent_counts": {"R": len(values_r), "S": len(values_s)},
            "samples": samples, "seed": seed, "skill_weights": {"R": "1/2", "S": "1/2"},
            "method": "paired-parent bootstrap within each skill; nearest-rank 95% interval",
            "powered_claim": False, "lower": _ratio(boot[lower_index]),
            "upper": _ratio(boot[upper_index])}


def _validate_cohort(cohort: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(isinstance(cohort, Mapping), "cohort must contain frozen rows")
    rows = cohort.get("rows")
    _require(isinstance(rows, list) and bool(rows), "cohort.rows must be a nonempty list")
    keys, hashes, renderings = set(), set(), set()
    parent_metadata: dict[str, tuple[str, str]] = {}
    validated = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"cohort row {index} must be an object")
        for name in ("key", "parent_id", "source", "role", "stratum", "rendering", "prompt_hash"):
            _require(isinstance(row.get(name), str) and bool(row[name]),
                     f"cohort row {index} requires nonempty {name}")
        _require(row["role"] in _ROLES, f"invalid role for {row['key']}")
        _require(bool(_SHA256.fullmatch(row["prompt_hash"])),
                 f"invalid SHA256 prompt_hash for {row['key']}")
        if "condition" in row:
            _require(type(row["condition"]) is int and row["condition"] >= 0,
                     f"condition for {row['key']} must be a nonnegative integer")
        _require(row["key"] not in keys, f"duplicate row key {row['key']}")
        _require(row["prompt_hash"] not in hashes, f"duplicate prompt_hash for {row['key']}")
        rendering_key = (row["parent_id"], row.get("condition"), row["rendering"])
        _require(rendering_key not in renderings,
                 f"duplicate parent/condition/rendering for {row['key']}")
        # A stratum may distinguish an UNKNOWN or misleading-cue variant of the
        # same parent. Role/source identity remains bound across renderings.
        metadata = (row["source"], row["role"])
        _require(row["parent_id"] not in parent_metadata or
                 parent_metadata[row["parent_id"]] == metadata,
                 f"parent identity drift for {row['parent_id']}")
        keys.add(row["key"])
        hashes.add(row["prompt_hash"])
        renderings.add(rendering_key)
        parent_metadata[row["parent_id"]] = metadata
        validated.append(dict(row))
    _require(all(any(row["role"] == role for row in validated) for role in ("R", "S", "RS")),
             "frozen cohort requires R, S, and RS coverage")
    return validated


def _validate_outcomes(outcomes: Mapping[str, Any], rows: list[dict[str, Any]], arm: str) -> None:
    _require(isinstance(outcomes, Mapping), f"{arm} rows must be a mapping")
    expected = {row["key"] for row in rows}
    _require(set(outcomes) == expected,
             f"{arm} coverage mismatch: missing={sorted(expected - set(outcomes))}, "
             f"extra={sorted(set(outcomes) - expected)}")
    for row in rows:
        outcome = outcomes[row["key"]]
        _require(isinstance(outcome, Mapping), f"{arm}/{row['key']} must be an object")
        for field in _BOOL_FIELDS:
            _require(type(outcome.get(field)) is bool,
                     f"{arm}/{row['key']} requires boolean {field}")
        for field in ("key", "parent_id", "source", "role", "stratum", "rendering", "prompt_hash"):
            if field in outcome:
                _require(outcome[field] == row[field],
                         f"{arm}/{row['key']} identity drift: {field}")
        if "condition" in outcome:
            _require("condition" in row and type(outcome["condition"]) is int and
                     outcome["condition"] == row["condition"],
                     f"{arm}/{row['key']} identity drift: condition")
        if "checkpoint_step" in outcome:
            expected_step = 0 if arm == "M0" else 27
            _require(type(outcome["checkpoint_step"]) is int and outcome["checkpoint_step"] == expected_step,
                     f"{arm}/{row['key']} is not the " +
                     ("unchanged step-0 source" if arm == "M0" else "final step-27 endpoint"))


def _completion_errors(candidate: Mapping[str, Any] | None) -> list[str]:
    if candidate is None:
        return ["candidate missing"]
    receipt = candidate.get("completion")
    if not isinstance(receipt, Mapping):
        return ["completion receipt missing"]
    expected = {"status": "COMPLETE", "completed_updates": 27,
                "consumed_sequences": 540, "final_step": 27, "failed": False}
    errors = []
    for field, value in expected.items():
        if type(receipt.get(field)) is not type(value) or receipt.get(field) != value:
            errors.append(f"completion.{field} must equal {value!r}")
    if "evaluated_step" in receipt and (type(receipt["evaluated_step"]) is not int or
                                         receipt["evaluated_step"] != 27):
        errors.append("completion.evaluated_step must equal 27")
    return errors


def _skill_metrics(outcomes: Mapping[str, Any], rows: list[dict[str, Any]],
                   baseline: Mapping[str, Any] | None, samples: int, seed: int) -> dict[str, Any]:
    result = {}
    role_values, role_gains = {}, {}
    for role_index, role in enumerate(("R", "S", "RS")):
        groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            if row["role"] == role:
                groups[row["parent_id"]].append(row["key"])
        parent_values = []
        parent_gains = []
        parents = []
        for parent_id, keys in sorted(groups.items()):
            correct = sum(outcomes[key]["correct"] for key in keys)
            value = Fraction(correct, len(keys))
            parent_values.append(value)
            parent = {"parent_id": parent_id, "row_count": len(keys), "correct": correct,
                      "accuracy": _ratio(value)}
            if baseline is not None:
                previous = Fraction(sum(baseline[key]["correct"] for key in keys), len(keys))
                gain = value - previous
                parent_gains.append(gain)
                parent["gain"] = _ratio(gain)
            parents.append(parent)
        metric = {"row_count": sum(len(keys) for keys in groups.values()),
                  "correct": sum(outcomes[key]["correct"] for keys in groups.values() for key in keys),
                  "parent_count": len(groups), "accuracy": _ratio(_mean(parent_values)),
                  "parents": parents,
                  "accuracy_interval": _bootstrap(parent_values, samples, seed + role_index)}
        if baseline is not None:
            metric["gain"] = _ratio(_mean(parent_gains))
            metric["gain_interval"] = _bootstrap(parent_gains, samples, seed + role_index)
        result[role] = metric
        role_values[role] = parent_values
        role_gains[role] = parent_gains
    result["macro_RS_atomic"] = _ratio((Fraction(result["R"]["accuracy"]["exact"]) +
                                         Fraction(result["S"]["accuracy"]["exact"])) / 2)
    result["macro_RS_atomic_interval"] = _macro_bootstrap(role_values["R"], role_values["S"],
                                                           samples, seed + 3)
    if baseline is not None:
        result["macro_RS_atomic_gain"] = _ratio((Fraction(result["R"]["gain"]["exact"]) +
                                                  Fraction(result["S"]["gain"]["exact"])) / 2)
        result["macro_RS_atomic_gain_interval"] = _macro_bootstrap(role_gains["R"], role_gains["S"],
                                                                   samples, seed + 3)
    return result


def _behavior(outcomes: Mapping[str, Any], rows: list[dict[str, Any]],
              baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Keep semantic, formatting, abstention and capped-output counts separate."""
    def summarize(keys: list[str]) -> dict[str, Any]:
        summary = {"rows": len(keys),
                   "semantic_correct": sum(outcomes[key]["correct"] for key in keys),
                   "format_correct": sum(outcomes[key]["format_correct"] for key in keys),
                   "abstained": sum(outcomes[key]["abstained"] for key in keys),
                   "capped": sum(outcomes[key]["capped"] for key in keys)}
        if baseline is not None:
            summary["format_new_failures"] = sum(baseline[key]["format_correct"] and
                                                  not outcomes[key]["format_correct"] for key in keys)
            summary["abstention_transitions"] = {
                "started": sum(not baseline[key]["abstained"] and outcomes[key]["abstained"] for key in keys),
                "stopped": sum(baseline[key]["abstained"] and not outcomes[key]["abstained"] for key in keys)}
        return summary
    strata: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row["key"])
    return {"overall": summarize([row["key"] for row in rows]),
            "by_stratum": {name: summarize(keys) for name, keys in sorted(strata.items())}}


def _retention(outcomes: Mapping[str, Any], baseline: Mapping[str, Any],
               rows: list[dict[str, Any]], sources: tuple[str, ...]) -> dict[str, Any]:
    enrolled_sources = {row["source"] for row in rows if row["role"] == "retention"}
    result = {}
    for source in sorted(enrolled_sources | set(sources) | {"long_context"}):
        keys = [row["key"] for row in rows if row["role"] == "retention" and row["source"] == source]
        supported = [key for key in keys if baseline[key]["correct"]]
        damaged = [key for key in supported if not outcomes[key]["correct"]]
        rescued = [key for key in keys if not baseline[key]["correct"] and outcomes[key]["correct"]]
        result[source] = {"row_count": len(keys), "source_correct_support": len(supported),
                          "retained": len(supported) - len(damaged), "damaged": len(damaged),
                          "rescued": len(rescued), "damaged_keys": sorted(damaged),
                          "rescued_keys": sorted(rescued),
                          "status": "UNDEFINED" if not supported else ("FAIL" if damaged else "PASS")}
    return result


def _follow_up(metrics: Mapping[str, Any], retention: Mapping[str, Any], eligible: bool,
               poor_r: bool) -> dict[str, Any]:
    acquisition = Fraction(metrics["macro_RS_atomic_gain"]["exact"]) > 0
    damage = any(source["damaged"] > 0 for source in retention.values())
    familiar_only = (Fraction(metrics["macro_RS_atomic_gain"]["exact"]) >= GAIN_THRESHOLD and
                     Fraction(metrics["RS"]["gain"]["exact"]) < GAIN_THRESHOLD)
    if damage and acquisition:
        intervention = "PRESERVATION_REPLAY_SUCCESSOR"
    elif poor_r:
        intervention = "VERIFIED_INTERMEDIATE_STATE_SUPERVISION_COMPARISON"
    elif familiar_only:
        intervention = "CONDITION_COVERAGE_AND_FRESH_TRANSFER_SUCCESSOR"
    elif eligible:
        intervention = "REPLICATE_ON_TWO_INDEPENDENT_TRAINING_DRAWS"
    else:
        intervention = "DIAGNOSE_BEFORE_REGISTERING_A_SUCCESSOR"
    return {"intervention": intervention,
            "poor_R_diagnosis_supplied": poor_r,
            "optional_RDD": "ONLY_AFTER_TEACHER_QUALIFICATION" if eligible else "NOT_ADMITTED",
            "separate_prospective_registration_required": True,
            "auto_launch": False, "fresh_confirmation_required_after_replication": eligible}


def evaluate_quality(baseline: Mapping[str, Any], candidates: Mapping[str, Any],
                     cohort: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a frozen, fully paired OFFLINE fixture; no execution authority.

    Malformed identifiers, outcome schemas or incomplete row coverage raise
    ValueError. Missing/incomplete/failed run receipts block selection even if
    their available outcome maps score well. Both M1 and M2 must be complete.
    Returned rational values are JSON-safe, exact numerator/denominator pairs.
    """
    _require(isinstance(policy, Mapping), "policy must be a mapping")
    supplied_sources = policy.get("required_retention_sources", list(REQUIRED_RETENTION_SOURCES))
    _require(isinstance(supplied_sources, (list, tuple)) and
             tuple(supplied_sources) == REQUIRED_RETENTION_SOURCES,
             "required retention sources must preserve the frozen four-source policy and order")
    sources = REQUIRED_RETENTION_SOURCES
    qualified = policy.get("long_context_qualified", False)
    _require(type(qualified) is bool, "long_context_qualified must be boolean")
    poor_r = policy.get("diagnosed_poor_reference_resolution", False)
    _require(type(poor_r) is bool, "diagnosed_poor_reference_resolution must be boolean")
    if "policy_id" in policy:
        _require(policy["policy_id"] == POLICY_ID, "policy_id must identify the new supplemental policy")
    samples = policy.get("bootstrap_samples", 1000)
    seed = policy.get("bootstrap_seed", 101)
    _require(type(samples) is int and 20 <= samples <= 10000, "bootstrap_samples must be 20..10000")
    _require(type(seed) is int, "bootstrap_seed must be integer")
    for threshold_name in ("macro_gain_threshold", "transfer_gain_threshold", "m2_advantage_threshold",
                           "minimum_macro_atomic_gain", "minimum_rs_transfer_gain", "minimum_m2_advantage"):
        if threshold_name in policy:
            try:
                threshold = Fraction(str(policy[threshold_name]))
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"{threshold_name} must remain exactly 3/100") from None
            _require(threshold == GAIN_THRESHOLD,
                     f"{threshold_name} must remain exactly 3/100")
    _require(isinstance(candidates, Mapping), "candidates must be a mapping")
    _require(not (set(candidates) - set(ARMS)), "only prespecified M1/M2 arms are permitted")
    rows = _validate_cohort(cohort)
    _validate_frozen_policy(policy, rows)
    _validate_outcomes(baseline, rows, "M0")
    result: dict[str, Any] = {
        "evidence_class": "OFFLINE_DECISION_FIXTURE",
        "evidence_kind": "mechanical_fixture",
        "decision_is_engineering_screen": True, "policy_id": POLICY_ID,
        "historical_registration_modified": False,
        "warning": "MOCKED FIXTURE EVIDENCE: no native run, quality claim, training or promotion authority",
        "native_execution_permitted": False, "quality_claim_permitted": False,
        "threshold": _ratio(GAIN_THRESHOLD), "weights": {"R": "1/2", "S": "1/2"},
        "endpoint": "FINAL_STEP_27_ONLY", "uncertainty_status": "EXPLORATORY_ENGINEERING_SCREEN_NOT_POWERED",
        "baseline": {"skills": _skill_metrics(baseline, rows, None, samples, seed),
                     "behavior": _behavior(baseline, rows)},
        "candidates": {}, "selected_arm": None,
        "broad_claim_status": "BLOCKED", "promotion_permitted": False}
    all_complete = True
    for arm in ARMS:
        candidate = candidates.get(arm)
        if candidate is not None:
            _require(isinstance(candidate, Mapping), f"{arm} must be an object")
        errors = _completion_errors(candidate)
        all_complete = all_complete and not errors
        if candidate is None:
            result["candidates"][arm] = {"status": "BLOCKED", "completion_errors": errors, "eligible": False}
            continue
        outcomes = candidate.get("rows")
        _validate_outcomes(outcomes, rows, arm)
        metrics = _skill_metrics(outcomes, rows, baseline, samples, seed)
        retention = _retention(outcomes, baseline, rows, sources)
        atomic_gain = Fraction(metrics["macro_RS_atomic_gain"]["exact"])
        checks = {
            "macro_R_S_gain_at_least_3pp": atomic_gain >= GAIN_THRESHOLD,
            "R_does_not_decline": Fraction(metrics["R"]["gain"]["exact"]) >= 0,
            "S_does_not_decline": Fraction(metrics["S"]["gain"]["exact"]) >= 0,
            "RS_transfer_gain_at_least_3pp": Fraction(metrics["RS"]["gain"]["exact"]) >= GAIN_THRESHOLD,
            "required_retention_source_correct_coverage": all(retention[source]["source_correct_support"] > 0 for source in sources),
            "every_enrolled_retention_source_correct_coverage": all(source["source_correct_support"] > 0
                for source in retention.values() if source["row_count"] > 0),
            "zero_new_retention_failures": all(source["damaged"] == 0 for source in retention.values()),
            "complete_final_endpoint": not errors}
        eligible = all(checks.values())
        long_context = retention["long_context"]
        broad_ready = qualified and long_context["status"] == "PASS" and eligible
        result["candidates"][arm] = {
            "status": "SCOPED_ELIGIBLE" if eligible else "BLOCKED_OR_FAILED_SCREEN",
            "completion_errors": errors, "eligible": eligible, "checks": checks,
            "skills": metrics, "retention": retention, "behavior": _behavior(outcomes, rows, baseline),
            "long_context": {"qualified": qualified, "guard_status": long_context["status"],
                             "scoped_guard_conditions_met": broad_ready,
                             "broad_claim_status": "BLOCKED_UNVERIFIED_QUALIFICATION" if broad_ready else "BLOCKED"},
            "follow_up": _follow_up(metrics, retention, eligible, poor_r)}
    result["all_prespecified_arms_complete"] = all_complete
    result["M2_advantage_claim"] = False
    result["M2_engineering_advantage_signal"] = False
    result["M2_advantage_signal_scope"] = "MECHANICAL_ENGINEERING_FIXTURE_ONLY_NO_MODEL_QUALITY_CLAIM"
    result["M2_advantage_claim_field_status"] = "LEGACY_ALIAS_FOR_ENGINEERING_SIGNAL_ONLY"
    if all_complete:
        m1 = result["candidates"]["M1"]
        m2 = result["candidates"]["M2"]
        difference = (Fraction(m2["skills"]["macro_RS_atomic"]["exact"]) -
                      Fraction(m1["skills"]["macro_RS_atomic"]["exact"]))
        result["M2_minus_M1_macro_R_S"] = _ratio(difference)
        result["M2_advantage_claim"] = m2["eligible"] and difference >= GAIN_THRESHOLD
        result["M2_engineering_advantage_signal"] = result["M2_advantage_claim"]
        if m1["eligible"] and m2["eligible"]:
            result["selected_arm"] = "M2" if result["M2_advantage_claim"] else "M1"
        elif m1["eligible"]:
            result["selected_arm"] = "M1"
        elif m2["eligible"]:
            result["selected_arm"] = "M2"
    chosen = result["selected_arm"]
    result["status"] = "SCOPED_ENGINEERING_SCREEN_ELIGIBLE" if chosen else "BLOCKED_OR_NO_ELIGIBLE_RECIPE"
    if chosen:
        result["broad_claim_status"] = result["candidates"][chosen]["long_context"]["broad_claim_status"]
    result["selection_rule"] = "Require complete M1 and M2 final endpoints; prefer eligible M1 unless eligible M2 exceeds M1 by >=3pp"
    return result
