"""Preregistered CPU arithmetic over already scored, explicitly frozen DEV rows.

No framework, model, tokenizer, protected bank or grader is imported. These
functions do not establish that caller-supplied binary scores or qualification
receipt hashes are truthful. Admission must independently verify their evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

VERSION = "synth-dev-statistics-1"
POLICY_PATH = Path(__file__).with_name("DEV_STATISTICS_POLICY.json")
ROW_FIELDS = {"row_id", "component_id", "skill", "condition", "rendering"}
DOMAIN_FIELDS = {"domain_id", "class", "row_ids", "component_ids"}
SCORER_FIELDS = {"scorer_sha256", "labels_sha256", "qualification_receipt_sha256", "qualified"}
MAX_GUARD_ROWS = 100000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("invalid statistics JSON") from error


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _id(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is not None


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate statistics JSON key")
        value[key] = item
    return value


def _nonfinite(value):
    raise ValueError("nonfinite statistics policy constant")


def policy():
    with POLICY_PATH.open("rb") as stream:
        raw = stream.read(65537)
    _require(len(raw) <= 65536, "statistics policy size")
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    expected = {
        "training_seed": 101, "component_counts_per_skill": {"R": 32, "S": 32, "RS": 32},
        "conditions": [0, 1], "renderings": ["familiar", "heldout_ledger"], "component_row_count": 4,
        "maximum_total_guard_rows": MAX_GUARD_ROWS,
    }
    _require(all(value.get(key) == item for key, item in expected.items()), "registered DEV policy drift")
    _require(value.get("population") == "EXPOSED_SINGLE_TRAINING_SEED_DEV_ONLY", "DEV population")
    bootstrap = value.get("bootstrap", {})
    _require(bootstrap.get("seed") == 2026100309 and bootstrap.get("resamples") == 10000
             and bootstrap.get("confidence") == .95 and bootstrap.get("quantile") == "linear interpolation at p*(B-1)",
             "registered paired-bootstrap settings changed")
    advancement = value.get("advancement", {})
    _require(advancement.get("atomic_M2_minus_M1_minimum") == .05 and all(
        advancement.get(key) is True for key in ("atomic_lower_CI_strictly_positive",
        "RS_M2_minus_M1_strictly_positive", "RS_lower_CI_strictly_positive",
        "all_retention_domains_and_both_candidates_must_pass")), "registered advancement margins changed")
    retention = value.get("retention", {})
    _require(retention.get("required_domain_classes") == ["task", "language", "long_context"]
             and retention.get("family_alpha") == .05 and retention.get("maximum_upper_failure_rate") == .05
             and retention.get("zero_observed_component_failures_required") is True,
             "registered retention settings changed")
    _require(all(value.get(key) is False for key in ("native_launch_authorized", "training_authorized",
        "protected_bank_access", "confirmation_eligible", "broad_retention_certified", "transfer_confirmation_established")),
        "closed statistics scope")
    return value


def _panel_rows(rows):
    _require(type(rows) is list and 0 < len(rows) <= 384, "bounded panel metadata required")
    ids, groups = set(), defaultdict(list)
    for row in rows:
        _require(type(row) is dict and set(row) == ROW_FIELDS and _id(row["row_id"])
                 and _id(row["component_id"]) and row["skill"] in ("R", "S", "RS")
                 and type(row["condition"]) is int and row["condition"] in (0, 1)
                 and row["rendering"] in ("familiar", "heldout_ledger"), "panel row metadata schema")
        _require(row["row_id"] not in ids, "duplicate panel row ID")
        ids.add(row["row_id"])
        groups[row["component_id"]].append(row)
    for component, members in groups.items():
        _require(len(members) == 4 and len({row["skill"] for row in members}) == 1
                 and {(row["condition"], row["rendering"]) for row in members}
                 == {(condition, rendering) for condition in (0, 1) for rendering in ("familiar", "heldout_ledger")},
                 "four complete condition/render rows per joined parent component required")
    return ids, groups


def _scores(scores, expected, label):
    _require(type(scores) is dict, "strict binary score map required: " + label)
    _require(len(scores) <= len(expected), "undeclared score IDs exceed the frozen cohort: " + label)
    _require(all(_id(key) and type(value) is bool for key, value in scores.items()),
             "strict binary score map required: " + label)
    _require(not (set(scores) - expected), "undeclared score IDs: " + label)
    return sorted(expected - set(scores))


def component_accuracies(rows, scores):
    """Aggregate the four rows before resampling; variants are never iid units."""
    ids, groups = _panel_rows(rows)
    _require(not _scores(scores, ids, "component aggregation"), "incomplete component scores")
    return {component: {"skill": members[0]["skill"],
            "accuracy": sum(scores[row["row_id"]] for row in members) / 4}
            for component, members in sorted(groups.items())}


def freeze_panel_metadata(rows):
    """Bind the complete joined-parent roster before any student score exists."""
    frozen = policy()
    _, groups = _panel_rows(rows)
    counts = Counter(members[0]["skill"] for members in groups.values())
    _require(dict(counts) == frozen["component_counts_per_skill"], "registered panel component quotas")
    ordered = sorted(rows, key=lambda row: row["row_id"])
    return {"policy_sha256": _digest(frozen), "panel_metadata_sha256": _digest(ordered),
            "rows": json.loads(_canonical(ordered)), "student_scores_used": False,
            "native_launch_authorized": False}


def _domains(domains):
    _require(type(domains) is list and len(domains) <= 64, "explicit bounded guard domain list required")
    seen_domains, seen_rows, component_domain = set(), set(), {}
    canonical_domains = []
    total_rows = 0
    for domain in domains:
        _require(type(domain) is dict and set(domain) == DOMAIN_FIELDS and _id(domain["domain_id"])
                 and _id(domain["class"]) and type(domain["row_ids"]) is list
                 and type(domain["component_ids"]) is list
                 and len(domain["row_ids"]) == len(domain["component_ids"]), "explicit guard domain metadata schema")
        total_rows += len(domain["row_ids"])
        _require(total_rows <= MAX_GUARD_ROWS, "guard cohort exceeds CPU arithmetic bound")
        name = domain["domain_id"]
        _require(name not in seen_domains, "duplicate guard domain ID")
        seen_domains.add(name)
        pairs = []
        for row, component in zip(domain["row_ids"], domain["component_ids"]):
            _require(_id(row) and _id(component) and row not in seen_rows, "duplicate/invalid guard row ID")
            _require(component not in component_domain or component_domain[component] == name,
                     "joined parent component crosses guard domains")
            seen_rows.add(row)
            component_domain[component] = name
            pairs.append((row, component))
        pairs.sort()
        canonical_domains.append({"domain_id": name, "class": domain["class"],
                                  "row_ids": [pair[0] for pair in pairs],
                                  "component_ids": [pair[1] for pair in pairs]})
    _require(len(seen_rows) <= MAX_GUARD_ROWS, "guard cohort exceeds CPU arithmetic bound")
    return sorted(canonical_domains, key=lambda domain: domain["domain_id"]), seen_rows, set(component_domain)


def _scorer_bindings(bindings, domain_ids):
    _require(type(bindings) is dict and set(bindings) <= domain_ids, "undeclared scorer domain binding")
    for name, record in bindings.items():
        _require(type(record) is dict and set(record) == SCORER_FIELDS and type(record["qualified"]) is bool
                 and all(type(record[key]) is str and re.fullmatch(r"[0-9a-f]{64}", record[key]) is not None
                         for key in SCORER_FIELDS - {"qualified"}), "scorer/label/qualification hash metadata")
    return json.loads(_canonical(bindings))


def freeze_guard_enrollment(declared_domains, m0_scores, *, baseline_id, scorer_bindings=None):
    """Freeze M0 membership with no candidate scores in the API or its output.

    Save/commit this packet before obtaining M1/M2 guard outputs. The function
    cannot verify chronology or receipts; those are separate admission duties.
    Missing classes, rows, or qualification metadata never gain default domains.
    """
    frozen = policy()
    _require(_id(baseline_id), "explicit baseline binding ID required")
    domains, ids, _ = _domains(declared_domains)
    missing = _scores(m0_scores, ids, "M0 guard enrollment")
    bindings = _scorer_bindings({} if scorer_bindings is None else scorer_bindings,
                                {domain["domain_id"] for domain in domains})
    classes = {domain["class"] for domain in domains}
    issues = ["MISSING_REQUIRED_CLASS:" + name for name in frozen["retention"]["required_domain_classes"]
              if name not in classes]
    issues += ["UNKNOWN_DOMAIN_CLASS:" + domain["domain_id"] for domain in domains
               if domain["class"] not in frozen["retention"]["required_domain_classes"]]
    if missing:
        issues.append("INCOMPLETE_M0_GUARD_COHORT")
    members = []
    for domain in domains:
        eligible = defaultdict(list)
        for row, component in zip(domain["row_ids"], domain["component_ids"]):
            if m0_scores.get(row) is True:
                eligible[component].append(row)
        members.append({**domain, "eligible_components": dict(sorted(eligible.items()))})
    value = {"version": VERSION, "policy_sha256": _digest(frozen), "baseline_id": baseline_id,
             "declared_domains": domains, "domains": members, "m0_scores": dict(sorted(m0_scores.items())),
             "scorer_bindings": bindings, "candidate_outputs_used": False,
             "membership_rule": "M0 correct only, frozen before candidate outputs",
             "status": "INCONCLUSIVE" if issues else "FROZEN_M0_MEMBERSHIP",
             "issues": issues, "missing_m0_row_ids": missing,
             "declared_domain_count": len(domains), "native_launch_authorized": False,
             "training_authorized": False, "protected_bank_access": False, "confirmation_eligible": False}
    value["sha256"] = _digest(value)
    return value


def _validate_enrollment(enrollment):
    _require(type(enrollment) is dict and enrollment.get("sha256") == _digest(
             {key: value for key, value in enrollment.items() if key != "sha256"}), "guard enrollment checksum")
    rebuilt = freeze_guard_enrollment(enrollment.get("declared_domains"), enrollment.get("m0_scores"),
        baseline_id=enrollment.get("baseline_id"), scorer_bindings=enrollment.get("scorer_bindings"))
    _require(_canonical(enrollment) == _canonical(rebuilt), "guard enrollment membership/source drift")


def _binomial_cdf(count, trials, probability):
    if probability == 0:
        return 1.0
    if probability == 1:
        return 0.0 if count < trials else 1.0
    logs = [math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1)
            + k * math.log(probability) + (trials - k) * math.log1p(-probability)
            for k in range(count + 1)]
    peak = max(logs)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(value - peak) for value in logs))


def one_sided_binomial_upper(failures, eligible_components, alpha):
    """Exact one-sided binomial upper bound, not a permutation NI estimate."""
    _require(type(failures) is int and type(eligible_components) is int
             and 0 < eligible_components <= MAX_GUARD_ROWS and 0 <= failures <= eligible_components
             and type(alpha) in (int, float) and math.isfinite(alpha) and 0 < alpha < 1,
             "binomial bound inputs")
    if failures == 0:
        # Numerically stable evaluation of 1-alpha**(1/n).
        return -math.expm1(math.log(alpha) / eligible_components)
    if failures == eligible_components:
        return 1.0
    lower, upper = failures / eligible_components, 1.0
    for _ in range(80):
        middle = (lower + upper) / 2
        if middle == lower or middle == upper:
            break
        if _binomial_cdf(failures, eligible_components, middle) > alpha:
            lower = middle
        else:
            upper = middle
    return upper


def _quantile(sorted_values, probability):
    index = probability * (len(sorted_values) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def paired_cluster_bootstrap(component_scores_m1, component_scores_m2):
    """Resample paired joined components within R, S and RS skill strata."""
    frozen = policy()
    _require(type(component_scores_m1) is dict and type(component_scores_m2) is dict
             and len(component_scores_m1) <= 96 and len(component_scores_m2) <= 96
             and set(component_scores_m1) == set(component_scores_m2),
             "paired component cohort differs")
    deltas = {skill: [] for skill in ("R", "S", "RS")}
    means = {arm: {skill: [] for skill in deltas} for arm in ("M1", "M2")}
    for component in sorted(component_scores_m1):
        a, b = component_scores_m1[component], component_scores_m2[component]
        _require(type(a) is dict and type(b) is dict and set(a) == set(b) == {"skill", "accuracy"}
                 and a["skill"] == b["skill"] and a["skill"] in deltas
                 and all(type(x["accuracy"]) in (int, float) and math.isfinite(x["accuracy"])
                         and 0 <= x["accuracy"] <= 1 for x in (a, b)), "component accuracy schema")
        skill = a["skill"]
        deltas[skill].append(b["accuracy"] - a["accuracy"])
        means["M1"][skill].append(a["accuracy"])
        means["M2"][skill].append(b["accuracy"])
    _require(all(len(deltas[skill]) == frozen["component_counts_per_skill"][skill] for skill in deltas),
             "registered component quotas differ")
    rng = random.Random(frozen["bootstrap"]["seed"])
    atomic, rs = [], []
    for _ in range(frozen["bootstrap"]["resamples"]):
        draws = {skill: sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                 for skill, values in deltas.items()}
        atomic.append((draws["R"] + draws["S"]) / 2)
        rs.append(draws["RS"])
    atomic.sort()
    rs.sort()
    skill_means = {arm: {skill: sum(values) / len(values) for skill, values in strata.items()}
                   for arm, strata in means.items()}
    point_atomic = (sum(deltas["R"]) / len(deltas["R"]) + sum(deltas["S"]) / len(deltas["S"])) / 2
    point_rs = sum(deltas["RS"]) / len(deltas["RS"])
    return {"method": frozen["bootstrap"]["method"], **frozen["bootstrap"],
            "component_counts": {skill: len(values) for skill, values in deltas.items()},
            "skill_means": skill_means,
            "atomic_R_S": {"M2_minus_M1": point_atomic, "CI95": [_quantile(atomic, .025), _quantile(atomic, .975)]},
            "RS": {"M2_minus_M1": point_rs, "CI95": [_quantile(rs, .025), _quantile(rs, .975)]}}


def analyze_dev_screen(panel_rows, arm_scores, enrollment, candidate_guard_scores, *, training_seed,
                       expected_enrollment_sha256, expected_panel_metadata_sha256):
    """Apply all predeclared gates; incomplete populations cannot be shrunk."""
    frozen = policy()
    _require(type(training_seed) is int and training_seed == frozen["training_seed"], "single frozen training seed 101")
    _require(type(expected_enrollment_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_enrollment_sha256)
             is not None and type(enrollment) is dict and enrollment.get("sha256") == expected_enrollment_sha256,
             "enrollment differs from the pre-candidate frozen baseline/cohort binding")
    ids, panel_components = _panel_rows(panel_rows)
    panel_hash = _digest(sorted(panel_rows, key=lambda row: row["row_id"]))
    _require(type(expected_panel_metadata_sha256) is str
             and re.fullmatch(r"[0-9a-f]{64}", expected_panel_metadata_sha256) is not None
             and panel_hash == expected_panel_metadata_sha256,
             "panel differs from the pre-score frozen row/joined-parent cohort binding")
    _require(type(arm_scores) is dict and set(arm_scores) <= {"M0", "M1", "M2"}, "undeclared DEV arm")
    _validate_enrollment(enrollment)
    domains, guard_ids, guard_components = _domains(enrollment["declared_domains"])
    _require(not ids.intersection(guard_ids) and not set(panel_components).intersection(guard_components),
             "guard enrollment overlaps acquisition/transfer rows or joined components")
    _require(type(candidate_guard_scores) is dict and set(candidate_guard_scores) <= {"M1", "M2"},
             "undeclared candidate guard arm")
    issues = list(enrollment["issues"])
    for arm in ("M0", "M1", "M2"):
        if arm not in arm_scores:
            issues.append("MISSING_PANEL_ARM:" + arm)
        if _scores(arm_scores.get(arm, {}), ids, arm + " panel"):
            issues.append("INCOMPLETE_PANEL_COHORT:" + arm)
    missing_candidates = {}
    for arm in ("M1", "M2"):
        if arm not in candidate_guard_scores:
            issues.append("MISSING_GUARD_ARM:" + arm)
        missing_candidates[arm] = _scores(candidate_guard_scores.get(arm, {}), guard_ids, arm + " guard")
        if missing_candidates[arm]:
            issues.append("INCOMPLETE_GUARD_COHORT:" + arm)
    component_counts = Counter(members[0]["skill"] for members in panel_components.values())
    if dict(component_counts) != frozen["component_counts_per_skill"]:
        issues.append("INCOMPLETE_REGISTERED_COMPONENT_QUOTAS")
    endpoint = None
    if not any(reason.startswith(("INCOMPLETE_PANEL", "INCOMPLETE_REGISTERED")) for reason in issues):
        aggregated = {arm: component_accuracies(panel_rows, arm_scores[arm]) for arm in arm_scores}
        endpoint = paired_cluster_bootstrap(aggregated["M1"], aggregated["M2"])
        endpoint["M0_skill_means"] = {skill: sum(cell["accuracy"] for cell in aggregated["M0"].values()
            if cell["skill"] == skill) / component_counts[skill] for skill in ("R", "S", "RS")}
    alpha = .05 / (2 * len(domains)) if domains else None
    retention, bindings_complete = [], True
    for domain in enrollment["domains"]:
        name = domain["domain_id"]
        binding = enrollment["scorer_bindings"].get(name)
        qualified = binding is not None and binding["qualified"] is True
        if not qualified:
            issues.append("SCORER_QUALIFICATION_MISSING_OR_UNQUALIFIED:" + name)
            bindings_complete = False
        eligible = domain["eligible_components"]
        n = len(eligible)
        if n == 0:
            issues.append("ZERO_ELIGIBLE_COMPONENTS:" + name)
        for arm in ("M1", "M2"):
            complete = not set(domain["row_ids"]).intersection(missing_candidates[arm])
            losses, lost_rows, repaired_rows = [], [], []
            if complete and not enrollment["missing_m0_row_ids"]:
                for component, enrolled_rows in eligible.items():
                    failures = [row for row in enrolled_rows if candidate_guard_scores[arm][row] is False]
                    if failures:
                        losses.append(component)
                        lost_rows.extend(failures)
                repaired_rows = [row for row in domain["row_ids"] if enrollment["m0_scores"][row] is False
                                 and candidate_guard_scores[arm][row] is True]
            bound = one_sided_binomial_upper(len(losses), n, alpha) if complete and n and not enrollment["missing_m0_row_ids"] else None
            zero_loss_precision = one_sided_binomial_upper(0, n, alpha) if n else None
            if zero_loss_precision is not None and zero_loss_precision > .05:
                issues.append("INSUFFICIENT_RETENTION_PRECISION:" + name)
            retention.append({"domain_id": name, "class": domain["class"], "arm": arm,
                "eligible_components": n, "lost_components": sorted(losses), "lost_prior_correct_rows": sorted(lost_rows),
                "repaired_previously_incorrect_rows": sorted(repaired_rows),
                "observed_component_failures": len(losses) if complete and not enrollment["missing_m0_row_ids"] else None,
                "complete_cohort": complete, "per_comparison_alpha": alpha,
                "exact_one_sided_upper_failure_rate": bound, "best_possible_zero_loss_upper_rate": zero_loss_precision,
                "caller_declared_scorer_qualified": qualified,
                "passed": complete and qualified and bound is not None and bound <= .05 and not losses})
    atomic_pass = endpoint is not None and endpoint["atomic_R_S"]["M2_minus_M1"] >= .05 and endpoint["atomic_R_S"]["CI95"][0] > 0
    rs_pass = endpoint is not None and endpoint["RS"]["M2_minus_M1"] > 0 and endpoint["RS"]["CI95"][0] > 0
    retention_pass = bool(retention) and all(row["passed"] for row in retention)
    issues = sorted(set(issues))
    advance = not issues and atomic_pass and rs_pass and retention_pass
    return {"version": VERSION, "policy": frozen, "policy_sha256": _digest(frozen), "training_seed": training_seed,
            "panel_metadata_sha256": panel_hash,
            "enrollment_sha256": enrollment["sha256"], "score_inputs_sha256": _digest(
                {"arms": arm_scores, "candidate_guards": candidate_guard_scores}),
            "status": "INCONCLUSIVE" if issues else "ADVANCE_DEV_SCREEN" if advance else "DO_NOT_ADVANCE",
            "issues": issues, "advancement_requirements_met": advance, "atomic_pass": atomic_pass,
            "RS_pass": rs_pass, "retention_pass": retention_pass,
            "endpoints": endpoint, "retention": retention, "declared_domain_count": len(domains),
            "computational_screen_only": not bindings_complete or bool(enrollment["issues"]),
            "scorer_qualification_independently_verified_by_this_module": False,
            "single_training_seed_cannot_confirm_transfer_or_broad_retention": True,
            "expected_pre_candidate_enrollment_sha256": expected_enrollment_sha256,
            "expected_pre_score_panel_metadata_sha256": expected_panel_metadata_sha256,
            "native_launch_authorized": False, "training_authorized": False, "protected_bank_access": False,
            "confirmation_eligible": False, "broad_retention_certified": False,
            "transfer_confirmation_established": False}
