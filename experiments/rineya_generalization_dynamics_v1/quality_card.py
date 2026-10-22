"""Public aggregate quality cards; pure metadata validation, never bank/model IO.

Schema gd001_quality_card_v1 binds study_id/owner, evidence_level mechanical /
native_qualification / exploratory / confirmed, intervention_type mechanical /
learned / serving, source {model_id,parameters,model_sha256,tokenizer_id,
tokenizer_sha256}, baseline/candidate {checkpoint_id,model_id,
source_model_sha256,tokenizer_sha256,content_sha256,policy_sha256,runtime_sha256},
runtime signed({identity,settings}), baseline_policy/candidate_policy
signed({identity,prompt,decoding,readout}), and changed_policy_dimensions.
Learned comparisons keep runtime/policies fixed; serving comparisons keep the
checkpoint bytes fixed and declare exactly which policy components change.

data {data_id,data_sha256,split,split_sha256,receipt_id} names metadata only.
provenance {declaration:"caller_supplied_public_receipts",custody_reviewed_by,
receipts:[{receipt_id,reference:{path,sha256},file_hash_verified_by,summary}]} retains
public receipt locators. summary is signed metadata with scope
public_aggregate_metadata, role, study_id, owner, evidence_level, native_executed,
source, baseline_checkpoint_id, candidate_checkpoint_id, runtime_sha256,
baseline, candidate (both exact full checkpoint projections), baseline_policy_sha256,
candidate_policy_sha256, and reference_sha256. role-specific
payloads below must exactly match their card projections. This module verifies
embedded digests, not external file bytes, human review, provenance or execution.
Hashes are not signatures. File paths are never opened, including confirmation.

resource_budget is signed {budget_id,study_id,owner,scope:"study_specific",limits,
observed,receipt_id}; limits/observed have wall_seconds, forward_calls, model_loads,
optimizer_updates. A resources receipt includes resource_context (budget excluding
sha256 and receipt_id); a data receipt includes data_context (data excluding
receipt_id). Budgets belong to this study and cannot inherit another study's permit.

domains has six fixed names in DOMAINS. Missing names become
{status:"UNMEASURED",endpoints:null}; measured entries have status MEASURED and
nonempty endpoints. Each endpoint has endpoint_id, construct, cohort, denominator,
source_successes,candidate_successes,gained,lost,retained,threshold (original
nonempty JSON object), retention,receipt_id. An endpoint receipt includes endpoint
(the exact endpoint except receipt_id), domain (the domain name), and data_context
(the exact data/split object except receipt_id). retention has
source_success_membership_sha256,candidate_evaluated_membership_sha256,
source_correct_support,candidate_evaluated_support,retained_successes,failures.
The candidate must evaluate the same source-success membership, including failures.
Repair gains on source-incorrect rows remain separate from source-correct losses.
No scalar mixes semantic, formatting, abstention, termination or groundedness.

native_qualification requires a native_qualification receipt, status
PASS_NATIVE_MECHANICS_ONLY. exploratory adds scoring_qualification, status
PASS_SCORING_ONLY. confirmed additionally requires confirmation mapping five IDs:
native_qualification,scoring_qualification,selection,authorization,confirmation.
All five have independent_review {reviewer,independent_of:owner,attested:true}.
selection payload: {status:LOCKED_DEV_ONLY,selected_checkpoint_id,dev_split_sha256,
locked_before_confirmation:true}. authorization: {scope:confirmation_only,
authorized_by,independent_authorization:true,confirmation_split_sha256,
selection_receipt_id}. confirmation: {status:COMPLETED,independent_confirmation:true,
selection_receipt_id,authorization_receipt_id,split_sha256,confirmation_attempt_id,
single_use:true}. Supplied claims remain caller declarations. Card validation
never certifies balanced quality, release, promotion, or execution authorization.
"""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path

from .common import Blocked, digest, require, signed, verify

SCHEMA = "gd001_quality_card_v1"
DOMAINS = ("instruction_following", "grounded_retrieval", "arithmetic_state_tracking",
           "skill_composition", "answerability_abstention", "language_long_context_preservation")
COUNTERS = {"wall_seconds", "forward_calls", "model_loads", "optimizer_updates"}
CONSTRUCTS = {"semantic_correctness", "formatting", "abstention", "termination",
              "groundedness", "joint_correctness"}


def _text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"Missing {label}")
    return value


def _hash(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"Missing/invalid exact {label} hash")
    return value


def _exact(actual, expected):
    return digest(actual) == digest(expected)


def _integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid {label}: exact integer required")
    return value


def _fields(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), f"Incomplete/unknown {label} fields")


def _signed_body(value, label):
    require(isinstance(value, dict) and "sha256" in value, f"Missing signed {label} identity")
    verify(value)
    return {k: v for k, v in value.items() if k != "sha256"}


def _checkpoint(value, source, policy, runtime, label):
    _fields(value, {"checkpoint_id", "model_id", "source_model_sha256", "tokenizer_sha256",
                    "content_sha256", "policy_sha256", "runtime_sha256"}, label)
    _text(value["checkpoint_id"], f"{label} checkpoint ID")
    require(value["model_id"] == source["model_id"]
            and value["source_model_sha256"] == source["model_sha256"]
            and value["tokenizer_sha256"] == source["tokenizer_sha256"],
            "Baseline/candidate must share the exact source model and tokenizer")
    _hash(value["content_sha256"], f"{label} checkpoint")
    require(value["policy_sha256"] == policy["sha256"]
            and value["runtime_sha256"] == runtime["sha256"], "Checkpoint policy/runtime identity differs")


def _receipts(card):
    provenance = card["provenance"]
    _fields(provenance, {"declaration", "custody_reviewed_by", "receipts"}, "caller provenance")
    require(provenance["declaration"] == "caller_supplied_public_receipts",
            "Explicit caller public-receipt provenance required")
    _text(provenance["custody_reviewed_by"], "custody reviewer")
    require(isinstance(provenance["receipts"], list) and bool(provenance["receipts"]),
            "No public aggregate receipts supplied")
    receipts = {}
    common = {"scope": "public_aggregate_metadata", "study_id": card["study_id"],
              "owner": card["owner"], "evidence_level": card["evidence_level"], "source": card["source"],
              "baseline_checkpoint_id": card["baseline"]["checkpoint_id"],
              "candidate_checkpoint_id": card["candidate"]["checkpoint_id"],
              "baseline": card["baseline"], "candidate": card["candidate"],
              "runtime_sha256": card["runtime"]["sha256"],
              "baseline_policy_sha256": card["baseline_policy"]["sha256"],
              "candidate_policy_sha256": card["candidate_policy"]["sha256"]}
    for receipt in provenance["receipts"]:
        _fields(receipt, {"receipt_id", "reference", "file_hash_verified_by", "summary"}, "receipt custody")
        identifier = _text(receipt["receipt_id"], "receipt ID")
        require(identifier not in receipts, "Duplicate receipt ID")
        reference = receipt["reference"]
        _fields(reference, {"path", "sha256"}, "public receipt reference")
        require(Path(_text(reference["path"], "public receipt path")).is_absolute(),
                "Public receipt needs an absolute retained locator")
        _hash(reference["sha256"], "public receipt file")
        _text(receipt["file_hash_verified_by"], "caller file-hash reviewer")
        summary = _signed_body(receipt["summary"], "public receipt summary")
        require(all(_exact(summary.get(k), v) for k, v in common.items()),
                "Receipt source/study/runtime/policy/evidence identity differs")
        require(summary.get("reference_sha256") == reference["sha256"], "Receipt file reference differs")
        require(type(summary.get("native_executed")) is bool, "Receipt must declare native execution provenance")
        _text(summary["role"], "public receipt role")
        if card["evidence_level"] == "mechanical":
            require(summary["native_executed"] is False, "Mechanical receipts cannot claim native execution")
        receipts[identifier] = summary
    return receipts


def _receipt(receipts, identifier, role):
    require(isinstance(identifier, str) and identifier in receipts, f"Missing {role} public receipt")
    summary = receipts[identifier]
    require(summary["role"] == role, f"Wrong {role} receipt role")
    return summary


def _endpoint(value, receipts, evidence, domain, data):
    expected = {"endpoint_id", "construct", "cohort", "denominator", "source_successes",
                "candidate_successes", "gained", "lost", "retained", "threshold", "retention", "receipt_id"}
    _fields(value, expected, "measured endpoint")
    _text(value["endpoint_id"], "endpoint ID")
    _text(value["cohort"], "endpoint cohort")
    require(value["construct"] in CONSTRUCTS, "Unknown endpoint construct; do not conflate scoring endpoints")
    n = _integer(value["denominator"], "endpoint denominator", 1)
    counts = {k: _integer(value[k], k) for k in
              ("source_successes", "candidate_successes", "gained", "lost", "retained")}
    require(counts["source_successes"] == counts["retained"] + counts["lost"]
            and counts["candidate_successes"] == counts["retained"] + counts["gained"]
            and counts["retained"] + counts["lost"] + counts["gained"] <= n,
            "Paired endpoint counts disagree or exceed their own denominator")
    require(isinstance(value["threshold"], dict) and bool(value["threshold"]),
            "Original endpoint threshold/policy required")
    retention = value["retention"]
    _fields(retention, {"source_success_membership_sha256", "candidate_evaluated_membership_sha256",
                       "source_correct_support", "candidate_evaluated_support", "retained_successes", "failures"},
            "fixed source-success retention")
    for field in ("source_success_membership_sha256", "candidate_evaluated_membership_sha256"):
        _hash(retention[field], field)
    require(retention["source_success_membership_sha256"] == retention["candidate_evaluated_membership_sha256"],
            "Candidate retention membership moved away from source-success set")
    for field in ("source_correct_support", "candidate_evaluated_support", "retained_successes", "failures"):
        _integer(retention[field], field)
    require(retention["source_correct_support"] == retention["candidate_evaluated_support"] == counts["source_successes"]
            and retention["retained_successes"] == counts["retained"] and retention["failures"] == counts["lost"],
            "Fixed retention support/failures disagree; gains cannot hide source-correct losses")
    receipt = _receipt(receipts, value["receipt_id"], "endpoint")
    require(receipt.get("domain") == domain
            and _exact(receipt.get("data_context"), {k: v for k, v in data.items() if k != "receipt_id"}),
            "Endpoint domain/data/split differs from its original public receipt")
    require(_exact(receipt.get("endpoint"), {k: v for k, v in value.items() if k != "receipt_id"}),
            "Endpoint or original threshold changed from its public receipt")
    require(receipt["native_executed"] is (evidence != "mechanical"),
            "Endpoint execution provenance cannot be promoted from mechanical to native")
    return {"endpoint_id": value["endpoint_id"], "construct": value["construct"], "cohort": value["cohort"],
            "denominator": n, "source_accuracy": counts["source_successes"] / n,
            "candidate_accuracy": counts["candidate_successes"] / n,
            "independent_source_incorrect_gains": counts["gained"], "source_correct_losses": counts["lost"],
            "source_correct_retained": counts["retained"],
            "retention_fraction": counts["retained"] / counts["source_successes"] if counts["source_successes"] else None,
            "threshold": copy.deepcopy(value["threshold"]), "threshold_reinterpreted": False,
            "evidence_scope": "aggregate_public_receipt_metadata_only"}


def _evidence(card, receipts):
    level = card["evidence_level"]
    roles = {role: [r for r in receipts.values() if r["role"] == role]
             for role in ("native_qualification", "scoring_qualification")}
    for role, status, required in (("native_qualification", "PASS_NATIVE_MECHANICS_ONLY", level != "mechanical"),
                                   ("scoring_qualification", "PASS_SCORING_ONLY", level in {"exploratory", "confirmed"})):
        if required:
            require(len(roles[role]) == 1 and roles[role][0].get("status") == status
                    and roles[role][0]["native_executed"] is True, f"Missing qualified {role} metadata")
    if level != "confirmed":
        require(card["confirmation"] is None and card["data"]["split"] != "confirmation",
                "Nonconfirmed cards cannot assert confirmation evidence")
        return False
    require(card["data"]["split"] == "confirmation", "Confirmed claims require an independent confirmation split")
    confirmation = card["confirmation"]
    required = {"native_qualification", "scoring_qualification", "selection", "authorization", "confirmation"}
    _fields(confirmation, required, "independent confirmation chain")
    require(len(set(confirmation.values())) == 5, "Confirmation roles need separate public receipts")
    resolved = {}
    for key, identifier in confirmation.items():
        role = "confirmation_authorization" if key == "authorization" else key
        receipt = _receipt(receipts, identifier, role)
        review = receipt.get("independent_review")
        _fields(review, {"reviewer", "independent_of", "attested"}, "independent confirmation review")
        require(_text(review["reviewer"], "independent reviewer") != card["owner"]
                and review["independent_of"] == card["owner"] and review["attested"] is True,
                "Confirmed claim needs explicitly independent supplied review metadata")
        resolved[key] = receipt
    selected = resolved["selection"].get("selection")
    require(isinstance(selected, dict) and selected.get("status") == "LOCKED_DEV_ONLY"
            and selected.get("selected_checkpoint_id") == card["candidate"]["checkpoint_id"]
            and selected.get("locked_before_confirmation") is True, "Missing locked DEV selection metadata")
    dev_hash = _hash(selected["dev_split_sha256"], "locked DEV split")
    require(dev_hash != card["data"]["split_sha256"], "Confirmation split cannot equal DEV split")
    authorization = resolved["authorization"].get("authorization")
    require(isinstance(authorization, dict) and authorization.get("scope") == "confirmation_only"
            and _text(authorization.get("authorized_by"), "confirmation authorizer") != card["owner"]
            and authorization.get("independent_authorization") is True
            and authorization.get("confirmation_split_sha256") == card["data"]["split_sha256"]
            and authorization.get("selection_receipt_id") == confirmation["selection"],
            "Missing separate source-bound confirmation authorization metadata")
    outcome = resolved["confirmation"].get("confirmation")
    require(isinstance(outcome, dict) and outcome.get("status") == "COMPLETED"
            and outcome.get("independent_confirmation") is True and outcome.get("single_use") is True
            and outcome.get("selection_receipt_id") == confirmation["selection"]
            and outcome.get("authorization_receipt_id") == confirmation["authorization"]
            and outcome.get("split_sha256") == card["data"]["split_sha256"]
            and resolved["confirmation"]["native_executed"] is True,
            "Missing completed independent confirmation metadata")
    _text(outcome["confirmation_attempt_id"], "single-use confirmation attempt")
    return True


def _validate(card):
    require(isinstance(card, dict), "Quality card must be an object")
    if "sha256" in card:
        verify(card)
    original_hash = digest({k: v for k, v in card.items() if k != "sha256"})
    normalized = copy.deepcopy({k: v for k, v in card.items() if k != "sha256"})
    expected = {"schema", "study_id", "owner", "source", "baseline", "candidate", "runtime",
                "baseline_policy", "candidate_policy", "changed_policy_dimensions", "data", "provenance",
                "resource_budget", "domains", "evidence_level", "intervention_type", "confirmation"}
    _fields(normalized, expected, "quality card")
    card = normalized
    require(card["schema"] == SCHEMA, "Unsupported/legacy quality card; full identities required")
    _text(card["study_id"], "study ID")
    _text(card["owner"], "study owner")
    require(card["evidence_level"] in {"mechanical", "native_qualification", "exploratory", "confirmed"},
            "Unknown quality evidence level")
    require(card["intervention_type"] in {"mechanical", "learned", "serving"}, "Unknown intervention type")
    require(card["evidence_level"] != "mechanical" or card["intervention_type"] == "mechanical",
            "Mechanical evidence cannot be relabeled native learning or serving")
    require(card["intervention_type"] != "mechanical"
            or card["evidence_level"] in {"mechanical", "native_qualification"},
            "Mechanical intervention cannot be relabeled native quality efficacy")
    source = card["source"]
    _fields(source, {"model_id", "parameters", "model_sha256", "tokenizer_id", "tokenizer_sha256"}, "exact source")
    _text(source["model_id"], "exact source model ID")
    _text(source["tokenizer_id"], "exact tokenizer ID")
    require(type(source["parameters"]) is int and 500_000_000 <= source["parameters"] <= 700_000_000,
            "Quality card requires one exact 500M-700M source")
    _hash(source["model_sha256"], "source model")
    _hash(source["tokenizer_sha256"], "matching tokenizer")
    runtime = _signed_body(card["runtime"], "runtime")
    _fields(runtime, {"identity", "settings"}, "runtime identity")
    _text(runtime["identity"], "runtime ID")
    require(isinstance(runtime["settings"], dict) and bool(runtime["settings"]), "Explicit runtime settings required")
    policies = []
    for key in ("baseline_policy", "candidate_policy"):
        policy = _signed_body(card[key], key)
        _fields(policy, {"identity", "prompt", "decoding", "readout"}, key)
        _text(policy["identity"], key)
        for field in ("prompt", "decoding", "readout"):
            require(isinstance(policy[field], dict) and bool(policy[field]), f"Explicit {field} policy required")
        policies.append(policy)
    _checkpoint(card["baseline"], source, card["baseline_policy"], card["runtime"], "baseline")
    _checkpoint(card["candidate"], source, card["candidate_policy"], card["runtime"], "candidate")
    changed = card["changed_policy_dimensions"]
    require(isinstance(changed, list) and all(isinstance(v, str) for v in changed)
            and len(set(changed)) == len(changed) and set(changed) <= {"prompt", "decoding", "readout"},
            "Declare exact changed policy dimensions")
    actual = {key for key in ("prompt", "decoding", "readout") if not _exact(policies[0][key], policies[1][key])}
    require(set(changed) == actual, "Declared policy contrast differs from actual policy components")
    if card["intervention_type"] == "learned":
        require(not actual and _exact(card["baseline_policy"], card["candidate_policy"]),
                "Learned comparison must keep prompt/decoding/readout policies fixed")
        require(card["baseline"]["checkpoint_id"] != card["candidate"]["checkpoint_id"],
                "Learned comparison needs separately identified saved checkpoints")
    if card["intervention_type"] == "serving":
        require(bool(actual) and card["baseline"]["content_sha256"] == card["candidate"]["content_sha256"],
                "Serving comparison must retain the same model bytes and change only declared policy")
    data = card["data"]
    _fields(data, {"data_id", "data_sha256", "split", "split_sha256", "receipt_id"}, "data/split metadata")
    _text(data["data_id"], "data identity")
    for field in ("data_sha256", "split_sha256"):
        _hash(data[field], field)
    require(data["split"] in {"qualification", "dev", "public_exploratory", "confirmation"}, "Unsupported split metadata")
    receipts = _receipts(card)
    require(_exact(_receipt(receipts, data["receipt_id"], "data").get("data_context"),
                   {k: v for k, v in data.items() if k != "receipt_id"}), "Data/split receipt binding differs")
    budget = _signed_body(card["resource_budget"], "study resource budget")
    _fields(budget, {"budget_id", "study_id", "owner", "scope", "limits", "observed", "receipt_id"}, "resource budget")
    _text(budget["budget_id"], "study budget ID")
    require(budget["study_id"] == card["study_id"] and budget["owner"] == card["owner"]
            and budget["scope"] == "study_specific", "Cannot inherit another study's budget or execution permit")
    for name in ("limits", "observed"):
        _fields(budget[name], COUNTERS, f"resource {name}")
        for key, value in budget[name].items():
            if key == "wall_seconds":
                require(type(value) in {int, float} and math.isfinite(value)
                        and (value > 0 if name == "limits" else value >= 0), "Invalid aggregate wall counter")
            else:
                _integer(value, f"resource {key}")
    require(all(budget["observed"][key] <= budget["limits"][key] for key in COUNTERS),
            "Observed resources exceed this study's signed budget")
    resource_receipt = _receipt(receipts, budget["receipt_id"], "resources")
    require(_exact(resource_receipt.get("resource_context"),
                   {k: v for k, v in budget.items() if k != "receipt_id"}), "Resource receipt binding differs")
    require(resource_receipt["native_executed"] is (card["evidence_level"] != "mechanical"),
            "Resource execution provenance disagrees with the card evidence level")
    if card["evidence_level"] == "mechanical":
        require(budget["observed"]["model_loads"] == budget["observed"]["optimizer_updates"] == 0,
                "Mechanical cards cannot report native loads or training updates")
    else:
        require(budget["observed"]["model_loads"] > 0 and budget["observed"]["forward_calls"] > 0,
                "Native cards require explicit nonzero load/forward provenance")
        if card["intervention_type"] == "serving":
            require(budget["observed"]["optimizer_updates"] == 0, "Serving comparison cannot hide learned updates")
    require(isinstance(card["domains"], dict) and set(card["domains"]) <= set(DOMAINS), "Unknown quality domain")
    derived, unmeasured, endpoint_ids = {}, [], set()
    for domain in DOMAINS:
        item = card["domains"].setdefault(domain, {"status": "UNMEASURED", "endpoints": None})
        _fields(item, {"status", "endpoints"}, f"domain {domain}")
        if item["status"] == "UNMEASURED":
            require(item["endpoints"] is None, "Unmeasured quality must be null, never zero")
            derived[domain] = {"status": "UNMEASURED", "endpoints": None}
            unmeasured.append(domain)
        else:
            require(item["status"] == "MEASURED" and isinstance(item["endpoints"], list) and bool(item["endpoints"]),
                    "Measured domain needs explicit endpoint denominators")
            output = []
            for endpoint in item["endpoints"]:
                result = _endpoint(endpoint, receipts, card["evidence_level"], domain, data)
                require(result["endpoint_id"] not in endpoint_ids, "Duplicate endpoint ID across quality domains")
                endpoint_ids.add(result["endpoint_id"])
                output.append(result)
            derived[domain] = {"status": "MEASURED", "endpoints": output}
    confirmed = _evidence(card, receipts)
    warnings = ["Only embedded aggregate metadata digests verified; external receipts, human review and native execution remain caller declarations.",
                "No aggregate gain certifies balanced quality, release or promotion; endpoint thresholds remain the original study's authority."]
    if unmeasured:
        warnings.append("Missing quality domains remain UNMEASURED: " + ", ".join(unmeasured))
    if card["evidence_level"] in {"mechanical", "native_qualification"}:
        warnings.append("Mechanical/native-qualification evidence cannot establish model-quality efficacy.")
    if card["intervention_type"] == "learned" and budget["observed"]["optimizer_updates"] == 0:
        warnings.append("No optimizer updates occurred in this study; historical training of the learned candidate remains separately supplied provenance, not an inherited execution permit.")
    return signed({"schema": SCHEMA, "status": "PUBLIC_AGGREGATE_CARD_METADATA_VALIDATED",
                   "input_card_sha256": original_hash, "card": signed(card), "domains": derived,
                   "unmeasured_domains": unmeasured, "confirmed_chain_metadata_complete": confirmed,
                   "evidence_scope": "aggregate_public_receipt_metadata_only", "balanced_quality_established": False,
                   "external_file_hashes_verified_by_this_module": False, "human_review_verified": False,
                   "native_execution_verified_by_this_module": False, "authorizes_execution": False,
                   "authorizes_training": False, "authorizes_confirmation": False, "permits_model_promotion": False,
                   "warnings": warnings})


def validate_quality_card(card: dict) -> dict:
    """Return a signed, type-sensitive validation/coverage receipt; perform no IO."""
    try:
        return _validate(card)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, Blocked):
            raise
        raise Blocked(f"Invalid/incomplete quality card: {exc}") from exc


def build_quality_card(*, study_id, owner, source, baseline, candidate, runtime,
                       baseline_policy, candidate_policy, data, provenance, resource_budget,
                       domains=None, evidence_level="mechanical", intervention_type="mechanical",
                       changed_policy_dimensions=None, confirmation=None) -> dict:
    """Build from supplied identities/receipts; provide no fabricated hash defaults.

    Suitable for explicitly labeled fixtures. Missing domains are normalized to
    UNMEASURED. A validated native declaration is still not execution authority.
    """
    card = {"schema": SCHEMA, "study_id": study_id, "owner": owner, "source": source,
            "baseline": baseline, "candidate": candidate, "runtime": runtime,
            "baseline_policy": baseline_policy, "candidate_policy": candidate_policy,
            "data": data, "provenance": provenance, "resource_budget": resource_budget,
            "domains": {} if domains is None else domains, "evidence_level": evidence_level,
            "intervention_type": intervention_type,
            "changed_policy_dimensions": [] if changed_policy_dimensions is None else changed_policy_dimensions,
            "confirmation": confirmation}
    return validate_quality_card(card)["card"]
