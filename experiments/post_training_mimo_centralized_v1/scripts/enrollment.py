"""DEV actual-response import contract, not a model runner or a qualification gate.

``validate_import(manifest, responses)`` checks declared frozen rows, identities,
coverage, raw outputs, tokens, termination and costs. The declarations are not
proof that a native model ran. The CLI only performs this structural import.

``enroll`` can apply a separately supplied Python scorer whose entire source file,
qualified name, result schema and panel scopes are bound by the frozen manifest.
Its diagnostic result is {correct: bool, malformed: bool}; targets/oracles may be
carried in each frozen row's optional scoring_payload. No tokenizer preparation
module is imported. All outputs remain structural and natively unqualified until
a concrete scoped native-capture validator exists. Supplied evidence bundles and
callback results are recorded only as unverified diagnostic claims; they cannot
promote a status or qualify source membership. Language always remains BLOCKED/null
until a concrete four-dimension calibrated grader adapter exists. No import grants
native execution, training, promotion, CHECK or FINAL access.

Manifest v1: status=FROZEN, role=DEV, frozen_at_utc, source_identity (model_id,
weight_sha256, tokenizer_sha256, chat_template_sha256, runtime_sha256), bank_sha256,
seeds, eos_token_ids, decoding (frozen JSON policy), scorer, rows. Each row declares row_id, parent_id, family,
stratum, variant, role, panel, prompt_token_hash. Panels are acquisition,
deterministic_retention, long_context, language. Long-context variants encode
<2024|4072>-<answerable|unanswerable>_<neutral|adversarial>. Retention abstention
parents require both answerable and missing_evidence twins. Rows are never deduped.

Response v1: role=DEV, endpoint=source|candidate, manifest_sha256, bank_sha256,
identity, decoding_sha256, started_at_utc, finished_at_utc, source_enrollment_sha256 (null for source),
records. Exactly one attempted record per frozen row and seed is required; there
is no best-of/retry selection. Records contain row_id, seed, prompt_token_hash,
identity, raw_response, generated_token_ids, prompt_tokens, completion_tokens,
eos_reached, stop_reason, failure, cost={wall_seconds,gpu_seconds,usd}. Failed,
malformed and non-EOS attempts remain observations; missing attempts are rejected.
Candidates must bind an already frozen source receipt with scored membership and
an independently supplied expected candidate_identity (CLI --candidate-identity).

Use ``python .../enrollment.py --manifest M.json --responses R.json --output O.json``.
Outputs are write-once. Rejected imports retain both original inputs as base64 in
the rejection receipt. Synthetic tests are contract fixtures, never model results.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys


VERSION = 1
STRUCTURE_STATUS = "VERIFIED_RESPONSE_STRUCTURE_NOT_NATIVE_QUALIFICATION"
PANELS = {"acquisition", "deterministic_retention", "long_context", "language"}
LONG_CONTEXT_VARIANTS = {
    f"{length}-{answer}_{distractor}"
    for length in (2024, 4072) for answer in ("answerable", "unanswerable")
    for distractor in ("neutral", "adversarial")
}
IDENTITY_KEYS = {"model_id", "weight_sha256", "tokenizer_sha256",
                 "chat_template_sha256", "runtime_sha256"}
SCORER_SCHEMA = {"correct": "strict_boolean", "malformed": "strict_boolean"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_json(value):
    if type(value) is dict:
        require(all(type(k) is str for k in value), "JSON object keys must be strings")
        for child in value.values():
            _finite_json(child)
    elif type(value) is list:
        for child in value:
            _finite_json(child)
    elif type(value) is float:
        require(math.isfinite(value), "nonfinite JSON number")
    else:
        require(value is None or type(value) in (str, int, bool), "JSON values required")


def digest(value):
    _finite_json(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _hash(value, name):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            name + " must be lowercase SHA256")


def _text(value, name):
    require(type(value) is str and bool(value.strip()) and value == value.strip(),
            name + " must be a nonblank canonical string")


def _keys(value, keys, name, optional=()):
    require(type(value) is dict and keys <= set(value) <= keys | set(optional),
            name + ": missing or unexpected fields")


def _integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + " must be an integer")


def _time(value):
    require(type(value) is str and value.endswith("Z"), "UTC timestamp ending Z required")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("invalid UTC timestamp") from exc
    require(result.utcoffset().total_seconds() == 0, "UTC required")
    return result


def _identity(value):
    _keys(value, IDENTITY_KEYS, "identity")
    _text(value["model_id"], "model_id")
    for key in IDENTITY_KEYS - {"model_id"}:
        _hash(value[key], key)


def scorer_binding(callback, scopes):
    """Bind a caller-owned scorer without importing or loading code from input."""
    require(callable(callback), "scorer callback required")
    path = inspect.getsourcefile(callback)
    require(path is not None and Path(path).is_file(), "scorer must have a source file")
    require(type(scopes) in (list, tuple, set) and bool(scopes)
            and set(scopes) <= PANELS, "invalid scorer scopes")
    return {"schema_version": VERSION, "callable": callback.__module__ + "." + callback.__qualname__,
            "code_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "result_schema": SCORER_SCHEMA.copy(), "scopes": sorted(set(scopes))}


def _scorer(value):
    _keys(value, {"schema_version", "callable", "code_sha256", "result_schema", "scopes"}, "scorer")
    require(type(value["schema_version"]) is int and value["schema_version"] == VERSION,
            "unsupported scorer schema")
    _text(value["callable"], "scorer callable")
    _hash(value["code_sha256"], "scorer code hash")
    require(value["result_schema"] == SCORER_SCHEMA, "scorer result schema drift")
    require(type(value["scopes"]) is list and bool(value["scopes"])
            and all(type(s) is str for s in value["scopes"])
            and value["scopes"] == sorted(set(value["scopes"]))
            and set(value["scopes"]) <= PANELS, "invalid scorer scope")


def validate_manifest(manifest):
    _finite_json(manifest)
    _keys(manifest, {"schema_version", "status", "role", "frozen_at_utc", "source_identity",
                     "bank_sha256", "seeds", "eos_token_ids", "decoding", "scorer", "rows"}, "manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == VERSION,
            "unsupported manifest schema")
    require(manifest["status"] == "FROZEN" and manifest["role"] == "DEV",
            "only frozen DEV admitted; CHECK and FINAL remain closed")
    _time(manifest["frozen_at_utc"])
    _identity(manifest["source_identity"])
    _hash(manifest["bank_sha256"], "bank hash")
    _scorer(manifest["scorer"])
    require(type(manifest["decoding"]) is dict and bool(manifest["decoding"]), "frozen decoding policy required")
    _integer(manifest["decoding"].get("max_new_tokens"), "frozen maximum new tokens", 1)
    for key in ("seeds", "eos_token_ids"):
        values = manifest[key]
        require(type(values) is list and bool(values), key + " must be nonempty")
        for value in values:
            _integer(value, key)
        require(len(values) == len(set(values)), "duplicate " + key)
    rows, parents, seen = manifest["rows"], {}, set()
    require(type(rows) is list and bool(rows), "nonempty frozen rows required")
    for row in rows:
        _keys(row, {"row_id", "parent_id", "family", "stratum", "variant", "role", "panel",
                    "prompt_token_hash"}, "row", {"scoring_payload"})
        for key in ("row_id", "parent_id", "family", "stratum", "variant"):
            _text(row[key], key)
        require(row["role"] == "DEV" and row["panel"] in PANELS, "closed role or unknown panel")
        _hash(row["prompt_token_hash"], "prompt token hash")
        require(row["row_id"] not in seen, "duplicate expected row_id")
        seen.add(row["row_id"])
        group = parents.setdefault(row["parent_id"], [])
        require(not group or all(row[key] == group[0][key] for key in ("family", "stratum", "panel")),
                "parent family/stratum/panel drift")
        require(row["variant"] not in {r["variant"] for r in group}, "duplicate parent variant")
        group.append(row)
    for group in parents.values():
        variants = {row["variant"] for row in group}
        if group[0]["panel"] == "long_context":
            require(variants == LONG_CONTEXT_VARIANTS, "long-context parent requires all eight variants")
        if group[0]["panel"] == "deterministic_retention":
            expected = ({"answerable", "missing_evidence"} if group[0]["family"] == "answerability_abstention"
                        else {"standard"})
            require(variants == expected, "retention parent missing twin or unexpected variant")
    return parents


def _source_receipt(source, manifest):
    require(type(source) is dict, "candidate requires an already frozen source enrollment")
    require(source.get("receipt_sha256") == digest({k: v for k, v in source.items() if k != "receipt_sha256"}),
            "source enrollment hash drift")
    require(source.get("endpoint") == "source" and source.get("manifest_sha256") == digest(manifest),
            "source enrollment manifest/endpoint drift")
    require(source.get("manifest") == manifest, "source frozen manifest drift")
    require(source.get("status") == STRUCTURE_STATUS
            and source.get("native_qualification_verified_externally") is False
            and source.get("diagnostic_scoring_only") is True,
            "source must remain an unqualified structural import")
    document = validate_import(manifest, source.get("response_document"))
    require(source.get("responses_sha256") == digest(document), "source response document hash drift")
    membership = source.get("source_membership")
    require(type(membership) is list and source.get("source_membership_sha256") == digest(membership),
            "source membership is not frozen")
    expected = {(row["row_id"], seed) for row in manifest["rows"] for seed in manifest["seeds"]}
    require(len(membership) == len(expected) and {(m["row_id"], m["seed"]) for m in membership} == expected,
            "source membership coverage drift")
    rows = {row["row_id"]: row for row in manifest["rows"]}
    require(all((type(m["correct"]) is bool and m["status"] == "SCORED")
                or (m["correct"] is None and m["status"] == "BLOCKED" and rows[m["row_id"]]["panel"] == "language")
                for m in membership), "source must be scored before candidate import")
    outcomes = source.get("outcomes")
    require(type(outcomes) is list and [{k: o[k] for k in ("row_id", "seed", "correct", "status")}
                                      for o in outcomes] == membership, "source outcomes/membership drift")
    records = {(r["row_id"], r["seed"]): r for r in document["records"]}
    for outcome in outcomes:
        record = records[(outcome["row_id"], outcome["seed"])]
        require(outcome["raw_response_sha256"] == hashlib.sha256(record["raw_response"].encode()).hexdigest()
                and outcome["eos_failure"] == (not record["eos_reached"])
                and outcome["failure"] == record["failure"], "source outcome/attempt binding drift")
        require(outcome["correct"] is not True or (outcome["semantic_correct"] is True
                and outcome["malformed"] is False and not outcome["eos_failure"] and outcome["failure"] is None),
                "source correctness contradicts failed attempt")
    _time(source["frozen_at_utc"])


def validate_import(manifest, responses, *, source_enrollment=None, candidate_identity=None):
    """Pure declaration/structure validation. No scorer, model or native permission."""
    validate_manifest(manifest)
    _finite_json(responses)
    _keys(responses, {"schema_version", "role", "endpoint", "manifest_sha256", "bank_sha256",
                      "identity", "decoding_sha256", "started_at_utc", "finished_at_utc", "source_enrollment_sha256", "records"},
          "responses")
    require(type(responses["schema_version"]) is int and responses["schema_version"] == VERSION,
            "unsupported response schema")
    require(responses["role"] == "DEV" and responses["endpoint"] in {"source", "candidate"},
            "only DEV source/candidate admitted")
    require(responses["manifest_sha256"] == digest(manifest)
            and responses["bank_sha256"] == manifest["bank_sha256"], "frozen bank/manifest hash drift")
    require(responses["decoding_sha256"] == digest(manifest["decoding"]), "frozen decoding policy drift")
    _identity(responses["identity"])
    expected_identity = manifest["source_identity"]
    if responses["endpoint"] == "source":
        require(responses["identity"] == expected_identity, "source model identity drift")
        require(responses["source_enrollment_sha256"] is None and source_enrollment is None and candidate_identity is None,
                "source cannot bind another enrollment")
    else:
        _identity(candidate_identity)
        require(responses["identity"] == candidate_identity, "candidate model identity drift")
        _source_receipt(source_enrollment, manifest)
        require(responses["source_enrollment_sha256"] == source_enrollment["receipt_sha256"],
                "candidate must bind the prior frozen source enrollment")
        require(_time(responses["started_at_utc"]) >= _time(source_enrollment["frozen_at_utc"]),
                "candidate collection predates source enrollment")
        for key in ("tokenizer_sha256", "chat_template_sha256", "runtime_sha256"):
            require(responses["identity"][key] == expected_identity[key], "candidate " + key + " drift")
    require(_time(responses["finished_at_utc"]) >= _time(responses["started_at_utc"])
            >= _time(manifest["frozen_at_utc"]), "collection predates frozen manifest or reverses time")
    rows = {row["row_id"]: row for row in manifest["rows"]}
    expected = {(row_id, seed) for row_id in rows for seed in manifest["seeds"]}
    records, seen = responses["records"], set()
    require(type(records) is list, "response records must be a list")
    for record in records:
        _keys(record, {"row_id", "seed", "prompt_token_hash", "identity", "raw_response",
                       "generated_token_ids", "prompt_tokens", "completion_tokens", "eos_reached",
                       "stop_reason", "failure", "cost"}, "response record")
        _text(record["row_id"], "response row_id")
        _integer(record["seed"], "response seed")
        key = (record["row_id"], record["seed"])
        require(key in expected and key not in seen, "unexpected or duplicate response row/seed")
        seen.add(key)
        require(record["prompt_token_hash"] == rows[key[0]]["prompt_token_hash"], "prompt token identity drift")
        require(record["identity"] == responses["identity"], "per-record model identity drift")
        require(type(record["raw_response"]) is str, "actual raw response string required")
        tokens = record["generated_token_ids"]
        require(type(tokens) is list, "generated token ids required, including failures")
        for token in tokens:
            _integer(token, "generated token id")
        _integer(record["prompt_tokens"], "prompt tokens", 1)
        if rows[key[0]]["panel"] == "long_context":
            require(record["prompt_tokens"] == int(rows[key[0]]["variant"].split("-", 1)[0]),
                    "long-context prompt length drift")
        _integer(record["completion_tokens"], "completion tokens")
        require(record["completion_tokens"] == len(tokens), "completion token accounting mismatch")
        require(len(tokens) <= manifest["decoding"]["max_new_tokens"], "generation exceeded frozen token ceiling")
        require(type(record["eos_reached"]) is bool, "strict EOS boolean required")
        require(record["stop_reason"] in {"eos", "length", "stop_sequence", "error", "cancelled"},
                "unregistered stop reason")
        eos_positions = [i for i, token in enumerate(tokens) if token in manifest["eos_token_ids"]]
        require(eos_positions == ([len(tokens)-1] if record["eos_reached"] else []),
                "EOS token evidence mismatch or tokens after EOS")
        require((record["stop_reason"] == "eos") == record["eos_reached"], "EOS/stop reason mismatch")
        require(record["failure"] is None or (type(record["failure"]) is str and bool(record["failure"].strip())),
                "failure must be null or nonblank text")
        require(record["stop_reason"] not in {"error", "cancelled"} or record["failure"] is not None,
                "failed attempt requires failure reason")
        _keys(record["cost"], {"wall_seconds", "gpu_seconds", "usd"}, "attempt cost")
        for cost in record["cost"].values():
            require(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                    "finite nonnegative attempt costs required")
    require(seen == expected, "missing response pairs; failed attempts must be recorded, not dropped")
    return deepcopy(responses)


def _qualification_claim(bundle, verifier, kind, context):
    """Record caller assertions without treating a callback as an evidence gate."""
    if bundle is None:
        require(verifier is None, "qualification verifier supplied without evidence bundle")
        return None
    _finite_json(bundle)
    _keys(bundle, {"schema_version", "kind", "status", "context_sha256", "evidence_sha256",
                   "collector_code_sha256", "qualification_receipt_sha256"}, "qualification bundle")
    require(type(bundle["schema_version"]) is int and bundle["schema_version"] == VERSION
            and bundle["kind"] == kind and bundle["status"] == "PASS", "qualification kind/status drift")
    for key in ("context_sha256", "evidence_sha256", "collector_code_sha256", "qualification_receipt_sha256"):
        _hash(bundle[key], key)
    require(bundle["context_sha256"] == digest(context), "qualification is not bound to this response/scorer scope")
    claim = {"status": "UNVERIFIED_DIAGNOSTIC_CLAIM", "kind": kind, "bundle": deepcopy(bundle),
             "callback_assertion": None, "callback_error": None,
             "native_qualification": False, "language_grader_qualification": False}
    if verifier is not None:
        require(callable(verifier), "diagnostic claim verifier must be callable")
        try:
            assertion = verifier(deepcopy(bundle), deepcopy(context))
            require(type(assertion) is bool, "diagnostic callback must return a strict boolean")
            claim["callback_assertion"] = assertion
        except Exception as exc:
            claim["callback_error"] = type(exc).__name__ + ": " + str(exc)
    return claim


def qualification_context(manifest, responses):
    return {"role": "DEV", "endpoint": responses["endpoint"], "manifest_sha256": digest(manifest),
            "bank_sha256": manifest["bank_sha256"], "identity": deepcopy(responses["identity"]),
            "responses_sha256": digest(responses), "scorer_sha256": digest(manifest["scorer"]),
            "decoding_sha256": digest(manifest["decoding"]),
            "maximum_required_context_tokens": max(r["prompt_tokens"] for r in responses["records"])
                                               + manifest["decoding"]["max_new_tokens"],
            "required_panels": sorted({row["panel"] for row in manifest["rows"]}),
            "required_contexts": [2048, 4096] if any(row["panel"] == "long_context" for row in manifest["rows"]) else [],
            "long_context_zero_update_extension_required": any(row["panel"] == "long_context" for row in manifest["rows"])}


def _abstention(row, raw_response):
    if row["panel"] == "long_context":
        return raw_response.strip() == "UNKNOWN"
    if row["panel"] == "deterministic_retention" and row["family"] == "answerability_abstention":
        try:
            answer = json.loads(raw_response, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
            return type(answer) is dict and set(answer) == {"answer"} and answer["answer"] is None
        except (ValueError, TypeError, RecursionError):
            return False
    return None


def _paired_counts(paired):
    complete = all(type(p["base_correct"]) is bool and type(p["candidate_correct"]) is bool for p in paired)
    denominator = sum(p["base_correct"] for p in paired) if complete else None
    losses = sum(p["base_correct"] and not p["candidate_correct"] for p in paired) if complete else None
    repair_denominator = len(paired) - denominator if complete else None
    repairs = sum(not p["base_correct"] and p["candidate_correct"] for p in paired) if complete else None
    return {"status": "DESCRIPTIVE_ONLY" if complete else "BLOCKED_INCOMPLETE_SCORING",
            "source_correct_denominator": denominator, "source_correct_losses": losses,
            "source_correct_loss_rate": losses / denominator if denominator else None,
            "source_incorrect_denominator": repair_denominator, "source_incorrect_repairs": repairs,
            "source_incorrect_repair_rate": repairs / repair_denominator if repair_denominator else None}


def enroll(manifest, responses, *, scorer=None, source_enrollment=None, candidate_identity=None,
           qualification_bundle=None, qualification_verifier=None,
           language_qualification_bundle=None, language_qualification_verifier=None):
    """Import all attempts; optional scoring is distinct from native qualification.

    Deterministic scores and frozen membership are diagnostic imported observations.
    Verifier callbacks are unverified caller assertions, not a native-capture or
    calibrated language-grader gate. This module implements neither concrete gate.
    """
    responses = validate_import(manifest, responses, source_enrollment=source_enrollment,
                                candidate_identity=candidate_identity)
    manifest = deepcopy(manifest)
    if scorer is not None:
        require(scorer_binding(scorer, manifest["scorer"]["scopes"]) == manifest["scorer"],
                "scorer code, callable, schema or scope drift")
    context = qualification_context(manifest, responses)
    claims = [_qualification_claim(qualification_bundle, qualification_verifier, "native_response_collection", context),
              _qualification_claim(language_qualification_bundle, language_qualification_verifier, "language_grader", context)]
    rows = {row["row_id"]: row for row in manifest["rows"]}
    outcomes = []
    for record in responses["records"]:
        row = rows[record["row_id"]]
        abstained = _abstention(row, record["raw_response"])
        answerable = (row["variant"] == "answerable" if row["panel"] == "deterministic_retention"
                      else "-answerable_" in row["variant"])
        outcome = {"row_id": row["row_id"], "seed": record["seed"], "correct": None,
                   "diagnostic_only": True,
                   "semantic_correct": None, "malformed": None,
                   "abstained": abstained, "false_abstention": abstained if answerable else None,
                   "eos_failure": not record["eos_reached"], "failure": record["failure"],
                   "status": "UNMEASURED", "raw_response_sha256": hashlib.sha256(record["raw_response"].encode()).hexdigest()}
        if row["panel"] == "language":
            outcome.update(status="BLOCKED", reason="concrete four-dimension calibrated language grader adapter is unavailable; no human ratings inferred")
        elif scorer is not None:
            require(row["panel"] in manifest["scorer"]["scopes"], "row outside frozen scorer scope")
            try:
                score = scorer(deepcopy(row), record["raw_response"])
                _keys(score, {"correct", "malformed"}, "scorer result")
                require(all(type(v) is bool for v in score.values()), "scorer must return strict booleans")
                require(not score["correct"] or not score["malformed"], "malformed answer cannot be correct")
                outcome.update(status="SCORED", semantic_correct=score["correct"], malformed=score["malformed"],
                               correct=score["correct"] and not outcome["eos_failure"] and record["failure"] is None)
            except Exception as exc:
                # Retain the attempted row and fail scoring closed; never drop a scorer failure.
                outcome.update(status="SCORER_FAILED", scorer_error=type(exc).__name__ + ": " + str(exc))
        outcomes.append(outcome)
    parents = []
    for parent_id, group in sorted(validate_manifest(manifest).items()):
        parent_observations = []
        for seed in manifest["seeds"]:
            vector = [o for o in outcomes if o["seed"] == seed and o["row_id"] in {r["row_id"] for r in group}]
            complete = all(type(o["correct"]) is bool for o in vector)
            both_twins = group[0]["panel"] == "deterministic_retention" and group[0]["family"] == "answerability_abstention"
            parent_observations.append({"seed": seed, "derivative_count": len(vector),
                                        "mean_correct": sum(o["correct"] for o in vector) / len(vector) if complete else None,
                                        "all_derivatives_correct": all(o["correct"] for o in vector) if complete else None,
                                        "both_twins_correct": all(o["correct"] for o in vector) if complete and both_twins else None,
                                        "false_abstentions": sum(o["false_abstention"] for o in vector if o["false_abstention"] is not None)
                                            if any(o["false_abstention"] is not None for o in vector) else None})
        parents.append({"parent_id": parent_id, "family": group[0]["family"], "stratum": group[0]["stratum"],
                        "parent_observations": parent_observations,
                        "panel": group[0]["panel"], "derivatives": [
                            {"row_id": r["row_id"], "variant": r["variant"], "observations": [
                                deepcopy(o) for o in outcomes if o["row_id"] == r["row_id"]]} for r in group]})
    result = {"schema_version": VERSION,
              "status": STRUCTURE_STATUS,
              "role": "DEV", "endpoint": responses["endpoint"], "manifest_sha256": digest(manifest),
              "frozen_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "manifest": manifest, "response_document": responses, "responses_sha256": digest(responses),
              "outcomes": outcomes, "parent_groups": parents, "parent_cluster_count": len(parents),
              "independence_unit": "parent_id; derivatives and seeds stay inside the parent",
              "qualification_claims": [claim for claim in claims if claim is not None],
              "native_qualification_verified_externally": False,
              "native_capture_qualified": False, "language_grader_qualified": False,
              "native_qualification_status": "BLOCKED_CONCRETE_CAPTURE_VALIDATOR_UNAVAILABLE",
              "language_qualification_status": "BLOCKED_CALIBRATED_GRADER_ADAPTER_UNAVAILABLE",
              "diagnostic_scoring_only": True,
              "source_membership_scope": "diagnostic imported DEV outcomes; unqualified native source membership",
              "source_membership": None, "source_membership_sha256": None, "paired_summary": None,
              "accounting": {"attempts": len(outcomes),
                             "failed_attempts": sum(o["failure"] is not None for o in outcomes),
                             "eos_failures": sum(o["eos_failure"] for o in outcomes),
                             "prompt_tokens": sum(r["prompt_tokens"] for r in responses["records"]),
                             "completion_tokens": sum(r["completion_tokens"] for r in responses["records"]),
                             "cost": {key: sum(r["cost"][key] for r in responses["records"])
                                      for key in ("wall_seconds", "gpu_seconds", "usd")}},
              "native_execution_permission": False, "training_permission": False,
              "automatic_promotion": False, "check_opened": False, "final_opened": False,
              "limitations": ["Imported declarations do not prove native execution or model identity.",
                              "Supplied qualification bundles and callback assertions never grant native or language qualification.",
                              "All attempted rows and failures are retained; no best-of selection is supported.",
                              "Source-correct membership is specific to the bound DEV prompts and scorer.",
                              "Derivative/seed counts are not independent sample sizes; no statistical certificate is issued."]}
    if responses["endpoint"] == "source":
        result["source_membership"] = [{k: o[k] for k in ("row_id", "seed", "correct", "status")} for o in outcomes]
        result["source_membership_sha256"] = digest(result["source_membership"])
    else:
        prior = {(m["row_id"], m["seed"]): m for m in source_enrollment["source_membership"]}
        prior_outcomes = {(o["row_id"], o["seed"]): o for o in source_enrollment["outcomes"]}
        paired = [{"row_id": o["row_id"], "seed": o["seed"],
                   "parent_id": rows[o["row_id"]]["parent_id"], "family": rows[o["row_id"]]["family"],
                   "stratum": rows[o["row_id"]]["stratum"], "variant": rows[o["row_id"]]["variant"],
                   **{endpoint + "_" + field: observation[field]
                      for endpoint, observation in (("base", prior_outcomes[(o["row_id"], o["seed"])]), ("candidate", o))
                      for field in ("malformed", "eos_failure", "false_abstention")},
                   "base_correct": prior[(o["row_id"], o["seed"])]["correct"], "candidate_correct": o["correct"]}
                  for o in outcomes]
        result["paired_summary"] = {"pairs": paired, **_paired_counts(paired),
                                    "per_family": [{"family": family, **_paired_counts([
                                        p for p in paired if rows[p["row_id"]]["family"] == family])}
                                        for family in sorted({r["family"] for r in rows.values()})],
                                    "count_unit": "dependent row-seed observations, not independent parents"}
        result["source_membership"] = deepcopy(source_enrollment["source_membership"])
        result["source_membership_sha256"] = source_enrollment["source_membership_sha256"]
        result["source_enrollment_sha256"] = source_enrollment["receipt_sha256"]
        result["paired_parent_groups"] = [{"parent_id": parent["parent_id"], "family": parent["family"],
                                            "stratum": parent["stratum"], "derivatives": [
                                                {"row_id": r["row_id"], "variant": r["variant"], "paired_observations": [
                                                    deepcopy(p) for p in paired if p["row_id"] == r["row_id"]]}
                                                for r in parent["derivatives"]]} for parent in parents]
    result["receipt_sha256"] = digest(result)
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--source-enrollment", type=Path)
    parser.add_argument("--candidate-identity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = {}
    try:
        require(not args.output.exists(), "output exists; receipts are write-once")
        for key, path in (("manifest", args.manifest), ("responses", args.responses),
                          ("source_enrollment", args.source_enrollment), ("candidate_identity", args.candidate_identity)):
            if path is not None:
                inputs[key] = path.read_bytes()
        decoded = {k: json.loads(v, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                   for k, v in inputs.items()}
        result = enroll(decoded["manifest"], decoded["responses"], source_enrollment=decoded.get("source_enrollment"),
                        candidate_identity=decoded.get("candidate_identity"))
        code = 0
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        if args.output.exists():
            print("ENROLLMENT_REJECTED: " + str(exc), file=sys.stderr)
            return 2
        result = {"schema_version": VERSION, "status": "REJECTED_RESPONSE_IMPORT", "error": str(exc),
                  "raw_inputs_base64": {k: base64.b64encode(v).decode("ascii") for k, v in inputs.items()},
                  "native_execution_permission": False, "training_permission": False}
        code = 2
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except OSError as exc:
        print("ENROLLMENT_WRITE_FAILED: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
