"""Offline data readiness review; never admits rows or executes model code.

Packet schema ``gd001_data_admission_packet_v1`` retains role-labelled file refs
``{path, sha256, role}``. The original dose binds source_refs, tokenizer_ref,
task_ref, filtering_ref and consumption_ref and the complete occurrence schedule.
All seven production gate receipts bind that same inventory. Each retains an
executed evidence object and a different, nonauthor reviewer's review object.
The 224-trial diagnostic uses aggregate assignment/outcome/history metadata only;
this module never opens its public packets, candidates or private truth payloads.

Digests establish retained byte identity, not execution authenticity or authority.
Even READY_FOR_OWNER_ADMISSION_REVIEW requires the owner's independent review.
Fixture markers cannot produce readiness. No receipt is a training/model permit.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .common import Blocked, digest, file_hash, require, signed
from .protected_paths import reject_protected_path

SCHEMA = "gd001_data_admission_packet_v1"
GATES = ("source_use_rights", "split_parent_family_closure", "raw_rendered_overlap",
         "independent_solver", "independent_scorer", "native_rendering_eos_loss",
         "production_grader")
_HEX = re.compile(r"[0-9a-f]{64}")
_FIXTURE = {"mechanical_fixture", "fixture", "toy_fixture", "synthetic_fixture",
            "finite_mechanical_witness", "excluded_mechanical_fixture",
            "pass_finite_mechanical_query_dependence_witness"}


def _text(value, label):
    require(type(value) is str and bool(value.strip()), f"{label}: missing nonempty text")
    return value


def _hash(value, label):
    require(type(value) is str and _HEX.fullmatch(value), f"{label}: invalid SHA256")
    return value


def _same(left, right, label):
    require(digest(left) == digest(right), f"{label}: exact identity differs")


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON field {key}")
            result[key] = value
        return result
    def nonfinite(value):
        raise Blocked(f"nonfinite JSON value {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def _path(value, label):
    path = Path(_text(value, label)).expanduser()
    require(path.is_absolute() and not path.is_symlink(), f"{label}: absolute original path required")
    path = reject_protected_path(path, label, include_instruments=True)
    require(path.is_file(), f"{label}: retained payload missing")
    return path


def _ref(value, label, role):
    require(type(value) is dict and set(value) == {"path", "sha256", "role"},
            f"{label}: retained path/hash/role reference required")
    require(value["role"] == role, f"{label}: unexpected access role")
    path = _path(value["path"], label)
    require(file_hash(path) == _hash(value["sha256"], label), f"{label}: retained payload hash changed")
    return _json(path)


def _fixture(value):
    if type(value) is dict:
        marked = any(type(value.get(key)) is str and value[key].casefold() in _FIXTURE
                     for key in ("scope", "evidence_kind", "evidence_level", "classification", "status"))
        return value.get("fixture") is True or marked or any(
            _fixture(v) for v in value.values())
    return type(value) is list and any(_fixture(v) for v in value)


def _integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: invalid integer")
    return value


def _tokens(value, label, vocab):
    require(type(value) is list and bool(value) and all(type(x) is int and 0 <= x < vocab for x in value),
            f"{label}: actual token payload required")
    return value


def _inventory(packet, marked):
    assets = packet["assets"]
    require(type(assets["source_refs"]) is list and assets["source_refs"], "source_refs: original sources required")
    source_rows = {}
    for ref in assets["source_refs"]:
        source = _ref(ref, "original TRAIN source", "TRAIN_SIDE_PAYLOAD")
        marked.append(source)
        source_id = _text(source["source_id"], "source ID")
        require(source["split"] == "TRAIN", "original source: only TRAIN-side bodies are permitted")
        for row in source["rows"]:
            key = (source_id, _text(row["source_row_id"], "source row ID"))
            require(key not in source_rows and type(row["raw_record"]) is dict, "source rows: duplicate or missing raw record")
            source_rows[key] = row
    require(source_rows, "source rows: empty original inventory")
    tasks = _ref(assets["task_ref"], "complete task payload", "TRAIN_SIDE_PAYLOAD")
    tokenizer = _ref(assets["tokenizer_ref"], "exact tokenizer inventory", "TOKENIZER_METADATA")
    filters = _ref(assets["filtering_ref"], "filtering history", "TRAIN_SIDE_PAYLOAD")
    consumed = _ref(assets["consumption_ref"], "rendered consumption payload", "TRAIN_SIDE_PAYLOAD")
    dose = _ref(packet["dose_ref"], "original proposed dose", "PROSPECTIVE_DOSE")
    marked.extend((tasks, tokenizer, filters, consumed, dose))
    require(dose["frozen_before_outcomes"] is True and dose["purpose"] == "proposed_training_dose",
            "original dose: prospective frozen proposal required")
    _same(dose["assets"], assets, "original proposed dose/assets")
    require(consumed["status"] == "PROPOSED_CONSUMPTION_NOT_TRAINING", "consumption: this review prepares data, not training evidence")
    vocab = _integer(tokenizer["vocab_size"], "tokenizer vocabulary", 1)
    eos = _integer(tokenizer["eos_token_id"], "tokenizer EOS")
    require(eos < vocab, "tokenizer EOS: outside vocabulary")
    require(tokenizer["files"], "tokenizer: original tokenizer/template payloads required")
    for ref in tokenizer["files"]:
        # Preserve tokenizer bytes; parsing or executing the tokenizer is forbidden here.
        require(type(ref) is dict and set(ref) == {"path", "sha256", "role"}
                and ref["role"] == "TOKENIZER_BYTES", "tokenizer files: exact byte references required")
        path = _path(ref["path"], "tokenizer file")
        require(file_hash(path) == _hash(ref["sha256"], "tokenizer file"), "tokenizer file: hash changed")
    rows = {}
    for row in tasks["rows"]:
        rid = _text(row["row_id"], "task row ID")
        require(rid not in rows and row["split"] == "TRAIN", "task rows: duplicate or non-TRAIN row")
        key = (row["source_id"], row["source_row_id"])
        require(key in source_rows, "task row: original source ancestry missing")
        for field in ("parent_id", "family"):
            _text(row[field], field)
            _same(row[field], source_rows[key][field], f"task {rid} {field} ancestry")
        _text(row["prompt"], "task prompt")
        require(type(row["target"]) is str, "task target: explicit string required; null is not empty output")
        rows[rid] = row
    require(rows, "tasks: complete nonempty inventory required")
    dispositions, task_coverage = {}, []
    for row in filters["rows"]:
        key = (row["source_id"], row["source_row_id"])
        require(key in source_rows and key not in dispositions, "filter history: duplicate or foreign source row")
        decision = row["decision"]
        require(decision in {"accepted", "rejected", "unused"}, "filter history: explicit final disposition required")
        history = row["history"]
        require(type(history) is list and history, "filter history: actual ordered history required")
        for item in history:
            _text(item["operation"], "filter operation")
            _text(item["reason"], "filter reason")
            require(item["decision"] in {"keep", "reject"}, "filter history: invalid operation disposition")
        require(history[-1]["decision"] == ("reject" if decision == "rejected" else "keep"),
                "filter history: final disposition disagrees")
        expected = sorted(rid for rid, task in rows.items() if (task["source_id"], task["source_row_id"]) == key)
        _same(sorted(row["task_ids"]), expected if decision == "accepted" else [], "filter task projection")
        if decision != "accepted":
            require(not expected, "filter history: rejected/unused source produced a task")
        if decision == "unused":
            _text(row["unused_reason"], "unused accepted source reason")
        task_coverage.extend(row["task_ids"])
        dispositions[key] = decision
    require(set(dispositions) == set(source_rows), "filter history: source rows omitted, including rejected inputs")
    require(sorted(task_coverage) == sorted(rows), "filter history: accepted tasks omitted or repeated")
    occurrences, seen, last_step = [], set(), 0
    for event in consumed["occurrences"]:
        oid = _text(event["occurrence_id"], "consumption occurrence ID")
        require(oid not in seen and event["row_id"] in rows, "consumption: duplicate occurrence or unknown task")
        seen.add(oid)
        row = rows[event["row_id"]]
        step = _integer(event["step"], "consumption step", 1)
        require(step >= last_step, "consumption: steps not in producer order")
        last_step = step
        require(event["tokenizer_sha256"] == assets["tokenizer_ref"]["sha256"], "consumption: tokenizer differs")
        _same(event["prompt"], row["prompt"], "consumed public prompt")
        _same(event["response_text"], row["target"], "consumed exact target")
        _text(event["raw_rendered_prompt"], "raw rendered prompt")
        _text(event["rendered_prompt"], "actual rendered prompt")
        require(type(event["rendering_history"]) is list and event["rendering_history"], "consumption: rendering history missing")
        for item in event["rendering_history"]:
            _text(item["operation"], "rendering operation")
        original = _tokens(event["original_prompt_ids"], "original prompt IDs", vocab)
        prompt = _tokens(event["prompt_ids"], "consumed prompt IDs", vocab)
        response = _tokens(event["response_ids"], "response IDs including EOS", vocab)
        require(response[-1] == eos and eos not in response[:-1], "response: final EOS required without premature EOS")
        if row["target"] == "":
            require(response == [eos], "legitimate empty target: EOS alone must be supervised")
        trunc = event["truncation"]
        if trunc["policy"] == "none":
            require(trunc["max_tokens"] is None and trunc["history"] == []
                    and prompt == original and event["raw_rendered_prompt"] == event["rendered_prompt"],
                    "consumption: silent truncation")
        else:
            side, limit = trunc["policy"], _integer(trunc["max_tokens"], "truncation limit", 1)
            require(side in {"left", "right"} and limit < len(original), "truncation: invalid actual policy")
            _same(prompt, original[-limit:] if side == "left" else original[:limit], "actual truncation slice")
            _same(trunc["history"], [{"operation": "truncate", "side": side,
                  "removed_tokens": len(original) - limit, "max_tokens": limit}], "full truncation history")
        _same(event["input_ids"], prompt + response, "actual complete input tokens")
        _same(event["attention_mask"], [1] * (len(prompt) + len(response)), "actual attention mask")
        _same(event["response_mask"], [0] * len(prompt) + [1] * len(response), "response-only mask")
        _same(event["labels"], [-100] * len(prompt) + response, "prompt loss exclusion and EOS label")
        occurrences.append(event)
    require(occurrences, "consumption: complete proposed occurrence inventory required")
    require({e["row_id"] for e in occurrences} == set(rows),
            "consumption: proposed dose task absent; declare unused source disposition explicitly")
    step_set = {e["step"] for e in occurrences}
    require(min(step_set) == 1 and len(step_set) == last_step,
            "consumption: optimizer-step schedule coverage incomplete")
    _same(dose["occurrence_schedule"], [{"occurrence_id": e["occurrence_id"], "row_id": e["row_id"],
                                        "step": e["step"]} for e in occurrences], "original dose/consume repeats")
    _same(dose["counts"], {"unique_tasks": len(rows), "occurrences": len(occurrences),
          "input_tokens": sum(len(e["input_ids"]) for e in occurrences),
          "supervised_tokens": sum(len(e["response_ids"]) for e in occurrences)}, "original dose exact counts")
    return assets, rows, occurrences


def _reviewed_gate(ref, gate, binding, owner, marked):
    receipt = _ref(ref, gate + " gate", "PUBLIC_GATE_RECEIPT")
    evidence = _ref(receipt["evidence_ref"], gate + " executed evidence", "PUBLIC_GATE_EVIDENCE")
    review = _ref(receipt["review_ref"], gate + " independent review", "PUBLIC_GATE_REVIEW")
    marked.extend((receipt, evidence, review))
    for item in (receipt, evidence, review):
        require(item["gate"] == gate and item["binding_sha256"] == binding,
                gate + ": exact source/dose/task/tokenizer binding missing")
        require(item["status"] == "PASS" and item["evidence_kind"] == "real_observation",
                gate + ": real completed reviewed production evidence required")
    author, reviewer = _text(evidence["producer"], gate + " producer"), _text(review["reviewer"], gate + " reviewer")
    require(reviewer not in {owner, author}, gate + ": author cannot independently accept its own evidence")
    require(review["evidence_sha256"] == receipt["evidence_ref"]["sha256"], gate + ": reviewed evidence bytes differ")
    _text(evidence["method"], gate + " executed method")
    require(review["scope"] == "production_data_readiness", gate + ": diagnostic/source-only review is insufficient")
    return evidence


def _gate_content(gate, evidence, assets, rows, occurrences):
    if gate == "source_use_rights":
        _same(evidence["source_refs"], assets["source_refs"], "rights source coverage")
        require(evidence["permitted_use"] == "model_training" and evidence["rights_basis_refs"], "rights: explicit model-development permission required")
        for ref in evidence["rights_basis_refs"]:
            basis = _ref(ref, "source-use rights basis", "PUBLIC_RIGHTS_EVIDENCE")
            require(not _fixture(basis), "rights: fixture-marked rights basis cannot establish permitted use")
    elif gate == "split_parent_family_closure":
        _same(evidence["row_inventory"], [{k: row[k] for k in ("row_id", "parent_id", "family", "split")}
                                        for row in rows.values()], "split/family/parent closure")
        require(evidence["cross_split_parent_collisions"] == [] and evidence["unresolved_ancestry"] == [], "split closure: unresolved or colliding parents")
        _hash(evidence["exclusion_registry_sha256"], "prospective exclusion registry")
    elif gate == "raw_rendered_overlap":
        _same(evidence["row_ids"], list(rows), "raw/rendered overlap task coverage")
        require(evidence["raw_prompt_collisions"] == [] and evidence["rendered_prompt_collisions"] == []
                and evidence["parent_collisions"] == [] and evidence["scope"] == "all_reserved_roles",
                "overlap: exact raw/rendered/parent checks across reserved roles required")
        _hash(evidence["reservation_manifest_sha256"], "overlap reservation manifest")
    elif gate in {"independent_solver", "independent_scorer"}:
        observed = evidence["rows"]
        require(type(observed) is list and len(observed) == len(rows), gate + ": incomplete per-row result coverage")
        seen = set()
        for item in observed:
            rid = item["row_id"]
            require(rid in rows and rid not in seen, gate + ": duplicate/foreign per-row result")
            seen.add(rid)
            row = rows[rid]
            require(item["public_prompt_sha256"] == digest(row["prompt"]), gate + ": public task binding differs")
            _same(item["derived_target"], row["target"], gate + ": exact target verification")
            if gate == "independent_scorer":
                require(type(item["exact_target_score"]) is int and item["exact_target_score"] == 1
                        and type(item["wrong_target_score"]) is int and item["wrong_target_score"] == 0
                        and type(item["wrong_target"]) is str and item["wrong_target"] != row["target"],
                        "independent scorer: actual correct/wrong-target controls missing")
    elif gate == "native_rendering_eos_loss":
        observed = evidence["occurrences"]
        require(type(observed) is list and len(observed) == len(occurrences), "native rendering: incomplete occurrence coverage")
        for item, occurrence in zip(observed, occurrences):
            _same(item["occurrence_id"], occurrence["occurrence_id"], "native occurrence identity")
            require(item["rendered_payload_sha256"] == digest(occurrence), "native rendering: actual token/mask payload differs")
            _same(item["supervised_positions"], list(range(len(occurrence["prompt_ids"]), len(occurrence["input_ids"]))), "native response/EOS supervision")
            require(type(item["prompt_loss_tokens"]) is int and item["prompt_loss_tokens"] == 0,
                    "native loss: prompt tokens were supervised")
            loss = item["observed_response_loss"]
            require(type(loss) in (float, int) and math.isfinite(loss) and loss >= 0,
                    "native loss: actual finite response-only observation required")
        _hash(evidence["runtime_sha256"], "exact native runtime")
        _hash(evidence["tokenizer_execution_sha256"], "actual tokenizer/EOS/mask qualification")
    else:
        calibration = _ref(evidence["calibration_ref"], "separate production grader calibration", "PUBLIC_PRODUCTION_CALIBRATION")
        require(not _fixture(calibration) and calibration["status"] == "COMPLETED"
                and calibration["scope"] == "separately_reserved_production_grader"
                and calibration["registered_before_outcomes"] is True,
                "production grader: completed diagnostic alone cannot qualify production grading")
        assigned = _integer(calibration["assigned"], "production calibration assigned", 1)
        decisive = _integer(calibration["eligible_decisive"], "production calibration decisive", 1)
        require(decisive <= assigned, "production calibration: denominator mismatch")
        for key in ("error_upper", "registered_max_error_upper", "registered_min_coverage"):
            value = calibration[key]
            require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
                    "production calibration: finite preregistered bounds required")
        require(calibration["error_upper"] <= calibration["registered_max_error_upper"]
                and decisive / assigned >= calibration["registered_min_coverage"], "production calibration: error/coverage gate failed")
        protocol = _ref(calibration["protocol_ref"], "production grader prospective protocol", "PUBLIC_PRODUCTION_PROTOCOL")
        require(not _fixture(protocol) and protocol["status"] == "PREREGISTERED",
                "production calibration: retained real prospective protocol required")
        _same(protocol["assigned"], assigned, "production calibration frozen denominator")
        _same(protocol["max_error_upper"], calibration["registered_max_error_upper"], "production calibration frozen error bound")
        _same(protocol["min_coverage"], calibration["registered_min_coverage"], "production calibration frozen coverage")


def _diagnostic(value, marked):
    contract = _ref(value["contract_ref"], "review diagnostic contract", "PUBLIC_DIAGNOSTIC_CONTRACT")
    protocol_path = _path(value["protocol_ref"]["path"], "review diagnostic protocol")
    require(value["protocol_ref"]["role"] == "PUBLIC_DIAGNOSTIC_PROTOCOL"
            and file_hash(protocol_path) == _hash(value["protocol_ref"]["sha256"], "diagnostic protocol"),
            "diagnostic protocol: retained byte identity differs")
    require(contract["parents"] == 28 and contract["renderings"] == ["A", "B"]
            and contract["candidates_per_packet"] == 2 and contract["roles"] == ["semantic", "adversarial"]
            and contract["assigned_trials"] == 224 and contract["fresh_context_per_trial"] is True
            and contract["tools_permitted"] is False and contract["no_source_or_gold_to_reviewers"] is True,
            "diagnostic: original fixed 224-trial public-only contract required")
    assignments = _ref(value["assignment_ref"], "frozen224 assignments", "PUBLIC_DIAGNOSTIC_ASSIGNMENTS")
    outcomes = _ref(value["outcome_ref"], "complete224 outcomes", "PUBLIC_DIAGNOSTIC_OUTCOMES")
    marked.extend((contract, assignments, outcomes))
    require(assignments["frozen_before_candidates"] is True and assignments["role"] == "EXCLUDED_DIAGNOSTIC",
            "diagnostic assignments: must be prospectively frozen and EXCLUDED")
    parents = assignments["parent_ids"]
    require(type(parents) is list and len(parents) == len(set(parents)) == 28, "diagnostic: exact28 parent inventory required")
    for parent in parents:
        _text(parent, "opaque diagnostic parent")
    expected = set(itertools.product(parents, ("A", "B"), ("clean", "fault"), ("semantic", "adversarial")))
    slots, contexts, combinations = {}, set(), set()
    for slot in assignments["assignments"]:
        aid, context = _text(slot["assignment_id"], "assignment ID"), _text(slot["context_id"], "fresh context ID")
        combination = tuple(slot[k] for k in ("parent_id", "rendering", "candidate", "role"))
        require(aid not in slots and context not in contexts and combination not in combinations,
                "diagnostic: duplicate assignment/context/parent-render-candidate-role")
        require(combination in expected, "diagnostic: assignment outside frozen Cartesian inventory")
        _hash(slot["public_packet_sha256"], "public packet")
        _hash(slot["candidate_sha256"], "opaque candidate")
        slots[aid], combinations = slot, combinations | {combination}
        contexts.add(context)
    require(len(slots) == 224 and combinations == expected, "diagnostic: all224 frozen assignments required")
    require(outcomes["assignment_sha256"] == value["assignment_ref"]["sha256"], "diagnostic outcome: assignment bytes changed")
    require(outcomes["status"] == "COMPLETED" and outcomes["role"] == "EXCLUDED_DIAGNOSTIC", "diagnostic: complete outcome receipt missing")
    custodian = _ref(value["custodian_ref"], "diagnostic candidate qualification", "PUBLIC_DIAGNOSTIC_CUSTODIAN")
    review = _ref(value["review_ref"], "diagnostic independently reviewed observed history", "PUBLIC_DIAGNOSTIC_REVIEW")
    marked.extend((custodian, review))
    require(custodian["status"] == review["status"] == "COMPLETED"
            and custodian["role"] == review["role"] == "EXCLUDED_DIAGNOSTIC",
            "diagnostic: actual aggregate custodian/history review receipts missing")
    require(custodian["assignment_sha256"] == review["assignment_sha256"] == value["assignment_ref"]["sha256"]
            and review["outcome_sha256"] == value["outcome_ref"]["sha256"]
            and review["custodian_sha256"] == value["custodian_ref"]["sha256"],
            "diagnostic: independent review does not bind exact assignments/outcomes/custodian bytes")
    require(_text(review["reviewer"], "diagnostic independent reviewer") not in {
            _text(outcomes["producer"], "diagnostic outcome producer"),
            _text(custodian["producer"], "diagnostic custodian producer")},
            "diagnostic: author cannot independently review observed history")
    qualified = {}
    for item in custodian["candidate_checks"]:
        aid = item["assignment_id"]
        require(aid in slots and aid not in qualified, "diagnostic: custodian candidate coverage duplicated or foreign")
        require(item["public_packet_sha256"] == slots[aid]["public_packet_sha256"]
                and item["candidate_sha256"] == slots[aid]["candidate_sha256"], "diagnostic: custodian checked different public/candidate bytes")
        require(item["qualification"] in {"QUALIFIED", "UNQUALIFIED"}, "diagnostic: explicit candidate qualification required")
        if item["qualification"] == "QUALIFIED":
            _hash(item["public_verification_sha256"], "executed public candidate verification")
            if slots[aid]["candidate"] == "fault":
                _hash(item["failure_witness_sha256"], "actual seeded-fault witness")
        qualified[aid] = item["qualification"]
    require(set(qualified) == set(slots), "diagnostic: all224 candidate slots need custodian dispositions")
    counts, seen = Counter(), set()
    for row in outcomes["outcomes"]:
        aid = row["assignment_id"]
        require(aid in slots and aid not in seen, "diagnostic: duplicate/foreign outcome slot")
        seen.add(aid)
        slot = slots[aid]
        require(row["context_id"] == slot["context_id"], "diagnostic: outcome context differs")
        disposition = row["disposition"]
        require(disposition in {"completed", "missing", "uncertain", "protocol_invalid", "exposure_invalid"}, "diagnostic: invalid outcome disposition")
        counts[disposition] += 1
        if disposition != "completed":
            continue  # Keep every assigned invalid/unknown slot in the denominator.
        require(row["history_observable"] is True and row["reviewer_is_author"] is False,
                "diagnostic: context isolation or observed access history missing")
        history = row["observed_history"]
        require([e["kind"] for e in history] == ["public_packet", "public_rubric", "commitment", "candidate", "assessment"],
                "diagnostic: source/gold/tool exposure or missing two-stage commitment")
        times = []
        for index, event in enumerate(history):
            require(type(event["ordinal"]) is int and event["ordinal"] == index, "diagnostic: access history order differs")
            _hash(event["sha256"], "observed history payload")
            stamp = datetime.fromisoformat(event["at_utc"].replace("Z", "+00:00"))
            require(stamp.tzinfo is not None, "diagnostic: timezone-bound exposure chronology required")
            times.append(stamp)
        require(times == sorted(times) and times[2] < times[3], "diagnostic: candidate delivered before immutable public commitment")
        require(history[0]["sha256"] == slot["public_packet_sha256"]
                and history[3]["sha256"] == slot["candidate_sha256"], "diagnostic: actual delivered payload hashes differ")
        require(row["candidate_qualification"] == "QUALIFIED_BY_CUSTODIAN" and qualified[aid] == "QUALIFIED",
                "diagnostic: unqualified fault/clean proposal cannot count as success")
        require(row["vote"] in {"ACCEPT", "REJECT"}, "diagnostic: decisive outcome required")
        counts["correct_assessments"] += row["vote"] == ("ACCEPT" if slot["candidate"] == "clean" else "REJECT")
    require(seen == set(slots), "diagnostic: missing slots must remain explicitly in assigned224 denominator")
    positive = counts["completed"] == counts["correct_assessments"] == 224
    return {"assigned_trials": 224, "independent_parents": 28, "reported_slots": len(seen),
            "counts": dict(counts), "positive_control_results": positive,
            "production_grader_qualification": False, "scope": "EXCLUDED_DIAGNOSTIC_ONLY"}


def make_pending_packet(*, owner: str, packet_id: str, public_metadata_refs: list,
                        diagnostic_contract_ref=None, proposed_dose_metadata_ref=None,
                        source_metadata_refs=None, tokenizer_metadata_ref=None,
                        task_metadata_ref=None) -> dict:
    """Construct a never-ready public-metadata packet without writing any file.

    Metadata refs have role PUBLIC_PENDING_METADATA, except the separately named
    diagnostic contract (PUBLIC_DIAGNOSTIC_CONTRACT). No original TRAIN/control
    payload is read by this helper or the pending assessment path. Completion,
    exposure and native/production prerequisites intentionally remain missing.
    """
    _text(owner, "data owner")
    _text(packet_id, "packet ID")
    require(type(public_metadata_refs) is list, "public pending metadata reference list required")
    return {"schema": SCHEMA, "state": "PENDING_OWNER_GATES", "evidence_kind": "pending_public_metadata",
        "owner": owner, "packet_id": packet_id, "public_metadata_refs": list(public_metadata_refs),
        "proposed_dose_metadata_ref": proposed_dose_metadata_ref,
        "source_metadata_refs": list(source_metadata_refs or []), "tokenizer_metadata_ref": tokenizer_metadata_ref,
        "task_metadata_ref": task_metadata_ref, "dose_ref": None, "assets": None,
        "diagnostic": {"contract_ref": diagnostic_contract_ref, "assignment_ref": None, "outcome_ref": None,
                       "custodian_ref": None, "review_ref": None},
        "gates": {gate: None for gate in GATES}}


def _pending(packet):
    blockers, references = [], []
    def retain(ref, label, role="PUBLIC_PENDING_METADATA", parse=False):
        if ref is None:
            blockers.append(label + ": no retained public metadata supplied")
            return None
        try:
            require(type(ref) is dict and set(ref) == {"path", "sha256", "role"}
                    and ref["role"] == role, label + ": public metadata path/hash/role required")
            path = _path(ref["path"], label)
            require(file_hash(path) == _hash(ref["sha256"], label), label + ": retained metadata hash changed")
            references.append(ref)
            return _json(path) if parse else None
        except (Blocked, KeyError, TypeError, ValueError, OSError, OverflowError) as exc:
            blockers.append(label + ": " + str(exc))
            return None
    for ref in packet.get("public_metadata_refs", []):
        retain(ref, "public current-status metadata")
    for ref in packet.get("source_metadata_refs", []):
        retain(ref, "source metadata")
    for key in ("proposed_dose_metadata_ref", "tokenizer_metadata_ref", "task_metadata_ref"):
        retain(packet.get(key), key)
    contract = retain(packet.get("diagnostic", {}).get("contract_ref"), "fixed224 diagnostic contract",
                      "PUBLIC_DIAGNOSTIC_CONTRACT", True)
    if contract is not None:
        require(contract.get("parents") == 28 and contract.get("assigned_trials") == 224
                and contract.get("renderings") == ["A", "B"]
                and contract.get("roles") == ["semantic", "adversarial"]
                and contract.get("candidates_per_packet") == 2, "pending diagnostic: exact fixed224 contract differs")
    blockers.extend([
        "physical proposed dose: complete frozen original dose and occurrence schedule not supplied",
        "physical source/task/tokenizer payloads: metadata summaries cannot substitute for exact retained inputs",
        "rendered consumption: complete actual token/EOS/response masks and empty-output supervision not supplied",
        "filtering/truncation: complete original dispositions, transformations and repeated consumption not supplied",
        "reviewer diagnostic: frozen224 assignment ledger not supplied; do not dispatch from this packet",
        "reviewer diagnostic: completed224 outcomes and observed exposure history not supplied; exposure remains UNKNOWN",
        "reviewer diagnostic: aggregate executed custodian evidence and independent history review not supplied"])
    blockers.extend(gate + ": completed independent reviewed production gate not supplied" for gate in GATES)
    return blockers, {"assigned_trials": 224, "independent_parents": 28, "reported_slots": 0,
        "completed_trials": 0, "exposure_history": "UNKNOWN", "positive_control_results": False,
        "production_grader_qualification": False, "scope": "PUBLIC_PENDING_METADATA_ONLY"}, references


def assess_data_admission(packet_path: str) -> dict:
    """Return BLOCKED or READY_FOR_OWNER_ADMISSION_REVIEW; grant nothing."""
    blockers, marked, diagnostic = [], [], None
    packet_ref, binding, rows, occurrences, assets = None, None, None, None, None
    def attempt(label, action):
        try:
            return action()
        except (Blocked, KeyError, TypeError, ValueError, OSError, OverflowError) as exc:
            blockers.append(f"{label}: {exc}")
            return None
    def read_packet():
        nonlocal packet_ref
        path = _path(packet_path, "data readiness packet")
        packet_ref = {"path": str(path), "sha256": file_hash(path)}
        data = _json(path)
        require(data["schema"] == SCHEMA, "unsupported data readiness packet schema")
        _text(data["packet_id"], "packet ID")
        _text(data["owner"], "data owner")
        return data
    packet = attempt("packet", read_packet)
    if packet is not None:
        marked.append(packet)
        if packet.get("state") == "PENDING_OWNER_GATES":
            pending = attempt("pending owner gates", lambda: _pending(packet))
            if pending is not None:
                pending_blockers, diagnostic, public_refs = pending
                blockers.extend(pending_blockers)
                binding = digest({"packet_id": packet["packet_id"], "public_metadata_refs": public_refs})
            return signed({"schema": "gd001_data_admission_assessment_v1", "status": "BLOCKED",
                "evidence_level": "PUBLIC_PENDING_METADATA_ONLY", "packet": packet_ref,
                "binding_sha256": binding, "blockers": blockers, "diagnostic": diagnostic,
                "actual_dataset_admission": False, "authorizes_training": False, "authorizes_native_run": False,
                "authorizes_confirmation": False, "permits_model_promotion": False,
                "model_quality_established": False, "native_execution_independently_verified": False,
                "limitations": ["Pending public summaries cannot substitute for retained production inputs or gates.",
                                "No TRAIN body, diagnostic packet/candidate or private truth payload is opened.",
                                "The existing data owner retains admission and diagnostic dispatch authority."]})
        inventory = attempt("retained dose/data provenance", lambda: _inventory(packet, marked))
        if inventory is not None:
            assets, rows, occurrences = inventory
            binding = digest({"packet_id": packet["packet_id"], "dose_ref": packet["dose_ref"], "assets": assets})
        diagnostic = attempt("reviewer diagnostic", lambda: _diagnostic(packet["diagnostic"], marked))
        if diagnostic is not None and not diagnostic["positive_control_results"]:
            blockers.append("reviewer diagnostic: missing/uncertain/invalid/incorrect outcomes remain in assigned224 denominator")
        for gate in GATES:
            if binding is None:
                blockers.append(gate + ": exact original dose/source/tokenizer/task binding unavailable")
                continue
            evidence = attempt(gate, lambda gate=gate: _reviewed_gate(packet["gates"][gate], gate, binding, packet["owner"], marked))
            if evidence is not None:
                attempt(gate + " coverage", lambda gate=gate, evidence=evidence: _gate_content(gate, evidence, assets, rows, occurrences))
        if any(_fixture(item) for item in marked):
            blockers.append("fixture provenance: retained mechanical fixtures cannot establish production/native/data readiness")
    return signed({"schema": "gd001_data_admission_assessment_v1",
        "status": "BLOCKED" if blockers else "READY_FOR_OWNER_ADMISSION_REVIEW",
        "evidence_level": "OFFLINE_METADATA_REVIEW_ONLY", "packet": packet_ref,
        "binding_sha256": binding, "blockers": blockers, "diagnostic": diagnostic,
        "actual_dataset_admission": False, "authorizes_training": False,
        "authorizes_native_run": False, "authorizes_confirmation": False,
        "permits_model_promotion": False, "model_quality_established": False,
        "native_execution_independently_verified": False,
        "limitations": ["Retained hashes establish byte identity, not authentic execution or rights authority.",
                        "All gate observations/reviews still require the existing owner's independent review.",
                        "The excluded224 diagnostic cannot qualify production grading or learner efficacy."]})
