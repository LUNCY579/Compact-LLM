"""Original deterministic retention preparation; no native or training admission.

Canonical semantic parents receive roles before rendering or solving. Answerable
and missing-evidence twins share a parent. CHECK exports contain hashes only and
FINAL has no candidates. Uses the Python standard library; no model/data imports.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

VERSION = "rineya-original-retention-v1"
VERIFIER_VERSION = "rineya-retention-independent-verifier-v1"
DOMAINS = ("retrieval", "instruction_following", "answerability_abstention")
ROLES = ("TRAIN", "TRAIN_SELECTION", "DEV", "CHECK", "FINAL")
OPEN_ROLES = ("TRAIN", "DEV")
ROLE_BUCKETS = ((7000, "TRAIN"), (7800, "TRAIN_SELECTION"), (8600, "DEV"),
                (9400, "CHECK"), (10000, "FINAL"))
ROLE_DOMAIN = b"rineya-retention-semantic-parent-role-v1\0"
POSITIONS = ("early", "middle", "late", "distributed")
PRIORITIES = ("base", "task", "final")
SLOTS = ("recipient", "message", "note")
FORMATS = ("To: {recipient}\nMessage: {message}", "TO[{recipient}] MSG[{message}]",
           "Recipient={recipient}; Message={message}", "{message} (for {recipient})")
WORDS = ("amber", "birch", "cedar", "dawn", "ember", "fern", "glen", "harbor",
         "iris", "jade", "kelp", "linen", "moss", "north", "opal", "pine",
         "quartz", "river", "silver", "tulip", "umber", "violet", "willow", "zenith")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _keys(value: Any, expected: set[str]) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"expected object with keys {sorted(expected)}")


def _integer(value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"expected integer in [{low}, {high}]")
    return value


def _text(value: Any) -> str:
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}", value):
        raise ValueError("expected bounded literal ASCII text")
    return value


def canonicalize(spec: dict) -> dict:
    """Normalize record/directive order and identical duplicates.

    Query order, directive priorities, exact case and format remain meaningful.
    A complete record world is required even for missing-evidence twins; a twin
    cannot be registered as a second independent incomplete parent.
    """
    if type(spec) is not dict or spec.get("domain") not in DOMAINS:
        raise ValueError("unknown retention domain")
    domain = spec["domain"]
    if domain != "instruction_following":
        _keys(spec, {"domain", "records", "queries"})
        if type(spec["records"]) is not list or not 3 <= len(spec["records"]) <= 32:
            raise ValueError("expected 3..32 original records")
        records = {}
        for record in spec["records"]:
            _keys(record, {"entity", "field", "value"})
            entity, field, value = (_text(record[key]) for key in ("entity", "field", "value"))
            key = (entity, field)
            if key in records and records[key] != value:
                raise ValueError("contradictory record evidence")
            records[key] = value
        if type(spec["queries"]) is not list or len(spec["queries"]) != 2:
            raise ValueError("exactly two ordered queries required")
        queries = []
        for query in spec["queries"]:
            _keys(query, {"entity", "field"})
            query = {key: _text(query[key]) for key in ("entity", "field")}
            if (query["entity"], query["field"]) not in records:
                raise ValueError("complete canonical parent is missing queried evidence")
            if query in queries:
                raise ValueError("query keys must be distinct")
            queries.append(query)
        if len(records) < 3:
            raise ValueError("at least one distractor record required")
        return {"domain": domain, "records": [
            {"entity": entity, "field": field, "value": records[(entity, field)]}
            for entity, field in sorted(records)], "queries": queries}
    _keys(spec, {"domain", "directives", "template"})
    if type(spec["directives"]) is not list or not 3 <= len(spec["directives"]) <= 18:
        raise ValueError("expected 3..18 instruction directives")
    if spec["template"] not in FORMATS:
        raise ValueError("unregistered output template")
    directives = {}
    for item in spec["directives"]:
        _keys(item, {"priority", "slot", "text"})
        if item["priority"] not in PRIORITIES or item["slot"] not in SLOTS:
            raise ValueError("unknown instruction priority or slot")
        key, value = (item["priority"], item["slot"]), _text(item["text"])
        if key in directives and directives[key] != value:
            raise ValueError("contradictory same-priority directives")
        directives[key] = value
    if any(("base", slot) not in directives for slot in SLOTS):
        raise ValueError("all base slots, including irrelevant note, must be specified")
    if not any(priority != "base" and slot != "note" for priority, slot in directives):
        raise ValueError("at least one relevant priority override is required")
    return {"domain": domain, "directives": [
        {"priority": priority, "slot": slot, "text": directives[(priority, slot)]}
        for priority, slot in sorted(directives)], "template": spec["template"]}


def _semantic_bytes(spec: dict) -> bytes:
    # Retrieval and answerability share a split key if the same world is reused.
    # Domain names, seeds, candidate indices and surface positions cannot resplit it.
    # Every query/format derivative of one literal world remains in one role.
    # The full spec still preserves output order/template for solving and rendering.
    semantic = {key: value for key, value in spec.items() if key not in {"domain", "queries", "template"}}
    return json_bytes({**semantic, "kind": "text_instruction" if spec["domain"] == "instruction_following" else "record_query"})


@dataclass(frozen=True)
class AllocatedParent:
    canonical_json: str
    canonical_parent_sha256: str
    parent_id: str
    split_sha256: str
    role: str
    stratum: str
    generator_seed: int
    candidate_index: int


def allocate_parent(spec: dict, seed: int = 20260928, candidate_index: int = 0) -> AllocatedParent:
    _integer(seed, 0, 2**63 - 1)
    _integer(candidate_index, 0, 2**63 - 1)
    normalized = canonicalize(spec)
    raw = _semantic_bytes(normalized)
    identity, split = sha256(raw), sha256(ROLE_DOMAIN + raw)
    bucket = int(split[:16], 16) % 10000
    role = next(role for upper, role in ROLE_BUCKETS if bucket < upper)
    if normalized["domain"] == "instruction_following":
        stratum = "format_" + str(FORMATS.index(normalized["template"]))
    else:
        stratum = POSITIONS[int(identity[:16], 16) % 4]
    return AllocatedParent(json_bytes(normalized).decode(), identity, "retention-" + identity,
                           split, role, stratum, seed, candidate_index)


def _validated_spec(parent: AllocatedParent, require_open: bool = True) -> dict:
    spec = json.loads(parent.canonical_json)
    if allocate_parent(spec, parent.generator_seed, parent.candidate_index) != parent:
        raise ValueError("parent identity, role or metadata mismatch")
    if require_open and parent.role not in OPEN_ROLES:
        raise PermissionError("only TRAIN and DEV may render/solve; CHECK and FINAL remain closed")
    return spec


def _variant(spec: dict, variant: str) -> None:
    allowed = {"answerable", "missing_evidence"} if spec["domain"] == "answerability_abstention" else {"standard"}
    if variant not in allowed:
        raise ValueError("variant does not belong to this domain")


def _visible_records(spec: dict, variant: str) -> list[dict]:
    withheld = spec["queries"][0] if variant == "missing_evidence" else None
    return [record for record in spec["records"]
            if withheld is None or any(record[key] != withheld[key] for key in ("entity", "field"))]


def reference_answer(parent: AllocatedParent, variant: str = "standard") -> Any:
    """Direct reference execution; verifier has separate semantic evaluation."""
    spec = _validated_spec(parent)
    _variant(spec, variant)
    if spec["domain"] == "instruction_following":
        slots = {}
        for priority in PRIORITIES:
            for directive in spec["directives"]:
                if directive["priority"] == priority:
                    slots[directive["slot"]] = directive["text"]
        return spec["template"].format(**slots)
    records, result = _visible_records(spec, variant), []
    for query in spec["queries"]:
        matches = [record["value"] for record in records if all(record[key] == query[key] for key in query)]
        if len(matches) != 1:
            return None
        result.append(matches[0])
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def parse_answer(response: str) -> Any:
    if type(response) is not str or len(response) > 4096:
        raise ValueError("answer must be a bounded JSON object string")
    parsed = json.loads(response, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    _keys(parsed, {"answer"})
    return parsed["answer"]


def verify_answer(parent: AllocatedParent, response: str, variant: str = "standard") -> bool:
    """Independent strict validator, without calling solver or trusting target."""
    spec = _validated_spec(parent)
    _variant(spec, variant)
    try:
        answer = parse_answer(response)
    except (ValueError, TypeError, RecursionError):
        return False
    if spec["domain"] == "instruction_following":
        if type(answer) is not str:
            return False
        slots = {}
        rank = {"base": 0, "task": 1, "final": 2}
        for slot in ("recipient", "message"):
            options = [(rank[item["priority"]], item["text"]) for item in spec["directives"] if item["slot"] == slot]
            highest = max(level for level, text in options)
            slots[slot] = next(text for level, text in options if level == highest)
        # Explicit substitution rather than reference execution/format dispatch.
        expected = spec["template"].replace("{recipient}", slots["recipient"]).replace("{message}", slots["message"])
        return answer == expected
    evidence = {}
    withheld_key = tuple(spec["queries"][0][key] for key in ("entity", "field")) if variant == "missing_evidence" else None
    for record in spec["records"]:
        key = (record["entity"], record["field"])
        if key != withheld_key:
            evidence.setdefault(key, set()).add(record["value"])
    keys = [(query["entity"], query["field"]) for query in spec["queries"]]
    if any(len(evidence.get(key, set())) != 1 for key in keys):
        return answer is None and spec["domain"] == "answerability_abstention"
    if type(answer) is not list or len(answer) != 2 or any(type(value) is not str for value in answer):
        return False
    return all(answer[i] in evidence[key] for i, key in enumerate(keys))


def render_prompt(parent: AllocatedParent, variant: str = "standard") -> str:
    spec = _validated_spec(parent)
    _variant(spec, variant)
    if spec["domain"] == "instruction_following":
        directives = list(spec["directives"])
        rng = random.Random(int(parent.canonical_parent_sha256, 16))
        rng.shuffle(directives)
        body = ("Resolve the following labeled directives. For each slot, final priority overrides task priority, "
                "and task overrides base, regardless of listed order. Copy literal text; preserve spaces and case.\n")
        body += "\n".join(json_bytes(item).decode() for item in directives)
        body += "\nUse only recipient and message; never include the note slot."
        body += " Replace {recipient} and {message} in this exact output template: " + json.dumps(spec["template"])
        body += ". The answer is the resulting string with exact punctuation, spaces and newlines; add nothing."
    else:
        # Establish full-world positions once. A missing-evidence twin removes
        # exactly one record without relocating the remaining evidence.
        records = spec["records"]
        query_keys = {(query["entity"], query["field"]) for query in spec["queries"]}
        evidence = [record for record in records if (record["entity"], record["field"]) in query_keys]
        distractors = [record for record in records if (record["entity"], record["field"]) not in query_keys]
        if parent.stratum == "early":
            ordered = evidence + distractors
        elif parent.stratum == "middle":
            cut = len(distractors) // 2
            ordered = distractors[:cut] + evidence + distractors[cut:]
        elif parent.stratum == "late":
            ordered = distractors + evidence
        else:
            ordered = evidence[:1] + distractors + evidence[1:]
        if variant == "missing_evidence":
            withheld = spec["queries"][0]
            ordered = [record for record in ordered if any(record[key] != withheld[key] for key in ("entity", "field"))]
        body = "Use only these original records. Match both entity and field exactly; do not guess.\n"
        body += "\n".join(json_bytes(record).decode() for record in ordered)
        body += "\nQueries in required answer order: " + json_bytes(spec["queries"]).decode()
        body += "\nReturn the two exact recorded value strings as a JSON array in query order. Preserve spaces and case."
        if spec["domain"] == "answerability_abstention":
            body += " If either query lacks one uniquely supported value, the answer must be null for the whole pair."
    return body + '\nReturn one JSON object with exactly the key "answer". No rationale or extra text.'


def generate_spec(domain: str, seed: int, index: int) -> dict:
    """Sample original latent facts only. No surface, answer, or scoring occurs."""
    if domain not in DOMAINS:
        raise ValueError("unknown domain")
    _integer(seed, 0, 2**63 - 1)
    _integer(index, 0, 2**63 - 1)
    rng = random.Random(int(sha256(json_bytes([VERSION, domain, seed, index])), 16))
    if domain == "instruction_following":
        directives = [{"priority": priority, "slot": slot,
                       "text": rng.choice(WORDS).title() + " " + rng.choice(WORDS)}
                      for priority, slot in (("base", "recipient"), ("base", "message"), ("base", "note"),
                                             ("task", "recipient"), ("task", "message"),
                                             ("final", "recipient" if index % 2 else "message"))]
        return {"domain": domain, "directives": directives, "template": FORMATS[index % 4]}
    entities = rng.sample([a + " " + b for a in WORDS for b in WORDS], 6)
    fields = ("badge", "drawer") if domain == "retrieval" else ("access key", "storage bay")
    records = [{"entity": entity, "field": field,
                "value": rng.choice(WORDS).title() + " " + rng.choice(WORDS) + " " + str(rng.randrange(10, 100))}
               for entity in entities for field in fields]
    selected = rng.sample(records, 2)
    return {"domain": domain, "records": records,
            "queries": [{key: record[key] for key in ("entity", "field")} for record in selected]}


def make_row(parent: AllocatedParent, source_revision: str, variant: str = "standard") -> dict:
    spec = _validated_spec(parent)
    _variant(spec, variant)
    prompt, answer = render_prompt(parent, variant), reference_answer(parent, variant)
    target = json_bytes({"answer": answer}).decode()
    wrong = {"answer": ["UNSUPPORTED", "UNSUPPORTED"] if answer is None else None}
    if not verify_answer(parent, target, variant) or verify_answer(parent, json_bytes(wrong).decode(), variant):
        raise RuntimeError("positive reference or negative mutation control failed")
    return {
        "schema_version": 1, "row_id": parent.parent_id + "-" + variant, "parent_id": parent.parent_id,
        "source_id": "original-rineya-retention", "source_revision": source_revision,
        "domain": spec["domain"], "family": spec["domain"], "role": parent.role,
        "stratum": parent.stratum, "variant_id": variant,
        "canonical_parent_sha256": parent.canonical_parent_sha256, "split_sha256": parent.split_sha256,
        "spec": spec, "prompt": prompt, "target": target,
        "prompt_sha256": sha256(prompt.encode()), "normalized_prompt_sha256": sha256(" ".join(prompt.split()).encode()),
        "prompt_normalization": "collapse Unicode whitespace only", "target_sha256": sha256(target.encode()),
        "tokenized_prompt_sha256": None, "tokenizer_identity": None, "tokenizer_sha256": None,
        "chat_template_sha256": None, "tokenization_status": "UNMEASURED",
        "prompt_tokens": None, "target_tokens": None, "total_nonpadding_tokens": None,
        "generator_seed": parent.generator_seed, "candidate_index": parent.candidate_index,
        "verifier_version": VERIFIER_VERSION, "verifier_result": True, "mutation_control_rejected": True,
        "policy_or_teacher_identity": "DETERMINISTIC_REFERENCE_NO_MODEL", "policy_version": None,
        "behavior_logprobs": None, "environment_identity": VERSION, "environment_revision": source_revision,
        "tool_actions": [], "tool_outputs": [], "reward": 1, "attempt_number": 1,
        "termination_reason": "verified_exact_answer", "source_correct_membership": None,
        "source_correct_membership_policy": "freeze DEV base-correct subset; TRAIN is fixed and never filtered/replaced by baseline outcomes",
        "source_correct_enrollment_qualified": False, "training_role_eligible": parent.role == "TRAIN",
        "training_eligibility": False,
        "training_eligibility_reasons": ["NATIVE_BASELINE_RESPONSES_UNMEASURED", "TOKENIZER_CONTEXT_FIT_UNMEASURED",
                                         "HISTORICAL_EXCLUSION_UNMEASURED", "COMPLETE_REPLAY_NOT_QUALIFIED"],
        "cost": {"model_calls": 0, "teacher_calls": 0, "model_generation_tokens": 0, "verifier_calls": 2,
                 "training_tokens": 0, "dataset_tokens": None},
        "license_provenance": {"origin": "original task logic, synthetic facts and prompt authored for the user project",
                               "external_dataset_or_teacher": False, "redistribution_license": "UNRESOLVED_NO_NEW_LICENSE_GRANT",
                               "code_notice_policy": "retain repository source notices"},
    }


def validate_registration(registration: dict) -> dict[str, dict[str, int]]:
    if registration.get("status") != "FROZEN_FAST_LANE_RETENTION_ONLY":
        raise ValueError("registration must be FROZEN_FAST_LANE_RETENTION_ONLY")
    authorization = registration.get("authorization", {})
    expected_auth = {"dataset_artifact_generation": True, "native_execution": False, "training": False,
                     "check_rendering": False, "final_rendering": False}
    if any(authorization.get(key) is not value for key, value in expected_auth.items()):
        raise ValueError("registration must authorize only FAST retention preparation, with CHECK/FINAL closed")
    if registration.get("domains") != list(DOMAINS):
        raise ValueError("domain catalog mismatch")
    if registration.get("model_outcome_based_selection", False) is not False:
        raise ValueError("fixed retention banks cannot be filtered or replaced by model outcomes")
    _integer(registration.get("selection_seed"), 0, 2**63 - 1)
    _keys(registration.get("counts"), set(ROLES))
    counts = {}
    for role in ROLES:
        _keys(registration["counts"][role], set(DOMAINS))
        counts[role] = {}
        for domain in DOMAINS:
            count = _integer(registration["counts"][role][domain], 0, 100000)
            if count % 4 or (role in {"TRAIN_SELECTION", "FINAL"} and count != 0):
                raise ValueError("four balanced strata required; TRAIN_SELECTION and FINAL counts must be zero")
            counts[role][domain] = count
    if not sum(sum(values.values()) for values in counts.values()):
        raise ValueError("empty registration")
    return counts


def prepare_parents(counts: dict[str, dict[str, int]], seed: int) -> tuple[list[AllocatedParent], dict]:
    """Exact parent/stratum quotas by deterministic order, before solve/render."""
    _integer(seed, 0, 2**63 - 1)
    _keys(counts, set(ROLES))
    quotas = {}
    for role in ROLES:
        _keys(counts[role], set(DOMAINS))
        for domain in DOMAINS:
            count = _integer(counts[role][domain], 0, 100000)
            if count % 4 or (role in {"TRAIN_SELECTION", "FINAL"} and count):
                raise ValueError("invalid balanced parent quota or closed FINAL population")
            for stratum in (["format_" + str(i) for i in range(4)] if domain == "instruction_following" else POSITIONS):
                quotas[(domain, role, stratum)] = count // 4
    selected, seen, attempts, duplicates = [], set(), 0, 0
    for domain in DOMAINS:
        keys = [key for key in quotas if key[0] == domain]
        ceiling = max(10000, sum(quotas[key] for key in keys) * 1000)
        index = 0
        while any(quotas[key] for key in keys):
            if index >= ceiling:
                raise RuntimeError("candidate ceiling exhausted; no automatic seed change")
            parent = allocate_parent(generate_spec(domain, seed, index), seed, index)
            index += 1
            attempts += 1
            key = (domain, parent.role, parent.stratum)
            if quotas.get(key, 0) <= 0:
                continue
            if parent.canonical_parent_sha256 in seen:
                duplicates += 1
                continue
            seen.add(parent.canonical_parent_sha256)
            selected.append(parent)
            quotas[key] -= 1
    return selected, {"candidate_specifications": attempts, "duplicate_candidates_skipped": duplicates,
                      "selection_basis": "fixed original candidate order, role/domain/stratum quotas; no answer or model scoring"}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(json_bytes(row) + b"\n")


def build_dataset(registration_path: Path, output: Path, source_revision: str) -> dict:
    if type(source_revision) is not str or not re.fullmatch("[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a full lowercase Git SHA")
    started = time.perf_counter()
    raw = registration_path.read_bytes()
    registration = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    counts = validate_registration(registration)
    if output.exists():
        raise FileExistsError("export is write-once; output directory already exists")
    parents, selection = prepare_parents(counts, registration["selection_seed"])
    rows, reservations, prompt_hashes = {role: [] for role in OPEN_ROLES}, [], {}
    strata = Counter()
    for parent in parents:
        spec = _validated_spec(parent, require_open=False)
        strata[(parent.role, spec["domain"], parent.stratum)] += 1
        if parent.role not in OPEN_ROLES:
            reservations.append({"parent_id": parent.parent_id, "role": parent.role,
                                 "canonical_parent_sha256": parent.canonical_parent_sha256,
                                 "split_sha256": parent.split_sha256, "status": "HASH_ONLY_UNOPENED"})
            continue
        variants = ("answerable", "missing_evidence") if spec["domain"] == "answerability_abstention" else ("standard",)
        for variant in variants:
            row = make_row(parent, source_revision, variant)
            digest = row["normalized_prompt_sha256"]
            if digest in prompt_hashes:
                raise RuntimeError("duplicate normalized prompt; no silent substitution")
            prompt_hashes[digest] = row["row_id"]
            rows[parent.role].append(row)
    output.mkdir(parents=True, exist_ok=False)
    (output / "REGISTRATION.json").write_bytes(raw)
    for role in OPEN_ROLES:
        _write_jsonl(output / (role + ".jsonl"), rows[role])
    _write_jsonl(output / "SEALED_RESERVATIONS.jsonl", reservations)
    files = {}
    for name in ("REGISTRATION.json", "TRAIN.jsonl", "DEV.jsonl", "SEALED_RESERVATIONS.jsonl"):
        content = (output / name).read_bytes()
        files[name] = {"sha256": sha256(content), "bytes": len(content)}
    total_rows = sum(map(len, rows.values()))
    receipt = {
        "schema_version": 1, "status": "MEASURED_DETERMINISTIC_RETENTION_MECHANICS_ONLY",
        "generator_version": VERSION, "verifier_version": VERIFIER_VERSION,
        "generator_file_sha256": sha256(Path(__file__).read_bytes()), "source_revision": source_revision,
        "source_revision_verification": "CALLER_SUPPLIED_GIT_SHA; actual generator bytes independently bound above",
        "registration_sha256": sha256(raw), "selection_seed": registration["selection_seed"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "runtime": {"python_version": sys.version, "platform": sys.platform, "dependencies": "Python standard library only",
                    "determinism_scope": "same code, registration and Python runtime"},
        "role_allocator": {"domain_separator_hex": ROLE_DOMAIN.hex(),
                           "hash": "SHA256(domain_separator + semantic_UTF8_JSON); first 64 bits modulo 10000",
                           "exclusive_upper_buckets": [[high, role] for high, role in ROLE_BUCKETS]},
        "independent_unit": "canonical_semantic_parent; answerability twins count once",
        "parent_counts": {role: sum(counts[role].values()) for role in ROLES},
        "domain_parent_counts": counts, "open_row_counts": {role: len(rows[role]) for role in OPEN_ROLES},
        "domain_row_counts": {role: dict(Counter(row["domain"] for row in rows[role])) for role in OPEN_ROLES},
        "stratum_parent_counts": {role: {domain: {stratum: number for (r, d, stratum), number in sorted(strata.items())
                                                  if r == role and d == domain} for domain in DOMAINS} for role in ROLES},
        "selection": selection, "sealed_exports": "CHECK hash identities only; no spec, prompt, target or verifier payload",
        "final_count_status": "ZERO_CLOSED_UNSPECIFIED_CONFIRMATION_POPULATION",
        "controls": {"positive_reference_answers_verified": total_rows, "wrong_answer_mutations_rejected": total_rows,
                     "canonical_parent_duplicates": 0, "cross_parent_prompt_collisions": 0},
        "normalization_scope": ["record ordering and duplicate identical facts", "same world across retrieval/abstention domains",
                                "all query/template derivatives grouped despite different exact targets", "priority-directive ordering and identical duplicates",
                                "twins share complete world parent"],
        "not_proven": ["all possible semantic equivalence", "historical/pretraining exclusion", "source-correct Titan membership",
                       "native token fit", "language preservation", "long-context preservation", "native learning efficacy"],
        "cost": {"wall_seconds": time.perf_counter() - started, "model_calls": 0, "teacher_calls": 0,
                 "gpu_seconds": 0, "model_generation_tokens": 0, "actual_training_tokens": 0,
                 "dataset_token_count": None, "dataset_token_count_status": "UNMEASURED"},
        "tokenizer_context_qualified": False, "historical_exclusion_qualified": False,
        "source_correct_enrollment_qualified": False, "retention_dataset_qualified": False,
        "language_preservation_qualified": False, "long_context_preservation_qualified": False,
        "broad_preservation_coverage_qualified": False, "native_qualified": False, "e1_launch_ready": False,
    }
    (output / "DATA_RECEIPT.json").write_bytes(json_bytes(receipt) + b"\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    receipt = build_dataset(args.registration, args.output, args.source_revision)
    path = args.output / "DATA_RECEIPT.json"
    print(json.dumps({"receipt": str(path.resolve()), "receipt_sha256": sha256(path.read_bytes()),
                      "parent_counts": receipt["parent_counts"], "open_row_counts": receipt["open_row_counts"],
                      "e1_launch_ready": False}, indent=2))


if __name__ == "__main__":
    main()
