"""Bind the four existing DEV populations to separated policy/evaluator inputs.

Stdlib only: no tokenizer, generator, scorer, native framework or environment
mutation. ``bind(package_root, config)`` reads only FILES, verifies the exact bytes
against the prospective config and recorded receipt identities, then checks all
joins, stored arrays and population contracts. The result is source preparation,
not proof of model execution or permission to grade language.

Config has exactly schema_version=1, status=FROZEN_DEV_INPUT_BINDING_ONLY,
primary_weight_sha256, file_identities={repo-relative path: {sha256, bytes}},
native_execution=false and language_scoring=false. FILES is the exact inventory.

Policy records have exactly six fields. Their row_id is routing metadata, never
model text. A future generator consumes only input_ids/attention_mask, or the
separately verified matching messages. Short-task stored input_ids include a
reference completion: only their verified prompt prefix leaves this binder.
Complete original rows, including targets and answerability metadata, appear only
in evaluator_rows. No response collection, scoring, publication or CLI lives here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


PACKAGE = "experiments/post_training_mimo_centralized_v1/"
ARTIFACTS = "artifacts/mimo_central/"
WEIGHT_SHA256 = "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7"
EOS = 151645
VOCAB_SIZE = 151936
TEMPLATE_KWARGS = {"add_generation_prompt": True, "enable_thinking": False}
PANELS = ("acquisition", "deterministic_retention", "long_context", "language")
COUNTS = {"acquisition": (256, 256), "deterministic_retention": (256, 192),
          "long_context": (256, 32), "language": (64, 64)}
ACQUISITION_FAMILIES = ("bounded_integer_arithmetic", "symbolic_variable_updates",
                        "finite_set_queries", "constrained_text_transformations")
RETENTION_FAMILIES = ("retrieval", "instruction_following", "answerability_abstention")
LC_FAMILIES = ("three_fact_keyed_join", "latest_authorized_record_selection")
POSITIONS = ("early", "middle", "late", "distributed")
LANGUAGE_STRATA = ("explanation", "meaning_preserving_rewrite", "synthesis", "clarification",
                   "supportive_conversation", "disagreement_and_compromise",
                   "contextual_recommendation", "followup_and_closure")
POLICY_FIELDS = {"row_id", "messages", "input_ids", "attention_mask", "prompt_tokens", "prompt_token_hash"}
EVALUATOR_FIELDS = {"row_id", "parent_id", "family", "stratum", "variant", "role", "panel",
                    "prompt_token_hash", "scoring_payload"}
PATHS = {
    "primary": PACKAGE + "receipts/PRIMARY_STUDENT_RECEIPT.json",
    "tokenizer": PACKAGE + "receipts/TOKENIZER_RECEIPT.json",
    "acquisition_receipt": PACKAGE + "datasets/bridge_v1/DATA_RECEIPT.json",
    "acquisition_registration": PACKAGE + "datasets/bridge_v1/REGISTRATION.json",
    "acquisition_source": PACKAGE + "datasets/bridge_v1/DEV.jsonl",
    "acquisition_profile": PACKAGE + "receipts/BRIDGE_TOKENIZATION_RECEIPT.json",
    "acquisition_tokens": ARTIFACTS + "bridge_tokenized_v1/DEV.tokenized.jsonl",
    "deterministic_retention_receipt": ARTIFACTS + "retention_v1/DATA_RECEIPT.json",
    "deterministic_retention_registration": ARTIFACTS + "retention_v1/REGISTRATION.json",
    "deterministic_retention_source": ARTIFACTS + "retention_v1/DEV.jsonl",
    "deterministic_retention_profile": PACKAGE + "receipts/RETENTION_TOKENIZATION_RECEIPT.json",
    "deterministic_retention_tokens": ARTIFACTS + "retention_tokenized_v1/DEV.tokenized.jsonl",
    "long_context_receipt": PACKAGE + "receipts/LONG_CONTEXT_DEV_DATA_RECEIPT.json",
    "long_context_registration": PACKAGE + "manifests/LONG_CONTEXT_FAST_REGISTRATION.json",
    "long_context_export_registration": ARTIFACTS + "long_context_dev_v1/REGISTRATION.json",
    "long_context_source": ARTIFACTS + "long_context_dev_v1/DEV.jsonl",
    "language_receipt": PACKAGE + "receipts/LANGUAGE_DEV_CAL_CANDIDATE_RECEIPT.json",
    "language_profile": PACKAGE + "receipts/LANGUAGE_DEV_CAL_TOKENIZATION_RECEIPT.json",
    "language_registration": PACKAGE + "manifests/LANGUAGE_AUTHORING_REGISTRATION.json",
    "language_freeze": PACKAGE + "manifests/LANGUAGE_WORLD_SPEC_FREEZE.json",
    "language_source": ARTIFACTS + "language_dev_cal_v1/DEV.jsonl",
    "language_tokens": ARTIFACTS + "language_eval_tokenized_v1/DEV.prompts.jsonl",
}
FILES = tuple(PATHS.values())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    if type(value) is dict:
        require(all(type(k) is str for k in value), "JSON keys must be strings")
        for child in value.values():
            _finite(child)
    elif type(value) is list:
        for child in value:
            _finite(child)
    elif type(value) is float:
        require(math.isfinite(value), "nonfinite JSON")
    else:
        require(value is None or type(value) in (str, int, bool), "JSON values required")


def canonical(value):
    _finite(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON: " + value)
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    _finite(result)
    return result


def _same(actual, expected, label):
    require(canonical(actual) == canonical(expected), label + " drift")


def _text(value, label):
    require(type(value) is str and bool(value.strip()), label + " string required")


def _integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, label + " integer/range")


def _hash(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, label + " SHA256")


def _hash_text(value, observed, label):
    require(type(value) is str and hashlib.sha256(value.encode("utf-8")).hexdigest() == observed, label + " hash drift")


def _ids(values, label, *, nonempty=True):
    require(type(values) is list and (bool(values) or not nonempty), label + " token array")
    require(all(type(v) is int and 0 <= v < VOCAB_SIZE for v in values), label + " token dtype/range")


def _path(root, label):
    require(type(label) is str and label in FILES, "file outside fixed DEV inventory")
    candidate = root / label
    # Reject redirecting file/directory symlinks, including links into a protected
    # bank, before opening. The resolved workspace root itself is caller supplied.
    require(candidate.resolve(strict=True) == candidate, "redirected DEV path")
    require(candidate.is_file(), "DEV input is not a regular file")
    return candidate


def _read(root, label):
    path = _path(root, label)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            and len(raw) == after.st_size, "input changed during read: " + label)
    _path(root, label)
    return raw


def _config(config):
    _finite(config)
    require(type(config) is dict and set(config) == {
        "schema_version", "status", "primary_weight_sha256", "file_identities", "native_execution", "language_scoring"},
        "DEV binding config field inventory")
    require(type(config["schema_version"]) is int and config["schema_version"] == 1
            and config["status"] == "FROZEN_DEV_INPUT_BINDING_ONLY", "DEV config stage/version")
    require(config["primary_weight_sha256"] == WEIGHT_SHA256 and config["native_execution"] is False
            and config["language_scoring"] is False, "DEV config scope/primary identity")
    pinned = config["file_identities"]
    require(type(pinned) is dict and set(pinned) == set(FILES), "config must pin exact fixed DEV file inventory")
    for label, value in pinned.items():
        require(type(value) is dict and set(value) == {"sha256", "bytes"}, "file identity fields")
        _hash(value["sha256"], label)
        _integer(value["bytes"], 1, 100_000_000, "file byte ceiling")


def _rows(raw, field):
    lines = raw.splitlines()
    require(lines and all(line.strip() for line in lines), "blank/empty DEV JSONL")
    rows = [_parse(line) for line in lines]
    require(all(type(row) is dict and row.get("role") == "DEV" for row in rows), "DEV rows only; closed/calibration role")
    keys = [row.get(field) for row in rows]
    require(all(type(key) is str and key and key == key.strip() for key in keys)
            and len(keys) == len(set(keys)), "missing/duplicate DEV row identity")
    return rows


def _messages(messages, *, single=False):
    require(type(messages) is list and len(messages) in ((1,) if single else (1, 3, 5, 7)), "conversation length")
    for i, message in enumerate(messages):
        require(type(message) is dict and set(message) == {"role", "content"}
                and message["role"] == ("user" if i % 2 == 0 else "assistant"), "conversation boundary/role")
        _text(message["content"], "conversation content")


def _short(source, token, panel, receipt, registration):
    require(source.get("schema_version") == 1 and type(source["schema_version"]) is int, "source schema")
    for field in ("row_id", "parent_id", "family", "role", "variant_id"):
        _same(token.get(field), source[field], "source/token " + field)
    require(source["role"] == "DEV" and source["training_role_eligible"] is False
            and token["training_role_eligible"] is False, "short DEV eligibility")
    require(source["source_revision"] == receipt["source_revision"] == source["environment_revision"], "source revision drift")
    _same(source["generator_seed"], registration["selection_seed"], "source generator seed")
    require(type(source["spec"]) is dict, "source specification required")
    for field in ("prompt", "target"):
        _hash_text(source[field], source[field + "_sha256"], "source " + field)
        require(token["source_" + field + "_sha256"] == source[field + "_sha256"], "token source text drift")
    _hash_text(" ".join(source["prompt"].split()), source["normalized_prompt_sha256"], "normalized prompt")
    require(token["source_row_sha256"] == digest(source), "source/token canonical row join drift")
    if panel == "acquisition":
        require(source["source_id"] == "original-rineya-e1-bridge" and source["spec"]["family"] == source["family"], "acquisition source identity")
        _integer(source["variant_id"], 0, 0, "acquisition variant")
        require(source["row_id"] == source["parent_id"] + "-v0", "acquisition row ID")
        stratum, variant = source["structure_id"], str(source["variant_id"])
    else:
        require(source["source_id"] == "original-rineya-retention" and source["spec"]["domain"] == source["domain"] == source["family"], "retention source identity")
        stratum, variant = source["stratum"], source["variant_id"]
        require(source["row_id"] == source["parent_id"] + "-" + variant, "retention row ID")
        for field in ("stratum", "domain", "canonical_parent_sha256", "split_sha256", "source_revision"):
            _same(token[field], source[field], "retention token " + field)
    _text(stratum, "short stratum")
    p, c, n = token["prompt_tokens"], token["completion_tokens"], token["nonpadding_tokens"]
    _integer(p, 1, 1023, "prompt length")
    _integer(c, 1, 128, "completion length")
    _integer(n, 1, 1024, "sequence length")
    require(n == p + c and type(token["padding_tokens"]) is int and token["padding_tokens"] == 0, "short length/padding")
    _integer(token["target_json_tokens"], 0, 127, "target length")
    require(token["target_json_tokens"] == c - 1, "target/EOS accounting")
    ids = token["input_ids"]
    _ids(ids, "stored short sequence")
    require(len(ids) == n and type(token["completion_eos_id"]) is int and token["completion_eos_id"] == EOS,
            "short EOS/length contract")
    require(ids[-1] == EOS and not {151643, 151644, EOS}.intersection(ids[p:-1]), "short target termination")
    _same(token["attention_mask"], [1] * n, "short attention mask")
    _same(token["completion_mask"], [0] * p + [1] * c, "short completion mask")
    _same(token["labels"], [-100] * p + ids[p:], "short positional labels")
    for field, values in (("input_token_ids_sha256", ids), ("tokenized_prompt_sha256", ids[:p]),
                          ("completion_token_ids_sha256", ids[p:])):
        require(token[field] == digest(values), field + " drift")
    _same(token["template_kwargs"], TEMPLATE_KWARGS, "short template kwargs")
    rendered = "<|im_start|>user\n" + source["prompt"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    _hash_text(rendered, token["rendered_prompt_sha256"], "short rendered prompt")
    return source["row_id"], source["family"], stratum, variant, [{"role": "user", "content": source["prompt"]}], ids[:p]


def _long(source):
    require(source["row_sha256"] == digest({k: v for k, v in source.items() if k != "row_sha256"}), "long-context row hash drift")
    p = source["prompt_tokens"]
    require(type(p) is int and p in (2024, 4072), "long-context registered prompt length")
    ids = source["input_ids"]
    _ids(ids, "long-context prompt")
    require(len(ids) == p and source["input_token_ids_sha256"] == digest(ids), "long-context prompt hash/length")
    for field in ("prompt", "rendered_prompt", "context_body", "target"):
        _hash_text(source[field], source[field + "_sha256"], "long-context " + field)
    _same(source["template_kwargs"], TEMPLATE_KWARGS, "long-context template kwargs")
    _same(source["generation_reserve_tokens"], 24, "long-context reserve")
    _same(source["total_sequence_envelope"], p + 24, "long-context envelope")
    require(type(source["answerable"]) is bool and type(source["adversarial"]) is bool, "long-context factor type")
    variant = ("answerable" if source["answerable"] else "unanswerable") + ("_adversarial" if source["adversarial"] else "_neutral")
    require(source["variant_id"] == variant and source["row_id"] == f"{source['parent_id']}-{p}-{variant}", "long-context factor identity")
    _ids(source["target_token_ids"], "long-context target")
    _integer(source["target_tokens"], 1, 16, "long-context target length")
    require(len(source["target_token_ids"]) == source["target_tokens"]
            and source["target_token_ids_sha256"] == digest(source["target_token_ids"]), "long-context target hash/length")
    require(not {151643, 151644, EOS}.intersection(source["target_token_ids"]), "long-context target special token")
    _same(source["policy_input_only"], ["prompt", "input_ids"], "long-context input scope")
    require(source["training_eligibility"] is False and source["native_qualification_inferred"] is False
            and source["source_correct_membership"] is None, "long-context remains unqualified")
    rendered = "<|im_start|>user\n" + source["prompt"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    require(source["rendered_prompt"] == rendered, "long-context wrapper drift")
    return source["row_id"], source["family"], source["evidence_position"], f"{p}-{variant}", [{"role": "user", "content": source["prompt"]}], list(ids)


def _language(source, token, freeze):
    require(source["row_sha256"] == digest({k: v for k, v in source.items() if k != "row_sha256"}), "language source row hash")
    parent = source["parent_id"]
    require(parent == token["parent_id"] and source["role"] == token["role"] == "DEV"
            and source["stratum"] == token["stratum"], "language source/token identity")
    world = {k: source[k] for k in ("parent_id", "role", "stratum", "conversation_form", "semantic_world", "author_id")}
    require(source["world_spec_sha256"] == digest(world), "language semantic world hash")
    entries = [entry for entry in freeze["worlds"] if entry["parent_id"] == parent]
    require(len(entries) == 1, "language frozen DEV membership")
    for field in ("parent_id", "role", "stratum", "conversation_form", "author_id", "world_spec_sha256", "spec_file_sha256"):
        _same(source[field], entries[0][field], "language freeze " + field)
    require(source["spec_file_sha256"] == freeze["source_files"][source["spec_source_path"]]["sha256"], "language spec-file lineage")
    payload = {k: source[k] for k in ("parent_id", "world_spec_sha256", "messages", "semantic_contract", "dev_exemplar", "cal_controls", "authorship")}
    require(source["payload_world_sha256"] == digest(payload), "language payload hash")
    require(source["authorship"] == "assistant_source_candidate" and source["cal_controls"] == [], "language DEV source scope")
    _text(source["dev_exemplar"], "language illustrative exemplar")
    _messages(source["messages"])
    require(source["conversation_form"] == ("single" if len(source["messages"]) == 1 else "multi"), "language conversation form")
    require(token["source_row_sha256"] == digest(source) and token["messages_sha256"] == digest(source["messages"]), "language source/token row join")
    _same(token["prior_authored_assistant_turns"], len(source["messages"]) // 2, "language prior assistant turns")
    _integer(token["prompt_tokens"], 1, 896, "language prompt cap")
    ids = token["prompt_token_ids"]
    _ids(ids, "language prompt")
    require(len(ids) == token["prompt_tokens"] and token["prompt_token_ids_sha256"] == digest(ids), "language prompt hash/length")
    require(token["truncation"] is False and token["native_qualified"] is False and token["training_eligibility"] is False,
            "language profile scope")
    refs = token["references"]
    require(type(refs) is list and len(refs) == 1 and refs[0]["reference_id"] == "dev_exemplar", "language DEV reference inventory")
    ref = refs[0]
    _hash_text(source["dev_exemplar"], ref["response_sha256"], "language reference")
    completion = ref["completion_token_ids"]
    _ids(completion, "language reference tokens")
    _integer(ref["completion_tokens_including_EOS"], 1, 128, "language reference cap")
    require(len(completion) == ref["completion_tokens_including_EOS"] and completion[-1] == EOS
            and not {151643, 151644, EOS}.intersection(completion[:-1])
            and ref["completion_token_ids_sha256"] == digest(completion)
            and ref["qualified_reference_labels"] is False, "language reference hash/termination/scope")
    return parent + "-dev", "general_language", source["stratum"], "primary", deepcopy(source["messages"]), list(ids)


def _population(panel, evaluator):
    row_count, parent_count = COUNTS[panel]
    require(len(evaluator) == row_count and len({r["parent_id"] for r in evaluator}) == parent_count, panel + " row/parent quota")
    groups = defaultdict(list)
    for row in evaluator:
        groups[row["parent_id"]].append(row)
    if panel == "acquisition":
        _same(dict(Counter(r["family"] for r in evaluator)), dict.fromkeys(ACQUISITION_FAMILIES, 64), "acquisition family quota")
        require(all(r["variant"] == "0" for r in evaluator), "acquisition fixed variant")
    elif panel == "deterministic_retention":
        _same(dict(Counter(r["family"] for r in evaluator)), {"retrieval": 64, "instruction_following": 64, "answerability_abstention": 128}, "retention family quota")
        parent_cells = Counter()
        for rows in groups.values():
            first = rows[0]
            require(all((r["family"], r["stratum"]) == (first["family"], first["stratum"]) for r in rows), "retention parent metadata")
            variants = {"answerable", "missing_evidence"} if first["family"] == "answerability_abstention" else {"standard"}
            require(len(rows) == len(variants) and {r["variant"] for r in rows} == variants, "retention twin coverage")
            parent_cells[(first["family"], first["stratum"])] += 1
        expected = {(family, stratum): 16 for family in RETENTION_FAMILIES
                    for stratum in (["format_" + str(i) for i in range(4)] if family == "instruction_following" else POSITIONS)}
        require(parent_cells == Counter(expected), "retention stratum quota")
    elif panel == "long_context":
        parent_cells = Counter()
        expected_variants = {f"{p}-{a}_{d}" for p in (2024, 4072) for a in ("answerable", "unanswerable") for d in ("neutral", "adversarial")}
        for rows in groups.values():
            first = rows[0]
            require(len(rows) == 8 and {r["variant"] for r in rows} == expected_variants, "long-context eight-variant coverage")
            for row in rows:
                require((row["family"], row["stratum"]) == (first["family"], first["stratum"]), "long-context parent metadata")
                for field in ("missing_index", "split_sha256", "canonical_parent_sha256", "query", "structural_cluster_id"):
                    _same(row["scoring_payload"]["source_row"][field], first["scoring_payload"]["source_row"][field], "long-context derivative " + field)
            parent_cells[(first["family"], first["stratum"])] += 1
        require(parent_cells == Counter({(f, s): 4 for f in LC_FAMILIES for s in POSITIONS}), "long-context family/placement quota")
    else:
        _same(dict(Counter(r["stratum"] for r in evaluator)), dict.fromkeys(LANGUAGE_STRATA, 8), "language stratum quota")
        for row in evaluator:
            require(row["parent_id"] in {f"LDEV-{row['stratum']}-{i:02d}" for i in range(1, 9)}, "language fixed DEV ID")
            ordinal = int(row["parent_id"].rsplit("-", 1)[1])
            expected_form = "single" if ordinal <= 4 else "multi"
            require(row["scoring_payload"]["source_row"]["conversation_form"] == expected_form, "language fixed conversation form")
    return {"rows": row_count, "parents": parent_count,
            "family_rows": dict(sorted(Counter(r["family"] for r in evaluator).items()))}


def _receipts(doc, ids):
    """Cross-check recorded artifact identities, never follow a receipt's paths."""
    primary, pin = doc["primary"], doc["tokenizer"]
    require(primary["weight_sha256"] == pin["weight_sha256"] == WEIGHT_SHA256, "static Titan identity")
    _same(primary["parameter_count"], 596049920, "Titan parameter count")
    require(primary["tokenizer_sha256"] == pin["tokenizer_sha256"] == pin["tokenizer_files"]["tokenizer.json"]["sha256"], "static tokenizer identity")
    require(primary["chat_template_sha256"] == pin["chat_template_sha256"], "static template identity")
    _hash_text(pin["chat_template"], pin["chat_template_sha256"], "pinned chat template")
    _same(pin["eos_token_id"], EOS, "pinned EOS")
    for panel, data_status, profile_status in (
        ("acquisition", "MEASURED_DETERMINISTIC_DATA_MECHANICS_ONLY", "MEASURED_OPEN_ROLE_TOKENIZATION_ONLY"),
        ("deterministic_retention", "MEASURED_DETERMINISTIC_RETENTION_MECHANICS_ONLY", "MEASURED_OPEN_RETENTION_ROLE_TOKENIZATION_ONLY")):
        receipt, profile, reg = (doc[panel + suffix] for suffix in ("_receipt", "_profile", "_registration"))
        require(receipt["status"] == data_status and profile["status"] == profile_status, "short receipt status")
        _same(receipt["files"]["DEV.jsonl"], ids[panel + "_source"], "construction DEV identity")
        _same(receipt["files"]["REGISTRATION.json"], ids[panel + "_registration"], "construction registration identity")
        require(receipt["registration_sha256"] == ids[panel + "_registration"]["sha256"], "construction registration hash")
        require(profile["source_data_receipt_sha256"] == ids[panel + "_receipt"]["sha256"]
                and profile["source_registration_sha256"] == ids[panel + "_registration"]["sha256"], "short profile source receipt link")
        _same(profile["source_open_files_verified"]["DEV.jsonl"], ids[panel + "_source"], "profile source DEV identity")
        _same(profile["files"]["DEV.tokenized.jsonl"], ids[panel + "_tokens"], "profile token DEV identity")
        _same(receipt["open_row_counts"]["DEV"], 256, "construction DEV count")
        _same(profile["source_row_counts"]["DEV"], 256, "profile DEV count")
        _same(profile["source_role_parent_counts"]["DEV"], COUNTS[panel][1], "profile parent count")
        require(profile["tokenizer_receipt_sha256"] == ids["tokenizer"]["sha256"], "profile tokenizer receipt")
        _same(profile["tokenizer_files"], pin["tokenizer_files"], "profile tokenizer assets")
        require(profile["chat_template_sha256"] == pin["chat_template_sha256"], "profile template")
        _same(profile["template_kwargs"], TEMPLATE_KWARGS, "profile template kwargs")
        require(profile["truncation"] is False and profile["native_qualification_inferred"] is False, "profile qualification scope")
        if panel == "acquisition":
            require(reg["status"] == "FROZEN_FAST_LANE_DATA_ONLY", "acquisition registration stage")
            _same(reg["counts"]["DEV"], {"acquisition_parents": 256, "per_family": 64}, "acquisition registration quota")
        else:
            require(reg["status"] == "FROZEN_FAST_LANE_RETENTION_ONLY" and reg["primary_student_weight_sha256"] == WEIGHT_SHA256, "retention registration scope")
            _same(reg["counts"]["DEV"], dict.fromkeys(RETENTION_FAMILIES, 64), "retention registration quota")
    lc, reg = doc["long_context_receipt"], doc["long_context_registration"]
    require(lc["status"] == "MEASURED_LONG_CONTEXT_DEV_TOKENIZER_MECHANICS_ONLY"
            and reg["status"] == "FROZEN_FAST_LANE_LONG_CONTEXT_ONLY", "long-context receipt/registration stage")
    _same(lc["files"]["DEV.jsonl"], ids["long_context_source"], "long-context DEV identity")
    _same(lc["files"]["REGISTRATION.json"], ids["long_context_export_registration"], "long-context exported registration")
    _same(lc["registration_identity"], ids["long_context_registration"], "long-context prospective registration")
    _same(reg, doc["long_context_export_registration"], "long-context registration semantic copy")
    _same(reg["counts"]["DEV"], 32, "long-context parent quota")
    _same(reg["prompt_lengths"], [2024, 4072], "long-context lengths")
    _same(lc["open_row_counts"]["DEV"], 256, "long-context row quota")
    _same(lc["tokenizer_receipt_identity"], ids["tokenizer"], "long-context tokenizer receipt")
    _same(lc["tokenizer_files"], pin["tokenizer_files"], "long-context tokenizer assets")
    require(lc["chat_template_sha256"] == reg["chat_template_sha256"] == pin["chat_template_sha256"]
            and reg["tokenizer_sha256"] == pin["tokenizer_sha256"] and reg["primary_student_weight_sha256"] == WEIGHT_SHA256,
            "long-context static identities")
    _same(lc["template_kwargs"], TEMPLATE_KWARGS, "long-context template kwargs")
    language, profile, reg, freeze = (doc[name] for name in ("language_receipt", "language_profile", "language_registration", "language_freeze"))
    require(language["status"] == "AUTHORED_LANGUAGE_CANDIDATES_UNQUALIFIED"
            and profile["status"] == "MEASURED_LANGUAGE_DEV_CAL_TOKENIZER_ONLY_UNQUALIFIED_GRADING"
            and reg["status"] == "ACCEPTED_DATA_ONLY_AUTHORING_NOT_GRADING_OR_NATIVE_EXECUTION"
            and freeze["status"] == "FROZEN_LANGUAGE_WORLD_SPECS_BEFORE_RENDERING", "language artifact stage")
    _same(language["files"]["DEV.jsonl"], ids["language_source"], "language source DEV identity")
    _same(profile["files"]["DEV.prompts.jsonl"], ids["language_tokens"], "language token DEV identity")
    _same(profile["source_files"]["DEV.jsonl"], ids["language_source"], "language profile source DEV")
    _same(profile["bank_receipt"], ids["language_receipt"], "language profile bank receipt")
    _same(profile["tokenizer_receipt"], ids["tokenizer"], "language profile tokenizer receipt")
    _same(profile["tokenizer_assets"], pin["tokenizer_files"], "language tokenizer assets")
    require(profile["chat_template_sha256"] == pin["chat_template_sha256"], "language template identity")
    _same(profile["template_kwargs"], TEMPLATE_KWARGS, "language template kwargs")
    for name in ("language_registration", "language_freeze"):
        label = PATHS[name][len(PACKAGE):]
        _same(language["input_files"][label], ids[name], "language source input " + label)
    _same(freeze["registration"], ids["language_registration"], "language frozen registration")
    _same(profile["parents"]["DEV"], 64, "language profile DEV count")
    _same(language["counts"]["DEV_parents"], 64, "language source DEV count")
    _same(reg["DEV_parent_count"], 64, "language registered DEV count")
    require(language["grading_qualified"] is False and language["reference_labels_qualified"] is False
            and profile["qualified_reference_labels"] is False and profile["native_qualified"] is False
            and profile["truncation"] is False, "language remains unqualified")


def bind(package_root, config):
    """Return bound DEV sources and prompt-only policy rows; never publish or score."""
    try:
        return _bind(package_root, config)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("incomplete or incompatible DEV source schema: " + str(exc)) from exc


def _bind(package_root, config):
    _config(config)
    config = deepcopy(config)
    package = Path(package_root).resolve(strict=True)
    root = package.parents[1]
    require(package == root / PACKAGE.rstrip("/"), "fixed experiment package root required")
    raw, identities = {}, {}
    for name, label in PATHS.items():
        value = _read(root, label)
        observed = identity(value)
        _same(observed, config["file_identities"][label], "frozen file " + label)
        raw[name], identities[name] = value, observed
    doc = {name: _parse(value) for name, value in raw.items() if PATHS[name].endswith(".json")}
    _receipts(doc, identities)
    policy_rows, evaluator_rows, counts, seen, parent_panels = [], [], {}, set(), {}
    for panel in PANELS:
        key = "parent_id" if panel == "language" else "row_id"
        source_rows = _rows(raw[panel + "_source"], key)
        tokens = None
        if panel != "long_context":
            token_rows = _rows(raw[panel + "_tokens"], key)
            require(len(token_rows) == len(source_rows), "source/token population size mismatch")
            tokens = {r[key]: r for r in token_rows}
            require(set(tokens) == {r[key] for r in source_rows}, "source/token population join mismatch")
        panel_evaluator = []
        for source in source_rows:
            if panel in ("acquisition", "deterministic_retention"):
                mapped = _short(source, tokens[source[key]], panel, doc[panel + "_receipt"], doc[panel + "_registration"])
            elif panel == "long_context":
                mapped = _long(source)
            else:
                mapped = _language(source, tokens[source[key]], doc["language_freeze"])
            row_id, family, stratum, variant, messages, ids = mapped
            require(row_id not in seen, "row ID reused across DEV panels")
            seen.add(row_id)
            parent = source["parent_id"]
            require(parent_panels.setdefault(parent, panel) == panel, "parent reused across DEV panels")
            _messages(messages)
            policy = {"row_id": row_id, "messages": deepcopy(messages), "input_ids": list(ids),
                      "attention_mask": [1] * len(ids), "prompt_tokens": len(ids), "prompt_token_hash": digest(ids)}
            evaluator = {"row_id": row_id, "parent_id": parent, "family": family, "stratum": stratum,
                         "variant": variant, "role": "DEV", "panel": panel, "prompt_token_hash": policy["prompt_token_hash"],
                         "scoring_payload": {"schema_version": 1, "panel": panel, "source_row_sha256": digest(source),
                                             "source_row": deepcopy(source)}}
            require(set(policy) == POLICY_FIELDS and set(evaluator) == EVALUATOR_FIELDS, "output field inventory")
            policy_rows.append(policy)
            evaluator_rows.append(evaluator)
            panel_evaluator.append(evaluator)
        counts[panel] = _population(panel, panel_evaluator)
    require(len(policy_rows) == 832 and len(parent_panels) == 544, "combined DEV population")
    # Re-open only already allowlisted paths and compare exactly the bytes parsed.
    for name, label in PATHS.items():
        require(_read(root, label) == raw[name], "source snapshot changed during DEV binding: " + label)
    bindings = {PATHS[name]: identities[name] for name in PATHS}
    result = {"schema_version": 1, "status": "BOUND_DEV_INPUTS_NOT_NATIVE_ADMISSION", "role": "DEV",
              "policy_rows": policy_rows, "evaluator_rows": evaluator_rows, "bindings": bindings, "counts": counts,
              "config_sha256": digest(config), "primary_weight_sha256": WEIGHT_SHA256,
              "policy_boundary": "row_id is routing only; model input is input_ids/attention_mask or matching messages, never serialized records",
              "independence_unit": "parent; retention twins and eight long-context derivatives remain grouped",
              "native_qualified": False, "native_launch_authorized": False, "native_execution": False,
              "training_eligibility": False, "training_authorized": False, "grading_qualified": False,
              "language_scoring": False, "reference_labels_qualified": False, "source_correct_membership": False,
              "e1_launch_ready": False, "model_calls": 0, "tokenizer_calls": 0,
              "held_out_access": {"CAL": False, "CHECK": False, "FINAL": False}}
    result["data_sha256"] = digest(result)
    return result
