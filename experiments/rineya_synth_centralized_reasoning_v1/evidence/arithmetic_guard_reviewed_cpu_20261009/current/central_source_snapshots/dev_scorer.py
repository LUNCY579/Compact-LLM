"""Fixed deterministic DEV evaluator; no tokenizer, model, plugin or native grant.

Only the three pinned stdlib verifier modules below are imported. Their source
bytes are checked afresh on every validation/scoring call; changing a verifier
requires a new explicit scorer registration. dependency_binding also includes
this callback's own file, which the response importer separately binds.

Evaluator metadata and complete original source rows belong only on the scoring
side. They must never be given to a model as policy inputs. The publisher/binder
must independently bind source files and token streams. For short banks, the
original source has no token IDs, so the outer prompt-token hash can only be
syntax-checked here. Long-context token arrays/hashes are checked without claiming
re-tokenization or native offset/resource qualification.

Malformed means an invalid registered response shape, distinct from a wrong value:
short replies require <=4096 characters of strict JSON with exactly 'answer'.
Family types are checked, excluding booleans as integers; sorted unique sets are
required. Both answerability twins admit null OR two strings structurally.
Wrong well-shaped values, including out-of-answer-range integers, are incorrect
but not malformed. LC uses the strict case-sensitive symbol/UNKNOWN parser with
surrounding whitespace allowed; diagnostic prefixes/JSON/prose are malformed.
Source/schema/lineage errors always raise before response parsing. EOS and runtime
failures remain the importer's responsibility. Language identity may be validated,
but language grading always raises: exemplars are never exact-match graders.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

VERSION = "rineya-dev-deterministic-scorer-v1"
PANELS = ("acquisition", "deterministic_retention", "long_context", "language")
HELPER_SHA256 = {
    "bridge_data.py": "d3a10f84713ae7164775e00514c5584bc4f03525b229f3afe4ebf05f52ea57f1",
    "retention_data.py": "44725c1ed0dd1d9de34a6f3e9b0009c10d8d0a3a5f9bd53f3ec84d58b1c1f09e",
    "long_context_tasks.py": "5563e038ee3b47a90b86ec4c8e1f6285938d2d2105e0dcbf5816a62d42101424",
}
EVALUATOR_FIELDS = frozenset("row_id parent_id family stratum variant role panel prompt_token_hash scoring_payload".split())
PAYLOAD_FIELDS = frozenset("schema_version panel source_row_sha256 source_row".split())
SHORT_COMMON_FIELDS = frozenset("""schema_version row_id parent_id source_id source_revision family role
variant_id canonical_parent_sha256 split_sha256 spec prompt prompt_sha256 normalized_prompt_sha256
prompt_normalization tokenized_prompt_sha256 tokenizer_identity tokenizer_sha256 chat_template_sha256
tokenization_status prompt_tokens target_tokens total_nonpadding_tokens target target_sha256
verifier_version verifier_result mutation_control_rejected generator_seed candidate_index attempt_number
policy_or_teacher_identity policy_version behavior_logprobs environment_identity environment_revision
tool_actions tool_outputs reward termination_reason cost training_role_eligible training_eligibility
training_eligibility_reasons license_provenance""".split())
BRIDGE_FIELDS = SHORT_COMMON_FIELDS | {"source_revision_verification", "structure_id", "prompt_characters", "generation_attempts"}
RETENTION_FIELDS = SHORT_COMMON_FIELDS | {"domain", "stratum", "source_correct_membership",
    "source_correct_membership_policy", "source_correct_enrollment_qualified"}
LC_FIELDS = frozenset("""schema_version row_id parent_id role generator_version semantic_generator_version
verifier_version support_unit split_sha256 family evidence_position variant_id distractor_modes answerable
adversarial missing_index canonical_parent_sha256 structural_cluster_id query context_body prompt rendered_prompt
input_ids prompt_tokens body_tokens total_sequence_envelope generation_reserve_tokens target target_token_ids
target_tokens prompt_sha256 rendered_prompt_sha256 context_body_sha256 target_sha256 input_token_ids_sha256
target_token_ids_sha256 template_kwargs support_spans controlled_slots body_whole_prompt_token_start
body_whole_prompt_token_end body_whole_prompt_char_start body_whole_prompt_char_end native_offset_mapping
visible_text_oracle semantic_verification individual_ablations_unknown policy_input_only training_eligibility
source_correct_membership native_qualification_inferred e1_launch_ready row_sha256""".split())
WORLD_FIELDS = frozenset("parent_id role stratum conversation_form semantic_world author_id".split())
LANGUAGE_PAYLOAD_FIELDS = frozenset("parent_id world_spec_sha256 messages semantic_contract dev_exemplar cal_controls authorship".split())
LANGUAGE_FIELDS = WORLD_FIELDS | LANGUAGE_PAYLOAD_FIELDS | {
    "spec_file_sha256", "spec_source_path", "payload_file_sha256", "payload_source_path", "payload_world_sha256", "row_sha256"}
LANGUAGE_STRATA = ("explanation", "meaning_preserving_rewrite", "synthesis", "clarification",
    "supportive_conversation", "disagreement_and_compromise", "contextual_recommendation", "followup_and_closure")


class EvaluatorDataError(ValueError):
    """Invalid host-side source/evaluator data, never a model malformed response."""


class DependencyDriftError(EvaluatorDataError):
    """An unregistered verifier implementation is present on disk."""


class UnqualifiedLanguageGraderError(ValueError):
    """No qualified language grading adapter is implemented."""


def _require(condition, message):
    if not condition:
        raise EvaluatorDataError(message)


def _finite(value):
    if type(value) is dict:
        _require(all(type(k) is str for k in value), "JSON object keys must be strings")
        for child in value.values():
            _finite(child)
    elif type(value) is list:
        for child in value:
            _finite(child)
    elif type(value) is float:
        _require(math.isfinite(value), "nonfinite JSON number")
    else:
        _require(value is None or type(value) in (str, int, bool), "JSON values required")


def canonical(value):
    _finite(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _text_hash(value):
    _require(type(value) is str, "source text must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _keys(value, expected, label):
    _require(type(value) is dict and set(value) == set(expected), label + " schema mismatch")


def _hash(value, label):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, label + " must be SHA256")


def _nonempty(value):
    return type(value) is str and bool(value.strip()) and value == value.strip()


def _same(left, right):
    return canonical(left) == canonical(right)


def _identity(path):
    before = path.stat()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "source changed during dependency hashing")
    return {"sha256": sha, "bytes": after.st_size}


def _verify_dependency_files():
    directory = Path(__file__).resolve().parent
    identities = {}
    for name, expected in HELPER_SHA256.items():
        path = directory / name
        if not path.is_file():
            raise DependencyDriftError("verifier module location drift: " + name)
        identity = _identity(path)
        if identity["sha256"] != expected:
            raise DependencyDriftError("unregistered verifier source bytes: " + name)
        identities[name] = {**identity, "expected_sha256": expected}
    return identities


# Check bytes before executing any local dependency, so an edited helper cannot
# import a tokenizer/native framework before the dispatch-time guard runs.
_verify_dependency_files()
from . import bridge_data as bridge
from . import retention_data as retention
from . import long_context_tasks as long_context


def _verify_dependencies():
    identities = _verify_dependency_files()
    directory = Path(__file__).resolve().parent
    for name, module in {"bridge_data.py": bridge, "retention_data.py": retention,
                         "long_context_tasks.py": long_context}.items():
        if Path(module.__file__).resolve() != directory / name:
            raise DependencyDriftError("verifier module location drift: " + name)
    return identities


def dependency_binding():
    """Fixed dependency closure, checked now; callers must recheck before scoring."""
    files = _verify_dependencies()
    files["dev_scorer.py"] = _identity(Path(__file__).resolve())
    return {"schema_version": 1, "version": VERSION,
            "callable": "experiments.post_training_mimo_centralized_v1.scripts.dev_scorer.score_response",
            "result_schema": {"correct": "strict_boolean", "malformed": "strict_boolean"},
            "scoring_panels": list(PANELS[:-1]), "language_scoring_qualified": False,
            "source_directory": "experiments/post_training_mimo_centralized_v1/scripts",
            "files": files, "dependency_sha256": digest(files)}


def validate_dependency_binding(binding):
    """No arbitrary paths, callbacks or expected-hash overrides are accepted."""
    _require(_same(binding, dependency_binding()), "scorer/dependency binding drift")


def _common_short(source, module, parent):
    expected = {"schema_version": 1, "role": "DEV", "parent_id": parent.parent_id,
        "canonical_parent_sha256": parent.canonical_parent_sha256, "split_sha256": parent.split_sha256,
        "spec": json.loads(parent.canonical_json), "environment_identity": module.VERSION,
        "verifier_version": module.VERIFIER_VERSION, "verifier_result": True,
        "mutation_control_rejected": True, "attempt_number": 1,
        "policy_or_teacher_identity": "DETERMINISTIC_REFERENCE_NO_MODEL", "policy_version": None,
        "behavior_logprobs": None, "tool_actions": [], "tool_outputs": [], "reward": 1,
        "termination_reason": "verified_exact_answer", "training_role_eligible": False,
        "training_eligibility": False, "prompt_normalization": "collapse Unicode whitespace only",
        "tokenization_status": "UNMEASURED"}
    for field in ("tokenized_prompt_sha256", "tokenizer_identity", "tokenizer_sha256", "chat_template_sha256",
                  "prompt_tokens", "target_tokens", "total_nonpadding_tokens"):
        expected[field] = None
    for field, value in expected.items():
        _require(_same(source[field], value), "short source metadata drift: " + field)
    _require(parent.role == "DEV", "short source canonical parent is not DEV")
    _require(type(source["source_revision"]) is str and re.fullmatch(r"[0-9a-f]{40}", source["source_revision"]), "source revision must be full Git SHA")
    _require(source["environment_revision"] == source["source_revision"], "environment/source revision drift")
    for field in ("prompt", "target"):
        _require(source[field + "_sha256"] == _text_hash(source[field]), field + " hash drift")
    _require(source["normalized_prompt_sha256"] == _text_hash(" ".join(source["prompt"].split())), "normalized prompt hash drift")
    _require(type(source["training_eligibility_reasons"]) is list and bool(source["training_eligibility_reasons"])
             and all(_nonempty(x) for x in source["training_eligibility_reasons"]), "source eligibility reason schema")
    _keys(source["license_provenance"], {"origin", "external_dataset_or_teacher", "redistribution_license", "code_notice_policy"}, "source provenance")
    _require(source["license_provenance"]["external_dataset_or_teacher"] is False
             and all(_nonempty(source["license_provenance"][key]) for key in ("origin", "redistribution_license", "code_notice_policy")), "source provenance drift")


def _short_source(source, panel):
    module = bridge if panel == "acquisition" else retention
    _keys(source, BRIDGE_FIELDS if panel == "acquisition" else RETENTION_FIELDS, "original short source")
    parent = module.allocate_parent(source["spec"], source["generator_seed"], source["candidate_index"])
    _common_short(source, module, parent)
    variant = source["variant_id"]
    if panel == "acquisition":
        _require(type(variant) is int and variant in (0, 1), "acquisition variant schema")
        _require(source["family"] == json.loads(parent.canonical_json)["family"]
                 and source["structure_id"] == parent.structure_id
                 and source["row_id"] == parent.parent_id + f"-v{variant}", "bridge lineage drift")
        _require(source["source_id"] == "original-rineya-e1-bridge"
                 and source["source_revision_verification"] == "CALLER_SUPPLIED_GIT_SHA; exact generator bytes bound by receipt"
                 and type(source["generation_attempts"]) is int and source["generation_attempts"] == 1
                 and type(source["prompt_characters"]) is int and source["prompt_characters"] == len(source["prompt"]), "bridge source metadata drift")
        expected_cost = {"model_calls": 0, "teacher_calls": 0, "model_generation_tokens": 0,
                         "verifier_calls": 2, "training_tokens": None, "tokenizer_tokens_status": "UNMEASURED"}
        valid_target = module.verify_answer(parent, source["target"])
    else:
        domain = json.loads(parent.canonical_json)["domain"]
        allowed = {"answerable", "missing_evidence"} if domain == "answerability_abstention" else {"standard"}
        _require(type(variant) is str and variant in allowed, "retention variant schema")
        _require(source["source_id"] == "original-rineya-retention" and source["domain"] == source["family"] == domain
                 and source["stratum"] == parent.stratum and source["row_id"] == parent.parent_id + "-" + variant, "retention lineage drift")
        _require(source["source_correct_membership"] is None and source["source_correct_enrollment_qualified"] is False
                 and source["source_correct_membership_policy"] == "freeze DEV base-correct subset; TRAIN is fixed and never filtered/replaced by baseline outcomes", "retention membership claim drift")
        expected_cost = {"model_calls": 0, "teacher_calls": 0, "model_generation_tokens": 0,
                         "verifier_calls": 2, "training_tokens": 0, "dataset_tokens": None}
        valid_target = module.verify_answer(parent, source["target"], variant)
    _require(_same(source["cost"], expected_cost), "source construction cost schema drift")
    _require(source["prompt"] == module.render_prompt(parent, variant), "source prompt differs from reconstructed parent")
    _require(valid_target, "stored target fails independent source verifier")
    answer = _parse_short(source["target"])
    _require(_answer_shape(answer, panel, source["family"]), "stored target shape invalid")
    _require(source["target"] == canonical({"answer": answer}).decode("utf-8"), "stored source target is not canonical JSON")
    return parent


def _ids(value, label):
    _require(type(value) is list and bool(value) and all(type(x) is int and 0 <= x < 151936 for x in value), label + " token schema")


def _long_source(source):
    _keys(source, LC_FIELDS, "original long-context source")
    _require(source["row_sha256"] == digest({k: v for k, v in source.items() if k != "row_sha256"}), "long-context source self-hash drift")
    fixed = {"schema_version": 1, "role": "DEV", "generator_version": "long-context-native-additive-note-dp-v1",
        "semantic_generator_version": long_context.VERSION, "verifier_version": long_context.VERIFIER_VERSION,
        "support_unit": "record_block", "generation_reserve_tokens": 24,
        "template_kwargs": {"add_generation_prompt": True, "enable_thinking": False},
        "individual_ablations_unknown": True, "policy_input_only": ["prompt", "input_ids"],
        "training_eligibility": False, "source_correct_membership": None,
        "native_qualification_inferred": False, "e1_launch_ready": False}
    for field, value in fixed.items():
        _require(_same(source[field], value), "long-context source metadata drift: " + field)
    _require(source["family"] in long_context.FAMILIES and source["evidence_position"] in long_context.POSITIONS
             and source["structural_cluster_id"] == source["family"], "long-context family/position drift")
    _require(type(source["parent_id"]) is str and re.fullmatch(r"lc-[0-9a-f]{64}", source["parent_id"]), "long-context parent identity schema")
    for field in ("canonical_parent_sha256", "split_sha256"):
        _hash(source[field], field)
    _require(7800 <= int(source["split_sha256"][:16], 16) % 10000 < 8600, "long-context split is not DEV")
    _require(type(source["answerable"]) is type(source["adversarial"]) is bool
             and type(source["missing_index"]) is int and 0 <= source["missing_index"] <= 2, "long-context factor schema")
    variant = ("answerable" if source["answerable"] else "unanswerable") + ("_adversarial" if source["adversarial"] else "_neutral")
    _require(source["variant_id"] == variant and type(source["prompt_tokens"]) is int
             and source["prompt_tokens"] in (2024, 4072)
             and source["row_id"] == f"{source['parent_id']}-{source['prompt_tokens']}-{variant}"
             and type(source["total_sequence_envelope"]) is int
             and source["total_sequence_envelope"] == source["prompt_tokens"] + 24, "long-context derivative identity drift")
    expected_modes = ["different_key", "obsolete_archive", "unauthorized_unsigned"] if source["adversarial"] else ["neutral"] * 3
    _require(source["distractor_modes"] == expected_modes, "long-context distractor mode drift")
    before = long_context.SYSTEM_PROMPT + "\nCONTEXT\n"
    _require(type(source["prompt"]) is str and source["prompt"].startswith(before), "long-context instruction drift")
    content = source["prompt"][len(before):]
    _require(content.count("QUERY\n") == 1, "long-context query boundary")
    body, query = content.split("QUERY\n")
    _require(body == source["context_body"] and query == source["query"], "visible source body/query drift")
    family_prefix = "Return the routed seal for " if source["family"] == long_context.FAMILIES[0] else "Return the latest authorized record for "
    _require(query.startswith(family_prefix), "visible query/family drift")
    for field in ("prompt", "rendered_prompt", "context_body", "target"):
        _require(source[field + "_sha256"] == _text_hash(source[field]), field + " hash drift")
    for field, count_field, hash_field in (("input_ids", "prompt_tokens", "input_token_ids_sha256"),
                                          ("target_token_ids", "target_tokens", "target_token_ids_sha256")):
        _ids(source[field], field)
        _require(type(source[count_field]) is int and len(source[field]) == source[count_field]
                 and digest(source[field]) == source[hash_field], "long-context token count/hash drift")
    _require(1 <= source["target_tokens"] <= 16 and not set(source["target_token_ids"]).intersection((151643,151644,151645)), "long-context target tokens")
    # These are bound historical measurements, not newly tokenized offsets.
    _require(type(source["native_offset_mapping"]) is list and len(source["native_offset_mapping"]) == source["prompt_tokens"], "long-context offset inventory")
    for pair in source["native_offset_mapping"]:
        _require(type(pair) is list and len(pair) == 2 and all(type(v) is int for v in pair)
                 and 0 <= pair[0] <= pair[1] <= len(source["rendered_prompt"]), "long-context offset schema")
    for kind, ceiling in (("token", source["prompt_tokens"]), ("char", len(source["rendered_prompt"]))):
        start, end = source[f"body_whole_prompt_{kind}_start"], source[f"body_whole_prompt_{kind}_end"]
        _require(type(start) is int and type(end) is int and 0 <= start < end <= ceiling, "long-context body bounds")
    _require(type(source["body_tokens"]) is int and source["body_tokens"] == source["body_whole_prompt_token_end"] - source["body_whole_prompt_token_start"], "long-context body token span")
    _require(source["rendered_prompt"][source["body_whole_prompt_char_start"]:source["body_whole_prompt_char_end"]] == body, "long-context rendered body span")
    for field in ("support_spans", "controlled_slots"):
        _require(type(source[field]) is list and len(source[field]) == 3
                 and [r.get("dependency_index") for r in source[field]] == [0,1,2], "long-context support inventory")
    regions = ("early", "middle", "late") if source["evidence_position"] == "distributed" else (source["evidence_position"],) * 3
    for index, (span, slot) in enumerate(zip(source["support_spans"], source["controlled_slots"])):
        present = source["answerable"] or index != source["missing_index"]
        _require(span.get("region") == regions[index] and span.get("required_fact_present") is present
                 and slot.get("present") is present, "long-context support presence/placement drift")
    solved = long_context.solve_visible(query, body)
    _require(source["target"] == source["visible_text_oracle"] == solved
             and long_context.parse_answer(source["target"]) == source["target"], "bound target disagrees with strict visible oracle")
    _require(source["answerable"] is (solved != "UNKNOWN"), "long-context answerability drift")
    _keys(source["semantic_verification"], {"visible_solution", "single_fact_ablations", "independent_oracle_agrees"}, "semantic verification")
    _require(source["semantic_verification"]["visible_solution"] == solved
             and source["semantic_verification"]["single_fact_ablations"] == ["UNKNOWN"] * 3
             and source["semantic_verification"]["independent_oracle_agrees"] is True, "semantic verification drift")
    return solved


def _language_source(source):
    """Identity and structure only; never judge the illustrative language reply."""
    _keys(source, LANGUAGE_FIELDS, "language source")
    _require(source["row_sha256"] == digest({k:v for k,v in source.items() if k != "row_sha256"}), "language row self-hash drift")
    _require(source["world_spec_sha256"] == digest({k:source[k] for k in WORLD_FIELDS})
             and source["payload_world_sha256"] == digest({k:source[k] for k in LANGUAGE_PAYLOAD_FIELDS}), "language source hash chain drift")
    _require(source["role"] == "DEV" and source["stratum"] in LANGUAGE_STRATA, "language DEV-only scope")
    match = re.fullmatch(r"LDEV-" + re.escape(source["stratum"]) + r"-(0[1-8])", source["parent_id"])
    _require(match is not None, "language parent identity")
    form = "single" if int(match[1]) <= 4 else "multi"
    author = "/root/baseline_enrollment" if LANGUAGE_STRATA.index(source["stratum"]) < 4 else "/root/e1_review"
    part = "A" if author.endswith("baseline_enrollment") else "B"
    _require(source["conversation_form"] == form and source["author_id"] == author
             and source["authorship"] == "assistant_source_candidate", "language author/form drift")
    for kind in ("spec", "payload"):
        _hash(source[kind + "_file_sha256"], kind + " file")
        label = "SPECS" if kind == "spec" else "PAYLOADS"
        _require(source[kind + "_source_path"] == f"sources/language_dev_cal_v1/{label}_{part}.json", "language source label drift")
    semantic = source["semantic_world"]
    lists = {"actors", "facts", "timeline", "constraints", "potential_confusions", "exclusion_tags"}
    _keys(semantic, lists | {"task_intent", "decisive_structure"}, "language semantic world")
    for field in lists:
        _require(type(semantic[field]) is list and bool(semantic[field]) and all(_nonempty(x) for x in semantic[field]), "language semantic list")
    _require(all(_nonempty(semantic[k]) for k in ("task_intent", "decisive_structure")), "language semantic text")
    messages = source["messages"]
    _require(type(messages) is list and len(messages) in ((1,) if form == "single" else (3,5,7)), "language conversation length")
    for index, message in enumerate(messages):
        _keys(message, {"role", "content"}, "language message")
        _require(message["role"] == ("assistant" if index % 2 else "user") and _nonempty(message["content"]), "language conversation boundary")
    contract = source["semantic_contract"]
    _keys(contract, {"must_include", "must_avoid", "acceptable_variation", "dimensions"}, "language contract")
    for field in ("must_include", "must_avoid", "acceptable_variation"):
        _require(type(contract[field]) is list and bool(contract[field]) and all(_nonempty(x) for x in contract[field]), "language contract list")
    _keys(contract["dimensions"], {"adequacy", "relevance", "fluency", "role_consistency"}, "language dimensions")
    _require(all(_nonempty(v) for v in contract["dimensions"].values()) and _nonempty(source["dev_exemplar"])
             and source["cal_controls"] == [], "language DEV source schema")


def _validated(row):
    _verify_dependencies()
    _finite(row)
    _keys(row, EVALUATOR_FIELDS, "evaluator row")
    _require(row["role"] == "DEV" and row["panel"] in PANELS, "evaluator DEV/panel scope")
    for field in ("row_id", "parent_id", "family", "stratum", "variant"):
        _require(_nonempty(row[field]), "evaluator identity: " + field)
    _hash(row["prompt_token_hash"], "prompt token hash")
    payload = row["scoring_payload"]
    _keys(payload, PAYLOAD_FIELDS, "scoring payload")
    _require(type(payload["schema_version"]) is int and payload["schema_version"] == 1
             and payload["panel"] == row["panel"], "scoring payload version/panel drift")
    source = payload["source_row"]
    _require(type(source) is dict and payload["source_row_sha256"] == digest(source), "canonical source row hash drift")
    _require(source.get("role") == "DEV", "source DEV-only scope")
    panel, parent, target = row["panel"], None, None
    if panel in PANELS[:2]:
        parent = _short_source(source, panel)
        mapping = {"row_id":source["row_id"], "parent_id":source["parent_id"], "family":source["family"],
                   "stratum":source["structure_id"] if panel == "acquisition" else source["stratum"], "variant":str(source["variant_id"])}
    elif panel == "long_context":
        target = _long_source(source)
        mapping = {"row_id":source["row_id"], "parent_id":source["parent_id"], "family":source["family"],
                   "stratum":source["evidence_position"], "variant":f"{source['prompt_tokens']}-{source['variant_id']}",
                   "prompt_token_hash":source["input_token_ids_sha256"]}
    else:
        _language_source(source)
        mapping = {"row_id":source["parent_id"] + "-dev", "parent_id":source["parent_id"], "family":"general_language",
                   "stratum":source["stratum"], "variant":"primary"}
    for field, value in mapping.items():
        _require(_same(row[field], value), "evaluator/source mapping drift: " + field)
    return source, parent, target


def validate_evaluator_row(row):
    """Validate source structure/identity only; None on success, otherwise raise."""
    _validated(row)


def _parse_short(response):
    if type(response) is not str or len(response) > 4096:
        raise ValueError("short response must be a JSON string of at most 4096 characters")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate answer key")
            out[key] = value
        return out
    def bad(value):
        raise ValueError("nonfinite response JSON: " + value)
    parsed = json.loads(response, object_pairs_hook=unique, parse_constant=bad)
    _finite(parsed)
    if type(parsed) is not dict or set(parsed) != {"answer"}:
        raise ValueError("exact answer envelope required")
    return parsed["answer"]


def _answer_shape(answer, panel, family):
    if panel == "acquisition":
        if family == "bounded_integer_arithmetic":
            return type(answer) is int
        if family == "symbolic_variable_updates":
            return type(answer) is list and len(answer) == 3 and all(type(v) is int for v in answer)
        if family == "finite_set_queries":
            return type(answer) is list and all(type(v) is int for v in answer) and answer == sorted(set(answer))
        return type(answer) is str
    if family == "instruction_following":
        return type(answer) is str
    if family == "answerability_abstention" and answer is None:
        return True
    return type(answer) is list and len(answer) == 2 and all(type(v) is str for v in answer)


def score_response(row, raw_response):
    source, parent, target = _validated(row)  # Never put host validation inside the response-error catch.
    panel = row["panel"]
    if panel == "language":
        raise UnqualifiedLanguageGraderError("language scoring blocked: no qualified four-dimension grader; exemplar matching is forbidden")
    if panel == "long_context":
        parsed = long_context.parse_answer(raw_response)
        return {"correct": bool(parsed is not None and long_context.score_answer(raw_response, target)), "malformed": parsed is None}
    try:
        answer = _parse_short(raw_response)
        well_shaped = _answer_shape(answer, panel, source["family"])
    except (ValueError, TypeError, RecursionError, OverflowError):
        return {"correct": False, "malformed": True}
    if not well_shaped:
        return {"correct": False, "malformed": True}
    correct = bridge.verify_answer(parent, raw_response) if panel == "acquisition" else retention.verify_answer(parent, raw_response, source["variant_id"])
    return {"correct": bool(correct), "malformed": False}
