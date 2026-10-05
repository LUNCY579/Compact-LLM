"""Additive SYNTH evidence packet: standard library, always BLOCKED.

Only an explicit allowlist of public source, TRAIN, and open DEV metadata is
read. References inside those documents are recorded and never dereferenced.
Hash equality is byte evidence; it does not establish native readiness, reviewer
authority, launch permission, a successful experiment, or model quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

VERSION = "synth-quality-readiness-supplement-1"
PACKET_ID = "SYNTH-QUALITY-READINESS-SUPPLEMENT-20261005"
PACKAGE = "experiments/rineya_synth_centralized_reasoning_v1"
BUNDLE = PACKAGE + "/data/crossed_source2_v1_20261003"
MAX_BYTES = 16 * 1024 * 1024
SOURCE_ASSET_PINS = {
    "weight_sha256": "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7",
    "config_sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "chat_template_sha256_recorded": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
}
PERMISSIONS = {name: False for name in (
    "native_diagnostic", "native_inference", "training", "protected_bank_access",
    "paid_services", "model_upload", "merge", "confirmation", "serving", "promotion")}
SOURCE_FILES = (
    "AGENTS.md", "RINEYA_SYNTH_CENTRALIZED_REASONING.md",
    *(PACKAGE + "/" + name for name in (
        "AGENTS.md", "README.md", "STATUS.md", "EXPERIMENT_PLAN.md", "CPU_DATA_MILESTONE.md",
        "PILOT_DECISIONS.json", "pilot_trainable_inventory.json", "study_schedule.py",
        "STUDY_DEV_REGISTRATION_DRAFT.json", "score_import.py", "DEV_STATISTICS_POLICY.json",
        "DEV_FINISH_POLICY_001.json", "EVALUATION_PANEL_POLICY.json", "EVALUATION_PANEL_POLICY_002.json",
        "task_compiler.py", "rs_contract.py", "rs_solver.py", "rs_verifier.py", "balanced_corpus.py",
        "TASK_RETENTION_GUARD_POLICY.json", "LONG_CONTEXT_GUARD_POLICY.json",
        "NATIVE_QUALIFICATION_PROPOSAL.json", "PREFIX_INPUT_DIAGNOSTIC_001.md")),
)
BUNDLE_FILES = tuple(BUNDLE + "/" + name for name in (
    "preparation_receipt.json", "plan.json", "enrollment.json", "candidate_records.jsonl",
    "pilot_decisions.json", "matching_policy.json", "schedule_result.json", "training_occurrences.json",
    "student_pool_M1.jsonl", "student_pool_M2.jsonl", "token_pool_M1.jsonl", "token_pool_M2.jsonl",
    "source_policy_002.json", "corpus_amendment_002.md"))
DEV_FILES = tuple(PACKAGE + "/evidence/fresh_dev_compact002_20261003/" + name
                  for name in ("panel.json", "policy/prompts.json", "prompt_token_inputs_cpu.json"))
DEV_FILE_HASHES = dict(zip(DEV_FILES, (
    "495cadc57d97d232acc0f093257b1dc0c4ae29897d17bd319e158e0cc7af9acb",
    "a7baf928b38ebccd86173340e27d3cae8467ef71e49df71c48785b2e682a2957",
    "4664c1770a88a6345c6bfa02cad3c7bdf667c3037f8811364049b503a6b90a4f")))
ALL_FILES = SOURCE_FILES + BUNDLE_FILES + DEV_FILES
SLOTS = {
    "native_failure_diagnosis": "Owner finishes CUDA failure diagnosis, retaining every failed attempt and distinct later supervisor abort.",
    "current_numerical_qualification": "Scope-matching current source, exact model/runtime, numerical checks and fresh host admission; old short mechanics cannot certify study fit.",
    "sustained_fit_save_reload_ownership": "Actual sustained fit, physical restore, owned worker/supervisor/auditor and terminal cleanup evidence.",
    "complete_open_dev_m0": "Six complete 64-row successful independently audited M0 batches with externally selected config/audit hashes; 256/384 unscored rows do not qualify.",
    "native_scoring_and_candidate_import": "Qualified whole-response cap/EOS scoring and M1/M2 collector/audit schema; current score_import.py admits M0 provenance only and no native grader.",
    "short_retention_enrollment": "Independent disjoint source-correct arithmetic, retrieval, abstention and dialogue guard membership frozen before candidate outputs.",
    "language_authority_and_reference_vectors": "Qualified language scorer authority, reference vectors and source-correct enrollment.",
    "long_context_qualification_and_enrollment": "Separate genuine native long-context qualification and disjoint source-correct enrollment, with missing coverage undefined.",
    "transfer_family_and_exposure_freeze": "Exact family separation, fixture derivatives, consumed-prompt occurrences, independent public semantics/alternatives and freeze before treatment outcomes.",
    "prospective_quality_policy_review": "Exact supplemental 3-point/zero-new-loss policy hash and owner review before outcomes; retain historical 5-point plus CI policy separately.",
    "synth_owner_review": "Owner reviews this additive packet and selects prospective anchors; no self-declared approval flags.",
    "study_completion_lineage": "Actual three STUDY chunks per trained arm, final step27/540 sequences; all112 factors, Adam moments/steps, RNG, cursor, global LR, costs and all checkpoints. FULL27 qualification artifacts cannot become study weights.",
    "original_peft_and_native_export_parity": "Preserved original exports/completion receipts and actual native parity at original steps9 and27, retaining every checkpoint1..27; metadata is insufficient.",
    "replication_and_fresh_confirmation": "Eligible recipe prospectively replicated on two additional independent training draws before separately registered fresh confirmation.",
    "separate_launch_serving_promotion_decisions": "Fresh scope-matching human native authorization and separate study launch, confirmation, serving and promotion decisions; nothing launches automatically.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_hash(value):
    return (type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            and value not in ("0" * 64, hashlib.sha256(b"").hexdigest()))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seal(value):
    require(type(value) is dict and "sha256" not in value, "unsealed body required")
    return {**value, "sha256": digest(value)}


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonfinite JSON constant")


def parse_json(raw):
    require(type(raw) is bytes and len(raw) <= MAX_BYTES, "bounded JSON bytes required")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("invalid bounded UTF-8 JSON") from error
    _bounded_tree(value)
    return value


def _bounded_tree(value, depth=0):
    require(depth <= 64, "JSON nesting bound")
    if type(value) is float:
        require(value not in (float("inf"), float("-inf")) and value == value, "nonfinite JSON number")
    elif type(value) is dict:
        for item in value.values():
            _bounded_tree(item, depth + 1)
    elif type(value) is list:
        for item in value:
            _bounded_tree(item, depth + 1)


def _no_link(path):
    """Reject POSIX links and Windows junction/reparse ancestors before resolving."""
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        info = ancestor.lstat()
        require(not stat.S_ISLNK(info.st_mode)
                and not getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
                "symlink/reparse path forbidden")
    return path


def safe_path(root, relative, *, must_exist=True):
    require(type(relative) is str and 0 < len(relative) <= 1024 and "\\" not in relative
            and ":" not in relative and "\x00" not in relative, "relative POSIX path required")
    parts = relative.split("/")
    require(all(part not in ("", ".", "..") for part in parts) and not PurePosixPath(relative).is_absolute(),
            "path traversal forbidden")
    root = _no_link(root)
    require(root.is_dir(), "regular root directory required")
    candidate = root.joinpath(*parts)
    if must_exist:
        _no_link(candidate)
        require(candidate.is_file(), "regular selected file required")
    else:
        _no_link(candidate.parent)
        require(not candidate.exists() and not candidate.is_symlink(), "new output required")
    require(candidate.resolve(strict=must_exist).is_relative_to(root.resolve()), "path escaped root")
    return candidate


def read_selected(root, relative, *, expected_sha256=None):
    path = safe_path(root, relative)
    require(path.suffix in (".py", ".md", ".json", ".jsonl"), "selected text/metadata only")
    if expected_sha256 is not None:
        require(is_hash(expected_sha256), "valid expected file hash required")
    before = path.stat()
    require(0 < before.st_size <= MAX_BYTES, "selected file must be nonempty and bounded")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    after = path.stat()
    require(before.st_size == after.st_size == len(raw) and before.st_mtime_ns == after.st_mtime_ns
            and before.st_ino == after.st_ino, "selected file changed during read")
    sha = hashlib.sha256(raw).hexdigest()
    require(expected_sha256 is None or sha == expected_sha256, "selected file byte drift: " + relative)
    return raw, {"relative_path": relative, "bytes": len(raw), "sha256": sha}


def jsonl(raw):
    return [parse_json(line) for line in raw.splitlines() if line.strip()]


def learning_rates_exact():
    """No study import: reconstruct the published pure Fraction definition."""
    peak = Fraction(1, 10000)
    rates = [peak / 2, peak] + [peak * Fraction(27 - step, 25) for step in range(3, 28)]
    return [{"update": step, "numerator": rate.numerator, "denominator": rate.denominator,
             "float_value": float(rate)} for step, rate in enumerate(rates, 1)]


def validate_recipe(decisions, draft):
    require(type(decisions) is dict and type(draft) is dict, "recipe documents required")
    student, adaptation, optimizer = decisions["student"], decisions["adaptation"], decisions["optimizer"]
    require(student["name"] == "Rineya-Titan-Experimental-1.0.0-devfreeze"
            and type(student["base_parameters"]) is int and student["base_parameters"] == 596049920,
            "exact Titan source required")
    for key, expected in SOURCE_ASSET_PINS.items():
        require(student[key] == expected, "exact source asset pin required: " + key)
    fixed = {"method": "qv_lora", "rank": 8, "alpha": 16, "base_frozen": True,
             "layers": list(range(28)), "projections": ["q_proj", "v_proj"], "dropout": 0.0,
             "target_format": "final_answer_only", "common_warmstart": False,
             "context_cap_tokens": 512, "microbatch_sequences": 1, "gradient_accumulation_sequences": 20,
             "packing": False, "padding": False, "trainable_parameters": 1146880, "trainable_tensors": 112,
             "loss": "completion_plus_EOS_mean_per_sequence_then_mean_over_accumulation"}
    require(all(adaptation.get(key) == value and type(adaptation.get(key)) is type(value) for key, value in fixed.items()),
            "frozen adaptation recipe drift")
    expected_optimizer = {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
                          "clip_norm": 1.0, "optimizer_state_dtype": "float32", "peak_lr": 1e-4,
                          "schedule": "5_percent_warmup_then_linear_decay_indexed_by_frozen_matched_optimizer_updates"}
    require(all(optimizer.get(key) == value for key, value in expected_optimizer.items()), "frozen AdamW recipe drift")
    require(decisions["seeds"]["first_development_seed"] == draft["seed"] == 101
            and decisions["selection"]["checkpoint"] == "final_only"
            and adaptation["replay"].startswith("none_in_first_atomic_dose_screen"), "seed/endpoint/replay recipe drift")
    require(draft["student"] == {key: value for key, value in student.items()
                                   if key not in ("native_load_performed", "static_asset_hashes_refreshed")}
            and draft["adaptation"] == adaptation, "draft decision binding drift")
    require(draft["optimizer"]["learning_rates_by_1based_update"] == [item["float_value"] for item in learning_rates_exact()]
            and draft["optimizer"]["warmup_updates"] == 2, "exact27 LR vector drift")
    require(all(draft["optimizer"].get(key) == value for key, value in expected_optimizer.items()), "draft optimizer drift")
    dose = draft["dose"]
    expected_dose = {"full_updates_per_adapted_arm": 27, "sequences_per_update": 20, "whole_sequences_per_arm": 540,
                     "M1_actual_input_tokens": 130562, "M2_actual_input_tokens": 130574,
                     "padding": False, "packing": False, "truncation": False, "dropping": False,
                     "replay_added": False, "warmstart": False, "RS_training": False}
    require(all(dose.get(key) == value for key, value in expected_dose.items()), "frozen dose recipe drift")
    require(draft["resume_contract"]["chunks_per_arm"] == 3
            and draft["resume_contract"]["full_updates_per_chunk"] == 9
            and draft["resume_contract"]["qualification_STATE1_warmstart_allowed"] is False,
            "three-chunk cold-start recipe drift")
    require(all(value is False for value in decisions["permissions"].values())
            and draft["native_launch_authorized"] is False, "historical execution gate changed")
    return {"student": {key: value for key, value in student.items()
                         if key not in ("native_load_performed", "static_asset_hashes_refreshed")},
            "adaptation": deepcopy(adaptation), "optimizer": deepcopy(expected_optimizer),
            "seed": 101, "primary_checkpoint": "final_only", "learning_rates": learning_rates_exact(),
            "warmup_updates": 2, "update27": "all20 sequences/backward/clip/AdamW moments consumed at LR0"}


def validate_quality_policy(policy):
    require(type(policy) is dict and policy.get("schema") == "synth-quality-engineering-screen-v1"
            and policy.get("policy_id") == "SYNTH-QUALITY-3PP-ZERO-DAMAGE-20261005"
            and policy.get("experiment") == "SYNTH-CRS-001"
            and policy.get("policy_status") == "PROSPECTIVE_ENGINEERING_SCREEN_NOT_NATIVE_APPROVAL",
            "distinct prospective quality policy required")
    require(all(type(policy.get(name)) is float and policy[name] == 0.03 for name in
                ("minimum_macro_atomic_gain", "minimum_rs_transfer_gain", "minimum_m2_advantage"))
            and policy.get("minimum_gain_fraction") == [3, 100]
            and policy.get("atomic_skill_weights") == {"R": [1, 2], "S": [1, 2]}, "exact3-point policy/weights required")
    require(policy.get("no_atomic_skill_decline") is True and type(policy.get("maximum_damaged_prior_correct_cases")) is int
            and policy["maximum_damaged_prior_correct_cases"] == 0 and policy.get("repairs_can_cancel_damage") is False
            and policy.get("required_retention_sources") == ["arithmetic", "retrieval", "abstention", "dialogue"],
            "zero-new-loss policy and complete independent guard classes required")
    require(policy.get("checkpoint_selection") == "final_only_27" and policy.get("both_candidates_must_complete") is True
            and policy.get("candidate_order") == ["M1", "M2"]
            and policy.get("replication") == {"additional_independent_training_draws": 2, "automatic_launch": False},
            "final endpoint/sequential full cohort/replication policy required")
    require(all(policy.get(name) is False for name in ("native_model_execution_authorized", "native_scoring_authorized",
                "training_authorized", "confirmation_authorized", "promotion_authorized", "long_context_qualified")),
            "policy must not claim native authorization/qualification")
    return policy["policy_id"]


def validate_token_row(row):
    for key in ("prompt_ids", "completion_ids", "input_ids", "labels", "attention_mask"):
        require(type(row.get(key)) is list and all(type(value) is int for value in row[key]), "typed token array: " + key)
    prompt, completion, ids = row["prompt_ids"], row["completion_ids"], row["input_ids"]
    require(prompt and completion and 1 < len(ids) <= 512
            and all(0 <= value < 151936 for value in prompt + completion), "bounded full token sequence")
    require(completion[-1] == 151645 and 151645 not in completion[:-1] and 151643 not in ids, "exact completion EOS; no padding")
    require(ids == prompt + completion and row["labels"] == [-100] * len(prompt) + completion
            and row["attention_mask"] == [1] * len(ids), "masks/EOS/causal concatenation drift")
    require(all(type(row.get(key)) is int and row[key] == expected for key, expected in
                (("prompt_tokens", len(prompt)), ("total_tokens", len(ids)), ("supervised_tokens", len(completion)))),
            "token accounting drift")


def audit_dose(schedule, occurrences, candidates, pools):
    require(schedule["status"] == "MATCHED_CPU_SCHEDULE" and schedule["optimizer_updates_per_arm"] == 27,
            "approved27-update schedule required")
    require(occurrences["row_ids"] == schedule["schedules"] and occurrences["schedule_sha256"] == schedule["sha256"],
            "occurrence schedule binding drift")
    records = {item["row_id"]: item for item in candidates}
    require(len(records) == len(candidates), "duplicate candidate row ID")
    summaries, parent_vectors, orders = {}, {}, {}
    for arm, expected in (("M1", (130562, 5614)), ("M2", (130574, 5615))):
        rows = pools[arm]
        pool = {row["row_id"]: row for row in rows}
        require(len(pool) == len(rows), "duplicate token row ID")
        for row in rows:
            validate_token_row(row)
            record = records[row["row_id"]]
            require(record["arm"] == arm and record["split"] == "train" and record["skill"] in ("R", "S")
                    and record["verification"]["accepted"] is True and record["verification"]["reason"] == "verified",
                    "verified atomic TRAIN rows only")
        ids = schedule["schedules"][arm]
        require(type(ids) is list and len(ids) == 540 and all(row_id in pool for row_id in ids), "whole540 occurrences required")
        consumed = [pool[row_id] for row_id in ids]
        input_count = sum(row["total_tokens"] for row in consumed)
        target_count = sum(row["supervised_tokens"] for row in consumed)
        require((input_count, target_count) == expected, "frozen corpus token dose drift")
        skills = Counter(records[row_id]["skill"] for row_id in ids)
        require(skills == {"R": 192, "S": 348}, "frozen atomic skill exposure drift")
        parents = Counter(records[row_id]["parent_id"] for row_id in ids)
        require(len(parents) == 20, "shared20 source-parent population")
        parent_vectors[arm] = dict(sorted(parents.items()))
        orders[arm] = [(records[row_id]["parent_id"], records[row_id]["skill"]) for row_id in ids]
        summaries[arm] = {"input_tokens": input_count, "supervised_tokens": target_count,
                          "sequences": 540, "updates": 27, "sequences_per_update": 20,
                          "R_sequences": skills["R"], "S_sequences": skills["S"],
                          "registered_candidate_rows": len(rows), "source_parent_exposures": parent_vectors[arm],
                          "occurrence_order_sha256": digest(ids), "chunks": [
                              {"chunk": chunk, "first_update": 1 + (chunk - 1) * 9, "last_update": chunk * 9,
                               "first_occurrence": 1 + (chunk - 1) * 180, "last_occurrence": chunk * 180,
                               "occurrence_order_sha256": digest(ids[(chunk - 1) * 180:chunk * 180])}
                              for chunk in range(1, 4)]}
    require(parent_vectors["M1"] == parent_vectors["M2"] and orders["M1"] == orders["M2"], "paired parent exposure/order drift")
    return summaries


def git_identity(root):
    root = _no_link(root)
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=15, check=False)
        require(result.returncode == 0, "git source identity unavailable")
        return result.stdout.strip()
    head, branch, status = git("rev-parse", "HEAD"), git("branch", "--show-current"), git("status", "--porcelain")
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None and branch, "committed branch identity required")
    require(status == "", "clean SYNTH source required; owner must finish tracked changes first")
    return {"head": head, "branch": branch, "clean": True}


def _future_refs(value):
    """Metadata only: these are never dereferenced or treated as authoritative."""
    require(type(value) is dict and set(value) <= set(SLOTS), "known unresolved slots only")
    result = {}
    for slot, refs in value.items():
        require(type(refs) is list and len(refs) <= 64, "bounded future metadata references")
        checked = []
        for ref in refs:
            require(type(ref) is dict and set(ref) == {"path", "sha256"}
                    and type(ref["path"]) is str and 0 < len(ref["path"]) <= 1024
                    and "\x00" not in ref["path"] and is_hash(ref["sha256"]), "valid future path/hash metadata required")
            checked.append(deepcopy(ref))
        result[slot] = checked
    return result


def build_packet(synth_root, external_root, policy_ref=None, *, supplementary_refs=None, runtime_refs=None):
    """Return an immutable-content prospective snapshot; no readiness graduation API.

    policy_ref is {relative_path, sha256} under external_root. runtime_refs is a
    list of the same references for explicitly supplied, copied runtime JSON
    metadata. Optional omissions remain explicit unresolved slots.
    """
    before = git_identity(synth_root)
    refs, raw = {}, {}
    for name in ALL_FILES:
        raw[name], refs[name] = read_selected(synth_root, name, expected_sha256=DEV_FILE_HASHES.get(name))
    decisions = parse_json(raw[PACKAGE + "/PILOT_DECISIONS.json"])
    draft = parse_json(raw[PACKAGE + "/STUDY_DEV_REGISTRATION_DRAFT.json"])
    recipe = validate_recipe(decisions, draft)
    require(decisions == parse_json(raw[BUNDLE + "/pilot_decisions.json"]), "root/corpus decisions drift")
    for name, pin in draft["source_and_decision_pins"].items():
        if name in refs:
            require(refs[name]["sha256"] == pin, "historical draft byte binding drift: " + name)
    dose = audit_dose(parse_json(raw[BUNDLE + "/schedule_result.json"]),
                      parse_json(raw[BUNDLE + "/training_occurrences.json"]),
                      jsonl(raw[BUNDLE + "/candidate_records.jsonl"]),
                      {arm: jsonl(raw[BUNDLE + "/token_pool_" + arm + ".jsonl"]) for arm in ("M1", "M2")})
    policy = None
    policy_id = "SYNTH-QUALITY-3PP-ZERO-DAMAGE-20261005"
    if policy_ref is not None:
        require(type(policy_ref) is dict and set(policy_ref) == {"relative_path", "sha256"}, "explicit policy path/hash required")
        policy_raw, policy = read_selected(external_root, policy_ref["relative_path"], expected_sha256=policy_ref["sha256"])
        policy_id = validate_quality_policy(parse_json(policy_raw))
    runtime = []
    require(runtime_refs is None or type(runtime_refs) is list and len(runtime_refs) <= 16, "bounded runtime metadata refs")
    for ref in runtime_refs or []:
        require(type(ref) is dict and set(ref) == {"relative_path", "sha256"}
                and ref["relative_path"].endswith(".json"), "explicit runtime JSON path/hash required")
        metadata, bound = read_selected(external_root, ref["relative_path"], expected_sha256=ref["sha256"])
        parse_json(metadata)
        runtime.append(bound)
    future = _future_refs(supplementary_refs or {})
    historical = parse_json(raw[PACKAGE + "/DEV_STATISTICS_POLICY.json"])
    require(historical["advancement"]["atomic_M2_minus_M1_minimum"] == 0.05
            and historical["advancement"]["atomic_lower_CI_strictly_positive"] is True,
            "historical5-point plus CI policy must remain unchanged")
    after = git_identity(synth_root)
    require(before == after, "SYNTH source changed while binding packet")
    for name, ref in refs.items():
        read_selected(synth_root, name, expected_sha256=ref["sha256"])
    packet = {
        "schema_version": 1, "id": PACKET_ID, "version": VERSION,
        "status": "BLOCKED_PROSPECTIVE_READINESS_AND_MEASUREMENT_PACKET",
        "experiment": "SYNTH-CRS-001", "artifact_scope": "additive_external_stdlib_metadata_snapshot",
        "source": {**before, "root": str(Path(synth_root).resolve()), "files": refs},
        "external_root": str(Path(external_root).resolve()),
        "frozen_recipe": recipe, "verified_token_accounting": dose,
        "source_asset_pins_are_declared_metadata_not_current_model_byte_qualification": True,
        "runtime_metadata_refs": runtime, "runtime_status": "UNQUALIFIED_METADATA_ONLY",
        "historical_policy": {"id": historical["id"], "file": refs[PACKAGE + "/DEV_STATISTICS_POLICY.json"],
                              "status": "UNCHANGED_HISTORICAL_5_POINT_PLUS_CI_EXPLORATORY_POLICY"},
        "prospective_quality_policy": {"supplement_id": policy_id,
                                       "file": policy, "status": "HASH_BOUND_PENDING_OWNER_REVIEW" if policy else "UNRESOLVED_MISSING_POLICY_HASH",
                                       "replaces_historical_policy": False, "claims_statistical_power": False},
        "measurement_freeze": {"open_DEV_rows": 384, "components_per_skill": {"R": 32, "S": 32, "RS": 32},
                               "conditions": [0, 1], "renderings": ["familiar", "heldout_ledger"],
                               "macro_weights": {"R": {"numerator": 1, "denominator": 2}, "S": {"numerator": 1, "denominator": 2}},
                               "RS_weight": {"numerator": 1, "denominator": 1}, "primary_checkpoint": "final_only",
                               "M0_audited_raw_rows_at_owner_handoff": 256, "M0_scored_rows": 0,
                               "complete_baseline_required": {"audits": 6, "rows_per_audit": 64, "rows": 384,
                                                              "externally_selected_config_and_audit_hashes": True},
                               "baseline_complete": False, "retention_membership": None,
                               "missing_source_correct_coverage": "UNDEFINED",
                               "long_context_qualification": "SEPARATE_UNRESOLVED",
                               "report_separately": ["formatting", "semantic_correctness", "abstention", "capped_outputs", "uncertainty", "new_correct", "newly_failed_guards", "repairs"],
                               "exclude_from_transfer": ["training_families", "fixture_derivatives", "previously_consumed_prompts"],
                               "UNKNOWN_transitions_harder_conditions_alternate_renderings": "PRESERVE_FROZEN_OPEN_DEV_DESIGN"},
        "curriculum_limits": {"input_token_residual": 12, "joint_difficulty_equivalence": False,
                             "S_cardinality_updates_with_different_distributions": 27,
                             "R_first_depth_updates_with_different_distributions": 26,
                             "whole_run_marginals_match": True,
                             "remaining_differences": ["joint_item_count_cardinality_solution_count", "requested_count_maximum_depth_total_hop", "target_range", "per_update_ordering"],
                             "interpretation": "Comparison of two frozen curricula; token matching does not isolate diversity from every difficulty or ordering effect."},
        "producer_contract": {"retain": ["raw_inputs", "rendered_inputs", "causal_input_ids_labels_masks", "source", "filter", "consume", "unused", "replay_exposures", "input_and_target_token_counts", "consumed_prompt_occurrences", "original_exports", "completion_receipts", "failed_attempts", "hash_bound_checkpoints"],
                              "retain_all_original_checkpoint_steps": list(range(1, 28)),
                              "GD_admission_original_steps": [9, 27], "actual_native_export_parity_required": True,
                              "metadata_substitutes_for_parity": False,
                              "qualification_full27_weights_are_study_weights": False,
                              "interrupted_or_shortened_runs_are_success": False},
        "successor_registration": {"automatic_launch": False, "branches": {
            "acquisition_with_forgetting": "matched preservation replay successor",
            "poor_reference_resolution": "final-answer versus independently verified concise intermediate-state supervision",
            "useful_sft_preserved_skills": "prepared RDD contrast only after teacher qualification",
            "familiar_template_gains_only": "condition coverage and transfer testing before dose increase"},
            "larger_models_architecture_sweeps_RL_restart_full_distillation_factorial": "OUTSIDE_THIS_CYCLE"},
        "unresolved_slots": {slot: {"status": "UNRESOLVED", "requirement": description,
                                    "candidate_metadata_refs": future.get(slot, []),
                                    "candidate_refs_qualify_gate": False} for slot, description in SLOTS.items()},
        "permissions": deepcopy(PERMISSIONS), "execution_permit": False, "native_ready": False,
        "quality_gain_established": False, "confirmation_eligible": False,
        "retention_certified": False, "owner_reviewed": False,
    }
    return seal(packet)


def write_packet(path, synth_root, external_root, policy_ref=None, **kwargs):
    output = Path(path)
    root = _no_link(external_root).resolve()
    require(output.absolute().is_relative_to(root), "output must stay in external packet workspace")
    output = safe_path(root, output.absolute().relative_to(root).as_posix(), must_exist=False)
    packet = build_packet(synth_root, root, policy_ref, **kwargs)
    with output.open("xb") as stream:
        stream.write(json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    raw, ref = read_selected(root, output.relative_to(root).as_posix())
    require(parse_json(raw) == packet, "written packet byte/content mismatch")
    return {"packet": packet, "file": ref, "status": packet["status"]}


def check_packet(path, expected_sha256, synth_root, external_root):
    """Verify externally pinned packet bytes and reconstruct all selected bindings."""
    root = _no_link(external_root).resolve()
    output = Path(path).absolute()
    require(output.is_relative_to(root), "packet outside external workspace")
    raw, ref = read_selected(root, output.relative_to(root).as_posix(), expected_sha256=expected_sha256)
    value = parse_json(raw)
    require(type(value) is dict and is_hash(value.get("sha256"))
            and value["sha256"] == digest({key: item for key, item in value.items() if key != "sha256"}), "packet canonical body checksum")
    require(value.get("status") == "BLOCKED_PROSPECTIVE_READINESS_AND_MEASUREMENT_PACKET"
            and value.get("permissions") == PERMISSIONS and value.get("execution_permit") is False
            and value.get("native_ready") is False, "packet cannot grant execution/readiness")
    policy = value["prospective_quality_policy"]["file"]
    policy_ref = {key: policy[key] for key in ("relative_path", "sha256")} if policy else None
    runtime_refs = [{key: item[key] for key in ("relative_path", "sha256")} for item in value["runtime_metadata_refs"]]
    future = {slot: item["candidate_metadata_refs"] for slot, item in value["unresolved_slots"].items()
              if item["candidate_metadata_refs"]}
    expected = build_packet(synth_root, root, policy_ref, supplementary_refs=future, runtime_refs=runtime_refs)
    require(value == expected, "packet/source/recipe/policy/measurement snapshot drift")
    return {"status": "PASS_BLOCKED_METADATA_PACKET_BYTE_BINDINGS", "file": ref,
            "unresolved_slots": len(SLOTS), "execution_permit": False, "native_ready": False,
            "permissions": deepcopy(PERMISSIONS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "check"))
    parser.add_argument("--synth-root", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-path")
    parser.add_argument("--policy-sha256")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    require((args.policy_path is None) == (args.policy_sha256 is None), "policy path/hash must be paired")
    if args.operation == "build":
        policy = {"relative_path": args.policy_path, "sha256": args.policy_sha256} if args.policy_path else None
        result = write_packet(args.output, args.synth_root, args.external_root, policy)
        result = {key: value for key, value in result.items() if key != "packet"}
    else:
        result = check_packet(args.output, args.expected_sha256, args.synth_root, args.external_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
