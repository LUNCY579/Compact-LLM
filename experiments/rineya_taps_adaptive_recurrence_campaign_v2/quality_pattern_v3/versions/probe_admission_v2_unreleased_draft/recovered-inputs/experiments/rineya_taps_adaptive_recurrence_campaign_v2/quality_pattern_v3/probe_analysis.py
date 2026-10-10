"""Fail-closed, standard-library passive-probe reader. Never imports a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

STUDY = Path(__file__).resolve().parent
CAMPAIGN = STUDY.parent
ROOT = CAMPAIGN.parents[1]
LANE = "GPU_FP32_PASSIVE_INPUT_FEATURE_PROBE"
STUDY_ID = "V3-PASSIVE-PROBE-001"
SCHEMA = "PASSIVE_INPUT_FEATURES_V1"
STATE = "b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551"
CUSTOM = "b77b7d709fac3690d86f5978d46de409fd7ad3548c66a9d400188c225d99fa3d"
QUAL_MODES = ("INTACT", "UNOBSERVED_K0", "OBSERVED_K0", "LEGACY_UNIT",
              "UNOBSERVED_UNIT", "OBSERVED_UNIT", "RESTORED_INTACT")
CONTRACT = {"cases": 192, "parents": 96, "task_full_forwards": 384,
    "qualification_smoke_full_attempts": 56, "qualification_max_full_attempts": 9,
    "qualification_full_attempts": 65, "qualification_completed_forwards": 64,
    "qualification_API_attempts": 67, "cache_rejections": 2, "expected_partial": 1,
    "model_API_attempts": 451, "full_forward_attempts": 449,
    "completed_forwards": 448, "state_adjoints": 0, "task_readouts": 384,
    "task_feature_events": 576, "maximum_prefix_tokens": 357,
    "qualification_completed_prefix_positions": 4921,
    "task_completed_prefix_positions": 49216, "completed_prefix_positions": 54137}
COUNTS = {k: CONTRACT[k] for k in ("model_API_attempts", "full_forward_attempts",
    "completed_forwards", "expected_partial", "cache_rejections", "completed_prefix_positions", "state_adjoints")}
QUAL_COUNTS = {"model_API_attempts": 67, "full_forward_attempts": 65,
    "completed_forwards": 64, "expected_partial": 1, "cache_rejections": 2,
    "completed_prefix_positions": 4921, "state_adjoints": 0}
EXPORT_PINS = {"DEVELOPMENT_BANK.json": "681a3e77e3d797e2faebcff3971dbb365a4479ebdac017bf3dcfcb48081ba03f",
    "DEVELOPMENT_TOKEN_BINDING.json": "c34948026ebaf332111ce635a473b1489ef0d2f3932100f9e0d3cdd6a08ce6d2",
    "MANIFEST.json": "f35dc01c54154c51fa6284807245767f01282128c23a09307a2485483951530b"}
OFFLINE_PINS = {"runs/native-bulk-restart-001/result/outcomes.jsonl": "858ba6c1b5105b460e600c93872575c78e2c3e43545538847e3d9eba30e01cd8",
    "runs/bulk-bank-apparatus-001/bank.json": "862a63648caf7efb3d287c11bc5caaefc884e0ec6cb3130ce8f196526efc40cd",
    "analyses/cpu-causal-replay-003-fp32-reviewed-001/REPORT.json": "ae15d0c9d425938d13ea981db4406269ae2ada5a6444d778e1a8fe0f84671911"}
NATIVE_PINS = {"probe_native.py": "c7a433c26c9d7b5d83f54e971fe5d656d8f6edbf047e5442c092b310dd6be914",
    "observer.py": "da6a75376d6984a54396dc91499fa8e797ad3a66d8ff66512b29123b449bdae0"}
RIVAL_SET = "ALL_HEAD_ROWS_EXCEPT_UNKNOWN_INCLUDING_EOS_AND_SPECIAL_UNFILTERED"
DENOM_POLICY = "NULL_IF_ZERO_NO_EPSILON_SUBSTITUTION"
BANK_KEYS = {"case_id", "split", "raw_prompt", "raw_prompt_sha256"}
TOKEN_KEYS = BANK_KEYS | {"rendered_prompt", "rendered_prompt_sha256", "input_token_ids", "input_tokens"}
POP_KEYS = {"case_id", "split", "prompt", "prompt_sha256", "rendered_prompt",
            "rendered_prompt_sha256", "input_token_ids", "prefix_ids_sha256", "prefix_tokens"}
TASK_CONTEXT = {"case_id", "split", "phase", "mode", "prefix_ids_sha256", "prompt_sha256", "rendered_prompt_sha256"}
FEATURE_BODY = {"schema", "sample_id", "forward_index", "iteration", "prefix_tokens", "last_token_index",
    "stage", "available_before_extra", "extra_block_already_computed", "prefix_kind", "preceding_gold_tokens",
    "target_independent", "teacher_conditioned", "alpha", "statistics", "task_measurement"}
STAT_KEYS = {"old_l2", "new_l2", "delta_l2", "old_rms", "new_rms", "normal_rms", "delta_rms",
    "state_dot", "state_cosine_denominator", "state_cosine", "delta_over_old_l2_denominator", "delta_over_old_l2",
    "zero_old_norm", "zero_new_norm", "zero_delta_norm", "denominator_policy", "previous_delta_l2",
    "delta_dot_previous", "delta_cosine_denominator", "delta_cosine_previous", "delta_over_previous_l2", "finite"}
READ_BODY = {"schema", "vocabulary_rows", "UNKNOWN_token_id", "UNKNOWN_logit", "rival_logit", "rival_token_id",
    "UNKNOWN_minus_rival_margin", "rival_set", "top1_token_id", "top1_probability", "top2_token_ids", "top2_logits",
    "top2_margin", "entropy_nats", "logsumexp", "logits_changed", "target_independent", "finite", "prefix_kind",
    "preceding_gold_tokens", "teacher_conditioned", "observer_seconds", "readout_observer_seconds",
    "extra_model_calls_for_statistics"}
FORBIDDEN = {"gold", "gold_answer", "gold_token", "gold_tokens", "target", "targets", "score", "correct",
    "category", "family", "parent_id", "answerability", "source_status", "teacher_tokens", "loss", "derivative",
    "metadata", "scorer", "answer_kind", "difficulty_stratum", "generator_family", "generator_variant", "query_role", "variant_id"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def clean(value):
    if isinstance(value, dict):
        return not FORBIDDEN.intersection(value) and all(clean(v) for v in value.values())
    if isinstance(value, list):
        return all(clean(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate_export(bank, tokens):
    """Same exact prompt-only projection as the native reader; no tokenizer import."""
    errors, result = [], []
    if not isinstance(bank, list) or not isinstance(tokens, list):
        return ["export arrays missing"], []
    if any(not isinstance(r, dict) or set(r) != BANK_KEYS for r in bank) or any(
            not isinstance(r, dict) or set(r) != TOKEN_KEYS for r in tokens):
        return ["export exact prompt-only schema/metadata firewall differs"], []
    by = {r["case_id"]: r for r in tokens}
    if len(bank) != 192 or len(tokens) != 192 or len(by) != 192 or len({r["case_id"] for r in bank}) != 192:
        errors.append("export complete192 unique membership differs")
    if set(by) != {r["case_id"] for r in bank}:
        errors.append("export bank/token membership differs")
    for row in bank:
        token = by.get(row["case_id"], {})
        ids = token.get("input_token_ids", [])
        valid = (row["split"] == token.get("split") == "DEVELOPMENT"
            and isinstance(row["raw_prompt"], str) and row["raw_prompt"] == token.get("raw_prompt")
            and text_sha(row["raw_prompt"]) == row["raw_prompt_sha256"] == token.get("raw_prompt_sha256")
            and isinstance(token.get("rendered_prompt"), str)
            and text_sha(token["rendered_prompt"]) == token.get("rendered_prompt_sha256")
            and isinstance(ids, list) and 1 <= len(ids) <= 232 and token.get("input_tokens") == len(ids)
            and all(type(i) is int and 0 <= i < 151936 for i in ids))
        if not valid:
            errors.append("export prompt/token hash/shape differs:" + row["case_id"])
            continue
        result.append({"case_id": row["case_id"], "split": "DEVELOPMENT", "prompt": row["raw_prompt"],
            "prompt_sha256": row["raw_prompt_sha256"], "rendered_prompt": token["rendered_prompt"],
            "rendered_prompt_sha256": token["rendered_prompt_sha256"], "input_token_ids": ids,
            "prefix_ids_sha256": digest(ids), "prefix_tokens": len(ids)})
    return errors, result


def analysis_membership(admission, manifest, current, reader_relative, plan_relative):
    """Actual frozen membership; there is no ancestry/manifest fallback."""
    errors = []
    dispatch = admission.get("dispatch_hashes", {})
    if not dispatch or manifest.get("sha256") != dispatch:
        errors.append("parent manifest exact admission map differs")
    for relative, field in ((reader_relative, "analysis_source"), (plan_relative, "analysis_plan")):
        if relative not in dispatch or current.get(relative) != dispatch.get(relative) or admission.get(field + "_sha256") != dispatch.get(relative):
            errors.append("actual frozen reader/plan membership missing/different:" + relative)
    return errors


def validate_statistics(s, iteration, ordinary=None):
    errors = []
    if not isinstance(s, dict) or set(s) != STAT_KEYS or not clean(s):
        return ["feature statistics exact schema/finite firewall differs"]
    scalars = STAT_KEYS - {"denominator_policy", "finite", "zero_old_norm", "zero_new_norm", "zero_delta_norm"}
    if any(v is not None and not number(v) for k, v in s.items() if k in scalars):
        errors.append("feature statistics nonnumeric")
        return errors
    if s["denominator_policy"] != DENOM_POLICY or s["finite"] is not True:
        errors.append("feature denominator/finite declaration differs")
    norms = ("old_l2", "new_l2", "delta_l2", "old_rms", "new_rms", "normal_rms", "delta_rms", "state_cosine_denominator", "delta_over_old_l2_denominator")
    if any(not number(s[k]) or s[k] < 0 for k in norms):
        return errors + ["feature norm operands invalid"]
    for flag, norm in (("zero_old_norm", "old_l2"), ("zero_new_norm", "new_l2"), ("zero_delta_norm", "delta_l2")):
        if s[flag] is not (s[norm] == 0):
            errors.append("feature zero-norm declaration differs")
    if s["state_cosine_denominator"] != s["old_l2"] * s["new_l2"] or s["delta_over_old_l2_denominator"] != s["old_l2"]:
        errors.append("feature raw denominators differ")
    relations = [("state_cosine", "state_dot", "state_cosine_denominator", True),
                 ("delta_over_old_l2", "delta_l2", "old_l2", False)]
    previous = ("previous_delta_l2", "delta_dot_previous", "delta_cosine_denominator", "delta_cosine_previous", "delta_over_previous_l2")
    if iteration == 0 and any(s[k] is not None for k in previous):
        errors.append("ordinary feature carries previous/extra statistics")
    if iteration == 1:
        if ordinary is None or s["previous_delta_l2"] != ordinary.get("delta_l2") or s["old_l2"] != ordinary.get("new_l2") or s["old_rms"] != ordinary.get("new_rms"):
            errors.append("postextra feature does not follow same-prefix ordinary delta")
        if not number(s["previous_delta_l2"]) or s["previous_delta_l2"] < 0 or s["delta_cosine_denominator"] != s["delta_l2"] * s["previous_delta_l2"]:
            errors.append("postextra raw previous denominator differs")
        relations += [("delta_cosine_previous", "delta_dot_previous", "delta_cosine_denominator", True),
                      ("delta_over_previous_l2", "delta_l2", "previous_delta_l2", False)]
    for field, numerator, denominator, cosine in relations:
        den, num = s[denominator], s[numerator]
        if not number(den) or not number(num):
            errors.append("feature raw ratio operands missing")
            continue
        expected = None if den == 0 else num / den
        if expected is not None and cosine:
            expected = max(-1., min(1., expected))
        if s[field] != expected:
            errors.append("feature ratio/null differs:" + field)
    return errors


def validate_readout(row):
    if set(row) != TASK_CONTEXT | READ_BODY or not clean(row):
        return ["readout exact schema/finite/label firewall differs"]
    errors = []
    required = {"schema": SCHEMA, "vocabulary_rows": 151936, "UNKNOWN_token_id": 51378,
        "rival_set": RIVAL_SET, "logits_changed": False, "target_independent": True, "finite": True,
        "prefix_kind": "ORIGINAL_INPUT_ONLY", "preceding_gold_tokens": 0, "teacher_conditioned": False,
        "extra_model_calls_for_statistics": 0}
    if any(row.get(k) != v or type(row.get(k)) is not type(v) for k, v in required.items()):
        errors.append("readout target/head/availability contract differs")
    numeric = ("UNKNOWN_logit", "rival_logit", "UNKNOWN_minus_rival_margin", "top1_probability", "top2_margin", "entropy_nats", "logsumexp", "observer_seconds", "readout_observer_seconds")
    if any(not number(row[k]) for k in numeric):
        return errors + ["readout raw numeric operands invalid"]
    ids, values = row["top2_token_ids"], row["top2_logits"]
    if not isinstance(ids, list) or not isinstance(values, list) or len(ids) != 2 or len(values) != 2 or len(set(ids)) != 2 or any(type(i) is not int or not 0 <= i < 151936 for i in ids) or any(not number(v) for v in values):
        return errors + ["readout raw top2 operands invalid"]
    if type(row["rival_token_id"]) is not int or not 0 <= row["rival_token_id"] < 151936 or row["rival_token_id"] == 51378 or type(row["top1_token_id"]) is not int or not 0 <= row["top1_token_id"] < 151936:
        errors.append("readout top1/rival IDs differ")
    if row["UNKNOWN_minus_rival_margin"] != fp32(row["UNKNOWN_logit"] - row["rival_logit"]) or row["top2_margin"] != fp32(values[0] - values[1]) or values[0] < values[1]:
        errors.append("readout recorded margin does not match raw FP32 operands")
    # argmax, topk and the rival-vector argmax need not return identical IDs at
    # ties. Top2 cannot prove that the second-ranked nonUNKNOWN row is unique.
    nonunknown_top2 = [values[i] for i in range(2) if ids[i] != 51378]
    if row["rival_logit"] != max(nonunknown_top2) or max(row["UNKNOWN_logit"], row["rival_logit"]) != values[0] or any(row["UNKNOWN_logit"] != values[i] for i in range(2) if ids[i] == 51378) or (51378 not in ids and row["UNKNOWN_logit"] > values[1]):
        errors.append("readout UNKNOWN/unfiltered rival operands inconsistent")
    if values[0] > values[1] and row["top1_token_id"] != ids[0]:
        errors.append("readout unique top1 identity differs")
    if row["top1_token_id"] == 51378 and row["UNKNOWN_logit"] != values[0] or row["top1_token_id"] != 51378 and row["rival_logit"] != values[0]:
        errors.append("readout top1 value inconsistent with UNKNOWN/rival maxima")
    if values[0] > values[1] and ids[0] != 51378 and row["rival_token_id"] != ids[0]:
        errors.append("readout unique nonUNKNOWN maximum identity differs")
    if not 0 <= row["top1_probability"] <= 1 or row["entropy_nats"] < 0 or row["top2_margin"] < 0 or row["observer_seconds"] < 0 or row["readout_observer_seconds"] < 0:
        errors.append("readout probability/entropy/timing bounds invalid")
    return errors


def readout_numeric_diagnostics(row):
    """Cross-kernel comparisons are descriptive, never required equalities."""
    try:
        reconstructed = math.exp(row["top2_logits"][0] - row["logsumexp"])
    except OverflowError:
        reconstructed = None
    difference = None if reconstructed is None else row["top1_probability"] - reconstructed
    ids, values = row["top2_token_ids"], row["top2_logits"]
    return {"case_id": row["case_id"], "mode": row["mode"],
        "raw_readout": {k: row[k] for k in READ_BODY},
        "cross_kernel_status": "UNVERIFIED_CROSS_FP32_KERNEL_COMPARISON",
        "kernel_provenance": {"observer_sha256": NATIVE_PINS["observer.py"],
            "top1_probability": "FP32 z.log_softmax(0).exp()[z.argmax()]",
            "logsumexp": "separate FP32 z.logsumexp(0)",
            "descriptive_reconstruction": "stdlib exp(recorded top2 maximum logit minus recorded LSE)",
            "margin_validation": "same-source recorded FP32 scalar subtraction; no cross-kernel equality gate"},
        "reconstructed_probability": reconstructed, "signed_probability_discrepancy": difference,
        "absolute_probability_discrepancy": None if difference is None else abs(difference),
        "logsumexp_minus_max_logit_descriptive": row["logsumexp"] - values[0],
        "operand_error_bound_available": False, "tolerance_or_quality_gate": None,
        "tie_identity": {"top2_max_value_tie_observed": values[0] == values[1],
            "argmax_vs_topk_id_equal": row["top1_token_id"] == ids[0],
            "rival_id_present_in_top2": row["rival_token_id"] in ids,
            "rival_tie_uniqueness": "UNVERIFIED_FROM_TOP2_ONLY" if ids[0] == 51378 or values[0] == values[1] else "UNIQUE_NONUNKNOWN_MAXIMUM_FROM_STRICT_TOP2_GAP"}}


def audit_records(population, forwards, feature_events, readouts):
    """Pure in-memory audit; never reads annotations or fits a classifier."""
    errors, paired = [], {}
    by = {x["case_id"]: x for x in population}
    if len(population) != 192 or len(by) != 192 or any(set(x) != POP_KEYS or not clean(x) for x in population):
        errors.append("native projected population schema/192 uniqueness differs")
    task_keys = {(cid, mode) for cid in by for mode in ("OBSERVED_K0", "OBSERVED_UNIT")}
    start, terminal = {}, {}
    counts = Counter(x.get("event") for x in forwards)
    expected_events = {"FORWARD_STARTED": 449, "FORWARD_COMPLETED": 448, "FORWARD_FAILED": 1,
                       "CACHE_REJECTION_STARTED": 2, "CACHE_REJECTED": 2}
    if counts != expected_events:
        errors.append("exact451 API/449 attempts/448 complete journal count differs")
    clocks = [x.get("parent_elapsed_seconds") for x in forwards]
    if any(not number(t) or not 0 <= t < 900 for t in clocks) or any(a > b for a, b in zip(clocks, clocks[1:]) if number(a) and number(b)):
        errors.append("forward parent clocks invalid/nonmonotonic")
    for row in forwards:
        if not clean(row):
            errors.append("forward contamination/nonfinite field")
        attempt, event = row.get("API_attempt"), row.get("event")
        dest = start if event in ("FORWARD_STARTED", "CACHE_REJECTION_STARTED") else terminal
        if type(attempt) is not int or attempt in dest:
            errors.append("duplicate/invalid started or terminal API identity")
        dest[attempt] = row
    if set(start) != set(range(1, 452)) or set(terminal) != set(start):
        errors.append("every451 API attempt needs exactly one start/terminal")
    completed = [x for x in forwards if x.get("event") == "FORWARD_COMPLETED"]
    failures = [x for x in forwards if x.get("event") == "FORWARD_FAILED"]
    positions = sum(x.get("prefix_tokens", 0) for x in completed if type(x.get("prefix_tokens")) is int)
    if positions != 54137:
        errors.append("completed54137 positions differs; partial work remains unknown")
    for attempt, end in terminal.items():
        begin = start.get(attempt, {})
        if begin.get("event") == "CACHE_REJECTION_STARTED":
            if end.get("event") != "CACHE_REJECTED" or end.get("model_compute_started") is not False or end.get("completed") is not False or end.get("task_measurement") is not False or begin.get("use_cache") != end.get("use_cache") or begin.get("cache_object_present") != end.get("cache_object_present"):
                errors.append("cache not rejected before compute")
        else:
            if begin.get("event") != "FORWARD_STARTED" or end.get("event") not in ("FORWARD_COMPLETED", "FORWARD_FAILED") or any(begin.get(k) != end.get(k) for k in ("case_id", "phase", "split", "mode", "prefix_ids_sha256", "prefix_tokens", "extra_loops", "native_layers")):
                errors.append("full-forward started/terminal join differs")
            if not number(end.get("forward_seconds")) or end["forward_seconds"] < 0:
                errors.append("forward time missing/nonfinite")
            if end.get("event") == "FORWARD_COMPLETED" and (end.get("completed") is not True or end.get("post_forward_guard_error") is not None):
                errors.append("completed forward failed postguard")
    cache = [x for x in terminal.values() if x.get("event") == "CACHE_REJECTED"]
    if {(x.get("use_cache"), x.get("cache_object_present")) for x in cache} != {(True, False), (False, True)}:
        errors.append("both exact cache variants required")
    if len(failures) != 1 or any(x.get("expected_forced_exception") is not True or x.get("phase") != "QUALIFICATION" or x.get("case_id") != "SYNTHETIC_MAX357_NO_TARGET" or x.get("mode") != "OBSERVED_UNIT" or x.get("prefix_tokens") != 357 or x.get("partial_compute") != "UNKNOWN" or "EXPECTED_PASSIVE_FORCED_EXCEPTION" not in x.get("error", "") for x in failures):
        errors.append("unexpected failure or unmeasured partial misrepresented")
    task = [x for x in completed if x.get("phase") == "TASK"]
    qualification = [x for x in completed if x.get("phase") == "QUALIFICATION"]
    if len(task) != 384 or {(x.get("case_id"), x.get("mode")) for x in task} != task_keys or len(qualification) != 64 or sum(x["prefix_tokens"] for x in qualification) != 4921 or sum(x["prefix_tokens"] for x in task) != 49216:
        errors.append("qualification/task forward matrix/position split differs")
    for row in task:
        ref = by.get(row.get("case_id"), {})
        if row.get("split") != "DEVELOPMENT" or row.get("prefix_tokens") != ref.get("prefix_tokens") or any(row.get(k) != ref.get(k) for k in ("prefix_ids_sha256", "prompt_sha256", "rendered_prompt_sha256")) or row.get("native_layers") != [12, 13] or row.get("extra_loops") != int(row.get("mode") == "OBSERVED_UNIT"):
            errors.append("task forward original input/operator binding differs")
    if task and qualification and min(x["parent_elapsed_seconds"] for x in task) <= max(x["parent_elapsed_seconds"] for x in qualification):
        errors.append("task preceded complete fresh qualification")
    grouped, qualification_groups = defaultdict(list), defaultdict(list)
    for row in feature_events:
        if not isinstance(row, dict) or not clean(row):
            errors.append("feature label/finite firewall differs")
            continue
        task_row = row.get("task_measurement") is True
        context = TASK_CONTEXT if task_row else {"case_id", "split", "phase", "mode"}
        if set(row) != context | FEATURE_BODY or row.get("schema") != SCHEMA:
            errors.append("feature exact schema differs")
            continue
        iteration = row.get("iteration")
        if type(iteration) is not int or iteration not in (0, 1) or row.get("stage") != ("ORDINARY_PRE_EXTRA" if iteration == 0 else "UNIT_PROPOSAL_POST_EXTRA") or row.get("available_before_extra") is not (iteration == 0) or row.get("extra_block_already_computed") is not (iteration == 1) or row.get("alpha") != 1. or row.get("sample_id") != row.get("case_id") or row.get("forward_index") != 1 or row.get("last_token_index") != row.get("prefix_tokens", 0) - 1 or row.get("preceding_gold_tokens") != 0 or row.get("target_independent") is not True or row.get("teacher_conditioned") is not False:
            errors.append("feature stage/causal availability/identity contract differs")
        expected_prefix = "SYNTHETIC_MAX357_NO_TARGET" if row.get("split") == "MECHANICAL" else "ORIGINAL_INPUT_ONLY"
        if row.get("prefix_kind") != expected_prefix:
            errors.append("feature original prefix/teacher firewall differs")
        if task_row:
            ref = by.get(row["case_id"], {})
            if row.get("phase") != "TASK" or row.get("split") != "DEVELOPMENT" or row.get("prefix_tokens") != ref.get("prefix_tokens") or any(row.get(k) != ref.get(k) for k in ("prefix_ids_sha256", "prompt_sha256", "rendered_prompt_sha256")):
                errors.append("feature task original-prefix hashes differ")
            grouped[(row["case_id"], row["mode"])].append(row)
        else:
            if row.get("phase") != "QUALIFICATION" or row.get("split") not in ("SMOKE", "MECHANICAL") or row.get("mode") not in ("OBSERVED_K0", "OBSERVED_UNIT"):
                errors.append("qualification feature context differs")
            qualification_groups[(row["case_id"], row["mode"])].append(row)
    if sum(x.get("task_measurement") is True for x in feature_events) != 576 or sum(x.get("task_measurement") is False for x in feature_events) != 27 or set(grouped) != task_keys:
        errors.append("exact576 task/27 qualification feature coverage differs")
    expected_qualification_keys = {(x.get("case_id"), x.get("mode")) for x in qualification
                                   if x.get("mode") in ("OBSERVED_K0", "OBSERVED_UNIT")}
    if len(expected_qualification_keys) != 18 or set(qualification_groups) != expected_qualification_keys:
        errors.append("qualification feature membership differs from actual observed qualification calls")
    for key, rows in list(grouped.items()) + list(qualification_groups.items()):
        if [x["iteration"] for x in rows] != ([0] if key[1] == "OBSERVED_K0" else [0, 1]):
            errors.append("ordinary/postextra feature stage order differs")
            continue
        for row in rows:
            errors += validate_statistics(row["statistics"], row["iteration"], rows[0]["statistics"] if row["iteration"] else None)
    for cid in by:
        k0, unit = grouped.get((cid, "OBSERVED_K0"), []), grouped.get((cid, "OBSERVED_UNIT"), [])
        if k0 and unit:
            if k0[0]["statistics"] != unit[0]["statistics"]:
                errors.append("A0/unit ordinary same-prefix statistics not exact")
            paired[cid] = {"ordinary": k0[0]["statistics"], "postextra": unit[-1]["statistics"]}
    rout = {}
    for row in readouts:
        errors += validate_readout(row)
        key = (row.get("case_id"), row.get("mode"))
        if key in rout:
            errors.append("duplicate readout cell")
        rout[key] = row
        ref = by.get(row.get("case_id"), {})
        if row.get("phase") != "TASK" or row.get("split") != "DEVELOPMENT" or any(row.get(k) != ref.get(k) for k in ("prefix_ids_sha256", "prompt_sha256", "rendered_prompt_sha256")):
            errors.append("readout original task-prefix hashes differ")
    if len(readouts) != 384 or set(rout) != task_keys:
        errors.append("complete384 readout matrix differs")
    return errors, {"API_attempts": len(start), "completed_forwards": len(completed), "completed_prefix_positions": positions,
        "partial_compute_positions": None, "task_features": sum(x.get("task_measurement") is True for x in feature_events),
        "task_readouts": len(readouts)}, paired, rout


def validate_packet(bundle):
    """Complete native gates, separated from label-bearing offline joins."""
    errors = list(bundle.get("binding_errors", []))
    admission, registration = bundle.get("admission", {}), bundle.get("registration", {})
    qualification, receipt, terminal = (bundle.get(k, {}) for k in ("qualification", "receipt", "terminal"))
    export_errors, expected_population = validate_export(bundle.get("export_bank"), bundle.get("export_tokens"))
    errors += export_errors
    population = bundle.get("population", [])
    if population != expected_population:
        errors.append("actual native population differs from exact prompt-only export")
    if admission.get("status") != "READY_FOR_PASSIVE_PROBE_FP32_DISPATCH" or admission.get("study_id") != STUDY_ID or admission.get("lane") != LANE or admission.get("count_contract") != CONTRACT or admission.get("training") is not False or admission.get("native_test") is not False:
        errors.append("frozen admission lane/count/no-training contract differs")
    required_reg = {"lane": LANE, "study_id": STUDY_ID, "schema": SCHEMA, "count_contract": CONTRACT,
        "native_layers": [12, 13], "extra_loops": [0, 1], "alpha": 1., "seed": 17, "prefix_kind": "ORIGINAL_INPUT_ONLY",
        "generation": False, "teacher_conditioned": False, "training": False, "state_adjoints": 0,
        "readout_changed": False, "controller_routing": False, "hard_deadline_seconds": 900,
        "body_deadline_seconds": 780, "internal_reserve_seconds": 120, "global_finalization_reserve_seconds": 1800}
    if any(registration.get(k) != v for k, v in required_reg.items()) or registration.get("admission_sha256") != bundle.get("admission_sha256"):
        errors.append("actual registration/reader study contract differs")
    if registration.get("data_sha256") != admission.get("dev_bank_sha256") or registration.get("token_binding_sha256") != admission.get("dev_token_binding_sha256"):
        errors.append("actual registration export binding differs")
    if qualification.get("status") != "PASS_PASSIVE_INPUT_FEATURE_MECHANICS" or qualification.get("counts") != QUAL_COUNTS or qualification.get("maximum_prefix_tokens") != 357 or qualification.get("cache_rejections") != 2:
        errors.append("fresh complete qualification/counts absent")
    flags = ("checkpoint_files_unchanged", "source_unchanged", "forced_exception_restoration", "L0_L2_one_shot", "head_and_operator_unchanged", "observer_output_identity")
    if any(qualification.get(k) is not True for k in flags) or qualification.get("parameter_gradients") != "NONE":
        errors.append("qualification preservation/hooks/head/operator guard absent")
    parity = qualification.get("parity_proofs", [])
    if len(parity) != 9 or len({x.get("context", {}).get("case_id") for x in parity}) != 9:
        errors.append("all8 SMOKE plusmax357 parity proofs absent")
    for proof in parity:
        hashes = proof.get("logits_sha256", [])
        if proof.get("modes") != list(QUAL_MODES) or len(hashes) != 7 or not (hashes[0] == hashes[1] == hashes[2] == hashes[6] and hashes[3] == hashes[4] == hashes[5]):
            errors.append("byte-exact observer/intact/unit/restoration parity differs")
        context = proof.get("context", {})
        for index, mode in enumerate(QUAL_MODES):
            actual = [x for x in bundle.get("forwards", []) if x.get("event") == "FORWARD_COMPLETED"
                      and x.get("phase") == "QUALIFICATION" and x.get("case_id") == context.get("case_id") and x.get("mode") == mode]
            if not actual or index >= len(hashes) or any(x.get("logits_sha256") != hashes[index] for x in actual):
                errors.append("qualification parity proof does not bind actual forward output hashes")
    for packet in (qualification, receipt):
        if packet.get("before_state_sha256") != STATE or packet.get("after_state_sha256") != STATE or packet.get("checkpoint_files_before") != admission.get("checkpoint_files") or packet.get("checkpoint_files_after") != admission.get("checkpoint_files") or packet.get("dispatch_hashes") != admission.get("dispatch_hashes"):
            errors.append("stateb23/ten-assets/all-admitted-source before/after differs")
    if len(admission.get("checkpoint_files", {})) != 10 or receipt.get("loaded_custom_code_sha256") != CUSTOM or receipt.get("parameter_gradients") != "NONE" or receipt.get("trainable_parameters") != 0 or receipt.get("native_model_loads") != 1 or receipt.get("model_load_attempts") != 1 or receipt.get("training") is not False or receipt.get("native_test") is not False or receipt.get("state_adjoints") != 0:
        errors.append("final frozen customhead/parameter/no-label firewall differs")
    if receipt.get("status") != "PASSIVE_INPUT_FEATURE_PROBE_COMPLETE_NOT_EFFICACY" or receipt.get("complete") is not True or receipt.get("counts") != COUNTS or receipt.get("cases") != 192 or receipt.get("task_readouts") != 384 or receipt.get("task_feature_events") != 576 or receipt.get("source_unchanged") is not True or receipt.get("checkpoint_files_unchanged") is not True or receipt.get("efficacy_claim") is not False:
        errors.append("final native completeness not qualified")
    forecast = bundle.get("forecast", {})
    estimate = forecast.get("remaining_forecast_seconds")
    if forecast.get("admitted") is not True or forecast.get("remaining_calls") != 384 or forecast.get("forecast_factor") != 1.5 or forecast.get("internal_reserve_seconds") != 120 or forecast.get("parent_deadline_seconds") != 900 or not number(estimate) or not number(forecast.get("max_observed_warm_forward_seconds")) or not number(forecast.get("elapsed_seconds")) or estimate != 384 * forecast["max_observed_warm_forward_seconds"] * 1.5 or forecast["elapsed_seconds"] + estimate + 120 > 900:
        errors.append("fresh complete conservative forecast missing/different")
    runtime = registration.get("runtime", {})
    if runtime.get("tf32") is not False or runtime.get("CPU_threads") != 4 or runtime.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or receipt.get("runtime") != runtime:
        errors.append("actual pinned deterministic GPU runtime differs")
    if terminal.get("status") != "WORKER_TERMINAL" or terminal.get("exit_code") != 0 or terminal.get("study_id") != STUDY_ID or terminal.get("lane") != LANE or terminal.get("admission_sha256") != bundle.get("admission_sha256") or not number(terminal.get("native_worker_seconds")) or not 0 < terminal["native_worker_seconds"] <= 900 or not number(receipt.get("parent_elapsed_seconds")) or not 0 <= receipt["parent_elapsed_seconds"] <= terminal["native_worker_seconds"]:
        errors.append("parent terminal/charged clock not complete")
    ledger = bundle.get("ledger", [])
    study_events = [x for x in ledger if x.get("study_id") == STUDY_ID]
    starts = [x for x in study_events if x.get("event") == "native_started"]
    ends = [x for x in study_events if x.get("event") == "native_completed"]
    if len(starts) != 1 or len(ends) != 1 or any(x.get("run") != terminal.get("run") for x in starts + ends) or not ends or any(ends[0].get(k) != terminal.get(k) for k in ("status", "exit_code", "worker_pid", "native_worker_seconds", "admission_sha256")):
        errors.append("study-specific ledger single-attempt parent join differs")
    spent = sum(x.get("native_worker_seconds", 0) for x in ledger if x.get("event") == "native_completed" and number(x.get("native_worker_seconds", 0)))
    if spent + 1800 > 21600:
        errors.append("cumulative charged ledger violates finalization reserve")
    record_errors, counts, paired, readouts = audit_records(population, bundle.get("forwards", []), bundle.get("features", []), bundle.get("readouts", []))
    errors += record_errors
    forwards = bundle.get("forwards", [])
    warm = [x for x in forwards if x.get("event") == "FORWARD_COMPLETED" and x.get("phase") == "QUALIFICATION"]
    task_starts = [x for x in forwards if x.get("event") == "FORWARD_STARTED" and x.get("phase") == "TASK"]
    if len(warm) != 64 or forecast.get("max_observed_warm_forward_seconds") != max((x.get("forward_seconds", -1) for x in warm[1:]), default=-1):
        errors.append("fresh forecast not computed from actual complete warm journal")
    if not task_starts or not number(forecast.get("elapsed_seconds")) or not warm or not max(x["parent_elapsed_seconds"] for x in warm) <= forecast["elapsed_seconds"] < min(x["parent_elapsed_seconds"] for x in task_starts):
        errors.append("forecast clock does not precede first task after qualification")
    initial = registration.get("initial_resources", {})
    if not number(initial.get("host_available_bytes")) or initial["host_available_bytes"] < 6 * 1024**3 or not number(initial.get("gpu_free_bytes")) or initial["gpu_free_bytes"] < 6 * 1024**3:
        errors.append("recorded preweight six-GiB host/GPU qualification floor missing")
    floors = {"gpu_allocated_bytes": (0, 5 * 1024**3), "gpu_free_bytes": (1024**3, None),
              "host_available_bytes": (1024**3, None), "rss_bytes": (0, 6 * 1024**3)}
    resources = [x.get("resources") for x in forwards if x.get("event") == "FORWARD_COMPLETED"]
    for sample in resources:
        if not isinstance(sample, dict) or any(not number(sample.get(k)) or sample[k] < low or high is not None and sample[k] > high for k, (low, high) in floors.items()):
            errors.append("completed forward worker resource samples/floors differ")
    parent_resources = bundle.get("resources", [])
    if not parent_resources or any(not number(x.get("rss_bytes")) or not 0 <= x["rss_bytes"] <= 6 * 1024**3 or not number(x.get("host_available_bytes")) or x["host_available_bytes"] < 1024**3 for x in parent_resources):
        errors.append("parent sampled tree RSS/host evidence absent/invalid")
    elif terminal.get("peak_rss_bytes") != max(x["rss_bytes"] for x in parent_resources):
        errors.append("parent peak RSS does not match actual samples")
    return errors, counts, paired, readouts


def distribution(values):
    known = [v for v in values if number(v)]
    return {"records": len(values), "known_records": len(known), "null_records": sum(v is None for v in values),
            "min": min(known) if known else None, "mean": statistics.mean(known) if known else None, "max": max(known) if known else None}


def summarize(population, paired, readouts, old_bank, old_outcomes):
    """Annotations join only after native firewall validation; descriptive only."""
    ids = {x["case_id"] for x in population}
    cases = {x["id"]: x for x in old_bank if x.get("split") == "DEVELOPMENT"}
    old = {(x.get("case_id"), x.get("arm")): x for x in old_outcomes if x.get("split") == "DEVELOPMENT" and x.get("physical_run") is True and x.get("arm") in ("A0", "A1") and x.get("seed") == 17}
    if set(cases) != ids or len(old) != 384 or any((cid, arm) not in old or old[cid, arm].get("score") not in (0, 1) for cid in ids for arm in ("A0", "A1")):
        raise ValueError("exact old192 DEV intact/unit score join differs")
    groups, parent, family, values = defaultdict(list), defaultdict(list), defaultdict(list), {}
    for cid in sorted(ids):
        a, u = old[cid, "A0"]["score"], old[cid, "A1"]["score"]
        group = "both_correct" if a and u else "A0_only" if a else "unit_only" if u else "both_wrong"
        numeric = {"ordinary." + k: v for k, v in paired[cid]["ordinary"].items() if number(v) or v is None}
        numeric.update({"postextra." + k: v for k, v in paired[cid]["postextra"].items() if number(v) or v is None})
        for mode, short in (("OBSERVED_K0", "K0"), ("OBSERVED_UNIT", "unit")):
            row = readouts[cid, mode]
            for k in ("UNKNOWN_minus_rival_margin", "top2_margin", "entropy_nats", "top1_probability"):
                numeric[short + "." + k] = row[k]
        numeric["unit_minus_K0.UNKNOWN_margin"] = numeric["unit.UNKNOWN_minus_rival_margin"] - numeric["K0.UNKNOWN_minus_rival_margin"]
        values[cid] = numeric
        groups[group].append(cid); parent[cases[cid]["parent_id"]].append(cid); family[cases[cid]["category"]].append(cid)
    def group_table(members):
        return {g: {"cases": len(ii), "parents": len({cases[i]["parent_id"] for i in ii}),
                   "features": {k: distribution([values[i][k] for i in ii]) for k in next(iter(values.values()))}}
                for g, ii in members.items()}
    group_summary = group_table({g: groups[g] for g in ("both_correct", "A0_only", "unit_only", "both_wrong")})
    overlaps = {}
    for key in next(iter(values.values())):
        left, right = (group_summary[g]["features"][key] for g in ("A0_only", "unit_only"))
        available = left["min"] is not None and right["min"] is not None
        low, high = (max(left["min"], right["min"]), min(left["max"], right["max"])) if available else (None, None)
        overlaps[key] = {"available": available, "range_intersection_min": low, "range_intersection_max": high,
                         "ranges_overlap": low <= high if available else None, "classification_claim": False}
    parents = [{"parent_id": p, "case_ids": ii, "old_correctness_groups": {g: sum(i in groups[g] for i in ii) for g in groups},
                "feature_means": {k: distribution([values[i][k] for i in ii])["mean"] for k in next(iter(values.values()))}} for p, ii in sorted(parent.items())]
    families = {f: {"cases": len(ii), "parents": len({cases[i]["parent_id"] for i in ii}),
                   "old_groups": {g: sum(i in groups[g] for i in ii) for g in groups}, "feature_ranges": group_table({f: ii})[f]["features"]}
                for f, ii in sorted(family.items())}
    return {"groups": group_summary, "A0_only_vs_unit_only_range_overlap": overlaps, "parent_rows": parents, "family": families,
            "case_rows": [{"case_id": cid, "parent_id": cases[cid]["parent_id"], "category": cases[cid]["category"],
                           "old_group": next(g for g, ii in groups.items() if cid in ii), "features": values[cid]} for cid in sorted(ids)],
            "scope": "Exposed192 DEV old generated correctness joined offline after native validation; no fitting, CI, gate-quality or efficacy claim."}


def load_bundle(run_root):
    """Enumerate every evidence requirement against the actual frozen contract."""
    bundle, errors, inputs = {}, [], []
    def read(path, lines=False):
        try:
            value = [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()] if lines else json.loads(path.read_text(encoding="utf-8-sig"))
            inputs.append({"path": str(path), "sha256": sha(path)})
            return value
        except (OSError, ValueError) as exc:
            errors.append("required evidence unavailable:" + str(path) + ":" + type(exc).__name__)
            return [] if lines else {}
    for key, name, lines in (("admission", "admission.json", False), ("terminal", "terminal_receipt.json", False), ("review", "review.json", False), ("manifest", "source_manifest.json", False)):
        bundle[key] = read(run_root / name, lines)
    for key, name, lines in (("registration", "registration.json", False), ("qualification", "qualification.json", False), ("receipt", "receipt.json", False), ("forecast", "matrix_forecast.json", False), ("population", "population.json", False), ("forwards", "forwards.jsonl", True), ("features", "feature_events.jsonl", True), ("readouts", "readouts.jsonl", True)):
        bundle[key] = read(run_root / "result" / name, lines)
    admission = bundle["admission"]
    bundle["admission_sha256"] = sha(run_root / "admission.json") if (run_root / "admission.json").exists() else None
    dispatch = admission.get("dispatch_hashes", {})
    current = {}
    for relative, expected in dispatch.items():
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents:
            errors.append("admitted source escapes workspace:" + relative); continue
        try:
            current[relative] = sha(path)
            if current[relative] != expected:
                errors.append("actual frozen source changed:" + relative)
        except OSError:
            errors.append("actual frozen source missing:" + relative)
    reader_relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    plan = STUDY / "PROBE_ANALYSIS_PLAN_001.md"
    plan_relative = plan.relative_to(ROOT).as_posix()
    errors += analysis_membership(admission, bundle["manifest"], current, reader_relative, plan_relative)
    for field, path in (("analysis_source", Path(__file__).resolve()), ("analysis_plan", plan),
                        ("authority", STUDY / "ACTIVE_GOAL_SCOPE_001.json"), ("goal_snapshot", STUDY / "ACTIVE_GOAL_SNAPSHOT_001.json")):
        if Path(admission.get(field + "_path", "")).resolve() != path.resolve():
            errors.append("exact analysis/scope evidence path differs:" + field)
        packet = read(path) if field in ("authority", "goal_snapshot") else None
        if packet is None:
            if path.exists():
                inputs.append({"path": str(path), "sha256": sha(path)})
            else:
                errors.append("analysis source/plan missing:" + str(path))
        if not path.exists() or sha(path) != admission.get(field + "_sha256"):
            errors.append("exact analysis/scope evidence hash differs:" + field)
        if field in ("authority", "goal_snapshot"):
            bundle[field] = packet
    scope, goal = bundle.get("authority", {}), bundle.get("goal_snapshot", {}).get("goal", {})
    if goal.get("status") != "active" or scope.get("objective") != goal.get("objective") or scope.get("status") != "DISTINCT_FROZEN_DEV_RESEARCH_AUTHORIZED_BY_NEW_GOAL" or scope.get("extra_budgets") != [0, 1] or any(scope.get(k) is not False for k in ("training", "protected_data", "cloud_spending", "readout_mutation", "operator_mutation", "retry_available")):
        errors.append("distinct active scope/goal snapshot not bound")
    for name, expected in NATIVE_PINS.items():
        if current.get((STUDY / name).relative_to(ROOT).as_posix()) != expected:
            errors.append("registered native/observer source differs:" + name)
    export_dir = STUDY / "data/dev_prefix_probe_001"
    for key, name in (("export_bank", "DEVELOPMENT_BANK.json"), ("export_token_packet", "DEVELOPMENT_TOKEN_BINDING.json"), ("export_manifest", "MANIFEST.json")):
        path = export_dir / name
        bundle[key] = read(path)
        relative = path.relative_to(ROOT).as_posix()
        field = {"export_bank": "dev_bank", "export_token_packet": "dev_token_binding", "export_manifest": "dev_export_manifest"}[key]
        if Path(admission.get(field + "_path", "")).resolve() != path.resolve() or current.get(relative) != EXPORT_PINS[name] or admission.get(field + "_sha256") != EXPORT_PINS[name]:
            errors.append("actual exact DEV export membership/path/hash differs:" + name)
    bundle["export_tokens"] = bundle["export_token_packet"].get("rows", [])
    em, token = bundle["export_manifest"], bundle["export_token_packet"]
    if em.get("status") != "PROMPT_ONLY_EXPOSED_DEV_EXPORT" or em.get("allowed_split") != "DEVELOPMENT" or em.get("case_count") != 192 or em.get("parent_units") != 96 or em.get("bank_sha256") != EXPORT_PINS["DEVELOPMENT_BANK.json"] or em.get("token_binding_sha256") != EXPORT_PINS["DEVELOPMENT_TOKEN_BINDING.json"] or token.get("status") != "PROMPT_ONLY_NATIVE_NOT_MEASURED" or token.get("native_model_loads") != 0 or token.get("canonical_source") != admission.get("source"):
        errors.append("DEV-only export/source/no-model registration differs")
    if bundle["review"].get("status") != "PASS_PASSIVE_PROBE_FP32_DISPATCH" or bundle["review"].get("admission_sha256") != bundle["admission_sha256"]:
        errors.append("independent actual admission dispatch review differs")
    review_path = CAMPAIGN / "quality_pattern_v3/reviews/probe_dispatch_receipt_002.json"
    if admission.get("review_receipt") != "quality_pattern_v3/reviews/probe_dispatch_receipt_002.json" or not review_path.exists() or not (run_root / "review.json").exists() or sha(review_path) != sha(run_root / "review.json"):
        errors.append("new v2 admission review002 path/copy binding differs")
    original = STUDY / "probe_admission_v2.json"
    if not original.exists() or sha(original) != bundle["admission_sha256"]:
        errors.append("copied actual admission differs from original freeze")
    for name, expected in admission.get("checkpoint_files", {}).items():
        try:
            if sha(Path(admission["source"]) / name) != expected:
                errors.append("canonical asset changed:" + name)
        except (KeyError, OSError):
            errors.append("canonical asset unavailable:" + name)
    for relative, expected in registration_hashes(bundle).items():
        if dispatch.get(relative) != expected:
            errors.append("inherited native registration source not admitted:" + relative)
    for field, name in (("model_binding", "MODEL_DATA_BINDING.json"), ("smoke_prompt_audit", "PROMPT_AUDIT.json")):
        path = CAMPAIGN / "runs/binding-001" / name
        value = read(path)
        if Path(admission.get(field + "_path", "")).resolve() != path.resolve() or admission.get(field + "_sha256") != sha(path) or dispatch.get(path.relative_to(ROOT).as_posix()) != sha(path):
            errors.append("exact inherited model/SMOKE admission binding differs:" + field)
        bundle[field] = value
    bundle["ledger"] = read(CAMPAIGN / "BUDGET_LEDGER.jsonl", True)
    bundle["resources"] = read(run_root / "resources.jsonl", True)
    for relative, expected in OFFLINE_PINS.items():
        path = CAMPAIGN / relative
        value = read(path, path.suffix == ".jsonl")
        if path.exists() and sha(path) != expected:
            errors.append("pinned offline reference changed:" + relative)
        if relative.endswith("outcomes.jsonl"):
            bundle["old_outcomes"] = value
        elif relative.endswith("bank.json"):
            bundle["old_bank"] = value
        else:
            bundle["old_causal_report"] = value
    bundle["binding_errors"], bundle["input_bindings"] = errors, inputs
    return bundle


def registration_hashes(bundle):
    return bundle.get("registration", {}).get("source_hashes", {})


def analyze(bundle):
    try:
        errors, counts, paired, readouts = validate_packet(bundle)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, struct.error) as exc:
        errors, counts, paired, readouts = list(bundle.get("binding_errors", [])) + ["required packet malformed:" + type(exc).__name__], {}, {}, {}
    primary = None
    if not errors:
        try:
            primary = summarize(bundle["population"], paired, readouts, bundle["old_bank"], bundle["old_outcomes"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append("offline DEV reference join refused:" + str(exc))
    return {"status": "COMPLETE_PASSIVE_DESCRIPTIVE_DEV_ONLY" if not errors else "INCOMPLETE_NO_PRIMARY_ASSOCIATION",
        "errors": sorted(set(errors)), "primary": primary, "counts": counts, "model_calls_by_reader": 0,
        "fitted_thresholds_or_policy": False, "generation_or_quality_effect": None, "unexposed_validation": None,
        "feature_availability": {"ordinary": "input-only before extra block; exactK0/unit statistics parity",
            "postextra": "unit proposal already computed; cannot claim a free pre-extra K decision",
            "readouts": "actual unchanged full-suffix UNKNOWN/all-other-row head logits; first prefix only"},
        "readout_numeric_diagnostics": [readout_numeric_diagnostics(row) for row in readouts.values()] if not errors else None,
        "scope": "All192 DEV exposed. Old generated correctness labels join only offline after native gates. CPU local canonical derivative and GPU first-prefix features are different measurements; no causal proof/CI/quality gate or fitted policy. Separate FP32 probability/LSE kernels have no established cross-expression error bound; discrepancies are UNVERIFIED and never repair/drop observations.",
        "bindings": bundle.get("input_bindings", [])}


def run(run_root, out):
    run_root, out = Path(run_root).resolve(), Path(out).resolve()
    if out == STUDY or STUDY not in out.parents or run_root == out or run_root in out.parents or out in run_root.parents:
        raise PermissionError("exclusive analysis output must be separate and V3-local")
    if run_root != (STUDY / "runs/native-prefix-probe-001").resolve():
        raise PermissionError("exact registered native-prefix-probe-001 run required")
    out.mkdir(parents=True, exist_ok=False)
    try:
        bundle = load_bundle(run_root)
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        bundle = {"binding_errors": ["required file/binding packet malformed:" + type(exc).__name__], "input_bindings": []}
    report = analyze(bundle)
    for name, value in (("REPORT.json", report), ("MANIFEST.json", {"reader_sha256": sha(__file__),
            "plan_sha256": sha(STUDY / "PROBE_ANALYSIS_PLAN_001.md") if (STUDY / "PROBE_ANALYSIS_PLAN_001.md").exists() else None, "run_root": str(run_root),
            "inputs": bundle["input_bindings"], "offline_reference_role": "Exact pinned historical artifacts, never native-qualified or manifest fallback",
            "model_calls": 0}), ("receipt.json", {"status": report["status"], "created_utc": datetime.now(timezone.utc).isoformat(),
            "reader_sha256": sha(__file__), "models": 0, "training": False, "native_TEST": False, "primary_available": report["primary"] is not None})):
        with (out / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False); handle.write("\n")
    with (out / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write("# " + report["status"] + "\n\n" + report["scope"] + "\n\n")
        if report["errors"]:
            handle.write("Required gates failed; primary association is null.\n\n")
            handle.write("\n".join("- " + error for error in report["errors"]) + "\n")
        else:
            handle.write("| Old DEV correctness group | Cases | Parents |\n|---|---:|---:|\n")
            for group, value in report["primary"]["groups"].items():
                handle.write(f"| {group} | {value['cases']} | {value['parents']} |\n")
            handle.write("\nEvery feature range/mean, parent/family association and range overlap is retained in REPORT.json. Post-extra features require the unit proposal to have already executed; none is a free pre-extra guard.\n")
    print(json.dumps({"status": report["status"], "errors": report["errors"], "output": str(out)}))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.run_root, args.out))
