"""Validate the fixed R3/R4 depth family and report all ten registered comparisons.

No model, tokenizer, or torch import. Historical artifacts are never changed. Missing
or invalid pairs use p=1 in the fixed ten-test Holm correction, and are unavailable
for interpretation. An incomplete family is explicitly labelled in the report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.architecture_benchmark_001 import retention
from experiments.rab001_exploration import headroom_audit

DIGEST = "a329eb7bb06fc003a9220ea41713e721fba575c478008240749c425774b2b58e"
ARMS = ("B28last8", "B28last12", "B28last16", "B28last20", "B28")
PAIRS = tuple(combinations(ARMS, 2))
ALPHA = Fraction(1, 20)
BASE = REPO / "experiments/architecture_benchmark_001"
EVIDENCE = BASE / "evidence/retention-design-v1"
RUN_REGISTRY = REPO / "experiments/rab001_exploration/run_registry.json"
SUPPORTED_RETRIES = {"B28_lr1e-05_b5200_seed17": ("B28_lr1e-05_b5200_seed17_retry1", "B28_lr1e-05_b5200_seed17_retry2")}


class InvalidArtifact(ValueError):
    """A result cannot enter a paired comparison."""


def require(condition, reason):
    if not condition:
        raise InvalidArtifact(reason)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject(value):
        raise InvalidArtifact(f"nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_unique_object, parse_constant=reject)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_initial_failure(metadata, canonical):
    require(metadata.get("no_endpoint_observed") is True, "initial failure no_endpoint_observed mismatch")
    expected_path = EVIDENCE / f"ARM_DOSE_{canonical}.json"
    require((REPO / metadata["failure_artifact_path"]).resolve() == expected_path.resolve(), "failure artifact path differs from canonical run")
    require(expected_path.is_file(), "registered canonical failure artifact missing")
    require(sha256(expected_path) == metadata["failure_artifact_sha256"], "preserved canonical failure hash mismatch")
    failure = read_json(expected_path)
    require(failure.get("completed") is False and failure.get("checkpoint") is None, "registered failure has a completed result or checkpoint")
    require(failure.get("aborted_reason") == metadata["failure_reason"] and bool(metadata["failure_reason"]), "registered failure reason mismatch")
    require(failure.get("supervised_target_tokens") == metadata["observed_supervised_target_tokens"]
            and failure.get("optimizer_steps") == metadata["observed_optimizer_steps"], "registered failed dose mismatch")
    require(failure.get("checkpoints") and all(point.get("tokens") == 0 and point.get("steps") == 0
                                               and point.get("milestone") == 0 for point in failure["checkpoints"]),
            "retry1 disallowed: a post-training checkpoint/endpoint was observed")
    for key, value in {"arm": "B28", "lr": 1e-5, "seed": 17, "token_budget": 5200,
                       "schedule_spec": "cosine:0.03", "beta2": .999, "reference_digest": DIGEST}.items():
        require(failure.get(key) == value, f"canonical failure recipe {key} mismatch")


def _validate_save_failure(metadata, canonical):
    first_retry = SUPPORTED_RETRIES[canonical][0]
    for key, value in {"previous_tag": first_retry, "no_endpoint_observed": False, "completed_training": True,
                       "failure_kind": "checkpoint_save", "retry_reason": "checkpoint_save_resource_failure_only",
                       "known_score_is_not_retry_criterion": True}.items():
        require(metadata.get(key) == value, f"save failure retry metadata {key} mismatch")
    path = REPO / "experiments/rab001_exploration/outputs/R4_RETRY1_SAVE_FAILURE_20260913.json"
    require((REPO / metadata["failure_artifact_path"]).resolve() == path.resolve(), "save failure observation path mismatch")
    require(sha256(path) == metadata["failure_artifact_sha256"], "save failure observation hash mismatch")
    snapshot = read_json(path)
    for key, value in {"schema": 1, "canonical_tag": canonical, "tag": first_retry, "run_label": "retry1",
                       "completed_training": True, "checkpoint_saved": False, "overall_completed": False,
                       "final_endpoint_observed": True, "implementation_digest": DIGEST}.items():
        require(snapshot.get(key) == value, f"save failure observation {key} mismatch")
    source_files = snapshot["source_files"]
    expected_names = {"start": "START.json", "initial_probe": "probe_0000000.json", "final_probe": "probe_0005200.json", "failure": "failure.json"}
    require(set(source_files) == set(expected_names), "save failure source-file inventory mismatch")
    parents = set()
    for name, binding in source_files.items():
        source_path = (REPO / binding["path"]).resolve()
        require(source_path.name == expected_names[name] and source_path.parent.parent == (BASE / "outputs/arm-runs").resolve()
                and source_path.parent.name.startswith(first_retry + "_"), "save failure source path mismatch")
        parents.add(source_path.parent)
        require(sha256(source_path) == binding["sha256"], f"save failure source-file hash mismatch: {name}")
        require(read_json(source_path) == snapshot["source_records"][name], f"save failure embedded record mismatch: {name}")
    require(len(parents) == 1, "save failure records came from different runs")
    start, final, failure = (snapshot["source_records"][name] for name in ("start", "final_probe", "failure"))
    require(start["tag"] == first_retry and start["canonical_tag"] == canonical and start["implementation_digest"] == DIGEST,
            "save failure START identity mismatch")
    require(failure["tag"] == first_retry and failure["completed"] is False and "CPU checkpoint copy" in failure["message"],
            "failure did not arise from the observed checkpoint copy guard")
    require(metadata["failure_reason"] == snapshot["failure_reason"] == failure["message"], "save failure reason mismatch")
    require(final["tokens"] == snapshot["supervised_target_tokens"] == metadata["observed_supervised_target_tokens"] == 5233
            and final["steps"] == snapshot["optimizer_steps"] == metadata["observed_optimizer_steps"] == 41
            and final["milestone"] == 5200, "save failure completed dose mismatch")
    for key, value in metadata["observed_final_summary"].items():
        require(snapshot["final_summary"].get(key) == value, f"known final observation mismatch: {key}")
    require(set(metadata["observed_final_summary"]) == {"correct_total", "retained_total", "gained_total", "lost_total", "gained_by_stratum", "lost_by_stratum"},
            "known final observation inventory mismatch")
    require(not (EVIDENCE / f"ARM_DOSE_{first_retry}.json").exists(), "retry1 unexpectedly has a successful ARM_DOSE artifact")
    require(not (BASE / "outputs/arm-checkpoints" / first_retry).exists(), "retry1 unexpectedly has saved checkpoint files")


def validate_run_registry(registry):
    """Validate the exact declared retry chain, retaining both different failure kinds."""
    require(registry.get("schema") == 1, "unsupported run registry schema")
    aliases, retries = registry.get("active_tags"), registry.get("retries")
    require(isinstance(aliases, dict) and isinstance(retries, dict), "registry requires active_tags and retries mappings")
    expected_retries = set()
    for canonical, active in aliases.items():
        allowed = SUPPORTED_RETRIES.get(canonical, ())
        require(active in allowed, f"unsupported registry alias: {canonical} -> {active}")
        chain = allowed[:allowed.index(active) + 1]
        expected_retries.update(chain)
        require(set(chain) <= set(retries), "registry aliases and retry chain metadata disagree")
        for number, tag in enumerate(chain, 1):
            metadata = retries[tag]
            for key, value in {"canonical_tag": canonical, "run_label": f"retry{number}", "retry_scope": "storage_only",
                               "checkpoint_saved": False, "preserve_failure_bytes": True}.items():
                require(metadata.get(key) == value, f"retry metadata {key} mismatch")
            if number == 1:
                _validate_initial_failure(metadata, canonical)
            else:
                _validate_save_failure(metadata, canonical)
                previous = registry.get("previous_registry", {})
                previous_path = REPO / "experiments/rab001_exploration/outputs/RUN_REGISTRY_BEFORE_RETRY2_20260913.json"
                require((REPO / previous.get("path", "")).resolve() == previous_path.resolve()
                        and sha256(previous_path) == previous.get("sha256"), "previous registry snapshot binding mismatch")
                original = read_json(previous_path)
                require(original["active_tags"] == {canonical: chain[0]} and original["retries"][chain[0]] == retries[chain[0]],
                        "original failure registry was not preserved")
    require(set(retries) == expected_retries, "registry aliases and retry metadata disagree")


def load_run_registry(path=RUN_REGISTRY):
    require(Path(path).is_file(), f"run registry unavailable: {path}")
    registry = read_json(path)
    validate_run_registry(registry)
    return {**registry, "registry_path": str(Path(path).resolve()), "registry_sha256": sha256(path)}


def resolve_run(canonical_tag, evidence=EVIDENCE, registry=None):
    """Resolve one declared tag; never enumerate files or choose the latest result."""
    registry = load_run_registry() if registry is None else registry
    active = registry["active_tags"].get(canonical_tag, canonical_tag)
    failure = None
    failures = []
    if active != canonical_tag:
        allowed = SUPPORTED_RETRIES.get(canonical_tag, ())
        require(active in allowed, "unsupported active run alias")
        failure = registry["retries"].get(active)
        label = active[len(canonical_tag) + 1:]
        require(failure is not None and failure.get("canonical_tag") == canonical_tag
                and failure.get("run_label") == label, "active alias missing or mismatched retry metadata")
        failures = [{"retry_tag": tag, **registry["retries"][tag]} for tag in allowed[:allowed.index(active) + 1]]
    return {"canonical_tag": canonical_tag, "active_tag": active,
            "artifact_path": str(evidence / f"ARM_DOSE_{active}.json"), "preserved_failure": failure, "preserved_failures": failures}


def validate_run_identity(body, selection):
    canonical, active = selection["canonical_tag"], selection["active_tag"]
    if active == canonical:
        require(body.get("canonical_tag", canonical) == canonical and body.get("run_label") in (None, ""),
                "unregistered or mismatched retry identity")
    else:
        require(active in SUPPORTED_RETRIES.get(canonical, ()) and selection.get("preserved_failure") is not None,
                "retry alias absent from registry")
        require(body.get("canonical_tag") == canonical and body.get("run_label") == active[len(canonical) + 1:],
                "retry body canonical_tag/run_label mismatch")


def mcnemar_exact(b, c):
    require(type(b) is int and type(c) is int and b >= 0 and c >= 0,
            "discordances must be nonnegative integers")
    n = b + c
    return min(Fraction(1), Fraction(2 * sum(comb(n, k) for k in range(min(b, c) + 1)), 2 ** n))


def holm(pvalues):
    """Exact rational Holm adjustment, preserving every supplied family member."""
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted, previous = {}, Fraction(0)
    for rank, key in enumerate(ordered):
        require(0 <= pvalues[key] <= 1, "p must lie in [0,1]")
        previous = min(Fraction(1), max(previous, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = previous
    return adjusted


def make_context(panel, frozen, headroom, labels, policy):
    ids = [row["id"] for row in panel]
    require(len(ids) == 150 and len(set(ids)) == 150, "panel must contain exactly 150 unique IDs")
    classes = {row["id"]: row["reply_class"] for row in panel}
    require(set(classes.values()) == set(policy["probe_classes"]), "panel classes differ from policy")
    require(policy["bare_policy"] == "permissive", "registered policy is not permissive")
    require(headroom["reference_digest"] == DIGEST, "headroom digest mismatch")
    source = {row["id"]: {"text": row["source"]["raw"], "terminated": row["source"]["terminated"]}
              for row in frozen["rows"]}
    require(len(frozen["rows"]) == 150 and set(source) == set(ids), "frozen source IDs mismatch")
    targets = retention.build_overlay(labels, "permissive")["targets"]
    require(set(targets) == set(ids), "label-overlay IDs mismatch")
    correct = {rid: bool(retention.score_prediction(out["text"], out["terminated"], targets[rid])["correct"])
               for rid, out in source.items()}
    require({rid for rid in correct if correct[rid]} == set(frozen["reference_ids_by_policy"]["permissive"]),
            "source re-score differs from frozen reference")
    strata = {}
    for name, members in headroom["policies"]["permissive"]["ids"].items():
        require(name in ("wrong_answer", "format_only", "runaway"), "unknown headroom stratum")
        for rid in members:
            require(rid not in strata, "duplicate/overlapping headroom IDs")
            strata[rid] = name
    require(set(strata) == {rid for rid in correct if not correct[rid]}, "headroom does not exactly partition source failures")
    require(sum(correct.values()) == 47, "frozen permissive reference must contain 47 successes")
    return {"ids": set(ids), "classes": classes, "source": source, "source_correct": correct,
            "strata": strata, "targets": targets, "policy": policy}


def summarize_checkpoint(checkpoint, context, require_outputs=False):
    records = checkpoint["retention"]
    correct = records["per_row"]
    require(set(correct) == context["ids"] and all(type(ok) is bool for ok in correct.values()),
            "checkpoint must have exactly the same 150 boolean correctness IDs")
    outputs = checkpoint.get("outputs")
    require(not require_outputs or outputs is not None, "endpoint has no saved outputs")
    if outputs is not None:
        require(set(outputs) == context["ids"], "output IDs differ from panel")
        for rid, out in outputs.items():
            require(isinstance(out.get("text"), str) and type(out.get("terminated")) is bool,
                    f"invalid output record: {rid}")
            actual = retention.score_prediction(out["text"], out["terminated"], context["targets"][rid])
            require(actual["correct"] == correct[rid], f"saved correctness disagrees with output re-score: {rid}")
    source = context["source_correct"]
    gains = sorted(rid for rid in correct if correct[rid] and not source[rid])
    losses = sorted(rid for rid in correct if not correct[rid] and source[rid])
    totals = {"correct_total": sum(correct.values()), "retained_total": sum(source.values()) - len(losses),
              "lost_total": len(losses), "gained_total": len(gains), "reference_n": sum(source.values())}
    declared = records["permissive"]
    for key, value in totals.items():
        require(declared.get(key) == value, f"stored {key} differs from recomputed count")
    classes = {}
    for name in context["policy"]["probe_classes"]:
        members = {rid for rid, cls in context["classes"].items() if cls == name}
        count = {"rows": len(members), "reference_n": sum(source[rid] for rid in members),
                 "retained": sum(source[rid] and correct[rid] for rid in members),
                 "lost": len(members & set(losses)), "gained": len(members & set(gains)),
                 "arm_correct": sum(correct[rid] for rid in members),
                 "source_correct_terminated": sum(source[rid] for rid in members)}
        count["terminated"] = (sum(outputs[rid]["terminated"] for rid in members) if outputs is not None
                               else declared["per_class"][name]["terminated"])
        require(type(count["terminated"]) is int and 0 <= count["terminated"] <= len(members),
                "invalid per-class termination count")
        for key, value in count.items():
            require(declared["per_class"][name].get(key) == value, f"stored {name}/{key} mismatch")
        classes[name] = count
    totals["terminated_total"] = sum(c["terminated"] for c in classes.values())
    require(declared["terminated_total"] == totals["terminated_total"], "termination total mismatch")
    verdict = retention.monitor_verdict(classes, context["policy"], context["policy"]["probe_classes"])
    require(declared["status"] == verdict["status"], "stored retention status mismatch")
    require(sorted(declared["failed_classes"]) == sorted(verdict["failed_classes"]), "stored failed classes mismatch")
    changed = (sum(outputs[rid]["text"] != context["source"][rid]["text"] for rid in outputs)
               if outputs is not None else checkpoint["answers_changed_vs_source"])
    require(changed == checkpoint["answers_changed_vs_source"], "stored answer-change count mismatch")
    gain_strata = {name: sum(context["strata"][rid] == name for rid in gains)
                   for name in ("wrong_answer", "format_only", "runaway")}
    loss_records = []
    for rid in losses:
        entry = {"id": rid, "reply_class": context["classes"][rid]}
        if outputs is not None:
            out = outputs[rid]
            scored = retention.score_prediction(out["text"], out["terminated"], context["targets"][rid])
            category = headroom_audit.failure_stratum(out["text"], out["terminated"], context["targets"][rid])
            require(category is not None, "lost row unexpectedly accepted by failure classifier")
            entry.update(out, scorer_reason=scored["reason"], stratum=category)
        else:
            entry["stratum"] = "output_unavailable"
        loss_records.append(entry)
    loss_counts = Counter(entry["stratum"] for entry in loss_records)
    loss_strata = {name: loss_counts[name] for name in ("wrong_answer", "format_only", "runaway")}
    if loss_counts["output_unavailable"]:
        loss_strata["output_unavailable"] = loss_counts["output_unavailable"]
    return {**totals, "status": verdict["status"], "failed_classes": verdict["failed_classes"],
            "per_class": classes, "net": len(gains) - len(losses), "gained_ids": gains,
            "gained_by_stratum": gain_strata, "lost_ids": losses, "loss_records": loss_records,
            "lost_by_stratum": loss_strata,
            "loss_strata_status": "COMPLETE" if not loss_counts["output_unavailable"] else "UNAVAILABLE: milestone outputs were not saved",
            "answers_changed_vs_source": changed, "output_rescored": outputs is not None}


def validate_arm(body, arm, context):
    for key, value in {"arm": arm, "completed": True, "aborted_reason": None, "lr": 1e-5,
                       "token_budget": 5200, "seed": 17, "schedule_spec": "cosine:0.03", "beta2": .999,
                       "train_device": "cuda", "probe_device": "cpu", "reference_digest": DIGEST,
                       "device_integrity_control_passed": True, "registered_bare_policy": "permissive"}.items():
        require(body.get(key) == value, f"{key} must equal {value!r}")
    require(body["numerics"] == {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                                 "float32_matmul_precision": "highest"}, "TF32/precision control mismatch")
    sequence = body["per_step_target_tokens"]
    require(sequence and all(type(n) is int and n > 0 for n in sequence), "invalid training token sequence")
    require(sum(sequence) == body["supervised_target_tokens"], "actual dose differs from token sequence")
    require(sum(sequence[:-1]) < 5200 <= sum(sequence), "training did not stop at first completed sequence crossing requested budget")
    require(len(sequence) == body["optimizer_steps"], "optimizer steps differ from token sequence")
    checkpoints = body["checkpoints"]
    require(len(checkpoints) >= 2, "missing initial/final endpoints")
    first, last = checkpoints[0], checkpoints[-1]
    require(first["tokens"] == 0 and first["steps"] == 0 and first["milestone"] == 0, "invalid initial endpoint")
    require(last["tokens"] == body["supervised_target_tokens"] and last["steps"] == body["optimizer_steps"]
            and last["milestone"] == 5200, "final endpoint does not match actual dose")
    require(first.get("outputs") == context["source"], "initial source text/termination integrity control mismatch")
    trajectory, previous = [], -1
    for index, point in enumerate(checkpoints):
        require(type(point["tokens"]) is int and point["tokens"] > previous, "checkpoint doses not strictly increasing")
        require(0 <= point["steps"] <= len(sequence) and point["tokens"] == sum(sequence[:point["steps"]]),
                "checkpoint dose differs from its step prefix")
        previous = point["tokens"]
        result = summarize_checkpoint(point, context, index in (0, len(checkpoints) - 1))
        trajectory.append({"tokens": point["tokens"], "steps": point["steps"], "milestone": point["milestone"], **result})
    endpoint = trajectory[-1]
    headline = body["headline"]
    for key, value in {"correct_start": 47, "correct_end": endpoint["correct_total"],
                       "gained_end": endpoint["gained_total"], "lost_end": endpoint["lost_total"],
                       "retained_end": endpoint["retained_total"], "net_end": endpoint["net"],
                       "answers_changed_end": endpoint["answers_changed_vs_source"],
                       "status_start": "PASS", "status_end": endpoint["status"]}.items():
        require(headline.get(key) == value, f"headline {key} differs from recomputation")
    consumed_ids = body.get("train_row_ids_consumed")
    if consumed_ids is not None:
        require(isinstance(consumed_ids, list) and len(consumed_ids) == len(sequence)
                and all(isinstance(rid, str) and rid for rid in consumed_ids), "invalid consumed row-ID sequence")
        require(body.get("train_row_ids_sha256") == hashlib.sha256("\n".join(consumed_ids).encode()).hexdigest(),
                "consumed row-ID hash mismatch")
    if body.get("data_sha256") is not None:
        require(isinstance(body["data_sha256"], dict), "data_sha256 must be a mapping")
        for key, expected in context.get("current_data_sha256", {}).items():
            require(body["data_sha256"].get(key) == expected, f"recorded {key} hash differs from current frozen input")
    binding_names = ("training_data_sha256", "train_data_sha256", "train_file_sha256", "data_hashes", "data_sha256",
                     "consumed_train_row_ids", "train_row_ids_consumed", "consumed_row_ids_sha256",
                     "train_row_ids_sha256", "consumed_train_row_ids_sha256", "data_bindings",
                     "source_weights_sha256", "source_assets_sha256")
    bindings = {key: body[key] for key in binding_names if body.get(key) is not None}
    saved = body.get("checkpoint")
    require(isinstance(saved, dict), "registered final checkpoint metadata missing")
    require(saved["parameters"] == body["trainable_parameters"] and saved["saved_after_tokens"] == body["supervised_target_tokens"]
            and saved["saved_after_steps"] == body["optimizer_steps"], "saved checkpoint count/dose mismatch")
    is_complete_model = saved["parameters"] == body["parameters"]
    checkpoint_metadata = {"parameters": saved["parameters"], "complete_model_declared": saved["complete_model"],
                           "complete_model_expected_from_parameter_count": is_complete_model,
                           "warning": ("Historical complete_model flag contradicts partial trainable parameter inventory; source reconstruction is required. Historical bytes preserved."
                                       if saved["complete_model"] != is_complete_model else None),
                           "weight_bytes_or_hash_independently_checked": False}
    return {"status": "VALID", "parameters": body["parameters"], "trainable_parameters": body["trainable_parameters"],
            "actual_tokens": body["supervised_target_tokens"], "steps": body["optimizer_steps"],
            "seed": body["seed"], "token_sequence": sequence, "recorded_training_bindings": bindings,
            "training_identity_limit": "legacy token lengths and seed only" if not bindings else "recorded bindings compared where common; absent legacy hashes remain a limitation",
            "endpoint": endpoint, "trajectory": trajectory, "checkpoint_metadata": checkpoint_metadata,
            "_per_row": dict(last["retention"]["per_row"])}


def compare_pair(a, b):
    require(a["status"] == b["status"] == "VALID", "one or both arms unavailable")
    require(a["actual_tokens"] == b["actual_tokens"] and a["steps"] == b["steps"], "unmatched actual endpoint dose")
    require(a["seed"] == b["seed"] and a["token_sequence"] == b["token_sequence"], "unmatched seed/token sequence")
    ba, bb = a["recorded_training_bindings"], b["recorded_training_bindings"]
    shared_bindings = sorted(set(ba) & set(bb))
    for key in shared_bindings:
        require(ba[key] == bb[key], f"training binding differs: {key}")
    ra, rb = a["_per_row"], b["_per_row"]
    require(len(ra) == 150 and set(ra) == set(rb), "paired IDs differ; intersections are forbidden")
    wins = sum(ra[rid] and not rb[rid] for rid in ra)
    losses = sum(rb[rid] and not ra[rid] for rid in ra)
    return {"status": "AVAILABLE", "a_only_correct": wins, "b_only_correct": losses,
            "discordant": wins + losses, "rows_compared": 150,
            "matched_training_binding_fields": shared_bindings,
            "training_identity_limit": (None if all(key in shared_bindings for key in ("train_row_ids_consumed", "train_row_ids_sha256", "data_sha256"))
                                        else "some identity bindings are absent; seed and token lengths cannot prove row identity"),
            "_p": mcnemar_exact(wins, losses)}


def fixed_pair_report(arms):
    pairs, pvalues = {}, {}
    for a, b in PAIRS:
        key = f"{a}~{b}"
        try:
            pair = compare_pair(arms[a], arms[b])
        except (InvalidArtifact, KeyError) as exc:
            pair = {"status": "UNAVAILABLE", "reason": str(exc), "_p": Fraction(1)}
        pvalues[key] = pair.pop("_p")
        pairs[key] = pair
    adjusted = holm(pvalues)
    complete = all(pair["status"] == "AVAILABLE" for pair in pairs.values())
    for key, pair in pairs.items():
        p, q = pvalues[key], adjusted[key]
        pair.update(p_raw_exact=str(p), p_raw=float(p), p_holm_exact=str(q), p_holm=float(q),
                    p_is_missing_placeholder=pair["status"] != "AVAILABLE",
                    significance_status="COMPLETE_FAMILY" if complete else "WITHHELD_INCOMPLETE_FAMILY",
                    significant=complete and pair["status"] == "AVAILABLE" and q < ALPHA)
        pair["favours"] = (key.split("~")[0 if pair["a_only_correct"] > pair["b_only_correct"] else 1]
                           if pair["significant"] else None)
    return pairs


def build_report(evidence, context, registry=None):
    registry = load_run_registry() if registry is None else registry
    validate_run_registry(registry)
    arms = {}
    for arm in ARMS:
        selection = resolve_run(f"{arm}_lr1e-05_b5200_seed17", evidence, registry)
        path = Path(selection["artifact_path"])
        if not path.is_file():
            arms[arm] = {"status": "MISSING", "path": str(path), "run_selection": selection}
            continue
        fingerprint = sha256(path)
        try:
            body = read_json(path)
            validate_run_identity(body, selection)
            arms[arm] = validate_arm(body, arm, context)
        except (InvalidArtifact, KeyError, TypeError, ValueError) as exc:
            arms[arm] = {"status": "INVALID", "reason": str(exc)}
        arms[arm].update(path=str(path), sha256=fingerprint, run_selection=selection)
    pairs = fixed_pair_report(arms)
    complete = all(v["status"] == "AVAILABLE" for v in pairs.values())
    for arm in arms.values():
        arm.pop("_per_row", None)
    replication = {"status": "UNAVAILABLE", "historical": {"correct_total": 73, "retained_total": 41, "gained_total": 32, "lost_total": 6}}
    if arms["B28"]["status"] == "VALID":
        replication["observed"] = {key: arms["B28"]["endpoint"][key] for key in replication["historical"]}
        replication["status"] = "AGGREGATE_MATCH" if replication["observed"] == replication["historical"] else "AGGREGATE_DIFFERENCE"
    replication["limitation"] = "Historical row outputs and tensors were not saved; aggregate agreement cannot establish determinism or identify the historical trained weights."
    return {"schema": 1, "evidence_kind": "exploratory_fixed_depth_family_report", "family_status": "COMPLETE" if complete else "INCOMPLETE",
            "registration": "DISTINGUISHABILITY_PREREG.md continuation 2026-09-13 22:31 UTC",
            "reference_digest": DIGEST, "fixed_arms": list(ARMS), "fixed_test_count": 10, "alpha": str(ALPHA),
            "correction": "Holm over all ten fixed tests; missing/invalid pairs remain p=1 placeholders",
            "trajectory_policy": "All intermediate milestones are descriptive; final completed endpoint is fixed and never selected by score.",
            "full_B28_endpoint_limit": "Full B28 also trains embeddings and final norm; tail arms freeze them, so this is not a pure layer-count control.",
            "arms": arms, "pairs": pairs, "historical_B28_aggregate_replication": replication,
            "run_registry": registry,
            "loss_classifier": {"function": "headroom_audit.failure_stratum", "script_sha256": sha256(Path(headroom_audit.__file__))},
            "licenses_registered_training": False, "licenses_efficacy_claim": False, "licenses_release": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New create-only JSON report path")
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args(argv)
    require(not args.out.exists(), f"output already exists: {args.out}")
    panel_path = BASE / "data/monitor_probe.jsonl"
    panel = [json.loads(line) for line in panel_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    reference_path = BASE / f"outputs/retention-{DIGEST[:8]}/B28/retention.json"
    input_paths = {"panel": panel_path, "reference": reference_path,
                   "headroom": args.evidence_dir / "HEADROOM_AUDIT.json",
                   "labels": args.evidence_dir / "PROBE_LABEL_AUDIT.json", "protocol": BASE / "protocol.json"}
    context = make_context(panel, read_json(reference_path), read_json(input_paths["headroom"]),
                           read_json(input_paths["labels"]), read_json(input_paths["protocol"])["G1_RETENTION"])
    context["current_data_sha256"] = {"train": sha256(BASE / "outputs/splits_large/train.jsonl"),
                                      "dev": sha256(BASE / "outputs/splits_large/dev.jsonl"),
                                      "monitor_probe": sha256(panel_path), "probe_label_audit": sha256(input_paths["labels"])}
    report = build_report(args.evidence_dir, context)
    report["input_sha256"] = {key: sha256(path) for key, path in input_paths.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{report['family_status']}: {sum(p['status'] == 'AVAILABLE' for p in report['pairs'].values())}/10 comparisons available; {args.out}")
    return 2 if any(arm["status"] == "INVALID" for arm in report["arms"].values()) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InvalidArtifact as exc:
        raise SystemExit(str(exc))
