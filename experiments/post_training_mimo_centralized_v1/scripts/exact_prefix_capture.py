"""Independent exact-prefix controls for one frozen mechanical trace; stdlib only."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import greedy_capture as old

PACKAGE = Path(__file__).resolve().parents[1]
REL = "experiments/post_training_mimo_centralized_v1/"
CONFIG_RELATIVE = "configs/CAPTURE_EXACT_PREFIX_FIXED_TRACE_V1.json"
CONFIG_SHA256 = "b1b5348680f7c2bf09272ed81e504d95ae60cbbef97a85763f4ac4da9311cb9b"
PASS_STATUS = "PASS_NATIVE_EXACT_PREFIX_FIXED_TRACE_CAPTURE_MECHANICS_ONLY"
NEW_FILES = (REL + "scripts/exact_prefix_capture.py", REL + "scripts/qualify_exact_prefix.py",
             REL + "tests/test_exact_prefix_capture.py", REL + CONFIG_RELATIVE,
             REL + "AMENDMENT_022.md")

require = old.require
canonical = old.canonical
write = old.write
file_identity = old.file_identity


def load_config(path):
    require(Path(path).resolve() == (PACKAGE / CONFIG_RELATIVE).resolve(), "registered exact-prefix config path required")
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, "exact-prefix config drift")
    cfg = old.parse(raw)
    validate_plan(cfg)
    return cfg


def validate_plan(cfg):
    require(type(cfg["prompt_ids"]) is list and len(cfg["prompt_ids"]) == 128
            and old.token_hash(cfg["prompt_ids"]) == cfg["prompt_ids_sha256"], "frozen prompt identity")
    require(cfg["expected_generated_ids"] == [16, 22, 151645]
            and all(type(x) is int for x in cfg["expected_generated_ids"])
            and old.token_hash(cfg["expected_generated_ids"]) == cfg["expected_generated_ids_sha256"], "frozen target identity")
    require(type(cfg["generation_cap"]) is int and cfg["generation_cap"] == 3
            and cfg["offline_order"] == [2, 1, 0] and all(type(x) is int for x in cfg["offline_order"]), "frozen exact-prefix schedule")
    require(cfg["scoring"]["logprob_max_atol"] == .03125 and cfg["scoring"]["logprob_mean_atol"] == .01
            and cfg["scoring"]["finite_required"] is True and cfg["scoring"]["behavior_logprob_comparison"] is False,
            "unchanged raw-model likelihood tolerances required")
    require(cfg["ceilings"] == {"pretrained_loads": 1, "model_constructs": 1, "tensor_copies": 310,
            "forward_calls": 6, "forward_input_positions": 774, "logical_calls": 318, "generated_tokens": 3,
            "optimizer_steps": 0, "training_positions": 0, "teacher_calls": 0, "wall_seconds": 600}, "registered exact-prefix budgets required")
    require(all(cfg[key] is False for key in ("general_capture_qualified", "DEV_collection_authorized",
            "language_grading_authorized", "E1_authorized", "CHECK_FINAL_access")), "exact-prefix authority cannot expand")


def prerequisite_inputs(root, cfg):
    for key in ("captured_registration", "failed_trace", "causal_receipt"):
        require(file_identity(root / cfg[key])["sha256"] == cfg[key + "_sha256"], key + " byte drift")
    capture = old.parse((root / cfg["captured_registration"]).read_bytes())
    failed = old.parse((root / cfg["failed_trace"]).read_bytes())
    causal = old.parse((root / cfg["causal_receipt"]).read_bytes())
    first = failed["fixtures"][0]
    require(failed["status"] == "BLOCKED_CAPTURE_QUALIFICATION" and first["prompt_ids"] == cfg["prompt_ids"]
            and first["generated_ids"] == cfg["expected_generated_ids"] and first["likelihood"]["passed"] is False,
            "preserved failed trace required")
    require(first["online_model_logprobs"] == cfg["historical_online_model_logprobs"], "historical online trace drift")
    require(causal["status"] == "COMPLETE_CAPTURE_CAUSAL_POSTIMPORT_WAIT_DIAGNOSTIC_ONLY"
            and causal["code_sha"] == cfg["verified_base"] and causal["fixed_shape_causal_controls_passed"] is True
            and len(causal["comparisons"]) == 6 and all(x["passed"] and x["bitwise_equal"] for x in causal["comparisons"]),
            "passed native causal prerequisite required")
    return capture, first["likelihood"]


def source_snapshot(root, cfg, capture):
    pins = dict(cfg["reused_source_files"])
    for relative in (*NEW_FILES, cfg["captured_registration"], cfg["failed_trace"], cfg["causal_receipt"],
                     capture["primary_receipt"].replace("\\", "/"), capture["qualified_core_config"],
                     REL + capture["qualified_CE512_reference"]):
        pins[relative] = file_identity(root / relative)
    # The inherited helper verifies every dependency against clean current HEAD.
    return old.source_snapshot(root, {"reused_source_files": pins,
            "installed_runtime_source_files": capture["installed_runtime_source_files"]})


def prefix_inputs(prompt, generated, ordinal):
    old.token_hash(prompt)
    old.token_hash(generated)
    require(type(ordinal) is int and 0 <= ordinal < len(generated), "target ordinal outside saved trace")
    require(not any(token in (151645, 151643) for token in generated[:-1]), "saved target after terminal EOS")
    return list(prompt) + list(generated[:ordinal]), generated[ordinal]


def capture_trace(cfg, forward, record, trace):
    """Forward callback returns fresh measurements; no online scores enter replay.

    Online target=None is mandatory: the native callback chooses raw argmax.
    Offline receives only newly assembled input IDs and one saved target ID.
    The supplied trace retains partial completed evidence if a later step fails.
    """
    validate_plan(cfg)
    trace.update(generated_ids=[], online=[], offline=[], behavior_logprobs=[])
    prompt = list(cfg["prompt_ids"])
    for ordinal in range(cfg["generation_cap"]):
        ids = prompt + trace["generated_ids"]
        measured = forward(ids, None, "online_" + str(ordinal))
        require(measured["input_ids"] == ids and measured["input_ids_sha256"] == old.token_hash(ids), "online prefix identity")
        token = measured["target_id"]
        behavior = old.validate_choice(token, measured["selected_raw_logit"], measured["raw_max_logit"], measured["first_maximum_token_id"])
        require(old.likelihood_measurement([measured["model_logprob"]], [measured["model_logprob"]])["finite"], "online finite model likelihood")
        trace["online"].append(measured)
        trace["generated_ids"].append(token)
        trace["behavior_logprobs"].append(behavior)
        record("online_token", ordinal=ordinal, measurement=measured, behavior_logprob=behavior)
        require(token == cfg["expected_generated_ids"][ordinal], "native raw choice differs from frozen mechanical trace")
        if old.termination_after_emit(prompt, trace["generated_ids"], 3, [151645, 151643]) is not None:
            break
    require(trace["generated_ids"] == cfg["expected_generated_ids"]
            and old.validate_sequence(trace["generated_ids"], 3, [151645, 151643]) == "eos", "complete frozen natural EOS trace required")
    saved = list(trace["generated_ids"])
    record("online_trace_complete", generated_ids=saved, generated_ids_sha256=old.token_hash(saved))
    for ordinal in cfg["offline_order"]:
        ids, target = prefix_inputs(prompt, saved, ordinal)
        measured = forward(ids, target, "offline_" + str(ordinal))
        trace["offline"].append({"ordinal": ordinal, **measured})
        record("offline_target", ordinal=ordinal, measurement=measured)
        require(measured["input_ids"] == ids and measured["input_ids_sha256"] == old.token_hash(ids)
                and measured["target_id"] == target, "independent offline prefix/target identity")
    aligned = sorted(trace["offline"], key=lambda x: x["ordinal"])
    trace["online_model_logprobs"] = [x["model_logprob"] for x in trace["online"]]
    trace["offline_model_logprobs"] = [x["model_logprob"] for x in aligned]
    trace["likelihood"] = old.likelihood_measurement(trace["online_model_logprobs"], trace["offline_model_logprobs"])
    trace["historical_reproduction"] = old.likelihood_measurement(cfg["historical_online_model_logprobs"], trace["online_model_logprobs"])
    trace["raw_logits_bitwise_equal"] = [left["raw_logits_sha256"] == right["raw_logits_sha256"]
                                         for left, right in zip(trace["online"], aligned)]
    record("likelihood_comparison", likelihood=trace["likelihood"], historical_reproduction=trace["historical_reproduction"],
           raw_logits_bitwise_equal=trace["raw_logits_bitwise_equal"])
    require(trace["likelihood"]["passed"] and trace["historical_reproduction"]["passed"], "exact-prefix model likelihood qualification failed")
    return trace


def persist_outcome(output, result, accounting, wall_seconds):
    """Freeze one current snapshot for every terminal/failure cost receipt."""
    result["call_accounting"] = accounting.snapshot()
    result["wall_seconds"] = wall_seconds
    result["generation_tokens"] = len(result.get("trace", {}).get("generated_ids", []))
    result = old.safe_numbers(result)
    if result["status"] != PASS_STATUS:
        write(output / "FAILURE_RECEIPT.json", result)
    write(output / "CAPTURE_RESULT.json", result)
    write(output / "COST_LEDGER.json", {key: result[key] for key in ("status", "wall_seconds", "pretrained_loads",
            "generation_tokens", "optimizer_steps", "training_positions", "teacher_calls", "call_accounting")})
    return result


def recover_worker_costs(output, supervision, cfg, source, reason):
    """Reconstruct durable intents after a watchdog kill; never invent completion.

    Preserve original worker receipts. Recovery files share one reconstructed
    snapshot, and canonical missing receipts are filled with the same bytes.
    An incomplete final journal line is preserved as explicit uncertainty.
    """
    from .qualify_native import verify_call_journal
    journal = output / "EVENTS.jsonl"
    events = []
    raw = b""
    error = None
    tail = b""
    reconstructed = None
    try:
        if journal.exists():
            require(journal.stat().st_size <= 8 * 1024**2, "recovery journal exceeds 8MiB metadata bound")
            with journal.open("rb") as stream:
                raw = stream.read(8 * 1024**2 + 1)
            require(len(raw) <= 8 * 1024**2, "recovery journal grew beyond metadata bound")
            lines = raw.splitlines(keepends=True)
            if lines and not lines[-1].endswith(b"\n"):
                tail = lines.pop()
            events = [old.parse(line) for line in lines]
        reconstructed = verify_call_journal(events, maximum_optimizer_calls=0)
    except Exception as exc:
        error = str(exc)
    records = [x for x in events if type(x) is dict]
    online = [x["measurement"] for x in records if x.get("event") == "online_token"
              and type(x.get("measurement")) is dict and type(x["measurement"].get("target_id")) is int]
    attempted_online = sum(x.get("event") == "operation_attempt" and x.get("operation") == "forward"
                           and type(x.get("stage")) is str and x["stage"].startswith("online_") for x in records)
    result = {"schema_version": 1, "status": "BLOCKED_NATIVE_EXACT_PREFIX_SUPERVISOR_RECOVERY", "error": reason,
              "experiment_id": cfg["experiment_id"], "code_sha": source["code_sha"], "config_sha256": CONFIG_SHA256,
              "evidence_level": "DURABLE_JOURNAL_RECONSTRUCTION_ONLY", "call_accounting": reconstructed,
              "wall_seconds": None, "pretrained_loads": reconstructed["operations"]["checkpoint_load"]["returned"] if reconstructed is not None else None,
              "generation_tokens": len(online) if error is None else None, "generation_tokens_confirmed": len(online),
              "generation_tokens_attempted_upper_bound": attempted_online if error is None else None,
              "optimizer_steps": reconstructed["operations"]["optimizer_step"]["returned"] if reconstructed is not None else None,
              "training_positions": reconstructed["training_forward_input_positions_attempted"] if reconstructed is not None else None,
              "teacher_calls": 0, "journal_verified": error is None, "journal_exists": journal.exists(),
              "journal_sha256": hashlib.sha256(raw).hexdigest(), "journal_bytes": len(raw), "journal_error": error,
              "incomplete_tail_bytes": len(tail), "incomplete_tail_sha256": hashlib.sha256(tail).hexdigest() if tail else None,
              "pending_work_is_not_a_returned_or_failed_operation": True,
              "generation_count_uncertain": error is not None or bool(tail) or attempted_online != len(online),
              "trace": {"generated_ids": [x["target_id"] for x in online], "online": online},
              "general_capture_qualified": False, "DEV_collection_authorized": False, "language_grading_authorized": False,
              "E1_authorized": False, "CHECK_FINAL_access": False, "interpretation_limit": cfg["interpretation_limit"]}
    watchdog = supervision / "SUPERVISOR.json"
    if watchdog.exists():
        watchdog_result = old.parse(watchdog.read_bytes())
        result["wall_seconds"] = watchdog_result.get("elapsed_seconds")
        result["watchdog"] = watchdog_result
    recovery = supervision / "worker_recovery"
    recovery.mkdir(exist_ok=False)
    terminal = canonical(old.safe_numbers(result)) + b"\n"
    ledger = canonical({key: result[key] for key in ("status", "wall_seconds", "pretrained_loads", "generation_tokens",
            "optimizer_steps", "training_positions", "teacher_calls", "call_accounting")}) + b"\n"
    output.mkdir(parents=True, exist_ok=True)
    for directory in (recovery, output):
        for name, data in (("FAILURE_RECEIPT.json", terminal), ("CAPTURE_RESULT.json", terminal), ("COST_LEDGER.json", ledger)):
            if directory == recovery or not (directory / name).exists():
                with (directory / name).open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
    return result
