"""Analysis-only receipt accounting; never loads models, weights or generations.

This file is separate from the frozen execution and scientific report files.
Complete reports require settled native/export/six-probe terminals. Failed or
unobserved attempts remain charged; incomplete accounting is printed, not saved
as a completed ledger. Worker wall time is not measured GPU-active time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ROOT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-integration001-20260920")
WALL_KEYS = ("model_worker_wall_seconds_and_conservative_gpu_seconds", "total_owned_work_wall_seconds")
LEGS = {"dose": (384, 128), "select": (384, 128), "retention": (150, 24),
        "long_context": (256, 24), "binding": (384, 24), "diagnostic": (896, 24)}
OUTPUT_ALLOWANCE = 2 * 2**20
MAX_RECEIPT_BYTES = 16 * 2**20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)) and path.stat().st_size <= MAX_RECEIPT_BYTES and
            path.suffix not in (".safetensors", ".pt", ".bin") and path.name != "GENERATIONS.json" and
            not ("eval" in path.parts and path.name in ("result.json", "RETENTION_GENERATOR_PREDICTIONS.json")),
            f"not a small regular cost receipt/code file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid wall seconds")
    return float(value)


def integer(value, label):
    require(type(value) is int and value >= 0, "invalid nonnegative integer: " + label)
    return value


def unseal(raw):
    body = {key: value for key, value in raw.items() if key != "sha256"}
    require(raw.get("sha256") == digest(body), "journal event seal differs")
    return body


def replay_events(paths, read, limits):
    """Keep active reservations; replace wall reservations only upon settlement."""
    require(isinstance(limits, dict) and bool(limits), "missing native limits")
    for key, value in limits.items():
        integer(value, key)
    used = {key: 0 for key in limits}; active = {}; started = set(); settlements = {}; previous = None
    for index, path in enumerate(paths):
        raw = read(path); event = unseal(raw)
        require(path.name == f"{index:06d}.json" and event["index"] == index and event["previous"] == previous,
                "native journal event order/chain differs")
        charges = event.get("charges", {})
        require(isinstance(charges, dict) and set(charges) <= set(limits), "unknown native charge field")
        for key, value in charges.items():
            used[key] += integer(value, key)
        if event["kind"] == "start":
            run = event["run_id"]
            require(isinstance(run, str) and run and not active and run not in started, "overlapping or reused native launch")
            require(all(key in charges for key in WALL_KEYS), "native start omitted wall reservation")
            active[run] = event; started.add(run)
        elif event["kind"] == "settle":
            run = event["run_id"]
            require(run in active and run not in settlements, "native settlement lacks a unique active launch")
            reservation = active.pop(run)
            for key in WALL_KEYS:
                used[key] += integer(event["actual_wall"][key], key) - reservation["charges"][key]
            settlements[run] = event
        else:
            require(event["kind"] == "charge", "unknown native budget event")
        previous = raw["sha256"]
    state = {"used_or_reserved": used, "active_reservations": sorted(active),
             "budget_breached": any(used[key] > limit for key, limit in limits.items()),
             "remaining": {key: max(0, limit - used[key]) for key, limit in limits.items()}}
    return state, started, settlements


def observed_worker_charge(supervisor):
    if (isinstance(supervisor, dict) and supervisor.get("status") in ("EXITED", "TIMEOUT_OR_INTERRUPTED") and
            type(supervisor.get("returncode")) is int and type(supervisor.get("wall_seconds")) in (int, float) and
            math.isfinite(supervisor["wall_seconds"]) and supervisor["wall_seconds"] >= 0):
        return float(supervisor["wall_seconds"]), "OBSERVED_SUPERVISOR_WALL_INCLUDING_FAILURE"
    return 900.0, "FULL_RESERVATION_WITHOUT_OBSERVED_TERMINAL"


def launch_identity(directory, supervisor, timeout, read):
    launch, spawn, claim = (read(directory / name) for name in ("launch.json", "spawn.json", "worker_claim.json"))
    require(Path(launch["out"]).resolve() == directory and launch["timeout_seconds"] == timeout,
            "supervised stage watchdog/output differs")
    require(spawn["launch_sha256"] == claim["launch_sha256"] == digest(launch) and
            spawn["supervisor_pid"] == claim["supervisor_pid"] == launch["supervisor_pid"] and
            spawn["child_pid"] == supervisor["child_pid"] and
            (claim["pid"] == spawn["child_pid"] or (claim.get("launcher_redirected") is True and claim["parent_pid"] == spawn["child_pid"])),
            "stage worker claim does not match its supervisor launch")
    return launch


def replay_steps(path, read_jsonl):
    progress = {"updates": 0, "processed_tokens": 0, "supervised_tokens": 0}; resources = []
    for step in read_jsonl(path):
        require(step["updates"] == progress["updates"] + 1 and step["processed_tokens"] > progress["processed_tokens"] and
                step["supervised_tokens"] > progress["supervised_tokens"], "optimizer cost log is discontinuous")
        progress = {key: integer(step[key], key) for key in progress}
        resources.append(step["resources"])
    return progress, resources


def resource_summary(samples):
    """Describe recorded observations, never infer device-wide or lifecycle peaks."""
    require(bool(samples), "resource observations absent")
    host = [integer(row["available_host_bytes"], "available host bytes") for row in samples]
    rss = [integer(row["rss_bytes"], "RSS bytes") for row in samples]
    gpu = [row["gpu0"] for row in samples if row.get("gpu0") is not None]
    result = {"recorded_samples": len(samples), "minimum_sampled_available_host_bytes": min(host),
              "maximum_sampled_process_rss_bytes": max(rss), "gpu_samples": len(gpu)}
    if gpu:
        result.update(minimum_sampled_gpu_free_bytes=min(integer(row["free"], "GPU free bytes") for row in gpu),
                      maximum_sampled_torch_allocated_bytes=max(integer(row["allocated"], "torch allocated") for row in gpu),
                      maximum_sampled_torch_reserved_bytes=max(integer(row["reserved"], "torch reserved") for row in gpu),
                      maximum_recorded_torch_peak_allocated_bytes=max(integer(row["peak"], "torch peak") for row in gpu))
    result["coverage"] = "pre-load snapshot and pre-update observations; torch peak is allocator telemetry at those samples, not a device-wide or final-stage peak"
    return result


def native_costs(root, envelope, read, read_jsonl):
    journal = root / "journal"; grant = read(journal / "grant.json")
    require(grant["limits"] == envelope["native_limits"] and grant["experiment"] == "BINDING-INTEGRATION-001",
            "native grant differs from registered integration envelope")
    require(grant == read(root / "GRANT.json"), "journal/root grant differ")
    paths = sorted((journal / "events").glob("*.json"))
    require(paths, "native budget journal absent")
    state, started, settlements = replay_events(paths, read, grant["limits"])
    runs = sorted((journal / "runs").iterdir())
    require({path.name for path in runs} == started == {"train-001"} and all(path.is_dir() and not path.is_symlink() for path in runs),
            "single registered native launch inventory differs")
    require(not state["active_reservations"] and not state["budget_breached"], "native journal active/breached; reservations remain charged")
    run = runs[0]; summary = read(run / "run_summary.json"); supervisor = read(run / "supervisor.json")
    require(summary["supervisor"] == supervisor and summary["budget_after"] == state, "native summary differs from replay")
    require(summary["experiment"] == "BINDING-INTEGRATION-001" and summary["run_id"] == run.name and
            type(summary["returncode"]) is int and summary["returncode"] == 0 and summary["worker_failed"] is False and
            supervisor["status"] == "EXITED" and type(supervisor["returncode"]) is int and supervisor["returncode"] == 0,
            "settled successful native terminal absent; failed charges remain in journal")
    require(not any((run / name).exists() for name in ("failure.json", "supervisor_failure.json")), "native failure receipt exists")
    wall = seconds(supervisor["wall_seconds"]); settlement = settlements[run.name]
    require(settlement["receipt_sha256"] == sha(run / "supervisor.json") and
            settlement["actual_wall"][WALL_KEYS[0]] == math.ceil(wall) and settlement["actual_wall"][WALL_KEYS[1]] >= math.ceil(wall),
            "native wall settlement differs from observed supervisor")
    launch_identity(run, supervisor, 2400, read)
    progress, resources = replay_steps(run / "steps.jsonl", read_jsonl)
    steps = read_jsonl(run / "steps.jsonl")
    result = read(run / "result.json"); before = read(run / "binding.json"); plan = read(root / "PLAN.json")
    require(result["record"] == "BINDING_INTEGRATION_TRAINING_RESULT" and result["status"] == "COMPLETE" and result["complete"] is True and
            progress == {"updates": 193, "processed_tokens": 577417, "supervised_tokens": 39502} and
            result["progress"] == {**progress, "next_group": 193}, "active optimizer inventory differs")
    require(result["native_clock_tokens"] == 380921 and
            [row["native_clock_tokens"] for row in steps] == plan["native_processed_cumulative"] and
            [row["lr"] for row in steps] == plan["applied_native_learning_rates"],
            "active native-only LR clock differs")
    require(result["binding"] == before["binding"] and digest(grant) == result["binding"]["grant_sha256"] and
            result["binding"]["plan_sha256"] == digest(plan) and result["steps_sha256"] == sha(run / "steps.jsonl"),
            "native result/plan/step cost binding differs")
    token_charges, call_charges = {}, {}
    native = read(run / "native_base_qualification.json")
    require(native["status"] == "PASS" and native["optimizer_steps"] == 0, "native source qualification failed")
    chat = [row for row in native["gradient_probes"] if row["probe"] == "native_chat"]
    require(len(chat) == 1 and integer(chat[0]["tokens"], "native chat tokens") == 31, "native qualification shape differs")
    token_charges["native_source"], call_charges["native_source"] = 2203, 16
    for name, filename in (("zero_B_adapter", "adapter_qualification.json"), ("final_merge", "merge_qualification.json")):
        value = read(run / filename)
        require(value["status"] == "PASS", name + " qualification failed")
        token_charges[name] = integer(value["budget_charges"]["processed_training_tokens_all_attempts"], name)
        call_charges[name] = integer(value["budget_charges"]["evaluation_and_qualification_model_calls"], name)
    require((token_charges["zero_B_adapter"], call_charges["zero_B_adapter"], token_charges["final_merge"], call_charges["final_merge"])
            == (2203, 26, 4390, 6), "source/adapter/merge qualification costs changed")
    zero_path = run / "zero_addend_qualification.json"; zero = read(zero_path)
    require(result["zero_addend_qualification_sha256"] == sha(zero_path) and Path(result["zero_addend_qualification_path"]).resolve() == zero_path,
            "zero-addend cost receipt binding differs")
    require(zero["record"] == "BINDING_INTEGRATION_ZERO_ADDEND_QUALIFICATION" and zero["status"] == "PASS" and zero["complete"] is True and
            zero["scope"] == "EXACT_TWO_UPDATE_PREFIX_ONLY" and zero["optimizer_step_count"] == 2 and
            len(zero["native_row_ids"]) == 28 and zero["binding_forward_calls"] == zero["generation_calls"] == 0,
            "two-update qualification accounting differs")
    require(zero["qualification_optimizer_discarded"] is True and zero["trainable_state_restored_exactly"] is True and
            zero["rng_restored_exactly"] is True and result["active_optimizer_started_fresh"] is True and
            result["active_initial_trainable_sha256"] == zero["restored_trainable_sha256"] and
            result["active_initial_rng_sha256"] == zero["rng_before_sha256"] == zero["rng_after_restore_sha256"],
            "qualification was not restored/discarded before a fresh active optimizer")
    token_charges["zero_addend_two_updates"] = integer(zero["budget_charges"]["processed_training_tokens_all_attempts"], "zero-addend tokens")
    call_charges["zero_addend_two_updates"] = integer(zero["budget_charges"]["evaluation_and_qualification_model_calls"], "zero-addend calls")
    require((token_charges["zero_addend_two_updates"], call_charges["zero_addend_two_updates"]) == (4032, 28),
            "zero-addend token/call inventory differs")
    used = state["used_or_reserved"]
    require(progress["processed_tokens"] + sum(token_charges.values()) == used["processed_training_tokens_all_attempts"] ==
            envelope["expected_native_charged_tokens"] == 590245 and sum(call_charges.values()) ==
            used["evaluation_and_qualification_model_calls"] == envelope["expected_native_qualification_calls"] == 76,
            "active/qualification decomposition does not reproduce the native ledger")
    require(used["model_worker_launches"] == 1 and used["retries"] == 0 and used["adaptive_model_selection_rounds"] == 1 and
            used["new_disk_bytes"] == 4 * 2**20 + 2 * (7255040 * 12 + 2**20) and used["generated_tokens_all_candidates"] == 0 and
            used["on_policy_rl_rollout_tokens"] == 0 and used["cloud_or_teacher_spend"] == 0, "unexpected native retry/generation/spend")
    return {"journal_state": state, "run": str(run), "actual_worker_wall_seconds": wall,
            "completed_optimizer_progress": progress, "native_learning_rate_clock_tokens": 380921,
            "active_optimizer_updates": 193, "qualification_optimizer_updates": 2, "total_executed_optimizer_updates": 195,
            "qualification_optimizer_discarded_and_initial_state_restored": True, "active_optimizer_started_fresh": True,
            "active_optimizer_data_tokens": 577417, "qualification_optimizer_data_tokens": 4032,
            "all_optimizer_data_tokens": 581449, "qualification_token_charges": token_charges, "qualification_call_charges": call_charges,
            "runtime": before["binding"]["runtime"], "resources": resource_summary([before["resources_before_load"], *resources]),
            "measured_gpu_active_seconds": None,
            "native_wall_charge_semantics": "Rounded observed supervisor wall, not measured GPU-active time. Qualification subsets reconcile the total; never add them twice."}



def preload_costs(root, envelope, read, read_jsonl):
    paths = [root / "PRELOAD_TERMINAL.json", *sorted((root / "preload-attempts").glob("*/PRELOAD_TERMINAL.json"))]
    require(paths[0].exists(), "initial preload terminal absent")
    attempts = []; dispatched = []
    for path in paths:
        value = read(path); samples = read_jsonl(path.parent / "preload_wait.jsonl")
        require(samples and all(seconds(row["elapsed_seconds"]) >= 0 for row in samples), "preload resource observations missing")
        dispatch = value.get("status") == "DISPATCHED_ONE_CONTROLLER"
        require(dispatch or (value.get("worker_launched") is False and value.get("controller_launched") is False), "preload terminal has ambiguous worker state")
        item = {"receipt": str(path), "status": value["status"], "wall_seconds": seconds(value["wall_seconds"]),
                "controller_dispatched": dispatch, "sampled_minimum_available_host_bytes": min(row["available_host_bytes"] for row in samples),
                "sampled_minimum_free_disk_bytes": min(row["free_disk_bytes"] for row in samples)}
        attempts.append(item)
        if dispatch:
            dispatched.append(path)
            command = read(path.parent / "TRAIN_COMMAND_TERMINAL.json")
            require(command["controller_pid"] == value["controller_pid"] and type(command["returncode"]) is int and command["returncode"] == 0,
                    "training foreground command did not exit successfully")
            require(value["dispatch_reservation_sha256"] == sha(root / "NATIVE_DISPATCH_RESERVATION.json") and
                    value["available_host_bytes"] >= envelope["pretrain_host_bytes"] and value["free_disk_bytes"] >= envelope["pretrain_free_disk_bytes"] and
                    value["lease_present"] is False, "dispatch reservation/resource admission differs")
    require(len(dispatched) == 1, "exactly one controller dispatch required")
    reservation = read(root / "NATIVE_DISPATCH_RESERVATION.json")
    require(reservation["native_owned_seconds_reserved"] == 2700 and reservation["deadline_unix"] == envelope["deadline_unix"] and
            reservation["prelaunch_binding_sha256"] == sha(root / "PRELAUNCH_BINDING.json"), "native dispatch reserve differs")
    return {"attempts": attempts, "wall_seconds": sum(row["wall_seconds"] for row in attempts),
            "included_in_session_elapsed": True, "not_added_again_to_session_elapsed": True}


def export_costs(root, envelope, read, partial=None):
    posttrain = root / "posttrain"; directories = [p for p in posttrain.iterdir() if p.is_dir() and p.name.startswith("build-")]
    require({p.name for p in directories} == {"build-integrated"}, "unexpected export attempt; all export attempts must be accounted")
    directory = directories[0]
    if partial is not None:
        partial.update(status="PARTIAL_NOT_COMPLETION_VERIFIED", attempts=1,
                       export_supervisor_wall_seconds=None,
                       export_worker_seconds_charged=envelope["cpu_export_seconds"],
                       accounting="FULL_EXPORT_RESERVATION_WITHOUT_OBSERVED_TERMINAL",
                       export_registered_watchdog_seconds=envelope["cpu_export_seconds"],
                       compression_registered_reservation_seconds=envelope["compression_seconds_each"],
                       compression_observed_wall_seconds=None, missing_terminal_is_not_zero_cost=True)
    supervisor = read(directory / "supervisor.json")
    if partial is not None and supervisor.get("status") in ("EXITED", "TIMEOUT_OR_INTERRUPTED") and type(supervisor.get("returncode")) is int:
        partial.update(export_supervisor_wall_seconds=seconds(supervisor["wall_seconds"]),
                       export_worker_seconds_charged=seconds(supervisor["wall_seconds"]),
                       accounting="OBSERVED_EXPORT_SUPERVISOR_INCLUDING_FAILURE")
    terminal = read(directory / "CPU_TERMINAL.json")
    compression = read(directory / "COMPRESSION.json")
    if partial is not None:
        partial["compression_observed_wall_seconds"] = seconds(compression["wall_seconds"])
    require(supervisor["status"] == "EXITED" and type(supervisor["returncode"]) is int and supervisor["returncode"] == 0 and
            terminal["status"] == "COMPLETE" and terminal["returncode"] == 0 and terminal["failure"] is None and terminal["model_execution"] is False,
            "CPU export lacks successful terminals")
    require(envelope["cpu_export_seconds"] == 600, "CPU export reservation differs")
    launch_identity(directory, supervisor, 600, read)
    require(type(compression["returncode"]) is int and compression["returncode"] == 0 and compression["model_execution"] is False and
            compression["before_sha256"] == compression["after_sha256"], "compression identity/terminal failed")
    arm = read(root / "arms/integrated/ARM.json"); material = read(root / "arms/integrated/MATERIALIZATION.json")
    require(compression["after_sha256"] == arm["arm_weight_sha256"] == material["out_weight_sha256"], "compression and export receipt identities differ")
    weight = root / "arms/integrated/model.safetensors"
    require(weight.is_file() and not weight.is_symlink() and weight.stat().st_size == compression["logical_bytes"], "exported logical size differs")
    worker, controller, compress = seconds(terminal["wall_seconds"]), seconds(supervisor["wall_seconds"]), seconds(compression["wall_seconds"])
    require(worker <= controller and worker <= envelope["cpu_export_seconds"] and compress <= envelope["compression_seconds_each"], "CPU stage exceeded registered wall bound")
    return {"attempts": 1, "export_worker_wall_seconds": worker, "export_supervisor_wall_seconds": controller,
            "compression_wall_seconds": compress, "weight_logical_bytes": compression["logical_bytes"],
            "available_host_bytes_before_export": arm["available_host_bytes_before"], "physical_free_bytes_after_compression": compression["free_disk_bytes"],
            "model_execution": False, "weight_contents_read": False, "overlap_note": "worker is inside supervisor; use supervisor plus compression, never sum worker and supervisor"}


def probe_costs(root, envelope, read, partial=None):
    """Charge every durable reservation, including failure/no-terminal attempts."""
    root = Path(root).resolve(); post = root / "posttrain"
    required = {"integrated-" + name for name in LEGS}
    excluded = {"PLAN.json", "TERMINAL.json", "DISPATCH_RESERVATION.json", "RECOVERY_PLAN.json", "RECOVERY_COMPLETE.json"}
    replacements = {}
    if (root / "ATTEMPT_MAP.json").exists():
        mapping = read(root / "ATTEMPT_MAP.json")
        require(mapping.get("schema") == 1 and isinstance(mapping.get("replacements"), dict), "invalid explicit retry map")
        replacements = mapping["replacements"]
    require(len(replacements) <= 1, "more than one global inference retry")
    for key, relative in replacements.items():
        require(key in required and Path(relative).parts == ("eval", key + "-retry1"), "unregistered retry path")
    allowed = required | {key + "-retry1" for key in replacements}
    receipt_paths = {p.stem: p for p in post.glob("*.json") if p.name not in excluded}
    reservation_paths = {p.stem: p for p in (post / "reservations").glob("*.json")}
    require(set(receipt_paths) <= allowed and set(reservation_paths) <= allowed, "unexpected six-leg attempt/reservation")
    require(set(receipt_paths) <= set(reservation_paths), "attempt has no prospective parent reservation")
    attempts, receipts, selected, cost_pins = [], {}, {}, {}
    if partial is not None:
        partial.update(status="PARTIAL_NOT_COMPLETION_VERIFIED", attempts=attempts)
    for slot, reservation_path in sorted(reservation_paths.items()):
        reserve = read(reservation_path)
        retry = slot.endswith("-retry1"); key = slot[:-7] if retry else slot
        suffix = key.removeprefix("integrated-"); n, maximum = LEGS[suffix]
        out = root / "eval" / slot
        require(reserve["suffix"] == suffix and reserve["attempt"] == (2 if retry else 1) and
                Path(reserve["out"]).resolve() == out and reserve["reserved_rows"] == n and
                reserve["reserved_generation_tokens"] == n * maximum and reserve["reserved_worker_seconds"] == 900 and
                reserve["session_envelope_sha256"] == sha(root / "SESSION_ENVELOPE.json"),
                "parent reservation geometry/envelope differs")
        value = read(receipt_paths[slot]) if slot in receipt_paths else None
        auxiliary = suffix in ("binding", "diagnostic")
        supervisor_path = out / ("supervisor.json" if auxiliary else "worker/supervisor.json")
        supervisor = read(supervisor_path) if supervisor_path.exists() else None
        charge, basis = observed_worker_charge(supervisor)
        code = value.get("returncode") if value else None
        item = {"leg": key, "attempt_slot": slot, "out": str(out), "returncode": code,
                "foreground_wall_seconds": seconds(value["wall_seconds"]) if value else None,
                "worker_seconds_charged": charge, "accounting": basis, "reserved_rows": n,
                "reserved_generation_tokens": n * maximum, "parent_reservation_sha256": sha(reservation_path)}
        attempts.append(item)
        totals = {"attempts": len(attempts), "successful_attempts": sum(x["returncode"] == 0 for x in attempts),
                  "failed_attempts": sum(type(x["returncode"]) is int and x["returncode"] != 0 for x in attempts),
                  "unobserved_parent_results": sum(x["returncode"] is None for x in attempts),
                  "worker_seconds_charged": sum(x["worker_seconds_charged"] for x in attempts),
                  "rows_reserved": sum(x["reserved_rows"] for x in attempts),
                  "generation_tokens_reserved": sum(x["reserved_generation_tokens"] for x in attempts)}
        if partial is not None:
            partial["observed_or_reserved_totals"] = dict(totals)
        if value is None:
            continue  # Reservation and any observed failed/successful worker remain charged.
        require(value["leg"] == key and Path(value["out"]).resolve() == out and type(code) is int and
                value["reserved_rows"] == n and value["reserved_generation_tokens"] == n * maximum and
                math.isclose(seconds(value["worker_seconds_charged"]), charge, rel_tol=0, abs_tol=1e-6),
                "attempt result omitted or changed failure/unobserved charges")
        receipts[slot] = value
        cost_pins[(post / (slot + ".json")).relative_to(root).as_posix()] = sha(receipt_paths[slot])
        if supervisor is not None:
            cost_pins[supervisor_path.relative_to(root).as_posix()] = sha(supervisor_path)
            require((supervisor.get("returncode") == 0) == (code == 0), "foreground/supervisor success states disagree")
        if code != 0:
            continue
        require(supervisor is not None and supervisor["status"] == "EXITED" and supervisor["returncode"] == 0, "successful leg has no observed successful supervisor")
        launch_identity(out if auxiliary else out / "worker", supervisor, 900, read)
        terminal_path = out / ("worker_terminal.json" if auxiliary else "worker/probe_terminal.json")
        terminal = read(terminal_path)
        require(terminal["status"] == ("COMPLETE" if auxiliary else "COMPLETED") and terminal["complete"] is True and terminal["failure"] is None,
                "successful leg has no complete worker terminal")
        require(terminal["returncode" if "returncode" in terminal else "system_exit_code"] == 0 if not auxiliary else True,
                "standard worker exit differs")
        filename = "AUX_BINDING.json" if auxiliary else "PROBE_WEIGHT_BINDING.json"; binding_path = out / filename
        binding = read(binding_path)
        require(binding["status"] == "COMPLETE" and binding["complete"] is True and
                binding["controller_returncode" if auxiliary else "returncode"] == 0 and
                binding["supervisor_sha256"] == sha(supervisor_path) and
                binding["worker_terminal_sha256" if auxiliary else "terminal_sha256"] == sha(terminal_path),
                "successful custody does not bind the cost terminals")
        if auxiliary:
            require(binding["worker_returncode"] == 0, "auxiliary worker did not exit successfully")
            local_path = root / "eval" / ("AUX_RESERVATION_" + slot + ".json")
            local = read(local_path)
            require(binding["reservation_sha256"] == sha(local_path) and local["parent_reservation_sha256"] == sha(reservation_path) and
                    local["rows"] == n and local["reserved_generated_tokens"] == n * maximum, "auxiliary local/global reservation differs")
        expected = root / replacements[key] if key in replacements else root / "eval" / key
        require(out == expected and key not in selected, "successful leg replaced without an explicit failed-attempt map")
        selected[key] = sha(binding_path)
    require(reservation_paths, "no probe reservations")
    expected_directories = set(reservation_paths)
    require({p.name for p in (root / "eval").iterdir() if p.is_dir()} == expected_directories, "probe output directory has no parent reservation")
    require(not (root / "POSTTRAIN_CONTROLLER_LEASE.json").exists() and not (root / "eval/AUX_CONTROLLER_LEASE.json").exists(),
            "posttraining/auxiliary admission lease is unsettled")
    for key, ceiling in (("attempts", "probe_workers_all_attempts"), ("failed_attempts", "probe_retries_total"),
                         ("worker_seconds_charged", "probe_worker_seconds_all_attempts"), ("rows_reserved", "probe_rows_reserved_all_attempts"),
                         ("generation_tokens_reserved", "probe_generation_tokens_reserved_all_attempts")):
        require(totals[key] <= envelope[ceiling], "global six-leg attempt budget exceeded")
    require(totals["unobserved_parent_results"] == 0 and len(receipts) == len(reservation_paths),
            "parent result missing; durable reservations retained in partial ledger")
    failures = [x for x in attempts if x["returncode"] != 0]
    require(set(selected) == required and len(attempts) == 6 + len(replacements) and
            len(failures) == len(replacements) and {x["leg"] for x in failures} == set(replacements),
            "six unique successful legs and all original failures required")
    original_path = post / "TERMINAL.json"; original = read(original_path); terminal = original
    require(isinstance(original.get("attempts"), list) and original["attempts"], "battery terminal omitted attempts")
    original_slots = [Path(x["out"]).name for x in original["attempts"]]
    require(len(original_slots) == len(set(original_slots)) and
            all(receipts.get(Path(x["out"]).name) == x for x in original["attempts"]), "battery changed/duplicated attempt records")
    if replacements:
        require(original["status"] == "FAILED_OR_BLOCKED" and set(replacements) <= set(original_slots), "retry lacks preserved original failure")
        terminal = read(post / "RECOVERY_COMPLETE.json")
        require(terminal["prior_terminal_sha256"] == sha(original_path) and terminal["attempt_map_sha256"] == sha(root / "ATTEMPT_MAP.json"),
                "recovery is not linked to its preserved terminal/map")
    require(terminal["status"] == "COMPLETE_SIX_PROBES" and terminal["final_binding_sha256"] == selected,
            "six-probe complete terminal/final bindings absent")
    require(isinstance(terminal.get("attempts"), list) and
            {Path(x["out"]).name: x for x in terminal["attempts"]} == receipts and len(terminal["attempts"]) == len(receipts),
            "final terminal must retain every exact attempt, including failures")
    # Exact schema produced by frozen report.probe_cost_accounting for normal completion.
    cost_accounting = {"probe_worker_seconds": totals["worker_seconds_charged"], "probe_rows_reserved": totals["rows_reserved"],
        "probe_generation_tokens_reserved": totals["generation_tokens_reserved"], "attempt_count": len(attempts),
        "successful_attempts": 6, "failed_attempts": len(failures),
        "individual_attempts": [{"leg": x["leg"], "out": x["out"], "returncode": x["returncode"],
            "worker_seconds_charged": x["worker_seconds_charged"], "reserved_rows": x["reserved_rows"],
            "reserved_generation_tokens": x["reserved_generation_tokens"]} for x in attempts],
        "cost_source_sha256": cost_pins}
    require(terminal["cost_accounting"] == cost_accounting, "terminal cost_accounting differs from independent attempt/supervisor hashes")
    for field in ("probe_worker_seconds", "probe_rows_reserved", "probe_generation_tokens_reserved"):
        require(math.isclose(seconds(terminal[field]), cost_accounting[field], rel_tol=0, abs_tol=1e-6), "terminal totals omitted or doubled costs")
    elapsed = seconds(terminal["session_elapsed_seconds"])
    require(elapsed <= envelope["total_owned_work_wall_seconds"], "completed probe phase exceeded overall envelope")
    return {"attempts": attempts, "totals": totals, "terminal_cost_accounting": cost_accounting,
            "session_elapsed_at_probe_completion_seconds": elapsed, "measured_gpu_active_seconds": None,
            "actual_generated_tokens": None, "generation_accounting": "Full per-attempt reservations, including failure/unobserved attempts; no generation body or scientific endpoint read."}



def storage_preparation_costs(root, envelope, read):
    """Existing-file compression is CPU/storage work, not a new training artifact."""
    stages = []
    for filename, status, change_key in (
        ("STORAGE_QUALIFICATION.json", "PASS_LOSSLESS_CHECKPOINT_COMPRESSION", "checkpoint_contents_changed"),
        ("SIDECAR_STORAGE_QUALIFICATION.json", "PASS_LOSSLESS_OWNED_SIDECAR_COMPRESSION", "contents_changed")):
        path = root / filename; value = read(path)
        require(value["schema"] == 1 and value["status"] == status and value["model_calls"] == 0 and
                value[change_key] is False and value["evidence_deleted"] is False, "storage preparation did not preserve content")
        inventory = value["file_sha256_before_after"]
        require(isinstance(inventory, dict) and len(inventory) == integer(value["files"], "compressed files") and bool(inventory) and
                all(isinstance(name, str) and isinstance(pin, str) and len(pin) == 64 and
                    all(ch in "0123456789abcdef" for ch in pin) for name, pin in inventory.items()),
                "storage preparation before/after hash inventory differs")
        if filename == "STORAGE_QUALIFICATION.json":
            require(isinstance(value["runs"], list) and len(value["runs"]) == 4 and
                    all(type(row["returncode"]) is int and row["returncode"] == 0 for row in value["runs"]),
                    "checkpoint compression command failed")
            commands = len(value["runs"])
        else:
            require(type(value["returncode"]) is int and value["returncode"] == 0, "sidecar compression command failed")
            commands = 1
        wall = seconds(value["wall_seconds"]); require(wall <= envelope["compression_seconds_each"], "storage compression bound exceeded")
        before = integer(value["volume_free_bytes_before"], "storage free bytes before")
        after = integer(value["volume_free_bytes_after"], "storage free bytes after")
        stages.append({"receipt": str(path), "receipt_sha256": sha(path), "status": status, "commands": commands,
            "files": len(inventory), "logical_existing_bytes_processed": integer(value["logical_bytes"], "existing logical bytes"),
            "wall_seconds": wall, "volume_free_bytes_before": before, "volume_free_bytes_after": after,
            "observed_volume_free_delta_bytes": after - before, "attributable_file_allocation_savings_bytes": None,
            "recorded_identity_inventory_sha256": digest(inventory), "payloads_rehashed_by_cost_report": False,
            "new_training_tokens": 0, "model_calls": 0, "new_artifact_logical_bytes_charge": 0})
    return {"stages": stages, "wall_seconds": sum(row["wall_seconds"] for row in stages),
            "logical_existing_bytes_processed": sum(row["logical_existing_bytes_processed"] for row in stages),
            "observed_volume_free_delta_bytes": sum(row["observed_volume_free_delta_bytes"] for row in stages),
            "new_artifact_logical_bytes_charge": 0, "model_calls": 0, "included_in_session_elapsed": True,
            "interpretation": "Recorded lossless before/after identities are audited as metadata only. Volume free-space deltas are observations, not isolated per-file allocation savings. Existing logical bytes are not added as new artifacts."}



def build_report(root=ROOT, *, now_unix=None):
    root = Path(root).resolve(); evidence = {}; costs = {}; blockers = []
    def read(path):
        path = Path(path); evidence[str(path)] = sha(path)
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        require(isinstance(value, dict), "cost receipt is not an object")
        return value
    def read_jsonl(path):
        path = Path(path); evidence[str(path)] = sha(path)
        with path.open(encoding="utf-8-sig") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    try:
        envelope = read(root / "SESSION_ENVELOPE.json"); frozen = read(root / "PRELAUNCH_BINDING.json")
        require(frozen["session_envelope_sha256"] == sha(root / "SESSION_ENVELOPE.json") and
                frozen["grant_sha256"] == sha(root / "GRANT.json") and
                all(frozen["artifact_files_sha256"][name] == sha(root / name) for name in ("STORAGE_QUALIFICATION.json", "SIDECAR_STORAGE_QUALIFICATION.json")),
                "cost envelope/grant/storage receipts differ from prelaunch freeze")
        require(envelope["experiment"] == "BINDING-INTEGRATION-001" and
                envelope["deadline_unix"] == envelope["session_start_unix"] + envelope["total_owned_work_wall_seconds"] and
                envelope["native_limits"]["model_worker_launches"] == 1 and envelope["native_limits"]["retries"] == 0 and
                envelope["probe_workers_all_attempts"] == 7 and envelope["probe_worker_seconds_all_attempts"] == 6300 and
                envelope["probe_rows_reserved_all_attempts"] == 3500 and envelope["probe_generation_tokens_reserved_all_attempts"] == 200000 and
                envelope["probe_retries_total"] == 1 and envelope["new_records_disk_bytes"] == 4 * 2**30 and envelope["free_disk_floor_bytes"] == 14 * 2**30,
                "registered integration envelope differs")
        # Preserve charged/reserved native totals even if a later success gate
        # fails; a blocked ledger must not make an observed failed attempt zero.
        if (root / "journal/grant.json").exists():
            journal_grant = read(root / "journal/grant.json")
            journal_paths = sorted((root / "journal/events").glob("*.json"))
            if journal_paths:
                costs["native_charged_or_reserved_before_success_gates"] = replay_events(journal_paths, read, journal_grant["limits"])[0]
        for name, operation in (("storage_preparation", lambda: storage_preparation_costs(root, envelope, read)),
                                ("preload", lambda: preload_costs(root, envelope, read, read_jsonl)),
                                ("native", lambda: native_costs(root, envelope, read, read_jsonl)),
                                ("export_and_compression", lambda: export_costs(root, envelope, read, partial=costs.setdefault("export_and_compression", {}))),
                                ("probes", lambda: probe_costs(root, envelope, read, partial=costs.setdefault("probes", {})))):
            try:
                costs[name] = operation()
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                blockers.append(name + ": " + str(exc))
        current = time.time() if now_unix is None else now_unix
        elapsed = seconds(current - envelope["session_start_unix"])
        costs["session_elapsed_at_analysis_seconds"] = elapsed
        require(current <= envelope["deadline_unix"], "analysis exceeded registered overall session deadline")
        files = list(root.rglob("*"))
        require(not any(path.is_symlink() for path in files), "disk accounting refuses symlinks")
        logical = sum(path.stat().st_size for path in files if path.is_file()); free = shutil.disk_usage(root).free
        costs["disk"] = {"logical_records_bytes_before_report": logical, "physical_volume_free_bytes_at_analysis": free,
                         "output_allowance_bytes": OUTPUT_ALLOWANCE, "weight_contents_read": False}
        require(logical + OUTPUT_ALLOWANCE <= envelope["new_records_disk_bytes"] and free - OUTPUT_ALLOWANCE >= envelope["free_disk_floor_bytes"], "cost report would exceed disk ceiling/floor")
        require(all(sha(path) == expected for path, expected in evidence.items()), "a cost receipt changed during analysis")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        blockers.append(str(exc))
    return {"schema": 1, "record": "BINDING_INTEGRATION_COST_REPORT", "status": "BLOCKED_COST_ACCOUNTING" if blockers else "COMPLETE_VERIFIED_COST_REPORT",
            "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "reporter_sha256": sha(Path(__file__)),
            "analysis_only_added_after_execution_freeze": True, "costs": costs, "blockers": blockers, "source_receipt_sha256": evidence,
            "scope_notes": ["Observed supervisor wall seconds include model loading, qualification, waiting and cleanup; they are not measured GPU-active seconds.",
                "Native rounded worker/owned charges come from the replayed journal. Optimizer and qualification figures reconcile this total and are not added again.",
                "All probe attempts, including failures, consume full row/generation-token reservations; without an observed terminal the full 900-second reservation remains charged.",
                "Export worker duration overlaps export supervisor duration. Compression is a separate CPU/filesystem stage.",
                "The two-update zero-addend qualification is charged separately, then its optimizer is discarded and initial trainable/RNG state restored. Active training starts fresh; this is not a warm start.",
                "Preparation compression processes existing checkpoint/sidecar files. Their logical bytes are not new artifact charges; observed volume free-space deltas do not isolate per-file allocation savings.",
                "Resource extrema cover recorded samples only. Torch allocator peak values are neither device-wide peaks nor complete lifecycle measurements.",
                "The receipt records elapsed wall through probe completion; elapsed at analysis also includes subsequent idle gaps and analysis. Neither is summed compute/GPU time; component durations overlap.",
                "Logical disk size includes all retained experiment artifacts, regardless of transparent compression; physical free space is a separate volume observation.",
                "This report reads small receipts and metadata only: no model weights, evaluation generations, endpoints, GPU runtime, or inferred monetary costs."],
            "model_execution": False, "source_weights_read": False, "quality_claim": False, "release_permitted": False}


def markdown(report):
    lines = ["# BINDING-INTEGRATION-001 cost report", "", "Status: **" + report["status"] + "**.", ""]
    if report["blockers"]:
        return "\n".join(lines + ["- " + v for v in report["blockers"]]) + "\n"
    costs = report["costs"]; native = costs["native"]; used = native["journal_state"]["used_or_reserved"]
    probes = costs["probes"]["totals"]; cpu = costs["export_and_compression"]
    rows = [("Native charged tokens including qualifications", used["processed_training_tokens_all_attempts"]),
            ("Active optimizer updates / zero-addend qualification updates", f"{native['active_optimizer_updates']} / {native['qualification_optimizer_updates']}"),
            ("Active optimizer-data / zero-addend optimizer-data tokens", f"{native['active_optimizer_data_tokens']} / {native['qualification_optimizer_data_tokens']}"),
            ("Native-only learning-rate clock tokens", native["native_learning_rate_clock_tokens"]),
            ("Qualification tokens / calls", f"{sum(native['qualification_token_charges'].values())} / {sum(native['qualification_call_charges'].values())}"),
            ("Native observed worker wall seconds", f"{native['actual_worker_wall_seconds']:.3f}"),
            ("Native rounded worker / owned wall charges", f"{used[WALL_KEYS[0]]} / {used[WALL_KEYS[1]]}"),
            ("Probe attempts / failures", f"{probes['attempts']} / {probes['failed_attempts']}"),
            ("Probe worker wall charges, seconds", f"{probes['worker_seconds_charged']:.3f}"),
            ("Probe row / generation-token reservations", f"{probes['rows_reserved']} / {probes['generation_tokens_reserved']}"),
            ("Export supervisor / worker wall seconds (overlap)", f"{cpu['export_supervisor_wall_seconds']:.3f} / {cpu['export_worker_wall_seconds']:.3f}"),
            ("Export compression wall seconds", f"{cpu['compression_wall_seconds']:.3f}"),
            ("Preparation compression wall seconds (existing files)", f"{costs['storage_preparation']['wall_seconds']:.3f}"),
            ("Preparation observed volume-free delta, bytes", costs["storage_preparation"]["observed_volume_free_delta_bytes"]),
            ("Session elapsed through analysis, seconds", f"{costs['session_elapsed_at_analysis_seconds']:.3f}"),
            ("Measured GPU-active time", "Not measured"),
            ("Logical records / physical free volume, GiB", f"{costs['disk']['logical_records_bytes_before_report']/2**30:.3f} / {costs['disk']['physical_volume_free_bytes_at_analysis']/2**30:.3f}")]
    lines += ["| Cost | Value |", "|---|---:|", *[f"| {name} | {value} |" for name, value in rows], "",
              "| Probe attempt | Return code | Worker wall charge, seconds | Rows reserved | Generation tokens reserved |", "|---|---:|---:|---:|---:|"]
    for item in costs["probes"]["attempts"]:
        lines.append(f"| {Path(item['out']).name} | {item['returncode']} | {item['worker_seconds_charged']:.3f} | {item['reserved_rows']} | {item['reserved_generation_tokens']} |")
    lines += ["", *["- " + value for value in report["scope_notes"]]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(); paths = args.root / "COST_REPORT.json", args.root / "COST_REPORT.md"
    require(not any(path.exists() for path in paths), "cost report outputs are create-only")
    value = build_report(args.root)
    if value["status"] != "COMPLETE_VERIFIED_COST_REPORT":
        print(json.dumps(value, indent=2, allow_nan=False)); return 2
    outputs = json.dumps(value, indent=2, allow_nan=False) + "\n", markdown(value)
    require(sum(len(text.encode()) for text in outputs) <= OUTPUT_ALLOWANCE, "cost report exceeds output reserve")
    for path, text in zip(paths, outputs):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(json.dumps({"status": value["status"], "outputs": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
