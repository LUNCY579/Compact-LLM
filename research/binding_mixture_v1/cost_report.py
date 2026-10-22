"""Analysis-only receipt accounting; never loads models, weights or generations.

This file is separate from the frozen execution and scientific report files.
Complete reports require settled native/export/five-probe terminals. Failed or
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

ROOT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-mixture001-20260920")
WALL_KEYS = ("model_worker_wall_seconds_and_conservative_gpu_seconds", "total_owned_work_wall_seconds")
LEGS = {"dose": (384, 128), "select": (384, 128), "retention": (150, 24),
        "long_context": (256, 24), "binding": (384, 24)}
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
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_RECEIPT_BYTES,
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
    require(grant["limits"] == envelope["native_limits"] and grant["experiment"] == "BINDING-MIXTURE-001", "native grant differs from registered envelope")
    require(grant == read(root / "GRANT.json"), "journal grant differs from frozen root grant")
    paths = sorted((journal / "events").glob("*.json"))
    require(paths, "native budget journal absent")
    state, started, settlements = replay_events(paths, read, grant["limits"])
    runs = sorted((journal / "runs").iterdir())
    require({path.name for path in runs} == started == {"train-001"} and all(path.is_dir() and not path.is_symlink() for path in runs),
            "native attempt inventory differs from the single registered launch")
    require(not state["active_reservations"] and not state["budget_breached"], "native journal is active or breached; reservations remain charged")
    run = runs[0]; summary = read(run / "run_summary.json"); supervisor = read(run / "supervisor.json")
    require(summary["supervisor"] == supervisor and summary["budget_after"] == state, "native summary does not reproduce journal/supervisor")
    require(summary["experiment"] == "BINDING-MIXTURE-001" and summary["run_id"] == run.name and
            type(summary["returncode"]) is int and summary["returncode"] == 0 and summary["worker_failed"] is False and
            supervisor["status"] == "EXITED" and type(supervisor["returncode"]) is int and supervisor["returncode"] == 0,
            "native success terminal absent; failed costs must remain in journal")
    require(not any((run / name).exists() for name in ("failure.json", "supervisor_failure.json")), "native failure receipt exists")
    wall = seconds(supervisor["wall_seconds"]); settlement = settlements[run.name]
    require(settlement["receipt_sha256"] == sha(run / "supervisor.json") and
            settlement["actual_wall"][WALL_KEYS[0]] == math.ceil(wall) and settlement["actual_wall"][WALL_KEYS[1]] >= math.ceil(wall),
            "native settlement is not bound to observed supervisor wall")
    launch_identity(run, supervisor, 2400, read)
    progress, resources = replay_steps(run / "steps.jsonl", read_jsonl)
    result = read(run / "result.json"); before = read(run / "binding.json")
    require(result["status"] == "COMPLETE" and result["complete"] is True and
            progress == {"updates": 296, "processed_tokens": 577417, "supervised_tokens": 39502} and
            result["progress"] == {**progress, "next_group": 296}, "completed optimizer inventory differs")
    require(result["binding"] == before["binding"] and digest(grant) == result["binding"]["grant_sha256"] and
            result["steps_sha256"] == sha(run / "steps.jsonl"), "native result/step cost provenance differs")
    qualification_tokens, qualification_calls = {}, {}
    native = read(run / "native_base_qualification.json")
    require(native["status"] == "PASS" and native["optimizer_steps"] == 0, "native source qualification did not pass")
    chat = [value for value in native["gradient_probes"] if value["probe"] == "native_chat"]
    require(len(chat) == 1, "native chat qualification accounting ambiguous")
    n = integer(chat[0]["tokens"], "native chat tokens")
    qualification_tokens["native_source"] = 5 * n + 2048
    qualification_calls["native_source"] = 5 + (n + 2) // 3
    for name, filename in (("zero_B_adapter", "adapter_qualification.json"), ("final_merge", "merge_qualification.json")):
        value = read(run / filename)
        require(value["status"] == "PASS", name + " qualification failed")
        qualification_tokens[name] = integer(value["budget_charges"]["processed_training_tokens_all_attempts"], name)
        qualification_calls[name] = integer(value["budget_charges"]["evaluation_and_qualification_model_calls"], name)
    used = state["used_or_reserved"]
    require(progress["processed_tokens"] + sum(qualification_tokens.values()) == used["processed_training_tokens_all_attempts"] == 586213 and
            sum(qualification_calls.values()) == used["evaluation_and_qualification_model_calls"] == 48,
            "optimizer/qualification cost decomposition does not reproduce native ledger")
    require(used["model_worker_launches"] == 1 and used["retries"] == 0 and used["generated_tokens_all_candidates"] == 0 and
            used["on_policy_rl_rollout_tokens"] == 0 and used["cloud_or_teacher_spend"] == 0, "unexpected native launches/retries/generation/spend")
    return {"journal_state": state, "run": str(run), "actual_worker_wall_seconds": wall,
            "completed_optimizer_progress": progress, "qualification_token_charges": qualification_tokens,
            "qualification_call_charges": qualification_calls, "runtime": before["binding"]["runtime"],
            "resources": resource_summary([before["resources_before_load"], *resources]),
            "measured_gpu_active_seconds": None, "native_wall_charge_semantics": "rounded supervisor wall; conservative GPU reservation accounting, not observed GPU-active time"}


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
    require({p.name for p in directories} == {"build-mixed"}, "unexpected export attempt; all export attempts must be accounted")
    directory = directories[0]
    supervisor = read(directory / "supervisor.json")
    if partial is not None:
        partial.update(status="PARTIAL_NOT_COMPLETION_VERIFIED", attempts=1,
                       export_supervisor_wall_seconds=seconds(supervisor["wall_seconds"]),
                       export_registered_watchdog_seconds=envelope["cpu_export_seconds"],
                       compression_registered_reservation_seconds=envelope["compression_seconds"],
                       compression_observed_wall_seconds=None, missing_terminal_is_not_zero_cost=True)
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
    arm = read(root / "arms/mixed/ARM.json"); material = read(root / "arms/mixed/MATERIALIZATION.json")
    require(compression["after_sha256"] == arm["arm_weight_sha256"] == material["out_weight_sha256"], "compression and export receipt identities differ")
    weight = root / "arms/mixed/model.safetensors"
    require(weight.is_file() and not weight.is_symlink() and weight.stat().st_size == compression["logical_bytes"], "exported logical size differs")
    worker, controller, compress = seconds(terminal["wall_seconds"]), seconds(supervisor["wall_seconds"]), seconds(compression["wall_seconds"])
    require(worker <= controller and worker <= envelope["cpu_export_seconds"] and compress <= envelope["compression_seconds"], "CPU stage exceeded registered wall bound")
    return {"attempts": 1, "export_worker_wall_seconds": worker, "export_supervisor_wall_seconds": controller,
            "compression_wall_seconds": compress, "weight_logical_bytes": compression["logical_bytes"],
            "available_host_bytes_before_export": arm["available_host_bytes_before"], "physical_free_bytes_after_compression": compression["free_disk_bytes"],
            "model_execution": False, "weight_contents_read": False, "overlap_note": "worker is inside supervisor; use supervisor plus compression, never sum worker and supervisor"}


def probe_costs(root, envelope, read, partial=None):
    posttrain = root / "posttrain"; excluded = {"PLAN.json", "TERMINAL.json", "RECOVERY_PLAN.json", "RECOVERY_COMPLETE.json"}
    replacements = {}
    if (root / "ATTEMPT_MAP.json").exists():
        mapping = read(root / "ATTEMPT_MAP.json")
        require(mapping.get("schema") == 1 and isinstance(mapping.get("replacements"), dict), "invalid explicit attempt map")
        replacements = mapping["replacements"]
    require(len(replacements) <= 1, "more than ONE inference retry across five legs")
    required = {"mixed-" + leg for leg in LEGS}
    for key, relative in replacements.items():
        require(key in required and Path(relative).parts == ("eval", key + "-retry1"), "unregistered replacement path")
    allowed = required | {key + "-retry1" for key in replacements}
    attempts, receipts, selected_bindings = [], {}, {}
    if partial is not None:
        partial.update(status="PARTIAL_NOT_COMPLETION_VERIFIED", attempts=attempts)
    for path in sorted(posttrain.glob("*.json")):
        if path.name in excluded:
            continue
        require(path.stem in allowed, "unexpected probe cost attempt receipt")
        value = read(path); key = value["leg"]; out = Path(value["out"]).resolve()
        require(key in required and out == root / "eval" / path.stem and path.stem in (key, key + "-retry1"), "probe receipt/directory identity differs")
        receipts[path.stem] = value
        n, maximum = LEGS[key.removeprefix("mixed-")]
        require(value["reserved_rows"] == n and value["reserved_generation_tokens"] == n * maximum and type(value["returncode"]) is int,
                "probe rows/token reservation or returncode differs")
        binding_leg = key == "mixed-binding"
        supervisor_path = out / ("supervisor.json" if binding_leg else "worker/supervisor.json")
        supervisor = read(supervisor_path) if supervisor_path.exists() else None
        charged, accounting = observed_worker_charge(supervisor)
        require(math.isclose(seconds(value["worker_seconds_charged"]), charged, rel_tol=0, abs_tol=1e-6), "failed or unobserved probe reservation omitted")
        success = value["returncode"] == 0
        attempts.append({"leg": key, "out": str(out), "returncode": value["returncode"],
                         "foreground_wall_seconds": seconds(value["wall_seconds"]), "worker_seconds_charged": charged,
                         "accounting": accounting, "reserved_rows": n, "reserved_generation_tokens": n * maximum})
        if partial is not None:
            partial["observed_receipt_totals"] = {
                "attempts": len(attempts), "reported_failed_attempts": sum(item["returncode"] != 0 for item in attempts),
                "worker_seconds_charged": sum(item["worker_seconds_charged"] for item in attempts),
                "rows_reserved": sum(item["reserved_rows"] for item in attempts),
                "generation_tokens_reserved": sum(item["reserved_generation_tokens"] for item in attempts)}
        if success:
            require(supervisor is not None and supervisor["status"] == "EXITED" and supervisor["returncode"] == 0, "successful probe lacks observed successful supervisor")
            launch_identity(out if binding_leg else out / "worker", supervisor, 900, read)
            terminal_path = out / ("worker_terminal.json" if binding_leg else "worker/probe_terminal.json")
            terminal = read(terminal_path)
            require(terminal["status"] == ("COMPLETE" if binding_leg else "COMPLETED") and terminal["complete"] is True and terminal["failure"] is None,
                    "successful probe worker terminal absent")
            filename = "BINDING.json" if binding_leg else "PROBE_WEIGHT_BINDING.json"
            binding = read(out / filename)
            require(binding["status"] == "COMPLETE" and binding["complete"] is True and
                    binding["controller_returncode" if binding_leg else "returncode"] == 0 and
                    binding["supervisor_sha256"] == sha(supervisor_path) and
                    binding["worker_terminal_sha256" if binding_leg else "terminal_sha256"] == sha(terminal_path), "probe completion receipt cost identities differ")
            expected = root / replacements[key] if key in replacements else root / "eval" / key
            require(out == expected and key not in selected_bindings, "successful result selection differs or duplicates one leg")
            selected_bindings[key] = sha(out / filename)
    require({p.name for p in (root / "eval").iterdir()} == set(receipts), "eval attempt directory has no cost receipt (or an extra receipt exists)")
    failures = [item for item in attempts if item["returncode"] != 0]
    require(set(selected_bindings) == required and len(attempts) == 5 + len(replacements) and
            len(failures) == len(replacements) and {item["leg"] for item in failures} == set(replacements), "five unique successes plus every original failure are required")
    totals = {"attempts": len(attempts), "successful_attempts": 5, "failed_attempts": len(failures),
              "worker_seconds_charged": sum(item["worker_seconds_charged"] for item in attempts),
              "rows_reserved": sum(item["reserved_rows"] for item in attempts),
              "generation_tokens_reserved": sum(item["reserved_generation_tokens"] for item in attempts)}
    for key, limit in (("attempts", "probe_workers_all_attempts"), ("failed_attempts", "probe_retries_total"),
                       ("worker_seconds_charged", "probe_worker_seconds_all_attempts"), ("rows_reserved", "probe_rows_reserved_all_attempts"),
                       ("generation_tokens_reserved", "probe_generation_tokens_reserved_all_attempts")):
        require(totals[key] <= envelope[limit], "global five-leg attempt budget exceeded")
    original_path = posttrain / "TERMINAL.json"; original = read(original_path); terminal = original
    require(isinstance(original.get("attempts"), list) and bool(original["attempts"]), "posttrain terminal omitted its attempts")
    for value in original["attempts"]:
        require(receipts.get(Path(value["out"]).name) == value, "original attempt receipt changed since terminal")
    original_names = [Path(value["out"]).name for value in original["attempts"]]
    require(len(original_names) == len(set(original_names)), "posttrain terminal duplicates an attempt")
    if replacements:
        require(original["status"] == "FAILED_OR_BLOCKED", "completed battery cannot replace a successful attempt")
        terminal = read(posttrain / "RECOVERY_COMPLETE.json")
        require(terminal.get("prior_terminal_sha256") == sha(original_path) and terminal.get("attempt_map_sha256") == sha(root / "ATTEMPT_MAP.json") and
                terminal.get("final_binding_sha256") == selected_bindings, "recovery completion does not bind preserved failure/map/final receipts")
        require(set(replacements) <= set(original_names), "original terminal omitted its retried failure")
    else:
        require(set(original_names) == set(receipts), "complete terminal omitted an attempt receipt")
    require(terminal["status"] == "COMPLETE_FIVE_PROBES", "complete five-probe terminal absent")
    for field, total in (("probe_worker_seconds", "worker_seconds_charged"), ("probe_rows_reserved", "rows_reserved"),
                         ("probe_generation_tokens_reserved", "generation_tokens_reserved")):
        require(math.isclose(seconds(terminal[field]), totals[total], rel_tol=0, abs_tol=1e-6), "posttrain terminal omits/double-counts costs")
    elapsed = seconds(terminal["session_elapsed_seconds"])
    require(elapsed <= envelope["total_owned_work_wall_seconds"], "completed session exceeded registered wall envelope")
    return {"attempts": attempts, "totals": totals, "session_elapsed_at_probe_completion_seconds": elapsed,
            "measured_gpu_active_seconds": None, "actual_generated_tokens": None,
            "generation_accounting": "full per-attempt reservations; generation bodies intentionally not read"}


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
                frozen["grant_sha256"] == sha(root / "GRANT.json"), "cost envelope/grant differs from prelaunch freeze")
        require(envelope["experiment"] == "BINDING-MIXTURE-001" and
                envelope["deadline_unix"] == envelope["session_start_unix"] + envelope["total_owned_work_wall_seconds"], "session deadline arithmetic differs")
        # Preserve charged/reserved native totals even if a later success gate
        # fails; a blocked ledger must not make an observed failed attempt zero.
        if (root / "journal/grant.json").exists():
            journal_grant = read(root / "journal/grant.json")
            journal_paths = sorted((root / "journal/events").glob("*.json"))
            if journal_paths:
                costs["native_charged_or_reserved_before_success_gates"] = replay_events(journal_paths, read, journal_grant["limits"])[0]
        for name, operation in (("preload", lambda: preload_costs(root, envelope, read, read_jsonl)),
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
    return {"schema": 1, "record": "BINDING_MIXTURE_COST_REPORT", "status": "BLOCKED_COST_ACCOUNTING" if blockers else "COMPLETE_VERIFIED_COST_REPORT",
            "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "reporter_sha256": sha(Path(__file__)),
            "analysis_only_added_after_execution_freeze": True, "costs": costs, "blockers": blockers, "source_receipt_sha256": evidence,
            "scope_notes": ["Observed supervisor wall seconds include model loading, qualification, waiting and cleanup; they are not measured GPU-active seconds.",
                "Native rounded worker/owned charges come from the replayed journal. Optimizer and qualification figures reconcile this total and are not added again.",
                "All probe attempts, including failures, consume full row/generation-token reservations; without an observed terminal the full 900-second reservation remains charged.",
                "Export worker duration overlaps export supervisor duration. Compression is a separate CPU/filesystem stage.",
                "Resource extrema cover recorded samples only. Torch allocator peak values are neither device-wide peaks nor complete lifecycle measurements.",
                "The receipt records elapsed wall through probe completion; elapsed at analysis also includes subsequent idle gaps and analysis. Neither is summed compute/GPU time; component durations overlap.",
                "Logical disk size includes all retained experiment artifacts, regardless of transparent compression; physical free space is a separate volume observation.",
                "This report reads small receipts and metadata only: no model weights, evaluation generations, endpoints, GPU runtime, or inferred monetary costs."],
            "model_execution": False, "source_weights_read": False, "quality_claim": False, "release_permitted": False}


def markdown(report):
    lines = ["# BINDING-MIXTURE-001 cost report", "", "Status: **" + report["status"] + "**.", ""]
    if report["blockers"]:
        return "\n".join(lines + ["- " + v for v in report["blockers"]]) + "\n"
    costs = report["costs"]; native = costs["native"]; used = native["journal_state"]["used_or_reserved"]
    probes = costs["probes"]["totals"]; cpu = costs["export_and_compression"]
    rows = [("Native charged tokens including qualifications", used["processed_training_tokens_all_attempts"]),
            ("Optimizer-data tokens", native["completed_optimizer_progress"]["processed_tokens"]),
            ("Qualification tokens / calls", f"{sum(native['qualification_token_charges'].values())} / {sum(native['qualification_call_charges'].values())}"),
            ("Native observed worker wall seconds", f"{native['actual_worker_wall_seconds']:.3f}"),
            ("Native rounded worker / owned wall charges", f"{used[WALL_KEYS[0]]} / {used[WALL_KEYS[1]]}"),
            ("Probe attempts / failures", f"{probes['attempts']} / {probes['failed_attempts']}"),
            ("Probe worker wall charges, seconds", f"{probes['worker_seconds_charged']:.3f}"),
            ("Probe row / generation-token reservations", f"{probes['rows_reserved']} / {probes['generation_tokens_reserved']}"),
            ("Export supervisor / worker wall seconds (overlap)", f"{cpu['export_supervisor_wall_seconds']:.3f} / {cpu['export_worker_wall_seconds']:.3f}"),
            ("Compression wall seconds", f"{cpu['compression_wall_seconds']:.3f}"),
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
