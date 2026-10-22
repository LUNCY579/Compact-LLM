"""Analysis-only three-model cost ledger; never imports a model or reads weights.

Run after the successful battery. Small frozen receipts and completed generation
records supply all costs. Unknown failed-attempt actuals remain null and their
full reservations stay charged. A blocked ledger is printed, never saved as a
completed report. This file is intentionally outside the execution freeze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

if __package__:
    from . import common as c
    from .paths import ROOT, HERE
else:
    import common as c
    from paths import ROOT, HERE

OUTPUT_ALLOWANCE = 2 * 2**20
ACTUAL_KEYS = ("rows", "generated_tokens_including_eos", "forward_calls", "processed_input_tokens")


def seconds(value):
    c.require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid observed wall seconds")
    return float(value)


def small_sha(path):
    path = c.absolute(path, file=True)
    c.require(path.stat().st_size <= 32 * 2**20 and path.suffix.lower() not in (".safetensors", ".pt", ".bin", ".ced"),
              "cost accounting permits small metadata/code/generation files only")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observed_wall(supervisor):
    if (supervisor is not None and supervisor.get("status") in ("EXITED", "TIMEOUT_OR_INTERRUPTED") and
            type(supervisor.get("returncode")) is int and type(supervisor.get("wall_seconds")) in (int, float) and
            math.isfinite(supervisor["wall_seconds"]) and supervisor["wall_seconds"] >= 0):
        return float(supervisor["wall_seconds"]), "OBSERVED_SUPERVISOR_WALL_INCLUDING_FAILURE"
    return 900.0, "FULL_RESERVATION_WITHOUT_OBSERVED_TERMINAL"


def census(plan, plan_path, grant_path, prefills, read):
    """Return every known charge even when inventory or an envelope fails."""
    out = c.absolute(plan["output_root"])
    totals = {"workers": 0, "total_worker_seconds": 0.0, "rows": 0, "reserved_generated_tokens": 0,
              "forward_calls": 0, "processed_input_tokens": 0, "retries": 0}
    attempts, blockers = [], []
    for path in sorted(out.glob("RESERVATION_*.json")):
        slot = path.stem.removeprefix("RESERVATION_")
        expected = {"rows": c.NROWS, "reserved_generated_tokens": c.NROWS * 24,
                    "forward_calls": c.NROWS * 24, "processed_input_tokens": prefills + c.NROWS * 24}
        item = {"slot": slot, "reservation_sha256": small_sha(path), "actual_budget_charges": None,
                "actuals_status": "UNKNOWN_FULL_RESERVATION_RETAINED", "reserved_worker_seconds": 900, **expected}
        attempts.append(item)
        supervisor_path = out / slot / "supervisor.json"
        supervisor = None
        if supervisor_path.is_file():
            try:
                supervisor = read(supervisor_path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                blockers.append(slot + ": unreadable supervisor; full reservation charged: " + str(exc))
        wall, label = observed_wall(supervisor)
        item.update(worker_seconds_charged=wall, worker_time_accounting=label)
        totals["workers"] += 1; totals["retries"] += int(slot.endswith("-retry1")); totals["total_worker_seconds"] += wall
        for key, value in expected.items():
            totals[key] += value
        try:
            reservation = read(path); role, attempt = reservation["model_role"], reservation["attempt"]
            c.require(slot == c.slot_name(role, attempt) and reservation["reserved_worker_seconds"] == 900 and
                      reservation["plan_sha256"] == small_sha(plan_path) and reservation["grant_sha256"] == small_sha(grant_path) and
                      all(type(reservation[key]) is int and reservation[key] == value for key, value in expected.items()),
                      "reservation differs from the frozen worker geometry")
            item.update(model_role=role, attempt=attempt)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            blockers.append(slot + ": " + str(exc))
    reserved = {item["slot"] for item in attempts}
    directories = {path.name for path in out.iterdir() if path.is_dir()} if out.exists() else set()
    if directories - reserved:
        blockers.append("Attempt directories without reservations; their actual costs are unknown: " + ", ".join(sorted(directories - reserved)))
    for key, value in totals.items():
        if value > c.LIMITS[key]:
            blockers.append("All-attempt budget exceeded: " + key)
    return {"totals": totals, "attempts": attempts, "blockers": blockers,
            "measured_gpu_active_seconds": None,
            "census_note": "Every recognizable reservation charges one full fixed row/token/forward/prefill allocation; malformed inventory blocks completion."}


def generation_costs(generations, inputs, reservation):
    """Recompute execution counters without decoding or scoring any answer."""
    predictions = generations["predictions"]; by_id = {row["id"]: row for row in predictions}
    original = {row["id"]: row for row in inputs}
    c.require(len(predictions) == len(by_id) == len(original) == c.NROWS and by_id.keys() == original.keys(), "generation accounting has missing/duplicate rows")
    eos = generations["eos_ids"]
    c.require(isinstance(eos, list) and bool(eos) and len(eos) == len(set(eos)) and all(type(t) is int for t in eos), "generation EOS accounting absent")
    for key, row in by_id.items():
        tokens = row["generated_ids_including_eos"]
        c.require(isinstance(tokens, list) and 1 <= len(tokens) <= 24 and all(type(t) is int and 0 <= t < 151936 for t in tokens) and
                  all(type(row[name]) is int for name in ("input_tokens", "generated_tokens", "forward_calls", "processed_input_tokens")) and
                  row["parent_id"] == original[key]["parent_id"] and row["input_tokens"] == len(original[key]["prompt_ids"]) and
                  row["generated_tokens"] == row["forward_calls"] == len(tokens) and
                  row["processed_input_tokens"] == row["input_tokens"] + len(tokens) - 1 and
                  type(row["terminated"]) is bool and row["terminated"] == (tokens[-1] in eos) and not any(t in eos for t in tokens[:-1]) and
                  (row["terminated"] or len(tokens) == 24), "row token/EOS/forward/prefill accounting differs")
    actual = {"rows": len(predictions), "generated_tokens_including_eos": sum(row["generated_tokens"] for row in predictions),
              "forward_calls": sum(row["forward_calls"] for row in predictions),
              "processed_input_tokens": sum(row["processed_input_tokens"] for row in predictions)}
    c.require(generations["actual_budget_charges"] == actual and generations["extra_warmup_or_qualification_calls"] == 0 and
              actual["rows"] <= reservation["rows"] and actual["generated_tokens_including_eos"] <= reservation["reserved_generated_tokens"] and
              all(actual[key] <= reservation[key] for key in ("forward_calls", "processed_input_tokens")), "actual model calls/tokens exceed or disagree with reservations")
    return actual


def verify_success(item, plan, plan_path, grant_path, freeze, inputs, read, bind_hash):
    slot = item["slot"]; role, attempt = item["model_role"], item["attempt"]
    out = Path(plan["output_root"]) / slot
    c.require(not any((out / name).exists() for name in ("FAILURE.json", "supervisor_failure.json")), "successful slot has a failure receipt")
    binding = read(out / "DIAGNOSTIC_BINDING.json")
    c.require(binding.get("status") == "COMPLETE" and binding.get("complete") is True and binding.get("failure") is None and
              binding.get("model_role") == role and binding.get("attempt") == attempt and
              type(binding.get("controller_returncode")) is int and binding["controller_returncode"] == 0 and binding.get("worker_returncode") == 0,
              "individual diagnostic controller did not complete successfully")
    for stem, filename in (("supervisor", "supervisor.json"), ("worker_terminal", "worker_terminal.json"), ("generations", "GENERATIONS.json")):
        c.require(bind_hash(out / filename) == binding[stem + "_sha256"], "completed cost artifact hash differs")
    supervisor, terminal, generations = (read(out / name) for name in ("supervisor.json", "worker_terminal.json", "GENERATIONS.json"))
    c.verify_terminal(supervisor, terminal)
    identity = c.verify_launch(out, plan_path, grant_path, plan, role, supervisor, attempt=attempt)
    c.require(all(binding.get(key) == value for key, value in identity.items()), "worker launch custody differs")
    for name in ("launch.json", "spawn.json", "worker_claim.json"):
        bind_hash(out / name)
    reservation = read(Path(plan["output_root"]) / ("RESERVATION_" + slot + ".json"))
    c.require(item["reservation_sha256"] == binding["reservation_sha256"], "successful binding reservation differs")
    for key, path in (("plan_sha256", plan_path), ("grant_sha256", grant_path)):
        c.require(binding[key] == terminal[key] == generations[key] == bind_hash(path), "successful worker belongs to another plan/grant")
    owner = reservation["controller_owner"]; launch = read(out / "launch.json")
    c.require(hashlib.sha256(c.canonical(owner) + b"\n").hexdigest() == reservation["controller_lease_sha256"] and
              owner["owner_pid"] == launch["supervisor_pid"] and owner["plan_sha256"] == bind_hash(plan_path) and
              owner["grant_sha256"] == bind_hash(grant_path), "reservation/worker controller ownership differs")
    before = generations["before"]
    c.require(before == generations["after"] == terminal["before"] == terminal["after"] == freeze["model_runtime_input_snapshots"][role] and
              before["model"]["weight_sha256"] == c.MODEL_SHAS[role] and before["runtime_versions"] == plan["runtime_versions"],
              "recorded runtime/input/model custody differs from the prelaunch snapshot")
    c.require(generations["record"] == "BINDING_DIAGNOSTIC_GENERATIONS" and generations["model_role"] == terminal["model_role"] == role and
              generations["rows"] == c.NROWS and generations["parent_worlds"] == 128 and generations["gold_file_read"] is False and
              generations["training_permission"] is False and generations["numerics"] == c.NUMERICS and
              terminal["generations_sha256"] == bind_hash(out / "GENERATIONS.json"), "generation scope or terminal charge identity differs")
    actual = generation_costs(generations, inputs, reservation)
    item.update(actual_budget_charges=actual, actuals_status="RECOMPUTED_COMPLETE_GENERATION_RECORD", returncode=0,
                worker_seconds_charged=seconds(supervisor["wall_seconds"]), binding_sha256=bind_hash(out / "DIAGNOSTIC_BINDING.json"),
                runtime=before["runtime_versions"])
    return binding


def build_report(root=ROOT):
    root = c.absolute(root, directory=True); evidence = {}; blockers = []; costs = {"all_attempts": {"totals": None, "attempts": [], "status": "NOT_YET_RECONCILED"}}
    def bind_hash(path):
        path = c.absolute(path, file=True); value = small_sha(path); evidence[str(path)] = value; return value
    def read(path):
        bind_hash(path); return c.read(path)
    try:
        c.require(not any(name in sys.modules for name in ("torch", "transformers", "tokenizers")), "cost report requires an offline model-free process")
        plan_path, grant_path = root / "PLAN.json", root / "GRANT.json"
        plan, grant = read(plan_path), read(grant_path)
        c.require(c.absolute(plan["output_root"]) == root / "eval", "cost report root differs from plan")
        bind_hash(plan["input_path"]); inputs = c.validate_inputs(c.read_jsonl(plan["input_path"]))
        prefills = sum(len(row["prompt_ids"]) for row in inputs)
        costs["all_attempts"] = census(plan, plan_path, grant_path, prefills, read)
        blockers.extend(costs["all_attempts"]["blockers"])
        freeze = read(root / "PRELAUNCH_BINDING.json")
        c.require(freeze.get("schema") == 1 and freeze.get("experiment") == c.EXPERIMENT and
                  freeze.get("status") == "PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION", "prelaunch freeze is not passed")
        c.require({"PLAN.json", "GRANT.json", "SESSION_ENVELOPE.json", "PREREG.md"} <= set(freeze["artifact_files_sha256"]), "freeze omits required phase identities")
        for base, pins in ((HERE, freeze["code_files_sha256"]), (root, freeze["artifact_files_sha256"])):
            for name, expected in pins.items():
                target = (base / name).resolve()
                c.require(target.is_relative_to(base.resolve()) and bind_hash(target) == expected, "frozen code/artifact changed")
        c.validate_plan(plan); c.validate_grant(plan_path, grant)
        envelope = read(root / "SESSION_ENVELOPE.json")
        c.require(envelope["session_start_unix"] == grant["session_start_unix"] and envelope["deadline_unix"] == grant["deadline_unix"] and
                  grant["deadline_unix"] - grant["session_start_unix"] == 7200 and bind_hash(plan["input_path"]) == plan["input_sha256"] and
                  prefills == freeze["prefill_tokens_per_model"], "phase envelope or prefill count differs")
        c.require(not (Path(plan["output_root"]) / "CONTROLLER_LEASE.json").exists(), "controller is active or ownership unsettled")
        attempts = costs["all_attempts"]["attempts"]; successes = {}; failures = []
        for item in attempts:
            try:
                slot = item["slot"]; out = Path(plan["output_root"]) / slot
                binding = read(out / "DIAGNOSTIC_BINDING.json")
                if binding.get("status") == "COMPLETE":
                    verify_success(item, plan, plan_path, grant_path, freeze, inputs, read, bind_hash)
                    role = item["model_role"]
                    c.require(role not in successes, "multiple successful attempts for one diagnostic model")
                    successes[role] = item
                else:
                    c.require(binding.get("status") == "FAILED" and binding.get("complete") is False and
                              binding.get("model_role") == item["model_role"] and binding.get("attempt") == item["attempt"] == 1 and
                              binding.get("plan_sha256") == bind_hash(plan_path) and binding.get("grant_sha256") == bind_hash(grant_path) and
                              binding.get("reservation_sha256") == item["reservation_sha256"], "failed original attempt lacks cost identity")
                    supervisor = read(out / "supervisor.json")
                    c.require(binding["supervisor_sha256"] == bind_hash(out / "supervisor.json") and
                              supervisor["status"] in ("EXITED", "TIMEOUT_OR_INTERRUPTED") and type(supervisor["returncode"]) is int and supervisor["returncode"] != 0 and
                              not (out / "GENERATIONS.json").exists(), "retry cannot replace success or surviving generations")
                    item["returncode"] = binding["controller_returncode"]; failures.append(item)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                blockers.append(item["slot"] + ": " + str(exc))
        costs["successful_actual_totals"] = {key: sum(item["actual_budget_charges"][key] for item in successes.values()) for key in ACTUAL_KEYS}
        costs["all_attempt_actual_totals"] = (dict(costs["successful_actual_totals"]) if len(successes) == len(attempts) else None)
        costs["actual_totals_note"] = "Failed/unobserved attempts have unknown actual token/call counts; their full reservation remains in all_attempts. Known successful totals are not asserted to equal programme actuals."
        c.require(set(successes) == set(c.MODEL_SHAS) and len(failures) <= 1 and len(attempts) == 3 + len(failures), "three unique successful models and every failure are required")
        for role, item in successes.items():
            prior = [failure for failure in failures if failure["model_role"] == role]
            c.require((item["attempt"] == 2 and len(prior) == 1) or (item["attempt"] == 1 and not prior), "retry selection differs from preserved failure")
        out = Path(plan["output_root"])
        c.require({p.name for p in out.iterdir() if p.is_dir()} == {item["slot"] for item in attempts}, "unaccounted model attempt directory")
        actual_common = c.attempt_costs(plan, grant_path)
        c.require(actual_common["totals"] == costs["all_attempts"]["totals"], "independent census differs from registered attempt accounting")
        terminal_path = root / "battery/TERMINAL.json"; terminal = read(terminal_path)
        c.require(terminal.get("status") == "COMPLETE_THREE_MODELS" and terminal["cost_accounting"] == actual_common and
                  terminal["final_binding_sha256"] == {role: item["binding_sha256"] for role, item in successes.items()}, "global battery terminal/cost/binding identities differ")
        outer = terminal["attempts"]
        c.require(isinstance(outer, list) and len(outer) == len(attempts), "battery terminal omitted or duplicated an attempt")
        by_slot = {c.slot_name(record["model_role"], record["attempt"]): record for record in outer}
        c.require(len(by_slot) == len(outer) and set(by_slot) == {item["slot"] for item in attempts}, "battery attempt inventory differs")
        for item in attempts:
            record = by_slot[item["slot"]]
            c.require(read(root / "battery" / (item["slot"] + ".json")) == record and record["returncode"] == item["returncode"], "foreground controller attempt receipt changed")
            item["controller_wall_seconds"] = seconds(record["controller_wall_seconds"])
        elapsed = seconds(terminal["session_elapsed_seconds"])
        c.require(elapsed <= 7200, "completed battery exceeded the registered phase deadline")
        costs["recorded_session_elapsed_at_battery_completion_seconds"] = elapsed
        current = time.time(); costs["session_elapsed_at_analysis_seconds"] = seconds(current - grant["session_start_unix"])
        costs["remaining_phase_seconds_at_analysis"] = max(0.0, grant["deadline_unix"] - current)
        costs["budget_remaining"] = {key: max(0, c.LIMITS[key] - value) for key, value in actual_common["totals"].items()}
        owned = set(HERE.rglob("*")) | set(root.rglob("*"))
        c.require(not any(path.is_symlink() for path in owned), "logical cost accounting refuses symlinks")
        logical = sum(path.stat().st_size for path in owned if path.is_file()); free = shutil.disk_usage(root).free
        costs["disk"] = {"logical_code_and_records_bytes_before_report": logical, "physical_volume_free_bytes": free,
                         "reserved_report_output_bytes": OUTPUT_ALLOWANCE, "logical_budget_remaining_after_output_reserve": c.LIMITS["new_disk_bytes"] - logical - OUTPUT_ALLOWANCE,
                         "physical_free_above_floor_after_output_reserve": free - c.LIMITS["free_disk_floor_bytes"] - OUTPUT_ALLOWANCE}
        c.require(current <= grant["deadline_unix"] and logical + OUTPUT_ALLOWANCE <= c.LIMITS["new_disk_bytes"] and
                  free - OUTPUT_ALLOWANCE >= c.LIMITS["free_disk_floor_bytes"], "final cost analysis/output cannot fit the registered time/disk envelope")
        c.require(all(small_sha(path) == expected for path, expected in evidence.items()), "a cost source changed during analysis")
        c.require(not any(name in sys.modules for name in ("torch", "transformers", "tokenizers")), "cost report imported a model runtime")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        blockers.append(str(exc))
    return {"schema": 1, "record": "BINDING_DIAGNOSTIC_COST_REPORT", "experiment": c.EXPERIMENT,
            "status": "BLOCKED_COST_ACCOUNTING" if blockers else "COMPLETE_VERIFIED_COST_REPORT",
            "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "reporter_sha256": small_sha(Path(__file__)),
            "analysis_only_added_outside_execution_freeze": True, "costs": costs, "blockers": blockers, "source_receipt_sha256": evidence,
            "scope_notes": ["Observed worker time is supervisor wall, including loading and custody checks. GPU-active time was not measured.",
                "Row, generated-token, forward-call and processed-input reservations cover every attempt, including failed or unobserved attempts.",
                "Actual charges are recomputed from completed generation token records. Missing failed-attempt actuals are unknown, never zero.",
                "Controller wall contains worker wall; these durations overlap and must not be added together.",
                "Recorded battery elapsed and analysis elapsed include preparation and idle gaps; neither is GPU-active time.",
                "Logical bytes include current diagnostic code and retained records. Physical free volume is measured separately; no weight contents are read.",
                "Recorded model/runtime identities are checked against prelaunch snapshots. This cost ledger does not independently rehash model weights or score answers."],
            "measured_gpu_active_seconds": None, "model_calls": 0, "weight_files_read": False,
            "quality_claim": False, "training_permission": False, "release_permitted": False}


def markdown(report):
    lines = ["# Controlled diagnostic cost report", "", "Status: **" + report["status"] + "**.", ""]
    if report["blockers"]:
        return "\n".join(lines + ["- " + value for value in report["blockers"]]) + "\n"
    costs = report["costs"]; total = costs["all_attempts"]["totals"]; actual = costs["successful_actual_totals"]
    lines += ["| Cost | Charged or reserved | Known successful actual |", "|---|---:|---:|",
              f"| Workers / explicit retries | {total['workers']} / {total['retries']} | 3 successful models |",
              f"| Worker supervisor wall, seconds | {total['total_worker_seconds']:.3f} | Included at left |",
              f"| Rows | {total['rows']} | {actual['rows']} |",
              f"| Generated tokens including EOS | {total['reserved_generated_tokens']} | {actual['generated_tokens_including_eos']} |",
              f"| Forward calls | {total['forward_calls']} | {actual['forward_calls']} |",
              f"| Processed input tokens including cached steps | {total['processed_input_tokens']} | {actual['processed_input_tokens']} |", "",
              "| Attempt | Worker wall, seconds | Controller wall, seconds (overlap) | Actual generated tokens | Actual forward calls | Actual processed inputs |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in costs["all_attempts"]["attempts"]:
        value = item["actual_budget_charges"] or {key: "unknown" for key in ACTUAL_KEYS}
        lines.append(f"| {item['slot']} | {item['worker_seconds_charged']:.3f} | {item['controller_wall_seconds']:.3f} | {value['generated_tokens_including_eos']} | {value['forward_calls']} | {value['processed_input_tokens']} |")
    lines += ["", f"Recorded battery elapsed: {costs['recorded_session_elapsed_at_battery_completion_seconds']:.3f} seconds. "
              f"Remaining phase time at analysis: {costs['remaining_phase_seconds_at_analysis']:.3f} seconds.",
              f"Logical code/records before this report: {costs['disk']['logical_code_and_records_bytes_before_report'] / 2**20:.3f} MiB. "
              f"Physical volume free: {costs['disk']['physical_volume_free_bytes'] / 2**30:.3f} GiB.", "",
              *["- " + value for value in report["scope_notes"]]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(); root = c.absolute(args.root, directory=True)
    outputs = root / "COST_REPORT.json", root / "COST_REPORT.md"
    c.require(not any(path.exists() for path in outputs), "cost report outputs are create-only")
    report = build_report(root)
    if report["status"] != "COMPLETE_VERIFIED_COST_REPORT":
        print(json.dumps(report, indent=2, allow_nan=False)); return 2
    payloads = json.dumps(report, indent=2, allow_nan=False) + "\n", markdown(report)
    c.require(sum(len(value.encode()) for value in payloads) <= OUTPUT_ALLOWANCE, "cost output exceeds its reserved allowance")
    for path, value in zip(outputs, payloads):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
    print(json.dumps({"status": report["status"], "outputs": [str(path) for path in outputs]})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
