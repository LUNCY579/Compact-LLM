"""Receipt-only cost accounting for the resumed DOSELORA session (stdlib only).

No model, source weights, evaluation generations, or quality endpoints are read.
Only complete successful training/export/probe terminals permit final files.
Failure returns partial cost accounting without publishing a completed ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

DEFAULT_ROOT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/highdose-lora001-resume-20260920")
DEFAULT_PRIOR = DEFAULT_ROOT.parent / "highdose-lora001-20260920"
WALL_KEYS = ("model_worker_wall_seconds_and_conservative_gpu_seconds", "total_owned_work_wall_seconds")
LEG_LIMITS = {"dose": (384, 128), "select": (384, 128), "retention": (150, 24), "long_context": (256, 24)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unseal(raw):
    body = {key: value for key, value in raw.items() if key != "sha256"}
    require(raw.get("sha256") == digest(body), "journal event seal differs")
    return body


def finite_seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid wall seconds")
    return float(value)


def replay_events(paths, read, limits):
    """Reproduce used/reserved charges; preserve full reservations while active."""
    used = {key: 0 for key in limits}
    active, previous = {}, None
    for index, path in enumerate(paths):
        raw = read(path); event = unseal(raw)
        require(path.name == f"{index:06d}.json" and event["index"] == index and event["previous"] == previous,
                "journal event order/chain differs")
        charges = event.get("charges", {})
        require(isinstance(charges, dict) and set(charges) <= set(limits)
                and all(type(value) is int and value >= 0 for value in charges.values()), "invalid native charge")
        for key, value in charges.items():
            used[key] += value
        if event["kind"] == "start":
            require(not active and event["run_id"] not in active, "overlapping native reservation")
            active[event["run_id"]] = event
        elif event["kind"] == "settle":
            require(event["run_id"] in active, "settlement lacks an active native reservation")
            start = active.pop(event["run_id"])
            for key in WALL_KEYS:
                actual = event["actual_wall"][key]
                require(type(actual) is int and actual >= 0, "invalid settled wall charge")
                used[key] += actual - start["charges"][key]
        else:
            require(event["kind"] == "charge", "unknown native budget event")
        previous = raw["sha256"]
    return {"used_or_reserved": used, "active_reservations": sorted(active),
            "budget_breached": any(used[key] > ceiling for key, ceiling in limits.items()),
            "remaining": {key: max(0, ceiling - used[key]) for key, ceiling in limits.items()}}


def build_report(root=DEFAULT_ROOT, prior=DEFAULT_PRIOR):
    root, prior = Path(root).resolve(), Path(prior).resolve()
    evidence, blockers, costs = {}, [], {}

    def read(path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular cost receipt: {path}")
        evidence[str(path)] = sha(path)
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        require(isinstance(value, dict), f"cost receipt is not an object: {path}")
        return value

    def blocked(message):
        blockers.append(message)

    try:
        envelope = read(root / "SESSION_ENVELOPE_V2.json")
        prior_waits = []
        for relative in ("PRELOAD_TERMINAL.json", "preload-attempts/wait-002/PRELOAD_TERMINAL.json"):
            value = read(prior / relative)
            require(value["status"] == "WAIT_EXPIRED_NO_WORKER" and value["worker_launched"] is False,
                    "prior wait is not verified as having no worker")
            require(value.get("controller_launched", False) is False, "prior wait dispatched a controller")
            prior_waits.append({"receipt": str(prior / relative), "wall_seconds": finite_seconds(value["wall_seconds"]),
                                "model_workers": 0})
        require(not (prior / "journal").exists(), "prior supposedly unspent LoRA session has a native journal")
        costs["carried_prior_waits"] = {"attempts": prior_waits, "wall_seconds": sum(item["wall_seconds"] for item in prior_waits),
                                        "model_workers": 0, "charged_to_fresh_session_deadline": False}
        journal = root / "journal"
        grant = read(journal / "grant.json")
        require(grant["limits"] == envelope["native_limits"], "native grant differs from resumed envelope")
        event_paths = sorted((journal / "events").glob("*.json"))
        require(event_paths, "native journal contains no events")
        state = replay_events(event_paths, read, grant["limits"])
        costs["native"] = {"journal_state": state, "attempts": []}
        if state["active_reservations"] or state["budget_breached"]:
            blocked("Native journal is active or breached; charges include its retained reservations.")
        runs = sorted(path for path in (journal / "runs").iterdir() if path.is_dir())
        require(len(runs) == 1, "expected one resumed native attempt")
        run = runs[0]
        item = {"run": str(run), "qualified_token_charges": {}, "qualified_call_charges": {}}
        costs["native"]["attempts"].append(item)
        progress = {"updates": 0, "processed_tokens": 0, "supervised_tokens": 0}
        steps = run / "steps.jsonl"
        if steps.exists():
            evidence[str(steps)] = sha(steps)
            with steps.open(encoding="utf-8") as stream:
                for line in stream:
                    step = json.loads(line)
                    require(step["updates"] == progress["updates"] + 1 and step["processed_tokens"] > progress["processed_tokens"],
                            "optimizer step-cost log is discontinuous")
                    progress = {key: step[key] for key in progress}
        item["completed_optimizer_progress"] = progress
        native_q = run / "native_base_qualification.json"
        if native_q.exists():
            value = read(native_q)
            require(value["status"] == "PASS" and value["optimizer_steps"] == 0, "native source qualification did not pass")
            native_chat = next(row for row in value["gradient_probes"] if row["probe"] == "native_chat")
            tokens = native_chat["tokens"]
            item["qualified_token_charges"]["native_source"] = 5 * tokens + 2048
            item["qualified_call_charges"]["native_source"] = 5 + (tokens + 2) // 3
        else:
            blocked("Native source qualification receipt is absent.")
        for label, filename in (("zero_B_adapter", "adapter_qualification.json"), ("final_merge", "merge_qualification.json")):
            path = run / filename
            if path.exists():
                value = read(path)
                item["qualified_token_charges"][label] = value["budget_charges"]["processed_training_tokens_all_attempts"]
                item["qualified_call_charges"][label] = value["budget_charges"]["evaluation_and_qualification_model_calls"]
                if value["status"] != "PASS":
                    blocked(f"{label} qualification failed; its charges remain included.")
            else:
                blocked(f"{label} qualification receipt is absent.")
        summary_path, supervisor_path = run / "run_summary.json", run / "supervisor.json"
        if summary_path.exists() and supervisor_path.exists():
            summary, supervisor = read(summary_path), read(supervisor_path)
            require(summary["supervisor"] == supervisor and summary["budget_after"] == state,
                    "native summary does not reproduce supervisor and journal costs")
            settlement = unseal(read(event_paths[-1]))
            require(settlement["kind"] == "settle" and settlement["run_id"] == run.name
                    and settlement["receipt_sha256"] == sha(supervisor_path),
                    "native final settlement is not bound to the observed supervisor")
            actual = finite_seconds(supervisor["wall_seconds"])
            item.update(returncode=summary["returncode"], actual_worker_wall_seconds=actual,
                        charged_worker_seconds=state["used_or_reserved"][WALL_KEYS[0]],
                        charged_owned_seconds=state["used_or_reserved"][WALL_KEYS[1]])
            if summary["returncode"] != 0 or summary["worker_failed"] or supervisor["status"] != "EXITED" or supervisor["returncode"] != 0:
                blocked("Native attempt failed; its observed/reserved costs remain included.")
            require(state["used_or_reserved"][WALL_KEYS[0]] == math.ceil(actual), "native rounded worker charge differs")
            require(state["used_or_reserved"][WALL_KEYS[1]] >= math.ceil(actual), "native owned wall omits worker time")
        else:
            blocked("Native attempt lacks a settled supervisor and run-summary terminal.")
        if (run / "failure.json").exists():
            read(run / "failure.json"); blocked("Native worker failure receipt is present.")
        if not blockers:
            native_result = read(run / "result.json")
            require(native_result["status"] == "COMPLETE" and native_result["complete"] is True, "native result is incomplete")
            used = state["used_or_reserved"]
            require(progress == {"updates": 193, "processed_tokens": 380921, "supervised_tokens": 37090}, "completed optimizer inventory differs")
            require(progress["processed_tokens"] + sum(item["qualified_token_charges"].values()) == used["processed_training_tokens_all_attempts"] == 389717,
                    "native optimizer/qualification token accounting does not reconcile")
            require(sum(item["qualified_call_charges"].values()) == used["evaluation_and_qualification_model_calls"] == 48,
                    "native qualification-call accounting does not reconcile")
            require(used["generated_tokens_all_candidates"] == 0 and used["model_worker_launches"] == 1, "unexpected native generation or launch count")
        exports = []
        costs["exports_and_compression"] = exports
        for arm in ("lora", "norm"):
            directory = root / "posttrain" / f"build-{arm}"
            record = {"arm": arm, "model_execution": False}
            exports.append(record)
            if not directory.exists():
                blocked(f"{arm} export has not run."); continue
            for filename in ("supervisor.json", "CPU_TERMINAL.json", "COMPRESSION.json"):
                path = directory / filename
                if not path.exists():
                    blocked(f"{arm}/{filename} terminal is absent."); continue
                value = read(path)
                if filename == "supervisor.json":
                    record["export_supervisor_wall_seconds"] = finite_seconds(value["wall_seconds"])
                    if value["status"] != "EXITED" or value["returncode"] != 0:
                        blocked(f"{arm} export supervisor failed.")
                elif filename == "CPU_TERMINAL.json":
                    record["export_worker_wall_seconds"] = finite_seconds(value["wall_seconds"])
                    if value["status"] != "COMPLETE" or value["returncode"] != 0 or value["failure"] is not None:
                        blocked(f"{arm} materialization failed.")
                else:
                    record["compression_wall_seconds"] = finite_seconds(value["wall_seconds"])
                    record["weight_logical_bytes"] = value["logical_bytes"]
                    if value["returncode"] != 0 or value["before_sha256"] != value["after_sha256"]:
                        blocked(f"{arm} compression/identity verification failed.")
        posttrain = root / "posttrain"
        costs["probe_attempts"] = []
        for path in sorted(posttrain.glob("*.json")) if posttrain.exists() else []:
            if path.name in ("TERMINAL.json", "RECOVERY_COMPLETE.json", "PLAN.json", "RECOVERY_PLAN.json"):
                continue
            leg = read(path)
            key = leg["leg"]
            arm, suffix = key.split("-", 1)
            require(arm in ("lora", "norm") and suffix in LEG_LIMITS, "unexpected probe cost receipt")
            out = Path(leg["out"]).resolve()
            require(out.parent == root / "eval" and out.name == path.stem, "probe cost directory differs")
            rows, allowance = LEG_LIMITS[suffix]
            require(leg["reserved_rows"] == rows and leg["reserved_generation_tokens"] == rows * allowance, "probe reservations differ from frozen limits")
            charged, accounting = 900.0, "FULL_RESERVATION_WITHOUT_OBSERVED_TERMINAL"
            supervisor_path = out / "worker/supervisor.json"
            if supervisor_path.exists():
                supervisor = read(supervisor_path)
                if type(supervisor.get("returncode")) is int:
                    charged = finite_seconds(supervisor["wall_seconds"]); accounting = "OBSERVED_SUPERVISOR_INCLUDING_FAILURE"
                    require((supervisor["returncode"] == 0) == (leg["returncode"] == 0),
                            "probe runner/supervisor success state differs")
            if leg["returncode"] == 0:
                terminal_receipt = read(out / "worker/probe_terminal.json")
                require(terminal_receipt["status"] == "COMPLETED" and terminal_receipt["complete"] is True
                        and terminal_receipt["system_exit_code"] == 0 and terminal_receipt["failure"] is None,
                        "successful probe cost receipt lacks a successful worker terminal")
            require(math.isclose(leg["worker_seconds_charged"], charged, rel_tol=0, abs_tol=1e-6), "failed/unobserved probe time was not fully charged")
            costs["probe_attempts"].append({"leg": key, "attempt_directory": str(out), "returncode": leg["returncode"],
                "foreground_wall_seconds": finite_seconds(leg["wall_seconds"]), "worker_seconds_charged": charged,
                "accounting": accounting, "reserved_rows": rows, "reserved_generation_tokens": rows * allowance})
        attempts = costs["probe_attempts"]
        totals = {"attempts": len(attempts), "successful_attempts": sum(item["returncode"] == 0 for item in attempts),
                  "failed_attempts": sum(item["returncode"] != 0 for item in attempts),
                  "worker_seconds_charged": sum(item["worker_seconds_charged"] for item in attempts),
                  "rows_reserved": sum(item["reserved_rows"] for item in attempts),
                  "generation_tokens_reserved": sum(item["reserved_generation_tokens"] for item in attempts)}
        costs["probe_totals"] = totals
        require(totals["attempts"] <= 9 and totals["failed_attempts"] <= 1
                and totals["worker_seconds_charged"] <= envelope["probe_worker_seconds_all_attempts"]
                and totals["rows_reserved"] <= envelope["probe_rows_reserved_all_attempts"]
                and totals["generation_tokens_reserved"] <= envelope["probe_generation_tokens_reserved_all_attempts"], "probe cost envelope exceeded")
        terminal_path = posttrain / "TERMINAL.json"
        if not terminal_path.exists():
            blocked("Posttraining terminal is absent.")
        else:
            terminal = read(terminal_path)
            if terminal["status"] != "COMPLETE_EIGHT_PROBES":
                blocked("Posttraining terminal failed; no complete cost report is published.")
            else:
                successful = [item["leg"] for item in attempts if item["returncode"] == 0]
                require(len(successful) == 8 and set(successful) == {f"{arm}-{leg}" for arm in ("lora", "norm") for leg in LEG_LIMITS},
                        "eight unique successful probe legs are not present")
                require(math.isclose(terminal["probe_worker_seconds"], totals["worker_seconds_charged"], rel_tol=0, abs_tol=1e-6)
                        and terminal["probe_rows_reserved"] == totals["rows_reserved"]
                        and terminal["probe_generation_tokens_reserved"] == totals["generation_tokens_reserved"], "aggregate terminal double-counts or omits individual attempts")
                require(envelope["deadline_unix"] == envelope["session_start_unix"] + envelope["total_owned_work_wall_seconds"]
                        and terminal["session_elapsed_seconds"] <= envelope["total_owned_work_wall_seconds"], "completed session deadline was exceeded")
                costs["fresh_session_elapsed_seconds"] = finite_seconds(terminal["session_elapsed_seconds"])
        logical = 0
        for path in root.rglob("*"):
            require(not path.is_symlink(), "logical disk accounting refuses symlinks")
            if path.is_file():
                logical += path.stat().st_size
        free = shutil.disk_usage(root).free
        costs["disk"] = {"logical_experiment_bytes_before_cost_report": logical, "physical_volume_free_bytes": free,
                         "ledger_output_allowance_bytes": 2 * 2**20, "weight_contents_read": False}
        require(logical + 2 * 2**20 <= envelope["new_records_disk_bytes"]
                and free - 2 * 2**20 >= envelope["free_disk_floor_bytes"], "cost report cannot fit the disk ceiling/free-space floor")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, StopIteration) as exc:
        blocked(f"{type(exc).__name__}: {exc}")
    return {"schema": 1, "record": "DOSELORA_RESUME_COST_REPORT",
            "status": "COMPLETE_VERIFIED_COST_REPORT" if not blockers else "BLOCKED_COST_ACCOUNTING",
            "created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "prior_root": str(prior),
            "reporter_sha256": sha(Path(__file__)), "costs": costs, "blockers": blockers, "source_receipt_sha256": evidence,
            "scope_notes": ["Native charges include qualification and any retained active reservations; completed optimizer tokens are reported separately.",
                "Probe row/token figures are reservations, not asserted actual generations. Every failed attempt remains charged.",
                "CPU export supervisor/worker durations overlap; report both but do not add them together. Compression is a separate CPU/filesystem duration.",
                "Prior 1800-second resource waits launched no model and belong to the expired session; they are carried descriptively, not subtracted again from the fresh deadline.",
                "Logical disk size covers the new experiment records root only, including all failed/partial artifacts. It is not allocated NTFS storage or total programme storage.",
                "No evaluation text, quality endpoint, model tensor contents, measured hardware peak or paid-compute charge is inferred."],
            "model_execution": False, "quality_claim": False, "release_permitted": False}


def markdown(report):
    costs = report["costs"]; native = costs["native"]["journal_state"]["used_or_reserved"]
    attempt = costs["native"]["attempts"][0]
    lines = ["# Resumed DOSELORA cost report", "", "Receipt accounting includes failures and keeps the expired session separate.", "",
             "| Cost | Value |", "|---|---:|",
             f"| Native processed-token charges including qualifications | {native['processed_training_tokens_all_attempts']:,} |",
             f"| Completed optimizer-data tokens | {attempt['completed_optimizer_progress']['processed_tokens']:,} |",
             f"| Separate qualification-token charges | {sum(attempt['qualified_token_charges'].values()):,} |",
             f"| Native qualification calls / generated tokens | {native['evaluation_and_qualification_model_calls']} / {native['generated_tokens_all_candidates']} |",
             f"| Observed native worker seconds | {attempt['actual_worker_wall_seconds']:.3f} |",
             f"| Native rounded worker / owned seconds | {native[WALL_KEYS[0]]} / {native[WALL_KEYS[1]]} |",
             f"| Probe attempts / failures | {costs['probe_totals']['attempts']} / {costs['probe_totals']['failed_attempts']} |",
             f"| Probe worker seconds including failures | {costs['probe_totals']['worker_seconds_charged']:.3f} |",
             f"| Probe rows / generation-token reservations | {costs['probe_totals']['rows_reserved']} / {costs['probe_totals']['generation_tokens_reserved']} |",
             f"| Fresh session elapsed seconds | {costs['fresh_session_elapsed_seconds']:.3f} |",
             f"| Prior no-model wait seconds | {costs['carried_prior_waits']['wall_seconds']:.3f} |",
             f"| Logical new records / physical free space, GiB | {costs['disk']['logical_experiment_bytes_before_cost_report']/2**30:.3f} / {costs['disk']['physical_volume_free_bytes']/2**30:.3f} |",
             "", "| Arm | Export worker seconds | Export supervisor seconds | Compression seconds |", "|---|---:|---:|---:|"]
    for record in costs["exports_and_compression"]:
        lines.append(f"| {record['arm']} | {record['export_worker_wall_seconds']:.3f} | {record['export_supervisor_wall_seconds']:.3f} | {record['compression_wall_seconds']:.3f} |")
    lines += ["", *[f"- {note}" for note in report["scope_notes"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR)
    args = parser.parse_args()
    paths = (args.root / "COST_REPORT.json", args.root / "COST_REPORT.md")
    require(not any(path.exists() for path in paths), "cost report outputs are create-only")
    report = build_report(args.root, args.prior_root)
    if report["status"] != "COMPLETE_VERIFIED_COST_REPORT":
        print(json.dumps(report, indent=2, allow_nan=False)); return 2
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    rendered = markdown(report)
    require(len(data.encode()) + len(rendered.encode()) <= 2 * 2**20, "report exceeds reserved output allowance")
    for path, text in zip(paths, (data, rendered)):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(json.dumps({"status": report["status"], "outputs": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
