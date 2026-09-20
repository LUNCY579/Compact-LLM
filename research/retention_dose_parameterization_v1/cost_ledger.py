"""Create the final DOSEINT cost ledger from settled receipts, including failures.

Standard library only. Never loads or hashes model weights. Native journal
counters, individual probe attempts, and CPU builds are separate accounting
categories; aggregate terminal totals are checked but never added a second time.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

DEFAULT_ROOT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/doseint001-20260919")
JOURNALS = ("journal-dual899", "journal-dual899-preloadretry", "journal-dual899-recovery")
ARMS = ("a100", "a025", "a050", "a075")
LEG_LIMITS = {"dose": (384, 128), "select": (384, 128), "retention": (150, 24), "long_context": (256, 24)}
NATIVE_CAPS = {"model_worker_launches": 3, "processed_training_tokens_all_attempts": 779484,
               "evaluation_and_qualification_model_calls": 304, "generated_tokens_all_candidates": 36864,
               "model_worker_wall_seconds_and_conservative_gpu_seconds": 3488,
               "total_owned_work_wall_seconds": 4688, "adaptive_model_selection_rounds": 3,
               "cloud_or_teacher_spend": 0, "on_policy_rl_rollout_tokens": 0, "retries": 0}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nonnegative(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"invalid {label}")
    return value


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6), f"{label} differs from its source receipts")


def observed_resources(snapshots):
    """Extrema of recorded observations, never a representative peak certificate."""
    host = [s["available_host_bytes"] for s in snapshots if type(s.get("available_host_bytes")) is int]
    rss = [s["rss_bytes"] for s in snapshots if type(s.get("rss_bytes")) is int]
    peaks = [s["gpu0"]["peak"] for s in snapshots if isinstance(s.get("gpu0"), dict)
             and type(s["gpu0"].get("peak")) is int]
    return {"recorded_snapshots": len(snapshots), "minimum_observed_available_host_bytes": min(host) if host else None,
            "maximum_observed_process_rss_bytes": max(rss) if rss else None,
            "maximum_recorded_cuda_allocator_peak_bytes": max(peaks) if peaks else None,
            "scope": "Recorded snapshots only; not continuous monitoring, total GPU consumption, or a representative/max-shape resource certificate."}


def build_ledger(root):
    root = Path(root).resolve()
    sources = {}

    def bind(path):
        path = Path(path).resolve()
        require(path.is_file(), f"missing receipt: {path}")
        sources[str(path)] = file_hash(path)
        return path

    def read(path):
        value = json.loads(bind(path).read_text(encoding="utf-8-sig"))
        require(isinstance(value, dict), f"receipt must be an object: {path}")
        return value

    envelope = read(root / "SESSION_ENVELOPE.json")
    terminal = read(root / "battery-resume-001/TERMINAL.json")
    result = read(root / "RESULTS.json")
    require(terminal.get("status") == "COMPLETE" and terminal.get("report_returncode") == 0,
            "final resumed battery has not completed successfully")
    require(result.get("status") == "COMPLETE_COMPARABLE", "scientific report is not complete/comparable")
    for name in envelope["amendments"]:
        path = root / name
        require(path.resolve().parent == root, "amendment path escapes root")
        bind(path)
    bind(root / "ATTEMPT_MAP.json")
    bind(Path(__file__))
    native, total_native, snapshots = [], Counter(), []
    optimizer_tokens = 0
    for index, name in enumerate(JOURNALS):
        journal = root / name
        run = journal / "runs/train-001"
        summary, supervisor, grant = read(run / "run_summary.json"), read(run / "supervisor.json"), read(journal / "grant.json")
        require(summary["supervisor"] == supervisor and supervisor["status"] == "EXITED", "native supervisor receipt differs")
        require(summary["returncode"] == supervisor["returncode"] == (0 if index == 2 else 2), "unexpected native attempt outcome")
        budget = summary["budget_after"]
        require(budget["active_reservations"] == [] and budget["budget_breached"] is False, "native reservation remains unsettled")
        charges = Counter({key: 0 for key in grant["limits"]})
        previous, active = None, None
        for event_index, event_path in enumerate(sorted((journal / "events").glob("*.json"))):
            event = read(event_path)
            seal = event.pop("sha256")
            require(seal == digest(event) and event["previous"] == previous
                    and event["index"] == event_index and event_path.name == f"{event_index:06d}.json",
                    "native journal event chain is invalid")
            previous = seal
            for key, value in event["charges"].items():
                require(key in charges and type(value) is int and value >= 0, "invalid native charge")
                charges[key] += value
            if event["kind"] == "start":
                require(active is None and event["run_id"] == "train-001", "overlapping native launch")
                active = event
            elif event["kind"] == "settle":
                require(active is not None and event["run_id"] == active["run_id"], "native settlement lacks its launch")
                require(event["receipt_sha256"] == file_hash(run / "supervisor.json"), "native settlement supervisor hash differs")
                for key in ("model_worker_wall_seconds_and_conservative_gpu_seconds", "total_owned_work_wall_seconds"):
                    charges[key] += event["actual_wall"][key] - active["charges"][key]
                active = None
            else:
                require(event["kind"] == "charge", "unknown native event kind")
        require(active is None and dict(charges) == budget["used_or_reserved"], "native journal totals do not reproduce summary")
        require(all(0 <= value <= grant["limits"][key] for key, value in charges.items()), "native attempt exceeded its grant")
        total_native.update(charges)
        updates = processed = 0
        steps = run / "microfit/steps.jsonl"
        if steps.exists():
            with bind(steps).open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    require(row["updates"] == updates + 1 and row["processed_tokens"] > processed, "native step log is discontinuous")
                    updates, processed = row["updates"], row["processed_tokens"]
                    snapshots.append(row["resources"])
        require((updates, processed) == ((0, 0), (191, 377281), (193, 380921))[index], "native optimizer progress differs")
        optimizer_tokens += processed
        for filename in ("failure.json", "qualification.json", "binding.json", "result.json"):
            path = run / filename
            if path.exists():
                record = read(path)
                if filename == "binding.json":
                    snapshots.append(record["resources_before_load"])
        actual = nonnegative(supervisor["wall_seconds"], "native supervisor wall")
        require(charges["model_worker_wall_seconds_and_conservative_gpu_seconds"] >= math.ceil(actual), "native wall charge undercounts observed worker")
        native.append({"journal": name, "returncode": supervisor["returncode"], "optimizer_updates": updates,
                       "optimizer_data_tokens": processed, "actual_worker_wall_seconds": actual,
                       "journal_used_or_reserved": dict(charges)})
    for key, expected in {"processed_training_tokens_all_attempts": 762608,
                          "evaluation_and_qualification_model_calls": 152, "generated_tokens_all_candidates": 15360,
                          "model_worker_launches": 3}.items():
        require(total_native[key] == expected, f"unexpected global native {key}")
    require(optimizer_tokens == 758202, "unexpected optimizer-data total")
    require(all(total_native[key] <= ceiling for key, ceiling in NATIVE_CAPS.items()), "global native amendment ceiling exceeded")
    original_plan, resume_plan = read(root / "battery-launch/PLAN.json"), read(root / "battery-resume-001/PLAN.json")
    original_terminal = read(root / "battery-launch/TERMINAL.json")
    require(original_terminal["status"] == "FAILED", "original battery failure was not retained")
    original_commands = dict(original_plan["actions"])
    resume_actions = {item["leg"]: item for item in resume_plan["actions"]}
    builds = []
    for arm in ARMS:
        record = read(root / "battery-launch" / f"build-{arm}.json")
        require(record["returncode"] == 0, "materialization failed")
        receipt = read(root / "arms" / arm / "ARM.json")
        read(root / "arms" / arm / "MATERIALIZATION.json")
        weight_bytes = (root / "arms" / arm / "model.safetensors").stat().st_size
        require(weight_bytes == receipt["arm_weight_bytes"], "arm weight file length differs from builder receipt")
        builds.append({"arm": arm, "cpu_only_materialization": True, "wall_seconds": nonnegative(record["wall_seconds"], "build wall"),
                       "weight_bytes": weight_bytes, "available_host_bytes_before": receipt["available_host_bytes_before"]})
    require(len(list((root / "battery-launch").glob("build-*.json"))) == 4, "unexpected materialization attempt inventory")
    probes, seen_paths, succeeded = [], set(), set()
    specifications = [(path, False) for path in sorted((root / "battery-launch").glob("probe-*.json"))]
    specifications += [(path, True) for path in sorted((root / "battery-resume-001").glob("*.json"))
                       if path.name not in ("PLAN.json", "TERMINAL.json")]
    for path, resumed in specifications:
        leg = read(path)
        key = leg["leg"] if resumed else leg["label"].removeprefix("probe-")
        arm, suffix = key.split("-", 1)
        require(arm in ARMS and suffix in LEG_LIMITS, "unregistered probe attempt")
        if resumed:
            out = Path(leg["out"]).resolve()
            require(out == Path(resume_actions[key]["out"]).resolve(), "resume output differs from dispatched action")
        else:
            command = original_commands[leg["label"]]
            out = Path(command[command.index("--out") + 1]).resolve()
        require(out.parent == root / "eval" and out not in seen_paths, "duplicated or external probe attempt")
        seen_paths.add(out)
        supervisor = read(out / "worker/supervisor.json")
        worker_terminal = read(out / "worker/probe_terminal.json")
        binding_before = read(out / "worker/probe_binding.json")
        snapshots.append(binding_before["resources_before"])
        require(supervisor["status"] == "EXITED", "probe supervisor is not settled")
        actual = nonnegative(supervisor["wall_seconds"], "probe worker wall")
        require(leg["probe_time_accounting"] == "OBSERVED_SUPERVISOR_WALL_INCLUDING_FAILURE", "probe time lacks observed settlement")
        close(leg["probe_seconds_charged"], actual, "probe wall charge")
        rows, allowance = LEG_LIMITS[suffix]
        require(leg["probe_rows_reserved"] == rows and leg["probe_generation_tokens_reserved"] == rows * allowance,
                "probe reservation differs from its fixed instrument")
        success = leg["returncode"] == 0
        require((supervisor["returncode"] == 0) == success and worker_terminal["complete"] == success,
                "probe outcome differs across runner/supervisor/worker")
        if success:
            require(key not in succeeded and worker_terminal["status"] == "COMPLETED", "duplicate successful leg")
            succeeded.add(key)
            binding = read(out / "PROBE_WEIGHT_BINDING.json")
            require(binding["complete"] is True and binding["returncode"] == 0, "successful probe lacks custody")
            require(result["custody"][arm][suffix]["binding_sha256"] == sources[str((out / "PROBE_WEIGHT_BINDING.json").resolve())],
                    "scientific report used a different successful attempt")
        else:
            require(key == "a025-long_context" and not resumed, "unexpected failed probe")
            require(worker_terminal["failure"]["type"] == "AcceleratorError", "expected CUDA failure receipt differs")
            read(out / "FAILURE.json")
        probes.append({"leg": key, "attempt_directory": str(out), "resumed": resumed,
                       "returncode": leg["returncode"], "worker_returncode": supervisor["returncode"],
                       "actual_worker_wall_seconds": actual, "foreground_wall_seconds": nonnegative(leg["wall_seconds"], "probe foreground wall"),
                       "rows_reserved": rows, "generation_tokens_reserved": rows * allowance})
    require(len(probes) == 17 and len(succeeded) == 16
            and succeeded == {f"{arm}-{leg}" for arm in ARMS for leg in LEG_LIMITS}, "expected 17 attempts, 16 successful legs, one failure")
    rows_reserved = sum(p["rows_reserved"] for p in probes)
    generation_reserved = sum(p["generation_tokens_reserved"] for p in probes)
    probe_seconds = sum(p["actual_worker_wall_seconds"] for p in probes)
    require(rows_reserved == 4952 and generation_reserved == 438336, "global probe reservations differ")
    close(probe_seconds, terminal["probe_worker_seconds_all_attempts"], "final aggregate probe time")
    require(terminal["probe_rows_reserved_all_attempts"] == rows_reserved
            and terminal["probe_generation_tokens_reserved_all_attempts"] == generation_reserved, "terminal probe totals differ")
    require(probe_seconds <= envelope["probe_worker_seconds_all_attempts"]
            and rows_reserved <= envelope["probe_rows_reserved_all_attempts"]
            and generation_reserved <= envelope["probe_generation_tokens_reserved_all_attempts"], "probe envelope exceeded")
    require(envelope["deadline_unix"] == envelope["session_start_unix"] + envelope["total_owned_work_wall_seconds"], "session deadline is inconsistent")
    elapsed = nonnegative(terminal["session_elapsed_seconds"], "session elapsed")
    result_time = datetime.fromisoformat(result["created_utc"]).timestamp()
    require(elapsed <= envelope["total_owned_work_wall_seconds"]
            and envelope["session_start_unix"] <= result_time <= envelope["deadline_unix"], "completed experiment exceeded session deadline")
    logical_bytes = 0
    for path in root.rglob("*"):
        require(not path.is_symlink(), "logical records size audit refuses symlinks")
        if path.is_file():
            logical_bytes += path.stat().st_size
    free = shutil.disk_usage(root).free
    output_allowance = 4 * 2**20
    require(logical_bytes + output_allowance <= envelope["new_records_disk_bytes"], "records plus ledger exceed disk ceiling")
    require(free - output_allowance >= envelope["free_disk_floor_bytes"], "ledger would violate free-space floor")
    native_actual = sum(value["actual_worker_wall_seconds"] for value in native)
    return {"schema": 1, "status": "COMPLETE_VERIFIED_COST_LEDGER", "created_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root), "native_attempts": native, "native_totals": dict(total_native), "native_global_ceilings": NATIVE_CAPS,
            "optimizer_data_tokens_all_attempts": optimizer_tokens,
            "zero_update_qualification_tokens": total_native["processed_training_tokens_all_attempts"] - optimizer_tokens,
            "native_actual_worker_wall_seconds": native_actual,
            "native_ceiled_journal_worker_seconds": total_native["model_worker_wall_seconds_and_conservative_gpu_seconds"],
            "cpu_materializations": builds, "cpu_materialization_wall_seconds": sum(b["wall_seconds"] for b in builds),
            "probe_attempts": probes, "probe_attempt_count": 17, "probe_successes": 16, "probe_failures": 1,
            "probe_worker_seconds_including_failure": probe_seconds,
            "failed_cuda_probe_worker_seconds": sum(p["actual_worker_wall_seconds"] for p in probes if p["returncode"] != 0),
            "probe_foreground_wall_seconds": sum(p["foreground_wall_seconds"] for p in probes),
            "probe_rows_reserved": rows_reserved, "probe_generation_tokens_reserved": generation_reserved,
            "model_worker_launches_including_failures": 20, "model_worker_observed_seconds": native_actual + probe_seconds,
            "session_envelope": envelope, "completed_session_elapsed_seconds": elapsed,
            "ledger_generated_after_session_deadline": time.time() > envelope["deadline_unix"],
            "logical_records_bytes_before_ledger": logical_bytes, "physical_volume_free_bytes_before_ledger": free,
            "ledger_output_allowance_bytes": output_allowance, "disk_and_deadline_checks": "PASS",
            "resource_observations": observed_resources(snapshots), "source_receipt_sha256": sources,
            "limitations": ["Disk reservations and measured logical file bytes are distinct; reservations are never added to measured disk usage.",
                            "Logical size counts all files under the experiment root, including failures, weights, code snapshots and prepared data; it is not allocated filesystem size or programme-wide storage.",
                            "Probe generation figures are reserved ceilings, not claimed actual generated tokens. Incomplete failed inference is conservatively charged its entire row/token allowance.",
                            "Native model-call accounting and probe evaluated-row reservations measure different operations and are not pooled.",
                            "Worker wall totals exclude CPU materialization; foreground wall overlaps worker wall and is not added again to model compute.",
                            "Journal bookkeeping rounds do not imply adaptive model selection. No hardware peak, paid-compute cost, or representative maximum-memory claim is inferred."],
            "quality_claim": False, "release_permitted": False}


def markdown(ledger):
    native = ledger["native_totals"]
    return "\n".join([
        "# DOSEINT-001 final cost ledger", "", "Includes all preserved failed attempts; receipt totals are independently reconciled.", "",
        "| Category | Total |", "|---|---:|",
        f"| Native worker attempts | {native['model_worker_launches']} |",
        f"| Native charged tokens, including qualification | {native['processed_training_tokens_all_attempts']:,} |",
        f"| Optimizer-data tokens across attempts | {ledger['optimizer_data_tokens_all_attempts']:,} |",
        f"| Native evaluation/qualification calls | {native['evaluation_and_qualification_model_calls']} |",
        f"| Native generation-token reservations | {native['generated_tokens_all_candidates']:,} |",
        f"| Native observed worker seconds | {ledger['native_actual_worker_wall_seconds']:.3f} |",
        f"| Native rounded journal seconds | {ledger['native_ceiled_journal_worker_seconds']} |",
        f"| CPU materializations / seconds | 4 / {ledger['cpu_materialization_wall_seconds']:.3f} |",
        f"| Probe attempts / successful / failed | 17 / 16 / 1 |",
        f"| Probe worker seconds, including failure | {ledger['probe_worker_seconds_including_failure']:.3f} |",
        f"| Failed CUDA probe seconds included above | {ledger['failed_cuda_probe_worker_seconds']:.3f} |",
        f"| Probe row reservations | {ledger['probe_rows_reserved']:,} |",
        f"| Probe generation-token reservations | {ledger['probe_generation_tokens_reserved']:,} |",
        f"| Completed session elapsed seconds | {ledger['completed_session_elapsed_seconds']:.3f} |",
        f"| Logical experiment records before this ledger, GiB | {ledger['logical_records_bytes_before_ledger']/2**30:.3f} |",
        f"| Physical volume free space before this ledger, GiB | {ledger['physical_volume_free_bytes_before_ledger']/2**30:.3f} |",
        "", "Deadline, native/probe budgets, records size and the free-space floor passed.", "",
        *[f"- {text}" for text in ledger["limitations"]], "",
        "COST_LEDGER.json includes every contributing receipt hash and separate attempt records.", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    try:
        paths = [args.root / "COST_LEDGER.json", args.root / "COST_LEDGER.md"]
        require(not any(path.exists() for path in paths), "cost ledger outputs are create-only")
        ledger = build_ledger(args.root)
        encoded = json.dumps(ledger, indent=2, allow_nan=False) + "\n"
        rendered = markdown(ledger)
        require(len(encoded.encode()) + len(rendered.encode()) <= ledger["ledger_output_allowance_bytes"], "cost ledger exceeds its output allowance")
        for path, text in zip(paths, (encoded, rendered)):
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        print(json.dumps({"status": ledger["status"], "outputs": [str(p) for p in paths]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "BLOCKED_COST_ACCOUNTING", "reason": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
