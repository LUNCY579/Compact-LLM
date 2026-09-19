"""Foreground-only qualification controller; no training or evaluation command.

Reuse the MQ child watchdog/claim/lease/guards and the RRP hash-chained Journal
without changing their implementations. A separate validator explicitly requires
training_permission=False instead of lying to RRP's training-grant validator.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

from experiments.model_quality_gpu_v1.integrity import (
    canonical, digest, file_hash, read_json, seal, unseal, write_new,
)
from . import contracts as c
from .boundary import require_native_review


def journal_for(binding, grant, *, create=False):
    # RRP's registry and journal are reused, not its training-only grant validator.
    from experiments.reasoning_posttraining_v1.budget import Journal, register
    register(c.EXPERIMENT, ceiling=c.CEILING, phases=c.PHASES, excludes=c.EXCLUDES)
    return Journal(c.external(Path(binding["journal_path"])), grant, create=create)


def inputs(binding_path, grant_path):
    binding = read_json(c.external(binding_path))
    grant = read_json(c.external(grant_path))
    c.validate_approval(grant, binding)  # fail before hashing GBs or importing torch
    c.revalidate(binding)
    return binding, grant


def student_receipt(journal, run_id, binding, runtime=None):
    """Require tangible, settled, successful evidence from this exact journal.

    Hash seals detect accidental edits; they are not signatures/security against a
    local operator who can rewrite code and all records. No receipt-by-string API.
    """
    journal.read()
    c.require(type(run_id) is str and run_id in journal.settled, "settled subject-specific student qualification absent")
    directory = journal.directory / "runs" / run_id
    value = unseal(read_json(directory / "RESULT.json"))
    supervisor = read_json(directory / "supervisor.json")
    terminal_binding = read_json(directory / "TERMINAL_BINDING.json")
    c.require(terminal_binding.get("supervisor_sha256") == file_hash(directory / "supervisor.json")
              and terminal_binding.get("result_sha256") == file_hash(directory / "RESULT.json"),
              "student terminal/result bytes changed")
    launch = read_json(directory / "launch.json")
    claim = read_json(directory / "worker_claim.json")
    c.require(supervisor.get("status") == "EXITED" and supervisor.get("returncode") == 0,
              "student supervisor did not observe successful exit")
    c.require(value.get("schema") == "SCP_NATIVE_RESULT_V1" and value.get("experiment") == c.EXPERIMENT
              and value.get("phase") == "student" and value.get("status") == "PASS_QUALIFICATION_ONLY"
              and value.get("evidence_class") == "NATIVE_QUALIFICATION_ONLY"
              and value.get("binding_sha256") == digest(binding), "foreign/fixture/stale student receipt")
    c.require(all(value.get(k) is False for k in c.CLOSED), "student receipt opens a closed lane")
    c.require(value.get("launch_sha256") == digest(launch) == claim.get("launch_sha256")
              and value.get("worker_pid") == claim.get("pid"), "student launch/claim differs")
    checks = value.get("checks", {})
    c.require(checks.get("zero_update") is True and checks.get("backward_calls") == 0
              and checks.get("optimizer_steps") == 0 and checks.get("currently_trainable_tensors") == 0
              and checks.get("qualified_sequence_tokens") == c.HOST["max_sequence"]
              and checks.get("state_before") == checks.get("state_after")
              and checks.get("state_after") == value.get("loading", {}).get("subject_state"),
              "student mechanical scope/state differs")
    c.require(value.get("loading", {}).get("subject_id") == c.SUBJECT
              and value["loading"].get("subject_weight_sha256") == c.SUBJECT_SHA,
              "student qualification belongs to another subject")
    if runtime is not None:
        c.require(value.get("runtime") == runtime, "student qualification runtime/host differs")
    # Bind the observed terminal supervisor bytes to the inherited settlement event.
    settlements = [unseal(read_json(p)) for p in sorted((journal.directory / "events").glob("*.json"))]
    settlements = [e for e in settlements if e["kind"] == "settle" and e["run_id"] == run_id]
    c.require(len(settlements) == 1 and settlements[0]["receipt_sha256"] == file_hash(directory / "TERMINAL_BINDING.json"),
              "student terminal receipt differs from journal settlement")
    return value


def launch(args):
    require_native_review()  # Before imports, claims, leases, journals or model work.
    from experiments.model_quality_gpu_v1.resources import lease
    from experiments.model_quality_gpu_v1.supervisor import bounded_child
    binding, grant = inputs(args.binding, args.grant)
    c.require(type(args.timeout) is int and 1 <= args.timeout <= c.HOST["max_stage_timeout_seconds"], "unregistered watchdog bound")
    c.require(args.phase in c.PHASES, "training and evaluation phases are closed")
    c.require((args.phase == "mr") == (args.student_run is not None), "MR requires exactly one prior student run; student takes none")
    started = time.monotonic()
    # Same cooperative lease across model jobs. No stale-lock removal or bypass.
    with lease(Path(binding["lease_path"])):
        directory = Path(binding["journal_path"])
        journal = journal_for(binding, grant, create=not directory.exists())
        if args.phase == "mr":
            student_receipt(journal, args.student_run, binding)
        journal.start(args.run_id, args.phase, args.timeout)
        out = journal.directory / "runs" / args.run_id
        command = [sys.executable, "-m", "experiments.sequential_complementary_native_v1", "_worker",
                   "--binding", str(c.external(args.binding)), "--grant", str(c.external(args.grant)),
                   "--run-id", args.run_id, "--phase", args.phase, "--out", str(out)]
        if args.student_run is not None:
            command.extend(["--student-run", args.student_run])
        code = bounded_child(command, out, args.timeout)
        terminal = read_json(out / "supervisor.json")
        write_new(out / "TERMINAL_BINDING.json", {
            "supervisor_sha256": file_hash(out / "supervisor.json"),
            "result_sha256": file_hash(out / "RESULT.json") if (out / "RESULT.json").exists() else None,
            "binding_sha256": digest(binding), "phase": args.phase, **{k: False for k in c.CLOSED}})
        journal.settle(args.run_id, terminal, file_hash(out / "TERMINAL_BINDING.json"), time.monotonic() - started)
        # Exit zero alone is never native qualification evidence.
        if code == 0:
            result = unseal(read_json(out / "RESULT.json"))
            c.require(result["status"] == "PASS_QUALIFICATION_ONLY" and result["binding_sha256"] == digest(binding),
                      "successful child did not leave its bound qualification result")
            c.require(not journal.read()["budget_breached"], "worker breached the approved budget")
        print({"status": "QUALIFICATION_ONLY" if code == 0 else "FAILED_PRESERVED", "phase": args.phase,
               "run": str(out), **{k: False for k in c.CLOSED}})
        return code


def worker(args):
    require_native_review()  # Before imports, claims, leases, journals or model work.
    # This MUST precede the runtime import, model construction, or worker output.
    from experiments.model_quality_gpu_v1.supervisor import claim_worker
    out = c.external(args.out)
    claim_worker(out)
    backend = None
    try:
        binding, grant = inputs(args.binding, args.grant)
        journal = journal_for(binding, grant)
        c.require(out == journal.directory / "runs" / args.run_id, "worker evidence directory differs")
        c.require(args.run_id in journal.active and journal.active[args.run_id]["phase"] == args.phase,
                  "worker lacks the exact persistent reservation")
        c.require(args.phase in c.PHASES and ((args.phase == "mr") == (args.student_run is not None)), "unregistered phase dependency")
        from experiments.model_quality_gpu_v1 import native as mq_native
        from experiments.model_quality_gpu_v1.resources import guard as mq_guard
        from . import native
        mq_native.configure_runtime(17)
        runtime = mq_native.runtime(require=True)
        predecessor = student_receipt(journal, args.student_run, binding, runtime) if args.phase == "mr" else None
        events = out / "raw"
        events.mkdir(exist_ok=False)
        event_number = 0

        def guard(*, before_load=False):
            value = mq_guard(before_load=before_load, phase="qualify")
            c.require(value["rss_bytes"] <= c.HOST["max_host_rss_bytes"], "additional host RSS ceiling exceeded")
            c.require(shutil.disk_usage(out).free >= c.HOST["min_free_disk_bytes"], "inherited 14 GiB disk floor refused")
            c.require(sum(p.stat().st_size for p in journal.directory.rglob("*") if p.is_file()) <= grant["limits"]["new_disk_bytes"],
                      "actual journal/evidence bytes exceeded approved disk budget")
            return value

        def reserve(charges):
            journal.read()
            c.require(args.run_id in journal.active, "native request has no active supervised reservation")
            journal.reserve(charges)

        def emit(value):
            nonlocal event_number
            reserve({"new_disk_bytes": len(canonical(value)) + 1})
            write_new(events / f"{event_number:06d}.json", value)
            event_number += 1
            guard()

        guard(before_load=True)
        backend, loading = native.open_backend(binding, reserve, guard, emit, out)
        if predecessor is not None:
            c.require(loading["subject_state"] == predecessor["checks"]["state_after"], "reloaded subject state differs from student qualification")
        launch_record = read_json(out / "launch.json")
        if args.phase == "student":
            checks = native.student_probes(backend)
        else:
            checks = native.mr_probes(backend, launch_record["timeout_seconds"])
        c.revalidate(binding)
        guard()
        result = {"schema": "SCP_NATIVE_RESULT_V1", "experiment": c.EXPERIMENT, "phase": args.phase,
                  "status": "PASS_QUALIFICATION_ONLY", "evidence_class": "NATIVE_QUALIFICATION_ONLY",
                  "binding_sha256": digest(binding), "launch_sha256": digest(launch_record),
                  "worker_pid": os.getpid(), "runtime": runtime, "loading": loading, "checks": checks,
                  "forward_calls_actual_including_failures": backend.forward_calls,
                  "generation_requests": backend.generation_requests, "generated_tokens_actual": backend.generated_tokens,
                  "student_run": args.student_run, "budget": journal.read(),
                  "scope": "Fixed unscored inference mechanics only; not SCP experiment completion or a training permit",
                  **{k: False for k in c.CLOSED}}
        sealed = seal(result)
        reserve({"new_disk_bytes": len(canonical(sealed)) + 1})
        write_new(out / "RESULT.json", sealed)
        return 0
    except Exception as exc:
        write_new(out / "failure.json", {"status": "FAILED_PRESERVED", "type": type(exc).__name__,
                  "message": str(exc), **{k: False for k in c.CLOSED}})
        raise
    finally:
        if backend is not None:
            backend.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    propose = commands.add_parser("propose", help="Hash local assets and write a PENDING request; no model call")
    for key in ("source", "trained", "incumbent", "journal", "lease", "out"):
        propose.add_argument("--" + key, type=Path, required=True)
    for name in ("qualify", "_worker"):
        sub = commands.add_parser(name, help="Bounded inference-only qualification" if name == "qualify" else argparse.SUPPRESS)
        for key in ("binding", "grant"):
            sub.add_argument("--" + key, type=Path, required=True)
        sub.add_argument("--phase", choices=c.PHASES, required=True)
        sub.add_argument("--run-id", required=True)
        sub.add_argument("--student-run")
        if name == "qualify":
            sub.add_argument("--timeout", type=int, required=True)
        else:
            sub.add_argument("--out", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "propose":
        c.propose(args.source, args.trained, args.incumbent, args.out, args.journal, args.lease)
        print({"status": "PENDING_INFERENCE_ONLY_APPROVAL", "native_pass": False, "model_calls": 0})
        return 0
    return launch(args) if args.command == "qualify" else worker(args)
