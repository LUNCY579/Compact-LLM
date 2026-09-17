"""Bound the unchanged RAB diagnostic scripts with the existing MQ watchdog.

Run from a repository root with its pinned RAB Python. This adds execution
accountability, not a new scorer, serving qualification, or scientific approval.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from experiments.model_quality_gpu_v1.integrity import decode, digest, read_json, regular, write_new
from experiments.model_quality_gpu_v1.supervisor import bounded_child, claim_worker

SCRIPTS = {
    "process": "experiments/rab001_exploration/process_probe.py",
    "retention": "experiments/rab001_exploration/retention_probe.py",
    "long_context": "experiments/rab001_exploration/long_context_probe.py",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_lease_path():
    return Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"


def require_free_lease(lock):
    """Refuse before dispatch; never infer that an existing lease is stale."""
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(f"GPU lease already present; no worker dispatched: {lock}")


def recover_own_timedout_lease(out, lock):
    """Recover only this parent's proven-dead, timed-out, claimed worker lease.

    This is terminal cleanup of the launch just supervised, not a general stale-lock
    cleaner. The unchanged MQ lease intentionally cannot recover other jobs. Keep
    exact bytes before removal, then recheck identity and process death immediately
    before unlink. This remains a cooperative protocol, not an adversarial lock.
    """
    import psutil

    supervisor = read_json(out / "supervisor.json")
    if supervisor.get("status") != "TIMEOUT_OR_INTERRUPTED":
        return {"status": "NOT_APPLICABLE", "removed": False}
    if not lock.exists() and not lock.is_symlink():
        return {"status": "NO_LEASE", "removed": False}
    evidence = {"lock": str(lock), "removed": False, "quality_claim": False}

    def refuse(reason):
        write_new(out / "lease_recovery.json", {
            **evidence, "status": "REFUSED", "reason": reason,
        })
        raise RuntimeError(f"timed-out lease recovery refused: {reason}")

    cleanup = supervisor.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("child_exited") is not True:
        refuse("supervisor did not establish child termination")
    try:
        claim = read_json(out / "worker_claim.json")
        launch = read_json(out / "launch.json")
        raw = regular(lock).read_bytes()
        value = decode(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        refuse(f"missing or invalid ownership evidence: {type(exc).__name__}: {exc}")
    if not isinstance(value, dict) or not isinstance(claim, dict) or not isinstance(launch, dict):
        refuse("ownership records must be objects")
    pid = value.get("pid")
    if (type(pid) is not int or pid <= 0 or claim.get("pid") != pid
            or not isinstance(value.get("token"), str) or not value["token"]):
        refuse("lease owner does not match the claimed worker")
    if (claim.get("supervisor_pid") != os.getpid()
            or launch.get("supervisor_pid") != os.getpid()
            or launch.get("out") != str(out.absolute())
            or claim.get("launch_sha256") != digest(launch)):
        refuse("claim is not bound to this parent and this launch")
    evidence.update({"owner_pid": pid, "lock_sha256": hashlib.sha256(raw).hexdigest(),
                     "worker_claim_sha256": sha(out / "worker_claim.json"),
                     "supervisor_sha256": sha(out / "supervisor.json")})
    if psutil.pid_exists(pid):
        refuse("lease owner PID still exists")
    archive = out / "timedout_lease.raw.json"
    with archive.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    evidence["archived_exact_bytes"] = str(archive)
    write_new(out / "lease_recovery_intent.json", {
        **evidence, "status": "MATCHING_DEAD_OWNER_ARCHIVED_BEFORE_REMOVAL",
    })
    try:
        unchanged = regular(lock).read_bytes() == raw
    except (OSError, ValueError):
        unchanged = False
    if not unchanged:
        refuse("lease bytes/path changed after archival")
    if psutil.pid_exists(pid):
        refuse("lease owner PID appeared before removal")
    # A second byte read immediately precedes removal; another worker's newly
    # acquired lease must never be treated as the archived lease.
    try:
        if regular(lock).read_bytes() != raw:
            refuse("lease bytes changed immediately before removal")
        lock.unlink()
    except OSError as exc:
        refuse(f"lease removal failed: {type(exc).__name__}: {exc}")
    result = {**evidence, "status": "RECOVERED_OWN_DEAD_TIMEDOUT_WORKER_LEASE", "removed": True}
    write_new(out / "lease_recovery.json", result)
    return result


def result_path(kind, arguments):
    """Resolve only the legacy result filename, without importing a model runtime."""
    if kind == "retention":
        if len(arguments) not in (2, 3):
            raise ValueError("retention requires label, target, and optional device")
        label = arguments[0]
    else:
        labels = [arguments[i + 1] for i, arg in enumerate(arguments[:-1])
                  if arg == "--label"]
        labels += [arg.split("=", 1)[1] for arg in arguments if arg.startswith("--label=")]
        if len(labels) != 1:
            raise ValueError("exactly one legacy --label is required")
        label = labels[0]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", label):
        raise ValueError("result label must be a simple, bounded filename component")
    prefix = {"process": "PROCESS", "retention": "RETENTION_PROBE",
              "long_context": "LONGCTX"}[kind]
    return ROOT / "experiments/architecture_benchmark_001/evidence/retention-design-v1" / f"{prefix}_{label}.json"


@contextmanager
def preserve_legacy_outputs(kind, result, out):
    """Change output custody only; preserve the exact generator and scorer results.

    The old retention script uses write_text without an existence guard and retains
    only 24 sample predictions. Make the one result write exclusive and separately
    retain the complete, unchanged return from its generator. No target, decoded
    text, generation setting, or score is modified.
    """
    previous_write = Path.write_text
    generator_owner = previous_generator = None

    def exclusive_write(path, data, encoding=None, errors=None, newline=None):
        if path.absolute() == result.absolute():
            with path.open("x", encoding=encoding, errors=errors, newline=newline) as handle:
                return handle.write(data)
        return previous_write(path, data, encoding=encoding, errors=errors, newline=newline)

    try:
        Path.write_text = exclusive_write
        if kind == "retention":
            from experiments.architecture_benchmark_001 import run
            generator_owner = run
            previous_generator = run._greedy_probe

            def preserve_predictions(*args, **kwargs):
                predictions = previous_generator(*args, **kwargs)
                write_new(out / "retention_predictions.json", {
                    "scope": "UNCHANGED_LEGACY_GENERATOR_RETURN",
                    "rows": len(predictions), "predictions": predictions,
                    "scorer_modified": False, "quality_claim": False,
                })
                return predictions

            run._greedy_probe = preserve_predictions
        yield
    finally:
        Path.write_text = previous_write
        if generator_owner is not None and previous_generator is not None:
            generator_owner._greedy_probe = previous_generator


def execute_worker(args, arguments, script, expected, out):
    """Every claimed worker records failure truth, including pre-load refusal.

    A hard kill cannot run this finally block; the existing supervisor's timeout
    receipt is then authoritative and absence of this terminal is not success.
    """
    import runpy

    started = time.monotonic()
    code, status, failure, phase = None, "FAILED", None, "preflight"
    result = None
    old_argv, old_path = sys.argv, sys.path[:]
    try:
        if sha(script) != expected:
            raise ValueError("scorer changed between dispatch and execution")
        result = result_path(args.kind, arguments)
        if result.exists() or result.is_symlink():
            raise FileExistsError(f"create-only legacy result already exists: {result}")
        from experiments.model_quality_gpu_v1.resources import guard, lease
        phase = "lease"
        with lease(gpu_lease_path()):
            phase = "resource_guard"
            before = guard(before_load=True, phase="qualify")
            write_new(out / "probe_binding.json", {
                "kind": args.kind, "script": str(script), "script_sha256": expected,
                "wrapper_sha256": sha(__file__), "arguments": arguments,
                "result": str(result), "result_write_policy": "CREATE_ONLY",
                "resources_before": before, "quality_claim": False,
                "qualification_scope": "historical diagnostic path; not native serving",
            })
            phase = "legacy_script"
            sys.argv = [str(script), *arguments]
            # run_path does not provide Python's direct-script sys.path[0]. The
            # unchanged retention script imports its sibling `hostbudget` by name.
            sys.path.insert(0, str(script.parent))
            with preserve_legacy_outputs(args.kind, result, out):
                try:
                    runpy.run_path(str(script), run_name="__main__")
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
                    if code:
                        raise
            phase = "verify_result"
            if sha(script) != expected:
                raise RuntimeError("legacy script changed during execution")
            if not result.is_file() or not isinstance(json.loads(result.read_text(encoding="utf-8")), dict):
                raise RuntimeError("legacy script returned without a complete JSON result")
            code, status = 0, "COMPLETED"
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or code is None:
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        failure = {"type": type(exc).__name__, "message": str(exc), "phase": phase}
        status = "FAILED"
        raise
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
        try:
            after_sha, after_error = sha(script), None
        except OSError as exc:
            after_sha, after_error = None, f"{type(exc).__name__}: {exc}"
        if status == "COMPLETED" and after_sha != expected:
            status, code = "FAILED", 1
            failure = {"type": "SourceIdentityChanged", "phase": "terminal_hash",
                       "message": after_error or "script changed before terminal receipt"}
        write_new(out / "probe_terminal.json", {
            "status": status, "complete": status == "COMPLETED",
            "wall_seconds": time.monotonic() - started,
            "script_sha256_after": after_sha, "script_unchanged": after_sha == expected,
            "script_hash_error": after_error, "system_exit_code": code,
            "failure": failure, "result": str(result) if result is not None else None,
            "quality_claim": False,
        })
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(SCRIPTS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--expected-script-sha")
    parser.add_argument("--_worker", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 900:
        raise ValueError("registered probe timeout must be between 1 and 900 seconds")
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    script = ROOT / SCRIPTS[args.kind]
    expected = args.expected_script_sha or sha(script)
    out = args.out.absolute()
    if not args._worker:
        if sha(script) != expected:
            raise ValueError("scorer differs from its requested source hash")
        require_free_lease(gpu_lease_path())
        command = [sys.executable, str(Path(__file__).resolve()), "--kind", args.kind,
                   "--out", str(out), "--timeout", str(args.timeout),
                   "--expected-script-sha", expected, "--_worker", "--", *arguments]
        code = bounded_child(command, out, args.timeout)
        try:
            recovery = recover_own_timedout_lease(out, gpu_lease_path())
        except Exception as exc:
            recovery = {"status": "REFUSED", "removed": False,
                        "error": f"{type(exc).__name__}: {exc}"}
            code = code or 2
        print(json.dumps({"returncode": code, "attempt": str(out),
                          "lease_recovery": recovery}), flush=True)
        return code

    claim_worker(out)
    return execute_worker(args, arguments, script, expected, out)


if __name__ == "__main__":
    raise SystemExit(main())
