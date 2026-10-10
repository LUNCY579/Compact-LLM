"""Start the revised PARENTQ supervisor detached through WMI Win32_Process.Create.

Processes created this way sit outside the calling session's process tree, so a session
teardown cannot kill the screen mid-run (the programme's recorded launch convention,
interference_mechanism_v1/RUNBOOK_GPU.md, including ShowWindow=0, which is not optional:
a visible console can deliver Ctrl+C to the supervisor and cancel its hard-wall kill).
The default is a dry run that only prints the command. `--launch` creates the process,
waits up to 30 s for the supervisor's start or failure receipt, and writes a create-only
launch receipt with what it saw.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

import parentq_safety as safety

HERE = Path(os.path.abspath(__file__)).parent
DEFAULT_PYTHON = "C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"
ATTEMPTS_DIR = "native_attempts"
START_NAME = "SUPERVISOR_START_PARENTQ.json"
CONFIRM_SECONDS = 30
# The command line travels in an environment variable, so no grant text is ever
# re-parsed by PowerShell.
POWERSHELL = (
    "$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
    "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
    "@{CommandLine=$env:PARENTQ_WMI_COMMAND; CurrentDirectory=$env:PARENTQ_WMI_CWD; ProcessStartupInformation=$si}; "
    "@{ProcessId=$r.ProcessId; ReturnValue=$r.ReturnValue} | ConvertTo-Json -Compress"
)


def build_command(python: str, freeze: Path, freeze_sha: str, grant: str, here: Path = HERE,
                  runtime_root: Path = safety.RUNTIME_ROOT, attempt_number: int = 1) -> str:
    return subprocess.list2cmdline([python, "-B", "-u", str(here / "parentq_supervise.py"), "--freeze", str(freeze),
                                    "--freeze-sha256", freeze_sha, "--grant", grant,
                                    "--runtime-root", str(runtime_root), "--attempt-number", str(attempt_number)])


def launch(command: str, cwd: Path, run=subprocess.run) -> dict:
    env = {**os.environ, "PARENTQ_WMI_COMMAND": command, "PARENTQ_WMI_CWD": str(cwd)}
    completed = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", POWERSHELL],
                    capture_output=True, text=True, timeout=60, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"STOP_WMI_LAUNCH_FAILED:{completed.stderr.strip()[-300:]}")
    result = json.loads(completed.stdout.strip())
    if result.get("ReturnValue") != 0 or not result.get("ProcessId"):
        raise RuntimeError(f"STOP_WMI_CREATE_FAILED:{result}")
    return result


def confirm_started(runtime_root: Path, since: float, timeout: float = CONFIRM_SECONDS, sleep=time.sleep,
                    clock=time.time, supervisor_pid: int | None = None) -> dict:
    """Wait for a start receipt or a failure receipt newer than `since`."""
    folder = runtime_root / ATTEMPTS_DIR
    deadline = clock() + timeout
    while True:
        if folder.exists():
            fresh = [path for path in folder.rglob("*.json") if path.stat().st_mtime >= since]
            failures = [path for path in fresh if path.name.startswith("SUPERVISOR_FAILURE_") and
                        (supervisor_pid is None or json.loads(path.read_bytes()).get("supervisor_pid") == supervisor_pid)]
            starts = [path for path in fresh if path.name == START_NAME and
                      (supervisor_pid is None or json.loads(path.read_bytes()).get("supervisor_pid") == supervisor_pid)]
            if failures:
                return {"supervisor": "FAILED", "receipt": str(failures[0]),
                        "error": json.loads(failures[0].read_text(encoding="utf-8")).get("error")}
            if starts:
                return {"supervisor": "STARTED", "receipt": str(starts[0])}
        if clock() >= deadline:
            return {"supervisor": "UNCONFIRMED", "waited_seconds": timeout}
        sleep(1)


def main(argv=None) -> dict:
    """Argument/WMI/confirmation errors each leave an external create-only receipt."""
    result = {"status": "LAUNCH_NOT_REQUESTED"}
    root, command = safety.RUNTIME_ROOT, None
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE_PARENTQ.json")
        parser.add_argument("--freeze-sha256", required=True)
        parser.add_argument("--grant", required=True)
        parser.add_argument("--python", default=DEFAULT_PYTHON)
        parser.add_argument("--attempt-number", type=int, required=True)
        parser.add_argument("--runtime-root", type=Path, default=safety.RUNTIME_ROOT)
        parser.add_argument("--launch", action="store_true", help="create process; requires reviewed freeze and grant")
        args = parser.parse_args(argv)
        root = safety.validate_runtime_root(args.runtime_root, HERE.parents[1])
        command = build_command(args.python, args.freeze, args.freeze_sha256, args.grant,
                                runtime_root=root, attempt_number=args.attempt_number)
        if not args.launch:
            return {"dry_run": True, "command": command}
        # Refuse before dispatch as well as inside the independent supervisor.
        import parentq_supervise
        freeze = parentq_supervise.verify_freeze(args.freeze, args.freeze_sha256)
        if root != Path(freeze["runtime_root"]).resolve():
            raise ValueError("STOP_NATIVE_RUNTIME_ROOT_PIN_MISMATCH")
        parentq_supervise.verify_grant(args.grant, args.attempt_number, args.freeze_sha256, root)
        launched_at = time.time() - 1
        result = launch(command, HERE)
        result.update(confirm_started(root, launched_at, supervisor_pid=result["ProcessId"]))
        return result
    except BaseException as exc:
        result = {"status": "LAUNCH_ERROR", "error": repr(exc)}
        raise
    finally:
        if result.get("status") == "LAUNCH_ERROR" or "ProcessId" in result:
            # Microseconds plus PID avoid same-second receipt collisions.
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            receipt_path = root / ATTEMPTS_DIR / f"LAUNCH_RECEIPT_{stamp}-{os.getpid()}.json"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "command": command, **result}
            with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
