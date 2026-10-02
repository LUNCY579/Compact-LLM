"""Start supervise.py detached through WMI Win32_Process.Create.

Processes created this way sit outside the calling session's process tree, so a session
teardown cannot kill the screen mid-run (the programme's recorded launch convention).
The default is a dry run that only prints the command; `--launch` creates the process
and writes a create-only LAUNCH_RECEIPT_SECTIONH.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
DEFAULT_PYTHON = "C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"
RECEIPT_NAME = "LAUNCH_RECEIPT_SECTIONH.json"
# The command line travels in an environment variable, so no grant text is ever
# re-parsed by PowerShell.
POWERSHELL = (
    "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
    "@{CommandLine=$env:SECTIONH_WMI_COMMAND; CurrentDirectory=$env:SECTIONH_WMI_CWD}; "
    "@{ProcessId=$r.ProcessId; ReturnValue=$r.ReturnValue} | ConvertTo-Json -Compress"
)


def build_command(python: str, freeze: Path, freeze_sha: str, grant: str, here: Path = HERE) -> str:
    return subprocess.list2cmdline([python, "-u", str(here / "supervise.py"), "--freeze", str(freeze),
                                    "--freeze-sha256", freeze_sha, "--grant", grant])


def launch(command: str, cwd: Path, run=subprocess.run) -> dict:
    env = {**os.environ, "SECTIONH_WMI_COMMAND": command, "SECTIONH_WMI_CWD": str(cwd)}
    completed = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", POWERSHELL],
                    capture_output=True, text=True, timeout=60, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"STOP_WMI_LAUNCH_FAILED:{completed.stderr.strip()[-300:]}")
    result = json.loads(completed.stdout.strip())
    if result.get("ReturnValue") != 0 or not result.get("ProcessId"):
        raise RuntimeError(f"STOP_WMI_CREATE_FAILED:{result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=HERE / "FREEZE_SECTIONH.json")
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--grant", required=True)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--launch", action="store_true", help="actually create the process (default: dry run)")
    args = parser.parse_args()
    command = build_command(args.python, args.freeze, args.freeze_sha256, args.grant)
    if not args.launch:
        print(json.dumps({"dry_run": True, "command": command}, indent=2))
    else:
        receipt_path = HERE / RECEIPT_NAME
        if receipt_path.exists():
            raise SystemExit(f"create-only receipt exists: {receipt_path.name}")
        result = launch(command, HERE)
        with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "command": command, **result},
                      stream, indent=2)
            stream.write("\n")
        print(json.dumps(result, indent=2))
