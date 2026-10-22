"""Launch run_eval.py fully detached so it survives the launching session's teardown.

The first battery attempt died with returncode 1073807364 (0x40010004,
DBG_TERMINATE_PROCESS) at 1628 s — not a crash, a kill: the parent Claude session
ended and took the whole process tree with it. For an unattended multi-hour run the
worker has to outlive its launcher.

DETACHED_PROCESS gives it no console to be signalled through.
CREATE_NEW_PROCESS_GROUP stops console control events propagating to it.
CREATE_BREAKAWAY_FROM_JOB escapes a parent job object configured to kill on close,
which is the mechanism that actually killed attempt 1. It is attempted first and
falls back if the job forbids breakaway.

Writes the child's pid so a later session can find it without guessing.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/compose4b001-20260921")
PY = Path("C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe")

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def spawn(flags):
    log = (OUT / "eval.log").open("a", encoding="utf-8", buffering=1)
    log.write(f"\n===== detached launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} flags={flags:#x} =====\n")
    log.flush()
    return subprocess.Popen(
        [str(PY), str(HERE / "run_eval.py")],
        cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)


def main():
    base = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    for flags in (base | CREATE_BREAKAWAY_FROM_JOB, base):
        try:
            child = spawn(flags)
        except OSError as exc:
            print(f"flags {flags:#x} refused ({exc}); falling back", flush=True)
            continue
        time.sleep(3)
        if child.poll() is not None:
            print(f"child exited immediately rc={child.returncode}", flush=True)
            continue
        (OUT / "WORKER_PID.json").write_text(json.dumps({
            "pid": child.pid, "creationflags": hex(flags),
            "breakaway_from_job": bool(flags & CREATE_BREAKAWAY_FROM_JOB),
            "launched_unix": time.time(),
            "argv": [str(PY), str(HERE / "run_eval.py")],
            "log": str(OUT / "eval.log"),
            "why": "attempt 1 died rc=1073807364 (DBG_TERMINATE_PROCESS) on session teardown",
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"pid": child.pid, "flags": hex(flags), "alive": True}, indent=2))
        return 0
    print("could not spawn a detached worker", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
