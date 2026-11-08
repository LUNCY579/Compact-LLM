"""Detached runner for SMALLVD-001, meant to be CREATED THROUGH WMI (parent WmiPrvSE, outside every
Claude tool's process tree -- the only launch that survived tool teardown on 2026-09-24).

WMI cannot redirect output, so this runner opens its own log, sets PATH explicitly (a WMI process
does not inherit the tool shell's PATH), records its own and its child's pid, and writes rc.json on
exit. "Complete" is judged from rc.json plus the queue's own log, never from the process vanishing.

usage: python smallvd_runner.py <run_dir> <SMALLVD_SCRATCH> <cwd> <cmd...>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

run_dir, scratch, cwd, cmd = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
run_dir.mkdir(parents=True, exist_ok=False)
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
env = dict(os.environ)
env["PATH"] = os.pathsep.join([r"C:\Program Files\Git\usr\bin", r"C:\Program Files\Git\bin", env.get("PATH", "")])
env["SMALLVD_SCRATCH"] = scratch
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"
(run_dir / "runner.json").write_text(json.dumps({"runner_pid": os.getpid(), "cmd": cmd, "cwd": cwd,
                                                 "scratch": scratch, "start": now()}, indent=1), encoding="utf-8")
t0 = time.time()
with open(run_dir / "runner.log", "ab", buffering=0) as log:
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    (run_dir / "child.json").write_text(json.dumps({"child_pid": p.pid, "start": now()}), encoding="utf-8")
    rc = p.wait()
(run_dir / "rc.json").write_text(json.dumps({"rc": rc, "seconds": round(time.time() - t0, 1), "end": now()}),
                                 encoding="utf-8")
