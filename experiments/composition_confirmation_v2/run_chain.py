"""Operator-authorised chain (2026-09-25: "chain Stage A and Stage B after the batteries").

Runs, in order, each under its own GPU lease (each driver takes and releases it):
  1. interference_mechanism_v1/run_gpu_batteries.py        (the two exploratory batteries)
  2. composition_confirmation_v2/run_stage_a.py             (always attempted: independent of step 1's outcome)
  3. composition_confirmation_v2/run_stage_b.py             (ONLY if step 2 exited 0; its own preflight also refuses
                                                             unless the Stage A canary passed and no extension is due)
Writes CHAIN_STATUS.json after every step so an interrupted chain says exactly how far it got. Create-only:
it refuses to start if CHAIN_STATUS.json exists. Launch through WMI (wmi_runner.py) with the hermes interpreter.
"""
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PY = sys.executable
STATUS = HERE / "CHAIN_STATUS.json"
STEPS = [("batteries", EXP / "interference_mechanism_v1" / "run_gpu_batteries.py"),
         ("stage_a", HERE / "run_stage_a.py"),
         ("stage_b", HERE / "run_stage_b.py")]


def write(status):
    STATUS.write_bytes((json.dumps(status, indent=2) + "\n").encode())


def main() -> int:
    if STATUS.exists():
        raise SystemExit("CHAIN_STATUS.json exists — the chain has run before")
    status = {"authorised": "operator, 2026-09-25", "python": PY, "steps": []}
    write(status)
    for name, script in STEPS:
        if name == "stage_b":
            a = next((s for s in status["steps"] if s["name"] == "stage_a"), None)
            if not a or a["rc"] != 0:
                status["steps"].append({"name": name, "skipped": True, "reason": "Stage A did not exit 0"})
                write(status); print("stage_b skipped: Stage A did not exit 0", flush=True)
                break
        t0 = time.time()
        print(f"=== chain step {name}: {script}", flush=True)
        rc = subprocess.call([PY, "-u", str(script)], cwd=str(script.parent))
        status["steps"].append({"name": name, "rc": rc, "seconds": round(time.time() - t0),
                                "end": time.strftime("%Y-%m-%d %H:%M:%S")})
        write(status)
        print(f"=== chain step {name} exited {rc} after {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
