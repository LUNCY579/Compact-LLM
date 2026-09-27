"""CLARE-RINEYA-001 Phase 3 + 4 chain. Takes NO arguments (so it can be launched through
experiments/interference_mechanism_v1/wmi_runner.py). Relaunching resumes: completed steps are
skipped by their completion markers.

Tiers run in priority order and the chain STOPS at the first failure or BLOCKED preflight
(exit code 3 from cli.py), writing CHAIN_STATUS.json after every step:

  tier 0  native qualification (must end QUALIFIED), then the registered LR build gate
  tier 1  seed 0, canonical order: arms A, D, E -> train, evaluate (base once), interact
  tier 2  seed 1, canonical order: arms A, D, E
  tier 3  seed 0, secondary order: arms D, E
  tier 4  seed 0, canonical order: secondary arms B, then C (from B)
  final   analyse every completed run

Every step is a separate process, so GPU memory and host RAM are returned between steps.
Artifacts go under runs/clare-rineya-001/ (gitignored: deltas are large and are the irreplaceable
product of GPU time; back them up by hash, never delete them to make room).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "clare-rineya-001"
PY = sys.executable
CLI = [PY, "-u", "-B", str(HERE / "cli.py")]
STATUS = ROOT / "CHAIN_STATUS.json"


def _write_status(state: dict) -> None:
    tmp = STATUS.with_name(STATUS.name + ".partial")
    tmp.write_bytes((json.dumps(state, indent=2, sort_keys=True) + "\n").encode("ascii"))
    os.replace(tmp, STATUS)


def _qualified() -> bool:
    return any(json.loads(p.read_bytes()).get("status") == "QUALIFIED"
               for p in ROOT.glob("native-qualify-*/NATIVE_QUALIFICATION.json"))


def steps() -> list[dict]:
    pilot = ROOT / "LR_PILOT.json"
    out = [{"tier": 0, "name": "native-qualify", "done": _qualified,
            "argv": ["native-qualify", "--out", str(ROOT / f"native-qualify-{int(time.time())}")]},
           {"tier": 0, "name": "lr-pilot", "done": pilot.exists, "argv": ["lr-pilot", "--out", str(pilot)]}]

    def arm(tier, a, seed, order, extra=()):
        run = ROOT / f"{a}__s{seed}__{order}"
        base = seed == 0 and order == "canonical" and a == "A_dense_sequential"
        out.extend([
            {"tier": tier, "name": f"train {run.name}", "done": (run / "TRAIN_COMPLETE.json").exists,
             "argv": ["train", "--arm", a, "--seed", str(seed), "--order", order, "--out", str(run),
                      "--lr-from", str(pilot), *extra]},
            {"tier": tier, "name": f"evaluate {run.name}", "done": lambda run=run: any(run.glob("EVAL_COMPLETE_*.json")),
             "argv": ["evaluate", "--run", str(run), *(["--include-base"] if base else [])]},
            {"tier": tier, "name": f"interact {run.name}", "done": (run / "INTERACTION.json").exists,
             "argv": ["interact", "--run", str(run)]},
        ])
    for a in ("A_dense_sequential", "D_random_disjoint", "E_learned_disjoint"):
        arm(1, a, 0, "canonical")
    for a in ("A_dense_sequential", "D_random_disjoint", "E_learned_disjoint"):
        arm(2, a, 1, "canonical")
    for a in ("D_random_disjoint", "E_learned_disjoint"):
        arm(3, a, 0, "secondary")
    arm(4, "B_independent_sum", 0, "canonical")
    arm(4, "C_magnitude_sparse", 0, "canonical",
        extra=("--independent-from", str(ROOT / "B_independent_sum__s0__canonical")))
    return out


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    state = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": PY, "steps": []}
    for st in steps():
        entry = {"tier": st["tier"], "name": st["name"]}
        if st["done"]():
            entry["status"] = "SKIPPED_ALREADY_COMPLETE"
            state["steps"].append(entry)
            _write_status(state)
            continue
        t0 = time.time()
        rc = subprocess.call(CLI + st["argv"], cwd=str(HERE))
        entry.update(rc=rc, seconds=round(time.time() - t0), argv=st["argv"],
                     status="OK" if rc == 0 else ("BLOCKED" if rc == 3 else "FAILED"))
        if rc == 0 and st["name"] == "native-qualify" and not _qualified():
            entry["status"] = "FAILED_GATES"
        state["steps"].append(entry)
        _write_status(state)
        if entry["status"] != "OK":
            state["stopped_at"] = st["name"]
            _write_status(state)
            return 3 if entry["status"] == "BLOCKED" else 1
    runs = sorted(str(p) for p in ROOT.glob("*__s*__*") if (p / "TRAIN_COMPLETE.json").exists())
    analysis = ROOT / f"analysis-{int(time.time())}"
    rc = subprocess.call(CLI + ["analyse", "--runs", *runs, "--out", str(analysis)], cwd=str(HERE))
    state["steps"].append({"tier": "final", "name": "analyse", "rc": rc, "out": str(analysis)})
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_status(state)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
