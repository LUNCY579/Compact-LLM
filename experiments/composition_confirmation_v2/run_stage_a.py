"""CONFIRM-002 Stage A driver (PREREG §4, AMENDMENT_1). One GPU lease; create-only outputs.

  1. build H25, TH25 and THFULL (build_arms_c2.py: canary + verify), unless they already exist with valid BUILD.json;
  2. transition skill on T_BANK_V3X (exposed): H25, TH25, THFULL. BASE and SPARSE come from CONFIRM-001's
     CONF_GATES.json;
  3. boundary skill on HSKILL_VIEWS_V2 (selection bank, exposed): BASE, SPARSE (existing arms), H25, TH25, THFULL;
  4. after the lease is released: analyse_stage_a.py (harness canary gate, development estimates, power).
Launch with the development interpreter (hermes venv).
"""
from __future__ import annotations

import json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FDS = EXP / "fresh_dual_skill_instrument_v1"
SV = EXP / "titan_opallo_skill_vector_v1" / "arms"
C1D = EXP / "composition_confirmation_v1"
VIEWS = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002/HSKILL_VIEWS_V2.jsonl")
sys.path.insert(0, str(FDS))
from gpu_lease import lease  # noqa: E402

PY = sys.executable
OUT = {"T": HERE / "STAGE_A_T_V3X.json", "H_REF": HERE / "STAGE_A_H_VIEWS_BASE_SPARSE.json",
       "H_NEW": HERE / "STAGE_A_H_VIEWS_NEW.json", "RESULT": HERE / "STAGE_A_RESULT.json"}


def run(cmd, cwd):
    print("  $", " ".join(map(str, cmd)), flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd))
    if rc != 0:
        raise SystemExit(f"step failed (exit {rc})")


def floors():
    import psutil
    a = psutil.virtual_memory().available / 2 ** 30; time.sleep(5); b = psutil.virtual_memory().available / 2 ** 30
    v = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split(",")
    vf = (float(v[0]) - float(v[1])) / 1024; disk = shutil.disk_usage("C:/").free / 2 ** 30
    print(f"  RAM {a:.2f}/{b:.2f} GiB, VRAM free {vf:.2f} GiB, disk free {disk:.1f} GiB", flush=True)
    need_disk = 10.0 if not all((HERE / "arms" / x / "BUILD.json").is_file() for x in ("H25", "TH25", "THFULL")) else 2.0
    if min(a, b) < 3.5 or vf < 3.5 or disk < need_disk:
        raise SystemExit("NOT STARTED: a resource floor is not met")


def main() -> int:
    for p in OUT.values():
        if p.exists():
            raise SystemExit(f"{p.name} exists — Stage A has run before (create-only)")
    floors()
    with lease(label="confirm002:stage-a", job="build 3 arms + 3 x 864 (V3X) + 5 x 224 (VIEWS)"):
        if not all((HERE / "arms" / x / "BUILD.json").is_file() for x in ("H25", "TH25", "THFULL")):
            run([PY, "-u", HERE / "build_arms_c2.py"], HERE)
        run([PY, "-u", FDS / "score_arms.py", "--bank", C1D / "T_BANK_V3X.json", "--arms-root", HERE / "arms",
             "--arms", "H25", "TH25", "THFULL", "--output", OUT["T"], "--no-lease"], FDS)
        run([PY, "-u", HERE / "score_boundary.py", "--bank", VIEWS, "--arms-root", SV, "--arms", "BASE", "SPARSE",
             "--output", OUT["H_REF"], "--no-lease"], HERE)
        run([PY, "-u", HERE / "score_boundary.py", "--bank", VIEWS, "--arms-root", HERE / "arms", "--arms", "H25", "TH25", "THFULL",
             "--output", OUT["H_NEW"], "--no-lease"], HERE)
    return subprocess.call([PY, "-u", str(HERE / "analyse_stage_a.py"), "--t-runs", str(C1D / "CONF_GATES.json"), str(OUT["T"]),
                            "--h-runs", str(OUT["H_REF"]), str(OUT["H_NEW"]), "--output", str(OUT["RESULT"])], cwd=str(HERE))


if __name__ == "__main__":
    raise SystemExit(main())
