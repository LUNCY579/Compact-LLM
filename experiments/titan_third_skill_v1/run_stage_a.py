"""TRISKILL-001 Stage A driver (PREREG §6). Launch through WMI with the hermes interpreter; wait on
STAGE_A_RC.json. Preconditions: δ_C consolidated and BUILD_CANARY.json passed (build_arm.py).

Holds one GPU lease for the window; every (arm, bank) unit is a FRESH subprocess with --no-lease,
retried up to twice (driver resets kill a process's CUDA context), and skipped on resume if its
output exists. One arm is materialized at a time and its weights are deleted after its last unit.

Order (registered):
  1. BASE (Supa) and TT: C on DEV+CAL; TT also T on V3X and H on VIEWS. These give the canaries.
  2. C100: C on DEV+CAL, then A1. If A1 fails, STOP: nothing is composed.
  3. C25, C50: C on DEV+CAL (solo effects, descriptive).
  4. TTC25, TTC50, TTC100: C on DEV+CAL, T on V3X, H on VIEWS.
  5. analyse_stage_a.py full, which runs the canaries, A1, A2 and A3 and writes STAGE_A_RESULT.json.
"""
from __future__ import annotations

import json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FDS = EXP / "fresh_dual_skill_instrument_v1"
C2 = EXP / "composition_confirmation_v2"
SA = HERE / "stage_a"
CORES = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
SUPA = CORES / "Rineya-Supa-1.7.0-devfreeze"
TT = CORES / "Rineya-Titan-Transition-1.0.0-devfreeze"
V3X = EXP / "composition_confirmation_v1" / "T_BANK_V3X.json"
VIEWS = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002/HSKILL_VIEWS_V2.jsonl")
PY = sys.executable
sys.path.insert(0, str(FDS))
from gpu_lease import lease  # noqa: E402


def run(cmd, cwd, tries=3) -> None:
    for t in range(1, tries + 1):
        print(f"  $ {' '.join(map(str, cmd))}  (try {t}/{tries})", flush=True)
        rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd))
        if rc == 0:
            return
        print(f"  step failed rc={rc}", flush=True)
        if t < tries:
            time.sleep(30)
    raise SystemExit(f"step failed after {tries} tries: {cmd[1] if len(cmd) > 1 else cmd}")


def c_units(name, arm_dir):
    for bank in ("dev", "cal"):
        out = SA / "c_preds" / f"{name}__{bank}.jsonl"
        if not out.exists():
            run([PY, "-u", HERE / "score_c.py", "--arm-dir", arm_dir, "--bank", bank, "--out", out, "--no-lease"], HERE)


def t_unit(name, root, dirname):
    out = SA / "t" / f"{name}.json"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        run([PY, "-u", FDS / "score_arms.py", "--bank", V3X, "--arms-root", root, "--arms", dirname,
             "--output", out, "--no-lease"], FDS)


def h_unit(name, root, dirname):
    out = SA / "h" / f"{name}.json"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        run([PY, "-u", C2 / "score_boundary.py", "--bank", VIEWS, "--arms-root", root, "--arms", dirname,
             "--output", out, "--no-lease"], C2)


def outputs(name, full):
    need = [SA / "c_preds" / f"{name}__{b}.jsonl" for b in ("dev", "cal")]
    if full:
        need += [SA / "t" / f"{name}.json", SA / "h" / f"{name}.json"]
    return all(p.exists() for p in need)


def with_arm(name, full):
    """Build the arm if needed, score it, then delete its weights (BUILD.json keeps the hash)."""
    if outputs(name, full):
        return
    d = HERE / "arms" / name
    if not (d / "model.safetensors").exists():
        if d.exists():
            raise SystemExit(f"{d} exists without weights and its outputs are incomplete: resolve by hand")
        run([PY, "-u", HERE / "build_arm.py", "build", "--arm", name], HERE, tries=1)
    c_units(name, d)
    if full:
        t_unit(name, HERE / "arms", name)
        h_unit(name, HERE / "arms", name)
    b = json.loads((d / "BUILD.json").read_text(encoding="utf-8"))
    (d / "model.safetensors").unlink()
    (d / "WEIGHTS_DELETED.json").write_text(json.dumps({"record": "WEIGHTS_DELETED", "weight_sha256": b["weight_sha256"],
        "rebuild": f"build_arm.py build --arm {name} (from trained_C_fp32 + the pinned base)", "deleted": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"  {name}: scored, weights deleted", flush=True)


def floors():
    import psutil
    a = psutil.virtual_memory().available / 2 ** 30; time.sleep(5); b = psutil.virtual_memory().available / 2 ** 30
    disk = shutil.disk_usage("C:/").free / 2 ** 30
    print(f"  RAM {a:.2f}/{b:.2f} GiB, disk free {disk:.1f} GiB", flush=True)
    if min(a, b) < 5.0 or disk < 6.0:
        raise SystemExit("NOT STARTED: a resource floor is not met (RAM >= 5.0 GiB for the arm build, disk >= 6 GiB)")


def main() -> int:
    if (HERE / "STAGE_A_RESULT.json").exists():
        raise SystemExit("STAGE_A_RESULT.json exists: Stage A has run (create-only)")
    can = json.loads((HERE / "BUILD_CANARY.json").read_text(encoding="utf-8"))
    if not can.get("pass"):
        raise SystemExit("build canary has not passed")
    floors()
    with lease(label="triskill001:stage-a", job="Stage A: 7 arms; C 256 + T 864 + H 224 rows"):
        c_units("BASE", SUPA)
        c_units("TT", TT); t_unit("TT", CORES, TT.name); h_unit("TT", CORES, TT.name)
        with_arm("C100", full=False)
        rc = subprocess.call([PY, "-u", str(HERE / "analyse_stage_a.py"), "a1"], cwd=str(HERE))
        if rc == 3:
            print("A1 FAILED: the donor did not install C; nothing is composed (PREREG §6)", flush=True)
        elif rc != 0:
            raise SystemExit(f"A1 analysis error rc={rc}")
        else:
            for name in ("C25", "C50"):
                with_arm(name, full=False)
            for name in ("TTC25", "TTC50", "TTC100"):
                with_arm(name, full=True)
    return subprocess.call([PY, "-u", str(HERE / "analyse_stage_a.py"), "full"], cwd=str(HERE))


if __name__ == "__main__":
    raise SystemExit(main())
