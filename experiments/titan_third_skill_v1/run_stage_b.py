"""TRISKILL-001 one-read Stage B scorer for the five preregistered arms.

The A3 choice is loaded from STAGE_A_RESULT.json. Each arm/bank unit runs in a
fresh process and is written through an attempt file; a driver reset can be
retried without accepting a partial score file. Rebuildable arm weights are
removed after all three banks have been scored for that arm.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from stage_b_guard import selected

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SB = HERE / "stage_b"
ARMS = SB / "arms"
CORES = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
SUPA = CORES / "Rineya-Supa-1.7.0-devfreeze"
TT = CORES / "Rineya-Titan-Transition-1.0.0-devfreeze"
T_BANK = HERE / "T_BANK_V5.json"
H_BANK = HERE / "H_FRESH.jsonl"
PY = sys.executable
sys.path.insert(0, str(EXP / "fresh_dual_skill_instrument_v1"))
from gpu_lease import lease  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_unit(cmd: list, output: Path, tries: int = 3) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        tmp = output.with_name(output.name + f".attempt{attempt}")
        if tmp.exists():
            raise SystemExit(f"partial attempt artifact exists: {tmp}; inspect before retry")
        call = [str(x) for x in cmd] + ["--output", str(tmp)]
        print(f"  $ {' '.join(call)} (try {attempt}/{tries})", flush=True)
        rc = subprocess.call(call, cwd=str(HERE))
        if rc == 0 and tmp.exists():
            tmp.replace(output)
            return
        print(f"  unit failed rc={rc}; output={tmp.exists()}", flush=True)
        if attempt < tries:
            time.sleep(30)
    raise SystemExit(f"unit failed after {tries} tries: {output}")


def score_c(name: str, arm_dir: Path) -> None:
    output = SB / "c_preds" / f"{name}.jsonl"
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        tmp = output.with_name(output.name + f".attempt{attempt}")
        if tmp.exists():
            raise SystemExit(f"partial C attempt artifact exists: {tmp}")
        cmd = [PY, "-u", HERE / "score_c.py", "--arm-dir", arm_dir,
               "--bank", "cfresh", "--out", tmp, "--no-lease"]
        print(f"  $ {' '.join(map(str, cmd))} (try {attempt}/3)", flush=True)
        rc = subprocess.call([str(x) for x in cmd], cwd=str(HERE))
        if rc == 0 and tmp.exists():
            tmp.replace(output)
            return
        print(f"  C unit failed rc={rc}; output={tmp.exists()}", flush=True)
        if attempt < 3:
            time.sleep(30)
    raise SystemExit(f"C unit failed after 3 tries: {output}")


def score_t(name: str, root: Path, dirname: str) -> None:
    out = SB / "t" / f"{name}.json"
    if not out.exists():
        run_unit([PY, "-u", EXP / "fresh_dual_skill_instrument_v1" / "score_arms.py",
                  "--bank", T_BANK, "--arms-root", root, "--arms", dirname, "--no-lease"], out)


def score_h(name: str, root: Path, dirname: str) -> None:
    out = SB / "h" / f"{name}.json"
    if not out.exists():
        run_unit([PY, "-u", HERE / "score_h_fresh.py", "--bank", H_BANK,
                  "--arms-root", root, "--arms", dirname, "--no-lease"], out)


def scored(name: str) -> bool:
    return all(p.exists() for p in (SB / "c_preds" / f"{name}.jsonl",
                                    SB / "t" / f"{name}.json", SB / "h" / f"{name}.json"))


def score_arm(name: str, arm_dir: Path, root: Path, dirname: str) -> None:
    if scored(name):
        return
    score_c(name, arm_dir)
    score_t(name, root, dirname)
    score_h(name, root, dirname)


def built_arm(name: str) -> None:
    if scored(name):
        return
    arm = ARMS / name
    weights = arm / "model.safetensors"
    if not weights.exists():
        if arm.exists():
            raise SystemExit(f"{arm} exists without weights and scores are incomplete")
        print(f"  building Stage B arm {name}", flush=True)
        rc = subprocess.call([PY, "-u", str(HERE / "stage_b_build_arm.py"), "--arm", name], cwd=str(HERE))
        if rc:
            raise SystemExit(f"arm build failed rc={rc}: {name}")
    score_arm(name, arm, ARMS, name)
    if not scored(name):
        raise SystemExit(f"refusing to delete weights: {name} scores incomplete")
    build = json.loads((arm / "BUILD.json").read_text(encoding="utf-8"))
    if sha256(weights) != build["weight_sha256"]:
        raise SystemExit(f"refusing to delete weights: {name} hash mismatch")
    weights.unlink()
    (arm / "WEIGHTS_DELETED.json").write_text(json.dumps({"record": "WEIGHTS_DELETED",
        "weight_sha256": build["weight_sha256"], "rebuild": f"stage_b_build_arm.py --arm {name}",
        "deleted": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"  {name}: scored and rebuildable weights deleted", flush=True)


def floors() -> None:
    import psutil
    a = psutil.virtual_memory().available / 2**30
    time.sleep(5)
    b = psutil.virtual_memory().available / 2**30
    disk = shutil.disk_usage("C:/").free / 2**30
    print(f"  RAM {a:.2f}/{b:.2f} GiB, disk {disk:.2f} GiB", flush=True)
    if min(a, b) < 5.0 or disk < 6.0:
        raise SystemExit("Stage B not started: resource floor not met")


def main() -> int:
    composite, solo, beta = selected()
    if (HERE / "STAGE_B_RESULT.json").exists():
        raise SystemExit("Stage B result already exists (create-only)")
    floors()
    print(f"  Stage A selected beta={beta} ({composite}); Stage B registered arms: BASE, TT, {composite}, {solo}, TTR", flush=True)
    with lease(label="triskill001:stage-b", job="Stage B: C_FRESH, T_BANK_V5, H_FRESH, five arms"):
        score_arm("BASE", SUPA, CORES, SUPA.name)
        score_arm("TT", TT, CORES, TT.name)
        for name in (composite, solo, "TTR"):
            built_arm(name)
    return subprocess.call([PY, "-u", str(HERE / "analyse_stage_b.py")], cwd=str(HERE))


if __name__ == "__main__":
    raise SystemExit(main())
