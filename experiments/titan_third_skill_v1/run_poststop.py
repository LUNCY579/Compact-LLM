"""One-arm POSTSTOP-001 runner; scores only exposed Stage A banks."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PN = HERE / "poststop_null"
ARM = PN / "arms" / "TTR"
T_BANK = EXP / "composition_confirmation_v1" / "T_BANK_V3X.json"
H_BANK = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002/HSKILL_VIEWS_V2.jsonl")
PY = sys.executable
sys.path.insert(0, str(EXP / "fresh_dual_skill_instrument_v1"))
from gpu_lease import lease  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unit(cmd: list, output: Path, flag: str, tries: int = 3) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        tmp = output.with_name(output.name + f".attempt{attempt}")
        if tmp.exists():
            raise SystemExit(f"partial output exists: {tmp}")
        call = [str(x) for x in cmd] + [flag, str(tmp)]
        print(f"  $ {' '.join(call)} (try {attempt}/{tries})", flush=True)
        rc = subprocess.call(call, cwd=str(HERE))
        if rc == 0 and tmp.exists():
            tmp.replace(output)
            return
        print(f"  unit failed rc={rc}; output={tmp.exists()}", flush=True)
        if attempt < tries:
            time.sleep(30)
    raise SystemExit(f"unit failed after {tries} tries: {output}")


def main() -> int:
    stage = json.loads((HERE / "STAGE_A_RESULT.json").read_text(encoding="utf-8"))
    if stage["verdict"] != "STOP_NO_ELIGIBLE_BETA" or (HERE / "STAGE_B_RESULT.json").exists():
        raise SystemExit("post-STOP control refused: Stage A/B state mismatch")
    if (HERE / "POSTSTOP_NULL_RESULT.json").exists():
        raise SystemExit("post-STOP result already exists")
    import psutil
    a = psutil.virtual_memory().available / 2**30
    time.sleep(5)
    b = psutil.virtual_memory().available / 2**30
    disk = shutil.disk_usage("C:/").free / 2**30
    print(f"  RAM {a:.2f}/{b:.2f} GiB, disk {disk:.2f} GiB", flush=True)
    if min(a, b) < 5.0 or disk < 6.0:
        raise SystemExit("post-STOP control not started: resource floor not met")
    with lease(label="triskill-poststop:random-null", job="TTR25 on exposed Stage A banks"):
        if not (ARM / "model.safetensors").exists():
            if ARM.exists():
                raise SystemExit(f"partial arm directory exists: {ARM}")
            rc = subprocess.call([PY, "-u", str(HERE / "poststop_build.py")], cwd=str(HERE))
            if rc:
                raise SystemExit(f"TTR25 build failed rc={rc}")
        for bank in ("dev", "cal"):
            unit([PY, "-u", HERE / "score_c.py", "--arm-dir", ARM, "--bank", bank, "--no-lease"],
                 PN / "c_preds" / f"TTR25__{bank}.jsonl", "--out")
        unit([PY, "-u", EXP / "fresh_dual_skill_instrument_v1" / "score_arms.py",
              "--bank", T_BANK, "--arms-root", PN / "arms", "--arms", "TTR", "--no-lease"],
             PN / "t" / "TTR25.json", "--output")
        unit([PY, "-u", EXP / "composition_confirmation_v2" / "score_boundary.py",
              "--bank", H_BANK, "--arms-root", PN / "arms", "--arms", "TTR", "--no-lease"],
             PN / "h" / "TTR25.json", "--output")
        receipt = json.loads((ARM / "BUILD.json").read_text(encoding="utf-8"))
        if sha256(ARM / "model.safetensors") != receipt["weight_sha256"]:
            raise SystemExit("TTR25 weight hash mismatch before deletion")
        (ARM / "model.safetensors").unlink()
        (ARM / "WEIGHTS_DELETED.json").write_text(json.dumps({"record": "WEIGHTS_DELETED",
            "weight_sha256": receipt["weight_sha256"], "deleted": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rebuild": "poststop_build.py (trained_C_fp32 + pinned TT)"}, indent=1) + "\n", encoding="utf-8", newline="\n")
    return subprocess.call([PY, "-u", str(HERE / "analyse_poststop.py")], cwd=str(HERE))


if __name__ == "__main__":
    raise SystemExit(main())
