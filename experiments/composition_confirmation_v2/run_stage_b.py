"""CONFIRM-002 Stage B driver: the ONE read of T_BANK_V4 and HSKILL_SOUP_V3 (PREREG §5, AMENDMENT_1).

Refuses to start unless STAGE_A_RESULT.json exists with canary PASS and no pending extension. Also refuses
unless every arm matches its BUILD.json hash and both banks match their committed hashes.
One GPU lease; create-only outputs; gates first:
  gates:  T_BANK_V4 <- BASE, SPARSE;  HSKILL_SOUP_V3 <- BASE, H25   -> analyse --gates-only (rc 3: both gate sets failed, stop)
  rest:   T_BANK_V4 <- TH25, THFULL;  HSKILL_SOUP_V3 <- SPARSE, TH25, THFULL
  then:   analyse_stage_b.py -> CONFIRM002_RESULT.json
"""
from __future__ import annotations

import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FDS = EXP / "fresh_dual_skill_instrument_v1"
SV = EXP / "titan_opallo_skill_vector_v1" / "arms"
A2 = HERE / "arms"
sys.path.insert(0, str(FDS))
from gpu_lease import lease  # noqa: E402

PY = sys.executable
T_BANK, T_SHA = HERE / "T_BANK_V4.json", "82f0a6423dd6b58b81319708e7fbda7c5880b9734742323ea11574ff354c93c0"
H_BANK, H_SHA = HERE / "HSKILL_SOUP_V3.jsonl", "686df45b94fa000be8df8d15e1f026a7f74e22deadfa4aacda96f5c6244c8ad5"
OUT = {"T_GATES": HERE / "B_T_GATES.json", "H_GATES_BASE": HERE / "B_H_GATES_BASE.json", "H_GATES_H25": HERE / "B_H_GATES_H25.json",
       "GATES": HERE / "B_GATES_RESULT.json", "T_REST": HERE / "B_T_COMPOSE.json", "H_SPARSE": HERE / "B_H_SPARSE.json",
       "H_REST": HERE / "B_H_COMPOSE.json", "RESULT": HERE / "CONFIRM002_RESULT.json", "PREFLIGHT": HERE / "B_PREFLIGHT.json"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd):
    print("  $", " ".join(map(str, cmd)), flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd))
    if rc != 0:
        raise SystemExit(f"step failed (exit {rc})")


def preflight():
    for p in OUT.values():
        if p.exists():
            raise SystemExit(f"{p.name} exists — Stage B has run before; the read budget is ONE")
    sa = HERE / "STAGE_A_RESULT.json"
    if not sa.is_file():
        raise SystemExit("Stage A has not run")
    s = json.loads(sa.read_text(encoding="utf-8"))
    if not s["canary"]["PASS"]:
        raise SystemExit("Stage A canary FAILED: the H limbs need the native-probe path (AMENDMENT_1 §5)")
    if s["rules"]["extend_T"] or s["rules"]["extend_boundary"]:
        raise SystemExit("Stage A requires a bank extension (AMENDMENT_1 §1) before Stage B")
    if sha(T_BANK) != T_SHA or sha(H_BANK) != H_SHA:
        raise SystemExit("a fresh bank does not match its committed hash")
    arms = {}
    for root, names in ((SV, ("BASE", "SPARSE")), (A2, ("H25", "TH25", "THFULL"))):
        for n in names:
            want = json.loads((root / n / "BUILD.json").read_text(encoding="utf-8"))["weight_sha256"]
            got = sha(root / n / "model.safetensors")
            arms[n] = {"match": got == want, "sha256": got}
            if got != want:
                raise SystemExit(f"arm {n} does not match its BUILD.json")
    import psutil
    a = psutil.virtual_memory().available / 2 ** 30; time.sleep(5); b = psutil.virtual_memory().available / 2 ** 30
    if min(a, b) < 3.5:
        raise SystemExit("NOT STARTED: host RAM below 3.5 GiB")
    OUT["PREFLIGHT"].write_bytes((json.dumps({"arms": arms, "t_bank": T_SHA, "h_bank": H_SHA, "ram": [a, b]}, indent=2) + "\n").encode())


def main() -> int:
    preflight()
    with lease(label="confirm002:stage-b", job="gates + 4 x 864 (T_BANK_V4) + 5 x 445 (HSKILL_SOUP_V3)"):
        run([PY, "-u", FDS / "score_arms.py", "--bank", T_BANK, "--arms-root", SV, "--arms", "BASE", "SPARSE",
             "--output", OUT["T_GATES"], "--no-lease"], FDS)
        run([PY, "-u", HERE / "score_boundary.py", "--bank", H_BANK, "--arms-root", SV, "--arms", "BASE",
             "--output", OUT["H_GATES_BASE"], "--no-lease"], HERE)
        run([PY, "-u", HERE / "score_boundary.py", "--bank", H_BANK, "--arms-root", A2, "--arms", "H25",
             "--output", OUT["H_GATES_H25"], "--no-lease"], HERE)
        rc = subprocess.call([PY, "-u", str(HERE / "analyse_stage_b.py"), "--gates-only", "--t-runs", str(OUT["T_GATES"]),
                              "--h-runs", str(OUT["H_GATES_BASE"]), str(OUT["H_GATES_H25"]), "--output", str(OUT["GATES"])], cwd=str(HERE))
        if rc == 3:
            print("BOTH GATE SETS FAILED: no confirmatory limb is evaluable; stopping.", flush=True)
            return 3
        if rc != 0:
            raise SystemExit(f"gate analysis failed (exit {rc})")
        run([PY, "-u", FDS / "score_arms.py", "--bank", T_BANK, "--arms-root", A2, "--arms", "TH25", "THFULL",
             "--output", OUT["T_REST"], "--no-lease"], FDS)
        run([PY, "-u", HERE / "score_boundary.py", "--bank", H_BANK, "--arms-root", SV, "--arms", "SPARSE",
             "--output", OUT["H_SPARSE"], "--no-lease"], HERE)
        run([PY, "-u", HERE / "score_boundary.py", "--bank", H_BANK, "--arms-root", A2, "--arms", "TH25", "THFULL",
             "--output", OUT["H_REST"], "--no-lease"], HERE)
    return subprocess.call([PY, "-u", str(HERE / "analyse_stage_b.py"), "--t-runs", str(OUT["T_GATES"]), str(OUT["T_REST"]),
                            "--h-runs", str(OUT["H_GATES_BASE"]), str(OUT["H_GATES_H25"]), str(OUT["H_SPARSE"]), str(OUT["H_REST"]),
                            "--output", str(OUT["RESULT"])], cwd=str(HERE))


if __name__ == "__main__":
    raise SystemExit(main())
