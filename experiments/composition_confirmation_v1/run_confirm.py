"""CONFIRM-001 driver: gates first, then build RAND_MASK, then score the remaining arms — one read.

One GPU lease covers construction AND scoring (a streaming build is not lighter than a forward pass on
this 15 GiB box). The driver refuses to start unless free host RAM clears the floor on two consecutive
reads a few seconds apart, because a single reading can be a transient peak.

Every output is create-only. If a step's output already exists the driver STOPS rather than skipping
it: this bank has a read budget of ONE, and a resume that silently skipped a finished step would be a
verifier that cannot fail.
"""
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FDS = EXP / "fresh_dual_skill_instrument_v1"
sys.path.insert(0, str(FDS))
from gpu_lease import lease  # noqa: E402

BANK = HERE / "T_BANK_V3.json"
RAM_FLOOR_GIB = 3.5
STEPS = {
    "gates": (EXP / "titan_opallo_skill_vector_v1" / "arms", ["BASE", "SPARSE"], HERE / "CONF_GATES.json"),
    "compose": (EXP / "titan_opallo_skill_vector_v1" / "arms", ["AVG", "ADD-100"], HERE / "CONF_COMPOSE.json"),
    "location": (EXP / "module_disjoint_composition_v1" / "arms", ["SPARSE_ATTN", "SCALAR_MATCHED"], HERE / "CONF_LOCATION.json"),
    "randmask": (HERE / "arms", ["RAND_MASK"], HERE / "CONF_RANDMASK.json"),
}


def free_gib() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def ram_ok() -> bool:
    a = free_gib(); time.sleep(5); b = free_gib()
    print(f"  RAM free {a:.2f} then {b:.2f} GiB (floor {RAM_FLOOR_GIB})", flush=True)
    return min(a, b) >= RAM_FLOOR_GIB


def score(step: str) -> None:
    root, arms, out = STEPS[step]
    if out.exists():
        raise SystemExit(f"{out.name} already exists — refusing to re-read the bank (read budget ONE)")
    for a in arms:
        if not (root / a / "model.safetensors").is_file():
            raise SystemExit(f"arm {a} has no weights under {root}")
    cmd = [sys.executable, str(FDS / "score_arms.py"), "--bank", str(BANK), "--arms-root", str(root),
           "--arms", *arms, "--output", str(out), "--no-lease"]
    print(f"  scoring {step}: {arms}", flush=True)
    rc = subprocess.call(cmd, cwd=str(FDS))
    if rc != 0 or not out.exists():
        raise SystemExit(f"scoring step {step} failed (exit {rc})")
    n = sum(len(a["predictions"]) for a in json.loads(out.read_text(encoding="utf-8"))["arms"])
    if n != 288 * len(arms):                                  # completeness is a COUNT
        raise SystemExit(f"{out.name}: {n} predictions, expected {288 * len(arms)}")


def main() -> int:
    if not ram_ok():
        print("NOT STARTED: host RAM below the floor. Nothing was read.", flush=True)
        return 4
    with lease(label="confirm001:T_BANK_V3", job="gates + build RAND_MASK + score 7 arms x 288 rows"):
        score("gates")
        rc = subprocess.call([sys.executable, str(HERE / "analyse_confirm.py"), "--gates-only",
                              "--runs", str(STEPS["gates"][2]), "--output", str(HERE / "GATES_RESULT.json")])
        if rc != 0:
            print("GATES FAILED: bank rejected, no further arm is scored, no confirmatory claim.", flush=True)
            return 3
        mask_dir = HERE / "arms" / "RAND_MASK"
        if mask_dir.exists():
            raise SystemExit("RAND_MASK already exists — refusing to reuse an arm built outside this run")
        (HERE / "arms").mkdir(exist_ok=True)
        rc = subprocess.call([sys.executable, str(HERE / "build_rand_mask.py"), "--output", str(mask_dir)])
        if rc != 0:
            raise SystemExit("RAND_MASK build or verify failed")
        for step in ("compose", "location", "randmask"):
            score(step)
    runs = [str(STEPS[s][2]) for s in STEPS]
    return subprocess.call([sys.executable, str(HERE / "analyse_confirm.py"), "--runs", *runs,
                            "--output", str(HERE / "CONFIRM_RESULT.json")])


if __name__ == "__main__":
    raise SystemExit(main())
