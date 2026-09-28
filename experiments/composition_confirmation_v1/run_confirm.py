"""CONFIRM-001 driver: pre-flight, gates, build RAND_MASK then SCALAR_P, score the remaining arms — one read.

Amendment 1: the bank is T_BANK_V3X (864 rows) and SCALAR_P is added. Pre-flight now re-hashes
every existing arm against its BUILD.json before anything is scored, and records the torch
precision flags the scoring subprocesses inherit.

One GPU lease covers construction AND scoring (a streaming build is not lighter than a forward pass on
this 15 GiB box). The driver refuses to start unless free host RAM clears the floor on two consecutive
reads a few seconds apart, because a single reading can be a transient peak.

Every output is create-only. If a step's output already exists the driver STOPS rather than skipping
it: this bank has a read budget of ONE, and a resume that silently skipped a finished step would be a
verifier that cannot fail.
"""
from __future__ import annotations

import hashlib, json, platform, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FDS = EXP / "fresh_dual_skill_instrument_v1"
sys.path.insert(0, str(FDS))
from gpu_lease import lease  # noqa: E402

BANK = HERE / "T_BANK_V3X.json"
RAM_FLOOR_GIB = 3.5
SV = EXP / "titan_opallo_skill_vector_v1" / "arms"
MD = EXP / "module_disjoint_composition_v1" / "arms"
STEPS = {
    "gates": (SV, ["BASE", "SPARSE"], HERE / "CONF_GATES.json"),
    "compose": (SV, ["AVG", "ADD-100"], HERE / "CONF_COMPOSE.json"),
    "location": (MD, ["SPARSE_ATTN", "SCALAR_MATCHED"], HERE / "CONF_LOCATION.json"),
    "randmask": (HERE / "arms", ["RAND_MASK", "SCALAR_P"], HERE / "CONF_RANDMASK.json"),
}
EXISTING = {"BASE": SV, "SPARSE": SV, "AVG": SV, "ADD-100": SV, "SPARSE_ATTN": MD, "SCALAR_MATCHED": MD}


def free_gib() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def ram_ok() -> bool:
    a = free_gib(); time.sleep(5); b = free_gib()
    print(f"  RAM free {a:.2f} then {b:.2f} GiB (floor {RAM_FLOOR_GIB})", flush=True)
    return min(a, b) >= RAM_FLOOR_GIB


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def preflight() -> None:
    """Every existing arm must still be the arm its BUILD.json describes. Written to PREFLIGHT.json."""
    out = HERE / "PREFLIGHT.json"
    if out.exists():
        raise SystemExit("PREFLIGHT.json already exists — this driver has run before")
    import torch
    rec = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
           "float32_matmul_precision": torch.get_float32_matmul_precision(), "arms": {}}
    for arm, root in EXISTING.items():
        want = json.loads((root / arm / "BUILD.json").read_text(encoding="utf-8"))["weight_sha256"]
        got = sha256(root / arm / "model.safetensors")
        rec["arms"][arm] = {"root": str(root), "build_json_sha256": want, "file_sha256": got, "match": got == want}
        print(f"  preflight {arm}: {'OK' if got == want else 'MISMATCH'}", flush=True)
    out.write_bytes((json.dumps(rec, indent=2) + "\n").encode())
    bad = [a for a, r in rec["arms"].items() if not r["match"]]
    if bad:
        raise SystemExit(f"PREFLIGHT FAILED: {bad} differ from their BUILD.json — nothing was scored")


def score(step: str, n_rows: int) -> None:
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
    if n != n_rows * len(arms):                               # completeness is a COUNT
        raise SystemExit(f"{out.name}: {n} predictions, expected {n_rows * len(arms)}")


def build(script: str, out_dir: Path, *extra: str) -> None:
    if out_dir.exists():
        raise SystemExit(f"{out_dir.name} already exists — refusing to reuse an arm built outside this run")
    rc = subprocess.call([sys.executable, str(HERE / script), *extra, "--output", str(out_dir)], cwd=str(HERE))
    if rc != 0:
        raise SystemExit(f"{out_dir.name} build or verify failed")


def main() -> int:
    n_rows = len(json.loads(BANK.read_text(encoding="utf-8"))["rows"])
    if n_rows != 864:
        raise SystemExit(f"{BANK.name} has {n_rows} rows, expected 864")
    if not ram_ok():
        print("NOT STARTED: host RAM below the floor. Nothing was read.", flush=True)
        return 4
    with lease(label="confirm001:T_BANK_V3X", job="preflight + gates + build RAND_MASK, SCALAR_P + score 8 arms x 864 rows"):
        preflight()
        score("gates", n_rows)
        rc = subprocess.call([sys.executable, str(HERE / "analyse_confirm.py"), "--gates-only",
                              "--runs", str(STEPS["gates"][2]), "--output", str(HERE / "GATES_RESULT.json")])
        if rc != 0:
            print("GATES FAILED: bank rejected, no further arm is scored, no confirmatory claim.", flush=True)
            return 3
        (HERE / "arms").mkdir(exist_ok=True)
        build("build_rand_mask.py", HERE / "arms" / "RAND_MASK")
        build("build_scalar_p.py", HERE / "arms" / "SCALAR_P", "--mask", str(HERE / "arms" / "RAND_MASK"))
        for step in ("compose", "location", "randmask"):
            score(step, n_rows)
    runs = [str(STEPS[s][2]) for s in STEPS]
    return subprocess.call([sys.executable, str(HERE / "analyse_confirm.py"), "--runs", *runs,
                            "--output", str(HERE / "CONFIRM_RESULT.json")])


if __name__ == "__main__":
    raise SystemExit(main())
