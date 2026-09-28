"""Driver for the two registered exploratory GPU batteries, under ONE GPU lease.

  A. CONVENTION-LOC-001 step 1 (convention_localization_v1/STEP1_PREREG.md): ADD-O50, ADD-O60, ADD-O90,
     H1, H1c, scored on T_BANK_V2 (192 rows).
  B. INTERFERENCE-MECH-001 step 1 (interference_mechanism_v1/STEP1_PREREG.md): SPARSE-125, ADD-A125,
     DARE-O50, DARE-O10, DISJOINT-DARE, scored on T_BANK_V3X (864 rows).

Disk. C: had 9.1 GB free when this was written and ten FP32 arms need about 24 GB. Each arm is
therefore BUILT, VERIFIED, SCORED and then its model.safetensors is DELETED before the next one is
built. At most one new arm is on disk at a time. Its BUILD.json keeps the weight sha256, and every
build is deterministic from committed code, so any arm can be rebuilt and checked against its hash.
No pre-existing file is deleted. A guard refuses to build unless at least 4 GiB of disk is free.

Controls. Each builder's positive controls (and the partition check) run ONCE, at the start of this
run, before anything is built. A control failure aborts the run. The per-arm builds then use
--no-controls.

Everything is create-only. If any output of this run already exists, the driver stops. Scoring uses
score_arms.py with --no-lease, because this process holds the lease for the whole window.
Launch it with the development interpreter (hermes venv: torch 2.6.0+cu124, transformers 5.14.1).
"""
from __future__ import annotations

import json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
CONV = EXP / "convention_localization_v1"
FDS = EXP / "fresh_dual_skill_instrument_v1"
sys.path.insert(0, str(FDS))
from gpu_lease import lease  # noqa: E402

PY = sys.executable
RAM_FLOOR_GIB, DISK_FLOOR_GIB = 3.5, 4.0
BATTERIES = [
    {"name": "A-convention", "dir": CONV, "builder": CONV / "build_beta_arms.py",
     "bank": FDS / "T_BANK_V2.json", "rows": 192, "merged": CONV / "STEP1_SCORES.json",
     "analysis": CONV / "analyse_step1.py",
     "arms": [("beta:0.5", "ADD-O50"), ("beta:0.6", "ADD-O60"), ("beta:0.9", "ADD-O90"), ("H1", "H1"), ("H1c", "H1c")]},
    {"name": "B-interference", "dir": HERE, "builder": HERE / "build_arms.py",
     "bank": EXP / "composition_confirmation_v1" / "T_BANK_V3X.json", "rows": 864, "merged": HERE / "IM_SCORES.json",
     "analysis": HERE / "analyse_step1.py",
     "arms": [(a, a) for a in ("SPARSE-125", "ADD-A125", "DARE-O50", "DARE-O10", "DISJOINT-DARE")]},
]


def free_gib(path="C:/") -> float:
    return shutil.disk_usage(path).free / 2 ** 30


def ram_gib() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


VRAM_FLOOR_GIB = 3.5     # scoring needs about 2.5 GiB; on Windows, overflow past VRAM silently pages into host RAM


def vram_free_gib() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split(",")
    return (float(out[0]) - float(out[1])) / 1024


def run(cmd, cwd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd))
    if rc != 0:
        raise SystemExit(f"step failed (exit {rc}): {cmd[1] if len(cmd) > 1 else cmd}")


def preflight():
    for b in BATTERIES:
        for p in (b["merged"], b["dir"] / "arms", b["dir"] / "scores", b["dir"] / "STEP1_RESULT.json"):
            if p.exists():
                raise SystemExit(f"{p} already exists — this driver has run before; refusing (create-only)")
    a = ram_gib()
    time.sleep(5)
    b_ = ram_gib()
    print(f"  RAM free {a:.2f} then {b_:.2f} GiB (floor {RAM_FLOOR_GIB}); disk free {free_gib():.1f} GiB", flush=True)
    if min(a, b_) < RAM_FLOOR_GIB:
        raise SystemExit("NOT STARTED: host RAM below the floor")
    if free_gib() < DISK_FLOOR_GIB:
        raise SystemExit("NOT STARTED: disk below the floor")


def merge(files, out):
    docs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    base = dict(docs[0])
    for d in docs[1:]:
        if d["bank_sha256"] != base["bank_sha256"] or d["runtime"] != base["runtime"]:
            raise SystemExit("per-arm score files disagree on bank or runtime; refusing to merge")
    base["arms"] = [a for d in docs for a in d["arms"]]
    base["merged_from"] = [Path(f).name for f in files]
    out.write_bytes((json.dumps(base, indent=2) + "\n").encode("utf-8"))


def battery(b):
    print(f"== battery {b['name']}", flush=True)
    (b["dir"] / "scores").mkdir()
    files = []
    for spec, name in b["arms"]:
        if free_gib() < DISK_FLOOR_GIB:
            raise SystemExit(f"disk below {DISK_FLOOR_GIB} GiB before building {name}")
        run([PY, "-u", b["builder"], "--arms", spec, "--no-controls"], b["dir"])
        out = b["dir"] / "scores" / f"{name}.json"
        v = vram_free_gib()
        if v < VRAM_FLOOR_GIB:        # a co-tenant on the card would make allocations spill into host RAM
            raise SystemExit(f"VRAM free {v:.2f} GiB < {VRAM_FLOOR_GIB} before scoring {name}; stopping")
        run([PY, "-u", FDS / "score_arms.py", "--bank", b["bank"], "--arms-root", b["dir"] / "arms",
             "--arms", name, "--output", out, "--no-lease"], FDS)
        n = sum(len(a["predictions"]) for a in json.loads(out.read_text(encoding="utf-8"))["arms"])
        if n != b["rows"]:
            raise SystemExit(f"{name}: {n} predictions, expected {b['rows']}")
        w = b["dir"] / "arms" / name / "model.safetensors"
        rec = json.loads((b["dir"] / "arms" / name / "BUILD.json").read_text(encoding="utf-8"))
        w.unlink()
        (b["dir"] / "arms" / name / "WEIGHTS_DELETED.json").write_bytes((json.dumps(
            {"deleted_after_scoring": time.strftime("%Y-%m-%d %H:%M:%S"), "weight_sha256": rec["weight_sha256"],
             "reason": "disk: at most one new arm on disk at a time; rebuild deterministically and check this hash"},
            indent=2) + "\n").encode())
        print(f"  {name}: scored {n} rows; weights deleted (sha {rec['weight_sha256'][:12]} kept); disk free {free_gib():.1f} GiB", flush=True)
        files.append(out)
    merge(files, b["merged"])


def main() -> int:
    preflight()
    with lease(label="soup:gpu-batteries", job="10 exploratory arms: 5 x 192 rows (V2) + 5 x 864 rows (V3X)"):
        for b in BATTERIES:                                   # controls once, before anything is built
            run([PY, "-u", b["builder"], "--control-only"], b["dir"])
        for b in BATTERIES:
            battery(b)
    rc = 0
    for b in BATTERIES:
        rc |= subprocess.call([PY, "-u", str(b["analysis"])], cwd=str(b["dir"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
