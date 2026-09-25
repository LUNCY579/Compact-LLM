"""OSIDE-001 driver (PREREG.md section 10): the ONE registered run.

Preflight (no GPU): no output exists; every source arm matches its BUILD.json; Supa and the four trained
checkpoints match their registered hashes; O_REPLY_NLL.json and ROWS_MANIFEST.json match theirs; the seed builder
reproduces OPALLO (positive control); host RAM >= 3.5 GiB; disk >= 6 GiB.

Then one GPU lease, stopping at the first failed check:
  1. BASE, OPALLO                      -> canary on both (exit 3 = harness failure: stop before any new arm)
  2. OPALLO-BF16 (in-memory cast), SPARSE, ADD-100  -> canary on all four
  3. ADD-O50, H1, H1c: rebuild (hash must equal the recorded one, else stop) -> score -> delete the weights
  4. SEED23, SEED29, SEED41: build -> score -> delete the weights
  5. analyse_o_side.py -> O_SIDE_RESULT.json
Scores are create-only; a rerun resumes after the last complete score file (a crash is a harness failure and
is logged in ATTEMPT_LOG.md; nothing already scored is rescored).
"""
from __future__ import annotations

import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
FDS = EXP / "fresh_dual_skill_instrument_v1"
SV = EXP / "titan_opallo_skill_vector_v1" / "arms"
sys.path.insert(0, str(FDS))
sys.path.insert(0, str(HERE))
from gpu_lease import lease  # noqa: E402
import build_o_arms as B  # noqa: E402
import analyse_o_side as A  # noqa: E402

PY = sys.executable
SCORES = HERE / "scores"
PREFLIGHT = HERE / "PREFLIGHT.json"
RECORDED_SHA = "f5cbaea0f3ee67932e55c79fa784ecf09d03a59b95651f49b10aef56e57004aa"   # O_REPLY_NLL.json, as committed
SOURCE = ("BASE", "OPALLO", "SPARSE", "ADD-100")
MIN_RAM_GIB, MIN_DISK_GIB = 3.5, 6.0


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, ok=(0,)):
    print("  $", " ".join(map(str, cmd)), flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=str(HERE))
    if rc not in ok:
        raise SystemExit(f"step failed (exit {rc})")
    return rc


def avail_gib() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def wait_ram(label: str, need: float = MIN_RAM_GIB, max_wait: int = 1800) -> None:
    t0 = time.time()
    while avail_gib() < need:
        if time.time() - t0 > max_wait:
            raise SystemExit(f"host RAM stayed below {need} GiB for {max_wait} s before {label}")
        time.sleep(60)


def build_sha(root: Path, arm: str) -> str:
    return json.loads((root / arm / "BUILD.json").read_text(encoding="utf-8"))["weight_sha256"]


def preflight() -> dict:
    if (HERE / "O_SIDE_RESULT.json").exists():
        raise SystemExit("O_SIDE_RESULT.json exists: the registered run is complete")
    rec = {"source_arms": {}, "trained": {}}
    for a in SOURCE:
        want, got = build_sha(SV, a), sha(SV / a / "model.safetensors")
        rec["source_arms"][a] = got
        if got != want:
            raise SystemExit(f"source arm {a} does not match its BUILD.json")
    if sha(B.SUPA) != B.SUPA_SHA:
        raise SystemExit("Supa does not match its registered hash")
    for s, want in B.TRAINED_SHA.items():
        rec["trained"][s] = got = sha(B.ckpt(s))
        if got != want:
            raise SystemExit(f"seed {s} trained file does not match its registered hash")
    if sha(A.RECORDED) != RECORDED_SHA:
        raise SystemExit("O_REPLY_NLL.json does not match its registered hash")
    if sha(A.MANIFEST) != A.MANIFEST_SHA:
        raise SystemExit("ROWS_MANIFEST.json does not match its registered hash")
    rec["seed_control"] = B.seed_control()
    if not rec["seed_control"]["PASS"]:
        raise SystemExit("seed builder positive control FAILED")
    rec["ram_gib"] = avail_gib()
    rec["disk_free_gib"] = shutil.disk_usage(HERE).free / 2 ** 30
    if rec["ram_gib"] < MIN_RAM_GIB or rec["disk_free_gib"] < MIN_DISK_GIB:
        raise SystemExit(f"NOT STARTED: RAM {rec['ram_gib']:.2f} GiB / disk {rec['disk_free_gib']:.1f} GiB")
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not PREFLIGHT.exists():
        PREFLIGHT.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    return rec


def score(arm: str, model_dir: Path, expect: str, cast: bool = False) -> None:
    out = SCORES / f"{arm}.json"
    if out.exists():
        print(f"  {arm}: already scored, skipping", flush=True)
        return
    wait_ram(f"scoring {arm}")
    cmd = [PY, "-u", "-B", HERE / "score_o_tokens.py", "--arm", arm, "--model-dir", model_dir,
           "--expect-sha", expect, "--output", out]
    run(cmd + (["--cast-bf16"] if cast else []))


def built_arm(arm: str, build_cmd: list) -> None:
    """Build (unless already scored), score, delete. A rebuild hash mismatch exits 4 and stops the run."""
    d = B.ARMS / arm
    if (SCORES / f"{arm}.json").exists():
        print(f"  {arm}: already scored, skipping", flush=True)
    else:
        if not (d / "model.safetensors").exists():
            if d.exists():
                raise SystemExit(f"{d} exists without weights: inspect before rerunning")
            wait_ram(f"building {arm}")
            run([PY, "-u", "-B", HERE / "build_o_arms.py"] + build_cmd)
        score(arm, d, build_sha(B.ARMS, arm))
    if (d / "model.safetensors").exists():
        run([PY, "-u", "-B", HERE / "build_o_arms.py", "delete", arm])


def main() -> int:
    preflight()
    with lease(label="oside-001", job="11 arms x 6741 held-out rows, per-token NLL (+6 CPU builds)", timeout_seconds=7200):
        score("BASE", SV / "BASE", build_sha(SV, "BASE"))
        score("OPALLO", SV / "OPALLO", build_sha(SV, "OPALLO"))
        run([PY, "-u", "-B", HERE / "analyse_o_side.py", "--canary-only", "BASE", "OPALLO"])
        score("OPALLO-BF16", SV / "OPALLO", build_sha(SV, "OPALLO"), cast=True)
        score("SPARSE", SV / "SPARSE", build_sha(SV, "SPARSE"))
        score("ADD-100", SV / "ADD-100", build_sha(SV, "ADD-100"))
        run([PY, "-u", "-B", HERE / "analyse_o_side.py", "--canary-only", *SOURCE])
        for arm in ("ADD-O50", "H1", "H1c"):
            built_arm(arm, ["rebuild", arm])
        for s in (23, 29, 41):
            built_arm(f"SEED{s}", ["seed", str(s)])
    run([PY, "-u", "-B", HERE / "analyse_o_side.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
