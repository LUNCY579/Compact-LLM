"""OSIDE-001 driver (PREREG.md section 10, as amended by AMENDMENT_1.md): the ONE registered run.

Preflight (no GPU): no result exists; every source arm matches its BUILD.json; Supa and the four trained
checkpoints match their registered hashes; O_REPLY_NLL.json and ROWS_MANIFEST.json match theirs; the seed builder
reproduces OPALLO (positive control, run in a subprocess so torch never enters this long-lived process); host RAM
>= 3.5 GiB; disk >= 6 GiB.

Then one GPU lease, stopping at the first failed check:
  1. BASE, OPALLO                      -> canary + provenance on both
  2. OPALLO-BF16 (in-memory cast), SPARSE, ADD-100  -> canary on all four, provenance on all five
  3. ADD-O50, H1, H1c: rebuild (hash must equal the REGISTERED one, else stop) -> score -> delete the weights
  4. SEED23, SEED29, SEED41: build -> score -> delete the weights
  5. analyse_o_side.py -> O_SIDE_RESULT.json (provenance of all 11 files, canary, gate, comparison)

Resume rules (AMENDMENT_1): a score file counts as complete only if it parses and carries 6,741 rows, no row
limit, the expected weight hash and cast flag, the registered runtime and device and the manifest's input hashes.
A complete file still in scores/ is never rescored; the only files ever rescored are ones the driver itself moved
to scores_failed/ (never deleted): an incomplete file, or every file of a stage whose canary/provenance check
failed (the permitted rerun rescores exactly those). A built arm's fresh score is checked BEFORE its weights are
deleted. An arm directory without a parseable BUILD.json is an unverified partial build: it is removed (strictly
inside arms/) and rebuilt; each completed build is fsynced. A final-analysis HARNESS_FAILURE (reachable only by
editing files after they passed these checks) ends the attempt; recovery then needs a logged operator decision.
Every stop exits nonzero (the WMI runner's rc file records 1); the log line says which stop it was.
"""
from __future__ import annotations

import hashlib, json, os, shutil, subprocess, sys, time
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
FAILED = HERE / "scores_failed"
RECORDED_SHA = "f5cbaea0f3ee67932e55c79fa784ecf09d03a59b95651f49b10aef56e57004aa"   # O_REPLY_NLL.json, as committed
SOURCE = ("BASE", "OPALLO", "SPARSE", "ADD-100")
STAGE1, STAGE2 = ("BASE", "OPALLO"), ("OPALLO-BF16", "SPARSE", "ADD-100")
MIN_RAM_GIB, MIN_DISK_GIB = 3.5, 6.0


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


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


def expected_sha(arm: str) -> str | None:
    """Registered hash for fixed arms (rebuilt arms included); a seed arm's own BUILD.json otherwise."""
    if arm in A.REGISTERED_SHA:
        return A.REGISTERED_SHA[arm]
    b = read_build(B.ARMS / arm)
    return b["weight_sha256"] if b else None


def read_build(d: Path) -> dict | None:
    """BUILD.json of an arm dir, or None if it is missing, torn (unparseable) or lacks a weight hash."""
    try:
        b = json.loads((d / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return b if isinstance(b, dict) and isinstance(b.get("weight_sha256"), str) else None


def fsync_tree(d: Path) -> None:
    """Make a finished build durable (build_o_arms.write_json and the asset copies do not fsync)."""
    for p in sorted(d.iterdir()):
        if p.is_file():
            with open(p, "r+b") as f:
                os.fsync(f.fileno())


def score_problems(arm: str, expect: str | None) -> list:
    """Why scores/<arm>.json is not a complete score file ([] = complete)."""
    p = SCORES / f"{arm}.json"
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                   # truncated / zero-filled after a power loss
        return [f"unreadable: {type(e).__name__}"]
    man = json.loads(A.MANIFEST.read_text(encoding="utf-8"))
    prov = A.provenance({arm: s}, man, {arm: expect})
    return prov["arms"][arm]["problems"]


def move_aside(paths, why: str) -> Path:
    FAILED.mkdir(exist_ok=True)
    n = 1 + sum(1 for x in FAILED.iterdir() if x.is_dir())
    dest = FAILED / f"attempt_{n}_{stamp()}"
    dest.mkdir()
    for p in paths:
        if p.exists():
            shutil.move(str(p), str(dest / p.name))
    (dest / "WHY.txt").write_bytes((why + "\n").encode("utf-8"))
    print(f"  moved {[p.name for p in paths]} to {dest}: {why}", flush=True)
    return dest


def preflight() -> dict:
    if (HERE / "O_SIDE_RESULT.json").exists():
        raise SystemExit("O_SIDE_RESULT.json exists: the registered run is complete")
    rec = {"source_arms": {}, "trained": {}}
    for a in SOURCE:
        got = sha(SV / a / "model.safetensors")
        rec["source_arms"][a] = got
        if got != build_sha(SV, a) or got != A.REGISTERED_SHA[a]:
            raise SystemExit(f"source arm {a} does not match its BUILD.json / registered hash")
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
    p = subprocess.run([PY, "-u", "-B", str(HERE / "build_o_arms.py"), "seed-control"], cwd=str(HERE),
                       capture_output=True, text=True)
    lines = p.stdout.strip().splitlines()
    rec["seed_control"] = json.loads(lines[-1]) if lines else {"PASS": False, "stderr": p.stderr[-2000:]}
    rec["seed_control"]["rc"] = p.returncode
    if p.returncode != 0 or not rec["seed_control"].get("PASS"):
        raise SystemExit(f"seed builder positive control FAILED: {rec['seed_control']}")
    rec["ram_gib"] = avail_gib()
    rec["disk_free_gib"] = shutil.disk_usage(HERE).free / 2 ** 30
    if rec["ram_gib"] < MIN_RAM_GIB or rec["disk_free_gib"] < MIN_DISK_GIB:
        raise SystemExit(f"NOT STARTED: RAM {rec['ram_gib']:.2f} GiB / disk {rec['disk_free_gib']:.1f} GiB")
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = HERE / "PREFLIGHT.json"
    if out.exists():
        out = HERE / f"PREFLIGHT_rerun_{stamp()}.json"
    out.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    return rec


def score(arm: str, model_dir: Path, expect: str, cast: bool = False) -> None:
    out = SCORES / f"{arm}.json"
    if out.exists():
        problems = score_problems(arm, expect)
        if not problems:
            print(f"  {arm}: complete score file present, skipping", flush=True)
            return
        move_aside([out], f"{arm}: incomplete score file ({'; '.join(problems)})")
    wait_ram(f"scoring {arm}")
    cmd = [PY, "-u", "-B", HERE / "score_o_tokens.py", "--arm", arm, "--model-dir", model_dir,
           "--expect-sha", expect, "--output", out]
    run(cmd + (["--cast-bf16"] if cast else []))


def canary_stage(arms_scored, arms_checked) -> None:
    rc = run([PY, "-u", "-B", HERE / "analyse_o_side.py", "--canary-only", *arms_checked], ok=(0, 3))
    if rc == 3:
        move_aside([SCORES / f"{a}.json" for a in arms_scored],
                   f"canary/provenance FAILED on {list(arms_checked)}; the permitted rerun rescores {list(arms_scored)}")
        raise SystemExit("canary FAILED: harness failure (PREREG section 0; AMENDMENT_1)")


def built_arm(arm: str, build_cmd: list) -> None:
    """Build (unless a complete score exists), score, check the fresh score, and only then delete the weights."""
    d = B.ARMS / arm
    if d.parent.resolve() != B.ARMS.resolve():
        raise SystemExit(f"refusing arm path outside arms/: {d}")
    reg = A.REGISTERED_SHA.get(arm)
    done = (SCORES / f"{arm}.json").exists() and not score_problems(arm, expected_sha(arm))
    if not done:
        if (d / "REBUILD_MISMATCH.json").exists():
            raise SystemExit(f"{arm}: REBUILD_MISMATCH.json present: this arm must not be scored")
        if d.exists() and read_build(d) is None:
            shutil.rmtree(d)
            print(f"  {arm}: removed an incomplete build directory (BUILD.json missing or unreadable)", flush=True)
        if d.exists():
            b = read_build(d)
            if reg is not None and (b.get("match") is not True or b["weight_sha256"] != reg):
                raise SystemExit(f"{arm}: BUILD.json does not certify the registered hash; refusing")
            if not (d / "model.safetensors").exists():
                raise SystemExit(f"{arm}: BUILD.json present but no weights and no complete score: inspect")
        else:
            wait_ram(f"building {arm}")
            run([PY, "-u", "-B", HERE / "build_o_arms.py"] + build_cmd)
            fsync_tree(d)
        expect = reg if reg is not None else read_build(d)["weight_sha256"]
        score(arm, d, expect)
        problems = score_problems(arm, expect)
        if problems:                        # weights still present, so the rerun can move it aside and rescore
            raise SystemExit(f"{arm}: fresh score file fails provenance ({'; '.join(problems)}); weights kept")
    else:
        print(f"  {arm}: complete score file present, skipping", flush=True)
    if (d / "model.safetensors").exists():
        run([PY, "-u", "-B", HERE / "build_o_arms.py", "delete", arm])


def main() -> int:
    preflight()
    if "torch" in sys.modules:
        raise SystemExit("torch was imported into the driver; it must stay out of this long-lived process")
    with lease(label="oside-001", job="11 arms x 6741 held-out rows, per-token NLL (+6 CPU builds)", timeout_seconds=7200):
        score("BASE", SV / "BASE", A.REGISTERED_SHA["BASE"])
        score("OPALLO", SV / "OPALLO", A.REGISTERED_SHA["OPALLO"])
        canary_stage(STAGE1, STAGE1)
        score("OPALLO-BF16", SV / "OPALLO", A.REGISTERED_SHA["OPALLO-BF16"], cast=True)
        score("SPARSE", SV / "SPARSE", A.REGISTERED_SHA["SPARSE"])
        score("ADD-100", SV / "ADD-100", A.REGISTERED_SHA["ADD-100"])
        canary_stage(STAGE2, STAGE1 + STAGE2)
        for arm in ("ADD-O50", "H1", "H1c"):
            built_arm(arm, ["rebuild", arm])
        for s in (23, 29, 41):
            built_arm(f"SEED{s}", ["seed", str(s)])
    run([PY, "-u", "-B", HERE / "analyse_o_side.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
