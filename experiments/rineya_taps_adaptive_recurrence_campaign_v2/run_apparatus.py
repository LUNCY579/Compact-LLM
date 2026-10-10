"""Reproducible evidence wrapper for explicitly no-model CPU apparatus commands.

This is not a native worker supervisor or a qualification bypass. CLI verbs are
restricted; no native-dev/model loader is available. Existing env is read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def sources():
    roots = [ROOT / "experiments/rineya_taps_adaptive_recurrence_v1",
             ROOT / "experiments/anchor_routed_middle_recurrence_v1",
             ROOT / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src", PACKAGE]
    result = {}
    for base in roots:
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".md") and not any(
                x in path.parts for x in ("runs", "__pycache__", ".pytest_cache")
            ):
                result[path.relative_to(ROOT).as_posix()] = sha(path)
    return result


def run(stage, out, source=None):
    out = out.resolve()
    if PACKAGE.resolve() not in out.parents:
        raise ValueError("evidence must stay inside campaign")
    if sum(p.stat().st_size for p in PACKAGE.rglob("*") if p.is_file()) > 1536 * 1024**2:
        raise RuntimeError("campaign must retain 512 MiB finalization reserve")
    if stage == "tests-v1":
        tail = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "experiments/rineya_taps_adaptive_recurrence_v1/tests", f"--junitxml={out / 'junit.xml'}"]
    elif stage == "tests-inherited":
        tail = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "experiments/anchor_routed_middle_recurrence_v1/tests", f"--junitxml={out / 'junit.xml'}"]
    elif stage == "fixture":
        tail = ["-m", "experiments.rineya_taps_adaptive_recurrence_v1", "fixture", "--out", str(out / "result")]
    elif stage == "native-preflight":
        if source is None:
            raise ValueError("verified local source required")
        tail = ["-m", "experiments.rineya_taps_adaptive_recurrence_v1", stage, "--source", str(source.resolve()), "--out", str(out / "result")]
    else:
        raise ValueError("unsupported stage")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "git_status": git("status", "--porcelain=v1"), "sha256": sources()}
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-B", *tail]
    started = time.perf_counter()
    receipt = {"stage": stage, "command": command, "cwd": str(ROOT),
               "started_utc": datetime.now(timezone.utc).isoformat(), "native_model_worker": False,
               "native_model_loads": 0, "optimizer_steps": 0}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (out / "stderr.log").open("w", encoding="utf-8") as stderr:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=600, check=False)
            receipt["exit_code"] = result.returncode
            receipt["status"] = "COMMAND_COMPLETE" if result.returncode == 0 else "COMMAND_NONZERO"
        except subprocess.TimeoutExpired:
            receipt.update(exit_code=None, status="APPARATUS_TIMEOUT")
    receipt.update(wall_seconds=time.perf_counter() - started,
                   ended_utc=datetime.now(timezone.utc).isoformat(),
                   inherited_bytes_unchanged=all(sha(ROOT / p) == h for p, h in manifest["sha256"].items() if not p.startswith(PACKAGE.relative_to(ROOT).as_posix())),
                   stdout_sha256=sha(out / "stdout.log"), stderr_sha256=sha(out / "stderr.log"))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt.get("exit_code") or (0 if receipt["status"] == "COMMAND_COMPLETE" else 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tests-v1", "tests-inherited", "fixture", "native-preflight"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.stage, args.out, args.source))
