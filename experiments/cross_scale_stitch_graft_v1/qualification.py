"""Source-bound, time-bounded CPU qualification. No GitHub Actions or native execution."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

from .artifacts import ROOT, file_sha256, new_directory, strict_json, write_json
from .contracts import digest, require


def source_paths(root: Path = ROOT) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and
                  not any(x in {"__pycache__", ".pytest_cache", "evidence"} for x in p.relative_to(root).parts) and
                  p.name != "SOURCE_MANIFEST.json")


def verify_source(root: Path = ROOT) -> dict:
    m = strict_json(root / "SOURCE_MANIFEST.json")
    require(m.get("schema") == "xsg-source-v1", "source manifest schema mismatch")
    expected = m["files"]
    actual = {p.relative_to(root).as_posix(): file_sha256(p) for p in source_paths(root)}
    require(all(not p.is_symlink() for p in root.rglob("*")), "symlink in source tree")
    require(expected == actual, "source inventory/hash mismatch")
    require(m["source_digest"] == digest(expected), "source digest mismatch")
    require(len(m["test_ids"]) > 0 and sorted(set(m["test_ids"])) == m["test_ids"], "invalid expected test inventory")
    return m


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def worker(path: Path) -> None:
    import torch
    import safetensors
    torch.set_num_threads(1)
    manifest = verify_source()
    repo_root = ROOT.parents[1]
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(repo_root))
    cases = list(flatten(suite))
    require(sorted(t.id() for t in cases) == manifest["test_ids"], "test inventory changed")
    stream = io.StringIO()
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(cases))
    verify_source()
    record = {"schema": "xsg-mechanics-worker-v1", "source_digest": manifest["source_digest"],
              "test_ids": manifest["test_ids"], "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
              "success": result.wasSuccessful() and not result.skipped,
              "wall_seconds": time.monotonic() - started, "log": stream.getvalue(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "safetensors": safetensors.__version__, "device": "cpu", "torch_threads": torch.get_num_threads()},
              "pretrained_model_calls": 0, "native_qualification": False}
    write_json(path, record)
    if not record["success"]:
        raise SystemExit(1)


def qualify(output: Path) -> dict:
    manifest = verify_source()
    output = new_directory(output)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-m", "experiments.cross_scale_stitch_graft_v1", "_worker", str(output / "worker.json")]
    start = time.monotonic()
    try:
        run = subprocess.run(command, cwd=ROOT.parents[1], env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        write_json(output / "QUALIFICATION.json", {"status": "TIMEOUT", "source_digest": manifest["source_digest"],
                   "native_qualification": False, "pretrained_model_calls": 0})
        raise RuntimeError("CPU qualification exceeded 120 seconds")
    worker_path = output / "worker.json"
    record = strict_json(worker_path) if worker_path.is_file() else {}
    verified = verify_source()
    ok = (run.returncode == 0 and record.get("success") is True and
          record.get("source_digest") == verified["source_digest"] and
          record.get("test_ids") == manifest["test_ids"] and
          record.get("tests_run") == len(manifest["test_ids"]) and
          all(record.get(k) == 0 for k in ("failures", "errors", "skips")))
    summary = {"schema": "xsg-qualification-v1", "status": "PASS_MECHANICS_ONLY" if ok else "FAIL",
               "source_digest": manifest["source_digest"], "tests": record.get("tests_run"),
               "failures": record.get("failures"), "errors": record.get("errors"), "skips": record.get("skips"),
               "wall_seconds": time.monotonic() - start, "environment": record.get("environment"),
               "worker_sha256": file_sha256(worker_path) if worker_path.is_file() else None,
               "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
               "pretrained_model_calls": 0, "native_qualification": False, "efficacy_claim": False}
    write_json(output / "QUALIFICATION.json", summary)
    require(ok, "qualification failed; inspect preserved logs")
    return summary
