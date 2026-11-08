"""Full-prefix frozen capture with explicit permission and hard resource limits."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from experiments.paired_lf_v1.bank import Bank
from experiments.paired_lf_v1.contracts import CLOSED, digest, exclusive, file_hash, manifest, read_json, require, write_json
from experiments.paired_lf_v1.data import build
from .host import ROOT, REPO, PARENT, capture, preflight


def binding() -> dict:
    from experiments.paired_lf_v1.runtime import ROOT as FRAMEWORK
    return {"parent": PARENT, "host_source": manifest(ROOT), "framework_source": manifest(FRAMEWORK)}


def approve(path: Path, source: Path) -> None:
    permit = read_json(path)
    require(permit.get("approved") is True and bool(permit.get("approved_by")) and
            bool(permit.get("authorization_reference")) and permit.get("operation") == "paired_frozen_capture" and
            permit.get("binding") == binding() and permit.get("claims") == CLOSED and
            permit.get("source_directory") == str(source.resolve()) and
            permit.get("dataset_sha256") == digest(build()[0]), "unapproved/stale capture permit")


def main() -> int:
    p = argparse.ArgumentParser(description="PLF-HOST-001: B28 / E24-D4, frozen diagnostic only")
    p.add_argument("stage", choices=("preflight", "permit-template", "capture", "_worker"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--permit", type=Path)
    args = p.parse_args()
    try:
        if args.stage == "_worker":
            require(args.permit is not None, "worker needs permit")
            approve(args.permit, args.source)
            resolved = args.out.absolute()
            require(not any(x.is_symlink() for x in (resolved, *resolved.parents)) and
                    resolved.resolve() != REPO and REPO not in resolved.resolve().parents and
                    resolved.resolve() not in REPO.parents, "worker output overlaps source or a symlink")
            if args.source is not None:
                input_root = args.source.resolve()
                require(resolved.resolve() != input_root and input_root not in resolved.resolve().parents and
                        resolved.resolve() not in input_root.parents, "worker output overlaps input tree")
            require(args.out.is_dir() and not (args.out / "result.json").exists(), "worker destination not fresh")
            import torch
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            rows, _ = build()
            bank_dir = args.out / "bank"
            bank_dir.mkdir(exist_ok=False)
            capture(args.source, rows, bank_dir)
            bank = Bank(bank_dir)
            write_json(args.out / "result.json", {"status": "CAPTURE_COMPLETE", "bank": bank.identity,
                       "claims": CLOSED, "optimizer_steps": 0, "cached_CED_qualified": False})
            return 0
        out = exclusive(args.out, REPO, args.source)
        if args.stage == "preflight":
            result = preflight(args.source)
            write_json(out / "preflight.json", result)
            print(result["status"])
            return 2 if result["status"] == "BLOCKED" else 0
        if args.stage == "permit-template":
            result = {"approved": False, "approved_by": "", "authorization_reference": "",
                      "operation": "paired_frozen_capture", "binding": binding(), "claims": CLOSED,
                      "source_directory": str(args.source.resolve()), "dataset_sha256": digest(build()[0])}
            write_json(out / "permit.json", result)
            return 0
        before, status, error, peak = binding(), "FAIL", None, 0
        child = None
        started = time.monotonic()
        try:
            require(args.permit is not None, "explicit capture permit required")
            approve(args.permit, args.source)
            check = preflight(args.source)
            require(check["status"] == "PREFLIGHT_ONLY", "; ".join(check["blockers"]))
            require(psutil.virtual_memory().available >= 10 * 1024**3, "10 GiB free host RAM required for bounded native loader")
            require(psutil.disk_usage(out).free >= 1024**3, "1 GiB free disk required")
            command = [sys.executable, "-m", "experiments.paired_lf_host_v1", "_worker", "--source", str(args.source.absolute()),
                       "--out", str(out), "--permit", str(args.permit.absolute())]
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            with (out / "worker.log").open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    require(time.monotonic() - started < 1800, "capture hard wall deadline exceeded")
                    try:
                        proc = psutil.Process(child.pid)
                        rss = sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)] if p.is_running())
                        peak = max(peak, rss)
                        require(rss < 10 * 1024**3, "capture RSS ceiling exceeded")
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(0.05)
                require(child.returncode == 0, f"capture worker failed: {child.returncode}")
            require(binding() == before, "source changed during capture")
            Bank(out / "bank")
            require((out / "result.json").is_file(), "capture result missing")
            status = "PASS"
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if child is not None and child.poll() is None:
                try:
                    proc = psutil.Process(child.pid)
                    for p in proc.children(recursive=True):
                        p.kill()
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
                child.wait(timeout=10)
        result = {"schema": "plf.capture-receipt/1", "status": status, "error": error,
                  "binding": before, "claims": CLOSED, "peak_sampled_rss": peak,
                  "wall_seconds": time.monotonic() - started, "optimizer_steps": 0,
                  "artifacts": {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}}
        result["sha256"] = digest(result)
        write_json(out / "receipt.json", result)
        require(status == "PASS", f"{error}; receipt retained")
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"PLF-HOST-001 BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
