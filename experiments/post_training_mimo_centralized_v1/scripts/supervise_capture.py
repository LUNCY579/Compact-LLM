"""Windows-contained short greedy qualification under its distinct frozen profile."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
from experiments.post_training_mimo_centralized_v1.scripts import greedy_capture as control

def main():
    control.require(os.name == "nt", "capture containment is registered for Windows only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); cfg = control.load_config(args.config)
    root = Path(__file__).resolve().parents[3]
    snapshot = control.source_snapshot(root, cfg)
    # Shared watchdog and package initializers are checked before importing them.
    from experiments.opallo_native_collector_v1.supervisor import run_bounded
    output = args.output.resolve(); supervision = output.with_name(output.name + "_supervisor")
    control.require(not output.exists() and not supervision.exists(), "capture run path already exists")
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True, timeout=15).strip())
    common = common.resolve() if common.is_absolute() else (root / common).resolve()
    lock = common / "rineya-central-gpu0.lock"
    with lock.open("x", encoding="utf-8") as stream: stream.write(str(os.getpid()))
    try:
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        control.require(not any(term in line.lower() for line in active.splitlines() for term in ("python", "torch", "llama", "ollama", "vllm", "train", "comfy")), "competing GPU runtime")
        command = [sys.executable, "-B", "-m", "experiments.post_training_mimo_centralized_v1.scripts.qualify_capture",
                   "--config", str(args.config.resolve()), "--output", str(output), "--go-path", str(supervision / "GO.json")]
        supervision.mkdir(parents=True)
        limits = {name: cfg[name] for name in ("resource_profile", "host_free_gib", "worker_free_gib", "worker_rss_limit_gib",
                    "gpu_free_gib", "gpu_worker_free_gib", "disk_free_gib", "wall_seconds", "admission_budget")}
        control.write(supervision / "LAUNCH.json", {"command": command, "config_sha256": control.CONFIG_SHA256,
                    "source_sha": snapshot["code_sha"], "supervisor_sha256": control.file_identity(__file__)["sha256"],
                    "watchdog_sha256": cfg["reused_source_files"]["experiments/opallo_native_collector_v1/supervisor.py"]["sha256"],
                    "resource_limits": limits, "observed_gpu_processes": active.splitlines()})
        run_bounded(command, root, supervision, seconds=600, min_free_gib=1.25, max_rss_gib=2.0,
                    go={"config_sha256": control.CONFIG_SHA256})
        return 0
    except Exception as error:
        supervision.mkdir(parents=True, exist_ok=True)
        control.write(supervision / "FAILURE_RECEIPT.json", {"status": "BLOCKED", "reason": str(error)})
        print(str(error), flush=True); return 2
    finally:
        lock.unlink()

if __name__ == "__main__":
    sys.exit(main())
