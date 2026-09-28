"""Reuse the audited Opallo process-tree watchdog for one bounded Titan E0."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.opallo_native_collector_v1.supervisor import run_bounded
from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import sha, validate_config, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    validate_config(cfg)
    root = Path(__file__).resolve().parents[3]
    output = a.output.resolve()
    supervision = output.parent / (output.name + "_supervisor")
    if output.exists() or supervision.exists():
        raise ValueError("run path exists")
    common = Path(subprocess.check_output(["git","rev-parse","--git-common-dir"],cwd=root,text=True).strip())
    common = (root/common).resolve() if not common.is_absolute() else common.resolve()
    lock = common / "rineya-central-gpu0.lock"
    with lock.open("x",encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        active = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader,nounits"],
                                         text=True,timeout=15).strip()
        # WDDM lists desktop compositor/UI processes as compute apps. Retain the
        # entire inventory; block known competing model/training runtimes.
        conflicts = [line for line in active.splitlines() if any(term in line.lower() for term in
                     ("python", "torch", "llama", "ollama", "vllm", "train", "comfy"))]
        if conflicts:
            raise RuntimeError("competing GPU runtime: " + repr(conflicts))
        command = [sys.executable,"-B","-m","experiments.post_training_mimo_centralized_v1.scripts.qualify_native",
                   "--config",str(a.config.resolve()),"--output",str(output),"--go-path",str(supervision/"GO.json")]
        supervision.mkdir(parents=True)
        write(supervision/"LAUNCH.json",dict(command=command,config_sha256=sha(a.config),
              supervisor_sha256=sha(__file__),watchdog_sha256=sha(root/"experiments/opallo_native_collector_v1/supervisor.py"),
              observed_gpu_processes=active.splitlines(),source_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()))
        run_bounded(command,root,supervision,seconds=cfg["wall_seconds"],min_free_gib=cfg["worker_free_gib"],
                    max_rss_gib=4.0,go={"config_sha256":sha(a.config)})
        return 0
    except Exception as exc:
        supervision.mkdir(parents=True,exist_ok=True)
        write(supervision/"FAILURE_RECEIPT.json",{"status":"BLOCKED","reason":str(exc)})
        print(str(exc),flush=True)
        return 2
    finally:
        lock.unlink()


if __name__ == "__main__":
    sys.exit(main())
