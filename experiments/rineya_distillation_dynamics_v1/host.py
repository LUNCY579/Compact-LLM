"""Independent bounded worker supervision; preserves existing conservative floors."""
from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .contracts import require, write_new

HOST_ADMISSION_GIB = 7.4
HOST_RUNNING_GIB = 6.0
GPU_ADMISSION_GIB = 4.0
GPU_RUNNING_GIB = 0.5
DISK_GIB = 4.0
WORKER_TREE_RSS_GIB = 4.0
WALL_SECONDS = 1800


def runtime_check():
    require(os.name == "nt", "native host is Windows-only pending a separately qualified port")
    require(sys.version_info[:2] == (3, 11), "existing qualified Python 3.11 environment required")
    for package, version in {"torch": "2.10.0+cu126", "transformers": "4.57.1",
                             "tokenizers": "0.22.2", "safetensors": "0.7.0"}.items():
        require(importlib.metadata.version(package) == version, "native runtime mismatch: " + package)
    import psutil  # Existing host dependency; never install/upgrade it automatically.
    return psutil


def admission(output_parent):
    psutil = runtime_check()
    for _ in range(2):
        require(psutil.virtual_memory().available / 2**30 >= HOST_ADMISSION_GIB, "HOST_RAM_ADMISSION")
        time.sleep(.2)
    require(shutil.disk_usage(output_parent).free / 2**30 >= DISK_GIB, "DISK_ADMISSION")
    free = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                                    "--format=csv,noheader,nounits"], text=True, timeout=15)
    require(float(free.strip()) / 1024 >= GPU_ADMISSION_GIB, "GPU_ADMISSION")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=process_name",
                                      "--format=csv,noheader,nounits"], text=True, timeout=15)
    require(not any(x in active.lower() for x in ("python", "ollama", "llama", "vllm", "comfy")),
            "competing GPU runtime; do not stop or kill it")
    return psutil


def worker_guard(output, started):
    import psutil
    import torch
    require(time.monotonic()-started < WALL_SECONDS, "WORKER_WALL_BUDGET")
    require(psutil.virtual_memory().available / 2**30 >= HOST_RUNNING_GIB, "WORKER_HOST_HEADROOM")
    require(shutil.disk_usage(output).free / 2**30 >= DISK_GIB, "WORKER_DISK_HEADROOM")
    if torch.cuda.is_initialized():
        require(torch.cuda.mem_get_info(0)[0] / 2**30 >= GPU_RUNNING_GIB, "WORKER_GPU_HEADROOM")


def launch(registration, output, *, arm=None, selection=None, confirmation=None,
           allow_model_load=False, allow_training=False, ack_host_available=False, allow_confirmation=False):
    require(allow_model_load and ack_host_available, "explicit model-load and repaired/available-host acknowledgement required")
    require((arm is not None) != (selection is not None), "one run mode required")
    require((arm is None or allow_training), "training includes disposable qualification updates; explicit authorization required")
    require(selection is None or (allow_confirmation and confirmation), "explicit confirmation permission/catalog required")
    from .registry import check_registration
    r, plan, tasks = check_registration(registration)
    del tasks
    root = Path(__file__).resolve().parents[2]
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                        cwd=root, text=True).strip(), "native source checkout must be clean")
    output = Path(output).resolve()
    require(not output.exists(), "native output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    psutil = admission(output.parent)
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=root, text=True).strip())
    common = common if common.is_absolute() else (root / common).resolve()
    lock = common / "rineya-central-gpu0.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    process = None
    try:
        # Recheck after acquiring the shared lock; never steal/delete another owner's lock.
        admission(output.parent)
        output.mkdir()
        if selection is not None:
            from .contracts import digest
            from .selection import spend_confirmation
            spend_confirmation(selection, digest(r), r["binding"]["confirmation_sha256"])
        token = uuid.uuid4().hex
        command = [sys.executable, "-B", "-m", "experiments.rineya_distillation_dynamics_v1.native",
                   "--registration", str(Path(registration).resolve()), "--output", str(output), "--token", token]
        if arm is not None:
            command += ["--arm", arm]
        else:
            command += ["--selection", str(Path(selection).resolve()), "--confirmation", str(Path(confirmation).resolve())]
        env = dict(os.environ, RDD001_WORKER_TOKEN=token, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        write_new(output / "SUPERVISOR_STARTED.json", {"wall_seconds": WALL_SECONDS,
                   "worker_tree_rss_gib": WORKER_TREE_RSS_GIB, "mode": "train" if arm else "confirm"})
        started = time.monotonic()
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                require(time.monotonic()-started < WALL_SECONDS, "SUPERVISOR_WALL_BUDGET")
                try:
                    parent = psutil.Process(process.pid)
                    procs = [parent] + parent.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in procs if p.is_running())
                    require(rss / 2**30 <= WORKER_TREE_RSS_GIB, "SUPERVISOR_RSS_BUDGET")
                except psutil.NoSuchProcess:
                    pass
                require(psutil.virtual_memory().available / 2**30 >= HOST_RUNNING_GIB, "SUPERVISOR_HOST_HEADROOM")
                require(shutil.disk_usage(output).free / 2**30 >= DISK_GIB, "SUPERVISOR_DISK_HEADROOM")
                time.sleep(.5)
        require(process.returncode == 0, "native worker failed; inspect worker.log (do not auto-retry)")
        write_new(output / "SUPERVISOR_RESULT.json", {"status": "WORKER_EXITED_ZERO_NOT_EFFICACY",
                                                      "wall_seconds": time.monotonic()-started})
    except BaseException as exc:
        if process is not None and process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                for child in reversed(parent.children(recursive=True)):
                    child.kill()
                parent.kill()
            except psutil.NoSuchProcess:
                pass
            process.wait(timeout=15)
        if output.exists():
            write_new(output / "SUPERVISOR_FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        lock.unlink()  # This invocation created it exclusively.
