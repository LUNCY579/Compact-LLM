#!/usr/bin/env python3
"""Prepare and run only the immutable D28-001 P1 on local exact Supa assets."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

HEAD = "cd6ef9bd49724da6879bfa4d945af3bfaa6d9758"
IMPLEMENTATION = "570fa72d5f07979a8e4a57954d1aa2006e607788e6f5ad684321868c0ca1ea74"
PACKAGE = "experiments/dual28_ced_v1"
LOCK = "experiments/architecture_benchmark_001/source_lock.json"
WEIGHT_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
CHECKPOINT_BYTES = (1156087424 * 2 + 979890816) * 4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_path(value):
    path = Path(value).expanduser().absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink() and not getattr(part, "is_junction", lambda: False)(),
                f"Symlink/junction paths are unsupported: {part}")
    return path.resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def environment(offline=False):
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT",
                "PYTEST_ADDOPTS", "PYTEST_PLUGINS", "VIRTUAL_ENV",
                "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               GIT_LFS_SKIP_SMUDGE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return env


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args],
                                   text=True, env=environment()).strip()


def execute(args, cwd, log, offline=False):
    print(f"Running {log.name}; full output: {log}", flush=True)
    with log.open("x", encoding="utf-8") as f:
        result = subprocess.run([str(a) for a in args], cwd=cwd, env=environment(offline),
                                stdout=f, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f"Command exited {result.returncode}; see {log}")


def host_check():
    require(platform.system() in {"Windows", "Linux"} and platform.machine().lower() in {"amd64", "x86_64"},
            "Use Windows/Linux x64 for this pinned +cpu lane; no macOS/ARM runtime substitution.")
    require((3, 11) <= sys.version_info[:2] <= (3, 13), "Use CPython 3.11, 3.12 or 3.13 (3.11 recommended).")
    require(platform.python_implementation() == "CPython", "Use CPython.")
    require(sys.maxsize > 2**32, "Use 64-bit Python.")


def verify_source(source, lock):
    require(lock["assets"]["model.safetensors"] == WEIGHT_SHA, "Unexpected source lock")
    for name, digest in lock["assets"].items():
        require(Path(name).name == name, "Unsafe asset name")
        path = safe_path(source / name)
        require(path.is_file(), f"Missing source asset: {path}")
        if name == "model.safetensors":
            require(path.stat().st_size == lock["weight_bytes"], "Weight size mismatch; pointers/other cores are invalid")
        require(sha(path) == digest, f"Source hash mismatch: {path}")


def check_checkout(repo):
    require(git(repo, "rev-parse", "HEAD") == HEAD, "Checkout must remain at the exact registered head")
    require(not git(repo, "status", "--porcelain", "--untracked-files=no"), "Tracked checkout files changed")


def python_at(root):
    return root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_check(root, log, receipt=None):
    code = (
        "import json, torch; from pathlib import Path; "
        "from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer; "
        "from experiments.dual28_ced_v1 import guards as g; "
        "torch.set_num_threads(4); "
        f"g.require(g.implementation() == {IMPLEMENTATION!r}, 'implementation changed'); "
        "b = g.binding(Path('Supa')); "
        f"g.require(b['checkout'] == {HEAD!r}, 'wrong checkout'); "
        "g.require(all(v is False for v in g.protocol()['permissions'].values()), 'campaign permission open'); "
    )
    if receipt is not None:
        code += f"print(g.validate_parent(Path({str(receipt)!r}), b)); "
    code += "print(json.dumps(b, sort_keys=True, indent=2))"
    execute([python_at(root), "-c", code], root / "checkout", log, offline=True)


def prepare(args):
    host_check()
    repo, source, root = map(safe_path, (args.repo, args.source, args.work_dir))
    require(not root.exists(), "Work directory already exists; choose a new directory and preserve the old attempt")
    require(root.parent.is_dir(), "Create the work directory's parent first")
    require(not root.is_relative_to(repo) and not root.is_relative_to(source), "Work directory must be outside repo and source")
    require(git(repo, "rev-parse", "--is-shallow-repository") == "false", "Fetch full history before preparing")
    sparse = subprocess.run(["git", "-C", str(repo), "config", "--bool", "core.sparseCheckout"],
                            capture_output=True, text=True, env=environment())
    require(sparse.returncode in {0, 1} and sparse.stdout.strip() != "true", "Sparse checkout is unsupported")
    require(git(repo, "rev-parse", HEAD + "^{commit}") == HEAD, "Fetch the registered commit first")
    lock = json.loads(git(repo, "show", HEAD + ":" + LOCK))
    print("Verifying all seven source assets before copying or installing...", flush=True)
    verify_source(source, lock)
    # Preparation reserve only. The official runner independently enforces its disk guard.
    need = CHECKPOINT_BYTES + sum((source / n).stat().st_size for n in lock["assets"]) + 8 * 1024**3
    require(shutil.disk_usage(root.parent).free >= need,
            "Need about 23 GB free for source copy, checkpoints and setup reserve; full checkout may need more")
    root.mkdir()
    try:
        write(root / "request.json", {"target_head": HEAD, "source": str(source), "source_hashes": lock["assets"],
                                      "launcher_sha256": sha(__file__), "official_execution": False})
        execute(["git", "-c", "core.autocrlf=false", "-C", repo, "worktree", "add", "--detach", root / "checkout", HEAD],
                root, root / "checkout.log")
        checkout = root / "checkout"
        check_checkout(checkout)
        destination = checkout / "Supa"
        require(not destination.exists(), "Supa destination already exists; refusing to replace checkout assets")
        destination.mkdir()
        for name in lock["assets"]:
            with (source / name).open("rb") as src, (destination / name).open("xb") as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
        verify_source(destination, lock)
        verify_source(source, lock)
        check_checkout(checkout)
        execute([sys.executable, "-m", "venv", root / "venv"], root, root / "venv.log")
        py = python_at(root)
        execute([py, "-m", "pip", "--isolated", "install", "--index-url", "https://download.pytorch.org/whl/cpu",
                 "torch==2.10.0+cpu"], root, root / "install_torch.log")
        execute([py, "-m", "pip", "--isolated", "install", "--index-url", "https://pypi.org/simple",
                 "transformers==4.51.3", "numpy==2.3.5", "safetensors==0.7.0", "pytest==9.0.2"],
                root, root / "install_libraries.log")
        execute([py, "-m", "pip", "check"], root, root / "pip_check.log")
        execute([py, "-m", "pip", "freeze", "--all"], root, root / "installed_packages.txt")
        runtime_check(root, root / "source_runtime_binding.json.log")
        execute([py, "-m", "experiments.dual28_ced_v1", "inspect"], checkout, root / "inspect.json.log", offline=True)
        execute([py, "-m", "pytest", "-q", PACKAGE + "/tests", "--junitxml=" + str(root / "scoped_tests.xml")],
                checkout, root / "scoped_tests.log", offline=True)
        check_checkout(checkout)
        write(root / "prepared.json", {"status": "PREPARED_NOT_P1", "target_head": HEAD,
                                       "official_execution": False, "p2_authorized": False})
        print(f"PREPARED_NOT_P1. To run P1: python local_p1.py run --work-dir \"{root}\"")
    except BaseException as exc:
        write(root / "preparation_failure.json", {"status": "PREPARATION_FAILED", "error": str(exc),
                                                  "official_execution": False})
        raise


def run_p1(args):
    host_check()
    root = safe_path(args.work_dir)
    prepared = json.loads((root / "prepared.json").read_text(encoding="utf-8"))
    require(prepared["status"] == "PREPARED_NOT_P1" and prepared["target_head"] == HEAD, "Invalid preparation record")
    checkout = root / "checkout"
    safe_path(checkout)
    check_checkout(checkout)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    logs = root / ("p1-" + stamp)
    logs.mkdir()
    output = checkout / PACKAGE / "outputs" / ("official-p1-" + stamp)
    try:
        runtime_check(root, logs / "binding_before.log")
        write(logs / "launch.json", {"status": "STARTING_P1", "head": HEAD, "output": str(output)})
        execute([python_at(root), "-m", "experiments.dual28_ced_v1", "qualify", "--source", "Supa", "--out", output],
                checkout, logs / "qualify.log", offline=True)
        check_checkout(checkout)
        runtime_check(root, logs / "validated_receipt.log", output / "result.json")
        write(logs / "completed.json", {"status": "P1_RECEIPT_VALIDATED", "result": str(output / "result.json"),
                                       "p2_executed": False, "campaigns_opened": False})
        print(f"Official P1 receipt validated: {output / 'result.json'}\nStopped after P1; no P2 executed.")
    except BaseException as exc:
        write(logs / "launcher_failure.json", {"status": "FAILED_OR_INTERRUPTED", "error": str(exc),
                                               "output": str(output), "p2_executed": False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Copy exact local source and install/check runtime; no P1 execution")
    prep.add_argument("--repo", required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument("--work-dir", required=True)
    run = sub.add_parser("run", help="Foreground official P1 only, in an already prepared work directory")
    run.add_argument("--work-dir", required=True)
    args = p.parse_args()
    try:
        (prepare if args.command == "prepare" else run_p1)(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
