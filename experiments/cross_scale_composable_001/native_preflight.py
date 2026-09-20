"""Read-only resource inventory and tokenizer-only qualification; never load a model."""
from __future__ import annotations

import argparse
import ctypes
import datetime
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .build_calibration import build, canonical, digest, file_sha, require

PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "bitsandbytes",
            "peft", "accelerate", "numpy", "psutil")
TOKEN_FILES = ("config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.json", "merges.txt")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def versions():
    result = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return {"python": sys.executable, "python_version": sys.version, "packages": result}


def run_readonly(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return {"command": list(map(str, command)), "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def host_memory():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available",
                 "virtual_total", "virtual_available", "extended_available")]
        value = MemoryStatus()
        value.length = ctypes.sizeof(value)
        require(bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))),
                "cannot inspect host memory")
        return {"total_host_bytes": value.total, "available_host_bytes": value.available}
    import psutil
    value = psutil.virtual_memory()
    return {"total_host_bytes": value.total, "available_host_bytes": value.available}


def inspect_lease(path):
    if not path.exists():
        return {"path": path.as_posix(), "present": False, "mutated": False}
    raw = path.read_bytes()
    result = {"path": path.as_posix(), "present": True, "sha256": file_sha(path),
              "mutated": False, "steal_or_delete_permitted": False}
    try:
        owner = json.loads(raw)
        result["record"] = owner
        import psutil
        pid = owner.get("pid")
        result["owner_pid_exists"] = isinstance(pid, int) and psutil.pid_exists(pid)
        if result["owner_pid_exists"]:
            try:
                process = psutil.Process(pid)
                result["process"] = {"pid": pid, "name": process.name(),
                                     "created": process.create_time(), "exe": process.exe()}
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                result["process_inspection_error"] = type(exc).__name__
    except (ValueError, ImportError) as exc:
        result["inspection_error"] = type(exc).__name__
    return result


def provenance(root):
    return {"commit": run_readonly(["git", "-C", str(root), "rev-parse", "HEAD"]),
            "worktree": run_readonly(["git", "-C", str(root), "status", "--porcelain=v1"]),
            "helper_sha256": file_sha(Path(__file__)),
            "selection_helper_sha256": file_sha(Path(__file__).with_name("build_calibration.py"))}


def resource_inventory(root, nvidia_smi, runtime_paths):
    result = {"schema": 1, "kind": "READ_ONLY_RESOURCE_AUDIT", "observed_utc": now(),
              "source": provenance(root), "host": host_memory(),
              "lease": inspect_lease(Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"),
              "model_loads": 0, "model_forwards": 0, "optimizer_steps": 0,
              "current_process_runtime": versions(), "runtime_inventory": []}
    snippet = ("import importlib.metadata as m,json,sys; "
               "names=" + repr(list(PACKAGES)) + "; "
               "d={x.metadata['Name'].lower():x.version for x in m.distributions()}; "
               "print(json.dumps({'python':sys.executable,'python_version':sys.version,"
               "'packages':{n:d.get(n.lower()) for n in names}}))")
    for path in runtime_paths:
        result["runtime_inventory"].append(run_readonly([str(path), "-B", "-c", snippet]))
    result["gpu"] = run_readonly([str(nvidia_smi),
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits"])
    result["gpu_processes"] = run_readonly([str(nvidia_smi),
        "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])
    result["gpu_query_units"] = "MiB; WDDM per-process memory may be N/A"
    import psutil
    processes = []
    for process in psutil.process_iter(["pid", "ppid", "name", "exe"]):
        if "python" in (process.info["name"] or "").lower():
            processes.append(process.info)
    result["python_processes"] = processes
    result["audit_pid"] = os.getpid()
    result["guard_observations"] = {
        "student_launcher_host_minimum_bytes": 7 * 2**30,
        "student_child_host_minimum_bytes": 6 * 2**30,
        "donor_planning_host_minimum_bytes": 12 * 2**30,
        "student_launcher_host_pass_at_snapshot": result["host"]["available_host_bytes"] >= 7 * 2**30,
        "donor_planning_host_pass_at_snapshot": result["host"]["available_host_bytes"] >= 12 * 2**30,
        "does_not_license_launch": True,
        "child_after_import_guard_unmeasured": True}
    result["lease_protocol"] = {"source": "experiments/model_quality_gpu_v1/resources.py",
        "sha256": file_sha(root / "experiments/model_quality_gpu_v1/resources.py"),
        "mechanism": "create-only pid/token/created JSON; owner deletes only if token matches; never steals/deletes a suspected stale lock"}
    result["limitations"] = ["Instantaneous availability is not a peak fit qualification",
        "Missing shared lease does not prove no noncooperative GPU users",
        "WDDM includes UI clients and can conceal process names/per-process memory",
        "No native model runtime import or GPU context initialized by this audit"]
    require("torch" not in sys.modules, "unexpected torch import")
    return result


def reconstruct_prompts(calibration):
    build(calibration)
    train = Path(calibration["prompt_contract"]["system_rules_source"])
    rules = {}
    for line in train.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rule = row["prompt"][0]["content"]
            require(rules.setdefault(row["family"], rule) == rule, "different family rules")
    chosen = {row["id"]: row for row in calibration["rows"]}
    found = {}
    source = Path(calibration["selection"]["source_path"])
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] not in chosen:
                continue
            require(row["id"] not in found, "duplicate selected id")
            require(row["parent_id"] == chosen[row["id"]]["parent_id"], "wrong parent")
            prompt = [{"role": "system", "content": rules[row["family"]]},
                {"role": "user", "content": json.dumps(
                    {"before": row["before"], "event": row["event"]},
                    sort_keys=True, separators=(",", ":"))}]
            require(digest(prompt) == chosen[row["id"]]["prompt_sha256"], "wrong prompt digest")
            found[row["id"]] = prompt
    require(set(found) == set(chosen), "selected prompt absent")
    return found


def framework_state(allow_framework_import):
    imported = "torch" in sys.modules
    require(allow_framework_import or not imported, "tokenizer path imported torch")
    initialized = bool(sys.modules["torch"].cuda.is_initialized()) if imported else False
    require(not initialized, "tokenizer preflight unexpectedly initialized CUDA")
    return {"torch_imported": imported, "cuda_initialized": initialized,
            "framework_import_explicitly_allowed": allow_framework_import}


def tokenize(root, calibration_path, assets_path, source, expected_transformers,
             allow_framework_import=False):
    require("torch" not in sys.modules and "transformers" not in sys.modules,
            "tokenizer-only qualification requires a fresh process")
    os.environ.update(USE_TORCH="0", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    runtime = versions()
    require(runtime["packages"]["transformers"] == expected_transformers, "wrong Transformers pin")
    require(runtime["packages"]["tokenizers"] == "0.22.2", "wrong tokenizer pin")
    calibration = json.loads(calibration_path.read_bytes())
    require(calibration["role"] == "open_calibration", "wrong calibration role")
    prompts = reconstruct_prompts(calibration)
    assets = json.loads(assets_path.read_bytes())
    expected = {Path(item["path"]).name: item["sha256"]
                for item in assets["groups"]["student_base"] if Path(item["path"]).name in TOKEN_FILES}
    require(set(expected) == set(TOKEN_FILES), "incomplete tokenizer asset binding")
    actual = {name: file_sha(source / name) for name in TOKEN_FILES}
    require(actual == expected, "wrong local tokenizer/config bytes")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    require(tokenizer.is_fast and tokenizer.eos_token_id == 151645, "wrong tokenizer identity")
    framework_state(allow_framework_import)
    template = tokenizer.get_chat_template()
    rows = []
    for row in calibration["rows"]:
        rendered = tokenizer.apply_chat_template(prompts[row["id"]], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(rendered, add_special_tokens=False, return_attention_mask=True,
                            truncation=False, padding=False)
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        require(len(ids) == len(mask) and ids and all(value == 1 for value in mask), "invalid mask")
        rows.append({**row, "prompt": prompts[row["id"]], "rendered_prompt": rendered,
            "rendered_prompt_sha256": __import__("hashlib").sha256(rendered.encode()).hexdigest(),
            "input_ids": ids, "attention_mask": mask, "position_ids": list(range(len(ids))),
            "input_ids_sha256": digest(ids), "attention_mask_sha256": digest(mask), "tokens": len(ids)})
    cap = calibration["prompt_contract"]["maximum_native_prompt_tokens"]
    overflows = [row["id"] for row in rows if row["tokens"] > cap]
    require({name: file_sha(source / name) for name in TOKEN_FILES} == actual,
            "tokenizer files changed during qualification")
    framework = framework_state(allow_framework_import)
    return {"schema": 1, "kind": "TOKENIZER_ONLY_CALIBRATION_PREFLIGHT", "observed_utc": now(),
        "status": "BLOCKED_PROMPT_OVERFLOW" if overflows else "TOKEN_ELIGIBILITY_PASSED",
        "source": provenance(root), "runtime": runtime, "host_after": host_memory(),
        "calibration_file_sha256": file_sha(calibration_path),
        "calibration_sha256": calibration["calibration_sha256"],
        "dataset_sha256": calibration["dataset_sha256"], "tokenizer_assets": actual,
        "tokenizer_source": source.as_posix(), "tokenizer_class": type(tokenizer).__name__,
        "template_sha256": __import__("hashlib").sha256(template.encode()).hexdigest(),
        "rendering": {"add_generation_prompt": True, "enable_thinking": False,
                      "add_special_tokens": False, "truncation": False, "padding": False},
        "rows": rows, "encoded_rows_sha256": digest(rows), "row_count": len(rows),
        "tokens_total": sum(row["tokens"] for row in rows),
        "tokens_min": min(row["tokens"] for row in rows),
        "tokens_max": max(row["tokens"] for row in rows), "registered_cap": cap,
        "overflow_ids": overflows, "native_tokenizer_encodings": len(rows),
        "model_loads": 0, "model_forwards": 0, "optimizer_steps": 0,
        **framework, "does_not_qualify_model_runtime": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("resources", "tokenize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvidia-smi", type=Path)
    parser.add_argument("--runtime", type=Path, action="append", default=[])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expected-transformers", default="4.57.1")
    parser.add_argument("--allow-framework-import", action="store_true",
                        help="Allow a framework import for tokenizer packages requiring it; CUDA must remain uninitialized")
    parser.add_argument("--calibration", type=Path,
                        default=Path(__file__).parent / "evidence/CALIBRATION_BINDING.v2.json")
    parser.add_argument("--assets", type=Path,
                        default=Path(__file__).parent / "evidence/BINDING_ASSETS.json")
    args = parser.parse_args()
    require(not args.output.exists(), "create-only output exists")
    root = Path(__file__).resolve().parents[2]
    if args.mode == "resources":
        require(args.nvidia_smi is not None, "explicit observed NVIDIA utility path required")
        result = resource_inventory(root, args.nvidia_smi, args.runtime)
    else:
        require(args.source is not None, "explicit pinned local tokenizer directory required")
        result = tokenize(root, args.calibration, args.assets, args.source, args.expected_transformers,
                          args.allow_framework_import)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    print(json.dumps({"path": args.output.as_posix(), "sha256": file_sha(args.output),
                      "status": result.get("status"), "row_count": result.get("row_count"),
                      "tokens_max": result.get("tokens_max"), "model_forwards": 0}))
    if result.get("overflow_ids"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
