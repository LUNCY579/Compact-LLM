"""Reviewed-callable CPU runner; Root supplies fresh inventory before launch.

No standalone CLI/default inventory permit. Reuses the original cooperative
600-second/512-MiB ResourceWatch, never imports the whole corpus preparer.
"""
from copy import deepcopy
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import time
from arithmetic_token_projection import project

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == "rineya_synth_centralized_reasoning_v1")
OUTPUT = Path("C:/Users/clclc/BusinessProject/synth-arithmetic-tokenmeasurement-20261009-attempt001")
PREPARER_SHA256 = "7a8985da55a627856b66d9b649f8fb5d1053deb65b9f5c8d0f72869ebb086c9d"
POPULATION_SHA256 = "2992454925eecfabda4a6046d1d59ef78e718c1639d58714aa1404f90888f5c3"


def require(value, message):
    if not value: raise ValueError(message)


def _protect(event, args):
    if event == "import" and args and args[0].split(".", 1)[0] in ("torch", "transformers", "tensorflow", "jax", "flax", "psutil"):
        raise PermissionError("CPU-only measurement forbids model/GPU framework imports")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])); low = path.as_posix().lower()
        require(path.name.lower() not in ("config.json", "generation_config.json")
                and path.suffix.lower() not in (".safetensors", ".pt", ".bin")
                and "/check/" not in low and "/final/" not in low, "weights/model config/protected-bank read forbidden")


def _watch():
    raw = (PACKAGE / "prepare_corpus.py").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PREPARER_SHA256, "exact original ResourceWatch source")
    tree = ast.parse(raw)
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in ("peak_working_set_bytes", "ResourceWatch")]
    require(len(definitions) == 2, "original ResourceWatch AST slice")
    import ctypes  # Original CPU process-memory diagnostic; no GPU/model API.
    def no_frameworks():
        require(not any(name.split(".", 1)[0] in ("torch", "transformers", "tensorflow", "jax", "flax") for name in sys.modules), "CPU-only process")
    namespace = {"ctypes": ctypes, "os": os, "sys": sys, "time": time, "require": require,
                 "no_frameworks": no_frameworks, "CPU_WALL_LIMIT_SECONDS": 600,
                 "CPU_PEAK_LIMIT_BYTES": 512 * 1024 ** 2}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(PACKAGE / "prepare_corpus.py"), "exec"), namespace)
    return namespace["ResourceWatch"]()


def _inventory(check, stage):
    value = check(stage)
    require(type(value) is dict and value.get("verified") is True
            and type(value.get("identity_sha256")) is str and len(value["identity_sha256"]) == 64,
            "fresh verified source/runtime/package/four-asset inventory required")
    return deepcopy(value)


def _write(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        require(stream.write(raw) == len(raw), "complete write-once CPU result")
        stream.flush(); os.fsync(stream.fileno())
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def run(*, population_path, model_dir, inventory_check):
    require(callable(inventory_check), "Root-bound live inventory helper required before tokenizer import")
    require(OUTPUT.is_absolute() and OUTPUT.parent.is_dir() and not OUTPUT.exists() and not OUTPUT.is_symlink(), "fresh exact owned CPU output leaf")
    sys.addaudithook(_protect)
    old_path = list(sys.path); watch = None; result = None; error = None; before = after = None
    OUTPUT.mkdir()
    try:
        before = _inventory(inventory_check, "before_cpu_tokenizer_import")
        if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
        import arithmetic_tokenmeasurement as measurement
        guard, tokens = measurement.modules()
        watch = _watch(); watch.check("before_cpu_loader")
        raw = Path(population_path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == POPULATION_SHA256, "exact original512 arithmetic population bytes")
        population = json.loads(raw, object_pairs_hook=guard._pairs, parse_constant=guard._bad)
        policy = guard.policy()
        tokenizer = tokens.load_local_tokenizer(model_dir=Path(model_dir))
        watch.check("after_cpu_loader")
        result = measurement.measure_population(population, tokenizer, policy, check=watch.check)
        after = _inventory(inventory_check, "after_all_cpu_arithmetic_rows")
        require(after["identity_sha256"] == before["identity_sha256"], "source/runtime/package/assets changed during CPU measurement")
        watch.check("after_inventory_recheck")
    except Exception as failure:
        error = {"error_type": type(failure).__name__, "error": str(failure)}
    finally:
        sys.path[:] = old_path
        files = {}
        try:
            if result is not None:
                files["policy_prompt_tokens.json"] = _write(OUTPUT / "policy_prompt_tokens.json", {"rows": result["policy_rows"], "native_scoring_authorized": False})
                files["oracle_CPU_only.json"] = _write(OUTPUT / "oracle_CPU_only.json", {"rows": result["oracle_rows_CPU_only"], "oracle_exposed_CPU_only": True})
                metadata = project(result, files["policy_prompt_tokens.json"], files["oracle_CPU_only.json"])
                files["measurement.json"] = _write(OUTPUT / "measurement.json", metadata)
            if watch is not None: watch.check("after_result_files_written")
        except Exception as failure:
            error = error or {"error_type": type(failure).__name__, "error": str(failure)}
        resources = watch.record() if watch is not None else None
        if resources is not None and resources["cpu_budget_satisfied"] is not True:
            error = error or {"error_type": "ResourceFailure", "error": "original CPU budget or telemetry failed"}
        receipt = {"kind": "SYNTH_ARITHMETIC_CPU_TOKEN_MEASUREMENT_RUN001", "files": files,
            "status": "FAILED_CPU_MEASUREMENT" if error else result["status"] if result is not None else "FAILED_BEFORE_MEASUREMENT",
            "error": error, "inventory_before": before, "inventory_after": after, "resources": resources,
            "import_path_restored": sys.path == old_path, "native_launch_authorized": False,
            "native_scoring_authorized": False, "model_GPU_optimizer_or_M0_performed": False}
        _write(OUTPUT / "RUN_RECEIPT.json", receipt)
    return receipt
