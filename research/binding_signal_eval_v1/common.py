"""Stdlib-only contracts for a fixed, gold-free two-model DEV baseline."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

EXPERIMENT = "BINDING-DEV-BASELINE-001"
MODEL_SHAS = {"source": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
              "lora": "e025d88a1c18b1127bb58069e478ce2f3ee9609382fdcdcc10864018f9c4bae4"}
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
LIMITS = {"workers": 2, "worker_seconds": 900, "total_worker_seconds": 1800,
          "rows": 768, "reserved_generated_tokens": 18432, "retries": 0, "cloud_spend": 0,
          "total_owned_work_seconds": 3600, "new_disk_bytes": 268435456,
          "free_disk_floor_bytes": 15032385536, "forward_calls": 19200, "processed_input_tokens": 225540}
INPUT_KEYS = {"id", "parent_id", "prompt", "prompt_ids"}
PLAN_KEYS = {"schema", "record", "experiment", "pilot_root", "pilot_head", "python_path", "output_root",
             "input_path", "input_sha256", "source_dir", "sidecars_sha256", "models", "code_sha256",
             "pilot_code_sha256", "runtime_versions"}
PILOT_CODE = {f"experiments/model_quality_gpu_v1/{name}.py" for name in ("supervisor", "resources", "integrity")}
HERE = Path(__file__).resolve().parent
NUMERICS = {"seed": 17, "dtype": "float32", "attention": "frozen_RAB_eager_DenseReference",
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest", "greedy": True, "max_new_tokens": 24,
            "max_sequence": 2048, "truncation": False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def absolute(value, *, file=False, directory=False):
    require(isinstance(value, (str, Path)), "absolute path is required")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts and
            not any(p.is_symlink() for p in (path, *path.parents)), "unsafe/nonabsolute path")
    if file:
        require(path.is_file(), f"missing input file: {path.name}")
    if directory:
        require(path.is_dir(), f"missing directory: {path}")
    return path.resolve()


def sha(path):
    with absolute(path, file=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def decode(text):
    def bad(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=bad)


def read(path):
    path = absolute(path, file=True)
    require(path.stat().st_size <= 16 * 1024**2, "JSON exceeds bounded reader")
    value = decode(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), "JSON object required")
    return value


def write_new(path, value):
    path = absolute(path)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())


def read_jsonl(path):
    path = absolute(path, file=True)
    require(path.stat().st_size <= 32 * 1024**2, "JSONL exceeds bounded reader")
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            require(0 < len(line) <= 1024**2 and bool(line.strip()), "invalid JSONL line")
            value = decode(line)
            require(isinstance(value, dict), "JSONL object required")
            rows.append(value)
            require(len(rows) <= 384, "more than the registered DEV rows")
    return rows


def validate_inputs(rows):
    require(len(rows) == 384, "complete 384-row DEV input required")
    ids, worlds = set(), Counter()
    for row in rows:
        require(set(row) == INPUT_KEYS, "gold/category/extra fields are forbidden in worker input")
        require(isinstance(row["id"], str) and row["id"] and row["id"] not in ids,
                "duplicate/missing DEV row ID")
        require(isinstance(row["parent_id"], str) and row["parent_id"], "DEV world ID missing")
        ids.add(row["id"]); worlds[row["parent_id"]] += 1
        prompt = row["prompt"]
        require(isinstance(prompt, list) and len(prompt) == 2 and
                all(isinstance(m, dict) and set(m) == {"role", "content"} for m in prompt) and
                [m["role"] for m in prompt] == ["system", "user"] and
                all(isinstance(m["content"], str) and m["content"].strip() for m in prompt),
                "worker prompt must contain only system and user messages")
        tokens = row["prompt_ids"]
        require(isinstance(tokens, list) and bool(tokens) and all(type(t) is int and 0 <= t < 151936 for t in tokens),
                "invalid compiled prompt IDs")
        require(len(tokens) + 24 <= 2048,
                "prompt plus registered generation reserve exceeds context; truncation forbidden")
    require(len(worlds) == 32 and set(worlds.values()) == {12}, "DEV requires 32 worlds with 12 views each")
    return rows


def runtime_versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors")}}


def pilot_code_files(root):
    root = absolute(root, directory=True)
    rab = root / "experiments/architecture_benchmark_001"
    paths = set(rab.rglob("*.py"))
    paths.update(rab / name for name in ("protocol.json", "source_lock.json", "requirements.txt", "requirements-native.txt"))
    for name in read(rab / "protocol.json").get("amendments", []):
        path = rab / name
        require(path.resolve().is_relative_to(rab.resolve()), "RAB amendment escapes frozen package")
        paths.add(path)
    paths.update(root / name for name in PILOT_CODE)
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)}


def validate_plan(plan):
    require(isinstance(plan, dict) and set(plan) == PLAN_KEYS, "worker plan has unknown fields (gold paths are forbidden)")
    require(plan["schema"] == 1 and plan["record"] == "BINDING_DEV_BASELINE_PLAN" and plan["experiment"] == EXPERIMENT,
            "wrong baseline plan")
    for key in ("pilot_root", "source_dir"):
        absolute(plan[key], directory=True)
    for key in ("input_path", "python_path"):
        absolute(plan[key], file=True)
    absolute(plan["output_root"])
    require(Path(plan["input_path"]).name == "DEV_INPUTS.jsonl", "only the admitted gold-free input file is a worker input")
    require(set(plan["sidecars_sha256"]) == set(SIDECARS), "incomplete original tokenizer sidecars")
    require(set(plan["models"]) == set(MODEL_SHAS), "only fixed source and qualified LoRA models are admitted")
    require(set(plan["code_sha256"]) == {"common.py", "probe.py"} and
            set(plan["pilot_code_sha256"]) == set(pilot_code_files(plan["pilot_root"])),
            "incomplete worker/supervisor code inventory")
    require(set(plan["runtime_versions"]) == {"python", "torch", "transformers", "safetensors"}, "runtime pin missing")
    for role, model in plan["models"].items():
        require(set(model) == {"path", "weight_sha256", "receipt_sha256"} and model["weight_sha256"] == MODEL_SHAS[role],
                "unregistered model or model plan fields")
        absolute(model["path"], directory=True)
        require(set(model["receipt_sha256"]) == (set() if role == "source" else {"ARM.json", "MATERIALIZATION.json"}),
                "model provenance receipt inventory differs")
    require(absolute(plan["models"]["source"]["path"]) == absolute(plan["source_dir"]), "source tokenizer/model paths differ")
    return plan


def validate_grant(plan_path, grant):
    keys = {"schema", "record", "experiment", "decision", "authorization_reference", "plan_path", "plan_sha256", "limits",
            "registration_path", "registration_sha256", "data_admission_path", "data_admission_sha256",
            "session_start_unix", "deadline_unix"}
    require(set(grant) == keys and grant["schema"] == 1 and grant["record"] == "BINDING_DEV_BASELINE_GRANT" and
            grant["experiment"] == EXPERIMENT and grant["decision"] == "APPROVED" and
            isinstance(grant["authorization_reference"], str) and grant["authorization_reference"].strip(),
            "explicit baseline grant is missing")
    require(absolute(grant["plan_path"]) == absolute(plan_path) and grant["plan_sha256"] == sha(plan_path),
            "grant does not bind this exact baseline plan")
    require(grant["limits"] == LIMITS and all(type(v) is int for v in grant["limits"].values()),
            "baseline grant changes the registered envelope")
    for name in ("registration", "data_admission"):
        require(sha(grant[name + "_path"]) == grant[name + "_sha256"], "registration/data-admission binding changed")
    admission = read(grant["data_admission_path"])
    require(admission.get("status") == "ADMITTED_FOR_NAMED_EXPLORATORY_SCOPE" and
            admission.get("training_permission") is False, "data is not admitted for the named non-training baseline scope")
    require(all(type(grant[k]) in (int, float) and math.isfinite(grant[k]) for k in ("session_start_unix", "deadline_unix")) and
            grant["session_start_unix"] > 0 and
            math.isclose(grant["deadline_unix"] - grant["session_start_unix"], 3600, rel_tol=0, abs_tol=.001),
            "grant session envelope differs from the registered 3600 seconds")


def code_snapshot(plan):
    actual = {name: sha(HERE / name) for name in plan["code_sha256"]}
    pilot = pilot_code_files(plan["pilot_root"])
    require(actual == plan["code_sha256"] and pilot == plan["pilot_code_sha256"], "worker/supervisor code identity changed")
    return {"worker": actual, "pilot": pilot}


def model_snapshot(plan, role):
    model = plan["models"][role]; folder = Path(model["path"]); source = Path(plan["source_dir"])
    weight = sha(folder / "model.safetensors")
    require(weight == model["weight_sha256"], "model weight bytes differ from fixed baseline identity")
    sidecars = {name: sha(folder / name) for name in SIDECARS}
    require(sidecars == plan["sidecars_sha256"] == {name: sha(source / name) for name in SIDECARS},
            "model/original tokenizer sidecars differ")
    receipts = {name: sha(folder / name) for name in model["receipt_sha256"]}
    require(receipts == model["receipt_sha256"], "model provenance receipt hash differs")
    if role == "lora":
        for name in receipts:
            receipt = read(folder / name)
            require(receipt["source_weight_sha256"] == MODEL_SHAS["source"] and
                    receipt["out_weight_sha256"] == receipt["arm_weight_sha256"] == weight and
                    receipt["stored_dtype"] == "float32", "qualified LoRA materialization provenance differs")
    return {"weight_sha256": weight, "sidecars_sha256": sidecars, "receipts_sha256": receipts}


def disk_guard(output_root):
    path = Path(output_root)
    while not path.exists():
        path = path.parent
    require(shutil.disk_usage(path).free >= LIMITS["free_disk_floor_bytes"], "unchanged 14 GiB free-disk guard refused")


def phase_guard(plan, grant, *, seconds_needed=0):
    require(time.time() + seconds_needed <= grant["deadline_unix"], "full remaining baseline reservation does not fit session deadline")
    disk_guard(plan["output_root"])
    paths = set(HERE.rglob("*")) | set(Path(plan["output_root"]).rglob("*"))
    used = sum(path.stat().st_size for path in paths if path.is_file())
    require(used <= LIMITS["new_disk_bytes"], "baseline owned code/output disk budget exhausted")
    return {"owned_logical_bytes": used, "session_elapsed_seconds": time.time() - grant["session_start_unix"]}


def snapshot(plan, role):
    require(sha(plan["input_path"]) == plan["input_sha256"], "gold-free DEV inputs changed")
    require(runtime_versions() == plan["runtime_versions"], "pinned runtime versions changed")
    return {"input_sha256": plan["input_sha256"], "code": code_snapshot(plan),
            "runtime_versions": runtime_versions(), "model": model_snapshot(plan, role)}


def verify_terminal(supervisor, terminal):
    require(supervisor.get("status") == "EXITED" and type(supervisor.get("returncode")) is int and
            supervisor["returncode"] == 0, "baseline supervisor did not establish successful exit")
    require(terminal.get("status") == "COMPLETE" and terminal.get("complete") is True and
            terminal.get("failure") is None and terminal.get("before") == terminal.get("after"),
            "baseline worker terminal is incomplete, failed, or changed")
    require(type(supervisor.get("wall_seconds")) in (int, float) and math.isfinite(supervisor["wall_seconds"]) and
            0 <= supervisor["wall_seconds"] <= LIMITS["worker_seconds"], "baseline worker exceeded its registered time")


def verify_launch(out, plan_path, grant_path, plan, role, supervisor):
    """Tie the successful supervisor to this exact private worker and output slot."""
    out = absolute(out, directory=True)
    launch, spawn, claim = (read(out / name) for name in ("launch.json", "spawn.json", "worker_claim.json"))
    argv = launch.get("argv")
    require(isinstance(argv, list) and len(argv) >= 2 and all(isinstance(v, str) for v in argv), "invalid baseline launch argv")
    def option(name):
        require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "missing/duplicate baseline launch option")
        return argv[argv.index(name) + 1]
    require(absolute(launch["out"]) == out == absolute(plan["output_root"]) / role and
            absolute(launch["cwd"]) == absolute(plan["pilot_root"]) and launch["timeout_seconds"] == 900 and
            absolute(argv[0]) == absolute(plan["python_path"]) and absolute(argv[1]) == HERE / "probe.py" and
            absolute(option("--plan")) == absolute(plan_path) and absolute(option("--grant")) == absolute(grant_path) and
            absolute(option("--out")) == out and option("--model") == role and
            argv.count("--confirm-model-execution") == argv.count("--_worker") == 1,
            "baseline supervisor launch differs from plan/model/output")
    require(spawn["launch_sha256"] == claim["launch_sha256"] == digest(launch) and
            spawn["supervisor_pid"] == claim["supervisor_pid"] == launch["supervisor_pid"] and
            type(supervisor.get("child_pid")) is int and supervisor["child_pid"] == spawn["child_pid"] and
            (claim["pid"] == spawn["child_pid"] or (claim.get("launcher_redirected") is True and claim["parent_pid"] == spawn["child_pid"])),
            "baseline worker claim does not match its successful supervisor")
    return {stem + "_sha256": sha(out / (stem + ".json")) for stem in ("launch", "spawn", "worker_claim")}
