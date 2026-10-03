"""Frozen local qualification bindings. No framework import or model call."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from rs_contract import canonical, digest, require, _pairs, _bad_constant

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
VERSION = "synth-native-contract-1"
AUTHORIZATION = {"scope": "mechanical_native_qualification_only",
                 "owner_authority": "user_goal_and_delegated_decisions_2026-10-03",
                 "production_training": False, "protected_bank_access": False, "paid_services": False}
RESOURCE_BOUNDS = {"worker_process_tree_RSS_GiB": 2.0, "system_available_reserve_GiB": 1.25,
                   "preworker_available_GiB": 3.3125, "GPU_free_at_admission_GiB": 4.0,
                   "GPU_free_during_work_GiB": 0.5, "disk_free_GiB": 4.0,
                   "whole_scope_seconds": 600, "external_watchdog_milliseconds": 100,
                   "CPU_sample_max_age_seconds": 0.2,
                   "GPU_probe_interval_seconds": 0.5, "GPU_probe_timeout_seconds": 2.0,
                   "GPU_sample_max_age_seconds": 2.0, "admission_seconds": 30,
                   "admission_poll_seconds": 2.0}
TOLERANCES = {"zero_adapter_max_error": 0.0, "CE_absolute_error": 2e-6,
              "selected_logprob_max_error": 0.03125, "selected_logprob_mean_error": 0.01,
              "reference_gradient_atol": 1e-6, "reference_gradient_rtol": 1e-5,
              "restoration": "bitwise"}
RUNTIME = {"python": "3.11.15", "torch": "2.6.0+cu124", "transformers": "5.14.1",
           "safetensors": "0.8.0", "psutil": "7.2.2"}
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_HUB_KERNELS": "NO",
               "USE_TORCH": "1", "USE_TF": "0", "USE_FLAX": "0", "TOKENIZERS_PARALLELISM": "false",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
MAX_JSON = 8 * 1024 * 1024


def filehash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path, limit: int = MAX_JSON):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "bounded JSON file exceeded")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_bad_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid frozen JSON") from exc


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open("rb") as stream:
        for _ in range(8193):
            line = stream.readline(65537)
            if not line:
                return result
            require(len(line) <= 65536, "source array line exceeds64KiB")
            value = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_bad_constant)
            require(type(value) is dict, "array row object")
            result.append(value)
    raise ValueError("too many source lines")


def runtime_files() -> dict[str, str]:
    names = {
        "torch": ["torch/nn/functional.py", "torch/optim/adamw.py", "torch/optim/adam.py",
                  "torch/nn/modules/module.py", "torch/version.py"],
        "transformers": ["transformers/models/qwen3/modeling_qwen3.py", "transformers/models/qwen3/configuration_qwen3.py",
                         "transformers/modeling_rope_utils.py", "transformers/masking_utils.py",
                         "transformers/integrations/hub_kernels.py", "transformers/dependency_versions_table.py",
                         "transformers/modeling_utils.py", "transformers/configuration_utils.py",
                         "transformers/modeling_layers.py", "transformers/initialization.py",
                         "transformers/utils/generic.py", "transformers/utils/output_capturing.py"],
        "safetensors": ["safetensors/__init__.py", "safetensors/torch.py"],
        "psutil": ["psutil/__init__.py", "psutil/_pswindows.py"],
    }
    paths = {}
    for name, relative_paths in names.items():
        distribution = importlib.metadata.distribution(name)
        require(distribution.version == RUNTIME[name], "runtime version changed: " + name)
        for relative in relative_paths:
            path = Path(distribution.locate_file(relative)).resolve()
            require(path.is_file(), "runtime source unavailable: " + str(path))
            paths[str(path)] = filehash(path)
        package = Path(distribution.locate_file(name))
        binaries = list(package.glob("*.pyd"))
        if name == "torch":
            binaries += list((package / "lib").glob("*.dll"))
        for path in sorted(binaries):
            paths[str(path.resolve())] = filehash(path)
    require(sys.version.split()[0] == RUNTIME["python"], "interpreter version differs")
    paths[str(Path(sys.executable).resolve())] = filehash(Path(sys.executable))
    from native_nvml import library_path
    nvml = library_path()
    paths[str(nvml)] = filehash(nvml)
    return paths


def package_sources() -> dict[str, str]:
    paths = list(PACKAGE.glob("*.py")) + [PACKAGE / "PILOT_DECISIONS.json", PACKAGE / "pilot_trainable_inventory.json",
             PACKAGE / "NATIVE_QUALIFICATION_PROPOSAL.json", PACKAGE / "AGENTS.md", ROOT / "AGENTS.md",
             ROOT / "RINEYA_SYNTH_CENTRALIZED_REASONING.md"]
    return {str(p.resolve()): filehash(p) for p in sorted(paths)}


def build_scope() -> dict:
    proposal = read_json(PACKAGE / "NATIVE_QUALIFICATION_PROPOSAL.json")
    decisions = read_json(PACKAGE / "PILOT_DECISIONS.json")
    static = read_json(PACKAGE.parent / "post_training_mimo_centralized_v1/receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json")
    model_dir = Path(static["checkpoint_path"]).resolve()
    assets = {name: static["files"][name]["sha256"] for name in
              ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")}
    require(assets["model.safetensors"] == decisions["student"]["weight_sha256"], "student asset differs")
    bundle = (ROOT / proposal["source_bundle"]).resolve()
    require(bundle.parent == PACKAGE / "data", "source bundle outside experiment")
    names = ("preparation_receipt.json", "plan.json", "enrollment.json", "candidate_records.jsonl",
             "pilot_decisions.json", "schedule_result.json", "token_pool_M1.jsonl", "token_pool_M2.jsonl")
    data_pins = {str(bundle / n): filehash(bundle / n) for n in names}
    schedule = read_json(bundle / "schedule_result.json")
    rows = {r["row_id"]: r for r in read_jsonl(bundle / "token_pool_M1.jsonl")}
    row_ids = schedule["schedules"]["M1"][:40]
    require(row_ids == proposal["mechanical_dose"]["row_ids"], "registered first40 occurrence drift")
    require(len(row_ids) == 40 and all(r in rows for r in row_ids), "40 TRAIN rows required")
    dose = {"row_ids": row_ids, "input_positions": sum(rows[r]["total_tokens"] for r in row_ids),
            "supervised_positions": sum(rows[r]["supervised_tokens"] for r in row_ids),
            "maximum_row_length": max(rows[r]["total_tokens"] for r in row_ids),
            "native_optimizer_steps": 2, "reference_optimizer_steps": 1,
            "capture_prompt_row_id": row_ids[0], "capture_max_generated_tokens": 32,
            "maximum_forward_calls": 200, "maximum_forward_input_positions": 102400}
    return {"schema_version": 1, "id": "SYNTH-NQ-001", "profile": "BF16_RESIDENT_FP32_ACTIVATIONS_QV8_V1",
            "authorization": dict(AUTHORIZATION), "resources": dict(RESOURCE_BOUNDS), "tolerances": dict(TOLERANCES),
            "offline_environment": dict(OFFLINE_ENV), "runtime_versions": dict(RUNTIME),
            "source_pins": package_sources(), "runtime_pins": runtime_files(), "data_pins": data_pins,
            "model_dir": str(model_dir), "model_pins": assets, "data_bundle": str(bundle), "dose": dose,
            "precision": proposal["precision"], "seed": 101, "production_training_authorized": False,
            "protected_bank_access": False, "quality_improvement_established": False}


def write_scope(path: Path) -> dict:
    target = path.resolve()
    require(target.parent == PACKAGE / "configs" and target.suffix == ".json" and not target.exists(), "new configs/*.json required")
    target.parent.mkdir(exist_ok=True)
    body = build_scope()
    body["sha256"] = digest(body)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(body, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return body


def preflight(config_path: str | Path, *, require_clean: bool = True) -> dict:
    path = Path(config_path).resolve()
    require(path.parent == PACKAGE / "configs" and path.is_file(), "owned native config required")
    body = read_json(path)
    require(type(body) is dict and body.get("sha256") == digest({k: v for k, v in body.items() if k != "sha256"}), "config checksum")
    for key, expected in (("authorization", AUTHORIZATION), ("resources", RESOURCE_BOUNDS), ("tolerances", TOLERANCES),
                          ("offline_environment", OFFLINE_ENV), ("runtime_versions", RUNTIME)):
        require(canonical(body.get(key)) == canonical(expected), "scope/guard change: " + key)
    require(body.get("profile") == "BF16_RESIDENT_FP32_ACTIVATIONS_QV8_V1" and
            body.get("production_training_authorized") is False and body.get("protected_bank_access") is False,
            "closed production/protected scope")
    for section in ("source_pins", "runtime_pins", "data_pins"):
        require(type(body.get(section)) is dict and bool(body[section]), "missing binding section")
        for filename, expected in body[section].items():
            require(type(filename) is str and type(expected) is str and filehash(Path(filename)) == expected,
                    "bound file changed: " + str(filename))
    require(body["source_pins"] == package_sources(), "registered source closure differs")
    require(body["runtime_pins"] == runtime_files(), "current runtime source closure differs")
    require(canonical({k: v for k, v in body.items() if k != "sha256"}) == canonical(build_scope()),
            "native scope differs from current registered proposal/student/data")
    for filename, expected in body["model_pins"].items():
        require(filename in {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"}
                and filehash(Path(body["model_dir"]) / filename) == expected, "model asset changed")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    if require_clean:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
        require(not status.strip(), "native execution requires clean committed checkout")
    return {**body, "bindings": {"source": body["source_pins"], "runtime": body["runtime_pins"],
             "data": body["data_pins"], "model": body["model_pins"]}, "worker_path": str(PACKAGE / "native_worker.py"),
            "config_path": str(path), "config_sha256": filehash(path), "source_commit": head}
