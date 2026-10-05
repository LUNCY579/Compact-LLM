"""Offline source-bound parent evidence checks; no native permission is granted.

The supported receipt is the RSD-native schema-2 ordinary-SFT export. Genuine
external SFT parents need a separately reviewed provenance adapter contract;
unrecognized evidence is rejected rather than inferred to be native training.
This module checks bytes and metadata. It does not load an adapter, inspect model
behavior, establish human review, or certify the provenance assertions as true.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from .contracts import BASE_SHA256, digest, file_hash, read_json, require

PACKAGES = {"torch": "2.10.0+cu126", "transformers": "4.57.1",
            "safetensors": "0.7.0", "tokenizers": "0.22.2"}
INPUT_NAMES = {"e0_config", "e0_result", "catalog", "exclusions", "catalog_review"}
CORE_SOURCES = {
    "experiments/rineya_recursive_self_distillation_v1/" + name
    for name in ("__init__.py", "__main__.py", "native.py", "contracts.py", "data.py",
                 "backend.py", "engine.py", "evidence.py", "objectives.py", "prompts.py",
                 "metrics.py", "parent_qualification.py", "e0_qualification.py",
                 "catalog_qualification.py", "readiness.py")
} | {"experiments/post_training_mimo_centralized_v1/scripts/qualify_native.py",
     "experiments/post_training_mimo_centralized_v1/scripts/supervise_native.py",
     "experiments/maxrl_comp002_pretraining_v1/mechanics.py",
     "experiments/opallo_native_collector_v1/supervisor.py",
     "experiments/opallo_native_collector_v1/contract.py"}


def _hash(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid SHA256: " + label)


def _bound_file(spec, base: Path, label: str) -> Path:
    require(type(spec) is dict and set(spec) == {"path", "sha256"},
            "invalid parent file binding: " + label)
    require(type(spec["path"]) is str and bool(spec["path"].strip()),
            "invalid parent file path: " + label)
    _hash(spec["sha256"], label)
    path = Path(spec["path"])
    path = (path if path.is_absolute() else base / path).resolve()
    require(path.is_file(), "missing parent evidence file: " + label)
    require(file_hash(path) == spec["sha256"], "parent evidence file drift: " + label)
    return path


def _identity(identity):
    require(type(identity) is dict and identity.get("evidence") == "native_exact_titan"
            and identity.get("base") == BASE_SHA256
            and identity.get("adapter") == "ZERO_LORA_ON_EXACT_TITAN"
            and identity.get("parent_evidence") == "parent_stage",
            "parent must be source-bound native exact-Titan ordinary SFT from zero LoRA")
    for name in ("plan", "catalog", "source"):
        _hash(identity.get(name), "identity." + name)
    files = identity.get("files")
    require(type(files) is dict and set(files) == INPUT_NAMES,
            "parent identity needs exact E0/catalog/exclusions/catalog-review bindings")
    for name, value in files.items():
        _hash(value, "identity.files." + name)
    sources = identity.get("sources")
    require(type(sources) is dict and CORE_SOURCES <= set(sources),
            "incomplete parent source identity")
    for name, value in sources.items():
        require(type(name) is str and "\\" not in name and ":" not in name
                and not PurePosixPath(name).is_absolute()
                and name == PurePosixPath(name).as_posix()
                and ".." not in PurePosixPath(name).parts and name.endswith(".py"),
                "invalid parent source path")
        _hash(value, "identity.sources." + name)
    require(digest(sources) == identity["source"], "parent source digest/map drift")


def _runtime(runtime):
    require(type(runtime) is dict and runtime.get("packages") == PACKAGES,
            "unregistered parent native packages")
    require(type(runtime.get("python")) is str
            and re.match(r"^3\.11\.15(?:\s|$)", runtime["python"]) is not None,
            "unregistered parent native Python")
    require(type(runtime.get("platform")) is str and runtime["platform"].startswith("Windows-")
            and runtime.get("cuda") == "12.6"
            and type(runtime.get("gpu")) is str and bool(runtime["gpu"].strip())
            and "cpu" not in runtime["gpu"].casefold()
            and runtime.get("base_dtype") == "bfloat16"
            and runtime.get("adapter_dtype") == "float32"
            and runtime.get("attention") == "eager"
            and runtime.get("sampler") == "full-prefix-no-cache-temp1"
            and runtime.get("evaluation") == "greedy-fixed-budget",
            "unregistered parent native runtime contract")


def validate_parent(manifest_path: str | Path, require_review: bool = True):
    """Return ``(adapter_path, receipt_path, evidence_summary)`` after checks.

    Paths are resolved and both files' bytes match the manifest. Historical
    source maps are checked for completeness and their recorded digest, without
    requiring equality to the current checkout. ``require_review=False`` is for
    checking candidate evidence before an operator review; it cannot authorize
    treatment enrollment. Unsupported external receipt formats fail closed.
    """
    require(type(require_review) is bool, "review requirement must be boolean")
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    require(type(manifest) is dict and type(manifest.get("schema")) is int and manifest["schema"] == 1
            and manifest.get("base_weight_sha256") == BASE_SHA256
            and manifest.get("native_sft_completed") is True,
            "real completed exact-Titan SFT parent manifest required")
    steps = manifest.get("optimizer_steps")
    require(type(steps) is int and steps > 0, "positive completed parent optimizer steps required")
    reviewed = manifest.get("parent_evaluation_reviewed")
    reviewer = manifest.get("reviewed_by")
    require(type(reviewed) is bool and type(reviewer) is str, "invalid parent review fields")
    require(not reviewed or bool(reviewer.strip()), "reviewed parent needs reviewer identity")
    require(not require_review or (reviewed and bool(reviewer.strip())),
            "explicit parent evaluation review required")
    adapter = _bound_file(manifest.get("adapter"), path.parent, "adapter")
    require(adapter.suffix == ".safetensors", "parent adapter must be safetensors")
    receipt_path = _bound_file(manifest.get("source_run_receipt"), path.parent, "source_run_receipt")
    receipt = read_json(receipt_path)
    require(type(receipt) is dict and receipt.get("schema") == 2
            and receipt.get("experiment") == "RSD-001" and receipt.get("mode") == "parent"
            and receipt.get("arm") == "sft"
            and receipt.get("status") == "COMPLETED_SFT_CANDIDATE_REVIEW_REQUIRED",
            "unsupported parent receipt; external SFT needs reviewed provenance adapter contract")
    require(receipt.get("native_efficacy_claim") is False, "parent receipt may not certify efficacy")
    exported = _bound_file(receipt.get("exported_adapter"), receipt_path.parent, "exported_adapter")
    require(exported == adapter
            and receipt["exported_adapter"]["sha256"] == manifest["adapter"]["sha256"],
            "parent manifest/export adapter mismatch")
    _identity(receipt.get("identity"))
    commit = receipt.get("source_commit")
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
            "parent source commit required")
    _runtime(receipt.get("runtime"))
    result = receipt.get("result")
    require(type(result) is dict and result.get("status") == "COMPLETE"
            and type(result.get("steps")) is int and result["steps"] == steps
            and result.get("native_efficacy_claim") is False,
            "unfinished/mismatched parent training result")
    costs = result.get("costs")
    require(type(costs) is dict and type(costs.get("optimizer_steps")) is int
            and costs["optimizer_steps"] == steps, "parent optimizer-step evidence mismatch")
    if "optimizer_steps" in result:
        require(type(result["optimizer_steps"]) is int and result["optimizer_steps"] == steps,
                "parent top-level optimizer-step mismatch")
    require(type(result.get("last_evaluated_step")) is int
            and result["last_evaluated_step"] == steps,
            "completed parent needs terminal evaluation evidence")
    summary = {"status": "SOURCE_BOUND_PARENT_METADATA_CHECKED",
               "supported_route": "RSD-001-native-parent-schema2", "source_commit": commit,
               "source": receipt["identity"]["source"], "base": BASE_SHA256,
               "plan": receipt["identity"]["plan"], "catalog": receipt["identity"]["catalog"],
               "input_hashes": receipt["identity"]["files"], "optimizer_steps": steps,
               "adapter_sha256": manifest["adapter"]["sha256"],
               "receipt_sha256": manifest["source_run_receipt"]["sha256"],
               "review_attested": reviewed, "reviewed_by": reviewer,
               "quality_certified": False, "native_efficacy_claim": False,
               "native_execution_authorized": False,
               "limitations": "Bytes and metadata only; source/provenance/review truth and model behavior "
                              "require independent evidence. No model or adapter was loaded."}
    return adapter, receipt_path, summary
