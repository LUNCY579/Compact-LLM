"""Offline source-bound parent evidence checks; no native permission is granted.

The supported receipt is the RSD-native schema-2 ordinary-SFT export. Genuine
external SFT parents need a separately reviewed provenance adapter contract;
unrecognized evidence is rejected rather than inferred to be native training.
This module checks bytes and metadata. It does not load an adapter, inspect model
behavior, establish human review, or certify the provenance assertions as true.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

from .contracts import BASE_SHA256, digest, file_hash, read_json, require

PACKAGES = {"torch": "2.10.0+cu126", "transformers": "4.57.1",
            "safetensors": "0.7.0", "tokenizers": "0.22.2"}
INPUT_NAMES = {"e0_config", "e0_result", "catalog", "exclusions", "catalog_review"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MAX_SOURCE_FILES = 128
MAX_SOURCE_PATH_BYTES = 1024
MAX_SOURCE_BLOB_BYTES = 2 * 2**20
MAX_SOURCE_TOTAL_BYTES = 16 * 2**20
GIT_TIMEOUT_SECONDS = 10
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
    require(len(sources) <= MAX_SOURCE_FILES, "parent source file-count limit")
    for name, value in sources.items():
        require(type(name) is str and "\\" not in name and ":" not in name
                and not any(ord(character) < 32 or ord(character) == 127 for character in name)
                and len(name.encode("utf-8")) <= MAX_SOURCE_PATH_BYTES
                and not PurePosixPath(name).is_absolute()
                and name == PurePosixPath(name).as_posix()
                and ".." not in PurePosixPath(name).parts and name.endswith(".py"),
                "invalid parent source path")
        _hash(value, "identity.sources." + name)
    require(digest(sources) == identity["source"], "parent source digest/map drift")


def _git(arguments, label, *, input_data=None, output_limit=4096):
    """Read local Git objects only, with a finite process timeout.

    Literal paths, disabled replacements and a sanitized Git environment keep
    the fixed repository authoritative. Missing partial-clone objects must not
    trigger a lazy fetch. Blob sizes are checked before requesting their bytes.
    """
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith("GIT_")}
    environment.update(GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        process = subprocess.run(["git", "--no-replace-objects", "--literal-pathspecs", *arguments],
                                 cwd=REPOSITORY_ROOT, input=input_data, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, env=environment, timeout=GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("parent historical source Git unavailable/timed out: " + label) from exc
    require(process.returncode == 0, "parent historical source Git check failed: " + label)
    require(len(process.stdout) <= output_limit, "parent historical source Git output limit: " + label)
    return process.stdout


def _historical_source(sources, commit):
    """Bind every validated source path to a regular blob at the exact commit."""
    top = _git(["rev-parse", "--show-toplevel"], "repository").decode("utf-8").strip()
    require(Path(top).resolve() == REPOSITORY_ROOT.resolve(), "parent historical source repository mismatch")
    resolved = _git(["rev-parse", "--verify", commit + "^{commit}"], "commit", output_limit=64)
    require(resolved.decode("ascii").strip() == commit, "parent source_commit must identify a commit")
    names = sorted(sources)
    tree = _git(["ls-tree", "-z", "--full-tree", commit, "--", *names], "source paths",
                output_limit=MAX_SOURCE_FILES * (MAX_SOURCE_PATH_BYTES + 100))
    entries = tree.split(b"\x00")
    require(entries[-1] == b"", "invalid parent historical source tree output")
    objects = {}
    for entry in entries[:-1]:
        header, separator, raw_name = entry.partition(b"\t")
        fields = header.split(b" ")
        require(separator and len(fields) == 3, "invalid parent historical source tree entry")
        mode, kind, object_id = fields
        name = raw_name.decode("utf-8")
        require(name in sources and name not in objects, "parent historical source path mismatch")
        require(mode in (b"100644", b"100755") and kind == b"blob"
                and re.fullmatch(rb"[0-9a-f]{40}", object_id) is not None,
                "parent historical source must be a regular Git blob: " + name)
        objects[name] = object_id
    require(set(objects) == set(sources), "missing parent historical source path")
    batch = b"\n".join(objects[name] for name in names) + b"\n"
    metadata = _git(["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                    "blob sizes", input_data=batch, output_limit=MAX_SOURCE_FILES * 100).splitlines()
    require(len(metadata) == len(names), "parent historical source blob inventory mismatch")
    sizes = []
    for name, entry in zip(names, metadata):
        fields = entry.split(b" ")
        require(len(fields) == 3 and fields[0] == objects[name] and fields[1] == b"blob"
                and fields[2].isdigit() and len(fields[2]) <= 10,
                "invalid parent historical source blob metadata: " + name)
        size = int(fields[2])
        require(size <= MAX_SOURCE_BLOB_BYTES, "parent historical source blob byte limit: " + name)
        sizes.append(size)
    total = sum(sizes)
    require(total <= MAX_SOURCE_TOTAL_BYTES, "parent historical source total byte limit")
    payload = _git(["cat-file", "--batch"], "blob bytes", input_data=batch,
                   output_limit=total + MAX_SOURCE_FILES * 100)
    cursor = 0
    for name, size in zip(names, sizes):
        end_header = payload.find(b"\n", cursor)
        expected = objects[name] + b" blob " + str(size).encode("ascii")
        require(end_header >= cursor and payload[cursor:end_header] == expected,
                "parent historical source blob header mismatch: " + name)
        start, end = end_header + 1, end_header + 1 + size
        require(end < len(payload) and payload[end:end + 1] == b"\n",
                "parent historical source blob length mismatch: " + name)
        require(hashlib.sha256(payload[start:end]).hexdigest() == sources[name],
                "parent historical source SHA256 mismatch: " + name)
        cursor = end + 1
    require(cursor == len(payload), "unexpected parent historical source blob bytes")
    return {"verification": "exact_commit_regular_source_blobs_sha256", "files": len(names), "bytes": total}


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
    source maps must match regular Git blobs at their recorded commit in this
    module's fixed repository; unavailable history fails closed. The recorded
    commit need not match current HEAD. ``require_review=False`` is for
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
    historical_source = _historical_source(receipt["identity"]["sources"], commit)
    summary = {"status": "SOURCE_BOUND_PARENT_METADATA_CHECKED",
               "supported_route": "RSD-001-native-parent-schema2", "source_commit": commit,
               "historical_source": historical_source,
               "source": receipt["identity"]["source"], "base": BASE_SHA256,
               "plan": receipt["identity"]["plan"], "catalog": receipt["identity"]["catalog"],
               "input_hashes": receipt["identity"]["files"], "optimizer_steps": steps,
               "adapter_sha256": manifest["adapter"]["sha256"],
               "receipt_sha256": manifest["source_run_receipt"]["sha256"],
               "review_attested": reviewed, "reviewed_by": reviewer,
               "quality_certified": False, "native_efficacy_claim": False,
               "native_execution_authorized": False,
               "limitations": "Historical source bytes are verified; training/provenance/review truth and model behavior "
                              "require independent evidence. No model or adapter was loaded."}
    return adapter, receipt_path, summary
