"""Content-bound stage planning. Planning does not authorize a model call.

Native stages deliberately have no launcher in this release: the remote source
contains no qualified SCP/Titan or verified 4B host binding. Existing supervisors
must qualify those exact subjects before a later additive integration enables them.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
from .integrity import code_digest, digest, file_hash, positive_number, read_json, require, valid_hash, write_new

STAGES = {
    "mechanics": (),
    "instrument": ("mechanics",),
    "student_qualification": ("mechanics",),
    "teacher_qualification": ("mechanics",),
    "compression": ("mechanics", "student_qualification"),
    "teacher_replication": ("instrument", "teacher_qualification"),
    "prepare_fp": ("instrument",),
    "prepare_t": ("prepare_fp", "teacher_qualification"),
    "baseline_reasoning": ("instrument", "student_qualification"),
    "student_training": ("prepare_t", "student_qualification", "teacher_replication"),
    "posttraining_reasoning": ("student_training", "baseline_reasoning"),
    "data_efficiency": ("student_training",),
    "skill_composition": ("compression", "student_training"),
}
NATIVE_STAGES = set(STAGES) - {"mechanics", "instrument", "prepare_fp", "prepare_t"}
BUDGET_FIELDS = ("wall_seconds", "model_calls", "generated_tokens", "processed_training_tokens",
                 "new_disk_bytes", "max_host_rss_bytes", "min_free_host_bytes", "min_free_disk_bytes")


def validate_budget(budget):
    require(type(budget) is dict and set(budget) == set(BUDGET_FIELDS), "explicit complete numeric session budget required")
    for key in BUDGET_FIELDS:
        require(positive_number(budget[key]), f"positive finite budget required: {key}")
        if key != "wall_seconds":
            require(type(budget[key]) is int, f"integer budget required: {key}")
    return budget


def validate_assets(binding):
    """Hash only explicit allowlisted assets; never discover/download or read a bank."""
    require(type(binding) is dict and binding.get("assets"), "asset manifest is empty")
    result = {}
    for name, spec in binding["assets"].items():
        require(set(spec) == {"path", "sha256"} and valid_hash(spec["sha256"]), "invalid asset identity")
        path = Path(spec["path"])
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular asset: {name}")
        require(file_hash(path) == spec["sha256"], f"asset changed: {name}")
        result[name] = spec["sha256"]
    return result


def outside_checkout(root: Path, destination: Path):
    root, destination = root.resolve(), destination.resolve()
    require(destination != root and root not in destination.parents, "mutable evidence must be outside checkout")
    return destination


def clean_checkout(root: Path, expected_head: str):
    require(len(expected_head) == 40 and all(c in "0123456789abcdef" for c in expected_head), "invalid source commit")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=15).strip()
    require(git("rev-parse", "HEAD") == expected_head, "checkout commit changed")
    require(not git("status", "--porcelain", "--untracked-files=normal"), "native execution requires clean checkout")


def receipt(stage, status, binding, artifacts, *, evidence_class="SOFTWARE", details=None):
    require(stage in STAGES and status in {"PASS", "FAIL", "BLOCKED"}, "invalid stage receipt")
    require(evidence_class in {"SOFTWARE", "DATA", "NATIVE_OPEN_DEV"}, "invalid evidence class")
    require(type(binding) is dict and binding, "empty binding")
    require(type(artifacts) is dict and artifacts, "receipt needs tangible artifacts")
    refs = {}
    for name, path in artifacts.items():
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), "missing/nonregular evidence artifact")
        refs[name] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    return {"schema": "SCP_STAGE_RECEIPT_V1", "experiment": "SCP-001", "stage": stage,
            "status": status, "evidence_class": evidence_class, "binding_sha256": digest(binding),
            "code_sha256": code_digest(), "artifacts": refs, "details": details or {},
            "confirmation": False, "release": False, "quality_claim": False}


def validate_receipt(value, stage, binding):
    require(value.get("schema") == "SCP_STAGE_RECEIPT_V1" and value.get("experiment") == "SCP-001", "foreign receipt")
    require(value.get("stage") == stage and value.get("status") == "PASS", "stage not passed")
    require(value.get("code_sha256") == code_digest() and value.get("binding_sha256") == digest(binding), "stale receipt")
    require(all(value.get(k) is False for k in ("confirmation", "release", "quality_claim")), "receipt widens claim")
    if stage in NATIVE_STAGES:
        require(value.get("evidence_class") == "NATIVE_OPEN_DEV", "fixture cannot satisfy native stage")
    elif stage != "mechanics":
        require(value.get("evidence_class") == "DATA", "software fixture cannot accept a dataset")
    require(value.get("artifacts"), "receipt without artifacts")
    for spec in value["artifacts"].values():
        require(valid_hash(spec.get("sha256")), "invalid artifact hash")
        path = Path(spec["path"])
        require(path.is_file() and not path.is_symlink() and file_hash(path) == spec["sha256"], "receipt artifact changed")
    return True


def status(binding=None, receipts=None):
    binding, receipts = binding or {}, receipts or {}
    results = {}
    for stage, dependencies in STAGES.items():
        blockers = [f"dependency:{name}" for name in dependencies if results[name]["status"] != "PASS"]
        if stage in receipts:
            try:
                validate_receipt(receipts[stage], stage, binding)
                if not blockers:
                    results[stage] = {"status": "PASS", "blockers": []}
                    continue
            except (ValueError, KeyError, OSError) as exc:
                blockers.append(f"invalid_receipt:{exc}")
        if stage in NATIVE_STAGES:
            blockers.append("SCP native supervisor/subject integration not qualified; no launcher shipped")
        elif stage == "instrument":
            blockers.append("fresh tokenizer/solver/shortcut/lineage/rights/human-review acceptance required")
        elif stage == "prepare_t":
            blockers.append("verified teacher outputs on ALL registered training worlds required")
        elif stage == "prepare_fp":
            blockers.append("accepted TRAIN worlds and shared preservation anchors required")
        else:
            blockers.append("run scoped CPU tests and retain their receipt")
        results[stage] = {"status": "BLOCKED", "blockers": blockers}
    return {"experiment": "SCP-001", "code_sha256": code_digest(), "stages": results,
            "confirmation": "CLOSED_SEPARATE_AUTHORITY", "release": False,
            "native_launch_available": False}


def freeze_binding(binding, out: Path):
    require(binding.get("experiment") == "SCP-001", "wrong experiment")
    require(binding.get("population_role") == "OPEN_DEV", "no confirmation in this pipeline")
    for field in ("protocol_sha256", "dataset_sha256", "tokenizer_sha256", "student_sha256"):
        require(valid_hash(binding.get(field)), f"missing binding: {field}")
    require(binding["protocol_sha256"] == file_hash(Path(__file__).with_name("protocol.json")), "protocol changed")
    require(binding.get("student_dtype") == "float32", "incumbent requires FP32; precision changes are separate arms")
    validate_budget(binding.get("budget"))
    require(binding.get("preservation_rule") == "ZERO_OBSERVED_DAMAGE_OPEN_SCOUT", "new preservation rule must be explicit")
    return write_new(out, {**binding, "code_sha256": code_digest(), "native_launch_available": False})
