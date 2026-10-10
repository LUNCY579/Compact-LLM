"""No model imports: immutable identity, existing E0 gate, and exact local assets."""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from .contracts import Plan, digest, file_hash, matrix, read_json, require, sealed, unseal, write_new
from .data import load_catalog, require_roles


def source_identity():
    root = Path(__file__).parent
    return {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}


def inventory(directory):
    root = Path(directory)
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "absolute local model directory required")
    require(not any(p.is_symlink() for p in root.iterdir()), "model symlinks are not admitted")
    files = sorted(p for p in root.iterdir() if p.is_file())
    require(0 < len(files) <= 256, "model file count")
    require(not any(p.suffix in {".bin", ".pt", ".pth", ".pkl", ".py"} for p in files),
            "no pickle, executable custom loader, or mixed checkpoint formats")
    require((root / "config.json").is_file() and (root / "tokenizer.json").is_file()
            and (root / "tokenizer_config.json").is_file(), "standard HF config/tokenizer files required")
    weights = [p for p in files if p.suffix == ".safetensors"]
    require(bool(weights), "safetensors checkpoint required")
    cfg = read_json(root / "config.json")
    require(cfg.get("model_type") == "qwen3" and cfg.get("architectures") == ["Qwen3ForCausalLM"],
            "only explicit standard Qwen3ForCausalLM exports are supported")
    require(not cfg.get("quantization_config") and not cfg.get("auto_map"), "quantized/custom checkpoints unsupported")
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = read_json(index)["weight_map"]
        require(type(mapping) is dict and mapping, "checkpoint index")
        require(set(mapping.values()) <= {p.name for p in weights}, "missing/out-of-directory checkpoint shard")
    else:
        require(len(weights) == 1 and weights[0].name == "model.safetensors", "unindexed checkpoint layout")
    manifest = {p.name: file_hash(p) for p in files if p.suffix in {".safetensors", ".json", ".txt", ".model", ".jinja"}}
    return {"path": str(root.resolve()), "files": manifest, "sha256": digest(manifest),
            "serialized_weight_bytes": sum(p.stat().st_size for p in weights)}


def checked_file(spec):
    require(type(spec) is dict and set(spec) == {"path", "sha256"}, "bound file fields")
    p = Path(spec["path"])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(), "absolute existing bound file required")
    require(file_hash(p) == spec["sha256"], "bound file drift: " + str(p))
    return p


def check_e0(config_path, result_path):
    result = read_json(result_path)
    require(result.get("status") == "PASS_E0_SCOPED_NO_EFFICACY_CLAIM"
            and result.get("config_sha256") == file_hash(config_path), "matching existing E0 PASS required")
    for key in ("exact_native_load", "save_reload_resume_bitwise", "frozen_base_unchanged",
                "finite_scoped_backward", "eos_pad_mask"):
        require(result.get("checks", {}).get(key) is True, "missing E0 check: " + key)


def check_completed_native_run(result_path, identity):
    """A COMPLETE arm cannot supersede later cleanup/supervisor failure evidence."""
    path = Path(result_path).resolve()
    outer = path.parent.parent
    require(not any((path.parent / name).exists() for name in ("FAILED.json", "RESTORE_FAILED.json"))
            and not (outer / "SUPERVISOR_FAILED.json").exists(), "failed native invocation cannot be consumed")
    supervisor = read_json(outer / "SUPERVISOR_RESULT.json")
    native = unseal(read_json(outer / "NATIVE_RESULT.json"))
    qualification = outer / "qualification" / "RESULT.json"
    require(supervisor.get("status") == "WORKER_EXITED_ZERO_NOT_EFFICACY"
            and native.get("status") == "COMPLETED_NOT_PROMOTED" and native.get("identity") == identity
            and native.get("result_sha256") == file_hash(path)
            and native.get("qualification_sha256") == file_hash(qualification),
            "completed supervised native invocation and qualification binding required")
    q = unseal(read_json(qualification))
    require(q.get("status") == "PASS_SCOPED_MECHANICS_AND_TEACHER_SCREEN"
            and q.get("source_restored_bitwise") is True and q.get("evidence_kind") == "qualified_native_hf",
            "successful native qualification and source restoration required")


def register(plan_path, binding_path, output):
    plan = Plan.load(plan_path)
    require(plan.teacher_min_accuracy > 0 and plan.teacher_min_advantage > 0
            and plan.min_acquisition_gain > 0, "positive native teacher/acquisition thresholds required")
    b = read_json(binding_path)
    fields = {"schema", "student", "teacher", "student_parameters", "teacher_parameters", "e0_config",
              "e0_result", "export_review", "catalog", "exclusions", "confirmation_sha256",
              "learning_rates_reviewed", "reviewed_by"}
    require(type(b) is dict and set(b) == fields and b["schema"] == 1, "binding schema")
    require(b["learning_rates_reviewed"] is True and type(b["reviewed_by"]) is str and b["reviewed_by"].strip(),
            "pilot learning rates and source assets require explicit review")
    require(type(b["student_parameters"]) is int and 500_000_000 <= b["student_parameters"] <= 700_000_000,
            "exact 0.6B-class Rineya parameter count required; not 344M")
    require(type(b["teacher_parameters"]) is int and b["teacher_parameters"] > 0, "exact teacher parameter count")
    require(type(b["confirmation_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", b["confirmation_sha256"]),
            "commit separate unopened confirmation catalog digest before training")
    assets = {}
    for role in ("student", "teacher"):
        require(type(b[role]) is dict and set(b[role]) == {"path", "sha256"}, "model binding fields")
        assets[role] = inventory(b[role]["path"])
        require(assets[role]["sha256"] == b[role]["sha256"], "model snapshot drift")
    files = {k: checked_file(b[k]) for k in ("e0_config", "e0_result", "export_review", "catalog", "exclusions")}
    check_e0(files["e0_config"], files["e0_result"])
    review = read_json(files["export_review"])
    require(review.get("status") == "REVIEWED_NATIVE_EXPORT"
            and review.get("student_snapshot") == assets["student"]["sha256"]
            and review.get("teacher_snapshot") == assets["teacher"]["sha256"]
            and review.get("e0_result_sha256") == file_hash(files["e0_result"])
            and review.get("native_greedy_parity") is True
            and review.get("teacher_prefix_reviewed") is True
            and bool(review.get("reviewed_by")), "reviewed export parity and student-prefix teacher review required")
    tasks, provenance, audit = load_catalog(files["catalog"], read_json(files["exclusions"]))
    require(provenance == "reviewed_external", "mechanical fixture cannot enroll natively")
    require_roles(tasks, plan.replay_tasks > 0)
    require(len({t.parent for t in tasks if t.role == "teacher_dev"}) >= 16, "at least 16 teacher-dev parents")
    if plan.design == "selected":
        require(len([t for t in tasks if t.role == "teacher_dev"]) == 16,
                "selected qualification freezes exactly 16 teacher-dev rows")
    payload = {"schema": 1, "plan": asdict(plan), "plan_sha256": plan.fingerprint, "binding": b,
               "assets": assets, "files": {k: str(v) for k, v in files.items()}, "audit": audit,
               "source": source_identity(), "native_qualified": False,
               "review_is_operator_attestation_not_cryptographic_proof": True}
    if plan.design == "selected":
        payload["stage_scope"] = "selected"
        payload["treatment_cells"] = [asdict(arm) for arm in matrix(plan)]
    write_new(output, sealed(payload))
    return payload


def check_registration(path):
    r = unseal(read_json(path))
    require(r.get("schema") == 1 and r["source"] == source_identity(), "registered source drift")
    data = dict(r["plan"])
    data["seeds"], data["learning_rates"] = tuple(data["seeds"]), tuple(data["learning_rates"])
    if "selected_cells" in data:
        data["selected_cells"] = tuple(data["selected_cells"])
    plan = Plan(**data)
    require(plan.fingerprint == r["plan_sha256"], "plan identity drift")
    if plan.design == "selected":
        require(r.get("stage_scope") == "selected"
                and r.get("treatment_cells") == [asdict(arm) for arm in matrix(plan)],
                "complete selected registration scope required; no full/screen widening")
        require(r["binding"].get("learning_rates_reviewed") is True
                and bool(r["binding"].get("reviewed_by", "").strip()),
                "selected frozen learning rate requires explicit review")
    else:
        require("stage_scope" not in r and "treatment_cells" not in r,
                "factorial registration cannot import selected scope")
    for role in ("student", "teacher"):
        require(inventory(r["assets"][role]["path"])["sha256"] == r["assets"][role]["sha256"], "asset drift")
    for key in r["files"]:
        require(str(checked_file(r["binding"][key])) == r["files"][key], "file path drift")
    check_e0(r["files"]["e0_config"], r["files"]["e0_result"])
    tasks, provenance, audit = load_catalog(r["files"]["catalog"], read_json(r["files"]["exclusions"]))
    require(audit == r["audit"] and provenance == "reviewed_external", "catalog drift")
    return r, plan, tasks
