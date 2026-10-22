"""Prospective exact fresh source/data admission; never imports a model runtime."""
import json
from pathlib import Path
import subprocess
import sys
from .fresh_scope import V3, CAMPAIGN, ROOT, FIXED_CONTRACT, SOURCE, INTERPRETER, sha, validate_scope_files
from ..binding import admitted_source


def prepare():
    target = V3 / "fresh_admission_v2.json"
    if target.exists():
        raise FileExistsError("preserve existing exact native freeze")
    if sys.executable != INTERPRETER:
        raise PermissionError("canonical borrowed interpreter required")
    _, inherited = admitted_source(SOURCE)
    old = json.loads((V3 / "probe_admission_v3.json").read_text(encoding="utf-8"))
    required_reviews = {
        "reviews/fresh_data_execution_review_001.json": "PASS_FRESH_OPEN_DATA_ONLY",
        "reviews/fresh_parent_preparation_review_002.json": "PASS_ROOT_PREPARATION_SOURCE_ONLY_NOT_NATIVE_PERMISSION",
        "reviews/fresh_worker_reader_source_review_002.json": "PASS_FRESH_WORKER_READER_SOURCE_ONLY_NOT_NATIVE_PERMISSION",
    }
    for name, status in required_reviews.items():
        packet = json.loads((V3 / name).read_text(encoding="utf-8"))
        unresolved = packet.get("unresolved_source_findings") if "parent_preparation" in name else packet.get("errors")
        if packet.get("status") != status or unresolved != []:
            raise PermissionError("independent preparation/data prerequisite missing:" + name)
    files = {ROOT / name for name in inherited}
    files.update((ROOT / "experiments/rineya_taps_adaptive_recurrence_v1").glob("*.py"))
    for name in ("binding.py", "control.py", "gpu_native.py", "native_supervisor.py", "native_entry.py",
                 "bulk_restart_supervisor.py", "supervision.py", "run_apparatus.py", "full_generation_restart_authority.py"):
        files.add(CAMPAIGN / name)
    files.update(V3.glob("*.py"))
    files.update((V3 / "tests").glob("*.py"))
    for name in ("FRESH_SCOPE_001.json", "ACTIVE_GOAL_SNAPSHOT_002.json", "PRIMARY_POLICY_FREEZE_001.json",
                 "FRESH_POLICY_SOURCE_FREEZE_001.json", "FRESH_NATIVE_DESIGN_DRAFT_002.md", "FRESH_ANALYSIS_PLAN_DRAFT_001.md",
                 "reviews/opposed_replay_execution_review_001.json", "reviews/fresh_bank_source_review_001.json", *required_reviews):
        files.add(V3 / name)
    data = V3 / "data/fresh_open_replication_001"
    files.update(data.glob("*.json")); files.update((data / "model_inputs").glob("*.json"))
    model = CAMPAIGN / "runs/binding-001/MODEL_DATA_BINDING.json"
    smoke = CAMPAIGN / "runs/binding-001/PROMPT_AUDIT.json"
    files.update((model, smoke, ROOT / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl"))
    for name in ("fresh_analysis.py", "tests/test_fresh_analysis.py", "fresh_native.py", "tests/test_fresh_native.py",
                 "opposed_gate_generation.py", "tests/test_opposed_gate_generation.py"):
        if not (V3 / name).is_file():
            raise FileNotFoundError("affected-contract actual source missing:" + name)
    admission = {**FIXED_CONTRACT, "status": "READY_FOR_FRESH_K01_FP32_DISPATCH", "source": str(SOURCE),
        "interpreter": sys.executable, "head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "primary_policy": "EVIDENCE_CONSISTENCY_GUARD", "DEV_K1_rate": 160 / 192,
        "checkpoint_files": old["checkpoint_files"], "gpu_uuid": old["gpu_uuid"], "gpu_driver_version": old["gpu_driver_version"],
        "model_binding_path": str(model), "model_binding_sha256": sha(model),
        "smoke_prompt_audit_path": str(smoke), "smoke_prompt_audit_sha256": sha(smoke),
        "review_receipt": "reviews/fresh_dispatch_review_002.json", "models_loaded_by_preparation": 0,
        "scope_sha256": sha(V3 / "FRESH_SCOPE_001.json"), "primary_freeze_sha256": sha(V3 / "PRIMARY_POLICY_FREEZE_001.json"),
        "fresh_policy_source_freeze_sha256": sha(V3 / "FRESH_POLICY_SOURCE_FREEZE_001.json"),
        "goal_snapshot_sha256": sha(V3 / "ACTIVE_GOAL_SNAPSHOT_002.json"),
        "analysis_source_path": str(V3 / "fresh_analysis.py"), "analysis_source_sha256": sha(V3 / "fresh_analysis.py"),
        "analysis_plan_path": str(V3 / "FRESH_ANALYSIS_PLAN_DRAFT_001.md"), "analysis_plan_sha256": sha(V3 / "FRESH_ANALYSIS_PLAN_DRAFT_001.md")}
    for name in ("reviews/opposed_replay_execution_review_001.json", "reviews/fresh_bank_source_review_001.json", *required_reviews):
        admission[name + "_sha256"] = sha(V3 / name)
    specs = {
        "scientific_bank": data / "model_inputs/SCIENTIFIC_PROMPTS.json",
        "scientific_token_binding": data / "model_inputs/SCIENTIFIC_TOKEN_BINDING.json",
        "preservation_bank": data / "model_inputs/PRESERVATION_PROMPTS.json",
        "preservation_token_binding": data / "model_inputs/PRESERVATION_TOKEN_BINDING.json",
        "prompt_export_manifest": data / "model_inputs/MODEL_EXPORT_MANIFEST.json",
        "scientific_labels": data / "SCIENTIFIC_LABELS.json", "preservation_labels": data / "PRESERVATION_LABELS.json",
        "corpus_manifest": data / "CORPUS_MANIFEST.json", "cpu_audit": data / "CPU_AUDIT.json",
        "token_shape_audit": data / "TOKEN_SHAPE_AUDIT.json",
    }
    for name, path in specs.items():
        admission[name + "_path"] = str(path); admission[name + "_sha256"] = sha(path)
    admission["dispatch_hashes"] = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(files)}
    validate_scope_files(admission)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(admission, stream, indent=2); stream.write("\n")
    if "torch" in sys.modules:
        raise RuntimeError("model runtime unexpectedly imported by preparation")
    print(json.dumps({"admission": str(target), "sha256": sha(target), "actual_files": len(files), "models": 0}))


if __name__ == "__main__":
    prepare()
