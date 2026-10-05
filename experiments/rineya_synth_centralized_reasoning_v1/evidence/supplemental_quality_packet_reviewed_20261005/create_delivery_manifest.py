"""Freeze this external offline handoff; no native execution or source edits."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_ref(path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


verification = json.loads((ROOT / "VERIFICATION_001.json").read_text(encoding="utf-8"))
assert verification["status"] == "PASS_MECHANICAL_ONLY"
assert verification["tests_run"] == 70 and not verification["framework_import_attempts"]
assert not verification["frameworks_loaded"]
for relative, expected in verification["implementation_hashes"].items():
    assert file_ref(ROOT / relative)["sha256"] == expected, "tested source drift: " + relative
paths = sorted([*ROOT.glob("*.py"), *(ROOT / "tests").glob("*.py"), ROOT / "README.md",
    ROOT / "QUALITY_SCREEN_POLICY_001.json", ROOT / "READINESS_PACKET_001.json",
    ROOT / "VERIFICATION_001.json", ROOT / "VERIFICATION_001.log"])
files = {path.relative_to(ROOT).as_posix(): file_ref(path) for path in paths}
packet = json.loads((ROOT / "READINESS_PACKET_001.json").read_text(encoding="utf-8"))
assert packet["status"] == "BLOCKED_PROSPECTIVE_READINESS_AND_MEASUREMENT_PACKET"
assert not any(packet["permissions"].values()) and not packet["execution_permit"]
source = Path(packet["source"]["root"])
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
assert head == packet["source"]["head"]
payload = {"schema": "synth-quality-offline-delivery-v1", "status": "IMPLEMENTED_AND_INDEPENDENTLY_REVIEWED_OFFLINE",
    "artifact_root": str(ROOT), "files": files,
    "owner_chat": {"title": "Build CPU-only R/S task compiler", "thread_id": "01a1002f-e445-7680-8aa3-1d7bb05c9774",
                   "tracked_and_native_ownership_preserved": True, "code_acceptance_by_owner": "PENDING_SEPARATE_OWNER_RECEIPT"},
    "source_at_packet": {"head": head, "branch": packet["source"]["branch"], "tracked_files_edited_by_this_chat": False},
    "verification": {"tests": 70, "failures": 0, "errors": 0, "skipped": 0,
                     "frameworks_loaded": [], "native_model_loads": 0, "native_forward_calls": 0,
                     "native_tokenizer_execution": 0, "optimizer_updates": 0,
                     "protected_or_confirmation_reads": 0},
    "independent_review": {"reviewer": "/root/selected_plan", "tests": 70,
        "status": "PASS_NO_REMAINING_ACTIONABLE_FINDING", "real_condition_rendering_identity_bug_fixed": True,
        "evidence_scope": "source/stdlib metadata and fixtures only; no Rineya capability evidence"},
    "native_state": {"ready": False, "unresolved_slots": 15, "M0_raw_audited_rows": 256,
                     "M0_required_rows": 384, "M0_scored_rows": 0, "STUDY_trajectories_completed": 0,
                     "native_export_parity": False, "quality_gain_established": False},
    "historical_policy": "5pp plus CI unchanged; supplemental3pp/zero-new-loss policy prospectively separate",
    "fixture_derivatives": "EXCLUDED_FROM_NATIVE_TRAIN_SELECTION_DEV_CHECK_FINAL_CONFIRMATION",
    "next_owner_action": "Review pinned implementation/tests; admit prospective integration only after missing native/scorer/cohort/qualification/permit gates. No automatic training or confirmation."}
manifest = {**payload, "sha256": digest(payload)}
path = ROOT / "IMPLEMENTATION_MANIFEST_001.json"
with path.open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
print(json.dumps({"path": str(path), "sha256": file_ref(path)["sha256"], "files": len(files)}, indent=2))
