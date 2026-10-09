from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent.parent
W = P / "artifacts/arithmetic_capture_preparation_work"
D = P / "evidence/arithmetic_capture_reviewed_cpu_20261009"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def selected(path, pin):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin
    value = json.loads(raw)
    assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
    return value


handoff = selected(W / "frozen_arithmetic_capture001/HANDOFF001.json",
    "df45c52148fbdb0da14f4e1f0ad458ef455296a0934718546a2f6697ad4ed3bb")
review = selected(W / "independent_capture_replay001/INDEPENDENT_CAPTURE_REVIEW001.json",
    "aa794c0808fe257250ce9c2b53b25ba9cf47525f5cfe49e98f78ea4f42cfbb8b")
actual_dir = P / "artifacts/arithmetic_capture_actual_review_work"
actual = selected(actual_dir / "INDEPENDENT_ACTUAL_PLAN_REVIEW001.json",
    "65accda3d94cfca5ff5e78a1d01de03653182320b62aa55cee13710903a6269a")
root = json.loads((W / "root_replay001/CPU_CONTROLS003.json").read_bytes())
assert root["sha256"] == digest({k: v for k, v in root.items() if k != "sha256"})
assert root["success"] is True and root["tests"] == 6 and root["source_unchanged"] is True
assert root["forbidden_import_attempts"] == []
assert actual["accepted"] is True and actual["input_bytes_unchanged"] is True
assert actual["M0_capture_or_scoring_performed"] is False
for name, ref in handoff["candidate_files"].items():
    assert hashlib.sha256((W / "frozen_arithmetic_capture001" / name).read_bytes()).hexdigest() == ref["sha256"]
assert not D.exists()
D.mkdir()
for source, dest in ((W / "frozen_arithmetic_capture001", "current"),
                     (W / "root_replay001", "root_replay"),
                     (W / "independent_capture_replay001", "independent_source_review"),
                     (P / "artifacts/arithmetic_capture_actual_plan_work", "actual_plan"),
                     (actual_dir, "independent_actual_plan_review")):
    shutil.copytree(source, D / dest)
for source in W.iterdir():
    if source.is_dir() and source.name.startswith("pre_"):
        shutil.copytree(source, D / "history" / source.name)
shutil.copy2(P / "artifacts/build_arithmetic_capture_plan001.py", D / "build_arithmetic_capture_plan001.py")
shutil.copy2(Path(__file__).resolve(), D / "preserve_arithmetic_capture001.py")
value = {"kind": "ROOT_PRESERVED_CLOSED_ARITHMETIC_CAPTURE_CPU001",
    "source_commit": "201229ed58c5c66edcd0390c79c51c44738c5e02",
    "rows": 512, "parents": 128, "batches": 8, "authored_controls": 6,
    "root_controls": 6, "independent_controls": 6,
    "root_control_body_sha256": root["sha256"],
    "independent_source_review_body_sha256": review["sha256"],
    "independent_actual_plan_review_body_sha256": actual["sha256"],
    "actual_prompt_population_and_plan_join_verified": True,
    "actual_M0_capture_or_scoring_performed": False,
    "native_grading_or_enrollment_registration_implemented": False,
    "native_qualification_established": False,
    "package_native_paths_promoted": False,
    "prior_tracks_and_original384_768_contracts_unchanged": True,
    "maximum_forwards": 16400, "maximum_input_positions": 2295832,
    "files": {f.relative_to(D).as_posix(): {"bytes": f.stat().st_size,
        "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        for f in sorted(D.rglob("*")) if f.is_file()}}
value["sha256"] = digest(value)
(D / "ROOT_PRESERVATION.json").write_bytes((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
print(json.dumps({"directory": str(D), "files": len(value["files"]), "sha256": value["sha256"]}))
