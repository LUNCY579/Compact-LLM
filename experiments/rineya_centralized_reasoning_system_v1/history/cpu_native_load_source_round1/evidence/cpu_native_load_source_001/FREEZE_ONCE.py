"""Create-only first CPU source freeze. No native load or qualification."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];PKG=ROOT/"experiments/rineya_centralized_reasoning_system_v1"
assert not (OUT/"FREEZE.json").exists()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
preserved={"native.py":"8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3",
           "operations.py":"ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30",
           "owned_runtime_identity_probe.py":"845d1e6624cea3ce0686aecc2f4e79e96cd1c37f0abd890877eb1212c8a9a8ab",
           "reader.py":"0299cb37b417bf142341509efd9d728263ddcbe89f02e6cc8b49b01f4ec851ab",
           "reader_execution.py":"c66a1b6453f1bad19e0f51eb7d2cf2b763eb0e6e7a09e124eebcb613ef53f9d3",
           "contracts/NATIVE_DIAGNOSTIC_V1.json":"d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a"}
for relative,digest in preserved.items():assert sha(PKG/relative)==digest
cfg=json.loads((PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V1.json").read_text())
for relative,digest in cfg["source_sha256"].items():assert sha(ROOT/relative)==digest
source=OUT/"source";source.mkdir(exist_ok=True)
for relative in ("authorization/DELEGATED_CPU_NATIVE_LOAD_FEASIBILITY_001.json",
                 "audits/CPU_NATIVE_FEASIBILITY_SCOPE_001.md","audits/CPU_NATIVE_LOAD_SCI_SCOPE_001.md","audits/CPU_NATIVE_LOAD_OPS_SCOPE_001.md"):
    with (source/Path(relative).name).open("xb") as handle:handle.write((PKG/relative).read_bytes())
candidates=["cpu_native_load.py","tests/test_cpu_native_load.py","contracts/CPU_NATIVE_LOAD_FEASIBILITY_V1.json",
            "authorization/DELEGATED_CPU_NATIVE_LOAD_FEASIBILITY_001.json"]
record={"schema_version":1,"status":"FIRST_CPU_SOURCE_CANDIDATE_FROZEN_PENDING_INDEPENDENT_SCI_OPS_REVIEW",
        "new_exact_CPU_source_round":1,"default_source_round_limit":3,"old_GPU_round3_quarantine_preserved":True,
        "builder":"/root/desktop_profile_builder","source_files":[{"path":r,"sha256":sha(PKG/r)} for r in candidates],
        "preserved_files":[{"path":r,"sha256":h} for r,h in preserved.items()],
        "evidence_files":[{"path":p.relative_to(PKG).as_posix(),"sha256":sha(p)} for p in sorted(OUT.rglob("*")) if p.is_file()],
        "CPU_fixture_invocations":1,"CPU_fixture_groups_passed":5,"CPU_fixture_raw_seconds":2.169,
        "CPU_fixture_command_wall_seconds":2.561999999998079,"actual_Windows_Job_flags":8704,
        "actual_Job_commit_limit_bytes":2147483648,"actual_stdlib_initial_RSS_bytes":2101248,
        "actual_stdlib_deadline_and_empty_exit_verified":True,"canonical_database_access":False,
        "actual_checkpoint_loads":0,"actual_model_framework_tokenizer_imports":0,"actual_GPU_observations":0,
        "CPU_model_load_fit":"UNMEASURED","canonical_tensor_byte_verification":"NOT_EXECUTED",
        "loaded_CPU_copy_digest_allocation_paths":"SOURCE_ONLY_NOT_FRAMEWORK_TESTED",
        "strict_owned_executable_fixture":"ACTUAL_POSITIVE_CAPTURE; unknown identity decision relies on preserved inventory predicates",
        "actual_CPU_forward_or_BF16_operator_support":"UNMEASURED","hidden_internal_C_activity":"UNMEASURED",
        "actual_registered_CPU_study_attempts":0,"remaining_CPU_fit_attempts":1,"CPU_fit_seconds_max":120,"automatic_retry":False,
        "existing_probe002_untouched":True,"source_accepted_or_launch_permitted":False}
with (OUT/"FREEZE.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(record,indent=2)+"\n")
print(json.dumps({"FREEZE_sha256":sha(OUT/"FREEZE.json"),"source_files":record["source_files"]},indent=2))
