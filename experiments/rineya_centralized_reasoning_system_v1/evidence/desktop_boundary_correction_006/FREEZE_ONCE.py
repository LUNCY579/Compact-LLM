"""Create-only sixth-round evidence freeze. Static AST/hash checks only."""
import ast
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PKG=ROOT/"experiments/rineya_centralized_reasoning_system_v1"
assert not (OUT/"FREEZE.json").exists()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def methods(path):
    tree=ast.parse(Path(path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="IdentityProbeTests")
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
assert methods(OUT/"source_before/test_owned_runtime_identity_probe.py")==methods(PKG/"tests/test_owned_runtime_identity_probe.py")
def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
before=functions(OUT/"source_before/owned_runtime_identity_probe.py");after=functions(PKG/"owned_runtime_identity_probe.py")
changed=[name for name in before if before[name]!=after[name]]
assert set(before)==set(after) and set(changed)=={"checked_contract","validate_desktop_context","collect_desktop_context"}
preserved={"native.py":"8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3",
           "operations.py":"ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30",
           "contracts/NATIVE_DIAGNOSTIC_V1.json":"d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a",
           "contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V2.json":"d7b2e5bc7673580c59a07ed5de27e8548bf13c5829fedb6e907dc9c12345a409",
           "contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V3.json":"3747f47c5788791336eb4679d6c2b4f16d7221faaac22e97d92cf8d2367ac86f",
           "contracts/DESKTOP_GPU_IDENTITIES_PROBE002_V1.json":"b433c5d1108678638a91304e71fe893fd4ae031c47ca3e325c7547463755a03d",
           "evidence/desktop_profile_integration_001/FREEZE.json":"dc9c9d277b71011912bd90964b3f43481ece652c826417714c73fad0eefd6394",
           "reader.py":"0299cb37b417bf142341509efd9d728263ddcbe89f02e6cc8b49b01f4ec851ab",
           "reader_execution.py":"c66a1b6453f1bad19e0f51eb7d2cf2b763eb0e6e7a09e124eebcb613ef53f9d3"}
for relative,expected in preserved.items():assert sha(PKG/relative)==expected
assert sha(OUT/"source_before/owned_runtime_identity_probe.py")=="ffe327f366420c521d06281c7e8a70694f929154159a7149f368572499709512"
assert sha(OUT/"source_before/test_owned_runtime_identity_probe.py")=="300d6a0b5caefd84654559ee1b3053521b4509a736962b93dadfe00c74707600"
candidates=["owned_runtime_identity_probe.py","tests/test_owned_runtime_identity_probe.py",
            "contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V4.json",
            "authorization/DELEGATED_DESKTOP_BOUNDARY_CORRECTION_006.json"]
record={"schema_version":1,"status":"FROZEN_ROUND6_CANDIDATE_PENDING_FRESH_INDEPENDENT_SCI_OPS_SOURCE_REVIEW",
        "same_milestone":True,"cumulative_source_review_round":6,"builder":"/root/desktop_profile_builder",
        "changed_functions":changed,"original_IdentityProbeTests_methods_AST_unchanged":True,
        "candidate_files":[{"path":r,"sha256":sha(PKG/r)} for r in candidates],
        "preserved_files":[{"path":r,"sha256":h} for r,h in preserved.items()],
        "evidence_files":[{"path":p.relative_to(PKG).as_posix(),"sha256":sha(p)} for p in sorted(OUT.rglob("*")) if p.is_file()],
        "findings_addressed":["DP5-SCI-001","DP5-OPS-001"],"CPU_invocations":1,"CPU_test_groups_passed":5,
        "CPU_raw_test_seconds":0.077,"CPU_command_wall_seconds":0.43800000000010186,
        "test_scope":"ACTUAL_COLLECTOR_WITH_MOCKED_PROCESS_HASH_PRINCIPAL_ADAPTERS",
        "actual_process_GPU_framework_model_tokenizer_DB_actions":0,"actual_probe_attempts":0,
        "unchanged_large_dependency_hash_repeats":0,"actual_noncompute_workload_proved":False,
        "hidden_inprocess_GPU_operations":"UNMEASURED","live_admission_unmeasured":True,
        "remaining_fixed_KEY002_max_runs":1,"remaining_fixed_KEY002_max_seconds":60,"extra_worker_allowance":0,
        "automatic_retry":False,"source_accepted_or_execution_permitted":False,"further_repair_within_round":False}
with (OUT/"FREEZE.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(record,indent=2)+"\n")
print(json.dumps({"FREEZE_sha256":sha(OUT/"FREEZE.json"),"candidate_files":record["candidate_files"]},indent=2))
