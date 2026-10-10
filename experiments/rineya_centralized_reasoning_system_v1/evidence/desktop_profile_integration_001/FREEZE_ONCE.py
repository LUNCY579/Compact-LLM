"""Create-only candidate evidence freeze; no test, native API or DB operation."""
import ast
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PKG=ROOT/"experiments/rineya_centralized_reasoning_system_v1"
assert not (OUT/"FREEZE.json").exists()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def methods(path):
    tree=ast.parse(Path(path).read_text(encoding="utf-8"))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="IdentityProbeTests")
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
assert methods(OUT/"source_before/tests/test_owned_runtime_identity_probe.py")==methods(PKG/"tests/test_owned_runtime_identity_probe.py")
preserved={"native.py":"8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3",
           "operations.py":"ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30",
           "contracts/NATIVE_DIAGNOSTIC_V1.json":"d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a",
           "contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V2.json":"d7b2e5bc7673580c59a07ed5de27e8548bf13c5829fedb6e907dc9c12345a409",
           "reader.py":"0299cb37b417bf142341509efd9d728263ddcbe89f02e6cc8b49b01f4ec851ab",
           "reader_execution.py":"c66a1b6453f1bad19e0f51eb7d2cf2b763eb0e6e7a09e124eebcb613ef53f9d3"}
for relative,expected in preserved.items(): assert digest(PKG/relative)==expected
candidates=["owned_runtime_identity_probe.py","tests/test_owned_runtime_identity_probe.py",
            "contracts/DESKTOP_GPU_IDENTITIES_PROBE002_V1.json","contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V3.json",
            "authorization/OWNER_EXCEPTION_DESKTOP_PROFILE_GRANTED_2026-10-04.json"]
evidence=[p for p in OUT.rglob("*") if p.is_file()]
freeze={"schema_version":1,"status":"FROZEN_CANDIDATE_PENDING_FRESH_INDEPENDENT_SCI_OPS_REVIEW",
        "same_milestone_round":5,"builder":"/root/desktop_profile_builder",
        "candidate_files":[{"path":r,"sha256":digest(PKG/r)} for r in candidates],
        "preserved_files":[{"path":r,"sha256":h} for r,h in preserved.items()],
        "evidence_files":[{"path":p.relative_to(PKG).as_posix(),"sha256":digest(p)} for p in sorted(evidence)],
        "original_IdentityProbeTests_methods_AST_unchanged":True,
        "CPU_invocations":1,"CPU_tests_passed":8,"CPU_tests_wall_seconds":1.346,
        "actual_selected_identity_bytes":965409492,"actual_selected_static_hash_seconds":0.7650000000012369,
        "actual_probe_attempts":0,"canonical_database_access":False,"framework_model_tokenizer_GPU_imports":0,
        "actual_noncompute_workload_proved":False,"hidden_inprocess_GPU_operations":"UNMEASURED",
        "live_context_and_GPU_admission_not_measured":True,"new_worker_allowance":0,
        "remaining_fixed_KEY002_max_runs":1,"remaining_fixed_KEY002_max_wall_seconds":60,"automatic_retry":False,
        "source_acceptance_and_execution_permission":False,"further_material_repair_in_this_round":False}
with (OUT/"FREEZE.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(freeze,indent=2)+"\n")
print(json.dumps({"FREEZE_sha256":digest(OUT/"FREEZE.json"),"candidate_files":freeze["candidate_files"]},indent=2))
