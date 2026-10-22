import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];PKG=ROOT/"experiments/rineya_centralized_reasoning_system_v1"
assert not (OUT/"FREEZE.json").exists()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
prior=json.loads((PKG/"evidence/cpu_native_load_source_001/FREEZE.json").read_text())
for entry in prior["preserved_files"]:assert sha(PKG/entry["path"])==entry["sha256"]
assert sha(PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V1.json")=="8160ba80aa52a547a96aa605ad31a0a88a3bc605da779de3aef09ca4025c97d8"
assert sha(OUT/"source_before/cpu_native_load.py")=="f1a05707751e3e56705b9105357c06f753e94372717dbea086c2cad30e715ec4"
assert sha(OUT/"source_before/test_cpu_native_load.py")=="fa37ae75fd563efa12af3eb320173cd51d410482f0c9df11e4fbd9c3d52ff40f"
paths=["cpu_native_load.py","tests/test_cpu_native_load.py","contracts/CPU_NATIVE_LOAD_FEASIBILITY_V2.json","authorization/DELEGATED_CPU_NATIVE_SOURCE_CORRECTION_002.json"]
record={"status":"ROUND2_SOURCE_FROZEN_PENDING_FRESH_SCI_OPS_REVIEW","source_round":2,"source_rounds_max":3,
        "builder":"/root/desktop_profile_builder","source_files":[{"path":p,"sha256":sha(PKG/p)} for p in paths],
        "preserved_files":prior["preserved_files"],"prior_V1_sha256":sha(PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V1.json"),
        "prior_freeze_sha256":sha(PKG/"evidence/cpu_native_load_source_001/FREEZE.json"),
        "evidence_files":[{"path":p.relative_to(PKG).as_posix(),"sha256":sha(p)} for p in sorted(OUT.rglob("*")) if p.is_file()],
        "CPU_fixture_invocations":1,"CPU_fixture_groups_passed":5,"CPU_fixture_raw_seconds":0.225,
        "CPU_command_wall_seconds":0.5629999999982829,"actual_known_suspended_preJob_handle_cleanup_verified":True,
        "model_framework_tokenizer_GPU_checkpoint_canonicalDB_actions":0,"actual_CPU_study_attempts":0,
        "CPU_model_fit":"UNMEASURED","hidden_C_effects":"UNMEASURED","actual_Torch_import_path":"NOT_EXECUTED",
        "new_CPU_attempts_granted":0,"same_fixed_CPU_key_one120s_attempt":True,"automatic_retry":False,"source_or_launch_accepted":False}
with (OUT/"FREEZE.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(record,indent=2)+"\n")
print(json.dumps({"freeze_sha256":sha(OUT/"FREEZE.json"),"source_files":record["source_files"]},indent=2))
