"""Final finite static registration, one no-model fixture invocation, freeze."""
import ast,hashlib,json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];PKG=ROOT/"experiments/rineya_centralized_reasoning_system_v1"
assert not (OUT/"FREEZE.json").exists() and not (OUT/"TEST_RESULT.json").exists()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
before=functions(OUT/"source_before/cpu_native_load.py");after=functions(PKG/"cpu_native_load.py")
changed=[name for name in before if before[name]!=after[name]]
assert set(before)==set(after) and set(changed)=={"checked_contract","allowed_cuda_import_call","arm_boundary"}
cfg=json.loads((PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V2.json").read_text())
assert sha(PKG/"authorization/DELEGATED_CPU_NATIVE_SOURCE_CORRECTION_003.json")=="a201e42c7d18439a8dead889c7da6c57a136d74081b90d765c3a96f2e08192b5"
cfg["contract_id"]="CRS-CPU-NATIVE-LOAD-FEASIBILITY-V3";cfg["new_CPU_source_round"]=3
cfg["status"]="FINAL_SOURCE_ROUND3_CANDIDATE_PENDING_FRESH_SCI_OPS_REVIEW"
cfg["source_round3_authority_sha256"]="a201e42c7d18439a8dead889c7da6c57a136d74081b90d765c3a96f2e08192b5"
root=Path(cfg["CUDA_main_source"]).parent
bootstrap={}
for leaf in ("__init__","_device_limits","gds","_utils","graphs","green_contexts","streams"):
    path=root/(leaf+".py");module="torch.cuda" if leaf=="__init__" else "torch.cuda."+leaf
    tree=ast.parse(path.read_text(encoding="utf-8"))
    classes=[{"name":node.name,"firstlineno":node.lineno} for node in ast.walk(tree) if isinstance(node,ast.ClassDef)]
    assert sha(path)==cfg["CUDA_definition_source_sha256"][str(path)]
    bootstrap[module]={"filename":str(path),"body_sha256":sha(path),"class_definitions":classes}
cfg["CUDA_bootstrap_definitions"]=bootstrap
for relative in cfg["source_sha256"]:cfg["source_sha256"][relative]=sha(ROOT/relative)
with (PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V3.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(cfg,indent=2)+"\n")
command=[sys.executable,"-B","-m","unittest","-f","-v","experiments.rineya_centralized_reasoning_system_v1.tests.test_cpu_native_load.CPUSourceRound3Tests"]
began=time.monotonic()
with (OUT/"CPU_STDOUT.txt").open("xb") as stdout,(OUT/"CPU_STDERR.txt").open("xb") as stderr:
    try:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=30,check=False)
        code=result.returncode;status="PASSED" if code==0 else "MATERIAL_FAILURE_FINAL_ROUND_STOP_NO_RETRY"
    except subprocess.TimeoutExpired:code=None;status="FINAL_FIXTURE_DEADLINE_STOP_NO_RETRY"
receipt={"status":status,"command":command,"source_round":3,"invocations":1,"fixture_groups":3,"wall_seconds":time.monotonic()-began,
         "wall_seconds_max":30,"exit_code":code,"actual_model_framework_tokenizer_GPU_checkpoint_canonicalDB_actions":0,
         "actual_CPU_study_attempts":0,"actual_Torch_bootstrap":"NOT_EXECUTED","CPU_fit":"UNMEASURED",
         "raw_sha256":{name:sha(OUT/name) for name in ("CPU_STDOUT.txt","CPU_STDERR.txt")}}
with (OUT/"TEST_RESULT.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(receipt,indent=2)+"\n")
prior=json.loads((PKG/"evidence/cpu_native_load_source_002/FREEZE.json").read_text())
for entry in prior["preserved_files"]:assert sha(PKG/entry["path"])==entry["sha256"]
assert sha(PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V2.json")=="019c202afa302a9172a2d025591b9ad19bc508ccae485fbc8290405442287413"
paths=["cpu_native_load.py","tests/test_cpu_native_load.py","contracts/CPU_NATIVE_LOAD_FEASIBILITY_V3.json","authorization/DELEGATED_CPU_NATIVE_SOURCE_CORRECTION_003.json"]
freeze={"status":"FINAL_CPU_SOURCE_ROUND3_FROZEN_PENDING_FRESH_SCI_OPS" if code==0 else "FINAL_ROUND3_FAILED_QUARANTINE",
        "source_round":3,"source_rounds_max":3,"no_fourth_repair_reset_or_extra_model_attempt":True,"changed_functions":changed,
        "builder":"/root/desktop_profile_builder","source_files":[{"path":p,"sha256":sha(PKG/p)} for p in paths],
        "preserved_files":prior["preserved_files"],"source2_freeze_sha256":sha(PKG/"evidence/cpu_native_load_source_002/FREEZE.json"),
        "preserved_V2_sha256":sha(PKG/"contracts/CPU_NATIVE_LOAD_FEASIBILITY_V2.json"),
        "evidence_files":[{"path":p.relative_to(PKG).as_posix(),"sha256":sha(p)} for p in sorted(OUT.rglob("*")) if p.is_file()],
        "test_receipt":receipt,"actual_CPU_study_attempts":0,"source_or_launch_accepted":False}
with (OUT/"FREEZE.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(freeze,indent=2)+"\n")
print(json.dumps({"freeze_sha256":sha(OUT/"FREEZE.json"),"source_files":freeze["source_files"],"test_receipt":receipt},indent=2))
