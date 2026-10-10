import hashlib,json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
assert not (OUT/"TEST_RESULT.json").exists()
command=[sys.executable,"-B","-m","unittest","-f","-v","experiments.rineya_centralized_reasoning_system_v1.tests.test_cpu_native_load.CPUSourceRound2Tests"]
began=time.monotonic()
with (OUT/"CPU_STDOUT.txt").open("xb") as stdout,(OUT/"CPU_STDERR.txt").open("xb") as stderr:
    try:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=30,check=False)
        code=result.returncode;status="PASSED" if code==0 else "MATERIAL_FAILURE_STOP_NO_RETRY"
    except subprocess.TimeoutExpired:code=None;status="CPU_FIXTURE_DEADLINE_STOP_NO_RETRY"
record={"status":status,"source_round":2,"command":command,"invocations":1,"wall_seconds":time.monotonic()-began,"wall_seconds_max":30,"exit_code":code,
        "fixture_groups":5,"actual_stdlib_suspended_preJob_cleanup":True,"actual_model_framework_tokenizer_GPU_checkpoint_actions":0,
        "canonical_database_access":False,"CPU_load_fit":"UNMEASURED",
        "raw_sha256":{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ("CPU_STDOUT.txt","CPU_STDERR.txt")}}
with (OUT/"TEST_RESULT.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
