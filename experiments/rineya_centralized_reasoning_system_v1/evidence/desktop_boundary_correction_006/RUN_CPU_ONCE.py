"""One fail-fast targeted CPU invocation; deterministic collector adapters only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
assert not (OUT/"TEST_RESULT.json").exists()
command=[sys.executable,"-B","-m","unittest","-f","-v",
         "experiments.rineya_centralized_reasoning_system_v1.tests.test_owned_runtime_identity_probe.DesktopBoundaryCorrectionTests"]
started=time.monotonic()
with (OUT/"CPU_STDOUT.txt").open("xb") as stdout,(OUT/"CPU_STDERR.txt").open("xb") as stderr:
    try:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=60,check=False)
        code=result.returncode;status="PASSED" if code==0 else "MATERIAL_FAILURE_STOP_NO_RETRY"
    except subprocess.TimeoutExpired:code=None;status="CPU_DEADLINE_FAILURE_STOP_NO_RETRY"
record={"status":status,"command":command,"invocations":1,"wall_seconds":time.monotonic()-started,
        "wall_seconds_max":60,"exit_code":code,"test_groups":5,"actual_collector_mocked":True,
        "actual_process_GPU_framework_model_tokenizer_DB_actions":0,"actual_probe_attempts":0,
        "unchanged_large_dependency_hash_repeats":0,"actual_noncompute_workload_proved":False,
        "raw_sha256":{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ("CPU_STDOUT.txt","CPU_STDERR.txt")}}
with (OUT/"TEST_RESULT.json").open("x",encoding="utf-8",newline="\n") as handle:handle.write(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
