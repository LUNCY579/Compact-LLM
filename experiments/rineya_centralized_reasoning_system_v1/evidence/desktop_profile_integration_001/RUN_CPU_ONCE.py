"""One targeted CPU invocation, wall ceiling60s, raw output retained."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[4]
assert not (OUT/"TEST_RESULT.json").exists()
module="experiments.rineya_centralized_reasoning_system_v1.tests.test_owned_runtime_identity_probe"
command=[sys.executable,"-B","-m","unittest","-f","-v",module+".DesktopProfileTests",
         module+".IdentityProbeTests.test_deadline_independent_of_blocked_sample_and_late_check_rejects",
         module+".IdentityProbeTests.test_exact_reserved_job_required_other_lease_pending_wrong_epoch_worker_reject",
         module+".IdentityProbeTests.test_hash_individual_distinct_bounds_before_reads"]
began=time.monotonic()
with (OUT/"CPU_STDOUT.txt").open("xb") as stdout,(OUT/"CPU_STDERR.txt").open("xb") as stderr:
    try:
        result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=60,check=False)
        exit_code=result.returncode;status="PASSED" if exit_code==0 else "MATERIAL_FAILURE_STOP_NO_RETRY"
    except subprocess.TimeoutExpired:
        exit_code=None;status="CPU_DEADLINE_FAILURE_STOP_NO_RETRY"
elapsed=time.monotonic()-began
record={"status":status,"command":command,"invocations":1,"wall_seconds":elapsed,"wall_seconds_max":60,
        "exit_code":exit_code,"framework_model_tokenizer_GPU_imports":0,"actual_probe_attempts":0,
        "canonical_database_access":False,"positive_context_scope":"SYNTHETIC_VALIDATOR_ONLY",
        "actual_noncompute_workload_proved":False,"live_GPU_context_admission_measured":False,
        "raw_sha256":{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ("CPU_STDOUT.txt","CPU_STDERR.txt")}}
(OUT/"TEST_RESULT.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print(json.dumps(record,indent=2))
