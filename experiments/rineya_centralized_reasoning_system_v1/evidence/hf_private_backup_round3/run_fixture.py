"""One candidate3 targeted+regression invocation. HF is fake; no live inventory."""
import time
START=time.monotonic()
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

OUT=Path(__file__).resolve().parent
PACKAGE=OUT.parents[1]
ROOT=PACKAGE.parents[1]
SCOPE_SHA='826946e67776b97d552b70ae23aebe97b7271267c3069696375122a68045384e'
CORRECTION_SHA='b124b16d75e4006104d1e2744baec842867b44d1a7a653c97d7297621a772a27'
assert hashlib.sha256((PACKAGE/'authorization/DELEGATED_HF_BACKUP_HTML_RETENTION_003.json').read_bytes()).hexdigest()==SCOPE_SHA
assert hashlib.sha256((PACKAGE/'authorization/HF_BACKUP_HTML_SCOPE_PATH_CORRECTION_003.json').read_bytes()).hexdigest()==CORRECTION_SHA
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('frozen_hf_backup_tests',PACKAGE/'tests/test_hub_backup.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
stream=io.StringIO()
suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.BackupTests)
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
elapsed=time.monotonic()-START
raw=stream.getvalue()
with (OUT/'RAW_TEST_OUTPUT.txt').open('x',encoding='utf-8',newline='\n') as f:f.write(raw)
summary={
 'status':'PASS_LOCAL_FAKE_HF_CANDIDATE3' if result.wasSuccessful() and elapsed<=60 else 'FAIL_FROZEN_NO_REPAIR_OR_RERUN',
 'candidate':3,'max_source_candidates':3,'source_round_budget_exhausted_after_this_outcome':True,
 'invocations':1,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
 'elapsed_through_tests_and_imports_seconds':elapsed,'clock_start':START,
 'scope_sha256':SCOPE_SHA,'original_preservation_scope_sha256':'397056076f5eb96b899ce5cfacc99ad37a6f4cb0029a8aef49c95cb077276fd4',
 'scope_path_correction_sha256':CORRECTION_SHA,
 'coverage':'Four new exact-pinned HTML source/archive/hash/case/path/UTF8/NUL/secret/forged-archive groups plus every existing meaningful backup regression in the same invocation.',
 'HF_network_upload_all_fake':True,'actual_helper_inventory_or_prepare_of_primary':False,
 'actual_HF_network_upload_UV_framework_model_hostDB_GPU_process_observation':0,
 'no_source007_mutation_or_source008_start':True,'no_after_outcome_source_repair_or_retry':True,
 'current_helper_qualified_for_actual_inventory_or_upload':False,
 'next_gate':'Independent nonbuilder SCI/OPS source/contract/raw-test review. Candidate3 rejection quarantines backup source; no fourth repair or upload workaround.'
}
with (OUT/'TEST_RESULT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(summary,f,indent=2);f.write('\n')
files=[]
for p in [PACKAGE/'hub_backup.py',PACKAGE/'tests/test_hub_backup.py',PACKAGE/'contracts/HF_PRIVATE_BACKUP_V1.json',
          PACKAGE/'history/hf_private_backup_round2/PRESERVATION.json',*sorted(OUT.iterdir())]:
    if p.is_file():files.append({'path':str(p.relative_to(PACKAGE)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
freeze={'status':'FROZEN_BACKUP_SOURCE_CANDIDATE3_STOP_FOR_INDEPENDENT_REVIEW_NO_ACTUAL_HELPER_CALL',
 'scope_sha256':SCOPE_SHA,'path_correction_sha256':CORRECTION_SHA,
 'preservation_manifest_sha256':'c89419c9bcb7b36c29833614e0e9aa9a1b940b0599998814b4ce2e6541fe4ffa',
 'files':files,'result_status':summary['status'],'fixture_invocations':1,
 'elapsed_through_raw_result_and_source_hashing_seconds':time.monotonic()-START,
 'candidate':3,'max_source_candidates':3,'candidate3_exhausts_source_round_budget':True,
 'current_source_approved':False,'actual_remote_or_native_actions':0}
with (OUT/'FREEZE.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(freeze,f,indent=2);f.write('\n')
print(json.dumps({'status':summary['status'],'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                  'whole_fixture_elapsed_through_final_source_freeze_seconds':time.monotonic()-START}))
raise SystemExit(0 if summary['status']=='PASS_LOCAL_FAKE_HF_CANDIDATE3' else 1)
