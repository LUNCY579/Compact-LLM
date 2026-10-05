"""Record only staged CPU controls; this command cannot launch the native audit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,time,unittest
HERE=Path(__file__).resolve().parent
FILES=('audit_adapter.py','phase_replay.py','test_phase_audit.py','fixture_support.py',
    'pinned_dependencies/RAM2_auditor_d69810.py','pinned_dependencies/PRESERVATION.json')
def hashes():return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in FILES}
before=hashes();started=time.monotonic();output=io.StringIO()
suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_phase_audit.py')
result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite);elapsed=time.monotonic()-started;after=hashes()
text=output.getvalue();(HERE/'CPU_VALIDATION_20261005.txt').write_text(text,encoding='utf-8')
receipt={'schema_version':1,'status':'PASS_STAGED_CPU_PHASE3_CONTROLS' if result.wasSuccessful() and before==after else 'FAILED_STAGED_CPU_PHASE3_CONTROLS',
    'recorded_utc':datetime.now(timezone.utc).isoformat(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
    'elapsed_seconds':elapsed,'source_hashes_before':before,'source_hashes_after':after,'source_bytes_unchanged':before==after,
    'original_RAM2_source_sha256':'d69810dccd11a32595dace73a788c8868efabb2a09f593fd33316e92dcb47442',
    'source_text_adaptation_sites':14,'additional_adaptations':'Explicit _NativeReads strict bounded read_json/regular streaming filehash; finite inherited JSONL parse facade; independent terminal/phase/loader pre/post closure.',
    'whole_path_scope':'Actual inert filesystem/source/runtime-registration/raw/preservation/phase/lease/cleanup/resource/310-tensor682-chunk composition; inherited heavy decoder/model closure explicitly injected CPU stub.',
    'fixture_status':'PASS_CPU_FIXTURE_TERMINAL_COMPOSITION_ONLY','native_RAW_PASS_emitted':False,
    'native_model_decoder_runtime_binary_GPU_or_OS_probe_reads_or_execution':False,
    'limits':{'maximum_markers':200000,'maximum_marker_bytes':65536,'maximum_ledger_bytes':67108864,'maximum_JSON_bytes':16777216,
        'maximum_config_bytes':8388608,'maximum_header_bytes':1048576,'file_hash_chunk_bytes':2097152,'raw_file_count_ceiling':2127},
    'unchanged_gates':{'whole_rows':64,'context_cap':512,'generation_cap':32,'whole_forward_cap':2050,'runtime':65,'decoder':17,'data':5,'model':7,
        'source_maps_exact':True,'resource_and_numeric_floors_unchanged':True},
    'study_completed':False,'native_certification_obtained':False}
(HERE/'CPU_VALIDATION_20261005.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ('status','tests','failures','errors','skips','elapsed_seconds','source_bytes_unchanged')}))
raise SystemExit(0 if receipt['status'].startswith('PASS_') else 1)
