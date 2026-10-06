"""Explicit legacy51 plus finite V2 actual shared-body CPU controls."""
from pathlib import Path
import builtins, hashlib, io, json, sys, time, unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
STAGE=ROOT/'experiments/rineya_synth_centralized_reasoning_v1/artifacts/prefix_input_owned_execution_work'
OUT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
names=('prefix_library_policy.py','test_library_policy.py','test_library_managed_scope.py','prefix_registered_loader.py')
before={name:sha(STAGE/name) for name in names}
forbidden={'torch','ctypes','psutil','native_worker','native_process','native_nvml','transformers','safetensors','numpy','tensorflow','jax'}
attempts=[];real_import=builtins.__import__
def imports(name,*args,**kwargs):
    if name.split('.',1)[0] in forbidden:
        attempts.append(name);raise AssertionError('actual native/framework/OS import prohibited: '+name)
    return real_import(name,*args,**kwargs)
bad_reads=[];real_open=Path.open
def file_open(path,*args,**kwargs):
    if path.suffix.lower() in {'.dll','.pyd','.exe','.zip','.pyc','.safetensors','.bin','.pt','.pth'}:
        if not any(part.startswith('CPU_inert_DLL_boundary_') for part in path.parts):
            bad_reads.append(str(path));raise AssertionError('real host/runtime/model/cache bytes prohibited')
    return real_open(path,*args,**kwargs)
sys.path.insert(0,str(STAGE));output=io.StringIO();started=time.monotonic()
with patch.object(builtins,'__import__',side_effect=imports),patch.object(Path,'open',file_open):
    import test_library_policy as legacy
    import test_library_managed_scope as managed
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(legacy.LibraryBoundaryTests),
                             unittest.defaultTestLoader.loadTestsFromTestCase(managed.ManagedScopeTests)])
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
after={name:sha(STAGE/name) for name in names}
stamp=str(time.time_ns());text_path=OUT/('LIBRARY_MANAGED_SCOPE_CPU_'+stamp+'.txt')
with text_path.open('x',encoding='utf-8') as stream:stream.write(output.getvalue())
receipt={'kind':'FINITE_MANAGED_LIBRARY_SCOPE_V2_AND_UNCHANGED_V1_CPU_SOURCE_CONTROLS',
 'passed':result.wasSuccessful(),'tests_run':result.testsRun,'legacy_tests':51,'V2_tests':result.testsRun-51,
 'skips':len(result.skipped),'elapsed_seconds':time.monotonic()-started,'sources_before':before,'sources_after':after,
 'sources_unchanged':before==after,'forbidden_import_attempts':attempts,
 'actual_host_runtime_model_cache_byte_attempts':bad_reads,'fixture_only':True,'native_PASS':False,
 'native_authority_issued':False,'native_observer_implemented':False,'root_tracked_source_edits':False,
 'output_file':str(text_path),'output_sha256':sha(text_path),
 'limits':['Directory/module/event/WinAPI observations are explicit inert suppliers only.',
 'Complete native observer, actual fresh identity/source/bootstrap provenance, current runtime/ownership and terminal observation export remain separately required and absent.',
 'No process-creation event coverage, universal foreign-cookie claim, preventive sandbox, cloned-namespace defense, scientific fit or M0/M1/M2 authority is established.']}
receipt['sha256']=digest(receipt);path=OUT/('LIBRARY_MANAGED_SCOPE_CPU_'+stamp+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'receipt':str(path),'body_sha256':receipt['sha256'],'passed':receipt['passed'],
 'tests_run':result.testsRun,'sources_unchanged':before==after,'forbidden_imports':attempts,'bad_byte_reads':bad_reads}))
if not result.wasSuccessful() or result.skipped or before!=after or attempts or bad_reads:
    print(output.getvalue());raise SystemExit(1)
