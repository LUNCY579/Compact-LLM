"""Independent finite actual owned-entry replay, explicit CPU fixture scope."""
from pathlib import Path
import builtins, hashlib, importlib.util, io, json, sys, time, unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
STAGE=ROOT/'experiments/rineya_synth_centralized_reasoning_v1/artifacts/prefix_input_owned_execution_work'
OUT=Path(__file__).resolve().parent
NAMES=('prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_native_pipeline.py',
       'prefix_registered_loader.py','prefix_library_policy.py','test_owned_library_entry.py')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
before={name:sha(STAGE/name) for name in NAMES}
forbidden={'torch','ctypes','psutil','native_worker','native_process','native_nvml','worker_ram_probe','transformers','safetensors','numpy','tensorflow','jax'}
attempts=[];real_import=builtins.__import__
def imports(name,*args,**kwargs):
    if name.split('.',1)[0] in forbidden:
        attempts.append(name);raise AssertionError('actual native/framework/OS import prohibited: '+name)
    return real_import(name,*args,**kwargs)
bad_reads=[];real_open=Path.open
def file_open(path,*args,**kwargs):
    if path.suffix.lower() in {'.dll','.pyd','.exe','.zip','.pyc','.safetensors','.bin','.pt','.pth'}:
        if not any(part.startswith('CPU_library_manifest_') for part in path.parts):
            bad_reads.append(str(path));raise AssertionError('real host/runtime/model/cache bytes prohibited')
    return real_open(path,*args,**kwargs)
sys.path.insert(0,str(STAGE));output=io.StringIO();started=time.monotonic()
with patch.object(builtins,'__import__',side_effect=imports),patch.object(Path,'open',file_open):
    path=STAGE/'test_owned_library_entry.py'
    module=importlib.util.module_from_spec(importlib.util.spec_from_file_location('independent_owned_library_entry',path))
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.OwnedLibraryEntryControls)
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
after={name:sha(STAGE/name) for name in NAMES};stamp=str(time.time_ns())
text_path=OUT/('OWNED_LIBRARY_ENTRY_REVIEW_'+stamp+'.txt')
with text_path.open('x',encoding='utf-8') as stream:stream.write(output.getvalue())
receipt={'kind':'INDEPENDENT_ACTUAL_OWNED_MAIN_INSTALLER_CONTEXT_PIPELINE_CPU_SOURCE_REVIEW',
 'passed':result.wasSuccessful(),'tests_run':result.testsRun,'skips':len(result.skipped),
 'elapsed_seconds':time.monotonic()-started,'source_before':before,'source_after':after,
 'sources_unchanged':before==after,'forbidden_import_attempts':attempts,
 'actual_host_runtime_model_cache_byte_attempts':bad_reads,'root_source_edits':False,
 'fixture_only':True,'native_PASS':False,'native_authority_issued':False,
 'library_terminal_evidence_join_complete':False,'native_observer_implemented':False,
 'output_file':str(text_path),'output_sha256':sha(text_path),
 'source_assessment':'Actual main captures its selected installer authority checker and original worker functions/C instance. The selected pipeline receives that original C/context, consumes the opaque current installation before suppliers and guards framework import. Caller callbacks/public selector/pipeline APIs refuse; revoked/wrong-instance/CPU-promoted authority refuses. Primary main exceptions must remain primary when cleanup also fails.',
 'limits':['The observations and framework/resource/lifecycle suppliers are inert CPU fixtures; historical842 runtime/Git/layout and already-loaded stdlib bootstrap remain explicit injected assumptions.',
 'Same-process malicious reconstruction/mutation of trusted Python globals/classes/closures is excluded from this approved control-flow claim; no security boundary is established.',
 'Complete native observer, current bootstrap/runtime/owned physical evidence and terminal scope/event/cleanup evidence joins remain pending.',
 'Public real OS CLI, native DLL/framework/GPU execution, full native readiness, scientific fit and M0/M1/M2 qualification are not established.']}
receipt['sha256']=digest(receipt);path=OUT/('OWNED_LIBRARY_ENTRY_REVIEW_'+stamp+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'receipt':str(path),'body_sha256':receipt['sha256'],'passed':result.wasSuccessful(),
 'tests_run':result.testsRun,'sources_unchanged':before==after,'forbidden_imports':attempts,'bad_byte_reads':bad_reads}))
if not result.wasSuccessful() or result.testsRun!=13 or result.skipped or before!=after or attempts or bad_reads:
    print(output.getvalue());raise SystemExit(1)
