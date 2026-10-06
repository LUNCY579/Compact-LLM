"""Actual C metadata/context/observer binding tests, inert bytes only."""
from pathlib import Path
import builtins,hashlib,io,json,sys,time,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
STAGE=ROOT/'experiments/rineya_synth_centralized_reasoning_v1/artifacts/prefix_input_owned_execution_work'
OUT=Path(__file__).resolve().parent
NAMES=('prefix_tensor_contract.py','test_library_observer_registration.py','outer_context_fixture.py')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
before={name:sha(STAGE/name) for name in NAMES};attempts=[];bad_reads=[]
forbidden={'torch','ctypes','psutil','native_worker','native_process','native_nvml','worker_ram_probe','transformers','safetensors','numpy'}
real_import=builtins.__import__;real_open=Path.open
def imports(name,*args,**kwargs):
    if name.split('.',1)[0] in forbidden:attempts.append(name);raise AssertionError('actual native/framework/OS import prohibited: '+name)
    return real_import(name,*args,**kwargs)
def file_open(path,*args,**kwargs):
    if path.suffix.lower() in {'.dll','.pyd','.exe','.zip','.pyc','.safetensors','.bin','.pt','.pth'}:
        if not any(part.startswith('CPU_observer_registration_') for part in path.parts):
            bad_reads.append(str(path));raise AssertionError('real host/runtime/model/cache bytes prohibited')
    return real_open(path,*args,**kwargs)
sys.path.insert(0,str(STAGE));output=io.StringIO();started=time.monotonic()
with patch.object(builtins,'__import__',side_effect=imports),patch.object(Path,'open',file_open):
    import test_library_observer_registration as test
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(test.ObserverRegistrationControls))
after={name:sha(STAGE/name) for name in NAMES};stamp=str(time.time_ns())
text_path=OUT/('OBSERVER_REGISTRATION_CPU_'+stamp+'.txt')
with text_path.open('x',encoding='utf-8') as stream:stream.write(output.getvalue())
value={'kind':'SOURCE_OBSERVER_REGISTRATION_AND_ORIGINAL_CONTEXT_CAPTURE_CPU_CONTROLS','passed':result.wasSuccessful(),
 'tests_run':result.testsRun,'skips':len(result.skipped),'elapsed_seconds':time.monotonic()-started,
 'source_before':before,'source_after':after,'sources_unchanged':before==after,
 'forbidden_import_attempts':attempts,'actual_host_runtime_model_cache_byte_attempts':bad_reads,
 'fixture_only':True,'native_PASS':False,'native_authority_issued':False,'compiled_DLL_loaded':False,
 'native_observer_qualified':False,'root_tracked_source_edits':False,
 'output_file':str(text_path),'output_sha256':sha(text_path),
 'limits':['Historical65/779/Git/layout are injected existing seams; new selected registration/receipt/source/inert-DLL hashes and physical source context reads actually execute.',
 'The CPU source class and synthetic positive handle are explicit fixtures, never DLL or ABI/provenance observations.',
 'Native source/toolchain/build/current binary/ABI/startup/bootstrap qualification and explicit launch permission remain Root obligations.']}
value['sha256']=digest(value);path=OUT/('OBSERVER_REGISTRATION_CPU_'+stamp+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'receipt':str(path),'body_sha256':value['sha256'],'passed':value['passed'],'tests_run':result.testsRun,'source_stable':before==after,'forbidden_imports':attempts,'bad_byte_reads':bad_reads}))
if not result.wasSuccessful() or result.skipped or before!=after or attempts or bad_reads:
    print(output.getvalue());raise SystemExit(1)
