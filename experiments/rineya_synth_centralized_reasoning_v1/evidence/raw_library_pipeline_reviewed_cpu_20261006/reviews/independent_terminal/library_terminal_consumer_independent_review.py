"""Independent final actual worker/copy/auditor14; explicit CPU fixture scope."""
from pathlib import Path
import builtins,hashlib,importlib.util,io,json,sys,time,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
STAGE=ROOT/'experiments/rineya_synth_centralized_reasoning_v1/artifacts/prefix_input_owned_execution_work'
OUT=Path(__file__).resolve().parent
NAMES=('prefix_tensor_contract.py','prefix_input_owned_worker.py','prefix_native_pipeline.py',
 'prefix_registered_loader.py','prefix_library_policy.py','prefix_library_evidence.py',
 'prefix_input_owned_supervisor.py','prefix_input_owned_auditor.py','native_fixture_support.py',
 'test_library_terminal_evidence.py','test_library_terminal_consumer.py','test_library_failure_evidence.py')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
before={name:sha(STAGE/name) for name in NAMES}
assert before['prefix_input_owned_auditor.py']=='b06fa0824ff13f1b1c51ec405e60408bdd31aa471716a4e587d40dfd52a544f5'
assert before['prefix_library_policy.py']=='2f8d5c30d85c066413accab77bfe80c2eb20cc184772cd44ec097acb8775b1c1'
forbidden={'torch','ctypes','psutil','native_worker','native_process','native_nvml','worker_ram_probe','transformers','safetensors','numpy','tensorflow','jax'}
attempts=[];real_import=builtins.__import__
def imports(name,*args,**kwargs):
    if name.split('.',1)[0] in forbidden:
        attempts.append(name);raise AssertionError('actual native/framework/OS import prohibited: '+name)
    return real_import(name,*args,**kwargs)
bad_reads=[];real_open=Path.open
def file_open(path,*args,**kwargs):
    if path.suffix.lower() in {'.dll','.pyd','.exe','.zip','.pyc','.safetensors','.bin','.pt','.pth'}:
        if not any(part.startswith(('CPU_library_manifest_','CPU_library_evidence_')) for part in path.parts):
            bad_reads.append(str(path));raise AssertionError('real host/runtime/model/cache bytes prohibited')
    return real_open(path,*args,**kwargs)
def source(name):
    path=STAGE/(name+'.py');module=importlib.util.module_from_spec(importlib.util.spec_from_file_location(name,path))
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__);return module
sys.path.insert(0,str(STAGE));output=io.StringIO();started=time.monotonic()
with patch.object(builtins,'__import__',side_effect=imports),patch.object(Path,'open',file_open):
    full=source('test_library_terminal_evidence');consumer=source('test_library_terminal_consumer')
    classes=[value for module in (full,consumer) for value in module.__dict__.values()
             if isinstance(value,type) and issubclass(value,unittest.TestCase) and value.__module__==module.__name__]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes])
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
after={name:sha(STAGE/name) for name in NAMES};stamp=str(time.time_ns())
text_path=OUT/('LIBRARY_TERMINAL_CONSUMER_REVIEW_'+stamp+'.txt')
with text_path.open('x',encoding='utf-8') as stream:stream.write(output.getvalue())
receipt={'kind':'INDEPENDENT_ACTUAL_LIBRARY_TERMINAL_PUBLICATION_PRESERVATION_CONSUMER_CPU_REVIEW',
 'passed':result.wasSuccessful(),'tests_run':result.testsRun,'skips':len(result.skipped),
 'elapsed_seconds':time.monotonic()-started,'source_before':before,'source_after':after,'sources_unchanged':before==after,
 'forbidden_import_attempts':attempts,'actual_host_runtime_model_cache_byte_attempts':bad_reads,
 'fixture_only':True,'native_PASS':False,'native_authority_issued':False,'root_source_edits':False,
 'output_file':str(text_path),'output_sha256':sha(text_path),
 'source_assessment':'Actual owned main/guard journal and separate publication pass through actual copy preservation and independently selected-source replay. Original context/anchors/owner/policy/provider/source maps, external boundary plan and executing interpreter PID/FILETIME join terminal birth population. Original/copy and source drift plus re-signed event or outside-owned-window journals reject. Float-window endpoints use exact adjacent-float integer ratios.',
 'limits':['This independently reviews Root consumer/orchestration; my own helper producer implementation is not counted as independent validation.',
 'Historical runtime842/Git/layout, framework/resources/cadence/death and loader observations are explicit inert suppliers; complete native observer/bootstrap/runtime/resources remain unqualified.',
 'Direct sidecar replay preserves pre-core incomplete/failure packets unqualified; the full native tensor-body path still requires actual CORE/worker/report and ownership provenance and does not certify pre-core failures.',
 'No native worker/framework/model/binary/GPU/OS action, NativePASS, scientific fit or M0/M1/M2 study evidence is issued.']}
receipt['sha256']=digest(receipt);path=OUT/('LIBRARY_TERMINAL_CONSUMER_REVIEW_'+stamp+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'receipt':str(path),'body_sha256':receipt['sha256'],'passed':result.wasSuccessful(),
 'tests_run':result.testsRun,'sources_unchanged':before==after,'forbidden_imports':attempts,'bad_byte_reads':bad_reads}))
if not result.wasSuccessful() or result.testsRun!=14 or result.skipped or before!=after or attempts or bad_reads:
    print(output.getvalue());raise SystemExit(1)
