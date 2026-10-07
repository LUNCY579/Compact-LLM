"""Static byte/source library expectations; not a loaded transition witness."""
from pathlib import Path
import ast,hashlib,json,sys
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def read(path):
    raw=path.read_bytes();v=json.loads(raw);assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});return v,{'path':str(path),'bytes':len(raw),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':v['sha256']}
def descriptor(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
def main(output):
    runtime,anchor=read(PACKAGE/'artifacts/runtime_refresh_work/TENSOR_RUNTIME_BYTE_CANDIDATE001.json');files=runtime['files']
    observation,oanchor=read(PACKAGE/'evidence/observer_startup_attempt009_20261007/preserved/OBSERVER_STARTUP_INVENTORY.json')
    snapshots=observation['initial_module_snapshots'];assert len(snapshots)==2 and snapshots[0]['libraries']==snapshots[1]['libraries'] and all(x['complete'] is True for x in snapshots)
    initial=sorted(x['path'] for x in snapshots[0]['libraries']);assert len(initial)==35 and len(set(initial))==35 and all(p in files for p in initial)
    libraryfiles={p:d for p,d in files.items() if Path(p).suffix.lower() in ('.exe','.dll','.pyd') and d['role']!='observer_binary'}
    directories=sorted({str(Path(p).parent) for p in libraryfiles});assert 0<len(directories)<=128 and len(libraryfiles)<=4096
    provider=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/history/WORKER_LIBRARY_PUBLICATION_SOURCE_FROZEN_017/source/prefix_library_policy.py'
    providerdesc=descriptor(provider);assert providerdesc['sha256']=='2f8d5c30d85c066413accab77bfe80c2eb20cc184772cd44ec097acb8775b1c1'
    site=Path(observation['bootstrap']['interpreter_profile']['executable']).parents[1]/'Lib/site-packages';torchinit=site/'torch/__init__.py';source=torchinit.read_bytes();assert hashlib.sha256(source).hexdigest()==files[str(torchinit)]['sha256']
    tree=ast.parse(source);function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_load_dll_libraries')
    literal_dlls=sorted({n.value for n in ast.walk(function) if isinstance(n,ast.Constant) and type(n.value) is str and n.value.lower().endswith('.dll')})
    torchdlls=sorted(p for p in libraryfiles if Path(p).parent==site/'torch/lib' and Path(p).suffix.lower()=='.dll')
    extensions=sorted(p for p in libraryfiles if Path(p).suffix.lower()=='.pyd' and p not in initial)
    plan={'kind':'STATIC_LIBRARY_EXPECTATION_INPUTS_PENDING_EXACT_TRANSITION_PLAN','runtime_candidate':anchor,'observer009_reference':oanchor,
        'provider_source':{'path':str(provider),'descriptor':providerdesc},'library_files':libraryfiles,'search_directories':directories,
        'system_directory':str(Path('C:/Windows/System32')),'expected_initial_loaded_libraries':initial,
        'initial_set_provenance':'OBSERVED_OBSERVER009_ONLY_EXPECTATION_FOR_FUTURE_TENSOR_PROGRAM','actual_tensor_initial_baseline_observed':False,
        'torch_source':{'path':str(torchinit),'descriptor':files[str(torchinit)]},'windows_torch_dll_function_sha256':hashlib.sha256(ast.dump(function,include_attributes=True).encode()).hexdigest(),
        'torch_windows_source_literal_DLL_names':literal_dlls,'explicit_torch_lib_dll_leaves':torchdlls,
        'potential_unpreloaded_extension_leaves':extensions,'framework_transition':None,'extension_transitions':None,'library_boundary_plan':None,
        'topology_source':'Existing reviewed library policy tests use inert synthetic DLLs, not recorded Torch imports',
        'exact_runtime_rejects_initial_or_transition_mismatch':True,'unregistered_search_allowed':False,
        'requirements':['exact per-extension create/exec new-library sets','exact framework directly-loaded new-library set after nested acknowledgements','ordered boundary id,parent_id,name plan derived before result','native own-process initial baseline and provider identity record'],
        'model':False,'forwards':0,'backwards':0,'optimizer_steps':0,'scoring':False,'native_runtime_qualified':False,'native_launch_authorized':False,'native_authority_issued':False,'native_PASS':False}
    plan['sha256']=digest(plan)
    with output.open('x',encoding='utf-8') as f:f.write(canonical(plan)+'\n')
    print(canonical({'initial':len(initial),'libraries':len(libraryfiles),'directories':len(directories),'torch_dlls':len(torchdlls),'potential_extensions':len(extensions),'body_sha256':plan['sha256'],'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
