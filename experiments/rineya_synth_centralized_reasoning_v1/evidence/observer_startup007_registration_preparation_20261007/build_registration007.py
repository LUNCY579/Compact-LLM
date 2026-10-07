"""Exact prospective prelude observer packet; passive file bytes, no native calls."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import hashlib,json,subprocess,sys,time

P=Path(__file__).resolve().parents[2]
ROOT=P.parents[1]
CFG=P/'configs'
IMPL=sys.argv[1]
assert len(IMPL)==40 and all(c in '0123456789abcdef' for c in IMPL)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==IMPL
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
config=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_CONFIG_006.json').read_text(encoding='utf-8'))
runtime=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_RUNTIME_006.json').read_text(encoding='utf-8'))
registration=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_REGISTRATION_006.json').read_text(encoding='utf-8'))
plan_path=P/'observer_startup_measurement_v1/tensor_prelude_plan_001.json'
plan=json.loads(plan_path.read_text(encoding='utf-8'))
candidate=json.loads((P/'artifacts/tensor_prelude_observer_work/ROOT_FINITE_CACHE_CANDIDATE_001.json').read_text(encoding='utf-8'))

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
def seal(v):
    body={k:x for k,x in v.items() if k!='sha256'}
    return {**body,'sha256':digest(body)}
def anchor(path,value):return {'path':str(path),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'body_sha256':value['sha256']}
def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')
    with path.open('xb') as stream:assert stream.write(raw)==len(raw)
    assert path.read_bytes()==raw
    return {'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256']}
def source(path,role='python_source'):
    raw=Path(path).read_bytes()
    blob=subprocess.check_output(['git','cat-file','blob',IMPL+':'+Path(path).relative_to(ROOT).as_posix()],cwd=ROOT)
    assert raw==blob
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}

assert plan['sha256']==digest({k:v for k,v in plan.items() if k!='sha256'})
assert candidate['sha256']==digest({k:v for k,v in candidate.items() if k!='sha256'})
for path,old in list(config['source_files'].items()):config['source_files'][path]=source(path,old['role'])
prelude_path=P/'observer_startup_measurement_v1/prelude.py'
config['source_files'][str(prelude_path)]=source(prelude_path)
config['source_files'][str(plan_path)]=source(plan_path,'source_json')
config['source_files'][plan['packet_path']]=source(plan['packet_path'],'source_json')
for entry in plan['sources'].values():
    actual=source(entry['path'])
    assert actual==entry['descriptor']
    config['source_files'][entry['path']]=actual
for path,d in plan['helper_sources'].items():
    assert source(path)==d
    config['source_files'][path]=d
assert len(config['source_files'])==27
assert len(candidate['files'])==57 and not set(candidate['files']).intersection(runtime['files'])
runtime['files'].update(candidate['files'])
assert len(runtime['files'])==2983
cache=runtime['cache_policy']
cache['mode']='MEASURE_FROZEN_TENSOR_PRELUDE_NO_SITE_V4'
cache['tensor_prelude']=anchor(plan_path,plan)
cache['allowed_cache_files']=deepcopy(candidate['files'])
cache['allowed_bootstrap_modules']['__main__']['source_sha256']=config['source_files'][config['worker_path']]['sha256']
config['implementation_commit']=runtime['implementation_commit']=registration['implementation_commit']=IMPL
config['runtime_files']=deepcopy(runtime['files'])

reader_path=P/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py'
raw=reader_path.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
reader=ModuleType('selected_prelude_passive_reader');reader.__file__=str(reader_path)
exec(compile(raw,str(reader_path),'exec'),reader.__dict__)
selection={**config['source_files'],**runtime['files']}
alias={k:runtime['stdlib_layout'][k] for k in ('logical_base_prefix','physical_base_prefix','file_attributes','reparse_tag','is_symlink')}
started=time.monotonic()
read=reader._collect(list(selection),selection,alias_table=alias)
assert {p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in read['files'].items()}==selection
record=seal({'kind':'ACTUAL_OBSERVER_STARTUP007_PASSIVE_BYTE_REVIEW','implementation_commit':IMPL,
    'reader_sha256':hashlib.sha256(raw).hexdigest(),'reader_roles_changed':False,
    'selected_sources':len(config['source_files']),'selected_runtime':len(runtime['files']),
    'actual_hashed_bytes':read['actual_hashed_bytes'],'selected_descriptor_map_sha256':read['selected_descriptor_map_sha256'],
    'seconds':time.monotonic()-started,'finite_added_cache_files':candidate['files'],
    'cache_source_equivalence_qualified':False,'current_tensor_bootstrap_qualified':False,'native_PASS':False,'resources_observed':False})
write(P/'evidence/observer_startup007_actual_byte_review_20261007.json',record)
review=seal({'kind':'ROOT_REVIEWED_OBSERVER_STARTUP_BYTE_ADMISSION_V1','implementation_commit':IMPL,
    'source_inventory_sha256':digest(config['source_files']),'runtime_inventory_sha256':digest(runtime['files']),
    'observer_registration_sha256':digest(runtime['library_event_observer']),
    'allowed_initial_loaded_libraries_sha256':digest(runtime['allowed_initial_loaded_libraries']),
    'bootstrap_policy_sha256':digest(cache),'complete_current_byte_review':True})
runtime['qualification_review']=write(P/'evidence/observer_startup_byte_admission_review_007.json',review)
runtime=seal(runtime);config=seal(config)
registration.update(config_body_sha256=config['sha256'],runtime_body_sha256=runtime['sha256'],source_files=deepcopy(config['source_files']),runtime_files=deepcopy(runtime['files']))
registration=seal(registration)
for name,value in [('CONFIG',config),('RUNTIME',runtime),('REGISTRATION',registration)]:
    print(name,json.dumps(write(CFG/('SYNTH_OBSERVER_STARTUP_'+name+'_007.json'),value)))
print('PASS',len(config['source_files']),len(runtime['files']),read['actual_hashed_bytes'])
