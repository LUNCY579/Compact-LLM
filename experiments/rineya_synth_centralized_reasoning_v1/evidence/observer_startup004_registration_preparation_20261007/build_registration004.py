"""Exact prospective observer004 packet build; passive bytes only."""
from pathlib import Path
from copy import deepcopy
from types import ModuleType
import json, hashlib, subprocess, time

P=Path(__file__).resolve().parents[2]
CFG=P/'configs'
IMPL=subprocess.run(['git','rev-parse','HEAD'],cwd=P.parents[1],capture_output=True,text=True,check=True).stdout.strip()
assert IMPL=='4d7e3882d9c0ddf21957a6b9bcce28bf091713a9'
config=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_CONFIG_003.json').read_text(encoding='utf-8'))
runtime=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_RUNTIME_003.json').read_text(encoding='utf-8'))
registration=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_REGISTRATION_003.json').read_text(encoding='utf-8'))
extra=json.loads((P/'artifacts/observer_bootstrap_retry_preparation_work/finite_bootstrap_candidate.json').read_text(encoding='utf-8'))
def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
def seal(v):
    body={k:x for k,x in v.items() if k!='sha256'}
    return {**body,'sha256':digest(body)}
def write(path,v):
    raw=(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')
    with path.open('xb') as f:f.write(raw)
    return {'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':v['sha256']}
for path,d in config['source_files'].items():
    raw=Path(path).read_bytes()
    d.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
assert not set(runtime['files']).intersection(extra['additive_files'])
runtime['files'].update(extra['additive_files'])
cache=runtime['cache_policy']
cache['mode']='MEASURE_BYTE_VERIFIED_STARTUP_V2'
cache['allowed_bootstrap_modules'].update(extra['additive_bootstrap_modules'])
cache['allowed_bootstrap_modules']['__main__']['source_sha256']=config['source_files'][config['worker_path']]['sha256']
site=Path(next(iter(extra['startup_hook_sources']))).parent
cache['startup_hook_sources']={'directory':str(site),'files':{path:extra['additive_files'][path] for path in extra['startup_hook_sources']}}
cache['allowed_namespace_packages']=extra['allowed_namespace_packages']
config['implementation_commit']=runtime['implementation_commit']=registration['implementation_commit']=IMPL
config['runtime_files']=deepcopy(runtime['files'])
reader_path=P/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py'
reader_raw=reader_path.read_bytes()
assert hashlib.sha256(reader_raw).hexdigest()=='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
reader=ModuleType('selected_passive_observer_startup_hook_projection')
reader.__file__=str(reader_path)
exec(compile(reader_raw,str(reader_path),'exec'),reader.__dict__)
original_roles=deepcopy(reader.ROLES)
reader.ROLES={**reader.ROLES,'startup_hook_source':{'.pth'}}
selection={**config['source_files'],**runtime['files']}
alias={k:runtime['stdlib_layout'][k] for k in ('logical_base_prefix','physical_base_prefix','file_attributes','reparse_tag','is_symlink')}
start=time.monotonic()
read=reader._collect(list(selection),selection,alias_table=alias)
assert {p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in read['files'].items()}==selection
assert reader.ROLES=={**original_roles,'startup_hook_source':{'.pth'}}
read_record=seal({'kind':'ACTUAL_OBSERVER_STARTUP004_PASSIVE_BYTE_REVIEW','implementation_commit':IMPL,
    'reader_sha256':hashlib.sha256(reader_raw).hexdigest(),
    'reader_role_projection':'Only exact selected six .pth descriptors add startup_hook_source suffix; shared tensor policy unchanged.',
    'selected_sources':len(config['source_files']),'selected_runtime':len(runtime['files']),
    'actual_hashed_bytes':read['actual_hashed_bytes'],'selected_descriptor_map_sha256':read['selected_descriptor_map_sha256'],
    'seconds':time.monotonic()-start,'namespace_execution_proven':False,'native_PASS':False,'resources_observed':False})
write(P/'evidence/observer_startup004_actual_byte_review_20261007.json',read_record)
review=seal({'kind':'ROOT_REVIEWED_OBSERVER_STARTUP_BYTE_ADMISSION_V1','implementation_commit':IMPL,
    'source_inventory_sha256':digest(config['source_files']),'runtime_inventory_sha256':digest(runtime['files']),
    'observer_registration_sha256':digest(runtime['library_event_observer']),
    'allowed_initial_loaded_libraries_sha256':digest(runtime['allowed_initial_loaded_libraries']),
    'bootstrap_policy_sha256':digest(cache),'complete_current_byte_review':True})
runtime['qualification_review']=write(P/'evidence/observer_startup_byte_admission_review_004.json',review)
runtime=seal(runtime);config=seal(config)
registration.update(config_body_sha256=config['sha256'],runtime_body_sha256=runtime['sha256'],
    source_files=deepcopy(config['source_files']),runtime_files=deepcopy(runtime['files']))
registration=seal(registration)
refs={name:write(CFG/name,value) for name,value in (
    ('SYNTH_OBSERVER_STARTUP_CONFIG_004.json',config),('SYNTH_OBSERVER_STARTUP_RUNTIME_004.json',runtime),
    ('SYNTH_OBSERVER_STARTUP_REGISTRATION_004.json',registration))}
print({'refs':refs,'byte_review_body':read_record['sha256'],'actual_bytes':read['actual_hashed_bytes'],
    'runtime_files':len(runtime['files']),'sources':len(config['source_files'])})
