"""Prospective no-site observer registration; byte reads only, no imports of runtime payloads."""
from pathlib import Path
from copy import deepcopy
from types import ModuleType
import hashlib, json, subprocess, sys, time

P=Path(__file__).resolve().parents[2]
CFG=P/'configs'
IMPL=sys.argv[1]
assert len(IMPL)==40 and all(c in '0123456789abcdef' for c in IMPL)
assert subprocess.run(['git','rev-parse','HEAD'],cwd=P.parents[1],capture_output=True,text=True,check=True).stdout.strip()==IMPL
assert not subprocess.run(['git','status','--porcelain'],cwd=P.parents[1],capture_output=True,text=True,check=True).stdout
config=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_CONFIG_005.json').read_text(encoding='utf-8'))
runtime=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_RUNTIME_005.json').read_text(encoding='utf-8'))
registration=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_REGISTRATION_005.json').read_text(encoding='utf-8'))
original=deepcopy(runtime)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()

def seal(value):
    body={k:v for k,v in value.items() if k!='sha256'}
    return {**body,'sha256':digest(body)}

def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode('ascii')
    with path.open('xb') as stream:
        assert stream.write(raw)==len(raw)
    assert path.read_bytes()==raw
    return {'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256']}

contract_path=P/'observer_startup_measurement_v1/contract.py'
contract=ModuleType('selected_no_site_observer_contract')
contract.__file__=str(contract_path)
exec(compile(contract_path.read_bytes(),str(contract_path),'exec'),contract.__dict__)

cache=runtime['cache_policy']
names=set(contract.STARTUP_MODULE_LEAVES)
removed={cache['allowed_bootstrap_modules'][n]['path'] for n in names}
removed.update(cache['startup_hook_sources']['files'])
for row in cache['allowed_namespace_packages'].values():
    removed.update(row['files'])
assert len(removed)==14 and sum(runtime['files'][p]['bytes'] for p in removed)==844093
for path in removed:
    del runtime['files'][path]
assert not set(removed).intersection(runtime['allowed_initial_loaded_libraries'])
modules={n:d for n,d in cache['allowed_bootstrap_modules'].items() if n not in names}
runtime['cache_policy']={'mode':'MEASURE_BYTE_VERIFIED_STARTUP_NO_SITE_V3',
    'allowed_bootstrap_modules':modules,'imports':list(cache['imports']),
    'interpreter_profile':contract.expected_no_site_profile(runtime['stdlib_layout'])}
assert len(runtime['files'])==2926 and len(modules)==772
for path,d in config['source_files'].items():
    raw=Path(path).read_bytes()
    blob=subprocess.run(['git','cat-file','blob',IMPL+':'+Path(path).relative_to(P.parents[1]).as_posix()],cwd=P.parents[1],capture_output=True,check=True).stdout
    assert raw==blob
    d.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
modules['__main__']['source_sha256']=config['source_files'][config['worker_path']]['sha256']
config['implementation_commit']=runtime['implementation_commit']=registration['implementation_commit']=IMPL
config['runtime_files']=deepcopy(runtime['files'])

reader_path=P/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py'
reader_raw=reader_path.read_bytes()
assert hashlib.sha256(reader_raw).hexdigest()=='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
reader=ModuleType('selected_no_site_passive_reader')
reader.__file__=str(reader_path)
exec(compile(reader_raw,str(reader_path),'exec'),reader.__dict__)
selection={**config['source_files'],**runtime['files']}
alias={k:runtime['stdlib_layout'][k] for k in ('logical_base_prefix','physical_base_prefix','file_attributes','reparse_tag','is_symlink')}
start=time.monotonic()
read=reader._collect(list(selection),selection,alias_table=alias)
assert {p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in read['files'].items()}==selection
record=seal({'kind':'ACTUAL_OBSERVER_STARTUP006_PASSIVE_BYTE_REVIEW','implementation_commit':IMPL,
    'reader_sha256':hashlib.sha256(reader_raw).hexdigest(),'reader_role_projection':'unchanged original passive reader; no startup-hook role',
    'selected_sources':len(config['source_files']),'selected_runtime':len(runtime['files']),
    'actual_hashed_bytes':read['actual_hashed_bytes'],'selected_descriptor_map_sha256':read['selected_descriptor_map_sha256'],
    'removed_file_descriptors':{p:original['files'][p] for p in sorted(removed)},
    'seconds':time.monotonic()-start,'source_execution_history_proven':False,'native_PASS':False,'resources_observed':False})
write(P/'evidence/observer_startup006_actual_byte_review_20261007.json',record)
review=seal({'kind':'ROOT_REVIEWED_OBSERVER_STARTUP_BYTE_ADMISSION_V1','implementation_commit':IMPL,
    'source_inventory_sha256':digest(config['source_files']),'runtime_inventory_sha256':digest(runtime['files']),
    'observer_registration_sha256':digest(runtime['library_event_observer']),
    'allowed_initial_loaded_libraries_sha256':digest(runtime['allowed_initial_loaded_libraries']),
    'bootstrap_policy_sha256':digest(runtime['cache_policy']),'complete_current_byte_review':True})
runtime['qualification_review']=write(P/'evidence/observer_startup_byte_admission_review_006.json',review)
runtime=seal(runtime)
config=seal(config)
registration.update(config_body_sha256=config['sha256'],runtime_body_sha256=runtime['sha256'],
    source_files=deepcopy(config['source_files']),runtime_files=deepcopy(runtime['files']))
registration=seal(registration)
for name,value in [('CONFIG',config),('RUNTIME',runtime),('REGISTRATION',registration)]:
    print(name,json.dumps(write(CFG/('SYNTH_OBSERVER_STARTUP_'+name+'_006.json'),value)))
print('PASS',len(config['source_files']),len(runtime['files']),read['actual_hashed_bytes'])
