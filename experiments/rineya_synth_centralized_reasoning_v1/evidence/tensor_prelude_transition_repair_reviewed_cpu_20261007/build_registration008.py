"""Exact observer transition-repair registration; passive bytes only."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import json,hashlib,subprocess,sys,time
P=Path(__file__).resolve().parents[2]
ROOT=P.parents[1]
CFG=P/'configs'
IMPL=sys.argv[1]
assert len(IMPL)==40 and subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==IMPL
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
c=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_CONFIG_007.json').read_text(encoding='utf-8'))
r=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_RUNTIME_007.json').read_text(encoding='utf-8'))
g=json.loads((CFG/'SYNTH_OBSERVER_STARTUP_REGISTRATION_007.json').read_text(encoding='utf-8'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def seal(v):
    b={k:x for k,x in v.items() if k!='sha256'}
    return {**b,'sha256':digest(b)}
def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:assert f.write(raw)==len(raw)
    assert path.read_bytes()==raw
    return {'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256']}
changes={}
for text,d in c['source_files'].items():
    raw=Path(text).read_bytes();blob=subprocess.check_output(['git','cat-file','blob',IMPL+':'+Path(text).relative_to(ROOT).as_posix()],cwd=ROOT);assert raw==blob
    pin=hashlib.sha256(raw).hexdigest()
    if pin!=d['sha256']:changes[text]={'before':d['sha256'],'after':pin}
    d.update(bytes=len(raw),sha256=pin)
assert set(changes)=={str(P/'observer_startup_measurement_v1/worker.py')}
assert len(c['source_files'])==27 and len(r['files'])==2983
r['cache_policy']['allowed_bootstrap_modules']['__main__']['source_sha256']=c['source_files'][c['worker_path']]['sha256']
c['implementation_commit']=r['implementation_commit']=g['implementation_commit']=IMPL
c['runtime_files']=deepcopy(r['files'])
reader_path=P/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py'
raw=reader_path.read_bytes();assert hashlib.sha256(raw).hexdigest()=='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
reader=ModuleType('selected_transition_passive_reader');reader.__file__=str(reader_path);exec(compile(raw,str(reader_path),'exec'),reader.__dict__)
selection={**c['source_files'],**r['files']};alias={k:r['stdlib_layout'][k] for k in ('logical_base_prefix','physical_base_prefix','file_attributes','reparse_tag','is_symlink')};start=time.monotonic();observed=reader._collect(list(selection),selection,alias_table=alias)
assert {p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in observed['files'].items()}==selection
record=seal({'kind':'ACTUAL_OBSERVER_STARTUP008_PASSIVE_BYTE_REVIEW','implementation_commit':IMPL,'reader_sha256':hashlib.sha256(raw).hexdigest(),'reader_roles_changed':False,'changed_source_descriptors':changes,'selected_sources':27,'selected_runtime':2983,'actual_hashed_bytes':observed['actual_hashed_bytes'],'selected_descriptor_map_sha256':observed['selected_descriptor_map_sha256'],'seconds':time.monotonic()-start,'cache_source_equivalence_qualified':False,'current_tensor_bootstrap_qualified':False,'native_PASS':False,'resources_observed':False})
write(P/'evidence/observer_startup008_actual_byte_review_20261007.json',record)
review=seal({'kind':'ROOT_REVIEWED_OBSERVER_STARTUP_BYTE_ADMISSION_V1','implementation_commit':IMPL,'source_inventory_sha256':digest(c['source_files']),'runtime_inventory_sha256':digest(r['files']),'observer_registration_sha256':digest(r['library_event_observer']),'allowed_initial_loaded_libraries_sha256':digest(r['allowed_initial_loaded_libraries']),'bootstrap_policy_sha256':digest(r['cache_policy']),'complete_current_byte_review':True})
r['qualification_review']=write(P/'evidence/observer_startup_byte_admission_review_008.json',review)
r=seal(r);c=seal(c);g.update(config_body_sha256=c['sha256'],runtime_body_sha256=r['sha256'],source_files=deepcopy(c['source_files']),runtime_files=deepcopy(r['files']));g=seal(g)
for name,value in [('CONFIG',c),('RUNTIME',r),('REGISTRATION',g)]:print(name,json.dumps(write(CFG/('SYNTH_OBSERVER_STARTUP_'+name+'_008.json'),value)))
print('PASS',27,2983,observed['actual_hashed_bytes'])
