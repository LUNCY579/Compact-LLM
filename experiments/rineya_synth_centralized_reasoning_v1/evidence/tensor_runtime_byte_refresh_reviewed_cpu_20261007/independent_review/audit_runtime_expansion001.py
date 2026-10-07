"""Independent finite map/role/roster audit and selected passive byte probes."""
from pathlib import Path
import ast, hashlib, json, os, stat, sys, time, types
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;PACKAGE=next(p for p in WORK.parents if p.name=='rineya_synth_centralized_reasoning_v1');READER=PACKAGE/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py';CONTRACT=PACKAGE/'artifacts/tensor_observed_admission_work/prefix_tensor_contract.py';RUNTIME=PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json'
FILES=[p for p in WORK.rglob('*') if p.is_file() and 'independent_review' not in p.parts]+[READER,CONTRACT,RUNTIME]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins()
def selected(name):
    p=WORK/name;v=json.loads(p.read_text(encoding='utf-8'));assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});return v
b=selected('RUNTIME009_BYTE_ROSTER_REFRESH002.json');m=selected('MISSING_ROSTER_LEAF_BYTES002.json');e=selected('EXPANDED_RUNTIME_BYTE_ROSTER001.json');t=selected('TENSOR_RUNTIME_BYTE_CANDIDATE001.json');old=json.loads(RUNTIME.read_text(encoding='utf-8'))
assert BEFORE[str(WORK/'RUNTIME009_BYTE_ROSTER_REFRESH002.json')]['sha256']=='1040c6f7898474d857b680794c9f1ce34f4514cfdd25ba69a107ba608089bedf'
assert BEFORE[str(WORK/'MISSING_ROSTER_LEAF_BYTES002.json')]['sha256']=='55798ee04a8a2e94c9d14c8946d364c19788f7f07e7fb96b34be25e98d4eb30b'
assert BEFORE[str(WORK/'EXPANDED_RUNTIME_BYTE_ROSTER001.json')]['sha256']=='1c14b81af3c55a87b3ec38d9189760d7b5045176626c5009682cd1a27324f811'
assert BEFORE[str(READER)]['sha256']=='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
assert len(b['selected_runtime_files'])==2986 and len(b['selected_current_sources'])==10 and len(m['files'])==5730
assert not set(b['selected_runtime_files'])&set(m['files']);assert e['runtime_files']=={**b['selected_runtime_files'],**m['files']};assert len(e['runtime_files'])==8716;assert e['current_source_files']==b['selected_current_sources']
assert sum(x['bytes'] for x in m['files'].values())==m['actual_hashed_bytes']==102009875;assert b['actual_hashed_bytes']+m['actual_hashed_bytes']==e['actual_hash_bytes_across_selected_windows']==4059803314
assert {p:{k:v[k] for k in ('bytes','sha256','role')} for p,v in b['observed_files'].items()}=={**b['selected_runtime_files'],**b['selected_current_sources']}
assert e['rosters_sha256']==e['roster_before_sha256']==e['roster_after_sha256']==digest(e['rosters'])=='916cd0cf144878696edc18169a56d75d8a3ecbb27d506400d06dfdbfd68cbaed'
source=CONTRACT.read_bytes();assert hashlib.sha256(source).hexdigest()==e['current_source_files'][str(CONTRACT)]['sha256'];tree=ast.parse(source)
names={'QUALIFIED_LAYOUT','SITE','WINDOWS_SYSTEM_DIRECTORY','EXTERNAL_PACKAGES'};nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in names for x in n.targets) or isinstance(n,ast.FunctionDef) and n.name in ('runtime_role','roster_policy')];assert len(nodes)==6
def require(v,msg):
    if not v:raise ValueError(msg)
env={'Path':Path,'require':require,'frozenset':frozenset};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CONTRACT),'exec'),env)
roles=0
for p,d in t['files'].items():
    expected='observer_binary' if p in t['observer_binary_files'] else 'windows_system_binary' if p in t['windows_system_files'] else env['runtime_role'](p,t['inherited65_map'])
    assert d['role']==expected;roles+=1
for p,d in m['files'].items():assert env['runtime_role'](p,old['inherited65_map'])==d['role']
started=time.monotonic();recaptured={}
for name,r in e['rosters'].items():
    root,suffixes,excluded,recursive=env['roster_policy'](name);assert (str(root),suffixes,excluded,recursive)==(r['root'],r['suffixes'],r['excluded_directory_names'],r['recursive']);files={};assert root.resolve()==root
    for folder,dirs,leaves in os.walk(root,followlinks=False):
        assert time.monotonic()-started<600;dirs[:]=sorted(x for x in dirs if x not in excluded)
        for x in dirs:
            p=Path(folder)/x;a=p.lstat();assert not p.is_symlink() and not getattr(a,'st_file_attributes',0)&0x400
        for leaf in leaves:
            p=Path(folder)/leaf
            if p.suffix.lower() not in suffixes:continue
            a=p.lstat();assert stat.S_ISREG(a.st_mode) and not p.is_symlink() and not getattr(a,'st_file_attributes',0)&0x400;files[str(p)]={'bytes':a.st_size}
        if not recursive:dirs[:]=[]
    assert files==r['files'];recaptured[name]={**r,'files':files}
assert digest(recaptured)==e['rosters_sha256'];covered=set().union(*(set(r['files']) for r in t['rosters'].values()));known=set(t['inherited65_map'])|set(t['stdlib779_map']);denominator=covered|known|set(t['external_dependency_files'])|set(t['windows_system_files'])|set(t['preloaded_module_cache_pins'])|set(t['observer_binary_files']);assert set(t['files'])==denominator and len(t['files'])==8714
removed=t['removed_observer_only_or_unobserved_leaves'];assert e['runtime_files']=={**t['files'],**removed};assert len(removed)==2 and sum(d['bytes'] for d in removed.values())==20770;assert not set(removed)&(covered|known);assert len(t['preloaded_module_cache_pins'])==59 and len(old['cache_policy']['allowed_cache_files'])==60;assert len(t['preloaded_bootstrap_modules'])==63 and set(t['preloaded_class_aliases'])=={'typing.io','typing.re'} and '__main__' not in t['preloaded_bootstrap_modules']
for v in (b,m,e,t):
    for key in ('native_runtime_qualified','native_launch_authorized','native_authority_issued','native_PASS'):assert v[key] is False
assert t['library_boundary_plan'] is t['library_search_policy'] is None and e['single_atomic_hash_window'] is False
# One smallest new leaf per formerly missing roster, each runtime role, and all
# ten selected sources. This is a new limited byte probe, not a 4GB replay.
sample={};owners=m['roster_owner']
for owner in sorted(set(owners.values())):
    p=min((p for p in owners if owners[p]==owner),key=lambda p:(m['files'][p]['bytes'],p));sample[p]=m['files'][p]
for role in sorted(set(d['role'] for d in t['files'].values())):
    p=min((p for p,d in t['files'].items() if d['role']==role),key=lambda p:(t['files'][p]['bytes'],p));sample[p]=t['files'][p]
sample.update(e['current_source_files']);reader=types.ModuleType('independent_passive_reader');reader.__file__=str(READER);exec(compile(READER.read_bytes(),str(READER),'exec'),reader.__dict__);assert reader.MAX_TOTAL==16*1024**3 and reader.MAX_FILES==32768
probe=reader._collect(sorted(sample),sample);assert {p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in probe['files'].items()}==sample
first=json.loads((WORK/'first_refresh001/FAILURE.json').read_text(encoding='utf-8'));miscount=json.loads((WORK/'missing_selection_count_failure001/FAILURE.json').read_text(encoding='utf-8'));assert first['selected_byte_hash_loop_returned'] is True and first['phase']=='physical_roster_coverage';assert miscount['actual_count']==5730 and miscount['expected_count']==5830 and miscount['bytes_read']==0
after=pins();assert after==BEFORE
receipt={'kind':'INDEPENDENT_PASSIVE_RUNTIME_EXPANSION_AUDIT001','source_and_receipt_pins_before':BEFORE,'source_and_receipt_pins_after':after,'selected_sources_unchanged':True,'original_runtime_files':2986,'original_current_sources':10,'new_roster_leaves':5730,'expanded_runtime_files':8716,'tensor_derivative_runtime_files':8714,'original_roles_called':roles,'missing_roles_called':5730,'source_roster_policy_calls':len(e['rosters']),'actual_rosters_recaptured':len(recaptured),'actual_roster_sha256':digest(recaptured),'tensor_denominator_exact':True,'removed_files':removed,'observed_cache_pins':59,'candidate_cache_pins':60,'sample_hash_paths':list(sorted(sample)),'sample_actual_hashed_bytes':probe['actual_hashed_bytes'],'sample_count':len(sample),'existing_hash_bytes_across_two_windows':4059803314,'fresh_full_map_hash_repeated':False,'first_roster_failure_hash_returned_but_coverage_incomplete':True,'missing_miscount_failed_before_read_bytes':0,'native_or_provider_or_resource_event_facts_observed':False,'native_runtime_qualified':False,'native_launch_authorized':False,'native_authority_issued':False}
receipt['sha256']=digest(receipt)
with (HERE/'INDEPENDENT_RUNTIME_EXPANSION_AUDIT001.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(canonical({'roles':roles,'missing_roles':5730,'rosters':len(recaptured),'sample_files':len(sample),'sample_bytes':probe['actual_hashed_bytes'],'body_sha256':receipt['sha256']}))
