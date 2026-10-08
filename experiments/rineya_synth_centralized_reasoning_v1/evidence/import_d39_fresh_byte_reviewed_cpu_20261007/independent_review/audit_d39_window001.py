"""Independent finite validation of one d39 passive byte window."""
from pathlib import Path
from types import ModuleType,CodeType
import ast,hashlib,importlib.util,json,marshal,os,stat,subprocess,sys
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_current_byte_review001';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');REPO=PACKAGE.parents[1];FILES=[p for p in FROZEN.rglob('*') if p.is_file()]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();receipt_path=FROZEN/'FRESH_IMPORT_CURRENT_BYTE_REVIEW001.json';v=json.loads(receipt_path.read_bytes());assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});assert v['implementation_commit']=='d39d64b1447c8f0baffdebcc4a0a1092f76bbd88';assert BEFORE[str(FROZEN/'review_current_bytes.py')]['sha256']==v['review_script_sha256']=='7286041069771edb48f5a801155c0c8903648279b3043138bddd697e7ab7f8c0';assert BEFORE[str(FROZEN/'selected_inventory.py')]['sha256']==v['reader_source_sha256']=='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892';assert BEFORE[str(FROZEN/'selected_fingerprint_source.py')]['sha256']==v['fingerprint_source_sha256']=='62c25f6cf5f843a08463dbb7c3c573f6a038f036917fc4130da0aab5d4b7e0c5'
sources=v['source_files'];runtime=v['runtime_files'];selected={**sources,**runtime};assert not set(sources)&set(runtime) and (len(sources),len(runtime),len(selected))==(196,8724,8920);assert sum(d['bytes'] for d in selected.values())==v['actual_hashed_bytes']==4068272183;assert v['physical_read_counters']=={'bytes':4068272183,'opens':8920,'opened_metadata_checks':17840}
assert {p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in v['observed_files'].items()}==selected;assert digest(sources)==v['source_inventory_sha256'] and digest(runtime)==v['runtime_inventory_sha256'] and digest(selected)==v['selected_descriptor_map_sha256'];assert digest(sorted(selected))==v['selected_path_list_sha256'];assert v['git_before']==v['git_after']=={'HEAD':v['implementation_commit'],'dirty_porcelain':''};assert v['known_alias_before']==v['known_alias_after'];assert v['byte_budget']==16*1024**3 and v['deadline_seconds']==600 and 0<v['single_window_seconds']<600
for label,anchor in v['metadata_anchors'].items():
    name=Path(anchor['path']).name;p=FROZEN/name;raw=p.read_bytes();assert len(raw)==anchor['bytes'] and hashlib.sha256(raw).hexdigest()==anchor['file_sha256'];body=json.loads(raw);assert body['sha256']==anchor['body_sha256']==digest({k:x for k,x in body.items() if k!='sha256'})
reader=ModuleType('independent_d39_passive_reader');reader.__file__=str(FROZEN/'selected_inventory.py');exec(compile((FROZEN/'selected_inventory.py').read_bytes(),reader.__file__,'exec'),reader.__dict__)
runtime_packet=json.loads((FROZEN/'IMPORT_OBSERVATION_RUNTIME_003.json').read_bytes());layout=runtime_packet['stdlib_layout'];alias={k:layout[k] for k in ('logical_base_prefix','physical_base_prefix','file_attributes','reparse_tag','is_symlink')};R=reader.Readers();rosters={}
for name,row in v['rosters_before'].items():
    root=reader._path(row['root'],R,alias);files={}
    for folder,dirs,leaves in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in row['excluded_directory_names'])
        for d in dirs:m=(Path(folder)/d).lstat();assert not getattr(m,'st_file_attributes',0)&0x400
        for leaf in leaves:
            p=Path(folder)/leaf
            if p.suffix.lower() not in row['suffixes']:continue
            m=p.lstat();assert stat.S_ISREG(m.st_mode) and not p.is_symlink() and not getattr(m,'st_file_attributes',0)&0x400;files[str(p)]={'bytes':m.st_size}
        if not row['recursive']:dirs[:]=[]
    assert files==row['files'];rosters[name]={**row,'files':files}
assert len(rosters)==18 and rosters==v['rosters_before']==v['rosters_after'] and digest(rosters)==v['rosters_sha256']
# Independent code graph encoding; marshal objects are never executed.
def fp(code):
    count=[0,0]
    def visit(x,depth):
        count[0]+=1;assert depth<=64 and count[0]<=65536
        if type(x) is CodeType:return visit((x.co_code,x.co_exceptiontable,x.co_linetable,x.co_argcount,x.co_posonlyargcount,x.co_kwonlyargcount,x.co_nlocals,x.co_stacksize,x.co_flags,x.co_names,x.co_varnames,x.co_freevars,x.co_cellvars,x.co_name,x.co_qualname,x.co_firstlineno,x.co_filename,x.co_consts),depth+1)
        if type(x) is tuple:return tuple(visit(y,depth+1) for y in x)
        if type(x) is frozenset:return frozenset(visit(y,depth+1) for y in x)
        assert x is None or x is Ellipsis or type(x) in (bytes,str,int,float,bool,complex)
        if type(x) in (bytes,str):count[1]+=len(x);assert count[1]<=32*1024**2
        return x
    return visit(code,0)
cache_rows=v['passive_cache_code_equivalence'];assert len(cache_rows)==v['historical_cache_pairs']+v['additional_passive_cache_pairs']==66;cache_bytes=0;cache_source_bytes=0
for text,row in cache_rows.items():
    cache_path=reader._path(text,R,alias);source_path=reader._path(row['source_path'],R,alias);cached=cache_path.read_bytes();source=source_path.read_bytes();cache_bytes+=len(cached);cache_source_bytes+=len(source);assert hashlib.sha256(cached).hexdigest()==row['cache_sha256']==selected[text]['sha256'];assert hashlib.sha256(source).hexdigest()==row['source_sha256']==selected[row['source_path']]['sha256'];assert cached[:4]==importlib.util.MAGIC_NUMBER;flags=int.from_bytes(cached[4:8],'little');assert flags==row['flags'] and flags in (0,1,3);code=marshal.loads(cached[16:]);assert type(code) is CodeType and reader._path(code.co_filename,R,alias)==source_path;assert fp(code)==fp(compile(source,code.co_filename,'exec'))
    assert cached[8:16]==importlib.util.source_hash(source) if flags&1 else int.from_bytes(cached[12:16],'little')==len(source)&0xffffffff
    assert row['cache_executed'] is row['source_execution_history_proven'] is False
# Recheck every current source byte, and one small runtime representative per
# role/roster. This is an independent finite sample, not another 4GB window.
sample=dict(sources)
for role in sorted({d['role'] for d in runtime.values()}):
    p=min((p for p,d in runtime.items() if d['role']==role),key=lambda p:(runtime[p]['bytes'],p));sample[p]=runtime[p]
for row in rosters.values():
    p=min(row['files'],key=lambda p:(runtime[p]['bytes'],p));sample[p]=runtime[p]
probe=reader._collect(sorted(sample),sample,alias_table=alias);assert {p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in probe['files'].items()}==sample
head=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,check=True,text=True).stdout.strip();dirty=subprocess.run(['git','status','--porcelain=v1'],cwd=REPO,capture_output=True,check=True,text=True).stdout;assert head==v['implementation_commit'] and dirty==''
for key in ('native_runtime_qualified','native_context_minted','native_authority_issued','native_launch_authorized','native_API_called','Torch_psutil_ctypes_imported','current_main_or_bootstrap_execution_observed','resources_or_library_baseline_observed','DLL_cookie_or_provider_observed','source_execution_history_proven','native_PASS'):assert v[key] is False
after=pins();assert after==BEFORE;result={'kind':'INDEPENDENT_D39_PASSIVE_CURRENT_BYTE_WINDOW_REVIEW001','basis_file_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),'basis_body_sha256':v['sha256'],'implementation_commit':head,'source_count':196,'runtime_count':8724,'single_window_selected_bytes':4068272183,'physical_stream_counters_joined':True,'rosters_recaptured':18,'roster_sha256':digest(rosters),'independent_cache_code_graph_pairs':66,'cache_bytes_rechecked':cache_bytes,'cache_source_bytes_rechecked':cache_source_bytes,'actual_sample_files':len(sample),'actual_sample_hash_bytes':probe['actual_hashed_bytes'],'all196_current_source_files_rehashed':True,'source_receipt_pins_before':BEFORE,'source_receipt_pins_after':after,'frozen_pins_unchanged':True,'Git_current_d39_clean_verified':True,'whole_4GB_read_not_repeated':True,'later_source_fix_or_future_commit_covered':False,'known_RAM_copy_launch_source_bugs_resolved':False,'native_bootstrap_or_runtime_qualified':False,'native_API_or_framework_called':False,'native_authority_issued':False};result['sha256']=digest(result)
with (HERE/'INDEPENDENT_D39_BYTE_WINDOW001.json').open('x',encoding='utf-8') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(canonical({'sources':196,'runtime':8724,'cache_pairs':66,'sample_files':len(sample),'sample_bytes':probe['actual_hashed_bytes'],'body_sha256':result['sha256']}))
