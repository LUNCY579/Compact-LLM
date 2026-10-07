"""Passive runtime009/current proposal bytes and exact source-derived rosters."""
from pathlib import Path
from types import ModuleType
import ast,hashlib,json,os,stat,sys,time
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
ROOT=PACKAGE.parents[1]
RUNTIME=PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json'
PROPOSAL=PACKAGE/'artifacts/tensor_two_tensor_proposal_work/TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2_001.json'
READER=PACKAGE/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py'
PIN='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def seal(v):return {**v,'sha256':digest(v)}
def metadata(path):
    raw=path.read_bytes();assert len(raw)<=16*1024**2
    v=json.loads(raw);assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'})
    return v,{'path':str(path),'bytes':len(raw),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':v['sha256']}
def main(output):
    started=time.monotonic();runtime,ranchor=metadata(RUNTIME);proposal,panchor=metadata(PROPOSAL)
    raw=READER.read_bytes();assert hashlib.sha256(raw).hexdigest()==PIN
    reader=ModuleType('passive_exact_runtime_refresh');reader.__file__=str(READER);exec(compile(raw,str(READER),'exec'),reader.__dict__)
    current_sources={item['path']:{'bytes':item['bytes'],'sha256':item['sha256'],'role':'python_source'} for item in proposal['closed_source_packet']['source_candidates'].values()}
    current_sources.update(proposal['selected_current_helpers'])
    selected={**runtime['files'],**current_sources};assert len(runtime['files'])==2986
    layout=runtime['stdlib_layout'];alias={k:layout[k] for k in ('logical_base_prefix','physical_base_prefix','file_attributes','reparse_tag','is_symlink')}
    observed=reader._collect(list(selected),selected,alias_table=alias)
    assert {p:{k:v[k] for k in ('bytes','sha256','role')} for p,v in observed['files'].items()}==selected
    # Inspect only the pinned contract definitions that declare finite roots.
    contract=PACKAGE/'artifacts/tensor_observed_admission_work/prefix_tensor_contract.py';source=current_sources[str(contract)];assert hashlib.sha256(contract.read_bytes()).hexdigest()==source['sha256']
    tree=ast.parse(contract.read_bytes());external=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXTERNAL_PACKAGES' for t in node.targets):
            assert isinstance(node.value,ast.Call) and node.value.func.id=='frozenset';external=ast.literal_eval(node.value.args[0])
    assert external is not None
    site=Path(runtime['cache_policy']['interpreter_profile']['executable']).parents[1]/'Lib/site-packages';base=Path(layout['physical_base_prefix'])
    policies={'stdlib_python_source':(base/'Lib',['.py'],['site-packages','__pycache__'],True),
        'stdlib_extension_runtime':(base/'DLLs',['.pyd','.dll'],['__pycache__'],True),
        'interpreter_root_executable_runtime':(base,['.exe','.dll','.zip'],['__pycache__'],False),
        'torch':(site/'torch',['.py','.pyi','.pyd','.dll'],['__pycache__'],True)}
    policies.update({'external_package:'+name:(site/name,['.py','.pyi','.pyd','.dll'],['__pycache__'],True) for name in external if name!='typing_extensions'})
    rosters={};covered=set();errors=[]
    for name,(root,suffixes,excluded,recursive) in policies.items():
        assert root.resolve()==root and root.is_dir();files={}
        for folder,dirs,names in os.walk(root,followlinks=False):
            assert time.monotonic()-started<=600
            dirs[:]=sorted(d for d in dirs if d not in excluded)
            for child in dirs:
                p=Path(folder)/child;m=p.lstat();assert not p.is_symlink() and not(getattr(m,'st_file_attributes',0)&0x400)
            for leaf in names:
                p=Path(folder)/leaf
                if p.suffix.lower() not in suffixes:continue
                m=p.lstat();assert stat.S_ISREG(m.st_mode) and not p.is_symlink() and not(getattr(m,'st_file_attributes',0)&0x400) and len(files)<32768
                files[str(p)]={'bytes':m.st_size}
            if not recursive:dirs[:]=[]
        missing=sorted(set(files)-set(runtime['files']));mismatch=[p for p,d in files.items() if p in runtime['files'] and runtime['files'][p]['bytes']!=d['bytes']]
        if missing or mismatch:errors.append({'roster':name,'unselected_files':missing,'size_mismatches':mismatch})
        rosters[name]={'root':str(root),'suffixes':suffixes,'excluded_directory_names':excluded,'recursive':recursive,'files':files};covered.update(files)
    caches=runtime['cache_policy']['allowed_cache_files'];assert len(caches)==60 and all(runtime['files'][p]==d for p,d in caches.items())
    known=set(runtime['inherited65_map'])|set(runtime['stdlib779_map']);windows={p for p,d in runtime['files'].items() if d['role']=='windows_system_binary'}
    binaries={p for p,d in runtime['files'].items() if d['role'] in ('observer_binary','observer_fixture_binary')};singleton=str(site/'typing_extensions.py')
    unaccounted=sorted(set(runtime['files'])-covered-known-windows-set(caches)-binaries-{singleton})
    assert not errors and not unaccounted
    result=seal({'kind':'PASSIVE_RUNTIME009_CURRENT_BYTE_ROSTER_REFRESH','metadata_anchors':{'runtime009':ranchor,'proposalV2':panchor},
        'reader_sha256':PIN,'contract_descriptor':source,'selected_current_sources':current_sources,'selected_runtime_files':runtime['files'],
        'observed_files':observed['files'],'actual_hashed_bytes':observed['actual_hashed_bytes'],'selected_descriptor_map_sha256':digest(selected),
        'rosters':rosters,'rosters_sha256':digest(rosters),'runtime_map_sha256':digest(runtime['files']),'cache_policy_sha256':digest(runtime['cache_policy']),
        'allowed_cache_files_sha256':digest(caches),'window_seconds':time.monotonic()-started,'physical_rosters_complete_for_selected_policies':True,
        'historical_observer_main_retained_in_bootstrap_map':True,'tensor_main_or_provider_bootstrap_observed':False,
        'current_native_program_execution_proven':False,'resources_or_library_baseline_observed':False,'native_runtime_qualified':False,
        'native_launch_authorized':False,'native_authority_issued':False,'native_PASS':False})
    assert hashlib.sha256(READER.read_bytes()).hexdigest()==PIN and metadata(RUNTIME)[1]==ranchor and metadata(PROPOSAL)[1]==panchor
    with output.open('x',encoding='utf-8') as f:f.write(canonical(result)+'\n')
    print(canonical({'path':str(output),'sources':len(current_sources),'runtime':len(runtime['files']),'rosters':len(rosters),'files':len(selected),'hashed_bytes':result['actual_hashed_bytes'],'seconds':result['window_seconds'],'body_sha256':result['sha256'],'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
