"""Finite tensor byte-map derivative; no registration or native qualification."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def main(output):
    expanded_path=HERE/'EXPANDED_RUNTIME_BYTE_ROSTER001.json';raw=expanded_path.read_bytes();expanded=json.loads(raw);assert expanded['sha256']==digest({k:v for k,v in expanded.items() if k!='sha256'})
    old=json.loads((PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json').read_bytes());proposal=json.loads((PACKAGE/'artifacts/tensor_two_tensor_proposal_work/TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2_001.json').read_bytes());maps=proposal['observed009_selection']['maps'];pins=maps['preloaded_module_cache_pins'];assert len(pins)==59
    removed={p:d for p,d in expanded['runtime_files'].items() if d['role']=='observer_fixture_binary' or d['role']=='qualified_bytecode_cache' and p not in pins}
    assert len(removed)==2 and sum(d['bytes'] for d in removed.values())==20770 and not set(removed).intersection(set(old['inherited65_map'])|set(old['stdlib779_map']))
    files={p:d for p,d in expanded['runtime_files'].items() if p not in removed};covered=set().union(*(set(r['files']) for r in expanded['rosters'].values()));assert not covered.intersection(removed)
    external={p:d for p,d in files.items() if p.startswith(str(Path(old['cache_policy']['interpreter_profile']['executable']).parents[1]/'Lib/site-packages')) and not Path(p).is_relative_to(Path(old['cache_policy']['interpreter_profile']['executable']).parents[1]/'Lib/site-packages/torch')}
    windows={p for p,d in files.items() if d['role']=='windows_system_binary'};observer={p for p,d in files.items() if d['role']=='observer_binary'};known=set(old['inherited65_map'])|set(old['stdlib779_map'])
    assert len(observer)==1 and set(files)==covered|known|set(external)|windows|set(pins)|observer
    result={'kind':'PASSIVE_TENSOR_RUNTIME_BYTE_ROSTER_DERIVATIVE_UNQUALIFIED','expanded_receipt':{'path':str(expanded_path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':expanded['sha256']},'files':files,'runtime_file_count':len(files),'removed_observer_only_or_unobserved_leaves':removed,'rosters':expanded['rosters'],'rosters_sha256':expanded['rosters_sha256'],'inherited65_map':old['inherited65_map'],'stdlib779_map':old['stdlib779_map'],'external_dependency_files':external,'windows_system_files':{p:files[p] for p in windows},'observer_binary_files':{p:files[p] for p in observer},'preloaded_module_cache_pins':pins,'preloaded_bootstrap_modules':maps['preloaded_bootstrap_modules'],'preloaded_class_aliases':maps['preloaded_class_aliases'],'interpreter_profile_source_assumptions':proposal['observed009_selection']['interpreter_profile'],'source_execution_history_proven':False,'tensor_main_source':proposal['separate_tensor_main']['staged_source'],'library_search_policy':None,'library_boundary_plan':None,'source_promotion_and_git_joins_complete':False,'current_full_byte_review_registration_issued':False,'native_runtime_qualified':False,'native_launch_authorized':False,'native_authority_issued':False,'native_PASS':False};result['sha256']=digest(result)
    with output.open('x',encoding='utf-8') as f:f.write(canonical(result)+'\n')
    print(canonical({'runtime_count':len(files),'removed':len(removed),'removed_bytes':20770,'cache_pins':len(pins),'body_sha256':result['sha256'],'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
