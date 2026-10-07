"""Execute only selected runtime_role/roster_policy definitions, no imports."""
from pathlib import Path
import ast,hashlib,json,sys
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def main(output):
    p=HERE/'EXPANDED_RUNTIME_BYTE_ROSTER001.json';raw=p.read_bytes();value=json.loads(raw);assert value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'})
    c=PACKAGE/'artifacts/tensor_observed_admission_work/prefix_tensor_contract.py';source=c.read_bytes();assert hashlib.sha256(source).hexdigest()==value['current_source_files'][str(c)]['sha256'];tree=ast.parse(source)
    names={'QUALIFIED_LAYOUT','SITE','WINDOWS_SYSTEM_DIRECTORY','EXTERNAL_PACKAGES'};nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets) or isinstance(n,ast.FunctionDef) and n.name in ('runtime_role','roster_policy')];assert len(nodes)==6
    env={'Path':Path,'require':require,'frozenset':frozenset};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(c),'exec'),env)
    original=json.loads((PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json').read_bytes());extra=json.loads((HERE/'MISSING_ROSTER_LEAF_BYTES002.json').read_bytes())
    for text,d in extra['files'].items():assert env['runtime_role'](text,original['inherited65_map'])==d['role']
    for name,policy in value['rosters'].items():
        root,suffixes,excluded,recursive=env['roster_policy'](name);assert (policy['root'],policy['suffixes'],policy['excluded_directory_names'],policy['recursive'])==(str(root),suffixes,excluded,recursive)
    caches=original['cache_policy']['allowed_cache_files'];proposal=json.loads((PACKAGE/'artifacts/tensor_two_tensor_proposal_work/TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2_001.json').read_bytes());observed=proposal['observed009_selection']['maps']['preloaded_module_cache_pins'];assert len(observed)==59 and all(caches[p]==d for p,d in observed.items())
    unobserved={p:d for p,d in caches.items() if p not in observed};assert len(unobserved)==1
    result={'kind':'PASSIVE_RUNTIME_EXPANSION_EXACT_SOURCE_ROLE_REVIEW','contract_sha256':hashlib.sha256(source).hexdigest(),'expanded_receipt_file_sha256':hashlib.sha256(raw).hexdigest(),'expanded_receipt_body_sha256':value['sha256'],'actual_runtime_role_calls':len(extra['files']),'actual_roster_policy_calls':len(value['rosters']),'original_runtime009_files_retained':2986,'expanded_runtime_files':8716,'current_sources_separate':10,'selected_roster_before_after_sha256':value['rosters_sha256'],'observed009_cache_pins':59,'candidate_cache_pins':60,'unobserved_cache_candidate':unobserved,'class_alias_names':sorted(proposal['observed009_selection']['maps']['preloaded_class_aliases']),'native_or_program_execution_observed':False,'native_runtime_qualified':False,'native_launch_authorized':False,'native_authority_issued':False};result['sha256']=digest(result)
    with output.open('x',encoding='utf-8') as f:f.write(canonical(result)+'\n')
    print(canonical({'roles':result['actual_runtime_role_calls'],'rosters':result['actual_roster_policy_calls'],'body_sha256':result['sha256'],'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
