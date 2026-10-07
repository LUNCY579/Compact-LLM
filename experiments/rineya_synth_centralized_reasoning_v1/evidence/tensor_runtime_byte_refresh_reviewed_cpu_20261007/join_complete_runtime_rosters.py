"""Join passive byte receipts and recapture exact declared physical rosters."""
from pathlib import Path
import hashlib,json,os,stat,sys,time
HERE=Path(__file__).resolve().parent
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def selected(path):
    raw=path.read_bytes();v=json.loads(raw);assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});return v,{'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':v['sha256']}
def capture(policies,started):
    output={}
    for name,policy in policies.items():
        root=Path(policy['root']);assert root.resolve()==root and root.is_dir();files={}
        for folder,dirs,names in os.walk(root,followlinks=False):
            assert time.monotonic()-started<=600;dirs[:]=sorted(d for d in dirs if d not in policy['excluded_directory_names'])
            for child in dirs:
                p=Path(folder)/child;m=p.lstat();assert not p.is_symlink() and not(getattr(m,'st_file_attributes',0)&0x400)
            for leaf in names:
                p=Path(folder)/leaf
                if p.suffix.lower() not in policy['suffixes']:continue
                m=p.lstat();assert stat.S_ISREG(m.st_mode) and not p.is_symlink() and not(getattr(m,'st_file_attributes',0)&0x400) and len(files)<32768;files[str(p)]={'bytes':m.st_size}
            if not policy['recursive']:dirs[:]=[]
        output[name]={**policy,'files':files}
    return output
def main(output):
    started=time.monotonic();basis,banchor=selected(HERE/'RUNTIME009_BYTE_ROSTER_REFRESH002.json');extra,eanchor=selected(HERE/'MISSING_ROSTER_LEAF_BYTES002.json')
    assert extra['basis_file_sha256']==banchor['file_sha256'];before=capture(basis['rosters'],started);assert before==basis['rosters']
    runtime={**basis['selected_runtime_files'],**extra['files']};sources=basis['selected_current_sources'];assert not set(basis['selected_runtime_files']).intersection(extra['files']) and len(runtime)==8716
    observed={p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in basis['observed_files'].items()}
    assert all(observed[p]==d for p,d in basis['selected_runtime_files'].items()) and all(observed[p]==d for p,d in sources.items())
    for policy in before.values():assert all(p in runtime and runtime[p]['bytes']==d['bytes'] for p,d in policy['files'].items())
    for p,d in {**runtime,**sources}.items():
        m=Path(p).lstat();assert stat.S_ISREG(m.st_mode) and m.st_size==d['bytes'] and not Path(p).is_symlink() and not(getattr(m,'st_file_attributes',0)&0x400)
    after=capture(before,started);assert after==before
    result={'kind':'CURRENT_PASSIVE_EXPANDED_RUNTIME_BYTE_ROSTER_CANDIDATE','basis':banchor,'missing_leaf_receipt':eanchor,'runtime_files':runtime,'current_source_files':sources,'runtime_count':len(runtime),'current_source_count':len(sources),'original_runtime_files_retained_unchanged':len(basis['selected_runtime_files']),'new_exact_roster_leaves':len(extra['files']),'actual_hash_bytes_across_selected_windows':basis['actual_hashed_bytes']+extra['actual_hashed_bytes'],'rosters':before,'rosters_sha256':digest(before),'roster_before_sha256':digest(before),'roster_after_sha256':digest(after),'runtime_map_sha256':digest(runtime),'source_map_sha256':digest(sources),'original_runtime009_cache_policy_sha256':basis['cache_policy_sha256'],'runtime009_cache_candidates':60,'observed009ordinary_cache_files':59,'unobserved_candidate_cache':'C:\\Users\\clclc\\AppData\\Roaming\\uv\\python\\cpython-3.11.15-windows-x86_64-none\\Lib\\ctypes\\__pycache__\\util.cpython-311.pyc','physical_rosters_complete_for_selected_policies':True,'seconds':time.monotonic()-started,'single_atomic_hash_window':False,'fresh_runtime_source_cache_hash_required_before_launch':True,'current_tensor_main_provider_bootstrap_observed':False,'library_transition_and_resource_event_facts_observed':False,'native_runtime_qualified':False,'native_launch_authorized':False,'native_authority_issued':False,'native_PASS':False};result['sha256']=digest(result)
    with output.open('x',encoding='utf-8') as f:f.write(canonical(result)+'\n')
    print(canonical({'runtime':len(runtime),'sources':len(sources),'rosters':len(before),'total_hash_bytes':result['actual_hash_bytes_across_selected_windows'],'body_sha256':result['sha256'],'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'roster_sha256':result['rosters_sha256']}))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
