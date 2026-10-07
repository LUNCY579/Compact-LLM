"""Hash only the finite missing executable leaves captured by refresh002."""
from pathlib import Path
from types import ModuleType
import hashlib,json,time,sys
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def main(output):
    basis=HERE/'RUNTIME009_BYTE_ROSTER_REFRESH002.json';raw=basis.read_bytes();assert hashlib.sha256(raw).hexdigest()=='1040c6f7898474d857b680794c9f1ce34f4514cfdd25ba69a107ba608089bedf'
    v=json.loads(raw);assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'})
    selected={};owners={}
    for gap in v['roster_selection_errors']:
        for text in gap['unselected_files']:
            p=Path(text);suffix=p.suffix.lower();role='python_source' if suffix=='.py' else 'python_stub' if suffix=='.pyi' else 'extension_binary'
            assert suffix in ('.py','.pyi','.pyd','.dll') and text not in selected
            selected[text]={'bytes':v['rosters'][gap['roster']]['files'][text]['bytes'],'sha256':None,'role':role};owners[text]=gap['roster']
    assert len(selected)==5730 and sum(d['bytes'] for d in selected.values())==102009875
    readerpath=PACKAGE/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py';source=readerpath.read_bytes();pin='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892';assert hashlib.sha256(source).hexdigest()==pin
    reader=ModuleType('passive_finite_missing_roster_reader');reader.__file__=str(readerpath);exec(compile(source,str(readerpath),'exec'),reader.__dict__)
    started=time.monotonic();observed=reader._collect(list(selected),selected);new={p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in observed['files'].items()}
    result={'kind':'PASSIVE_EXACT_MISSING_ROSTER_LEAF_BYTE_CANDIDATE','basis_file_sha256':hashlib.sha256(raw).hexdigest(),'basis_body_sha256':v['sha256'],'reader_sha256':pin,'selection':selected,'roster_owner':owners,'files':new,'actual_hashed_bytes':observed['actual_hashed_bytes'],'seconds':time.monotonic()-started,'new_roots_or_suffixes_added':False,'source_execution_observed':False,'native_runtime_qualified':False,'native_launch_authorized':False,'native_authority_issued':False,'native_PASS':False}
    result['sha256']=digest(result)
    with output.open('x',encoding='utf-8') as f:f.write(canonical(result)+'\n')
    print(canonical({'files':len(new),'bytes':result['actual_hashed_bytes'],'seconds':result['seconds'],'body_sha256':result['sha256'],'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
