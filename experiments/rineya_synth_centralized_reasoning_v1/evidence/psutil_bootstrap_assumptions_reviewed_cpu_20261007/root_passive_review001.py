from pathlib import Path
from types import CodeType
import hashlib,json,marshal,importlib.util
HERE=Path(__file__).resolve().parent
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
p=HERE/'PASSIVE_PSUTIL_BOOTSTRAP_ASSUMPTIONS001.json';raw=p.read_bytes();v=json.loads(raw);assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'})
def read(path,d):
    path=Path(path);m=path.stat();assert path.resolve()==path and m.st_size==d['bytes'] and m.st_size<=16*1024**2
    with path.open('rb') as f:b=f.read(m.st_size+1)
    a=path.stat();assert len(b)==m.st_size and (m.st_size,m.st_mtime_ns,m.st_ino)==(a.st_size,a.st_mtime_ns,a.st_ino) and hashlib.sha256(b).hexdigest()==d['sha256'];return b
def compare(a,b):
    assert type(a) is type(b) is CodeType
    for n in ('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags','co_code','co_names','co_varnames','co_freevars','co_cellvars','co_name','co_qualname','co_firstlineno','co_linetable','co_exceptiontable'):assert getattr(a,n)==getattr(b,n),n
    assert len(a.co_consts)==len(b.co_consts)
    for x,y in zip(a.co_consts,b.co_consts):
        if type(x) is CodeType or type(y) is CodeType:compare(x,y)
        else:assert type(x) is type(y) and x==y
bytes_read=0;cache_count=0
for row in v['source_modules'].values():
    source=read(row['path'],row['descriptor']);bytes_read+=len(source)
    selected=next((path,c) for path,c in v['additional_cache_candidates'].items() if c['source_path']==row['path']);path,c=selected;cache=read(path,c['descriptor']);bytes_read+=len(cache);assert cache[:4]==importlib.util.MAGIC_NUMBER;code=marshal.loads(cache[16:]);assert Path(code.co_filename).resolve()==Path(row['path']);compare(code,compile(source,code.co_filename,'exec'));cache_count+=1
for row in v['extension_modules'].values():bytes_read+=len(read(row['path'],row['descriptor']))
new=v['new_exact_System32_candidate'];bytes_read+=len(read(new['path'],new['descriptor']))
assert len(v['source_modules'])==cache_count==7 and len(v['extension_modules'])==3 and v['source_derived_baseline_count']==len(v['source_derived_baseline_expectation'])==39
assert set(v['explicit_baseline_added_paths'])=={x['path'] for x in v['extension_modules'].values()}|{new['path']}
assert all(v[k] is False for k in ('psutil_source_or_extension_imported','native_API_or_DLL_load_performed','native_bootstrap_qualified','native_authority_issued'))
result={'kind':'ROOT_INDEPENDENT_PASSIVE_PSUTIL_SOURCE_CACHE_BYTES001','source_modules':7,'cache_pairs':7,'extensions':3,'pdh_selected':True,'actual_hashed_bytes':bytes_read,'source_cache_filename_and_code_graph_equal':True,'baseline39_is_falsifiable_not_observed':True,'native_API_or_DLL_load_performed':False,'native_authority_issued':False,'basis_file_sha256':hashlib.sha256(raw).hexdigest()};result['sha256']=digest(result);(HERE/'ROOT_PASSIVE_PSUTIL_REVIEW001.json').write_bytes((json.dumps(result,sort_keys=True,indent=2)+'\n').encode());print(canonical(result))
