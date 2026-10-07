"""Passive selected Torch PE import tables; no native loader execution."""
from pathlib import Path
import ast,hashlib,json,os,stat,struct,sys,time
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def fingerprint(m):return (stat.S_IFMT(m.st_mode),m.st_size,m.st_mtime_ns,m.st_ino)
def main(output):
    inputs=HERE/'LIBRARY_EXPECTATION_INPUTS001.json';raw=inputs.read_bytes();value=json.loads(raw);assert value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'})
    parserpath=PACKAGE/'evidence/observer_startup_dispatch_reviewed_20261006/passive_pe_candidates/passive_pe_startup_candidates.py';parserraw=parserpath.read_bytes();tree=ast.parse(parserraw);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pe_imports');env={'struct':struct,'require':require};exec(compile(ast.Module(body=[node],type_ignores=[]),str(parserpath),'exec'),env)
    files=value['library_files'];initial=set(value['expected_initial_loaded_libraries']);seeds=value['explicit_torch_lib_dll_leaves']+[p for p in value['potential_unpreloaded_extension_leaves'] if Path(p).parent.name=='torch'];assert len(seeds)==38
    byname={}
    for p in files:byname.setdefault(Path(p).name.casefold(),[]).append(p)
    started=time.monotonic();total=0;records={};edges=[]
    for text in seeds:
        p=Path(text);d=files[text];before=p.lstat();assert p.resolve()==p and stat.S_ISREG(before.st_mode) and before.st_size==d['bytes'] and d['bytes']<=8*1024**3
        total+=d['bytes'];assert total<=16*1024**3 and time.monotonic()-started<=600
        chunks=[];h=hashlib.sha256();remaining=d['bytes']
        with p.open('rb') as stream:
            assert fingerprint(os.fstat(stream.fileno()))==fingerprint(before)
            while remaining:
                chunk=stream.read(min(1024**2,remaining));assert chunk and len(chunk)<=remaining;h.update(chunk);chunks.append(chunk);remaining-=len(chunk);assert time.monotonic()-started<=600
            assert stream.read(1)==b'' and fingerprint(os.fstat(stream.fileno()))==fingerprint(before)
        assert fingerprint(p.lstat())==fingerprint(before) and h.hexdigest()==d['sha256'];parsed=env['pe_imports'](b''.join(chunks));records[text]={'descriptor':d,'machine':parsed['machine'],'optional_magic':parsed['optional_magic'],'imports':parsed['imports']}
        for row in parsed['imports']:
            names=byname.get(row['name'].casefold(),[]);kind='API_SET_UNRESOLVED' if row['name'].casefold().startswith(('api-ms-','ext-ms-')) else 'SELECTED_UNIQUE_LITERAL_NAME' if len(names)==1 else 'SELECTED_AMBIGUOUS_LITERAL_NAME' if names else 'NOT_IN_SELECTED_BINARY_MAP'
            edges.append({'from':text,**row,'classification':kind,'selected_candidates':names,'candidate_already_in_observer009_baseline':bool(names) and all(p in initial for p in names)})
    required=sorted({p for row in edges if row['kind']=='import' and row['classification']=='SELECTED_UNIQUE_LITERAL_NAME' for p in row['selected_candidates']});unresolved=[row for row in edges if row['classification']!='SELECTED_UNIQUE_LITERAL_NAME' or row['kind']=='delay_import']
    result={'kind':'PASSIVE_TORCH_WINDOWS_STATIC_BINARY_DEPENDENCY_OBLIGATIONS','input_reference':{'path':str(inputs),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256']},'parser_source':{'path':str(parserpath),'sha256':hashlib.sha256(parserraw).hexdigest()},'selected_pe_records':records,'literal_dependency_edges':edges,'ordinary_import_unique_selected_candidate_paths':required,'new_vs_observer009_candidate_paths':[p for p in required if p not in initial],'unresolved_or_delay_edges':unresolved,'explicit_torch_source_load_dll_paths':value['explicit_torch_lib_dll_leaves'],'Torch_C_extension_source_obligation':[p for p in seeds if Path(p).suffix.lower()=='.pyd'],'bytes_read':total,'seconds':time.monotonic()-started,'actual_loader_resolution_observed':False,'actual_per_boundary_effects_observed':False,'native_runtime_qualified':False,'native_launch_authorized':False,'native_authority_issued':False,'native_PASS':False};result['sha256']=digest(result)
    with output.open('x',encoding='utf-8') as f:f.write(canonical(result)+'\n')
    print(canonical({'pe_files':len(records),'bytes':total,'seconds':result['seconds'],'literal_edges':len(edges),'unique_required_candidates':len(required),'unresolved_or_delay':len(unresolved),'body_sha256':result['sha256'],'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
