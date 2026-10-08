"""Verify one extra source/cache pair without importing copy or native modules."""
from pathlib import Path
from types import ModuleType,CodeType
import ast,hashlib,importlib.util,json,marshal,types
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
ORIGINAL=PACKAGE/'evidence/psutil_bootstrap_assumptions_reviewed_cpu_20261007/PASSIVE_PSUTIL_BOOTSTRAP_ASSUMPTIONS001.json'
ORIGINAL_PIN='4a47c0cf98946ce5f31ace0f358dcf5e39652b2293ea7b97785f51635cebf3ae'
RUNTIME=PACKAGE/'artifacts/import_exact_registration_work/IMPORT_OBSERVATION_RUNTIME_003.json'
READER=PACKAGE/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py'
READER_PIN='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
FP=PACKAGE/'evidence/tensor_gate_dispatch_reviewed_cpu_20261007/frozen_gate_dispatch002/candidate/prefix_registered_loader.py'
FP_PIN='62c25f6cf5f843a08463dbb7c3c573f6a038f036917fc4130da0aab5d4b7e0c5'
def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def bounded(p):
    before=p.stat();require(p.resolve()==p and not p.is_symlink() and before.st_size<=16*1024**2,'bounded physical passive bytes')
    with p.open('rb') as f:raw=f.read(before.st_size+1)
    after=p.stat();require(len(raw)==before.st_size and (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'stable passive bytes');return raw
def descriptor(raw,role):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
def main():
    prior_raw=bounded(ORIGINAL);require(hashlib.sha256(prior_raw).hexdigest()==ORIGINAL_PIN,'preserved original seven-source passive receipt');prior=json.loads(prior_raw);require(prior['sha256']==digest({k:v for k,v in prior.items() if k!='sha256'}),'original passive receipt seal')
    runtime_raw=bounded(RUNTIME);runtime=json.loads(runtime_raw);require(runtime['sha256']==digest({k:v for k,v in runtime.items() if k!='sha256'}),'selected current runtime seal')
    base=Path(runtime['stdlib_layout']['physical_base_prefix']);source=base/'Lib/copy.py';cache=base/'Lib/__pycache__/copy.cpython-311.pyc'
    raw=bounded(source);cached=bounded(cache);d=descriptor(raw,'python_source');cd=descriptor(cached,'qualified_bytecode_cache')
    require(d==runtime['files'][str(source)] and d['bytes']==8681 and d['sha256'].startswith('27dcfc53'),'selected exact original copy source');require(cd['bytes']==11243 and cd['sha256'].startswith('9753f133'),'exact additional copy cache candidate')
    require('copy' not in runtime['cache_policy']['preloaded_bootstrap_modules'] and 'copy' not in prior['source_modules'] and str(cache) not in runtime['files'],'copy is separate additional assumption, no historical retagging')
    imports=[node for node in ast.parse(raw,str(source)).body if isinstance(node,(ast.Import,ast.ImportFrom))];dependency_names={name.name.split('.')[0] for node in imports if isinstance(node,ast.Import) for name in node.names}|{node.module.split('.')[0] for node in imports if isinstance(node,ast.ImportFrom) and node.module};require(dependency_names=={'types','weakref','copyreg'} and dependency_names<=set(runtime['cache_policy']['preloaded_bootstrap_modules']),'copy dependencies already selected historical sources')
    require(len(cached)>=16 and cached[:4]==importlib.util.MAGIC_NUMBER,'copy selected CPython3.11 magic');flags=int.from_bytes(cached[4:8],'little');require(flags in (0,1,3),'copy finite PEP552 flags');code=marshal.loads(cached[16:]);require(type(code) is CodeType,'copy passive cache code metadata only')
    logical=Path(runtime['stdlib_layout']['logical_base_prefix']);filename=Path(code.co_filename);physical=base/filename.relative_to(logical) if filename.is_relative_to(logical) else filename;require(physical==source and filename.resolve()==source,'copy cache code filename exact selected alias/source')
    fpraw=bounded(FP);require(hashlib.sha256(fpraw).hexdigest()==FP_PIN,'selected complete fingerprint source');nodes=[n for n in ast.parse(fpraw,str(FP)).body if isinstance(n,ast.FunctionDef) and n.name=='code_fingerprint'];require(len(nodes)==1,'one complete code fingerprint');ns={'types':types};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(FP),'exec'),ns);require(ns['code_fingerprint'](code)==ns['code_fingerprint'](compile(raw,code.co_filename,'exec')),'copy complete source/cache code equivalence')
    if flags&1:require(cached[8:16]==importlib.util.source_hash(raw),'copy hash-based source header')
    else:require(int.from_bytes(cached[12:16],'little')==len(raw),'copy timestamp-cache source size')
    reader_raw=bounded(READER);require(hashlib.sha256(reader_raw).hexdigest()==READER_PIN,'reviewed passive handle reader');reader=ModuleType('copy_passive_reader');reader.__file__=str(READER);exec(compile(reader_raw,str(READER),'exec'),reader.__dict__);observed=reader._collect([str(source),str(cache)],{str(source):d,str(cache):cd})
    result={k:v for k,v in prior.items() if k!='sha256'};result.update(kind='PASSIVE_IMPORT_BOOTSTRAP_SOURCE_CACHE_PE_ASSUMPTIONS002',schema_version=2,predecessor_reference={'path':str(ORIGINAL),'file_sha256':ORIGINAL_PIN,'body_sha256':prior['sha256']})
    result['source_modules']={**prior['source_modules'],'copy':{'path':str(source),'descriptor':d,'top_level_imports':[ast.unparse(n) for n in imports],'already_observed009':False,'source_execution_history_proven':False}}
    result['additional_cache_candidates']={**prior['additional_cache_candidates'],str(cache):{'descriptor':cd,'code_source_equivalence':True,'source_path':str(source),'code_filename':code.co_filename,'magic_hex':cached[:4].hex(),'flags':flags,'cache_executed':False}}
    result.update(additional_python_module_assumptions=[*prior['additional_python_module_assumptions'],'copy'],cache_candidate_count=8,passive_module_count=11,copy_source_or_cache_executed=False,copy_dependencies_already_historical=list(sorted(dependency_names)),copy_passive_reader_sha256=READER_PIN,copy_fingerprint_source_sha256=FP_PIN,copy_pair_actual_hashed_bytes=observed['actual_hashed_bytes'],native_runtime_map_after_addition_count=8725)
    result['notes']=[*prior['notes'],'Version2 adds copy source/cache only; historical63/59 remain unchanged and baseline39 remains an unobserved falsifiable expectation.'];result['sha256']=digest(result)
    output=HERE/'PASSIVE_IMPORT_BOOTSTRAP_ASSUMPTIONS002.json'
    with output.open('x',encoding='utf-8') as f:f.write(canonical(result)+'\n')
    print(canonical({'path':str(output),'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'body_sha256':result['sha256'],'sources':len(result['source_modules']),'caches':8,'extensions':3,'baseline_count':39,'copy_source':d,'copy_cache':cd,'native_authority_issued':False}))
if __name__=='__main__':main()
