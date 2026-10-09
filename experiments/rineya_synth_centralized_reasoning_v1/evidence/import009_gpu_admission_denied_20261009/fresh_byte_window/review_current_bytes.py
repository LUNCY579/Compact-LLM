"""One passive source/runtime/cache read window; never imports selected modules."""
from pathlib import Path
from types import ModuleType, CodeType
import ast, hashlib, importlib.util, json, marshal, os, stat, subprocess, sys, time, traceback

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
ROOT=PACKAGE.parents[1]
INPUTS=PACKAGE/'artifacts/import_exact_registration_work'
SELECTION=INPUTS/'ACTUAL_SOURCE_RUNTIME_SELECTION009.json'
SELECTION_PIN='08ea63633960affa384c7ae41cf5141a37497d6ee4ba9284fea95553e8faa3d2'
READER=PACKAGE/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py'
READER_PIN='5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
FP_SOURCE=PACKAGE/'evidence/tensor_gate_dispatch_reviewed_cpu_20261007/frozen_gate_dispatch002/candidate/prefix_registered_loader.py'
FP_PIN='62c25f6cf5f843a08463dbb7c3c573f6a038f036917fc4130da0aab5d4b7e0c5'
EXPECTED_HEAD='048d7a43efe705776d94f888656e58593fb467b7'

def require(value,message):
    if not value: raise ValueError(message)
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def seal(value): return {**value,'sha256':digest(value)}
def selected_json(path,pin=None):
    raw=path.read_bytes();require(len(raw)<=16*1024**2,'bounded selected metadata')
    file_sha=hashlib.sha256(raw).hexdigest();require(pin is None or file_sha==pin,'selected metadata pin')
    value=json.loads(raw);require(value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'selected metadata seal')
    return value,{'path':str(path),'bytes':len(raw),'file_sha256':file_sha,'body_sha256':value['sha256']}
def write_new(path,value):
    with path.open('x',encoding='utf-8') as stream: stream.write(canonical(value)+'\n')

def main():
    started=time.monotonic()
    def check(): require(time.monotonic()-started<=600,'single complete source/runtime qualification600s deadline')
    def git(*args):
        check();return subprocess.run(['git',*args],cwd=ROOT,check=True,capture_output=True,timeout=max(.001,600-(time.monotonic()-started))).stdout
    selection,selection_anchor=selected_json(SELECTION,SELECTION_PIN)
    runtime,runtime_anchor=selected_json(INPUTS/'IMPORT_OBSERVATION_RUNTIME_009.json')
    config,config_anchor=selected_json(INPUTS/'IMPORT_OBSERVATION_CONFIG_009.json')
    registration,registration_anchor=selected_json(INPUTS/'IMPORT_OBSERVATION_REGISTRATION_009.json')
    require(selection['implementation_commit']==EXPECTED_HEAD and config['source_commit']==EXPECTED_HEAD,'exact current selected implementation')
    sources=selection['source_files'];runtime_files=selection['runtime_files']
    require(sources==config['source_files']==registration['source_files'] and runtime_files==runtime['files']==config['runtime_files']==registration['runtime_files'],'exact complete source/runtime joins')
    require(len(sources)==selection['source_count'] and len(runtime_files)==selection['runtime_count']==8725 and len(runtime['rosters'])==18,'derived finite selected populations')
    require(not set(sources)&set(runtime_files),'disjoint source/runtime domains')
    selected={**sources,**runtime_files};require(sum(d['bytes'] for d in selected.values())<=16*1024**3,'one16GiB combined read budget')
    require(selection['source_inventory_sha256']==digest(sources) and selection['runtime_inventory_sha256']==digest(runtime_files),'selected inventory digests')
    require(config['owner_approval'] is None and config['native_launch_authorized'] is False and config['native_authority_issued'] is False,'no grant or launch authority')
    head_before=git('rev-parse','HEAD').decode().strip();dirty_before=git('status','--porcelain=v1').decode();require(head_before==EXPECTED_HEAD and dirty_before=='','unchanged clean current implementation before read')
    raw=READER.read_bytes();require(hashlib.sha256(raw).hexdigest()==READER_PIN,'reviewed passive collector bytes')
    reader=ModuleType('selected_passive_current_reader');reader.__file__=str(READER);exec(compile(raw,str(READER),'exec'),reader.__dict__)
    layout=runtime['stdlib_layout'];alias={k:layout[k] for k in ('logical_base_prefix','physical_base_prefix','file_attributes','reparse_tag','is_symlink')}
    cache=runtime['cache_policy'];pairs={}
    for name,row in cache['preloaded_bootstrap_modules'].items():
        if row['cache_path'] is not None:
            cp=row['cache_path'];require(cp not in pairs,'unique historical cache pair');pairs[cp]={'name':name,'source_path':row['path'],'origin':'historical_observed009_assumption'}
    require(set(pairs)==set(cache['preloaded_module_cache_pins']) and len(pairs)==59,'exact59 historical cache pairs')
    for name,row in cache['additional_preload_sources'].items():
        if row['cache_path'] is not None:
            cp=row['cache_path'];require(cp not in pairs,'additional cache never rewrites historical pairs');pairs[cp]={'name':name,'source_path':row['path'],'origin':'passive_additional_preload_assumption'}
    require(len(pairs)==67,'exact59 plus8 unique passive cache pairs')
    retain=set(pairs)|{r['source_path'] for r in pairs.values()}
    require(retain<=set(selected),'all cache/source pairs byte-selected before reads')
    buffers={};counts={'opens':0,'bytes':0,'opened_metadata_checks':0}
    class RetainedStream:
        def __init__(self,stream,path): self.stream=stream;self.path=path;self.chunks=[]
        def __enter__(self): self.stream.__enter__();return self
        def __exit__(self,*args):
            if self.path in retain: buffers[self.path]=b''.join(self.chunks)
            return self.stream.__exit__(*args)
        def fileno(self): return self.stream.fileno()
        def read(self,n):
            check();raw=self.stream.read(n);counts['bytes']+=len(raw)
            if self.path in retain:self.chunks.append(raw)
            return raw
    class WindowReader(reader.Readers):
        def clock(self): check();return time.monotonic()
        def metadata(self,path): check();return super().metadata(path)
        def open(self,path):
            check();counts['opens']+=1;return RetainedStream(super().open(path),str(path))
        def opened_metadata(self,stream): check();counts['opened_metadata_checks']+=1;return super().opened_metadata(stream)
    readers=WindowReader()
    def capture_rosters():
        result={}
        for name,policy in runtime['rosters'].items():
            check();root=reader._path(policy['root'],readers,alias);require(root.is_dir(),'selected physical roster root')
            require(set(policy)=={'root','suffixes','excluded_directory_names','recursive','files'},'exact original physical roster schema')
            files={}
            for folder,dirs,names in os.walk(root,followlinks=False):
                check();dirs[:]=sorted(d for d in dirs if d not in policy['excluded_directory_names'])
                for child in dirs:
                    m=readers.metadata(Path(folder)/child);require(not m['is_symlink'] and not m['file_attributes']&0x400,'no roster directory reparse')
                for leaf in names:
                    p=Path(folder)/leaf
                    if p.suffix.lower() not in policy['suffixes']:continue
                    m=readers.metadata(p);require(stat.S_ISREG(m['mode']) and not m['is_symlink'] and not m['file_attributes']&0x400 and len(files)<32768,'bounded physical roster leaf')
                    files[str(p)]={'bytes':m['bytes']}
                if not policy['recursive']:dirs[:]=[]
            result[name]={**policy,'files':files}
        require(result==runtime['rosters'],'exact current complete selected18 rosters')
        for row in result.values():require(all(p in runtime_files and runtime_files[p]['bytes']==d['bytes'] for p,d in row['files'].items()),'all roster leaves included in single hash map')
        return result
    before=capture_rosters();alias_before=readers.metadata(Path(alias['logical_base_prefix']))
    print(canonical({'event':'PASSIVE_SINGLE_WINDOW_STARTED','source_count':len(sources),'runtime_count':len(runtime_files),'selected_bytes':sum(d['bytes'] for d in selected.values()),'rosters':len(before),'native_API_called':False}),flush=True)
    observed=reader._collect(sorted(selected),selected,readers,alias_table=alias)
    actual={p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in observed['files'].items()};require(actual==selected,'all source/runtime original byte pins matched in one window')
    # Only the selected pure code-metadata function is compiled. No selected
    # module/extension or marshalled code object is executed.
    fpraw=FP_SOURCE.read_bytes();require(hashlib.sha256(fpraw).hexdigest()==FP_PIN and str(FP_SOURCE) in sources,'externally selected complete code fingerprint source')
    nodes=[n for n in ast.parse(fpraw,str(FP_SOURCE)).body if isinstance(n,ast.FunctionDef) and n.name=='code_fingerprint'];require(len(nodes)==1,'one complete bounded code fingerprint definition')
    import types
    ns={'types':types};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(FP_SOURCE),'exec'),ns)
    equivalent={}
    for cache_path,row in pairs.items():
        check();cache_physical=reader._path(cache_path,readers,alias);source_physical=reader._path(row['source_path'],readers,alias)
        cached=buffers[str(cache_physical)];source=buffers[str(source_physical)]
        require(len(cached)>=16 and cached[:4]==importlib.util.MAGIC_NUMBER,'actual selected CPython3.11 cache magic')
        flags=int.from_bytes(cached[4:8],'little');require(flags in (0,1,3),'finite PEP552 cache header flags')
        code=marshal.loads(cached[16:]);require(type(code) is CodeType,'passive marshalled code metadata only')
        require(reader._path(code.co_filename,readers,alias)==source_physical,'cache code filename exact selected source/known alias')
        expected=compile(source,code.co_filename,'exec');require(ns['code_fingerprint'](code)==ns['code_fingerprint'](expected),'complete passive cache/source code equivalence')
        if flags&1:require(cached[8:16]==importlib.util.source_hash(source),'hash-based cache source header')
        else:require(int.from_bytes(cached[12:16],'little')==(len(source)&0xffffffff),'timestamp cache source length header')
        equivalent[cache_path]={**row,'source_sha256':selected[row['source_path']]['sha256'],'cache_sha256':selected[cache_path]['sha256'],'magic_hex':cached[:4].hex(),'flags':flags,'code_filename':code.co_filename,'complete_code_source_equivalence':True,'cache_executed':False,'source_execution_history_proven':False}
    after=capture_rosters();require(after==before and readers.metadata(Path(alias['logical_base_prefix']))==alias_before,'rosters and exact known alias stable after all reads')
    head_after=git('rev-parse','HEAD').decode().strip();dirty_after=git('status','--porcelain=v1').decode();require(head_after==head_before and dirty_after==dirty_before,'Git source HEAD/dirty state stable across window')
    for path,anchor in ((SELECTION,selection_anchor),(INPUTS/'IMPORT_OBSERVATION_RUNTIME_009.json',runtime_anchor),(INPUTS/'IMPORT_OBSERVATION_CONFIG_009.json',config_anchor),(INPUTS/'IMPORT_OBSERVATION_REGISTRATION_009.json',registration_anchor)):
        require(selected_json(path)[1]==anchor,'selected packet bytes stable after window')
    require(hashlib.sha256(READER.read_bytes()).hexdigest()==READER_PIN and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==script_sha,'collector/script bytes stable')
    check();result=seal({'kind':'FRESH_PASSIVE_IMPORT_SOURCE_RUNTIME_BYTE_REVIEW_V1','implementation_commit':head_before,'source_files':sources,'runtime_files':runtime_files,'observed_files':observed['files'],'source_count':len(sources),'runtime_count':len(runtime_files),'source_inventory_sha256':digest(sources),'runtime_inventory_sha256':digest(runtime_files),'selected_descriptor_map_sha256':digest(selected),'selected_path_list_sha256':observed['selected_path_list_sha256'],'actual_hashed_bytes':observed['actual_hashed_bytes'],'physical_read_counters':counts,'single_window_seconds':time.monotonic()-started,'byte_budget':16*1024**3,'deadline_seconds':600,'reader_source_sha256':READER_PIN,'review_script_sha256':script_sha,'fingerprint_source_sha256':FP_PIN,'metadata_anchors':{'selection':selection_anchor,'runtime':runtime_anchor,'config':config_anchor,'registration':registration_anchor},'rosters_before':before,'rosters_after':after,'rosters_sha256':digest(before),'passive_cache_code_equivalence':equivalent,'historical_cache_pairs':59,'additional_passive_cache_pairs':8,'git_before':{'HEAD':head_before,'dirty_porcelain':dirty_before},'git_after':{'HEAD':head_after,'dirty_porcelain':dirty_after},'known_alias_before':alias_before,'known_alias_after':readers.metadata(Path(alias['logical_base_prefix'])),'whole_selected_byte_review_completed':True,'independent_byte_review_completed':False,'native_runtime_qualified':False,'native_context_minted':False,'native_authority_issued':False,'native_launch_authorized':False,'native_API_called':False,'Torch_psutil_ctypes_imported':False,'current_main_or_bootstrap_execution_observed':False,'resources_or_library_baseline_observed':False,'DLL_cookie_or_provider_observed':False,'source_execution_history_proven':False,'native_PASS':False})
    write_new(HERE/'FRESH_IMPORT_CURRENT_BYTE_REVIEW009.json',result)
    print(canonical({'event':'PASSIVE_SINGLE_WINDOW_COMPLETED','sources':len(sources),'runtime':len(runtime_files),'bytes':result['actual_hashed_bytes'],'seconds':result['single_window_seconds'],'cache_pairs':len(equivalent),'body_sha256':result['sha256'],'file_sha256':hashlib.sha256((HERE/'FRESH_IMPORT_CURRENT_BYTE_REVIEW009.json').read_bytes()).hexdigest(),'native_launch_authorized':False}),flush=True)

if __name__=='__main__':
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:main()
    except BaseException as error:
        failure={'kind':'PASSIVE_CURRENT_BYTE_REVIEW_FAILURE','error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'review_script_sha256':script_sha,'native_authority_issued':False,'native_launch_authorized':False}
        partial=getattr(error,'__inventory_partial__',None)
        if partial is not None:failure['collector_partial']=partial
        path=HERE/'FRESH_IMPORT_CURRENT_BYTE_REVIEW_FAILURE009.json'
        if not path.exists():write_new(path,seal(failure))
        raise
