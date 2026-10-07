"""Exact selected tensor source definitions and bounded startup metadata only.

No tensor entry, framework, ctypes, Win32 API or observer operation is invoked
by this source prelude. Native fixture work remains in the original observer.
"""
from pathlib import Path
from types import ModuleType,CodeType,FunctionType
import hashlib,importlib.machinery,importlib.util,json,marshal,os,stat,sys,time,zipimport
ORDER=('prefix_input_owned_worker.py','registration_git_join.py','prefix_registered_loader.py')
def require(value,message):
    if not value:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode('ascii')).hexdigest()
def bounded_bytes(path,maximum=16*1024**2):
    p=Path(path).absolute();require(p.resolve()==p,'exact prelude physical byte path')
    for q in (p,*p.parents):require(not q.is_symlink() and not(getattr(q.lstat(),'st_file_attributes',0)&0x400),'prelude aliases/reparse refused')
    b=p.lstat();require(stat.S_ISREG(b.st_mode) and 0<=b.st_size<=maximum,'bounded regular prelude file')
    with p.open('rb') as f:o=os.fstat(f.fileno());raw=f.read(b.st_size+1);c=os.fstat(f.fileno())
    a=p.lstat();identity=lambda x:(x.st_size,x.st_mtime_ns,x.st_ino,stat.S_IFMT(x.st_mode))
    require(len(raw)==b.st_size and identity(b)==identity(o)==identity(c)==identity(a),'stable bounded opened prelude bytes');return raw
def normalized(path,layout):
    p=Path(path).absolute();logical=Path(layout['logical_base_prefix']);physical=Path(layout['physical_base_prefix'])
    if p.is_relative_to(logical):
        require(logical.resolve()==physical and p.resolve()==physical/p.relative_to(logical),'one selected interpreter alias');return str(physical/p.relative_to(logical))
    require(p.resolve()==p,'no unselected source/cache alias');return str(p)
def fingerprint(code):
    budget=[0,0]
    def visit(x,depth=0):
        budget[0]+=1;require(depth<=64 and budget[0]<=32768,'bounded code metadata nodes')
        if type(x) is CodeType:return visit((x.co_code,x.co_argcount,x.co_posonlyargcount,x.co_kwonlyargcount,x.co_nlocals,x.co_stacksize,x.co_flags,x.co_exceptiontable,x.co_names,x.co_varnames,x.co_freevars,x.co_cellvars,x.co_name,x.co_qualname,x.co_firstlineno,x.co_linetable,x.co_filename,x.co_consts),depth+1)
        if type(x) is tuple:return tuple(visit(v,depth+1) for v in x)
        if type(x) is frozenset:return frozenset(visit(v,depth+1) for v in x)
        if type(x) in (str,bytes):budget[1]+=len(x);require(budget[1]<=32*1024**2,'bounded code metadata bytes')
        if type(x) is int:require(x.bit_length()<=32768,'bounded code integer')
        require(type(x) in (str,bytes,int,float,complex,bool,type(None),type(Ellipsis)),'literal metadata only');return x
    require(type(code) is CodeType,'cached object is code metadata');return visit(code)
def cache_metadata(module,source,layout,selected_files=None,cache_files=None):
    cached=vars(module).get('__cached__');row={'path':cached if type(cached) is str else None,'bytes':None,'sha256':None,'source_equivalence':None,'error':None}
    if cached is None:return row
    try:
        require(type(cached) is str and source.endswith('.py'),'source/cache exact metadata types');candidate=str(Path(cached).absolute());logical=Path(layout['logical_base_prefix']);physical=Path(layout['physical_base_prefix']);candidate=str(physical/Path(candidate).relative_to(logical)) if Path(candidate).is_relative_to(logical) else candidate
        require(type(cache_files) is dict and candidate in cache_files,'unselected cache path recorded without byte read');path=normalized(cached,layout);require(Path(path).parent.name=='__pycache__' and Path(path).name==Path(source).stem+'.cpython-311.pyc' and Path(path).parent.parent==Path(source).parent,'exact selected CPython3.11 source-cache leaf')
        raw=bounded_bytes(path);pin=cache_files[path];require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'selected cache byte identity');row.update(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());require(len(raw)>=16 and raw[:4]==importlib.util.MAGIC_NUMBER and int.from_bytes(raw[4:8],'little') in (0,1,3),'exact bounded cache header')
        code=marshal.loads(raw[16:]);require(type(code) is CodeType and normalized(code.co_filename,layout)==source,'cached code source filename')
        require(type(selected_files) is dict and source in selected_files,'source admitted before cache source-equivalence read');source_raw=bounded_bytes(source);pin=selected_files[source];require(len(source_raw)==pin['bytes'] and hashlib.sha256(source_raw).hexdigest()==pin['sha256'],'selected cache source byte identity');row['source_equivalence']=fingerprint(code)==fingerprint(compile(source_raw,code.co_filename,'exec'))
        require(row['source_equivalence'] is True,'cached code differs from selected source')
    except (OSError,ValueError,EOFError,TypeError,MemoryError) as e:row['error']={'type':type(e).__name__,'message':str(e)[:2048]}
    return row
def module_snapshot(layout,selected_files=None,cache_files=None):
    require(len(sys.modules)<=4096,'bounded current module population');rows=[]
    for name,module in sorted(tuple(sys.modules.items())):
        if module is None:continue
        namespace=vars(module);spec=namespace.get('__spec__');origin=getattr(spec,'origin',None) or namespace.get('__file__');row={'name':name,'spec_origin':getattr(spec,'origin',None),'origin':origin if type(origin) is str else None,'source_sha256':None,'source_bytes':None,'cached':None,'cached_attribute_present':'__cached__' in namespace,'cached_value_kind':'missing' if '__cached__' not in namespace else 'none' if namespace['__cached__'] is None else type(namespace['__cached__']).__name__,'error':None,'current_execution_history_proven':False}
        if origin not in ('built-in','frozen'):
            try:
                require(type(origin) is str,'unselected fileless namespace');candidate=str(Path(origin).absolute());logical=Path(layout['logical_base_prefix']);physical=Path(layout['physical_base_prefix']);candidate=str(physical/Path(candidate).relative_to(logical)) if Path(candidate).is_relative_to(logical) else candidate
                require(type(selected_files) is dict and candidate in selected_files,'unselected module origin recorded without byte read');path=normalized(origin,layout);raw=bounded_bytes(path);pin=selected_files[path];require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'selected original module byte identity');row.update(origin=path,source_sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw));row['cached']=cache_metadata(module,path,layout,selected_files,cache_files)
            except (OSError,ValueError,TypeError) as e:row['error']={'type':type(e).__name__,'message':str(e)[:2048]}
        rows.append(row)
    return rows
def importer_snapshot():
    standard=(importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder)
    require(len(sys.meta_path)==3 and all(a is b for a,b in zip(sys.meta_path,standard)),'original standard meta importer identities')
    cache=[];require(len(sys.path_importer_cache)<=4096,'bounded importer cache')
    for path,item in sorted(sys.path_importer_cache.items()):
        require(type(path) is str and (item is None or type(item) in (importlib.machinery.FileFinder,zipimport.zipimporter)),'only observed standard FileFinder or absent cache')
        cache.append({'path':path,'kind':None if item is None else type(item).__name__,'finder_path':None if item is None else item.path if type(item) is importlib.machinery.FileFinder else item.archive,'method_code_sha256':hashlib.sha256(type(item).find_spec.__code__.co_code).hexdigest() if type(item) is importlib.machinery.FileFinder else None})
    return {'path':list(sys.path),'meta_path':[{'module':x.__module__,'name':x.__name__} for x in sys.meta_path],'path_hooks':[{'module':x.__module__,'name':x.__name__} for x in sys.path_hooks],'importer_cache':cache,'source_execution_history_proven':False}
def source_prelude(plan,contract,value,*,cpu_only):
    require(tuple(plan['compile_order'])==ORDER,'exact frozen prelude compile order');require(cpu_only is value['fixture_only'],'exact fixture provenance')
    source_map=value['config']['source_files'];selected_files={**value['runtime'].get('files',{}),**source_map};cache_files={p:d for p,d in selected_files.items() if d.get('role')=='qualified_bytecode_cache'};modules={};records=[];costs=[];before=importer_snapshot();retained=None
    contract_row=plan['sources']['prefix_tensor_contract.py'];require(source_map.get(contract_row['path'])==contract_row['descriptor'],'transitive worker contract admitted before eager compilation');contract.file_descriptor(contract_row['path'],contract_row['descriptor'])
    for name in ORDER:
        entry=plan['sources'].get(name) if name!='registration_git_join.py' else {'path':next(iter(plan['helper_sources'])),'descriptor':next(iter(plan['helper_sources'].values()))}
        path=entry['path'];d=entry['descriptor'];require(source_map.get(path)==d and Path(path).name==name,'selected actual frozen prelude source descriptor');contract.file_descriptor(path,d);raw=bounded_bytes(path);require(len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'],'exact prelude source bytes before compile')
        started=time.monotonic();module=ModuleType('selected_tensor_prelude_'+Path(name).stem);module.__file__=path
        if name=='prefix_native_pipeline.py':module.__dict__['__owned_contract__']=retained
        exec(compile(raw,path,'exec'),module.__dict__);modules[name]=module
        if name=='prefix_input_owned_worker.py':retained=module.C;require(retained.__file__==plan['sources']['prefix_tensor_contract.py']['path'],'worker retained exact sibling contract')
        if name=='prefix_native_pipeline.py':require(module.C is retained,'pipeline retains original worker contract instance')
        require(retained.stage_native_gate.__code__.co_filename==plan['sources']['prefix_tensor_contract.py']['path'],'closed tensor gate selected source filename')
        try:retained.stage_native_gate()
        except ValueError:pass
        else:raise ValueError('tensor source gate unexpectedly open')
        elapsed=time.monotonic()-started;costs.append({'source':path,'compile_attempted':1,'compile_returned':1,'elapsed_seconds':elapsed});records.append({'name':module.__name__,'file':path,'descriptor':d,'definition_execution_observed':True,'main_executed':False,'contract_instance_retained':name!='prefix_native_pipeline.py' or module.C is retained})
    contract_row=plan['sources']['prefix_tensor_contract.py'];require(source_map.get(contract_row['path'])==contract_row['descriptor'],'transitive worker contract admitted before eager compilation')
    roots=plan['registered_roots'];require(len(roots)==2 and all(type(r) is str and Path(r).is_dir() and Path(r).resolve()==Path(r) for r in roots),'exact physical registered root transition')
    original=sys.path;initial=tuple(original);require(all(r not in original for r in roots),'registered roots absent at initial no-site profile')
    original.extend(roots);transition={'before':list(initial),'after':list(original),'registered_roots':roots,'hook_invoked':False}
    owned={'container':original,'before':initial,'after':list(original)}
    try:
        result={'kind':'OBSERVED_FROZEN_TENSOR_SOURCE_PRELUDE','compiled_helpers':records,'source_costs':costs,'initial_importers':before,'registered_root_transition':transition,'after_prelude_importers':importer_snapshot(),'modules_after_prelude':module_snapshot(value['runtime']['stdlib_layout'],selected_files,cache_files),'observer_main_renamed_to_tensor':False,'tensor_main_executed':False,'Torch_called':False,'native_authority_issued':False}
        return result,{'container':original,'before':initial,'after':list(original)},modules
    except BaseException as primary:
        try:restore_roots(owned)
        except BaseException as cleanup:raise primary from cleanup
        raise
def restore_roots(transition):
    require(sys.path is transition['container'] and sys.path==transition['after'],'original owned root transition unchanged; foreign mutations preserved');sys.path[:]=transition['before']


def validate_record(record,plan,selected_files,cache_files,*,guard_boundary,ordinary_modules=None):
    require(record['kind']=='OBSERVED_FROZEN_TENSOR_SOURCE_PRELUDE' and record['observer_main_renamed_to_tensor'] is record['tensor_main_executed'] is record['Torch_called'] is record['native_authority_issued'] is False,'exact definition-only prelude record')
    require(len(record['compiled_helpers'])==len(ORDER) and len(record['source_costs'])==len(ORDER),'exact selected helper/cost population')
    expected=[]
    for name in ORDER:
        row=plan['sources'][name] if name!='registration_git_join.py' else {'path':next(iter(plan['helper_sources'])),'descriptor':next(iter(plan['helper_sources'].values()))};expected.append(row)
    for name,row,cost,pin in zip(ORDER,record['compiled_helpers'],record['source_costs'],expected):
        require(row['name']=='selected_tensor_prelude_'+Path(name).stem and row['file']==pin['path'] and row['descriptor']==pin['descriptor'] and row['definition_execution_observed'] is True and row['main_executed'] is False,'original exact ordered source compile facts')
        require(cost['source']==pin['path'] and type(cost['compile_attempted']) is type(cost['compile_returned']) is int and cost['compile_attempted']==cost['compile_returned']==1 and type(cost['elapsed_seconds']) in (int,float) and 0<=cost['elapsed_seconds']<=600,'exact finite source compile costs')
    snapshots=['modules_after_prelude','modules_after_ctypes']+(['modules_at_guard_boundary'] if guard_boundary else [])
    for field in snapshots:
        require(type(record.get(field)) is list and 0<len(record[field])<=4096,'original complete prelude module snapshots');names=set()
        for row in record[field]:
            require(type(row['name']) is str and row['name'] and row['name'] not in names and row['error'] is None and row['current_execution_history_proven'] is False,'unique source/cache metadata without hidden error/history claim');names.add(row['name'])
            present=row.get('cached_attribute_present');kind=row.get('cached_value_kind');require(type(present) is bool and kind in ('missing','none','str'),'exact observed cached attribute shape')
            cached=row['cached']
            if not present:require(kind=='missing' and (cached is None or cached['path'] is None),'missing cached attribute not erased/qualified')
            elif kind=='none':require(cached is None or cached['path'] is None,'explicit None cached attribute has no byte identity')
            else:require(kind=='str' and type(cached) is dict and type(cached['path']) is str,'actual cached string observation cannot be erased')
            if row['origin'] in ('built-in','frozen'):require(row['source_sha256'] is row['source_bytes'] is None,'builtin/frozen documentary metadata is not source byte qualification')
            else:
                pin=selected_files.get(row['origin']);require(type(pin) is dict and row['source_sha256']==pin['sha256'] and row['source_bytes']==pin['bytes'],'original observed module source descriptor joins')
                cached=row['cached']
                if cached is not None and cached['path'] is not None:
                    pin=cache_files.get(cached['path']);require(type(pin) is dict and cached['error'] is None and cached['sha256']==pin['sha256'] and cached['bytes']==pin['bytes'] and cached['source_equivalence'] is True,'original observed selected cache bytes/source equivalence')
        require({'sys','__main__','builtins'}<=names,'mandatory current interpreter module population')
    if guard_boundary:
        require(type(ordinary_modules) is list and ordinary_modules,'original ordinary bootstrap population for guard join')
        ordinary={r['name']:r for r in ordinary_modules};after={r['name']:r for r in record['modules_after_ctypes']};guard={r['name']:r for r in record['modules_at_guard_boundary']}
        require(set(after)<=set(guard) and set(ordinary)<=set(guard),'guard-boundary contains original after-ctypes/ordinary modules')
        for name,row in after.items():require(guard[name]['origin']==row['origin'] and guard[name]['source_sha256']==row['source_sha256'],'original module origins/bytes conserved at guard')
        for name,row in ordinary.items():require(guard[name]['origin']==row['origin'] and guard[name]['source_sha256']==row['source_sha256'],'ordinary observer bootstrap origins/bytes at guard')
    transition=record['registered_root_transition'];require(transition['registered_roots']==plan['registered_roots'] and transition['hook_invoked'] is False and transition['after']==transition['before']+plan['registered_roots'],'exact original root transition')
    require(record['initial_importers']['path']==transition['before'] and record['after_prelude_importers']['path']==record['importers_after_ctypes']['path']==transition['after'],'original snapshot root joins')
    if guard_boundary:require(record['importers_at_guard_boundary']['path']==transition['after'],'original guard-boundary root snapshot')
    return True
