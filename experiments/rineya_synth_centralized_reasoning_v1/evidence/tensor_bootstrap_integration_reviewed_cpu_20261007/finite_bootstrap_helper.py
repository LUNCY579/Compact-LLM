"""Finite tensor-startup CPU helper candidate. Native enrollment stays closed.

No site hook is executed here. Source metadata and CPU identity handles do not
prove the execution history of live startup modules.
"""
from pathlib import Path
import contextlib,hashlib,importlib.machinery,json,os,stat,types,zipimport
SITE=Path(r'C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages')
MODULE_LEAVES={'__editable___hermes_agent_0_17_0_finder':'__editable___hermes_agent_0_17_0_finder.py','__editable___recursive_state_forgetting_0_1_0_finder':'__editable___recursive_state_forgetting_0_1_0_finder.py','__editable___rineya_tgsa_research_0_2_0_finder':'__editable___rineya_tgsa_research_0_2_0_finder.py','_distutils_hack':'_distutils_hack/__init__.py','_virtualenv':'_virtualenv.py','pywin32_bootstrap':'win32/lib/pywin32_bootstrap.py'}
HOOK_LEAVES=('__editable__.hermes_agent-0.17.0.pth','__editable__.recursive_state_forgetting-0.1.0.pth','__editable__.rineya_tgsa_research-0.2.0.pth','_virtualenv.pth','distutils-precedence.pth','pywin32.pth')
NS_LEAVES=('pythoncom311.dll','pywintypes311.dll')
FINDER_ROLES=(('_distutils_hack','DistutilsMetaFinder','instance','find_spec'),('_virtualenv','_Finder','instance','find_spec'),('__editable___hermes_agent_0_17_0_finder','_EditableFinder','class','find_spec'),('__editable___recursive_state_forgetting_0_1_0_finder','_EditableFinder','class','find_spec'),('__editable___rineya_tgsa_research_0_2_0_finder','_EditableFinder','class','find_spec'))
EDITABLE='__editable___hermes_agent_0_17_0_finder'
PLACEHOLDER='__editable__.hermes_agent-0.17.0.finder.__path_hook__'
def require(value,message):
    if not value:raise ValueError(message)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode('ascii')).hexdigest()
def physical(path):
    p=Path(path).absolute();require(p.resolve()==p,'physical startup path')
    for x in (p,*p.parents):require(not x.is_symlink() and not(getattr(x.lstat(),'st_file_attributes',0)&0x400),'startup alias/reparse refused')
    return p
def checked_bytes(path,d):
    require(type(d) is dict and set(d)=={'bytes','sha256','role'} and type(d['bytes']) is int and 0<=d['bytes']<=2*1024**2,'finite selected startup descriptor')
    p=physical(path);b=p.lstat();require(stat.S_ISREG(b.st_mode) and b.st_size==d['bytes'],'exact startup size')
    with p.open('rb') as f:o=os.fstat(f.fileno());raw=f.read(d['bytes']+1);c=os.fstat(f.fileno())
    a=p.lstat();identity=lambda s:(s.st_size,s.st_mtime_ns,s.st_ino,stat.S_IFMT(s.st_mode))
    require(identity(b)==identity(o)==identity(c)==identity(a) and len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'],'stable opened startup byte identity')
    return raw
def roster(root,files,*,suffix=None):
    root=physical(root);require(root.is_dir() and type(files) is dict and 0<len(files)<=64,'finite startup roster')
    def observe():
        children=tuple(root.iterdir());require(len(children)<=32768,'bounded startup directory')
        chosen={}
        for p in children:
            if suffix is not None and p.suffix.lower()!=suffix:continue
            require(p.parent==root and stat.S_ISREG(p.lstat().st_mode),'regular immediate startup leaves');physical(p);chosen[str(p)]=(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino)
        require(set(chosen)==set(files),'complete startup roster');return chosen
    before=observe()
    for path,d in files.items():require(Path(path).parent==root,'immediate roster descriptor');checked_bytes(path,d)
    require(observe()==before,'stable startup roster');return files
def project_policy(runtime,*,runtime_file_sha256):
    """Project observer bytes prospectively; no bootstrap or native authority."""
    require(type(runtime_file_sha256) is str and len(runtime_file_sha256)==64,'runtime file anchor')
    cache=runtime['cache_policy'];require(cache['mode']=='MEASURE_BYTE_VERIFIED_STARTUP_V2','explicit observed V2 policy')
    files=runtime['files'];modules={};selected={}
    for name,leaf in MODULE_LEAVES.items():
        path=str(SITE/leaf);d=files.get(path);require(type(d) is dict and d['role']=='python_source' and cache['allowed_bootstrap_modules'].get(name)=={'path':path,'source_sha256':d['sha256']},'exact startup source identity')
        modules[name]={'path':path,'source_sha256':d['sha256']};selected[path]=d
    hooks=cache['startup_hook_sources'];require(hooks['directory']==str(SITE) and set(hooks['files'])=={str(SITE/p) for p in HOOK_LEAVES},'exact six startup hook leaves')
    for path,d in hooks['files'].items():require(files.get(path)==d and d['role']=='startup_hook_source','hook runtime join');selected[path]=d
    ns=cache['allowed_namespace_packages'];require(set(ns)=={'pywin32_system32'},'exact namespace singleton');entry=ns['pywin32_system32'];root=SITE/'pywin32_system32'
    require(entry['search_locations']==[str(root)] and set(entry['files'])=={str(root/p) for p in NS_LEAVES} and entry['files_sha256']==digest(entry['files']),'exact namespace two-file identity')
    for path,d in entry['files'].items():require(files.get(path)==d and d['role']=='extension_binary','namespace runtime join');selected[path]=d
    require(len(selected)==14,'exact additive fourteen leaves')
    return {'kind':'CPU_TENSOR_FINITE_STARTUP_PROJECTION_V1','observer_runtime_file_sha256':runtime_file_sha256,'startup_files':selected,'modules':modules,'hooks':hooks,'namespace':entry,'finder_roles':[list(x) for x in FINDER_ROLES],'namespace_execution_proven':False,'native_authority_issued':False,'native_entry_implemented':False}
def verify_policy_bytes(policy):
    for path,d in policy['startup_files'].items():checked_bytes(path,d)
    roster(policy['hooks']['directory'],policy['hooks']['files'],suffix='.pth')
    roster(policy['namespace']['search_locations'][0],policy['namespace']['files'])
def code_fingerprint(code):
    require(type(code) is types.CodeType,'exact startup callable code')
    budget=[0,0]
    def visit(x,depth=0):
        budget[0]+=1;require(depth<=64 and budget[0]<=32768,'bounded startup code nesting/nodes')
        if type(x) is types.CodeType:return visit((x.co_code,x.co_argcount,x.co_posonlyargcount,x.co_kwonlyargcount,x.co_nlocals,x.co_stacksize,x.co_flags,x.co_exceptiontable,x.co_names,x.co_varnames,x.co_freevars,x.co_cellvars,x.co_name,x.co_qualname,x.co_consts),depth+1)
        if type(x) is tuple:return tuple(visit(v,depth+1) for v in x)
        if type(x) in (str,bytes):budget[1]+=len(x);require(budget[1]<=32*1024**2,'bounded startup code constant bytes')
        require(type(x) in (str,bytes,int,float,complex,type(None),bool),'literal startup code metadata')
        if type(x) is int:require(x.bit_length()<=4096,'bounded startup integer constant')
        return x
    return visit(code)
def expected_method(raw,path,class_name,method):
    top=compile(raw,path,'exec');cls=next((x for x in top.co_consts if type(x) is types.CodeType and x.co_name==class_name),None)
    require(cls is not None,'source-defined startup class');fn=next((x for x in cls.co_consts if type(x) is types.CodeType and x.co_name==method),None);require(fn is not None,'source-defined startup method');return fn

def _identity_registry():
    capabilities=set();records={};leases={}
    class Capability:__slots__=()
    class Handle:__slots__=()
    class Lease:__slots__=()
    def capability():x=Capability();capabilities.add(x);return x
    def check(cap):require(type(cap) is Capability and cap in capabilities,'private CPU-only startup capability')
    def enroll(policy,modules,meta_roles,path_hooks,standard_meta,standard_hooks,*,_capability):
        check(_capability);verify_policy_bytes(policy)
        require(type(modules) is dict and type(meta_roles) is list and len(meta_roles)==5 and len(standard_meta)==3 and type(path_hooks) is list and len(path_hooks)==1 and len(standard_hooks)==2,'complete authored startup identity fixture')
        standard=(importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder)
        require(all(a is b for a,b in zip(standard_meta,standard)),'exact interpreter standard meta-path identities')
        require(standard_hooks[0] is zipimport.zipimporter and type(standard_hooks[1]) is types.FunctionType and standard_hooks[1].__globals__ is importlib.machinery.FileFinder.path_hook.__globals__ and standard_hooks[1].__qualname__=='FileFinder.path_hook.<locals>.path_hook_for_FileFinder','exact standard interpreter path hook kinds')
        cells=standard_hooks[1].__closure__;require(type(cells) is tuple and len(cells)==2 and cells[0].cell_contents is importlib.machinery.FileFinder and type(cells[1].cell_contents) is tuple,'source-owned interpreter FileFinder hook closure')
        expected_details=((importlib.machinery.ExtensionFileLoader,importlib.machinery.EXTENSION_SUFFIXES),(importlib.machinery.SourceFileLoader,importlib.machinery.SOURCE_SUFFIXES),(importlib.machinery.SourcelessFileLoader,importlib.machinery.BYTECODE_SUFFIXES))
        details=cells[1].cell_contents;require(len(details)==3 and all(type(row) is tuple and len(row)==2 and row[0] is expected[0] and row[1]==expected[1] for row,expected in zip(details,expected_details)),'exact interpreter FileFinder loader detail identities')
        expected_hook=importlib.machinery.FileFinder.path_hook(*expected_details);require(code_fingerprint(standard_hooks[1].__code__)==code_fingerprint(expected_hook.__code__),'exact interpreter standard path-hook code metadata')
        for item,role in zip(meta_roles,FINDER_ROLES):
            name,class_name,kind,method=role;module=modules.get(name);require(type(module) is types.ModuleType and module.__name__==name,'exact selected startup module object')
            path=policy['modules'][name]['path'];require(module.__file__==path,'startup module source origin');cls=vars(module).get(class_name);require(type(cls) is type and cls.__module__==name,'exact module-owned startup class')
            require(item is cls if kind=='class' else type(item) is cls,'source module/class finder identity')
            if name=='_distutils_hack':require(vars(module).get('DISTUTILS_FINDER') is item,'distutils singleton finder identity')
            fn=vars(cls).get(method);fn=fn.__func__ if type(fn) is classmethod else fn;require(type(fn) is types.FunctionType and fn.__globals__ is vars(module) and fn.__module__==name,'exact source-owned startup method globals')
            expected=expected_method(checked_bytes(path,policy['startup_files'][path]),path,class_name,method);require(code_fingerprint(fn.__code__)==code_fingerprint(expected) and fn.__code__.co_filename==path,'selected startup method code metadata; execution unproven')
        module=modules[EDITABLE];cls=vars(module).get('_EditableNamespaceFinder');require(type(cls) is type and cls.__module__==EDITABLE,'exact editable namespace class');fn=vars(cls).get('_path_hook');require(type(fn) is classmethod and fn.__func__.__globals__ is vars(module),'exact editable class hook globals')
        hook=path_hooks[0];require(type(hook) is types.MethodType and hook.__self__ is cls and hook.__func__ is fn.__func__,'exact editable path hook identity')
        path=policy['modules'][EDITABLE]['path'];require(code_fingerprint(fn.__func__.__code__)==code_fingerprint(expected_method(checked_bytes(path,policy['startup_files'][path]),path,'_EditableNamespaceFinder','_path_hook')),'editable hook source code metadata')
        require(vars(module).get('PATH_PLACEHOLDER')==PLACEHOLDER,'exact editable placeholder')
        handle=Handle();records[handle]=(digest(policy),dict(modules),tuple(meta_roles),tuple(path_hooks),tuple(standard_meta),tuple(standard_hooks));return handle
    def verify_namespace(policy,module):
        require(type(module) is types.ModuleType and module.__name__=='pywin32_system32','exact authored namespace module');ns=vars(module);spec=ns.get('__spec__');require(spec is not None and spec.origin is None and spec.has_location is False and ns.get('__file__') is None,'originless namespace metadata')
        require(list(spec.submodule_search_locations)==list(ns.get('__path__',()))==policy['namespace']['search_locations'],'exact spec/module namespace paths');roster(policy['namespace']['search_locations'][0],policy['namespace']['files'])
    def quarantine(handle,policy,surface,*,_capability):
        check(_capability);require(type(handle) is Handle and handle in records,'original source-reviewed CPU handle');pin,modules,custom,hooks,standard,standard_hooks=records[handle];require(digest(policy)==pin,'same exact startup policy');verify_policy_bytes(policy)
        # Reconsume exact source/code identities; an enrollment handle is not a
        # license for later method/global replacement.
        refreshed=enroll(policy,modules,list(custom),list(hooks),standard,standard_hooks,_capability=_capability);del records[refreshed]
        require(type(surface) is types.SimpleNamespace and type(surface.meta_path) is list and type(surface.path_hooks) is list and type(surface.path) is list and type(surface.path_importer_cache) is dict and type(surface.modules) is dict,'explicit CPU in-memory startup surface')
        require(all(surface.modules.get(n) is m for n,m in modules.items()),'original startup module identities remain')
        require(len(surface.meta_path)==8 and all(a is b for a,b in zip(surface.meta_path,(*custom[:2],*standard,*custom[2:]))),'exact ordered standard/custom finder population')
        require(len(surface.path_hooks)==3 and all(a is b for a,b in zip(surface.path_hooks,(*standard_hooks,*hooks))),'exact ordered standard/editable path hook population')
        require(len(surface.path)<=256 and all(type(p) is str and 0<len(p)<=4096 for p in surface.path),'exact bounded startup path strings before comparison')
        require(surface.path.count(PLACEHOLDER)==1 and all(not(type(p) is str and p.startswith('__editable__') and p!=PLACEHOLDER) for p in surface.path),'exact singleton editable placeholder')
        require(len(set(surface.path))==len(surface.path) and all(p==PLACEHOLDER or physical(p).is_dir() for p in surface.path),'finite distinct physical CPU path directories')
        require(len(surface.path_importer_cache)<=256,'finite current importer cache population')
        for path,cached in surface.path_importer_cache.items():
            require(type(path) is str and path in surface.path,'importer cache key in current selected path list')
            if path==PLACEHOLDER:require(cached is vars(modules[EDITABLE])['_EditableNamespaceFinder'],'exact placeholder cached namespace finder')
            else:require(cached is None or (type(cached) is importlib.machinery.FileFinder and cached.path==path),'only standard exact-path cached finder; no foreign cache objects')
        verify_namespace(policy,surface.modules.get('pywin32_system32'))
        # A FileFinder's mutable cache/loader state is not source authentication.
        # No old importer-cache object remains callable during quarantine.
        before=(surface.meta_path,surface.path_hooks,surface.path,surface.path_importer_cache);snap=(tuple(before[0]),tuple(before[1]),tuple(before[2]),dict(before[3]));replacement=(list(standard),list(standard_hooks),[p for p in before[2] if p!=PLACEHOLDER],{})
        surface.meta_path[:]=replacement[0];surface.path_hooks[:]=replacement[1];surface.path[:]=replacement[2];surface.path_importer_cache.clear();surface.path_importer_cache.update(replacement[3])
        lease=Lease();leases[lease]=(surface,before,snap,replacement);return lease
    def restore(lease,*,_capability):
        check(_capability);require(type(lease) is Lease and lease in leases,'original unconsumed CPU startup lease');surface,before,snap,expected=leases[lease]
        now=(surface.meta_path,surface.path_hooks,surface.path,surface.path_importer_cache);require(all(a is b for a,b in zip(now,before)),'original startup containers remain')
        require(len(now[0])==len(expected[0]) and all(a is b for a,b in zip(now[0],expected[0])) and len(now[1])==len(expected[1]) and all(a is b for a,b in zip(now[1],expected[1])) and tuple(now[2])==tuple(expected[2]) and set(now[3])==set(expected[3]) and all(now[3][k] is expected[3][k] for k in now[3]),'quarantined startup state unchanged; concurrent mutation not clobbered')
        now[0][:]=snap[0];now[1][:]=snap[1];now[2][:]=snap[2];now[3].clear();now[3].update(snap[3]);del leases[lease]
    return capability,enroll,quarantine,restore
_CPU_capability,_CPU_enroll,_CPU_quarantine,_CPU_restore=_identity_registry()
@contextlib.contextmanager
def _CPU_quarantined(handle,policy,surface,*,_capability):
    lease=_CPU_quarantine(handle,policy,surface,_capability=_capability)
    try:yield lease
    except BaseException as primary:
        try:_CPU_restore(lease,_capability=_capability)
        except BaseException as cleanup:raise primary from cleanup
        raise
    else:_CPU_restore(lease,_capability=_capability)
def native_enroll(*args,**kwargs):raise ValueError('native startup provider qualification is not implemented by CPU candidate')
