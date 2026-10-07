"""Qualified finite source importer: source compilation, no cached pyc execution.

Installed only after the integrated owned worker repeats full byte qualification.
Every non-builtin/frozen Python or extension module must be in its exact map.
"""
from pathlib import Path
import hashlib,importlib.abc,importlib.machinery,importlib.util,marshal,sys,types
_CPU_FILEFINDER_METHOD=importlib.machinery.FileFinder.find_spec
_CPU_FILEFINDER_CODE=_CPU_FILEFINDER_METHOD.__code__
QUALIFIED_LAYOUT={'logical_base_prefix':r'C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none','physical_base_prefix':r'C:\Users\clclc\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none','file_attributes':1040,'reparse_tag':2684354563,'is_symlink':False,'python_version':'3.11.15'}
def physical(path,layout):
    return _physical_body(path,layout,QUALIFIED_LAYOUT)

def _physical_body(path,layout,expected_layout):
    if layout!=expected_layout:raise ValueError('exact reviewed interpreter alias layout')
    path=Path(path).absolute();logical=Path(layout['logical_base_prefix']);base=Path(layout['physical_base_prefix'])
    if path.is_relative_to(logical):
        info=logical.lstat()
        if logical.is_symlink() or getattr(info,'st_file_attributes',0)!=layout['file_attributes'] or getattr(info,'st_reparse_tag',None)!=layout['reparse_tag'] or logical.resolve()!=base:raise ValueError('qualified known interpreter alias changed')
        resolved=base/path.relative_to(logical)
        if path.resolve()!=resolved:raise ValueError('known alias contained mapping changed')
        return str(resolved)
    if path.resolve()!=path:raise ValueError('unregistered source origin alias')
    return str(path)
def code_fingerprint(code):
    # Pure code metadata; no execution, globals, locals, tensors or repr.
    budget=[0,0]
    def visit(value,depth=0):
        budget[0]+=1
        if budget[0]>32768 or depth>64:raise ValueError('bounded bootstrap code metadata')
        if type(value) is types.CodeType:return visit((value.co_code,value.co_argcount,value.co_posonlyargcount,value.co_kwonlyargcount,value.co_nlocals,value.co_flags,value.co_stacksize,value.co_exceptiontable,value.co_names,value.co_varnames,value.co_freevars,value.co_cellvars,value.co_name,value.co_qualname,value.co_firstlineno,value.co_linetable,value.co_filename,value.co_consts),depth+1)
        if type(value) in (str,bytes):
            budget[1]+=len(value)
            if budget[1]>32*1024**2:raise ValueError('bounded bootstrap code metadata bytes')
            return value
        if type(value) is int and value.bit_length()>32768:raise ValueError('bounded bootstrap integer metadata')
        if type(value) in (type(None),bool,int,float,complex):return value
        if type(value) is tuple:return tuple(visit(x,depth+1) for x in value)
        if type(value) is frozenset:return frozenset(visit(x,depth+1) for x in value)
        if value is Ellipsis:return Ellipsis
        raise ValueError('unsupported bootstrap code constant')
    return visit(code)
class RegisteredLoader(importlib.abc.Loader):
    def __init__(self,actual,path,descriptor,contract,observed,budget=None,library_guard=None):self.actual=actual;self.path=path;self.descriptor=descriptor;self.contract=contract;self.observed=observed;self.budget=budget;self.library_guard=library_guard
    def create_module(self,spec):
        try:
            self.contract.file_descriptor(self.path,self.descriptor,budget=self.budget)
            if Path(self.path).suffix=='.py':self._source_bytes();return None
            self.contract.require(type(self.actual) is importlib.machinery.ExtensionFileLoader,'exact registered extension loader before create')
            self.contract.require(self.library_guard is not None,'enforced observed DLL boundary before extension create')
            with self.library_guard.extension(self.path,'create'):
                return self.actual.create_module(spec)
        except BaseException:
            if self.library_guard is not None:self.library_guard.poison('registered loader create refusal')
            raise
    def _source_bytes(self):
        raw=self.contract.read_bytes(self.path,budget=self.budget);self.contract.require(len(raw)==self.descriptor['bytes'] and hashlib.sha256(raw).hexdigest()==self.descriptor['sha256'],'actual compile consumes exact qualified source bytes');return raw
    def exec_module(self,module):
        try:
            self.contract.file_descriptor(self.path,self.descriptor,budget=self.budget)
            if Path(self.path).suffix=='.py':
                raw=self._source_bytes();exec(compile(raw,self.path,'exec'),module.__dict__)
            else:
                self.contract.require(type(self.actual) is importlib.machinery.ExtensionFileLoader,'exact registered extension loader before execution')
                self.contract.require(self.library_guard is not None,'enforced observed DLL boundary before extension execution')
                with self.library_guard.extension(self.path,'exec'):
                    self.actual.exec_module(module)
            if hasattr(self,'CPU_module_registry'):self.CPU_module_registry[module.__name__]=(module,self)
            self.observed[module.__name__]={'path':self.path,'sha256':self.descriptor['sha256'],'loader_kind':'registered_source' if Path(self.path).suffix=='.py' else 'registered_extension','current_module_execution_observed':True,'bootstrap_execution_inferred':False}
        except BaseException:
            if self.library_guard is not None:self.library_guard.poison('registered loader execution refusal')
            raise
class RegisteredFinder(importlib.abc.MetaPathFinder):
    def __init__(self,files,contract,layout,namespace_roots=None,budget=None,resolver=None,library_guard=None):self.files=files;self.contract=contract;self.layout=layout;self.namespace_roots={} if namespace_roots is None else namespace_roots;self.observed={};self.budget=budget;self.resolver=physical if resolver is None else resolver;self.library_guard=library_guard;self.library_cleanup=None;self.library_raw_evidence=None;self.library_cleanup_error=None;self.library_export_error=None
    def framework_import(self):
        self.contract.require(self.library_guard is not None,'enforced observed DLL boundary before framework import')
        return self.library_guard.framework_import()
    def find_spec(self,fullname,path=None,target=None):
        if hasattr(self,'CPU_cache_tracker'):_check_installed_record(self,self.CPU_context,_CPU_capability=self.CPU_capability)
        try:return self._find_spec_body(fullname,path,target)
        except ModuleNotFoundError:
            # An absent optional module has executed no source/native code.
            raise
        except BaseException:
            if self.library_guard is not None:self.library_guard.poison('registered finder validation refusal')
            raise
    def _find_spec_body(self,fullname,path=None,target=None):
        if self.budget is not None:self.budget.check()
        # Builtin/frozen modules have interpreter byte identity in the qualified
        # receipt. PathFinder excludes this finder and therefore cannot recurse.
        if importlib.machinery.BuiltinImporter.find_spec(fullname) or importlib.machinery.FrozenImporter.find_spec(fullname):return None
        tracker=getattr(self,'CPU_cache_tracker',None)
        if tracker is not None:tracker.before(fullname,path)
        try:spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        finally:
            if tracker is not None:tracker.after()
        if spec is None:raise ModuleNotFoundError('unregistered meta-path fallback refused: '+fullname)
        if spec.origin is None:
            roots=[] if spec.submodule_search_locations is None else [self.resolver(directory,self.layout) for directory in spec.submodule_search_locations]
            self.contract.require(fullname in self.namespace_roots and roots==self.namespace_roots[fullname] and roots,'exact qualified finite namespace package roots');return spec
        origin=self.resolver(spec.origin,self.layout)
        self.contract.require(origin in self.files and Path(origin).suffix in ('.py','.pyd'),'unregistered source/extension/cache import refused')
        expected=importlib.machinery.SourceFileLoader if Path(origin).suffix=='.py' else importlib.machinery.ExtensionFileLoader
        self.contract.require(type(spec.loader) is expected and spec.loader.name==fullname and self.resolver(spec.loader.path,self.layout)==origin,'exact registered PathFinder source/extension loader; no fallback/create')
        if spec.submodule_search_locations is not None:
            roots=[self.resolver(directory,self.layout) for directory in spec.submodule_search_locations];self.contract.require(Path(origin).name=='__init__.py' and roots==[str(Path(origin).parent)],'exact regular package source root')
            spec.submodule_search_locations=roots
        self.contract.file_descriptor(origin,self.files[origin],budget=self.budget);spec.origin=origin;spec.cached=None
        spec.loader=RegisteredLoader(spec.loader,origin,self.files[origin],self.contract,self.observed,self.budget,self.library_guard)
        if hasattr(self,'CPU_module_registry'):spec.loader.CPU_module_registry=self.CPU_module_registry
        return spec
def _installation_registry_factory():
    # Source-owned identity records reject caller-created finders and callbacks.
    # This is not a defense against arbitrary mutation of Python process memory.
    records={};authorities={}
    class InstalledAuthority:
        __slots__=()
    def remember(finder,value,fixture_only):
        guard=finder.library_guard
        records[finder]=(type(finder),type(finder).framework_import,guard,
            None if guard is None else type(guard),None if guard is None else type(guard).framework_import,
            finder.contract.canonical(value),fixture_only,
            getattr(finder,'CPU_cache_tracker',None),finder._find_spec_body.__func__,
            None if not hasattr(finder,'CPU_cache_tracker') else (type(finder.CPU_cache_tracker),finder.CPU_cache_tracker.before.__func__,finder.CPU_cache_tracker.after.__func__))
        token=InstalledAuthority();authorities[token]=finder;return token
    def check(finder,value,*,_CPU_capability=None):
        contract=finder.contract
        contract.require(finder in records,'actual source-owned installation record')
        cls,method,guard,guard_cls,guard_method,projection,fixture_only,tracker,resolver,tracker_methods=records[finder]
        contract.require(sum(item is finder for item in sys.meta_path)==1,'one current actual source-owned installed finder')
        contract.require(getattr(finder,'CPU_cache_tracker',None) is tracker and type(finder._find_spec_body) is types.MethodType and finder._find_spec_body.__func__ is resolver,'original installed cache tracker and resolver method')
        if tracker is not None:contract.require(type(tracker) is tracker_methods[0] and type(tracker.before) is types.MethodType and tracker.before.__func__ is tracker_methods[1] and type(tracker.after) is types.MethodType and tracker.after.__func__ is tracker_methods[2],'original installed cache tracker methods')
        contract.require(type(finder) is cls and cls is RegisteredFinder and finder.framework_import.__func__ is method,'original installed finder class and method')
        contract.require(finder.library_guard is guard and (guard is None or (type(guard) is guard_cls and guard.framework_import.__func__ is guard_method)),'original installed guard class and method')
        if _CPU_capability is not None:
            contract._check_CPU_native_context_fixture(_CPU_capability)
            contract.require(fixture_only and value.get('fixture_only') is True and value.get('native_authority_issued') is False,'explicit CPU installed-selector fixture only')
        else:
            contract.require(not fixture_only and value.get('fixture_only',False) is False,'CPU installation record has no native selector authority')
        contract.require(projection==contract.canonical(value),'original installed context projection')
        contract.require(guard is not None or (_CPU_capability is not None and fixture_only),'native installation requires actual library guard')
        if guard is not None:contract.require(guard.active is True and guard.failed is False and guard.closed is False,'active unfailed unclosed installed guard')
        return None if guard is None else finder.framework_import
    def authority(token,value,*,finder=None,_CPU_capability=None):
        if type(token) is not InstalledAuthority or token not in authorities:raise ValueError('actual selected installer opaque authority')
        if finder is not None:finder.contract.require(authorities[token] is finder,'opaque authority belongs to actual main installation')
        return check(authorities[token],value,_CPU_capability=_CPU_capability)
    def forget(finder):
        records.pop(finder,None)
        for token,recorded in list(authorities.items()):
            if recorded is finder:del authorities[token]
    return remember,check,authority,forget
_remember_installation,_check_installed_record,_check_installation_authority,_forget_installation=_installation_registry_factory()

def install(context,contract,*,library_bootstrap_verifier=None,library_event_observer=None):
    contract.stage_native_gate();value=contract.context_value(context);contract.require(value.get('fixture_only',False) is False,'CPU native-context fixture has no native loader authority');return _install_body(value,contract,library_bootstrap_verifier=library_bootstrap_verifier,library_event_observer=library_event_observer)
def _CPU_install_fixture(context,contract,*,_capability,library_supplier=None):
    contract._check_CPU_native_context_fixture(_capability);value=contract.context_value(context);contract.require(value.get('fixture_only') is True and value.get('native_PASS') is False and value.get('native_authority_issued') is False,'explicit CPU loader source fixture only')
    layout=contract.parse(contract.canonical(value['runtime']['stdlib_layout']));contract.require(layout==contract.QUALIFIED_LAYOUT,'CPU resolver exact selected context layout')
    def resolver(path,selected_layout):
        contract._check_CPU_native_context_fixture(_capability)
        return _physical_body(path,selected_layout,layout)
    finder=_install_body(value,contract,resolver=resolver,fixture_only=True,library_supplier=library_supplier)
    finder.fixture_only=True;finder.native_PASS=False;finder.native_authority_issued=False
    return finder

def _interpreter_module_assumption(name,module,contract):
    """Validate importer metadata, not current namespace execution.

    Frozen modules may expose a documentary source filename. Their importer
    identity must be resolved before applying source/cache origin policy.
    Qualified interpreter bytes and fresh-bootstrap assumptions remain external
    prerequisites; this helper does not prove the module's current execution.
    """
    spec=getattr(module,'__spec__',None)
    if spec is None or spec.origin not in ('built-in','frozen'):return False
    expected=importlib.machinery.BuiltinImporter if spec.origin=='built-in' else importlib.machinery.FrozenImporter
    contract.require(type(module) is types.ModuleType and spec.name==name and spec.loader is expected,'exact builtin/frozen module importer identity')
    actual=expected.find_spec(name)
    contract.require(actual is not None and actual.name==name and actual.origin==spec.origin and actual.loader is expected,'builtin/frozen identity independently recognized by interpreter importer')
    return True
def _selected_library_guard(value,contract,budget,*,fixture_only,library_supplier,library_bootstrap_verifier,library_event_observer=None):
    policy=value['runtime'].get('library_search_policy',{}).get('enforced_boundary')
    if policy is None:
        contract.require(fixture_only,'selected enforced DLL boundary required before native finder installation')
        contract.require(library_supplier is None,'CPU library supplier needs selected enforced DLL boundary')
        return None
    source=policy.get('provider_source',{});path=source.get('path');descriptor=source.get('descriptor')
    contract.require(path==str(Path(__file__).absolute().with_name('prefix_library_policy.py')) and path in value['config']['source_files'] and value['config']['source_files'][path]==descriptor,'exact selected shared library provider source before compilation')
    contract.file_descriptor(path,descriptor,budget=budget);raw=contract.read_bytes(path,budget=budget)
    contract.require(len(raw)==descriptor['bytes'] and hashlib.sha256(raw).hexdigest()==descriptor['sha256'],'actual shared library guard source bytes before compilation')
    module=types.ModuleType('prefix_selected_library_policy');module.__file__=path
    exec(compile(raw,path,'exec'),module.__dict__);module.__compiled_source_descriptor__=dict(descriptor)
    module.validate_policy(policy)
    if fixture_only:
        contract.require(library_supplier is not None,'explicit CPU loaded-library/search-scope supplier required')
        return module._CPU_make_guard(policy,contract,library_supplier,_capability=module._CPU_library_capability(),budget=budget)
    contract.require(library_supplier is None,'injected CPU library supplier has no native loader authority')
    return module.make_native_guard(policy,contract,bootstrap_verifier=library_bootstrap_verifier,library_event_observer=library_event_observer,budget=budget)

def _install_body(value,contract,*,resolver=None,fixture_only=False,library_supplier=None,library_bootstrap_verifier=None,library_event_observer=None,_CPU_namespace_policy=None):
    resolver=physical if resolver is None else resolver
    files={**value['runtime']['files'],**value['config']['source_files']}
    cache_policy=value['runtime']['cache_policy'];budget=contract.QualificationBudget();finder=RegisteredFinder(files,contract,value['runtime']['stdlib_layout'],cache_policy['namespace_roots'],budget,resolver)
    contract.require(all(item in (importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder) for item in sys.meta_path),'unregistered preexisting meta-path finder')
    for name,module in list(sys.modules.items()):
        budget.check()
        contract.require(name.split('.',1)[0]!='torch','Torch must be fresh before any cache/source check')
        if _interpreter_module_assumption(name,module,contract):continue
        path=getattr(module,'__file__',None)
        if path is None:
            if fixture_only and _CPU_namespace_policy is not None and name=='pywin32_system32':
                spec=vars(module).get('__spec__');entry=_CPU_namespace_policy
                contract.require(spec is not None and spec.origin is None and spec.has_location is False and list(spec.submodule_search_locations)==list(vars(module).get('__path__',()))==entry['search_locations'],'exact CPU overlay namespace metadata')
                for selected,d in entry['files'].items():contract.file_descriptor(selected,d,budget=budget)
                finder.observed[name]={'loader_kind':'CPU_prospective_namespace_metadata','spec_search_locations':list(spec.submodule_search_locations),'module_search_locations':list(vars(module)['__path__']),'namespace_files_sha256':contract.digest(entry['files']),'current_module_execution_observed':False,'bootstrap_execution_inferred':False}
                continue
            contract.require(False,'fileless preloaded module is not qualified builtin/frozen')
        origin=resolver(path,value['runtime']['stdlib_layout']);contract.require(origin in files and Path(origin).suffix in ('.py','.pyd'),'cached module absent from exact source/runtime qualification')
        contract.require(not Path(origin).is_relative_to(Path(cache_policy['Torch_root'])),'Torch source must be fresh even without cached bytecode')
        assumptions=cache_policy['preloaded_bootstrap_modules'];contract.require(name in assumptions and assumptions[name]['path']==origin and assumptions[name]['source_sha256']==files[origin]['sha256'],'explicit external-qualified fresh-bootstrap source assumptions')
        contract.file_descriptor(origin,files[origin],budget=budget)
        cache=getattr(module,'__cached__',None)
        contract.require((cache is None and assumptions[name]['cache_path'] is None) or (cache is not None and resolver(cache,value['runtime']['stdlib_layout'])==assumptions[name]['cache_path']),'exact explicit bootstrap cache path assumptions')
        if Path(origin).suffix=='.py' and cache:
            cached=resolver(cache,value['runtime']['stdlib_layout']);pins=value['runtime']['cache_policy']['preloaded_module_cache_pins'];contract.require(cached in pins and cached in files and files[cached]==pins[cached] and not Path(origin).is_relative_to(Path(value['runtime']['cache_policy']['Torch_root'])),'exact qualified preloaded stdlib/dependency cache; Torch must be fresh')
            contract.file_descriptor(cached,pins[cached],budget=budget);raw=contract.read_bytes(cached,budget=budget);contract.require(len(raw)==pins[cached]['bytes'] and hashlib.sha256(raw).hexdigest()==pins[cached]['sha256'] and len(raw)>=16 and raw[:4]==importlib.util.MAGIC_NUMBER,'qualified exact CPython cache bytes/header')
            code=marshal.loads(raw[16:]);contract.require(type(code) is types.CodeType and resolver(code.co_filename,value['runtime']['stdlib_layout'])==origin,'qualified cache source filename')
            source=contract.read_bytes(origin,budget=budget);contract.require(len(source)==files[origin]['bytes'] and hashlib.sha256(source).hexdigest()==files[origin]['sha256'],'bootstrap compile uses exact registered source');expected=compile(source,code.co_filename,'exec');contract.require(code_fingerprint(code)==code_fingerprint(expected),'cache code matches registered source')
        finder.observed[name]={'path':origin,'sha256':files[origin]['sha256'],'loader_kind':'externally_qualified_bootstrap_assumption','cache_source_equivalence_verified':bool(cache),'current_module_execution_observed':False,'bootstrap_execution_inferred':False}
    finder.library_guard=_selected_library_guard(value,contract,budget,fixture_only=fixture_only,library_supplier=library_supplier,library_bootstrap_verifier=library_bootstrap_verifier,library_event_observer=library_event_observer)
    finder.selected_context_projection=contract.canonical(value)
    finder.selected_loader_source=globals().get('__compiled_source_descriptor__')
    finder.selected_library_guard=finder.library_guard
    finder.installed_authority=_remember_installation(finder,value,fixture_only)
    try:sys.meta_path.insert(0,finder)
    except BaseException:
        _forget_installation(finder)
        if finder.library_guard is not None:finder.library_guard.close()
        raise
    return finder
def remove(finder):
    _forget_installation(finder)
    if finder in sys.meta_path:sys.meta_path.remove(finder)
    if finder.library_guard is not None:
        try:finder.library_cleanup=finder.library_guard.close()
        except BaseException as error:finder.library_cleanup_error=error
        # close stops/drains the observer before this retained-byte export.
        # Failed close/export never creates completion or restores authority.
        try:finder.library_raw_evidence=finder.library_guard.export_raw_packet(getattr(finder,'library_evidence_identity',None))
        except BaseException as error:finder.library_export_error=error
        if finder.library_cleanup_error is not None:
            if finder.library_export_error is not None:raise finder.library_cleanup_error from finder.library_export_error
            raise finder.library_cleanup_error
        if finder.library_export_error is not None:raise finder.library_export_error


def _CPU_cache_snapshot(cache,contract):
    result={}
    for key,item in cache.items():
        contract.require(type(key) is str and len(key)<=4096,'CPU owned cache path string')
        if item is None:result[key]=(None,None);continue
        contract.require(type(item) is importlib.machinery.FileFinder and item.path==key and type(item).find_spec is _CPU_FILEFINDER_METHOD and code_fingerprint(type(item).find_spec.__code__)==code_fingerprint(_CPU_FILEFINDER_CODE),'exact CPU owned FileFinder method/class')
        data=vars(item);contract.require(set(data)=={'_loaders','path','_path_mtime','_path_cache','_relaxed_path_cache'},'no substituted instance finder callable/state')
        loaders=data['_loaders'];expected=tuple((suffix,cls) for cls,suffixes in ((importlib.machinery.ExtensionFileLoader,importlib.machinery.EXTENSION_SUFFIXES),(importlib.machinery.SourceFileLoader,importlib.machinery.SOURCE_SUFFIXES),(importlib.machinery.SourcelessFileLoader,importlib.machinery.BYTECODE_SUFFIXES)) for suffix in suffixes)
        contract.require(type(loaders) is list and len(loaders)==len(expected) and all(type(row) is tuple and len(row)==2 and type(row[0]) is str and row[0]==wanted[0] and row[1] is wanted[1] for row,wanted in zip(loaders,expected)),'exact flattened CPU owned FileFinder loader details')
        for field in ('_path_cache','_relaxed_path_cache'):contract.require(type(data[field]) is set and len(data[field])<=32768 and all(type(v) is str and len(v)<=4096 for v in data[field]),'bounded CPU directory cache strings')
        contract.require(type(data['_path_mtime']) in (int,float),'CPU cache directory timestamp')
        result[key]=(item,(type(item).find_spec,code_fingerprint(type(item).find_spec.__code__),tuple(loaders),data['path'],data['_path_mtime'],frozenset(data['_path_cache']),frozenset(data['_relaxed_path_cache'])))
    return result

def _CPU_install_bootstrap_fixture(context,contract,overlay,helper,handle,surface,*,_capability,_helper_capability,library_supplier=None):
    """Private finite source integration; no live-provider/native entry."""
    contract._check_CPU_native_context_fixture(_capability);value=contract.context_value(context)
    policy=contract._CPU_consume_startup_overlay(overlay,value,_capability=_capability)
    contract.require(surface.modules is sys.modules and surface.meta_path is sys.meta_path and surface.path_hooks is sys.path_hooks and surface.path is sys.path and surface.path_importer_cache is sys.path_importer_cache,'original CPU interpreter container identities')
    helper.verify_policy_bytes(policy);lease=helper._CPU_quarantine(handle,policy,surface,_capability=_helper_capability);finder=None
    try:
        layout=contract.parse(contract.canonical(value['runtime']['stdlib_layout']));contract.require(layout==contract.QUALIFIED_LAYOUT,'CPU resolver exact selected context layout')
        def resolver(path,selected_layout):
            contract._check_CPU_native_context_fixture(_capability);return _physical_body(path,selected_layout,layout)
        finder=_install_body(value,contract,resolver=resolver,fixture_only=True,library_supplier=library_supplier,_CPU_namespace_policy=policy['namespace'])
        finder.fixture_only=True;finder.native_PASS=False;finder.native_authority_issued=False
        finder.CPU_startup_lease=(helper,lease,_helper_capability,surface,{})
        finder.CPU_module_registry={}
        startup_namespace=surface.modules['pywin32_system32']
        allowed_cache_paths=set(surface.path)
        class Tracker:
            def before(self,fullname,path):
                cache=finder.CPU_startup_lease[4]
                contract.require(_CPU_cache_snapshot(surface.path_importer_cache,contract)==cache,'only previously owned importer cache identities/state before CPU resolution')
                if path is not None:
                    contract.require(type(path) is list and path and len(path)<=64 and all(type(p) is str and len(p)<=4096 for p in path),'exact finite package path argument')
                    parent=fullname.rpartition('.')[0];module=surface.modules.get(parent)
                    contract.require(type(module) is types.ModuleType and module.__name__==parent,'current parent package module identity')
                    spec=vars(module).get('__spec__');contract.require(spec is not None and spec.name==parent and list(spec.submodule_search_locations or ())==path==vars(module).get('__path__'),'current parent spec/module package paths exact')
                    if parent in finder.CPU_module_registry:
                        original,loader=finder.CPU_module_registry[parent];origin=vars(module).get('__file__')
                        contract.require(module is original and vars(module).get('__loader__') is loader and type(loader) is RegisteredLoader and spec.loader is loader and origin==loader.path and spec.origin==origin and Path(origin).name=='__init__.py' and path==[str(Path(origin).parent)] and origin in finder.files and loader.descriptor==finder.files[origin],'actual registered compiled parent package and source identity')
                        contract.file_descriptor(origin,finder.files[origin],budget=finder.budget)
                    else:
                        roots=value['runtime']['cache_policy']['namespace_roots'].get(parent)
                        contract.require(parent=='pywin32_system32' and module is startup_namespace and spec.origin is None and vars(module).get('__file__') is None and roots==path==policy['namespace']['search_locations'],'only exact enrolled startup namespace parent')
                    for directory in path:
                        contract.require(resolver(directory,layout)==directory and Path(directory).is_dir(),'physical registered package cache path');allowed_cache_paths.add(directory)
                    contract.require(_CPU_cache_snapshot(surface.path_importer_cache,contract)==cache,'package source/path callbacks cannot mutate owned cache before resolution')
            def after(self):
                cache=finder.CPU_startup_lease[4]
                current=_CPU_cache_snapshot(surface.path_importer_cache,contract)
                contract.require(all(k in current and current[k]==v for k,v in cache.items()),'CPU resolver conserves earlier owned cache state')
                contract.require(all(k in allowed_cache_paths for k in current),'CPU resolver cache paths in exact quarantined path list')
                cache.clear();cache.update(current)
        finder.CPU_cache_tracker=Tracker();finder.CPU_context=value;finder.CPU_capability=_capability
        _forget_installation(finder);finder.installed_authority=_remember_installation(finder,value,True)
        return finder
    except BaseException as primary:
        if finder is not None:
            try:
                if hasattr(finder,'CPU_startup_lease'):_CPU_remove_bootstrap_fixture(finder)
                else:
                    remove(finder);helper._CPU_restore(lease,_capability=_helper_capability)
            except BaseException as cleanup:raise primary from cleanup
        else:
            try:helper._CPU_restore(lease,_capability=_helper_capability)
            except BaseException as cleanup:raise primary from cleanup
        raise

def _CPU_remove_bootstrap_fixture(finder):
    require=getattr(finder,'contract').require
    require(hasattr(finder,'CPU_startup_lease'),'original CPU integrated startup finder')
    helper,lease,cap,surface,cache=finder.CPU_startup_lease
    primary=None
    try:remove(finder)
    except BaseException as failure:primary=failure
    try:
        require(_CPU_cache_snapshot(surface.path_importer_cache,finder.contract)==cache,'exact own resolver cache identities/state before cleanup; foreign additions retained')
        for key in tuple(cache):del surface.path_importer_cache[key]
        helper._CPU_restore(lease,_capability=cap)
    except BaseException as cleanup:
        if primary is not None:raise primary from cleanup
        raise
    if primary is not None:raise primary
