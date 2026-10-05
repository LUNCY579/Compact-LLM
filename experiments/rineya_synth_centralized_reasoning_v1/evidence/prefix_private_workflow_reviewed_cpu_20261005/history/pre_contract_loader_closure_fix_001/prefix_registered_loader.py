"""Qualified finite source importer: source compilation, no cached pyc execution.

Installed only after the integrated owned worker repeats full byte qualification.
Every non-builtin/frozen Python or extension module must be in its exact map.
"""
from pathlib import Path
import hashlib,importlib.abc,importlib.machinery,marshal,sys,types
def physical(path,layout):
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
    budget=[0]
    def visit(value,depth=0):
        budget[0]+=1
        if budget[0]>32768 or depth>64:raise ValueError('bounded bootstrap code metadata')
        if type(value) is types.CodeType:return (value.co_code,value.co_argcount,value.co_posonlyargcount,value.co_kwonlyargcount,value.co_flags,value.co_stacksize,value.co_exceptiontable,value.co_names,value.co_varnames,value.co_freevars,value.co_cellvars,value.co_name,value.co_qualname,value.co_firstlineno,value.co_linetable,value.co_filename,tuple(visit(x,depth+1) for x in value.co_consts))
        if type(value) in (type(None),bool,int,float,complex,str,bytes):return value
        if type(value) is tuple:return tuple(visit(x,depth+1) for x in value)
        if type(value) is frozenset:return frozenset(visit(x,depth+1) for x in value)
        if value is Ellipsis:return Ellipsis
        raise ValueError('unsupported bootstrap code constant')
    return visit(code)
class RegisteredLoader(importlib.abc.Loader):
    def __init__(self,actual,path,descriptor,contract,observed):self.actual=actual;self.path=path;self.descriptor=descriptor;self.contract=contract;self.observed=observed
    def create_module(self,spec):
        self.contract.file_descriptor(self.path,self.descriptor);return self.actual.create_module(spec)
    def exec_module(self,module):
        self.contract.file_descriptor(self.path,self.descriptor)
        if Path(self.path).suffix=='.py':
            raw=self.contract.read_bytes(self.path);exec(compile(raw.decode('utf-8'),self.path,'exec'),module.__dict__)
        else:self.actual.exec_module(module)
        self.observed[module.__name__]={'path':self.path,'sha256':self.descriptor['sha256'],'loader_kind':'registered_source' if Path(self.path).suffix=='.py' else 'registered_extension'}
class RegisteredFinder(importlib.abc.MetaPathFinder):
    def __init__(self,files,contract,layout):self.files=files;self.contract=contract;self.layout=layout;self.observed={}
    def find_spec(self,fullname,path=None,target=None):
        # Builtin/frozen modules have interpreter byte identity in the qualified
        # receipt. PathFinder excludes this finder and therefore cannot recurse.
        if importlib.machinery.BuiltinImporter.find_spec(fullname) or importlib.machinery.FrozenImporter.find_spec(fullname):return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is None:raise ModuleNotFoundError('unregistered meta-path fallback refused: '+fullname)
        if spec.origin is None:
            self.contract.require(spec.submodule_search_locations is not None and all(any(Path(directory).is_relative_to(Path(p).parent) for p in self.files) for directory in spec.submodule_search_locations),'registered finite namespace package roots');return spec
        origin=physical(spec.origin,self.layout)
        self.contract.require(origin in self.files and Path(origin).suffix in ('.py','.pyd'),'unregistered source/extension/cache import refused')
        spec.loader=RegisteredLoader(spec.loader,origin,self.files[origin],self.contract,self.observed);return spec
def install(context,contract):
    contract.stage_native_gate();value=contract.context_value(context);files={**value['runtime']['files'],**value['config']['source_files']}
    finder=RegisteredFinder(files,contract,value['runtime']['stdlib_layout'])
    contract.require(all(item in (importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder) for item in sys.meta_path),'unregistered preexisting meta-path finder')
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path is None:continue
        origin=physical(path,value['runtime']['stdlib_layout']);contract.require(origin in files and Path(origin).suffix in ('.py','.pyd'),'cached module absent from exact source/runtime qualification')
        contract.file_descriptor(origin,files[origin])
        cache=getattr(module,'__cached__',None)
        if Path(origin).suffix=='.py' and cache:
            cached=physical(cache,value['runtime']['stdlib_layout']);pins=value['runtime']['cache_policy']['preloaded_module_cache_pins'];contract.require(cached in pins and cached in files and files[cached]==pins[cached] and not Path(origin).is_relative_to(Path(value['runtime']['cache_policy']['Torch_root'])),'exact qualified preloaded stdlib/dependency cache; Torch must be fresh')
            contract.file_descriptor(cached,pins[cached]);raw=contract.read_bytes(cached);contract.require(len(raw)>=16 and raw[:4]==importlib.util.MAGIC_NUMBER,'qualified CPython cache header')
            code=marshal.loads(raw[16:]);contract.require(type(code) is types.CodeType and physical(code.co_filename,value['runtime']['stdlib_layout'])==origin,'qualified cache source filename')
            source=contract.read_bytes(origin);expected=compile(source.decode('utf-8'),code.co_filename,'exec');contract.require(code_fingerprint(code)==code_fingerprint(expected),'cache code matches registered source')
        finder.observed[name]={'path':origin,'sha256':files[origin]['sha256'],'loader_kind':'qualified_preloaded_module','cache_verified':bool(cache)}
    sys.meta_path.insert(0,finder);return finder
def remove(finder):
    if finder in sys.meta_path:sys.meta_path.remove(finder)
