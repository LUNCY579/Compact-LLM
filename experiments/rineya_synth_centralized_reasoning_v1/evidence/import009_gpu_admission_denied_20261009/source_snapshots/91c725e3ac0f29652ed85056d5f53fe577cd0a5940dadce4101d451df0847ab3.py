"""Finite own-process preload/main proof; historical009 assumptions unchanged."""
from pathlib import Path
from types import ModuleType,CodeType
import hashlib,importlib,importlib.abc,importlib.machinery,importlib.util,marshal,sys

def verify_current_module(name,module,row,contract,files,layout):
    contract.require(type(module) is ModuleType,'exact current preload module type');data=vars(module);spec=data.get('__spec__');origin=data.get('__file__');declared_name=name
    if name=='psutil._psutil_windows':
        # The exact installed extension declares psutil_windows; its import spec
        # and loader retain the full psutil._psutil_windows namespace.
        contract.require(row['path']==str(contract.SITE/'psutil/_psutil_windows.pyd') and row['descriptor']=={'bytes':70656,'role':'extension_binary','sha256':'0035450801bd7d938e9e146c5ec28e619cb5a5f4a18cdc53ac7e9734c7f94f78'},'exact observed psutil extension declared-name binding');declared_name='psutil_windows'
    contract.require(data.get('__name__')==declared_name and spec is not None and spec.name==name and type(origin) is str and type(spec.origin) is str,'exact selected preload module/spec name and origin')
    loader=data.get('__loader__');wanted=importlib.machinery.SourceFileLoader if Path(row['path']).suffix=='.py' else importlib.machinery.ExtensionFileLoader
    contract.require(type(loader) is wanted and spec.loader is loader and loader.name==name and str(contract.qualified_path(loader.path,layout=layout))==row['path'],'actual original standard preload loader identity/name/path')
    locations=spec.submodule_search_locations
    if locations is not None:
        contract.require(type(data.get('__path__')) is list and list(locations)==data['__path__'] and len(locations)==1 and str(contract.qualified_path(locations[0],layout=layout))==str(Path(row['path']).parent),'exact selected regular preload package dual roots')
    else:contract.require('__path__' not in data,'ordinary preload module cannot carry unregistered package path')
    physical=str(contract.qualified_path(origin,layout=layout));spec_origin=str(contract.qualified_path(spec.origin,layout=layout));contract.require(physical==spec_origin==row['path'] and files.get(physical)==row['descriptor'],'current preload source/spec exact descriptor')
    contract.file_descriptor(physical,row['descriptor'],layout=layout);present='__cached__' in data;cache=data.get('__cached__');kind='missing' if not present else 'none' if cache is None else 'str' if type(cache) is str else 'unknown'
    contract.require(present is row['cached_attribute_present'] and kind==row['cached_value_kind'],'exact current preload cache field shape')
    if kind=='str':
        cache_path=str(contract.qualified_path(cache,layout=layout));contract.require(cache_path==row['cache_path'] and files.get(cache_path)==row['cache_descriptor'],'selected current preload cache path/pin');contract.file_descriptor(cache_path,row['cache_descriptor'],layout=layout);cached=contract.read_bytes(cache_path,layout=layout);source=contract.read_bytes(physical,layout=layout);contract.require(len(cached)>=16 and cached[:4]==importlib.util.MAGIC_NUMBER,'current preload CPython cache header');code=marshal.loads(cached[16:]);contract.require(type(code) is CodeType,'preload cache metadata only code');expected=compile(source,code.co_filename,'exec');contract.require(contract.code_fingerprint(code)==contract.code_fingerprint(expected) and str(contract.qualified_path(code.co_filename,layout=layout))==physical,'current selected cache/source complete code equivalence')
    else:contract.require(row['cache_path'] is row['cache_descriptor'] is None,'no invented cache pin')
    contract.require(row['source_execution_history_proven'] is False,'current metadata/source equivalent is not original import history')
    return {'path':physical,'source_sha256':row['descriptor']['sha256'],'cache_path':row['cache_path'],'cache_sha256':None if row['cache_descriptor'] is None else row['cache_descriptor']['sha256'],'cached_attribute_present':present,'cached_value_kind':kind,'current_state_source_verified':True,'source_execution_history_proven':False}

def verify_additional_preloads(value,contract,modules):
    policy=value['runtime']['cache_policy'];extra=policy['additional_preload_sources'];historical=policy['preloaded_bootstrap_modules'];contract.require(type(extra) is dict and not set(extra)&set(historical),'additional preloads never overwrite historical009 names');files={**value['runtime']['files'],**value['config']['source_files']};result={}
    for name,row in extra.items():
        contract.require(type(name) is str and name in modules,'every selected additional preload is actually present');result[name]=verify_current_module(name,modules[name],row,contract,files,value['runtime']['stdlib_layout'])
    return result

class _FinitePreludeFinder(importlib.abc.MetaPathFinder):
    def __init__(self,value,contract):contract.require('pwd' not in sys.modules,'preloaded unselected Windows pwd refused');self.value=value;self.contract=contract;self.extra=value['runtime']['cache_policy']['additional_preload_sources']
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='pwd':
            self.contract.require(importlib.machinery.BuiltinImporter.find_spec(fullname) is None and importlib.machinery.FrozenImporter.find_spec(fullname) is None and importlib.machinery.PathFinder.find_spec(fullname,path,target) is None,'available unselected Windows pwd refused');raise ModuleNotFoundError("No module named "+repr(fullname),name=fullname)
        if importlib.machinery.BuiltinImporter.find_spec(fullname) or importlib.machinery.FrozenImporter.find_spec(fullname):return None
        if fullname not in self.extra:
            self.contract.require(fullname in self.value['runtime']['cache_policy']['preloaded_bootstrap_modules'],'unknown additional prelude import refused');return None
        row=self.extra[fullname];spec=importlib.machinery.PathFinder.find_spec(fullname,path,target);self.contract.require(spec is not None and spec.origin is not None and str(self.contract.qualified_path(spec.origin,layout=self.value['runtime']['stdlib_layout']))==row['path'],'finite standard prelude resolution matches selected source');self.contract.file_descriptor(row['path'],row['descriptor'],layout=self.value['runtime']['stdlib_layout'])
        if row['cache_path'] is not None:self.contract.file_descriptor(row['cache_path'],row['cache_descriptor'],layout=self.value['runtime']['stdlib_layout'])
        return spec

def preload_ownership_dependencies(value,contract):
    contract.stage_native_gate();before=dict(sys.modules);finder=_FinitePreludeFinder(value,contract);sys.meta_path.insert(0,finder)
    try:
        importlib.import_module('psutil')
        result=verify_additional_preloads(value,contract,sys.modules)
        allowed=set(value['runtime']['cache_policy']['preloaded_bootstrap_modules'])|set(value['runtime']['cache_policy']['additional_preload_sources'])|{'_datetime','typing.io','typing.re','__main__'}
        for name,module in sys.modules.items():
            if name in before or name in allowed:continue
            spec=vars(module).get('__spec__') if type(module) is ModuleType else None
            contract.require(spec is not None and spec.origin in ('built-in','frozen'),'unknown new non-builtin module during ownership prelude')
        return result
    finally:
        contract.require(finder in sys.meta_path,'owned prelude finder identity retained');sys.meta_path.remove(finder)

def verify_current_main(namespace,module,frame,descriptor,contract,path):
    path=str(Path(path).absolute());contract.require(type(module) is ModuleType and vars(module) is namespace and namespace.get('__name__')=='__main__' and module.__file__==path and module.__spec__ is None and '__cached__' in namespace and namespace['__cached__'] is None,'own direct import worker main module/file/cache shape')
    contract.require(frame.f_globals is namespace and frame.f_code.co_filename==path,'own current executing worker frame');contract.file_descriptor(path,descriptor);raw=contract.read_bytes(path);root=compile(raw,path,'exec');codes=[]
    def collect(code):
        contract.require(len(codes)<4096,'finite own main code graph');codes.append(code)
        for item in code.co_consts:
            if type(item) is CodeType:collect(item)
    collect(root);contract.require(any(contract.code_fingerprint(frame.f_code)==contract.code_fingerprint(code) for code in codes),'own executing frame matches selected source graph')
    return {'path':path,'descriptor':descriptor,'loader_kind':'direct_import_worker_current_frame_source_equivalence','cached_attribute_present':True,'cached_value_kind':'none','current_frame_source_equivalent':True,'source_execution_history_proven':False,'native_authority_issued':False}
