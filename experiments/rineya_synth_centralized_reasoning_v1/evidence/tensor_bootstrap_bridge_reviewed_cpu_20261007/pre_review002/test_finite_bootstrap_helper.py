import builtins,copy,hashlib,importlib.machinery,json,sys,tempfile,types,unittest,zipimport
from pathlib import Path
ORIGINAL=builtins.__import__
def trap(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','ctypes','_ctypes','pywin32_bootstrap','pywin32_system32','_virtualenv','_distutils_hack'} or name.startswith('__editable__'):raise AssertionError('forbidden actual startup/native import')
    return ORIGINAL(name,*args,**kwargs)
builtins.__import__=trap
HERE=Path(__file__).resolve().parent
H=types.ModuleType('CPU_helper');H.__file__=str(HERE/'finite_bootstrap_helper.py');exec(compile(Path(H.__file__).read_bytes(),H.__file__,'exec'),H.__dict__)
def desc(path,role):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_tensor_bootstrap_');self.site=Path(self.temp.name)/'site';self.site.mkdir();self.oldsite=H.SITE;H.SITE=self.site;self.files={};self.modules={};self.cap=H._CPU_capability();self.meta=[]
        for name,leaf in H.MODULE_LEAVES.items():
            path=self.site/leaf;path.parent.mkdir(parents=True,exist_ok=True)
            if name.startswith('__editable__'):
                text='class _EditableFinder:\n    @classmethod\n    def find_spec(cls, fullname, path=None, target=None):\n        return None\nclass _EditableNamespaceFinder:\n    @classmethod\n    def _path_hook(cls, path):\n        return cls\nPATH_PLACEHOLDER='+repr(H.PLACEHOLDER)+'\n'
            elif name=='_distutils_hack':text='class DistutilsMetaFinder:\n    def find_spec(self, fullname, path=None, target=None):\n        return None\nDISTUTILS_FINDER=DistutilsMetaFinder()\n'
            elif name=='_virtualenv':text='class _Finder:\n    def find_spec(self, fullname, path=None, target=None):\n        return None\n'
            else:text='# inert authored startup source\n'
            path.write_text(text,encoding='utf-8');self.files[str(path)]=desc(path,'python_source');module=types.ModuleType(name);module.__file__=str(path);exec(compile(text,str(path),'exec'),module.__dict__);self.modules[name]=module
        for name,cls,kind,method in H.FINDER_ROLES:
            module=self.modules[name];self.meta.append(module.DISTUTILS_FINDER if name=='_distutils_hack' else vars(module)[cls] if kind=='class' else vars(module)[cls]())
        self.hooks={}
        for leaf in H.HOOK_LEAVES:
            path=self.site/leaf;path.write_bytes(b'# inert authored hook; never executed\n');self.files[str(path)]=self.hooks[str(path)]=desc(path,'startup_hook_source')
        self.ns=self.site/'pywin32_system32';self.ns.mkdir();nsfiles={}
        for leaf in H.NS_LEAVES:
            path=self.ns/leaf;path.write_bytes(b'CPU authored bytes, never a loaded DLL');self.files[str(path)]=nsfiles[str(path)]=desc(path,'extension_binary')
        self.entry={'search_locations':[str(self.ns)],'files':nsfiles,'files_sha256':H.digest(nsfiles)}
        self.runtime={'files':self.files,'cache_policy':{'mode':'MEASURE_BYTE_VERIFIED_STARTUP_V2','allowed_bootstrap_modules':{name:{'path':str(self.site/leaf),'source_sha256':self.files[str(self.site/leaf)]['sha256']} for name,leaf in H.MODULE_LEAVES.items()},'startup_hook_sources':{'directory':str(self.site),'files':self.hooks},'allowed_namespace_packages':{'pywin32_system32':self.entry}}}
        self.policy=H.project_policy(self.runtime,runtime_file_sha256='a'*64);self.pathhook=self.modules[H.EDITABLE]._EditableNamespaceFinder._path_hook;self.standard=[importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder];self.standardhooks=[zipimport.zipimporter,importlib.machinery.FileFinder.path_hook((importlib.machinery.ExtensionFileLoader,importlib.machinery.EXTENSION_SUFFIXES),(importlib.machinery.SourceFileLoader,importlib.machinery.SOURCE_SUFFIXES),(importlib.machinery.SourcelessFileLoader,importlib.machinery.BYTECODE_SUFFIXES))]
        ns=types.ModuleType('pywin32_system32');ns.__spec__=types.SimpleNamespace(origin=None,has_location=False,submodule_search_locations=[str(self.ns)]);ns.__path__=[str(self.ns)];self.modules['pywin32_system32']=ns
        self.surface=types.SimpleNamespace(modules=dict(self.modules),meta_path=[*self.meta[:2],*self.standard,*self.meta[2:]],path_hooks=[*self.standardhooks,self.pathhook],path=[str(self.site),H.PLACEHOLDER],path_importer_cache={str(self.site):None,H.PLACEHOLDER:self.modules[H.EDITABLE]._EditableNamespaceFinder})
    def tearDown(self):H.SITE=self.oldsite;self.temp.cleanup()
    def enroll(self):return H._CPU_enroll(self.policy,self.modules,self.meta,[self.pathhook],self.standard,self.standardhooks,_capability=self.cap)
    def quarantine(self,handle=None):return H._CPU_quarantine(handle or self.enroll(),self.policy,self.surface,_capability=self.cap)
    def test_projection_and_exact_quarantine_restore(self):
        self.assertEqual(len(self.policy['startup_files']),14);containers=(self.surface.meta_path,self.surface.path_hooks,self.surface.path,self.surface.path_importer_cache);snapshot=(list(containers[0]),list(containers[1]),list(containers[2]),dict(containers[3]));lease=self.quarantine()
        self.assertEqual(self.surface.meta_path,self.standard);self.assertEqual(self.surface.path_hooks,self.standardhooks);self.assertNotIn(H.PLACEHOLDER,self.surface.path);self.assertNotIn(H.PLACEHOLDER,self.surface.path_importer_cache)
        H._CPU_restore(lease,_capability=self.cap)
        for now,before in zip((self.surface.meta_path,self.surface.path_hooks,self.surface.path,self.surface.path_importer_cache),containers):self.assertIs(now,before)
        self.assertEqual((self.surface.meta_path,self.surface.path_hooks,self.surface.path,self.surface.path_importer_cache),snapshot)
    def test_counterfeit_finder_and_rebound_source_code_refused(self):
        self.meta[0]=object()
        with self.assertRaises(ValueError):self.enroll()
        self.meta[0]=self.modules['_distutils_hack'].DISTUTILS_FINDER
        fn=self.modules['_virtualenv']._Finder.find_spec;fn.__code__=fn.__code__.replace(co_names=('counterfeit',))
        with self.assertRaises(ValueError):self.enroll()
    def test_changed_hook_extra_pth_or_source_refused(self):
        extra=self.site/'extra.pth';extra.write_bytes(b'# inert')
        with self.assertRaises(ValueError):self.enroll()
        extra.unlink();p=Path(next(iter(self.hooks)));old=p.read_bytes();p.write_bytes(b'# altered')
        with self.assertRaises(ValueError):self.enroll()
        p.write_bytes(old);source=self.site/H.MODULE_LEAVES['_virtualenv'];source.write_bytes(b'# changed source')
        with self.assertRaises(ValueError):self.enroll()
    def test_unknown_finder_hook_placeholder_or_module_refused_before_mutation(self):
        handle=self.enroll()
        for name,value in [('meta_path',object()),('path_hooks',object()),('path','__editable__.unknown.__path_hook__')]:
            getattr(self.surface,name).append(value);before=list(getattr(self.surface,name))
            with self.assertRaises(ValueError):self.quarantine(handle)
            self.assertEqual(getattr(self.surface,name),before);getattr(self.surface,name).pop()
        self.surface.modules['_virtualenv']=types.ModuleType('_virtualenv')
        with self.assertRaises(ValueError):self.quarantine(handle)
    def test_namespace_path_roster_or_origin_refused(self):
        handle=self.enroll();ns=self.modules['pywin32_system32'];ns.__path__.append(str(self.site))
        with self.assertRaises(ValueError):self.quarantine(handle)
        ns.__path__.pop();extra=self.ns/'extra.dll';extra.write_bytes(b'inert')
        with self.assertRaises(ValueError):self.quarantine(handle)
        extra.unlink();ns.__file__=str(self.ns/'__init__.py')
        with self.assertRaises(ValueError):self.quarantine(handle)
    def test_restore_detects_concurrent_mutation_and_retains_it(self):
        lease=self.quarantine();addition=object();self.surface.meta_path.append(addition)
        with self.assertRaises(ValueError):H._CPU_restore(lease,_capability=self.cap)
        self.assertIs(self.surface.meta_path[-1],addition);self.surface.meta_path.pop();H._CPU_restore(lease,_capability=self.cap)
        with self.assertRaises(ValueError):H._CPU_restore(lease,_capability=self.cap)
    def test_restore_replaced_container_and_cache_mutation_refused(self):
        lease=self.quarantine();original=self.surface.path;self.surface.path=list(original)
        with self.assertRaises(ValueError):H._CPU_restore(lease,_capability=self.cap)
        self.surface.path=original;new=object();self.surface.path_importer_cache['new']=new
        with self.assertRaises(ValueError):H._CPU_restore(lease,_capability=self.cap)
        self.assertIs(self.surface.path_importer_cache['new'],new);del self.surface.path_importer_cache['new'];H._CPU_restore(lease,_capability=self.cap)
    def test_post_enrollment_method_replacement_refused(self):
        handle=self.enroll();module=self.modules['_virtualenv'];fn=module._Finder.find_spec;fn.__code__=fn.__code__.replace(co_names=('changed',))
        with self.assertRaises(ValueError):self.quarantine(handle)
    def test_unknown_cached_finder_and_body_failure_restoration(self):
        handle=self.enroll();self.surface.path_importer_cache[str(self.site)]=self.meta[0]
        with self.assertRaises(ValueError):self.quarantine(handle)
        self.surface.path_importer_cache[str(self.site)]=None;snapshot=list(self.surface.meta_path)
        failure=RuntimeError('authored body failure')
        with self.assertRaises(RuntimeError) as caught:
            with H._CPU_quarantined(handle,self.policy,self.surface,_capability=self.cap):raise failure
        self.assertIs(caught.exception,failure);self.assertEqual(self.surface.meta_path,snapshot)
    def test_wrong_policy_forged_handle_missing_site_data_and_native_refused(self):
        with self.assertRaises(ValueError):self.quarantine(object())
        handle=self.enroll();self.policy['startup_files']=dict(self.policy['startup_files']);self.policy['startup_files'].pop(next(iter(self.policy['startup_files'])))
        with self.assertRaises(ValueError):self.quarantine(handle)
        with self.assertRaises(ValueError):H.native_enroll()
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));receipt={'kind':'CPU_FINITE_TENSOR_BOOTSTRAP_HELPER_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'site_disabled':sys.flags.no_site==1,'helper_sha256':hashlib.sha256(Path(H.__file__).read_bytes()).hexdigest(),'actual_site_hook_executed':False,'native_provider_implemented':False,'native_authority_issued':False,'public_contract_changed':False}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
