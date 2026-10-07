"""Authored CPU-only exact startup V2 controls; run with -I -S -B."""
import builtins,copy,hashlib,json,os,sys,tempfile,types,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT/'observer_startup_measurement_v1'
_original_import=builtins.__import__
_FORBIDDEN={'torch','ctypes','_ctypes','pywin32_bootstrap','pywin32_system32','_virtualenv','_distutils_hack'}
def guarded_import(name,*args,**kwargs):
    if name.split('.',1)[0] in _FORBIDDEN or name.startswith('__editable__'):raise AssertionError('CPU task must not import startup/native modules: '+name)
    return _original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
def load(name):
    path=PUBLIC/(name+'.py');module=types.ModuleType('CPU_'+name);module.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__);return module
C=load('contract');W=load('worker');A=load('audit')
def descriptor(path,role):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='observer_cpu_bootstrap_');self.base=Path(self.temp.name);self.site=self.base/'site';self.site.mkdir();self.old_site=C.STARTUP_SITE;C.STARTUP_SITE=self.site
        self.files={};self.modules={}
        for name,leaf in C.STARTUP_MODULE_LEAVES.items():
            p=self.site/leaf;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'# authored inert source\n');d=descriptor(p,'python_source');self.files[str(p)]=d;self.modules[name]={'path':str(p),'source_sha256':d['sha256']}
        self.hooks={}
        for leaf in C.STARTUP_HOOK_LEAVES:
            p=self.site/leaf;p.write_bytes(b'# authored inert startup hook\n');d=descriptor(p,'startup_hook_source');self.files[str(p)]=d;self.hooks[str(p)]=d
        self.ns=self.site/'pywin32_system32';self.ns.mkdir();self.nsfiles={}
        for leaf in C.STARTUP_NAMESPACE_LEAVES:
            p=self.ns/leaf;p.write_bytes(b'authored inert DLL bytes; never loaded\n');d=descriptor(p,'extension_binary');self.files[str(p)]=d;self.nsfiles[str(p)]=d
        self.entry={'search_locations':[str(self.ns)],'files':self.nsfiles,'files_sha256':C.digest(self.nsfiles)}
        self.cache={'mode':'MEASURE_BYTE_VERIFIED_STARTUP_V2','imports':['ctypes','ctypes.wintypes','_ctypes'],'allowed_bootstrap_modules':self.modules,'startup_hook_sources':{'directory':str(self.site),'files':self.hooks},'allowed_namespace_packages':{'pywin32_system32':self.entry}}
        self.runtime={'cache_policy':self.cache,'files':self.files,'stdlib_layout':{'logical_base_prefix':str(self.base/'logical'),'physical_base_prefix':str(self.base/'physical')},'inherited65_map':{},'stdlib779_map':{}}
        self.config={'source_files':{}};self.value={'config':self.config,'runtime':self.runtime,'fixture_only':True}
    def tearDown(self):C.STARTUP_SITE=self.old_site;self.temp.cleanup()
    def namespace(self,**changes):
        module=types.ModuleType('pywin32_system32');module.__spec__=types.SimpleNamespace(origin=None,has_location=False,submodule_search_locations=[str(self.ns)]);module.__path__=[str(self.ns)]
        for k,v in changes.items():setattr(module,k,v)
        return module
    def measure(self,module=None,name='pywin32_system32'):
        return W._measure_bootstrap_modules(self.value,C,((name,module or self.namespace()),))
    def assert_denied_measure(self,module=None,name='pywin32_system32'):
        result=self.measure(module,name);self.assertTrue(result['unknown_modules']);self.assertFalse(result['namespace_execution_proven'])
        with self.assertRaises(ValueError):A.replay_bootstrap_modules(result,self.runtime,fixture_only=True)
    def test_positive_exact_namespace_and_leaf_roles(self):
        self.assertEqual(len(C._startup_admitted_roles(self.config,self.runtime)),14)
        self.assertEqual(C._directory_roster(self.site,self.hooks,suffix='.pth'),self.hooks)
        result=self.measure();self.assertEqual(result['unknown_modules'],[]);self.assertEqual(A.replay_bootstrap_modules(result,self.runtime,fixture_only=True),{'pywin32_system32'})
        self.assertEqual(result['modules'][0]['namespace_files_sha256'],self.entry['files_sha256']);self.assertFalse(result['namespace_execution_proven'])
    def test_extra_pth_or_changed_hook_refused(self):
        extra=self.site/'new.pth';extra.write_bytes(b'# inert')
        with self.assertRaises(ValueError):C._directory_roster(self.site,self.hooks,suffix='.pth')
        extra.unlink();p=Path(next(iter(self.hooks)));p.write_bytes(b'# changed')
        with self.assertRaises(ValueError):C._directory_roster(self.site,self.hooks,suffix='.pth')
    def test_wrong_hook_roster_refused(self):
        self.cache['startup_hook_sources']['files']=dict(self.hooks);self.cache['startup_hook_sources']['files'].pop(next(iter(self.hooks)))
        with self.assertRaises(ValueError):C._startup_admitted_roles(self.config,self.runtime)
    def test_namespace_extra_paths_and_disagreement_refused(self):
        module=self.namespace();module.__path__.append(str(self.base));self.assert_denied_measure(module)
        module=self.namespace();module.__spec__.submodule_search_locations.append(str(self.base));module.__path__.append(str(self.base));self.assert_denied_measure(module)
    def test_namespace_alias_metadata_refused(self):
        module=self.namespace();module.__path__=[str(self.ns/'..'/'pywin32_system32')];module.__spec__.submodule_search_locations=list(module.__path__);self.assert_denied_measure(module)
    def test_extra_namespace_file_directory_or_init_refused(self):
        for leaf,isdir in [('extra.dll',False),('child',True),('__init__.py',False)]:
            p=self.ns/leaf
            if isdir:p.mkdir()
            else:p.write_bytes(b'authored inert')
            self.assert_denied_measure()
            if isdir:p.rmdir()
            else:p.unlink()
    def test_changed_namespace_bytes_refused(self):
        p=Path(next(iter(self.nsfiles)));p.write_bytes(b'changed');self.assert_denied_measure()
    def test_unknown_originless_and_source_origin_namespace_refused(self):
        self.assert_denied_measure(name='different_originless')
        module=self.namespace();module.__file__=str(self.ns/'__init__.py');self.assert_denied_measure(module)
    def test_wrong_module_source_hash_refused(self):
        name='_virtualenv';entry=self.modules[name];p=Path(entry['path']);p.write_bytes(b'# modified source')
        module=types.ModuleType(name);module.__file__=str(p);result=self.measure(module,name)
        self.assertTrue(result['unknown_modules'])
    def test_auditor_rejects_old_null_row_path_roster_or_error(self):
        result=self.measure();row=result['modules'][0]
        for field,value in [('spec_search_locations',[str(self.base)]),('namespace_files',{}),('namespace_files_sha256','0'*64),('error',{'type':'ValueError'})]:
            bad=copy.deepcopy(result);bad['modules'][0][field]=value
            with self.assertRaises(ValueError):A.replay_bootstrap_modules(bad,self.runtime,fixture_only=True)
        old=copy.deepcopy(result);old['modules'][0]={k:row[k] for k in ['name','origin','classification','source_sha256','error']};old['modules'][0]['classification']='source_or_extension'
        with self.assertRaises(ValueError):A.replay_bootstrap_modules(old,self.runtime,fixture_only=True)
    def test_wrong_namespace_registered_roster_refused(self):
        self.entry['files']=dict(self.entry['files']);self.entry['files'].pop(next(iter(self.entry['files'])));self.entry['files_sha256']=C.digest(self.entry['files'])
        with self.assertRaises(ValueError):C._startup_admitted_roles(self.config,self.runtime)
    def test_unchanged_envelope_and_old_v1_roles(self):
        self.assertEqual(C.OPS,W.OPS);self.assertEqual(C.OPS,A.OPS);self.assertEqual(C.RESOURCES['GPU_free_at_admission_GiB'],4.0);self.assertFalse(C.PERMISSION['model']);self.assertFalse(C.PERMISSION['torch_import'])
        self.runtime['cache_policy']={'mode':'MEASURE_BYTE_VERIFIED_STARTUP_V1'};self.assertEqual(C._startup_admitted_roles(self.config,self.runtime),{})
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Controls);result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'kind':'CPU_AUTHORED_OBSERVER_BOOTSTRAP_V2_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_files':{str(PUBLIC/(n+'.py')):hashlib.sha256((PUBLIC/(n+'.py')).read_bytes()).hexdigest() for n in ['contract','worker','audit']},'site_disabled':sys.flags.no_site==1,'native_run':False,'native_authority_issued':False,'namespace_execution_proven':False}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
