"""Exact real CPython typing class aliases; no native/framework operations."""
from pathlib import Path
import builtins,copy,hashlib,json,sys,types,typing,unittest,warnings
HERE=Path(__file__).resolve().parent
ORIGINAL=builtins.__import__
def trap(name,*args,**kwargs):
    if name.split('.',1)[0] in {'ctypes','_ctypes','torch','psutil','native_worker','native_supervisor','native_process','native_nvml'}:raise AssertionError('alias controls forbid native/framework imports')
    return ORIGINAL(name,*args,**kwargs)
builtins.__import__=trap
def load(name):
    p=HERE/name;m=types.ModuleType('CPU_'+p.stem);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
P=load('prelude.py');C=load('contract.py');W=load('worker.py');A=load('audit.py')
class Controls(unittest.TestCase):
    def setUp(self):
        path=str(Path(typing.__file__).resolve());raw=Path(path).read_bytes();self.files={path:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}};self.path=path;self.layout={'logical_base_prefix':str(Path(sys.base_prefix)),'physical_base_prefix':str(Path(sys.base_prefix).resolve())};self.population={'typing':typing,'typing.io':vars(typing)['io'],'typing.re':vars(typing)['re']};self.value={'config':{'source_files':{str(HERE/'prelude.py'):{'bytes':len((HERE/'prelude.py').read_bytes()),'sha256':hashlib.sha256((HERE/'prelude.py').read_bytes()).hexdigest(),'role':'python_source'}}},'runtime':{'files':self.files,'stdlib_layout':self.layout,'cache_policy':{'imports':['ctypes','ctypes.wintypes','_ctypes'],'allowed_bootstrap_modules':{'typing':{'path':self.path,'source_sha256':self.files[self.path]['sha256']}}}},'fixture_only':True}
    def test_actual_alias_objects_emit_distinct_owner_bound_rows_without_warning(self):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('error',DeprecationWarning)
            for name in ('typing.io','typing.re'):
                meta=P.typing_alias_metadata(name,self.population[name],self.population,self.files,self.layout);self.assertTrue(P.validate_typing_alias_row(name,meta,self.files));self.assertFalse(meta['source_execution_history_proven'])
            result=W._measure_bootstrap_modules(self.value,C,tuple(self.population.items()));self.assertEqual(result['unknown_modules'],[]);self.assertEqual([r['classification'] for r in result['modules'] if r['name']!='typing'],['stdlib_class_alias','stdlib_class_alias']);self.assertEqual(seen,[])
        A.replay_bootstrap_modules(result,self.value['runtime'],fixture_only=True)
    def test_spoof_owner_metaclass_and_member_identity_refuse(self):
        owner=types.ModuleType('typing');owner.__dict__.update(vars(typing));bad=dict(self.population);bad['typing']=owner
        with self.assertRaises(ValueError):P.typing_alias_metadata('typing.io',bad['typing.io'],bad,self.files,self.layout)
        class Fake:pass
        with self.assertRaises(ValueError):P.typing_alias_metadata('typing.io',Fake,self.population,self.files,self.layout)
        alias=self.population['typing.io'];members=type.__getattribute__(alias,'__dict__');old=members['IO'];type.__setattr__(alias,'IO',object())
        try:
            with self.assertRaises(ValueError):P.typing_alias_metadata('typing.io',alias,self.population,self.files,self.layout)
        finally:type.__setattr__(alias,'IO',old)
    def test_wrong_selected_owner_bytes_and_unselected_alias_refuse_before_read(self):
        bad=copy.deepcopy(self.files);bad[self.path]['sha256']='0'*64
        with self.assertRaises(ValueError):P.typing_alias_metadata('typing.re',self.population['typing.re'],self.population,bad,self.layout)
        with self.assertRaises(ValueError):P.typing_alias_metadata('typing.unknown',self.population['typing.io'],self.population,self.files,self.layout)
        with self.assertRaises(ValueError):P.typing_alias_metadata('typing.io',self.population['typing.io'],self.population,{},self.layout)
    def test_unknown_originless_remains_unknown_and_audit_owner_rebind_refuses(self):
        unknown=types.ModuleType('unknown_originless');result=W._measure_bootstrap_modules(self.value,C,tuple({**self.population,'unknown_originless':unknown}.items()));self.assertEqual([r['name'] for r in result['unknown_modules']],['unknown_originless'])
        good=W._measure_bootstrap_modules(self.value,C,tuple(self.population.items()));bad=copy.deepcopy(good);row=next(r for r in bad['modules'] if r['name']=='typing.io');row['alias_metadata']['owner_sha256']='0'*64
        with self.assertRaises(ValueError):A.replay_bootstrap_modules(bad,self.value['runtime'],fixture_only=True)
    def test_prelude_collector_records_alias_without_invented_cache_source(self):
        original=sys.modules
        try:
            sys.modules=dict(self.population);rows=P.module_snapshot(self.layout,self.files,{})
        finally:sys.modules=original
        for row in rows:
            if row['name'] in ('typing.io','typing.re'):
                self.assertEqual(row['classification'],'stdlib_class_alias');self.assertIsNone(row['origin']);self.assertIsNone(row['source_sha256']);self.assertIsNone(row['cached']);self.assertIsNone(row['error'])
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));r={'kind':'CPU_EXACT_TYPING_CLASS_ALIAS_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_typing_alias_objects':True,'deprecated_attribute_hook_invoked':False,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['worker.py','prelude.py','audit.py','contract.py','test_typing_alias.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
