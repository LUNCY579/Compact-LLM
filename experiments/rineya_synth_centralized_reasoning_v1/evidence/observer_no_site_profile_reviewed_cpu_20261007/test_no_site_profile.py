"""No-site profile and source-derived original lifecycle CPU controls."""
from pathlib import Path
import ast,builtins,copy,hashlib,importlib.machinery,json,sys,tempfile,types,unittest
HERE=Path(__file__).resolve().parent
ORIGINAL_IMPORT=builtins.__import__
def trap(name,*args,**kwargs):
    if name.split('.',1)[0] in {'ctypes','_ctypes','torch','psutil','native_supervisor','native_process','native_nvml','native_worker','pywin32_bootstrap','pywin32_system32'} or name.startswith('__editable__'):raise AssertionError('CPU no-site controls forbid native/site imports')
    return ORIGINAL_IMPORT(name,*args,**kwargs)
builtins.__import__=trap
def load(name):
    p=HERE/(name+'.py');m=types.ModuleType('CPU_'+name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
C=load('contract');W=load('worker');S=load('supervisor');A=load('audit');D=load('launch_driver')
PACKAGE=C.PACKAGE
oldtest=PACKAGE/'evidence/observer_startup_measurement_reviewed_cpu_20261006/test_supervisor.py';tree=ast.parse(oldtest.read_bytes());host_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Host');ns={'Path':Path};exec(compile(ast.Module(body=[host_node],type_ignores=[]),str(oldtest),'exec'),ns);Host=ns['Host']
class Controls(unittest.TestCase):
    def setUp(self):
        self.profile=C.expected_no_site_profile();self.cache={'mode':C.NO_SITE_MODE,'allowed_bootstrap_modules':{},'imports':['ctypes','ctypes.wintypes','_ctypes'],'interpreter_profile':self.profile};self.value={'runtime':{'cache_policy':self.cache},'fixture_only':True}
    def interpreter(self):
        return types.SimpleNamespace(**{k:self.profile[k] for k in ['executable','prefix','exec_prefix','base_prefix','base_exec_prefix']},flags=types.SimpleNamespace(**self.profile['flags']),path=list(self.profile['path']),meta_path=list(sys.meta_path),path_hooks=list(sys.path_hooks))
    def test_current_actual_metadata_matches_known_profile_without_hooks(self):
        observed=W._measure_no_site_profile(C,self.value,sys);self.assertEqual(observed,self.profile);self.assertFalse(observed['source_execution_history_proven']);self.assertEqual(C.OPS,W.OPS);self.assertEqual(C.OPS,A.OPS);self.assertEqual(C.PERMISSION['allocations'],0);self.assertFalse(C.PERMISSION['torch_import']);self.assertEqual(C.RESOURCES['GPU_free_at_admission_GiB'],4.0)
    def test_flags_prefix_path_and_custom_importers_refuse(self):
        for field,value in [('no_site',0),('isolated',True),('ignore_environment',0)]:
            actor=self.interpreter();setattr(actor.flags,field,value)
            with self.assertRaises(ValueError):W._measure_no_site_profile(C,self.value,actor)
        actor=self.interpreter();actor.prefix='wrong'
        with self.assertRaises(ValueError):W._measure_no_site_profile(C,self.value,actor)
        actor=self.interpreter();actor.path.append(str(PACKAGE))
        with self.assertRaises(ValueError):W._measure_no_site_profile(C,self.value,actor)
        actor=self.interpreter();actor.meta_path.append(object())
        with self.assertRaises(ValueError):W._measure_no_site_profile(C,self.value,actor)
        actor=self.interpreter();actor.path_hooks[1]=lambda path:None
        with self.assertRaises(ValueError):W._measure_no_site_profile(C,self.value,actor)
    def run_supervisor(self,mode):
        temp=tempfile.TemporaryDirectory(prefix='CPU_no_site_supervisor_');self.addCleanup(temp.cleanup);host=Host(temp.name);output=Path(temp.name)/'output';output.mkdir()
        value={'config_path':str(Path(temp.name)/'config.json'),'expected_head':'a'*40,'config':{'kind':'REGISTERED_OBSERVER_STARTUP_MEASUREMENT_V1','worker_path':str(PACKAGE/S.WORKER_RELATIVE),'resources':dict(C.RESOURCES),'source_files':{}},'runtime':{'files':{},'cache_policy':{'mode':mode,'interpreter_profile':self.profile}},'registration':{'profile_sha256':'b'*64},'anchors':{'expected_config_file':'c'*64},'fixture_only':True,'native_PASS':False,'native_authority_issued':False}
        context=C._CPU_context_value_fixture(value,_capability=C._CPU_capability());result=S._CPU_supervise_fixture(context,host,output,contract=C,_capability=S._CPU_capability());return result,host,value,output
    def test_original_supervisor_v3_argv_indices_and_auditor_exact_join(self):
        result,host,value,output=self.run_supervisor(C.NO_SITE_MODE);sup=result['receipt'];self.assertEqual(result['source_adaptation']['mutation_count'],6);self.assertEqual(sup['status'],'SUPERVISED_WORKER_EXITED_ZERO');self.assertEqual(host.argv,[self.profile['executable'],'-I','-S','-B',value['config']['worker_path'],'--config',value['config_path'],'--output-dir',str(output)]);A.replay_worker_command(sup,value,output)
        bad=dict(sup);bad['attempted_worker_argv']=list(host.argv);bad['attempted_worker_argv'].remove('-S')
        with self.assertRaises(ValueError):A.replay_worker_command(bad,value,output)
        self.assertEqual(sup['bounds'],S.BOUNDS);self.assertTrue(result['cleanup_verified']);self.assertLess(host.calls.index('Job_assigned'),host.calls.index('resume'))
    def test_legacy_supervisor_retains_original_three_changes(self):
        result,host,value,output=self.run_supervisor('MEASURE_BYTE_VERIFIED_STARTUP_V1');self.assertEqual(result['source_adaptation']['mutation_count'],3);self.assertEqual(host.argv[1:3],['-I','-B']);self.assertEqual(host.argv[3],value['config']['worker_path']);self.assertEqual(host.argv[5],value['config_path'])
    def test_profile_auditor_unknown_source_and_missing_measurement_refuse(self):
        runtime={'cache_policy':self.cache,'files':{},'stdlib_layout':{}}
        bootstrap={'kind':'MEASURED_BOOTSTRAP_IMPORTS_V1','imports':self.cache['imports'],'modules':[],'unknown_modules':[],'fixture_only':True,'namespace_execution_proven':False,'interpreter_profile':self.profile};A.replay_bootstrap_modules(bootstrap,runtime,fixture_only=True)
        for bad in [dict(bootstrap),copy.deepcopy(bootstrap),copy.deepcopy(bootstrap)]:
            if bad is not bootstrap:
                if 'interpreter_profile' in bad and bad['interpreter_profile'] is self.profile:bad.pop('interpreter_profile')
                elif bad['unknown_modules']==[]:bad['unknown_modules']=[{'name':'unselected'}]
            with self.assertRaises(ValueError):A.replay_bootstrap_modules(bad,runtime,fixture_only=True)
        bad=copy.deepcopy(bootstrap);bad['interpreter_profile']['flags']['no_site']=0
        with self.assertRaises(ValueError):A.replay_bootstrap_modules(bad,runtime,fixture_only=True)
    def test_profile_record_before_explicit_ctypes_and_driver_before_native_helpers(self):
        source=(HERE/'worker.py').read_text();body=source[source.index('def _native_bootstrap('):source.index('def main():')];self.assertLess(body.index('_measure_no_site_profile'),body.index("__import__('ctypes')"));self.assertEqual(body.count("__import__('ctypes')"),1)
        source=(HERE/'launch_driver.py').read_text();body=source[source.index('def run_registered('):source.index('def _CPU_run_fixture(')];self.assertLess(body.index('_measure_no_site_profile'),body.index('namespaces.begin()'))
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));r={'kind':'CPU_OBSERVER_NO_SITE_PROFILE_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_metadata_only_profile_observed':True,'original_supervisor_body_executed_with_inert_host':True,'ctypes_site_hooks_DLL_model_imported':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['contract.py','worker.py','supervisor.py','audit.py','launch_driver.py','test_no_site_profile.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
