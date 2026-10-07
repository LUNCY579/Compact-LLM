"""Composed actual source-prelude four-to-six transition, no native imports."""
from pathlib import Path
import builtins,copy,hashlib,json,sys,types,unittest
HERE=Path(__file__).resolve().parent
ORIGINAL=builtins.__import__
def trap(name,*args,**kwargs):
    if name.split('.',1)[0] in {'ctypes','_ctypes','torch','psutil','native_worker','native_supervisor','native_process','native_nvml','pywin32_bootstrap'} or name.startswith('__editable__'):raise AssertionError('CPU transition controls forbid native/site imports')
    return ORIGINAL(name,*args,**kwargs)
builtins.__import__=trap
def load(name):
    p=HERE/name;m=types.ModuleType('CPU_'+p.stem);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
C=load('contract.py');W=load('worker.py');P=load('prelude.py')
OLD=C.PACKAGE/'artifacts/tensor_prelude_observer_work'
class Controls(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((OLD/'PRELUDE_PLAN_003.json').read_text());runtime=json.loads((C.PACKAGE/'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_007.json').read_text());source_map={r['path']:r['descriptor'] for r in self.plan['sources'].values()};source_map.update(self.plan['helper_sources']);self.value={'config':{'source_files':source_map},'runtime':{'files':runtime['files'],'stdlib_layout':runtime['stdlib_layout'],'cache_policy':{'interpreter_profile':C.expected_no_site_profile(runtime['stdlib_layout'])}},'fixture_only':True}
    def prelude(self):
        initial=W._measure_no_site_profile(C,self.value,sys);record,transition,modules=P.source_prelude(self.plan,C,self.value,cpu_only=True);return initial,record,transition
    def test_actual_prelude_second_profile_and_original_rollback(self):
        before=list(sys.path);initial,record,transition=self.prelude()
        try:
            measured=W._measure_registered_root_profile(C,self.value,sys,transition,self.plan,initial);self.assertEqual(measured['path'],before+self.plan['registered_roots']);self.assertEqual(initial['path'],before);self.assertEqual(len(sys.path),6)
            with self.assertRaises(ValueError):W._measure_no_site_profile(C,self.value,sys)
        finally:P.restore_roots(transition)
        self.assertEqual(sys.path,before)
    def test_foreign_path_container_and_flags_refuse_without_clobber(self):
        initial,record,transition=self.prelude()
        try:
            sys.path.append('CPU_FOREIGN')
            with self.assertRaises(ValueError):W._measure_registered_root_profile(C,self.value,sys,transition,self.plan,initial)
            with self.assertRaises(ValueError):P.restore_roots(transition)
            self.assertEqual(sys.path[-1],'CPU_FOREIGN');sys.path.pop()
            actor=types.SimpleNamespace(**{k:getattr(sys,k) for k in ['executable','prefix','exec_prefix','base_prefix','base_exec_prefix','meta_path','path_hooks']},path=sys.path,flags=types.SimpleNamespace(**initial['flags']));actor.flags.no_site=0
            with self.assertRaises(ValueError):W._measure_registered_root_profile(C,self.value,actor,transition,self.plan,initial)
            actor.flags.no_site=1;actor.path=list(sys.path)
            with self.assertRaises(ValueError):W._measure_registered_root_profile(C,self.value,actor,transition,self.plan,initial)
        finally:P.restore_roots(transition)
    def test_rebound_root_plan_or_initial_profile_refuses(self):
        initial,record,transition=self.prelude()
        try:
            bad=copy.deepcopy(self.plan);bad['registered_roots']=list(reversed(bad['registered_roots']))
            with self.assertRaises(ValueError):W._measure_registered_root_profile(C,self.value,sys,transition,bad,initial)
            bad=copy.deepcopy(initial);bad['flags']['isolated']=0
            with self.assertRaises(ValueError):W._measure_registered_root_profile(C,self.value,sys,transition,self.plan,bad)
        finally:P.restore_roots(transition)
    def test_native_path_routes_second_call_to_transition_checker(self):
        source=(HERE/'worker.py').read_text();body=source[source.index('def _native_bootstrap('):source.index('def main():')];self.assertEqual(body.count('_measure_no_site_profile'),1);self.assertEqual(body.count('_measure_registered_root_profile'),1);self.assertLess(body.index('_measure_registered_root_profile'),body.index("__import__('ctypes')"));self.assertEqual(len(C.OPS),12)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));r={'kind':'CPU_EXACT_PRELUDE_TRANSITION_REPAIR_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_definition_prelude_then_second_profile':True,'ctypes_observer_framework_tensor_called':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['contract.py','worker.py','prelude.py','test_transition.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
