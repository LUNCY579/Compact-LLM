"""Six frozen producer/consumer controls with inert files and import surfaces."""
from pathlib import Path
from unittest.mock import patch
import ast, contextlib, hashlib, importlib, json, sys, tempfile, types, unittest
HERE=Path(__file__).resolve().parent;EVIDENCE=HERE.parent;GRAPH=EVIDENCE/'final_producer004';CONSUMER=EVIDENCE/'final_consumer';PACKAGE=next(p for p in EVIDENCE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
PATHS=list(GRAPH.glob('*'))+[CONSUMER/'prefix_input_owned_auditor.py',CONSUMER/'test_entry_auditor_producer004.py']
PATHS=[p for p in PATHS if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in PATHS}
BEFORE=pins()
for n,h in {'prefix_input_owned_worker.py':'b71075e252b5d2c18fe32644d3d22257131ba23a931cdedc52597f4f43e02afc','prefix_native_pipeline.py':'9efb6f15444a6449734fe85d5eddd72d3cefb2a76c0b7af46264b44ed68cbb2b','prefix_tensor_contract.py':'0d12fd01bb1f06c29054943a3a114efc6eac9619c33989e5ce030d97a3cf4263','prefix_registered_loader.py':'62c25f6cf5f843a08463dbb7c3c573f6a038f036917fc4130da0aab5d4b7e0c5'}.items():
    if BEFORE[str(GRAPH/n)]['sha256']!=h:raise ValueError('frozen producer pin changed')
if BEFORE[str(CONSUMER/'prefix_input_owned_auditor.py')]['sha256']!='c6685314a8fc41c36ece59a9ad367f7d7081879cc960bee50bd791399752d78a':raise ValueError('frozen consumer changed')

def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);raw=path.read_bytes();m.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};exec(compile(raw,str(path),'exec'),m.__dict__);return m
class NoNative:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
TRAP=NoNative();sys.meta_path.insert(0,TRAP)
W=load(GRAPH/'prefix_input_owned_worker.py','independent004worker');P=load(GRAPH/'prefix_native_pipeline.py','independent004pipeline')
# Explicit test-only relocation of one local package expression in the pinned
# writer bootstrap. All original writer operations and source pins are retained.
tree=ast.parse((GRAPH/'prefix_input_owned_worker.py').read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_entry_journal_bootstrap');changed=0
for n in node.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='package':
        if ast.unparse(n.value)!="HERE.parents[1] if HERE.parent.name == 'artifacts' else HERE":raise ValueError('unexpected fixture package expression')
        n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(PACKAGE))],keywords=[]);changed+=1
if changed!=1:raise ValueError('one package fixture adaptation')
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(GRAPH/'prefix_input_owned_worker.py'),'exec'),W.__dict__)

# Load frozen consumer fixture definitions without its source-copy initialization.
# GRAPH and FIXTURE_LAYOUT both refer directly to frozen producer bytes.
raw=(CONSUMER/'test_entry_auditor_producer004.py').read_bytes();ct=ast.parse(raw);definitions=[n for n in ct.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and getattr(n,'name',None) in ('descriptor','write','Controls')]
A=load(CONSUMER/'prefix_input_owned_auditor.py','independent004auditor');F=load(W.C.CORE/'fixture_support.py','independent004api')
ns={'Path':Path,'deepcopy':__import__('copy').deepcopy,'hashlib':hashlib,'json':json,'shutil':__import__('shutil'),'sys':sys,'tempfile':tempfile,'types':types,'unittest':unittest,'load':load,'W':W,'P':P,'A':A,'F':F,'FIXTURE_LAYOUT':GRAPH,'GRAPH':GRAPH,'HERE':CONSUMER,'PREFIX':json.loads(W.source_core()['C'].INPUT.read_bytes())['prefix_ids']}
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(CONSUMER/'test_entry_auditor_producer004.py'),'exec'),ns);ConsumerBase=ns['Controls']
DETAILS={}

class ProducerControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='independent004producer_');self.root=Path(self.tmp.name).resolve();self.site=self.root/'site';self.site.mkdir();self.output=self.root/'output';self.output.mkdir()
        self.C=load(GRAPH/'prefix_tensor_contract.py','independent004contract');self.L=load(GRAPH/'prefix_registered_loader.py','independent004loader');self.cap=self.C._mint_CPU_native_context_fixture();self.prior=W.C;W.C=self.C
        pkg=self.site/'independent_inert_pkg';pkg.mkdir();(pkg/'__init__.py').write_bytes(b'from . import child\nVALUE=child.VALUE\n');(pkg/'child.py').write_bytes(b'VALUE=29\n')
        files={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'python_source'} for p in pkg.glob('*.py')}
        self.value={'fixture_only':True,'native_PASS':False,'native_authority_issued':False,'config':{'source_files':{}},'runtime':{'files':files,'stdlib_layout':self.C.QUALIFIED_LAYOUT,'cache_policy':{'Torch_root':str(self.site/'torch'),'namespace_roots':{},'preloaded_bootstrap_modules':{},'preloaded_module_cache_pins':{}}}}
        self.context=self.C._mint_native_context(self.value);self.owner={'worker_pid':11,'supervisor_pid':22,'job_name':'CPU_independent004'};self.modules={'sys':sys};self.cache={};self.path=[str(self.site)];self.meta=[importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder];self.hooks=[__import__('zipimport').zipimporter,importlib.machinery.FileFinder.path_hook((importlib.machinery.ExtensionFileLoader,importlib.machinery.EXTENSION_SUFFIXES),(importlib.machinery.SourceFileLoader,importlib.machinery.SOURCE_SUFFIXES),(importlib.machinery.SourcelessFileLoader,importlib.machinery.BYTECODE_SUFFIXES))]
    def tearDown(self):W.C=self.prior;self.tmp.cleanup()
    @contextlib.contextmanager
    def active(self):
        with patch.object(sys,'modules',self.modules),patch.object(sys,'meta_path',self.meta),patch.object(sys,'path_hooks',self.hooks),patch.object(sys,'path',self.path),patch.object(sys,'path_importer_cache',self.cache):yield
    def install(self,*args):return self.L._CPU_install_observed_fixture(self.context,self.C,_capability=self.cap)
    def verify(self,*args):return {'actual_worker_pid':11,'actual_parent_pid':22,'launcher_root_pid':11,'supervisor_pid':22,'job_name':'CPU_independent004','owned_job_membership_verified':True,'query_job_handle_closed_before_framework_import':True}
    def state(self):return {'inputs':{'context':self.context,'owner':self.owner,'output':self.output}}
    def test_actual_common_install_acquisition_one_use_handoff(self):
        with self.active():
            state=self.state();events=[]
            def prepare(*args):events.append('prepare');self.assertNotIn('independent_inert_pkg',sys.modules)
            result=W._prepare_acquired_entry(state,install=self.install,verify=self.verify,prepare=prepare,_CPU_capability=self.cap);entry=W._consume_entry_handoff(state['entry_handoff'],self.context,self.owner,self.output)
            self.assertEqual(events,['prepare']);self.assertTrue(result['fixture_only']);self.assertFalse(result['native_authority_issued']);m=importlib.import_module('independent_inert_pkg');self.assertEqual(m.VALUE,29);self.assertTrue(entry['finder'].observed['independent_inert_pkg.child']['current_module_execution_observed']);self.L.remove(entry['finder']);self.assertEqual(self.cache,{})
            with self.assertRaises(ValueError):W._consume_entry_handoff(state['entry_handoff'],self.context,self.owner,self.output)
        DETAILS['common_install_actual']=True
    def test_acquisition_failure_cleanup_forged_handoff_closed_native_gate(self):
        with self.active():
            for stage in ('verify','prepare'):
                state=self.state()
                def failed(*args):raise RuntimeError('independent '+stage+' failed')
                with self.assertRaisesRegex(RuntimeError,'independent '+stage):W._prepare_acquired_entry(state,install=self.install,verify=failed if stage=='verify' else self.verify,prepare=failed if stage=='prepare' else lambda *a:None,_CPU_capability=self.cap)
                self.assertIn('acquired_finder',state);self.assertNotIn('entry_handoff',state);self.L.remove(state['acquired_finder']);self.assertEqual(self.cache,{})
            with self.assertRaises(ValueError):W._consume_entry_handoff(object(),self.context,self.owner,self.output)
            with self.assertRaises(ValueError):W._prepare_acquired_entry(self.state(),install=lambda *a:self.fail('native gate bypassed'),verify=self.verify,prepare=lambda *a:None)
        DETAILS['acquisition_failure_cleanup_actual']=True
    def test_cleanup_meter_restored_after_roots_failure_and_phase5_source_order(self):
        meter=W._SelectedSourceMeter();original=self.C.read_bytes;meter.bind_contract(self.C);actions,_=W._native_entry_actions(types.SimpleNamespace(),types.SimpleNamespace(),P.STARTUP_PHASES)
        with self.active():
            state={'source_meter':meter,'transition':{'original_container':self.path,'path_after':['foreign mutation'],'before':tuple(self.path)}}
            with self.assertRaisesRegex(ValueError,'transition unchanged'):actions[P.STARTUP_PHASES[-1]](state,{})
        self.assertIs(self.C.read_bytes,original);self.assertNotIn('__source_meter__',vars(self.C))
        t=ast.parse((GRAPH/'prefix_input_owned_worker.py').read_bytes());native=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_native_entry_actions');provider=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='provider');self.assertIn('_prepare_acquired_entry',ast.unparse(provider));self.assertIn("install=state['install']",ast.unparse(provider));execute=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='execute');self.assertIn("_entry_handoff=state['entry_handoff']",ast.unparse(execute));prepare=next(n for n in ast.parse((GRAPH/'prefix_native_pipeline.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_owned_main_entry');self.assertIn('one source-owned prepared core consumption',ast.unparse(prepare));DETAILS['native_prepare_source_only']=True

class ConsumerControls(ConsumerBase):
    def test_independent_mixed_category_replay_and_resealed_forgery(self):
        self.test_actual_mixed_source_cache_cost_derivation_and_resealed_forgery();DETAILS['mixed_category_original_writer_reader_replay']=True
    def test_independent_resealed_order_source_override_and_partial_failure(self):
        self.test_resealed_order_and_source_pin_override_refused();DETAILS['order_and_source_forgery_refused']=True
    def test_independent_partial_original_writer_reader_no_completion(self):
        self.test_actual_precore_failure_absent_worker_result_is_partial();DETAILS['partial_failure_reader_replay']=True

if __name__=='__main__':
    suite=unittest.TestSuite();suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ProducerControls));suite.addTests(ConsumerControls(n) for n in ('test_independent_mixed_category_replay_and_resealed_forgery','test_independent_resealed_order_source_override_and_partial_failure','test_independent_partial_original_writer_reader_no_completion'))
    result=unittest.TextTestRunner(verbosity=2).run(suite);sys.meta_path.remove(TRAP);after=pins()
    receipt={'kind':'INDEPENDENT_FROZEN_PRODUCER_CONSUMER004_REVIEW','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_source_files_unchanged':BEFORE==after,'details':DETAILS,'fixture_relocation':{'producer_bootstrap_one_local_PACKAGE_AST_substitution':True,'consumer_fixture_definitions_only_selected':True,'GRAPH_and_FIXTURE_LAYOUT':str(GRAPH),'candidate_bytes_modified':False},'native_import_trap_used':True,'native_prepare_factory_or_pipeline_executed':False,'native_API_or_framework_called':False,'native_authority_issued':False,'whole_native_terminal_consumer_qualified':False}
    receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_PRODUCER_CONSUMER004_REVIEW.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() and BEFORE==after else 1)
