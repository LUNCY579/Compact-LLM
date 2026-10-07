"""Frozen native-body schema joins; all SUP/resource/death facts CPU invented."""
from pathlib import Path
from copy import deepcopy
import ast, hashlib, json, shutil, sys, tempfile, types, unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'final_native_entry_consumer001';GRAPH=FROZEN/'final_placement_category004';EARLY_GRAPH=FROZEN/'final_early_failure001';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');SUPERVISOR=PACKAGE/'evidence/site_disabled_tensor_packet_reviewed_cpu_20261007/prefix_input_owned_supervisor.py'
INPUT=PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json'
FILES=[p for p in FROZEN.rglob('*') if p.is_file()]+[SUPERVISOR,INPUT]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins()
for path,expected in [(FROZEN/'prefix_input_owned_auditor.py','e06f987f252a6b1f6c5fa3557b4cc545905b305c3dc17c10eb727606a6993b58'),(FROZEN/'test_native_entry_auditor.py','68697e8a91ec108a34d50c5c4ed69b66e52b02ed0f2c9278813cf001d93ab24d'),(EARLY_GRAPH/'prefix_input_owned_worker.py','36c5f9d1462721adc76e46dd62dd79e42654790b442cd5cc824fbcda0ee7fd20')]:
    if BEFORE[str(path)]['sha256']!=expected:raise ValueError('selected frozen pin changed')
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);raw=path.read_bytes();m.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};exec(compile(raw,str(path),'exec'),m.__dict__);return m
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
TRAP=Trap();sys.meta_path.insert(0,TRAP)
W=load(GRAPH/'prefix_input_owned_worker.py','independent_join_core');EARLY=load(EARLY_GRAPH/'prefix_input_owned_worker.py','independent_join_early');P=load(GRAPH/'prefix_native_pipeline.py','independent_join_pipeline');A=load(FROZEN/'prefix_input_owned_auditor.py','independent_join_consumer');F=load(W.C.CORE/'fixture_support.py','independent_join_api')
input_tree=ast.parse((FROZEN/'prefix_input_owned_auditor.py').read_bytes());input_fn=next(n for n in input_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_input');input_changes=0
for n in input_fn.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='path':
        if ast.unparse(n.value)!="PACKAGE / 'evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json'":raise ValueError('unexpected fixture input path')
        n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(INPUT))],keywords=[]);input_changes+=1
if input_changes!=1:raise ValueError('one auditor input fixture expression')
exec(compile(ast.fix_missing_locations(ast.Module(body=[input_fn],type_ignores=[])),str(FROZEN/'prefix_input_owned_auditor.py'),'exec'),A.__dict__)
ADAPTATIONS=[]
for module,graph in [(W,GRAPH),(EARLY,EARLY_GRAPH)]:
    tree=ast.parse((graph/'prefix_input_owned_worker.py').read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_entry_journal_bootstrap');changes=0
    for n in fn.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='package':
            if ast.unparse(n.value)!="HERE.parents[1] if HERE.parent.name == 'artifacts' else HERE":raise ValueError('unexpected bootstrap fixture expression')
            n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(PACKAGE))],keywords=[]);changes+=1
    if changes!=1:raise ValueError('one explicit local package fixture expression')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(graph/'prefix_input_owned_worker.py'),'exec'),module.__dict__);ADAPTATIONS.append(str(graph/'prefix_input_owned_worker.py'))
# Preserve base consumer definitions, omit only fixture source-copy initialization.
base_tree=ast.parse((FROZEN/'test_entry_auditor_producer004.py').read_bytes());base_nodes=[n for n in base_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and getattr(n,'name',None) in ('descriptor','write','Controls')]
base_ns={'Path':Path,'deepcopy':deepcopy,'hashlib':hashlib,'json':json,'shutil':shutil,'sys':sys,'tempfile':tempfile,'types':types,'unittest':unittest,'load':load,'W':W,'P':P,'A':A,'F':F,'FIXTURE_LAYOUT':GRAPH,'GRAPH':GRAPH,'HERE':FROZEN,'PREFIX':json.loads(W.source_core()['C'].INPUT.read_bytes())['prefix_ids']}
exec(compile(ast.Module(body=base_nodes,type_ignores=[]),str(FROZEN/'test_entry_auditor_producer004.py'),'exec'),base_ns);BASE=types.SimpleNamespace(**base_ns)
native_tree=ast.parse((FROZEN/'test_native_entry_auditor.py').read_bytes());controls=next(n for n in native_tree.body if isinstance(n,ast.ClassDef) and n.name=='Controls')
# This fixture's helper file exists as tracked package source rather than in its
# isolated freeze. Only the helper path expression is replaced; hash is pinned.
changes=0
for n in ast.walk(controls):
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='helper':
        if ast.unparse(n.value)!="str(HERE / 'prefix_input_owned_supervisor.py')":raise ValueError('unexpected helper path expression')
        n.value=ast.Constant(str(SUPERVISOR));changes+=1
if changes!=1:raise ValueError('one fixture helper path expression')
native_ns={**base_ns,'BASE':BASE,'EARLY':EARLY}
exec(compile(ast.fix_missing_locations(ast.Module(body=[controls],type_ignores=[])),str(FROZEN/'test_native_entry_auditor.py'),'exec'),native_ns);Base=native_ns['Controls'];TRACES=[]
class Independent(Base):
    def test_actual_early_writer_join_absent_core_facts(self):
        self.fixture(early=True,actual_early=True);envelope=json.loads((self.output/'OWNED_WORKER_RETURN.json').read_bytes());final=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes())
        self.assertIsNone(envelope['runtime_execution_modules']);self.assertIsNone(envelope['owned_membership_verified']);self.assertEqual(envelope['report']['core_membership_status'],'NOT_CORE_ENROLLED');self.assertEqual(final['IO']['owned_return_write']['returned'],1)
        value=self.native_replay();self.assertEqual(value['status'],'CPU_NATIVE_SCHEMA_FAILURE_PRESERVED');self.assertTrue(value['precore_facts_unavailable']);self.assertFalse(value['native_shared_entry_terminal_qualified']);TRACES.append({'case':'actual early writer','status':value['status'],'markers':final['markers'],'raw_envelope_consumed':True})
    def test_actual_early_writer_failure_native_body_refuses_missing_envelope(self):
        self.fixture(early=True,actual_early=True,write_failure=True);self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists());final=json.loads((self.output/'PHASE_ACCOUNTING_FINAL.json').read_bytes());self.assertEqual(final['IO']['owned_return_write'],{'attempted':1,'returned':0,'failed':1,'pending':0})
        with self.assertRaisesRegex(ValueError,'all original native supervisor/core/log raw files'):self.native_replay()
        TRACES.append({'case':'actual failed early writer','returned_envelope':False,'failed_IO':final['IO']['owned_return_write']})
    def test_full_inert_core_native_schema_flags(self):
        self.fixture();value=self.native_replay();self.assertEqual(value['status'],'CPU_NATIVE_SCHEMA_COMPLETE');self.assertTrue(value['shared_entry_composition_source_exercised']);self.assertFalse(value['native_shared_entry_terminal_qualified']);self.assertFalse(value['native_authority_issued']);TRACES.append({'case':'full inert core','status':value['status']})
    def test_native_gate_and_resealed_membership_refuse(self):self.test_nonfixture_gate_and_resealed_missing_membership_refuse()
    def test_resealed_called_source_pin_override_refuse(self):self.test_resealed_startup_source_pin_override_refuses_in_native_body()
if __name__=='__main__':
    names=('test_actual_early_writer_join_absent_core_facts','test_actual_early_writer_failure_native_body_refuses_missing_envelope','test_full_inert_core_native_schema_flags','test_native_gate_and_resealed_membership_refuse','test_resealed_called_source_pin_override_refuse');result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(TRAP);after=pins()
    receipt={'kind':'INDEPENDENT_FROZEN_EARLY_NATIVE_SCHEMA_JOIN001','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_sources_unchanged':BEFORE==after,'actual_original_writer_early_envelope_and_failed_IO_exercised':True,'actual_native_preserved_body_Readers_replay_exercised':True,'traces':TRACES,'fixture_relocations':{'bootstrap_one_local_PACKAGE_AST_expression_per_worker':ADAPTATIONS,'base_test_definitions_only_selected':True,'GRAPH_and_FIXTURE_LAYOUT':str(GRAPH),'supervisor_path_one_AST_expression':str(SUPERVISOR),'auditor_input_one_AST_expression':str(INPUT),'candidate_bytes_modified':False},'initial_fixture_failures_preserved':['early_join001_initial_fixture_failure','early_join001_input_fixture_failure'],'SUP_Job_resource_death_metadata_invented_CPU_only':True,'native_import_trap_active':True,'native_API_or_framework_called':False,'native_context_or_gate_bypass':False,'native_shared_entry_terminal_qualified':False,'native_authority_issued':False}
    receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_EARLY_NATIVE_JOIN001_SUCCESSOR.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() and BEFORE==after else 1)
