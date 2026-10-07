"""Frozen native-body schema joins; all SUP/resource/death facts CPU invented."""
from pathlib import Path
from copy import deepcopy
import ast, hashlib, json, shutil, sys, tempfile, types, unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parents[1]/'tensor_observed_admission_work/final_native_entry_consumer001';GRAPH=FROZEN/'final_placement_category004';EARLY_GRAPH=FROZEN/'final_early_failure001';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');SUPERVISOR=PACKAGE/'evidence/site_disabled_tensor_packet_reviewed_cpu_20261007/prefix_input_owned_supervisor.py'
INPUT=PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json'
FIX=HERE.parent/'gate_dispatch_candidate/final_V2terminal_join003'
FILES=[p for p in FROZEN.rglob('*') if p.is_file()]+[p for p in FIX.rglob('*') if p.is_file()]+[SUPERVISOR,INPUT]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins()
for path,expected in [(FIX/'prefix_input_owned_auditor.py','c632d57cc5d992506c0577c2f939cf620d84f472ca76cdf6173108325c8299a8'),(FIX/'test_V2terminal_join.py','f7fb7e957883cd7ccd78763c1e975105dca3dee49b25929119cc61eea75d491b'),(EARLY_GRAPH/'prefix_input_owned_worker.py','36c5f9d1462721adc76e46dd62dd79e42654790b442cd5cc824fbcda0ee7fd20')]:
    if BEFORE[str(path)]['sha256']!=expected:raise ValueError('selected frozen pin changed')
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);raw=path.read_bytes();m.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};exec(compile(raw,str(path),'exec'),m.__dict__);return m
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
TRAP=Trap();sys.meta_path.insert(0,TRAP)
W=load(GRAPH/'prefix_input_owned_worker.py','independent_join_core');EARLY=load(EARLY_GRAPH/'prefix_input_owned_worker.py','independent_join_early');P=load(GRAPH/'prefix_native_pipeline.py','independent_join_pipeline');A=load(FIX/'prefix_input_owned_auditor.py','independent_join_consumer');F=load(W.C.CORE/'fixture_support.py','independent_join_api')
input_tree=ast.parse((FIX/'prefix_input_owned_auditor.py').read_bytes());input_fn=next(n for n in input_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_input');input_changes=0
for n in input_fn.body:
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='path':
        if ast.unparse(n.value)!="PACKAGE / 'evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json'":raise ValueError('unexpected fixture input path')
        n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(INPUT))],keywords=[]);input_changes+=1
if input_changes!=1:raise ValueError('one auditor input fixture expression')
exec(compile(ast.fix_missing_locations(ast.Module(body=[input_fn],type_ignores=[])),str(FIX/'prefix_input_owned_auditor.py'),'exec'),A.__dict__)
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

V2tree=ast.parse((FIX/'test_V2terminal_join.py').read_bytes());V2class=next(n for n in V2tree.body if isinstance(n,ast.ClassDef) and n.name=='Controls')
B=types.SimpleNamespace(Controls=Base,BASE=BASE,A=A);V2ns={**native_ns,'B':B,'A':A};exec(compile(ast.Module(body=[V2class],type_ignores=[]),str(FIX/'test_V2terminal_join.py'),'exec'),V2ns);V2Base=V2ns['Controls']
class Independent(V2Base):
    def test_actual_full_V2_bytes_and_gate_reference_reader(self):
        self.prepare();result=self.audit();self.assertEqual(result['status'],'CPU_NATIVE_SCHEMA_COMPLETE');self.assertFalse(result['native_authority_issued']);TRACES.append({'case':'fullV2','status':result['status']})
    def test_missing_schema_and_rebound_document_refuse(self):
        self.test_missing_schema_and_rebound_reference_refuse_after_reseal()
    def test_selected_gate_reference_rebind_refuses(self):
        self.test_rebound_actual_gate_document_refused()
    def test_nine_anchor_and_actual_document_bytes_refuse(self):
        self.test_ninth_anchor_rebind_and_document_bytes_refused()
if __name__=='__main__':
    names=('test_actual_full_V2_bytes_and_gate_reference_reader','test_missing_schema_and_rebound_document_refuse','test_selected_gate_reference_rebind_refuses','test_nine_anchor_and_actual_document_bytes_refuse');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(TRAP);after=pins();receipt={'kind':'INDEPENDENT_FULL_V2_TERMINAL_JOIN003_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'pins_unchanged':BEFORE==after,'traces':TRACES,'actual_native_preserved_body_and_original_writer_core_reader_executed':True,'full_V2_record_config_nineanchors_both_refs_bound':True,'actual_gate_document_bytes_read':True,'SUP_Job_resource_death_and_gate_documents_explicit_CPU_metadata':True,'fixture_AST_PACKAGE_input_helper_GRAPH_relocations_recorded_in_script':True,'native_API_or_framework_called':False,'native_authority_issued':False,'real_human_grant_present':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_V2TERMINAL_JOIN003.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
