"""Frozen real reader/qualifier integration, disclosed CPU gate/mint boundaries."""
from pathlib import Path
from types import ModuleType
import ast,hashlib,json,shutil,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;NATIVE=HERE.parent;FROZEN=NATIVE/'frozen_reader_qualifier_integration001';COMPANIONS=NATIVE/'frozen_partial_binder002';FILES=[p for p in FROZEN.rglob('*') if p.is_file()]+[p for p in COMPANIONS.rglob('*') if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'import_observation_contract.py')]['sha256']=='f19c53af565a1227a3cd7f116fef2350307c390267e6ace74d5ce6d5476fdd6f';assert BEFORE[str(FROZEN/'import_runtime_qualification.py')]['sha256']=='9c270b3eeffab6acd69d3d1f618645b5fa7f3d57dff64819088879dce83c0607';assert BEFORE[str(FROZEN/'test_import_reader_qualification.py')]['sha256']=='e8219746ffc812a718472f7358e5dd4648f08ed8839d9b12d65ff22218b9541a'
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);mirror=tempfile.TemporaryDirectory(prefix='CPU_reader_integration_frozen_',dir=NATIVE);LOCATION=Path(mirror.name)
for p in COMPANIONS.glob('*.py'):shutil.copy2(p,LOCATION/p.name)
shutil.copytree(COMPANIONS/'predecessors',LOCATION/'predecessors')
for p in FROZEN.glob('*.py'):shutil.copy2(p,LOCATION/p.name)
TEST=LOCATION/'test_import_reader_qualification.py';T=ModuleType('independent_reader_integration_fixture');T.__file__=str(TEST);exec(compile(TEST.read_bytes(),str(TEST),'exec'),T.__dict__);Base=T.Controls;DETAILS={}
class Independent(Base):
    def test_actual_selected_source_exec_full_qualification_before_CPU_return(self):
        self.setup_reader();result=self.call_reader();self.assertEqual(result['kind'],'CPU_READER_QUALIFIER_RETURN_ONLY');self.assertEqual(len(self.minted),1);self.assertEqual(self.gate_calls,['CPU_fixture_gate_boundary']);self.assertEqual(self.exec_calls,['actual_Git_helper_source_exec','actual_qualifier_source_exec']);self.assertEqual(len(self.git_calls),1);self.assertFalse(result['qualification']['current_module_execution_proven']);self.assertFalse(result['native_authority_issued']);self.assertEqual(len(self.minted[0]['runtime']['files']),8724);self.assertEqual(len(self.minted[0]['runtime']['cache_policy']['additional_preload_sources']),10)
        fn=next(n for n in ast.parse((FROZEN/'import_observation_contract.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_native_manifest');calls=[n.func.id for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)];self.assertEqual(calls.count('stage_native_gate'),1);self.assertEqual(calls.count('_mint_import_native'),1);self.assertIn('qualifier.qualify_import_value(value, budget)',ast.unparse(fn));DETAILS['gate_calls']=1;DETAILS['CPU_receipt_returns']=1;DETAILS['actual_selected_exec_order']=list(self.exec_calls)
    def test_actual_main_role_mutations_refuse_before_return(self):self.test_source_bound_main_and_role_mutations_refuse_before_mint()
    def test_actual_additive_cache_refuses_before_return(self):self.test_cache_association_rebind_refuses_before_mint()
    def test_wrong_selected_qualifier_bytes_refuse_before_exec_or_return(self):self.test_wrong_qualifier_pin_refuses_before_execution_or_mint()
if __name__=='__main__':
    names=('test_actual_selected_source_exec_full_qualification_before_CPU_return','test_actual_main_role_mutations_refuse_before_return','test_actual_additive_cache_refuses_before_return','test_wrong_selected_qualifier_bytes_refuse_before_exec_or_return');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);mirror.cleanup();after=pins();receipt={'kind':'INDEPENDENT_ACTUAL_READER_QUALIFIER_INTEGRATION001_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_sources_unchanged':BEFORE==after,'details':DETAILS,'temporary_HERE_companion_mirror_from_pinned_freeze':True,'actual_sealed_packet_reader_and_selected_qualifier_source_executed':True,'full_qualifier_result_not_stubbed':True,'exact_one_gate_and_one_mint_boundary_CPU_fixture_only':True,'file_descriptor_and_physical_roster_read_seams_injected_after_real_exec':True,'Git_immutable_input_join_spy_not_actual_repository_qualification':True,'actual_runtime_byte_hash_or_process_bootstrap_qualified':False,'native_API_or_framework_called':False,'native_context_or_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_READER_INTEGRATION001.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
