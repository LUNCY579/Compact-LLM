"""Independent installed TorchVersion metadata normalization source controls."""
from pathlib import Path
import ast,hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_version_source_fix001';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');TEST=FROZEN/'test_version_source_fix.py';FILES=[p for p in FROZEN.rglob('*') if p.is_file()]+[PACKAGE/'import_observation_contract.py']
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'import_observation_worker.py')]['sha256']=='16452323d6a6f000f630864e2726d50cd4621ba396686be95638196cec94b4db';assert BEFORE[str(TEST)]['sha256']=='0f4de4e7143ec3d26cbc326c5ff291154b00ce97de2ab98139b87403c7b074bf'
tree=ast.parse(TEST.read_bytes());nodes=[];changes=0
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PACKAGE' for t in n.targets):n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(PACKAGE))],keywords=[]);changes+=1
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets):break
    nodes.append(n)
assert changes==1;ns={'__name__':'independent_version_fixture','__file__':str(TEST)};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(TEST),'exec'),ns);Base=ns['Controls'];DETAILS={}
class Independent(Base):
    def test_actual_TorchVersion_built_in_plain_normalization_without_hooks(self):self.test_actual_TorchVersion_is_plain_exact_metadata_without_comparison_hook();self.assertEqual(ns['hooks'],[]);DETAILS['actual_selected_TorchVersion_AST_no_torch_import']=True
    def test_version_mismatch_journal_failure_pending_zero_cleanup_raw(self):self.test_version_mismatch_is_failed_inside_journal_and_cleanup_raw_survive();DETAILS['version_failed_counts']={'attempted':1,'returned':0,'failed':1,'pending':0}
    def test_arbitrary_str_hook_refused_and_native_expected_pin_private_source(self):self.test_arbitrary_object_string_hook_refused_and_native_pin_source_bound();self.assertEqual(ns['hooks'],[]);DETAILS['arbitrary_str_or_comparison_hooks_called']=0
if __name__=='__main__':
    names=('test_actual_TorchVersion_built_in_plain_normalization_without_hooks','test_version_mismatch_journal_failure_pending_zero_cleanup_raw','test_arbitrary_str_hook_refused_and_native_expected_pin_private_source');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(ns['trap']);after=pins();receipt={'kind':'INDEPENDENT_TORCH_VERSION_METADATA_SOURCE_FIX001_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'pins_unchanged':BEFORE==after,'details':DETAILS,'one_test_PACKAGE_AST_location_adaptation':str(PACKAGE),'actual_original_Ledger_and_version_operation_source_executed':True,'installed_TorchVersion_class_and_comparison_loop_AST_executed':True,'framework_measurement_resources_inert':True,'native_expected_version_binding_source_inspected_only':True,'full_raw_effect_or_terminal_replay_qualified':False,'Torch_ctypes_psutil_imported':False,'native_API_or_tensor_model_called':False,'native_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_VERSION_FIX001.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
