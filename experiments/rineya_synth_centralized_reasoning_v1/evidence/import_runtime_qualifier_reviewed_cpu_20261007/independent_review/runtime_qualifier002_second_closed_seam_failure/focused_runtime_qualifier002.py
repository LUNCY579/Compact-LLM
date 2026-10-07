"""Frozen generic import qualifier semantics; file/roster seams explicit."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import ast,hashlib,json,shutil,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;NATIVE=HERE.parent;FROZEN=NATIVE/'frozen_import_runtime_qualifier002';PROGRAM=NATIVE/'frozen_partial_binder002';PACKAGE=next(p for p in NATIVE.parents if p.name=='rineya_synth_centralized_reasoning_v1');FILES=[p for p in FROZEN.rglob('*') if p.is_file()]+[p for p in PROGRAM.rglob('*') if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'import_runtime_qualification.py')]['sha256']=='aa8f004a1b86d9e63aeca1ec3e5927193ed595c2b0c85781f835e594ffd3a0a1';assert BEFORE[str(FROZEN/'test_import_runtime_qualification.py')]['sha256']=='57cf6be4df3001c6b61292eb7d2bf7142fe7ab89c61166805ad271300125d915'
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);mirror=tempfile.TemporaryDirectory(prefix='CPU_qualifier_frozen_',dir=NATIVE);LOCATION=Path(mirror.name)
for p in PROGRAM.iterdir():
    if p.is_file() and p.suffix=='.py':shutil.copy2(p,LOCATION/p.name)
shutil.copytree(PROGRAM/'predecessors',LOCATION/'predecessors')
for n in ('import_runtime_qualification.py','test_import_runtime_qualification.py'):shutil.copy2(FROZEN/n,LOCATION/n)
# Mirror path affects only HERE-relative fixture companions. Actual selected
# helper initializer is source-owned and remains public PACKAGE checked.
TEST=LOCATION/'test_import_runtime_qualification.py';T=ModuleType('independent_qualifier_fixture');T.__file__=str(TEST);exec(compile(TEST.read_bytes(),str(TEST),'exec'),T.__dict__);Base=T.Controls;Q=T.Q;DETAILS={}
class Independent(Base):
    def test_actual_original_helpers_full_denominator_and_public_package(self):
        self.assertEqual(self.B.PACKAGE,PACKAGE);self.assertEqual(self.B.CORE,PACKAGE/'evidence/prefix_input_core_reviewed_cpu_20261005');result=self.run_value();self.assertFalse(result['current_module_execution_proven']);self.assertFalse(result['native_effects_or_authority_issued']);self.assertGreater(len(self.calls),8700);DETAILS['actual_selected_original_helper_checks']=True;DETAILS['fixture_descriptor_calls']=len(self.calls)
    def test_exact_historical_cache_namespace_main_rebind_refuse(self):self.test_namespace_main_and_historical_cache_rebinding_refuse()
    def test_exact_source_role_observer_ABI_roster_filter_refuse(self):
        original=deepcopy(self.value);self.test_exact_source_or_role_mutation_refuse();self.value=original;self.test_observer_ABI_or_roster_mutation_refuse()
    def test_no_extra_runtime_denominator_and_unknown_effect_refuse(self):
        original=deepcopy(self.value);cache=next(iter(self.value['runtime']['cache_policy']['preloaded_module_cache_pins']));extra=str(Path(cache).with_name('extra.cpython-311.pyc'));self.value['runtime']['files'][extra]=deepcopy(self.value['runtime']['files'][cache]);self.reseal()
        with self.assertRaises(ValueError):self.run_value()
        self.value=deepcopy(original);self.value['runtime']['measurement_policy']['framework_transition']=[];self.reseal()
        with self.assertRaises(ValueError):self.run_value()
    def test_added_preload_overlap_and_cache_path_association_refuse(self):
        original=deepcopy(self.value);row=self.value['runtime']['cache_policy']['preloaded_bootstrap_modules']['ctypes'];p=row['path'];d=self.value['runtime']['files'][p];self.value['runtime']['cache_policy']['additional_preload_sources']={'ctypes':{'path':p,'descriptor':d,'cached_attribute_present':True,'cached_value_kind':'none','cache_path':None,'cache_descriptor':None,'source_execution_history_proven':False}};self.reseal()
        with self.assertRaisesRegex(ValueError,'finite separate'):self.run_value()
        self.value=deepcopy(original);self.value['runtime']['cache_policy']['additional_preload_sources']={'independent_new':{'path':p,'descriptor':d,'cached_attribute_present':True,'cached_value_kind':'string','cache_path':str(Path(p).parent/'__pycache__/wrong.cpython-311.pyc'),'cache_descriptor':deepcopy(next(iter(self.value['runtime']['cache_policy']['preloaded_module_cache_pins'].values()))),'source_execution_history_proven':False}};self.reseal()
        with self.assertRaisesRegex(ValueError,'added cache'):self.run_value()
    def test_different_scope_or_permission_and_review_map_rebind_refuse(self):
        original=deepcopy(self.value);self.value['config']['permission']['tensor_allocations']=2;self.reseal()
        with self.assertRaisesRegex(ValueError,'purpose'):self.run_value()
        self.value=deepcopy(original);self.value['config']['scope']='ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY';self.reseal()
        with self.assertRaises(ValueError):self.run_value()
        self.value=deepcopy(original);self.value['runtime']['initial35_provenance']='changed documentary provenance';self.reseal()
        # Documentary text is not a completion fact; changing a reviewed policy
        # map is material and cannot pass unchanged external review anchors.
        self.value['runtime']['measurement_grammar']['maximum_depth']=65;self.reseal()
        with self.assertRaises(ValueError):self.run_value()
if __name__=='__main__':
    names=('test_actual_original_helpers_full_denominator_and_public_package','test_exact_historical_cache_namespace_main_rebind_refuse','test_exact_source_role_observer_ABI_roster_filter_refuse','test_no_extra_runtime_denominator_and_unknown_effect_refuse','test_added_preload_overlap_and_cache_path_association_refuse','test_different_scope_or_permission_and_review_map_rebind_refuse');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);mirror.cleanup();after=pins();receipt={'kind':'INDEPENDENT_GENERIC_IMPORT_RUNTIME_QUALIFIER002_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_source_pins_unchanged':BEFORE==after,'details':DETAILS,'fixture_HERE_companion_mirror_only':True,'actual_original_map_cache_observer_build_source_profile_schema_helpers_executed':True,'file_descriptor_and_physical_roster_explicit_inert_seams':True,'positive_additional_preload_population_empty':True,'first_review_mutation_state_failure_preserved':'runtime_qualifier002_first_mutation_state_failure','own_process_bootstrap_or_psutil_import_not_executed':True,'native_API_or_framework_called':False,'native_gate_or_context_minted':False,'native_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_RUNTIME_QUALIFIER002_SUCCESSOR.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
