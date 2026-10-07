"""Independent finite partial binder components; no native readiness claim."""
from pathlib import Path
from types import ModuleType
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_partial_binder002';TEST=FROZEN/'test_native_binder_components.py';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');FILES=[p for p in FROZEN.rglob('*') if p.is_file()]+[PACKAGE/'native_worker.py',PACKAGE/'prefix_library_observer.py',PACKAGE/'evidence/prefix_input_core_reviewed_cpu_20261005/pipeline.py']
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(TEST)]['sha256']=='83f20428e5a4a090aaa9fd99b6c9e3c45b2b0bbcc04f2a73fd50e23afebb2857';assert BEFORE[str(FROZEN/'import_observation_contract.py')]['sha256']=='1bda856263df1f3fc6105788559c2a69bdb2a9c916aab7375ac57cca4f216fc0';assert BEFORE[str(FROZEN/'import_observation_worker.py')]['sha256']=='d6d0f6397dd7934bca565c144a37bc57d180883d8ac6d97253c47e7a7aced77a'
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker','pynvml'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);T=ModuleType('independent_partial_binder_fixture');T.__file__=str(TEST);exec(compile(TEST.read_bytes(),str(TEST),'exec'),T.__dict__);Base=T.Controls;W=T.W;C=T.C;DETAILS={}
class Independent(Base):
    def test_actual_separate_registry_typed_context_and_native_refusal(self):
        self.test_separate_context_actual_observer_registry_and_crosscontext_refusal();cap=C._CPU_capability()
        with self.assertRaises(ValueError):C._CPU_context({'fixture_only':True,'scope':'ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY'},cap)
        with self.assertRaises(ValueError):C._CPU_context({'fixture_only':True,'scope':C.SCOPE},True)
        DETAILS['original_observer_registry_ImportContext_actual']=True
    def test_actual_seven_original_source_helper_slice_and_no_path_mutation(self):
        self.test_original_ownership_source_slice_avoids_module_path_mutation();DETAILS['original_seven_helper_definitions_selected_no_APIs']=True
    def test_acquisition_failure_always_restores_roots_and_cleanup_priority(self):
        self.test_acquisition_failure_always_restores_roots_and_retains_cleanup_priority();DETAILS['actual_root_cleanup_primary_secondary']=True
    def test_partial_raw_packet_original_writer_without_synthetic_measurement(self):
        self.test_constructor_partial_packet_actual_writer_and_no_false_measurement();DETAILS['original_writer_partial_packet_actual']=True
    def test_selected_measurement_context_and_class_refusal(self):
        self.test_selected_measurement_loader_forged_object_and_context_refused();DETAILS['source_owned_selector_forgery_refusals_actual']=True
if __name__=='__main__':
    names=('test_actual_separate_registry_typed_context_and_native_refusal','test_actual_seven_original_source_helper_slice_and_no_path_mutation','test_acquisition_failure_always_restores_roots_and_cleanup_priority','test_partial_raw_packet_original_writer_without_synthetic_measurement','test_selected_measurement_context_and_class_refusal');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);after=pins();receipt={'kind':'INDEPENDENT_PARTIAL_IMPORT_NATIVE_BINDER_COMPONENT_REVIEW002','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_sources_unchanged':BEFORE==after,'details':DETAILS,'registry_metadata_selector_injected':True,'ctypes_helper_object_explicit_inert_standin':True,'actual_original_writer_component_called':True,'full_native_factory_main_or_registered_runtime_executed':False,'generic_reader_cache_role_observer_bootstrap_qualification_complete':False,'psutil_additional_bootstrap_and_main_baseline_qualified':False,'native_API_or_framework_called':False,'native_permission_or_authority_issued':False,'partial_candidate_runnable':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_PARTIAL_BINDER002.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
