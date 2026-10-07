"""Frozen generic import qualifier semantics; file/roster seams explicit."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import ast,hashlib,json,shutil,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;NATIVE=HERE.parent;FROZEN=NATIVE/'frozen_import_runtime_qualifier003';PROGRAM=NATIVE/'frozen_partial_binder002';PACKAGE=next(p for p in NATIVE.parents if p.name=='rineya_synth_centralized_reasoning_v1');FILES=[p for p in FROZEN.rglob('*') if p.is_file()]+[p for p in PROGRAM.rglob('*') if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'import_runtime_qualification.py')]['sha256']=='9c270b3eeffab6acd69d3d1f618645b5fa7f3d57dff64819088879dce83c0607';assert BEFORE[str(FROZEN/'test_import_runtime_qualification.py')]['sha256']=='abc8018f28554abc1fd95c8f08599870f0da90c41cc3f2865d1b0112b07493aa'
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
    def test_actual_ten_overlay_full_qualifier_preserves_historical_maps(self):
        self.add_passive_overlay();runtime=self.value['runtime'];result=self.run_value();self.assertFalse(result['current_module_execution_proven']);self.assertEqual(len(runtime['files']),8724);self.assertEqual(len(runtime['cache_policy']['additional_preload_sources']),10);self.assertEqual(len(runtime['measurement_policy']['initial_loaded_libraries']),39);self.assertEqual(len(runtime['cache_policy']['preloaded_bootstrap_modules']),63);self.assertEqual(len(runtime['cache_policy']['preloaded_module_cache_pins']),59);self.assertEqual(set(runtime['cache_policy']['preloaded_class_aliases']),{'typing.io','typing.re'});DETAILS['positive_overlay']={'files':8724,'rows':10,'candidate_baseline':39,'observed_modules':63,'observed_cache_pins':59}
    def test_actual_overlay_association_role_and_historical_overlap_refuse(self):
        self.test_actual_passive_overlay_full_positive_and_cache_role_overlap_refusals()
    def test_external_passive_receipt_file_rebind_refuses(self):
        self.add_passive_overlay();self.value['runtime']['metadata_references']['additional_preload']['file_sha256']='0'*64;self.reseal()
        with self.assertRaises(ValueError):self.run_value()
    def test_baseline_unknown_path_and_cache_value_string_refuse(self):
        self.add_passive_overlay();original=deepcopy(self.value);self.value['runtime']['measurement_policy']['initial_loaded_libraries'].append('C:\\unselected.dll');self.reseal()
        with self.assertRaises(ValueError):self.run_value()
        self.value=deepcopy(original);self.value['runtime']['cache_policy']['additional_preload_sources']['psutil']['cached_value_kind']='string';self.reseal()
        with self.assertRaises(ValueError):self.run_value()
    def test_full_role_source_namespace_main_and_original_ABI_refusals(self):
        self.test_namespace_main_and_historical_cache_rebinding_refuse();Base.setUp(self);self.test_exact_source_or_role_mutation_refuse();Base.setUp(self);self.test_observer_ABI_or_roster_mutation_refuse()
if __name__=='__main__':
    names=('test_actual_ten_overlay_full_qualifier_preserves_historical_maps','test_actual_overlay_association_role_and_historical_overlap_refuse','test_external_passive_receipt_file_rebind_refuses','test_baseline_unknown_path_and_cache_value_string_refuse','test_full_role_source_namespace_main_and_original_ABI_refusals');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);mirror.cleanup();after=pins();receipt={'kind':'INDEPENDENT_GENERIC_IMPORT_RUNTIME_QUALIFIER003_OVERLAY_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_source_pins_unchanged':BEFORE==after,'details':DETAILS,'fixture_HERE_companion_mirror_only':True,'actual_original_map_cache_observer_build_source_profile_schema_helpers_executed':True,'file_descriptor_and_physical_roster_explicit_inert_seams':True,'positive_additional_preload_population_ten_source_derived_rows':True,'own_process_bootstrap_or_psutil_import_not_executed':True,'native_API_or_framework_called':False,'native_gate_or_context_minted':False,'native_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_RUNTIME_QUALIFIER003.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
