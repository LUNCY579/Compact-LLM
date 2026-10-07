"""Known interpreter-alias comparisons and raw ownership, CPU fixture only."""
from pathlib import Path
import hashlib, importlib, importlib.machinery, json, runpy, sys, unittest
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
FILES=('prefix_registered_loader.py','prefix_tensor_contract.py','test_observed_admission.py')
before={name:hashlib.sha256((WORK/name).read_bytes()).hexdigest() for name in FILES}
assert before=={'prefix_registered_loader.py':'eb195f3c7cda8abc6180d3290ab02ea3ac40aab2fa71658203018a2788a91266','prefix_tensor_contract.py':'fd86efe022e1c2f36b5a431b37e271cedfd4dd1654ffd3d6d7e0890acea8895f','test_observed_admission.py':'f26bfffc730708d9a0240eb3ca1150d38c311a56a7681ea747f73aef60edb3d2'}
ns=runpy.run_path(str(WORK/'test_observed_admission.py'),run_name='independent_known_alias_fixture');Base=ns['Controls']
class IndependentControls(Base):
    def test_actual_alias_parent_source_resolution_child_and_raw_mutation_refusal(self):
        self.test_exact_declared_logical_alias_bootstrap_parent_child_and_raw_mutation()
    def test_regular_active_parent_and_raw_owned_cache_still_restore(self):
        cache=self.surface.path_importer_cache;baseline=dict(cache)
        with self.active():
            finder=self.install();token=finder.installed_authority;module=importlib.import_module('CPU_observed_pkg');self.assertEqual(module.VALUE,19)
            self.assertTrue(finder.observed['CPU_observed_pkg.child']['current_module_execution_observed']);self.assertEqual(finder.CPU_active_registry,{})
            self.L._check_installation_authority(token,self.value,finder=finder,_CPU_capability=self.capC);self.L.remove(finder)
            self.assertIs(self.surface.path_importer_cache,cache);self.assertEqual(cache,baseline)
    def test_foreign_cache_container_callbacks_and_native_route_remain_refused(self):
        with self.active():
            finder=self.install();original=sys.path_importer_cache;calls=[];foreign=importlib.machinery.FileFinder(str(self.site));foreign.find_spec=lambda *args:calls.append(True);replacement={str(self.site):foreign};sys.path_importer_cache=replacement
            try:
                with self.assertRaises(ValueError):finder.find_spec('CPU_unknown')
                self.assertEqual(calls,[])
                with self.assertRaises(ValueError):self.L.remove(finder)
                self.assertIs(sys.path_importer_cache,replacement)
            finally:sys.path_importer_cache=original
        with self.assertRaises(ValueError):self.L.install(self.context,self.C)
METHODS=('test_actual_alias_parent_source_resolution_child_and_raw_mutation_refusal','test_regular_active_parent_and_raw_owned_cache_still_restore','test_foreign_cache_container_callbacks_and_native_route_remain_refused')
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS));after={name:hashlib.sha256((WORK/name).read_bytes()).hexdigest() for name in FILES};assert before==after
receipt={'kind':'INDEPENDENT_CPU_DECLARED_INTERPRETER_ALIAS_ADMISSION_REVIEW_007','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_before':before,'source_after':after,'actual_source_bytes_PathFinder_child_execution_checked':True,'known_alias_resolve_and_reparse_metadata_injected_CPU_fixture':True,'actual_live_junction_identity_proven':False,'raw_parent_spec_path_and_cache_ownership_preserved':True,'original_installer_and_public_remove_exercised':True,'foreign_cache_callbacks_not_called':True,'earlier_reviews_retained':['INDEPENDENT_ADMISSION_REVIEW003.json','INDEPENDENT_ADMISSION_REVIEW005.json'],'ledger_or_native_provider_whole_registration_qualified':False,'site_disabled':sys.flags.no_site==1,'ctypes_Torch_DLL_native_API_called':False,'native_PASS':False,'native_authority_issued':False}
receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();out=HERE/'INDEPENDENT_ADMISSION_REVIEW007.json'
with out.open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'file_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'body_sha256':receipt['sha256'],'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'path':str(out)},sort_keys=True));raise SystemExit(0 if result.wasSuccessful() else 1)
