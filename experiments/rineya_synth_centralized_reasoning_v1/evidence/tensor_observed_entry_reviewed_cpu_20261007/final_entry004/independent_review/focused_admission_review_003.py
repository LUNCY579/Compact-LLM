"""Actual V2 policy consumer and common installer CPU controls, no native work."""
from pathlib import Path
import copy, hashlib, importlib, json, runpy, sys, tempfile, unittest
from unittest.mock import patch
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
FILES = ('prefix_registered_loader.py', 'prefix_tensor_contract.py', 'test_observed_admission.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before == {'prefix_registered_loader.py':'cbd53d17f08306e671e3d12a41e772b76e977962ab7ece4df2759168c0f3e854','prefix_tensor_contract.py':'fd86efe022e1c2f36b5a431b37e271cedfd4dd1654ffd3d6d7e0890acea8895f','test_observed_admission.py':'c75e846d0dcd8a159e3dc6dfe37fe5ebfa1a692443de5f37feae869563983a2f'}
ns = runpy.run_path(str(WORK / 'test_observed_admission.py'), run_name='independent_observed_fixture')
Base = ns['Controls']
POLICY_COVERAGE = []
class IndependentControls(Base):
    def test_actual_policy_consumer_009_anchor_and_rebound_map_profile_main_refusal(self):
        C = ns['load'](WORK / 'prefix_tensor_contract.py', 'CPU_exact_policy_consumer')
        admission = json.loads((WORK / 'OBSERVED_ADMISSION_001.json').read_bytes())
        original = json.loads(Path(admission['observation_path']).read_bytes())
        runtime = json.loads((C.PACKAGE / 'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json').read_bytes())
        maps, caches, aliases = C.observed_bootstrap_maps(original)
        self.assertEqual(maps, admission['preloaded_bootstrap_modules']); self.assertEqual(caches, admission['preloaded_module_cache_pins']); self.assertEqual(aliases, admission['preloaded_class_aliases'])
        with tempfile.TemporaryDirectory(prefix='CPU_observed_policy_') as folder:
            root = Path(folder).resolve(); main = root / 'prefix_input_owned_worker.py'; main.write_bytes(b'# authored CPU main, not executed\n')
            descriptor = {'bytes': main.stat().st_size, 'sha256': hashlib.sha256(main.read_bytes()).hexdigest(), 'role':'python_source'}
            implementation = 'a' * 40
            review = {'kind':'ROOT_QUALIFIED_FRESH_BOOTSTRAP_BYTE_SOURCE_ASSUMPTIONS','source_commit':implementation,'preloaded_bootstrap_modules_sha256':C.digest(maps),'fresh_process_bootstrap_qualified':True,'environment_and_loader_assumptions_qualified':True,'current_module_execution_proven':False}
            review['sha256'] = C.digest(review); review_path = root / 'authored_review.json'; raw = (C.canonical(review)+'\n').encode('ascii'); review_path.write_bytes(raw)
            policy = {'preloaded_module_cache_pins':caches,'preloaded_bootstrap_modules':maps,'namespace_roots':{},'bootstrap_admission_version':2,'interpreter_profile':original['bootstrap']['interpreter_profile'],'preloaded_class_aliases':aliases,'bootstrap_observation':{'path':admission['observation_path'],'file_sha256':hashlib.sha256(Path(admission['observation_path']).read_bytes()).hexdigest(),'body_sha256':original['sha256']},'tensor_main_source':{'path':str(main),'descriptor':descriptor,'cached_attribute_present':True,'cached_value_kind':'none','cache_path':None,'cache_sha256':None},'fresh_bootstrap_evidence':{'path':str(review_path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':review['sha256']}}
            value = {'config':{'source_files':{str(main):descriptor}},'runtime':{'files':runtime['files'],'stdlib_layout':runtime['stdlib_layout'],'cache_policy':policy},'expected_head':'b'*40,'implementation_commit':implementation}
            with patch.object(C,'PACKAGE',root):
                C._bootstrap_policy(value)
                for target in ('source','profile','main'):
                    changed = copy.deepcopy(value)
                    if target == 'source': changed['runtime']['cache_policy']['preloaded_bootstrap_modules'].pop(next(iter(maps)))
                    elif target == 'profile': changed['runtime']['cache_policy']['interpreter_profile']['flags']['no_site']=0
                    else: changed['runtime']['cache_policy']['tensor_main_source']['path']=str(root/'different_main.py')
                    with self.assertRaises(ValueError): C._bootstrap_policy(changed)
            self.assertNotIn('__main__',maps); self.assertEqual(len(caches),59)
            POLICY_COVERAGE.append({'source_assumptions':len(maps),'cache_pins':len(caches),'aliases':list(aliases),'actual_bootstrap_policy_executed':True,'sealed_fresh_review_authored_CPU_only':True,'fresh_process_qualification_proven':False})
    def test_common_source_package_initializer_child_token_public_remove_restore(self):
        cache = self.surface.path_importer_cache; before_cache = dict(cache)
        with self.active():
            finder = self.install(); token = finder.installed_authority
            module = importlib.import_module('CPU_observed_pkg'); self.assertEqual(module.VALUE,19)
            self.assertTrue(finder.observed['CPU_observed_pkg.child']['current_module_execution_observed'])
            self.assertEqual(finder.CPU_active_registry,{})
            self.L._check_installed_record(finder,self.value,_CPU_capability=self.capC)
            self.L._check_installation_authority(token,self.value,finder=finder,_CPU_capability=self.capC)
            self.L.remove(finder); self.assertIs(self.surface.path_importer_cache,cache); self.assertEqual(cache,before_cache)
            with self.assertRaises(ValueError): self.L._check_installation_authority(token,self.value,_CPU_capability=self.capC)
    def test_actual_aliases_and_separate_main_common_install_then_bad_cache_refusal(self):
        self.test_version2_actual_aliases_separate_inert_main_common_source_install()
        self.assertNotIn('__main__',json.loads((WORK/'OBSERVED_ADMISSION_001.json').read_bytes())['preloaded_bootstrap_modules'])
    def test_foreign_owned_cache_callback_refused_and_retained_without_native_permission(self):
        with self.active():
            finder = self.install(); importlib.import_module('CPU_observed_pkg'); cached=self.surface.path_importer_cache[str(self.site)];called=[]
            cached.find_spec=lambda *args:called.append(True)
            with self.assertRaises(ValueError): finder.find_spec('CPU_foreign')
            self.assertEqual(called,[])
            with self.assertRaises(ValueError): self.L.remove(finder)
            self.assertIs(self.surface.path_importer_cache[str(self.site)],cached)
        with self.assertRaises(ValueError): self.L.install(self.context,self.C)

METHODS=('test_actual_policy_consumer_009_anchor_and_rebound_map_profile_main_refusal','test_common_source_package_initializer_child_token_public_remove_restore','test_actual_aliases_and_separate_main_common_install_then_bad_cache_refusal','test_foreign_owned_cache_callback_refused_and_retained_without_native_permission')
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after={name:hashlib.sha256((WORK/name).read_bytes()).hexdigest() for name in FILES};assert before==after
receipt={'kind':'INDEPENDENT_CPU_OBSERVED_BOOTSTRAP_ADMISSION_REVIEW_003','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_before':before,'source_after':after,'policy_consumer_coverage':POLICY_COVERAGE,'actual_common_package_initializer_child_import':True,'original_installer_token_and_public_remove_cache_restore_exercised':True,'actual_typing_aliases_and_separate_inert_main_exercised':True,'foreign_cache_callbacks_not_executed':True,'native_permission_gates_remain_closed':True,'CPU_source_context_and_review_metadata_authored':True,'actual_future_tensor_main_or_fresh_bootstrap_qualified':False,'startup_ledger_budget_verified_by_this_suite':False,'site_disabled':sys.flags.no_site==1,'ctypes_Torch_DLL_native_API_called':False,'native_PASS':False,'native_authority_issued':False}
receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
out=HERE/'INDEPENDENT_ADMISSION_REVIEW003.json'
with out.open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'file_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'body_sha256':receipt['sha256'],'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'path':str(out)},sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
