"""Independent preloaded-parent/cache-container successor checks, CPU only."""
from pathlib import Path
import hashlib, importlib, importlib.machinery, json, runpy, sys, types, unittest
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
FILES=('prefix_registered_loader.py','prefix_tensor_contract.py','test_observed_admission.py')
before={name:hashlib.sha256((WORK/name).read_bytes()).hexdigest() for name in FILES}
assert before=={'prefix_registered_loader.py':'d96eb4b0b713f4b41a703f760530ec4e3b00ab8f281d219cf98c579f2482723f','prefix_tensor_contract.py':'fd86efe022e1c2f36b5a431b37e271cedfd4dd1654ffd3d6d7e0890acea8895f','test_observed_admission.py':'dd9f38ceefec0af626daf9ba71d1ba98e54c8b65eb9d5b8db3e41786911d84bf'}
ns=runpy.run_path(str(WORK/'test_observed_admission.py'),run_name='independent_common_successor');Base=ns['Controls']
class IndependentControls(Base):
    def preload(self):
        root=self.site/'CPU_bootstrap_parent';root.mkdir();init=root/'__init__.py';init.write_bytes(b'# authored bootstrap package\n');child=root/'child.py';child.write_bytes(b'VALUE=31\n')
        for path in (init,child):self.files[str(path)]=ns['B'].B.desc(path,'python_source')
        module=types.ModuleType('CPU_bootstrap_parent');module.__file__=str(init);module.__cached__=None;loader=importlib.machinery.SourceFileLoader(module.__name__,str(init));module.__loader__=loader;module.__path__=[str(root)];module.__spec__=importlib.machinery.ModuleSpec(module.__name__,loader,origin=str(init),is_package=True);module.__spec__.submodule_search_locations=module.__path__
        self.surface.modules[module.__name__]=module;self.value['runtime']['cache_policy']['preloaded_bootstrap_modules'][module.__name__]={'path':str(init),'source_sha256':self.files[str(init)]['sha256'],'cache_path':None,'cache_sha256':None};self.context=self.C._mint_native_context(self.value)
        return module,root
    def test_preloaded_parent_new_child_retains_assumption_not_completed_execution(self):
        parent,root=self.preload();cache=self.surface.path_importer_cache;baseline=dict(cache)
        with self.active():
            finder=self.install();token=finder.installed_authority;child=importlib.import_module('CPU_bootstrap_parent.child');self.assertEqual(child.VALUE,31)
            self.assertFalse(finder.observed[parent.__name__]['current_module_execution_observed']);self.assertNotIn(parent.__name__,finder.CPU_module_registry)
            self.assertTrue(finder.observed['CPU_bootstrap_parent.child']['current_module_execution_observed'])
            self.L._check_installation_authority(token,self.value,finder=finder,_CPU_capability=self.capC);self.L.remove(finder)
            self.assertIs(self.surface.path_importer_cache,cache);self.assertEqual(cache,baseline)
    def test_bootstrap_parent_path_or_source_mutation_refused_then_clean_removed(self):
        parent,root=self.preload()
        with self.active():
            finder=self.install();parent.__path__.append(str(self.site))
            with self.assertRaises(ValueError):finder.find_spec('CPU_bootstrap_parent.child',parent.__path__)
            parent.__path__.pop();source=Path(parent.__file__);raw=source.read_bytes();source.write_bytes(b'# changed authored parent\n')
            with self.assertRaises(ValueError):finder.find_spec('CPU_bootstrap_parent.child',parent.__path__)
            source.write_bytes(raw);self.L.remove(finder)
    def test_foreign_cache_container_replacement_never_delegates_or_clobbers(self):
        with self.active():
            finder=self.install();original=sys.path_importer_cache;called=[];foreign=importlib.machinery.FileFinder(str(self.site));foreign.find_spec=lambda *args:called.append(True);replacement={str(self.site):foreign};sys.path_importer_cache=replacement
            try:
                with self.assertRaises(ValueError):finder.find_spec('CPU_unselected')
                self.assertEqual(called,[])
                with self.assertRaises(ValueError):self.L.remove(finder)
                self.assertIs(sys.path_importer_cache,replacement);self.assertIs(replacement[str(self.site)],foreign)
            finally:sys.path_importer_cache=original
    def test_active_initializer_common_path_and_native_gate_remain_closed(self):
        self.test_common_installer_actual_package_init_child_original_checker_public_remove()
        with self.assertRaises(ValueError):self.L.install(self.context,self.C)
METHODS=('test_preloaded_parent_new_child_retains_assumption_not_completed_execution','test_bootstrap_parent_path_or_source_mutation_refused_then_clean_removed','test_foreign_cache_container_replacement_never_delegates_or_clobbers','test_active_initializer_common_path_and_native_gate_remain_closed')
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS));after={name:hashlib.sha256((WORK/name).read_bytes()).hexdigest() for name in FILES};assert before==after
receipt={'kind':'INDEPENDENT_CPU_COMMON_IMPORTER_SUCCESSOR_REVIEW_005','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_before':before,'source_after':after,'actual_inert_preloaded_parent_child_and_active_initializer_executed':True,'bootstrap_parent_execution_history_not_promoted':True,'original_opaque_installer_token_and_public_remove_exercised':True,'foreign_cache_container_callbacks_not_called':True,'foreign_cache_replacement_retained_on_cleanup_failure':True,'previous_actual_V2_policy_consumer_review_retained':'INDEPENDENT_ADMISSION_REVIEW003.json','startup_ledger_native_provider_or_whole_registration_qualified':False,'site_disabled':sys.flags.no_site==1,'ctypes_Torch_DLL_native_API_called':False,'native_PASS':False,'native_authority_issued':False}
receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest();out=HERE/'INDEPENDENT_ADMISSION_REVIEW005.json'
with out.open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'file_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'body_sha256':receipt['sha256'],'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'path':str(out)},sort_keys=True));raise SystemExit(0 if result.wasSuccessful() else 1)
