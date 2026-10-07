"""Independent regular-package CPU successor controls, no native work."""
from pathlib import Path
import hashlib, importlib, json, runpy, sys, unittest
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
P = WORK.parents[1]
FILES = ('prefix_registered_loader.py', 'prefix_tensor_contract.py', 'finite_bootstrap_helper.py', 'test_package_paths.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before == {'prefix_registered_loader.py': '47f8633b2b1d8ca153eba84e9ee3572ed15f5ec38210258ed0eca2cbc3b29f06', 'prefix_tensor_contract.py': 'a51442898930c99f2d62e13344832878ec8677b157c676898acfacba7685821d', 'finite_bootstrap_helper.py': 'f0263bdf122d724e47d42f2a5cb8e9d43ce2e7da153715d64aef0e1308be0539', 'test_package_paths.py': 'eb85a1c80ed6b8b79702e3312e9b2c9abdba58328413ded2fde47eaa4b84bb77'}
ns = runpy.run_path(str(WORK / 'test_package_paths.py'), run_name='independent_package_fixture')
Base = ns['Controls']
class IndependentControls(Base):
    def test_actual_regular_package_children_nested_execution_tokens_and_restore(self):
        objects = (self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache)
        contents = (tuple(objects[0]), tuple(objects[1]), tuple(objects[2]), dict(objects[3]))
        with self.active():
            finder = self.install(); token = finder.installed_authority
            self.assertEqual(importlib.import_module('CPU_pkg.nested.deep').VALUE, 4)
            self.assertEqual(importlib.import_module('CPU_pkg.child').VALUE, 2)
            for name in ('CPU_pkg', 'CPU_pkg.child', 'CPU_pkg.nested', 'CPU_pkg.nested.deep'):
                module = self.surface.modules[name]
                self.assertIs(type(module.__loader__), self.L.RegisteredLoader)
                self.assertTrue(finder.observed[name]['current_module_execution_observed'])
                self.assertEqual(finder.observed[name]['sha256'], self.files[module.__file__]['sha256'])
            self.assertEqual(set(finder.CPU_startup_lease[4]), {str(self.site), str(self.pkg), str(self.pkg / 'nested')})
            self.L._check_installed_record(finder, self.value, _CPU_capability=self.capC)
            self.L._check_installation_authority(token, self.value, finder=finder, _CPU_capability=self.capC)
            self.L._CPU_remove_bootstrap_fixture(finder)
            with self.assertRaises(ValueError): self.L._check_installation_authority(token, self.value, _CPU_capability=self.capC)
        current = (self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache)
        for now, old in zip(current, objects): self.assertIs(now, old)
        self.assertEqual((tuple(current[0]), tuple(current[1]), tuple(current[2]), dict(current[3])), contents)
    def test_coherent_parent_rebinding_current_source_and_execution_record_refused(self):
        with self.active():
            finder = self.install(); parent = importlib.import_module('CPU_pkg'); loader = parent.__loader__
            original = (loader.path, loader.descriptor, parent.__file__, parent.__spec__.origin, tuple(parent.__path__), tuple(parent.__spec__.submodule_search_locations))
            replacement = self.pkg / 'nested/__init__.py'
            loader.path = str(replacement); loader.descriptor = self.files[str(replacement)]
            parent.__file__ = str(replacement); parent.__spec__.origin = str(replacement)
            parent.__path__[:] = [str(replacement.parent)]; parent.__spec__.submodule_search_locations[:] = parent.__path__
            with self.assertRaises(ValueError): finder.find_spec('CPU_pkg.child', parent.__path__)
            loader.path, loader.descriptor, parent.__file__, parent.__spec__.origin = original[:4]
            parent.__path__[:] = original[4]; parent.__spec__.submodule_search_locations[:] = original[5]
            source = self.pkg / '__init__.py'; raw = source.read_bytes(); source.write_bytes(b'VALUE=9\n')
            with self.assertRaises(ValueError): finder.find_spec('CPU_pkg.child', parent.__path__)
            source.write_bytes(raw)
            finder.observed['CPU_pkg']['current_module_execution_observed'] = False
            with self.assertRaises(ValueError): finder.find_spec('CPU_pkg.child', parent.__path__)
            finder.observed['CPU_pkg']['current_module_execution_observed'] = True
            self.L._CPU_remove_bootstrap_fixture(finder)
    def test_foreign_package_cache_key_refused_and_retained_on_cleanup_failure(self):
        with self.active():
            finder = self.install(); importlib.import_module('CPU_pkg.child')
            foreign = self.site / 'foreign_cache'; foreign.mkdir()
            self.surface.path_importer_cache[str(foreign)] = None
            with self.assertRaises(ValueError): finder.find_spec('CPU_pkg.next', [str(self.pkg)])
            with self.assertRaises(ValueError): self.L._CPU_remove_bootstrap_fixture(finder)
            self.assertIn(str(foreign), self.surface.path_importer_cache)
            self.assertNotIn(finder, self.surface.meta_path)
    def test_dotted_none_and_even_selected_new_fileless_namespace_refused(self):
        namespace = self.site / 'CPU_selected_ns'; namespace.mkdir()
        leaf = namespace / 'leaf.py'; leaf.write_bytes(b'VALUE=5\n')
        self.files[str(leaf)] = ns['B'].B.desc(leaf, 'python_source')
        self.value['runtime']['cache_policy']['namespace_roots']['CPU_selected_ns'] = [str(namespace)]
        self.surface.modules['sys'] = sys
        self.context = self.C._mint_native_context(self.value)
        self.overlay = self.C._CPU_prepare_startup_overlay(self.value, self.policy, _capability=self.capC)
        with self.active():
            finder = self.install(); importlib.import_module('CPU_pkg')
            with self.assertRaisesRegex(ValueError, 'dotted source resolution requires parent path proof'): finder.find_spec('CPU_pkg.child', None)
            with self.assertRaisesRegex(ValueError, 'no new fileless namespace enrollment'): importlib.import_module('CPU_selected_ns')
            self.assertNotIn('CPU_selected_ns', self.surface.modules)
            self.L._CPU_remove_bootstrap_fixture(finder)

METHODS = ('test_actual_regular_package_children_nested_execution_tokens_and_restore', 'test_coherent_parent_rebinding_current_source_and_execution_record_refused', 'test_foreign_package_cache_key_refused_and_retained_on_cleanup_failure', 'test_dotted_none_and_even_selected_new_fileless_namespace_refused')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before == after
public_sha = hashlib.sha256((P / 'prefix_tensor_contract.py').read_bytes()).hexdigest()
assert public_sha == '93df1a1c1c7eb78a0e3bb0b5f728e140fcc19021bd2980a9680ee78c3760c3d2'
receipt = {'kind': 'INDEPENDENT_CPU_REGULAR_PACKAGE_PATH_REVIEW_006', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'public_tensor_contract_sha256': public_sha, 'actual_authored_regular_package_child_nested_deep_execution': True, 'original_installer_record_and_authority_checks_exercised': True, 'four_container_identity_and_content_restore_exercised': True, 'immutable_parent_binding_and_current_source_checks_exercised': True, 'foreign_cache_retained_on_cleanup_failure': True, 'new_fileless_namespaces_explicitly_refused': True, 'only_preloaded_pywin32_namespace_supported': True, 'site_disabled': sys.flags.no_site == 1, 'CPU_process_import_containers_temporarily_replaced_by_authored_fixture': True, 'actual_site_hooks_executed': False, 'native_API_model_DLL_called': False, 'native_provider_implemented': False, 'native_PASS': False, 'native_authority_issued': False, 'future_Torch_import_readiness_claimed': False, 'scope': 'Regular authored CPU package/submodule/nested package imports; fresh fileless namespace support remains closed.', 'original_top_level_integration_receipts_preserved': True, 'old_eight_control_regression_rerun_claimed': False}
receipt['sha256'] = ns['B'].H.digest(receipt)
out = HERE / 'INDEPENDENT_REVIEW_006.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
