"""Independent authored CPU integration checks, no hooks/native libraries/models."""
from pathlib import Path
import copy, hashlib, importlib, importlib.util, json, marshal, runpy, sys, unittest
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
P = WORK.parents[1]
FILES = ('prefix_registered_loader.py', 'prefix_tensor_contract.py', 'finite_bootstrap_helper.py', 'test_bootstrap_integration.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before == {'prefix_registered_loader.py': 'cf814c0c597934cea9a5f3ff1dda39f0fc1dd3145cadc998205fd0558a17e0cd', 'prefix_tensor_contract.py': 'a51442898930c99f2d62e13344832878ec8677b157c676898acfacba7685821d', 'finite_bootstrap_helper.py': 'f0263bdf122d724e47d42f2a5cb8e9d43ce2e7da153715d64aef0e1308be0539', 'test_bootstrap_integration.py': '2323e4ca5438dcd509ea94cdfd7026dc3751a67d1bbdd53db04c755e6eaa402d'}
ns = runpy.run_path(str(WORK / 'test_bootstrap_integration.py'), run_name='independent_fixture_loader')
Base, H = ns['Controls'], ns['H']

class IndependentControls(Base):
    def baseline(self):
        return (self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache), (tuple(self.surface.meta_path), tuple(self.surface.path_hooks), tuple(self.surface.path), dict(self.surface.path_importer_cache))
    def assert_restored(self, prior):
        objects, contents = prior
        current = (self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache)
        for now, old in zip(current, objects): self.assertIs(now, old)
        self.assertEqual((tuple(current[0]), tuple(current[1]), tuple(current[2]), dict(current[3])), contents)
    def test_actual_inert_import_token_checks_and_four_container_restore(self):
        prior = self.baseline()
        with self.active():
            finder = self.install(); token = finder.installed_authority
            self.L._check_installed_record(finder, self.value, _CPU_capability=self.capC)
            self.L._check_installation_authority(token, self.value, finder=finder, _CPU_capability=self.capC)
            imported = importlib.import_module('CPU_inert')
            self.assertEqual(imported.VALUE, 37)
            self.assertIs(type(imported.__loader__), self.L.RegisteredLoader)
            self.assertTrue(finder.observed['CPU_inert']['current_module_execution_observed'])
            self.assertFalse(finder.observed['pywin32_system32']['current_module_execution_observed'])
            self.L._CPU_remove_bootstrap_fixture(finder)
            with self.assertRaises(ValueError): self.L._check_installation_authority(token, self.value, _CPU_capability=self.capC)
        self.assert_restored(prior)
    def test_immediate_capture_precedes_callback_mutation_and_refuses_cleanup(self):
        with self.active():
            finder = self.install(); original = self.C.file_descriptor; observations = []
            def mutate(path, descriptor, **kw):
                result = original(path, descriptor, **kw)
                if path == str(self.target):
                    cache = finder.CPU_startup_lease[4]
                    self.assertTrue(cache)
                    key, captured = next(iter(cache.items()))
                    actual = self.surface.path_importer_cache[key]
                    self.assertIs(captured[0], actual)
                    self.assertGreater(len(captured[1][2]), 0)
                    actual._loaders = []; observations.append(actual)
                return result
            self.C.file_descriptor = mutate
            importlib.import_module('CPU_inert')
            self.assertTrue(observations)
            with self.assertRaises(ValueError): self.L._CPU_remove_bootstrap_fixture(finder)
            self.assertTrue(any(item is observations[0] for item in self.surface.path_importer_cache.values()))
    def test_loader_state_and_replaced_installer_resolver_refused(self):
        with self.active():
            finder = self.install(); importlib.import_module('CPU_inert')
            key, entry = next(iter(finder.CPU_startup_lease[4].items())); cached = entry[0]
            original = list(cached._loaders); cached._loaders[0] = ('.counterfeit', original[0][1])
            with self.assertRaises(ValueError): finder.find_spec('unseen')
            cached._loaders = original
            method = finder._find_spec_body; finder._find_spec_body = lambda *args: None
            with self.assertRaises(ValueError): self.L._check_installed_record(finder, self.value, _CPU_capability=self.capC)
            finder._find_spec_body = method
            self.L._CPU_remove_bootstrap_fixture(finder)
    def test_forged_overlay_and_changed_context_refused_before_mutation(self):
        prior = self.baseline()
        with self.active():
            with self.assertRaises(ValueError): self.install(overlay=object())
            changed = copy.deepcopy(self.value); changed['config']['changed'] = True
            self.context = self.C._mint_native_context(changed)
            with self.assertRaises(ValueError): self.install()
        self.assert_restored(prior)
    def test_wrong_bootstrap_pyc_equivalence_restores_normal_install_failure(self):
        prior = self.baseline(); name = '_virtualenv'; source = Path(self.policy['modules'][name]['path'])
        cache = source.parent / '__pycache__' / (source.stem + '.cpython-311.pyc'); cache.parent.mkdir()
        raw = importlib.util.MAGIC_NUMBER + bytes(12) + marshal.dumps(compile('FOREIGN=True\n', str(source), 'exec'))
        cache.write_bytes(raw); desc = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'role': 'qualified_bytecode_cache'}
        self.files[str(cache)] = desc
        self.value['runtime']['cache_policy']['preloaded_module_cache_pins'][str(cache)] = desc
        self.value['runtime']['cache_policy']['preloaded_bootstrap_modules'][name].update(cache_path=str(cache), cache_sha256=desc['sha256'])
        self.modules[name].__cached__ = str(cache)
        self.context = self.C._mint_native_context(self.value)
        self.overlay = self.C._CPU_prepare_startup_overlay(self.value, self.policy, _capability=self.capC)
        with self.active():
            with self.assertRaisesRegex(ValueError, 'cache code matches'): self.install()
        self.assert_restored(prior)
    def test_post_creation_install_failure_restores_and_preserves_primary(self):
        prior = self.baseline(); original = self.L._remember_installation; calls = []
        primary = RuntimeError('independent installation-record failure after finder created')
        def remember(*args, **kw):
            calls.append(True)
            if len(calls) == 2: raise primary
            return original(*args, **kw)
        self.L._remember_installation = remember
        with self.active():
            with self.assertRaises(RuntimeError) as caught: self.install()
        self.assertIs(caught.exception, primary)
        self.assertIsNone(caught.exception.__cause__)
        self.assert_restored(prior)
    def test_post_creation_failure_foreign_cache_retained_and_cleanup_chained(self):
        original = self.L._remember_installation; calls = []; primary = RuntimeError('independent primary plus foreign cache')
        def remember(*args, **kw):
            calls.append(True)
            if len(calls) == 2:
                self.surface.path_importer_cache['foreign'] = None
                raise primary
            return original(*args, **kw)
        self.L._remember_installation = remember
        with self.active():
            with self.assertRaises(RuntimeError) as caught: self.install()
        self.assertIs(caught.exception, primary)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertIn('foreign', self.surface.path_importer_cache)
        self.assertNotIn(H.PLACEHOLDER, self.surface.path)

METHODS = ('test_actual_inert_import_token_checks_and_four_container_restore', 'test_immediate_capture_precedes_callback_mutation_and_refuses_cleanup', 'test_loader_state_and_replaced_installer_resolver_refused', 'test_forged_overlay_and_changed_context_refused_before_mutation', 'test_wrong_bootstrap_pyc_equivalence_restores_normal_install_failure', 'test_post_creation_install_failure_restores_and_preserves_primary', 'test_post_creation_failure_foreign_cache_retained_and_cleanup_chained')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before == after
public_sha = hashlib.sha256((P / 'prefix_tensor_contract.py').read_bytes()).hexdigest()
assert public_sha == '93df1a1c1c7eb78a0e3bb0b5f728e140fcc19021bd2980a9680ee78c3760c3d2'
receipt = {'kind': 'INDEPENDENT_CPU_TENSOR_BOOTSTRAP_INTEGRATION_REVIEW_003', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'public_tensor_contract_sha256': public_sha, 'actual_inert_registered_source_import_exercised': True, 'original_installation_record_and_token_exercised': True, 'fixture_source_context_injected': True, 'CPU_process_import_containers_temporarily_replaced_by_authored_fixture': True, 'actual_site_hooks_executed': False, 'native_API_DLL_model_called': False, 'native_provider_implemented': False, 'native_tensor_entry_implemented': False, 'site_disabled': sys.flags.no_site == 1, 'native_PASS': False, 'native_authority_issued': False, 'bootstrap_execution_inferred': False, 'scope_limit': 'Authored CPU fixtures and real inert registered-source execution; no live startup identity or native readiness evidence.', 'separate_unresolved_git_registration_cycle': True}
receipt['sha256'] = H.digest(receipt)
out = HERE / 'INDEPENDENT_REVIEW_003.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
