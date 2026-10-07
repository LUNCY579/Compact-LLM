"""Independent first-initializer CPU source/frame/cleanup controls."""
from pathlib import Path
import hashlib, importlib, json, runpy, sys, unittest
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
P = WORK.parents[1]
FILES = ('prefix_registered_loader.py', 'prefix_tensor_contract.py', 'finite_bootstrap_helper.py', 'test_active_initializer.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before['prefix_registered_loader.py'] == 'df3ca0b59a740bbda773a0640d0415ea690dd8cc19b913e2d49248d61604dec5'
assert before['test_active_initializer.py'] == '17855f30777da85ffeafd0444616d0a1ec9b6bafeb78976062d944e6dd2ba05c'
ns = runpy.run_path(str(WORK / 'test_active_initializer.py'), run_name='independent_active_fixture')
Base = ns['Controls']
TRACE = []
class IndependentControls(Base):
    def baseline(self):
        objects = (self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache)
        return objects, (tuple(objects[0]), tuple(objects[1]), tuple(objects[2]), dict(objects[3]))
    def assert_restored(self, baseline):
        objects, contents = baseline; now = (self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache)
        for actual, original in zip(now, objects): self.assertIs(actual, original)
        self.assertEqual((tuple(now[0]), tuple(now[1]), tuple(now[2]), dict(now[3])), contents)
    def test_actual_initializer_child_nested_and_circular_cached_parent(self):
        self.set_source('__init__.py', b'from . import child\nfrom . import nested\nVALUE=child.VALUE+nested.VALUE\n')
        self.set_source('child.py', b'import CPU_pkg\nVALUE=2\n')
        self.set_source('nested/__init__.py', b'from . import deep\nVALUE=deep.VALUE\n')
        self.set_source('nested/deep.py', b'import CPU_pkg.nested\nVALUE=4\n')
        baseline = self.baseline()
        with self.active():
            finder = self.install(); token = finder.installed_authority; original = self.C.file_descriptor
            def observe(path, descriptor, **kw):
                for parent, child in (('CPU_pkg', self.pkg / 'child.py'), ('CPU_pkg.nested', self.pkg / 'nested/deep.py')):
                    if path == str(child):
                        state = finder.CPU_active_registry[parent]
                        self.assertIs(state[0], self.surface.modules[parent])
                        self.assertIs(state[4], self.surface.modules[parent].__dict__)
                        self.assertTrue(self.surface.modules[parent].__spec__._initializing)
                        self.assertNotIn(parent, finder.CPU_module_registry)
                        self.assertNotIn(parent, finder.observed)
                        TRACE.append({'parent': parent, 'code_filename': state[3].co_filename, 'source_active': True, 'completion_present': False})
                return original(path, descriptor, **kw)
            self.C.file_descriptor = observe
            parent = importlib.import_module('CPU_pkg'); self.assertEqual(parent.VALUE, 6)
            self.assertEqual(finder.CPU_active_registry, {})
            for name in ('CPU_pkg', 'CPU_pkg.child', 'CPU_pkg.nested', 'CPU_pkg.nested.deep'):
                self.assertTrue(finder.observed[name]['current_module_execution_observed'])
            self.L._check_installed_record(finder, self.value, _CPU_capability=self.capC)
            self.L._check_installation_authority(token, self.value, finder=finder, _CPU_capability=self.capC)
            self.L._CPU_remove_bootstrap_fixture(finder)
        self.assert_restored(baseline)
    def test_failed_initializer_revokes_parent_and_preserves_completed_child(self):
        self.set_source('__init__.py', b'from . import child\nraise RuntimeError("independent initializer failure")\n')
        baseline = self.baseline()
        with self.active():
            finder = self.install()
            with self.assertRaisesRegex(RuntimeError, 'independent initializer failure'): importlib.import_module('CPU_pkg')
            self.assertEqual(finder.CPU_active_registry, {})
            self.assertNotIn('CPU_pkg', finder.CPU_module_registry)
            self.assertNotIn('CPU_pkg', finder.observed)
            self.assertNotIn('CPU_pkg', self.surface.modules)
            self.assertTrue(finder.observed['CPU_pkg.child']['current_module_execution_observed'])
            self.L._CPU_remove_bootstrap_fixture(finder)
        self.assert_restored(baseline)
    def test_active_path_or_parent_bytes_mutation_refused_before_child(self):
        sources = (b'__path__.append("foreign")\nfrom . import child\n', b'with open(__file__,"ab") as f: f.write(b"# drift\\n")\nfrom . import child\n')
        for source in sources:
            self.set_source('__init__.py', source); baseline = self.baseline()
            with self.active():
                finder = self.install()
                with self.assertRaises(ValueError): importlib.import_module('CPU_pkg')
                self.assertEqual(finder.CPU_active_registry, {})
                self.assertNotIn('CPU_pkg.child', self.surface.modules)
                self.assertNotIn('CPU_pkg', finder.observed)
                self.L._CPU_remove_bootstrap_fixture(finder)
            self.assert_restored(baseline)
    def test_repeat_refusal_preserves_existing_completion_without_exec(self):
        self.set_source('__init__.py', b'COUNT=globals().get("COUNT",0)+1\nfrom . import child\n')
        baseline = self.baseline()
        with self.active():
            finder = self.install(); parent = importlib.import_module('CPU_pkg')
            record = finder.CPU_module_registry['CPU_pkg']; observation = dict(finder.observed['CPU_pkg'])
            with self.assertRaisesRegex(ValueError, 'repeated execution is outside'): parent.__loader__.exec_module(parent)
            self.assertEqual(parent.COUNT, 1)
            self.assertIs(finder.CPU_module_registry['CPU_pkg'], record)
            self.assertEqual(finder.observed['CPU_pkg'], observation)
            self.assertEqual(finder.CPU_active_registry, {})
            self.L._CPU_remove_bootstrap_fixture(finder)
        self.assert_restored(baseline)
    def test_forged_metadata_without_current_selected_frame_refused(self):
        baseline = self.baseline()
        with self.active():
            finder = self.install(); parent = importlib.import_module('CPU_pkg')
            original = finder.CPU_module_registry.pop('CPU_pkg'); observation = finder.observed.pop('CPU_pkg')
            parent.__spec__._initializing = True
            code = compile((self.pkg / '__init__.py').read_bytes(), str(self.pkg / '__init__.py'), 'exec')
            finder.CPU_active_registry['CPU_pkg'] = (parent, parent.__loader__, original[2], code, parent.__dict__)
            with self.assertRaisesRegex(ValueError, 'currently active'): finder.find_spec('CPU_pkg.child', parent.__path__)
            finder.CPU_active_registry.clear(); parent.__spec__._initializing = False
            finder.CPU_module_registry['CPU_pkg'] = original; finder.observed['CPU_pkg'] = observation
            self.L._CPU_remove_bootstrap_fixture(finder)
        self.assert_restored(baseline)

METHODS = ('test_actual_initializer_child_nested_and_circular_cached_parent', 'test_failed_initializer_revokes_parent_and_preserves_completed_child', 'test_active_path_or_parent_bytes_mutation_refused_before_child', 'test_repeat_refusal_preserves_existing_completion_without_exec', 'test_forged_metadata_without_current_selected_frame_refused')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before == after
public_sha = hashlib.sha256((P / 'prefix_tensor_contract.py').read_bytes()).hexdigest()
assert public_sha == '93df1a1c1c7eb78a0e3bb0b5f728e140fcc19021bd2980a9680ee78c3760c3d2'
receipt = {'kind': 'INDEPENDENT_CPU_ACTIVE_INITIALIZER_REVIEW_003', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'public_tensor_contract_sha256': public_sha, 'actual_authored_initializer_child_nested_and_circular_imports': True, 'active_incomplete_observation_trace': TRACE, 'active_failure_revocation_exercised': True, 'path_and_current_parent_source_mutation_refused': True, 'repeat_execution_refused_before_source': True, 'original_completion_retained_on_repeat_refusal': True, 'inactive_forged_binding_refused': True, 'original_installer_and_four_container_restore_exercised': True, 'site_disabled': sys.flags.no_site == 1, 'authored_fixture_import_surface_only': True, 'actual_startup_hook_executed': False, 'native_API_DLL_model_called': False, 'native_provider_implemented': False, 'native_PASS': False, 'native_authority_issued': False, 'reload_support_implemented': False, 'scope': 'First verified regular-package initializer and child imports on authored CPU source only; repeat execution and new fileless namespaces remain closed.'}
receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
out = HERE / 'INDEPENDENT_REVIEW_003.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'trace_rows': len(TRACE), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
