"""Independent CPU helper controls and actual finite bytes; no live import surface."""
from pathlib import Path
import copy, hashlib, importlib.machinery, json, runpy, sys, types, unittest
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
P = WORK.parents[1]
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in ('finite_bootstrap_helper.py', 'test_finite_bootstrap_helper.py', 'prefix_registered_loader.py', 'prefix_tensor_contract.py')}
assert before['finite_bootstrap_helper.py'] == 'f0263bdf122d724e47d42f2a5cb8e9d43ce2e7da153715d64aef0e1308be0539'
assert before['test_finite_bootstrap_helper.py'] == 'b6000fb5f88d94d69fc8adbfa9ef6281700fb51c551fafdb85aae02d9e0f7cad'
ns = runpy.run_path(str(WORK / 'test_finite_bootstrap_helper.py'), run_name='independent_loader')
H, Base = ns['H'], ns['Controls']

class IndependentControls(Base):
    def test_mutable_cached_finder_is_removed_without_execution_and_restored(self):
        calls = []
        old = importlib.machinery.FileFinder(str(self.site))
        old.find_spec = lambda *args: calls.append('foreign cache executed')
        self.surface.path_importer_cache[str(self.site)] = old
        baseline = (self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache)
        lease = self.quarantine()
        self.assertEqual(self.surface.path_importer_cache, {})
        self.assertEqual(calls, [])
        H._CPU_restore(lease, _capability=self.cap)
        self.assertIs(self.surface.path_importer_cache[str(self.site)], old)
        for now, original in zip((self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache), baseline): self.assertIs(now, original)

    def test_bad_path_objects_strings_duplicates_and_file_leaves_refused(self):
        called = []
        class Foreign:
            def __eq__(self, other): called.append(other); return False
        for value in (Foreign(), '', str(self.site), str(self.site / H.HOOK_LEAVES[0])):
            self.surface.path.append(value)
            with self.assertRaises((ValueError, OSError)): self.quarantine()
            self.surface.path.pop()
        self.assertEqual(called, [])

    def test_exception_stack_and_bounded_code_metadata_refused(self):
        fn = self.modules['_virtualenv']._Finder.find_spec
        original = fn.__code__
        for change in ({'co_exceptiontable': b'counterfeit'}, {'co_stacksize': original.co_stacksize + 1}):
            fn.__code__ = original.replace(**change)
            with self.assertRaises(ValueError): self.enroll()
        fn.__code__ = original
        too_deep = 0
        for _ in range(65): too_deep = (too_deep,)
        for constants in ((too_deep,), (1 << 4097,), tuple(range(32769))):
            with self.assertRaises(ValueError): H.code_fingerprint(original.replace(co_consts=constants))

    def test_method_globals_substitution_after_enrollment_refused(self):
        handle = self.enroll()
        cls = self.modules['_virtualenv']._Finder
        original = vars(cls)['find_spec']
        fake = types.FunctionType(original.__code__, dict(original.__globals__), original.__name__)
        fake.__module__ = original.__module__
        cls.find_spec = fake
        with self.assertRaises(ValueError): self.quarantine(handle)

    def test_normal_primary_failure_restores_all_four_original_contents(self):
        original = (self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache)
        contents = (tuple(original[0]), tuple(original[1]), tuple(original[2]), dict(original[3]))
        failure = RuntimeError('independent authored primary failure')
        with self.assertRaises(RuntimeError) as caught:
            with H._CPU_quarantined(self.enroll(), self.policy, self.surface, _capability=self.cap): raise failure
        self.assertIs(caught.exception, failure)
        self.assertIsNone(caught.exception.__cause__)
        for now, prior in zip((self.surface.meta_path, self.surface.path_hooks, self.surface.path, self.surface.path_importer_cache), original): self.assertIs(now, prior)
        self.assertEqual((tuple(original[0]), tuple(original[1]), tuple(original[2]), dict(original[3])), contents)

    def test_concurrent_mutation_retained_and_primary_error_chains_cleanup(self):
        failure = RuntimeError('independent authored primary with foreign mutation')
        addition = object()
        with self.assertRaises(RuntimeError) as caught:
            with H._CPU_quarantined(self.enroll(), self.policy, self.surface, _capability=self.cap):
                self.surface.path_hooks.append(addition)
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertIs(self.surface.path_hooks[-1], addition)
        self.assertNotIn(H.PLACEHOLDER, self.surface.path)
        self.assertEqual(self.surface.path_importer_cache, {})

    def test_missing_namespace_path_and_policy_or_capability_change_refused(self):
        handle = self.enroll()
        original = self.modules['pywin32_system32'].__spec__.submodule_search_locations
        self.modules['pywin32_system32'].__spec__.submodule_search_locations = []
        with self.assertRaises(ValueError): self.quarantine(handle)
        self.modules['pywin32_system32'].__spec__.submodule_search_locations = original
        modified = copy.deepcopy(self.policy); modified['namespace_execution_proven'] = True
        with self.assertRaises(ValueError): H._CPU_quarantine(handle, modified, self.surface, _capability=self.cap)
        with self.assertRaises(ValueError): H._CPU_quarantine(handle, self.policy, self.surface, _capability=object())
        with self.assertRaises(ValueError): H.native_enroll(handle, self.policy, self.surface)

methods = ('test_mutable_cached_finder_is_removed_without_execution_and_restored', 'test_bad_path_objects_strings_duplicates_and_file_leaves_refused', 'test_exception_stack_and_bounded_code_metadata_refused', 'test_method_globals_substitution_after_enrollment_refused', 'test_normal_primary_failure_restores_all_four_original_contents', 'test_concurrent_mutation_retained_and_primary_error_chains_cleanup', 'test_missing_namespace_path_and_policy_or_capability_change_refused')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in methods))
runtime_path = P / 'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_005.json'
runtime_raw = runtime_path.read_bytes()
assert hashlib.sha256(runtime_raw).hexdigest() == 'd76bad7cc062b8b0f6bb32c552112d442b1d215adde8f73ab5e8dab5154f348e'
runtime = json.loads(runtime_raw); body = dict(runtime); seal = body.pop('sha256'); assert H.digest(body) == seal
projection = H.project_policy(runtime, runtime_file_sha256=hashlib.sha256(runtime_raw).hexdigest())
H.verify_policy_bytes(projection)
assert len(projection['startup_files']) == 14 and sum(d['bytes'] for d in projection['startup_files'].values()) == 844093
assert projection['namespace_execution_proven'] is projection['native_authority_issued'] is projection['native_entry_implemented'] is False
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in before}
assert before == after
public = {name: hashlib.sha256((P / name).read_bytes()).hexdigest() for name in ('prefix_tensor_contract.py',)}
assert public['prefix_tensor_contract.py'] == '93df1a1c1c7eb78a0e3bb0b5f728e140fcc19021bd2980a9680ee78c3760c3d2'
receipt = {'kind': 'INDEPENDENT_CPU_FINITE_TENSOR_BOOTSTRAP_HELPER_REVIEW_003', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'public_source_sha256': public, 'observed_runtime_file_sha256': hashlib.sha256(runtime_raw).hexdigest(), 'observed_runtime_body_sha256': seal, 'actual_finite_projection_files': 14, 'actual_finite_projection_bytes': 844093, 'actual_finite_bytes_and_rosters_verified_only': True, 'CPU_identity_objects_authored': True, 'live_startup_identity_objects_observed': False, 'source_fingerprint_execution_proof_claimed': False, 'mutation_policy': 'Unexpected mutation retains additions and quarantine, reports cleanup failure and preserves primary exception; unchanged surfaces restore original four objects and contents.', 'site_disabled': sys.flags.no_site == 1, 'actual_startup_hooks_executed': False, 'live_sys_import_surface_mutated': False, 'native_provider_implemented': False, 'integrated_tensor_bridge_implemented': False, 'native_API_or_DLL_called': False, 'native_PASS': False, 'native_authority_issued': False, 'namespace_execution_proven': False, 'followup_native_obligation': 'Fresh separately authorized owned-process observation and independently reviewed live enrollment/loader integration remain required.'}
receipt['sha256'] = H.digest(receipt)
out = HERE / 'INDEPENDENT_REVIEW_003.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
