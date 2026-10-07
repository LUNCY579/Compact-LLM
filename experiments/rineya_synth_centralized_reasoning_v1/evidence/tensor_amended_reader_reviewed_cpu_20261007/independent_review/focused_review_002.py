"""Independent complete amended-reader CPU execution with explicit provider seams."""
from pathlib import Path
import hashlib, json, runpy, sys, unittest
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
P = WORK.parents[1]
FILES = ('prefix_tensor_contract.py', 'registration_git_join.py', 'test_full_amended_reader.py', 'outer_context_fixture.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before['prefix_tensor_contract.py'] == '610fb451315de7c79fa9b9b968926f3a16bf7f65329845e350284a9092a363cc'
assert before['registration_git_join.py'] == '04eff7f510e16820f71ff4c371454a9048b5309fea8b8a18cc9918a65e473bdc'
assert before['test_full_amended_reader.py'] == 'd9278dc4d8e23d0d0b4c242543818930798c6d91bddf1bc84f7c1ca18a04ed5b'
dependency = P / 'artifacts/prefix_input_owned_execution_work/outer_context_fixture.py'
assert dependency.read_bytes() == (WORK / 'outer_context_fixture.py').read_bytes()
ns = runpy.run_path(str(WORK / 'test_full_amended_reader.py'), run_name='independent_amended_reader_fixture')
Base = ns['Controls']
COVERAGE = []
class IndependentControls(Base):
    def test_complete_actual_reader_metadata_rosters_cache_descriptors_and_git(self):
        context = self.read(); value = self.C.context_value(context)
        self.assertEqual(value['implementation_commit'], self.implementation)
        self.assertEqual(value['expected_head'], self.head)
        self.assertNotEqual(self.implementation, self.head)
        expected = {'native_metadata': 1, '_CPU_amended_metadata': 1, 'qualification_rosters': 2, 'physical_roster': 14, '_bootstrap_policy': 2, 'load_input': 1}
        self.assertEqual(self.actual_functions, expected)
        self.assertEqual(self.fixture.git_calls, [])
        self.assertEqual(value['CPU_git_join']['dispatch_head'], self.head)
        self.assertEqual(value['CPU_git_join']['implementation_commit'], self.implementation)
        self.assertEqual(value['CPU_git_join']['source_inventory_sha256'], self.C.digest(self.fixture.sources))
        self.assertTrue(value['fixture_only'])
        self.assertFalse(value['native_PASS']); self.assertFalse(value['native_authority_issued'])
        self.assertFalse(value['CPU_git_join']['native_runtime_qualified'])
        self.assertEqual(len(self.fixture.actual_file_checks), 70)
        self.assertEqual(len(set(self.fixture.actual_file_checks)), 35)
        self.assertEqual(len(self.fixture.historical_runtime_checks), 1684)
        self.assertEqual(len(set(self.fixture.historical_runtime_checks)), 842)
        COVERAGE.append({'function_calls': dict(self.actual_functions), 'actual_descriptor_calls': 70, 'actual_descriptor_paths': 35, 'injected_historical_descriptor_calls': 1684, 'injected_historical_descriptor_paths': 842, 'source_map_files': len(self.fixture.sources), 'runtime_map_files': len(self.fixture.runtime['files']), 'fixture_Git_mock_calls': [], 'real_implementation_commit': self.implementation, 'real_dispatch_commit': self.head, 'returned_context_status': value['status']})
    def test_unrostered_package_source_refused_by_actual_roster(self):
        extra = self.fixture.package / 'extra_unregistered.py'
        extra.write_bytes(b'# authored inert extra source; never executed\n')
        with self.assertRaisesRegex(ValueError, 'exact complete current package/tests/transitive source roster'): self.read()
        self.assertGreater(self.actual_functions.get('qualification_rosters', 0), 0)
    def test_resealed_bootstrap_review_bound_to_dispatch_refused(self):
        f = self.fixture
        f.bootstrap = f._seal({**f.bootstrap, 'source_commit': self.head})
        f.runtime['cache_policy']['fresh_bootstrap_evidence'] = f.anchor(f.bootstrap_path, f.bootstrap)
        f.runtime = f._seal(f.runtime)
        f.review = f._seal({**f.review, 'cache_policy_sha256': self.C.digest(f.runtime['cache_policy'])})
        f.registration = f._seal({**f.registration, 'runtime_body_sha256': f.runtime['sha256'], 'qualification_review_file_sha256': f.anchor(f.review_path, f.review)['file_sha256'], 'qualification_review_body_sha256': f.review['sha256']})
        self.rewrite_dispatch()
        with self.assertRaisesRegex(ValueError, 'external bootstrap assumptions'): self.read()
        self.assertGreater(self.actual_functions.get('_bootstrap_policy', 0), 0)
    def test_inert_runtime_drift_and_invalid_capability_or_native_entry_refused(self):
        f = self.fixture; path = f.site / 'typing_extensions.py'; raw = path.read_bytes()
        path.write_bytes(b'# changed authored runtime bytes\n')
        with self.assertRaises(ValueError): self.read()
        path.write_bytes(raw)
        with self.active():
            with self.assertRaises(ValueError): self.C._CPU_read_amended_manifest(f.config_path, _capability=object(), **f.anchors)
        with self.assertRaises(ValueError): self.C.native_entry(None)

METHODS = ('test_complete_actual_reader_metadata_rosters_cache_descriptors_and_git', 'test_unrostered_package_source_refused_by_actual_roster', 'test_resealed_bootstrap_review_bound_to_dispatch_refused', 'test_inert_runtime_drift_and_invalid_capability_or_native_entry_refused')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before == after
assert dependency.read_bytes() == (WORK / 'outer_context_fixture.py').read_bytes()
public_sha = hashlib.sha256((P / 'prefix_tensor_contract.py').read_bytes()).hexdigest()
assert public_sha == '93df1a1c1c7eb78a0e3bb0b5f728e140fcc19021bd2980a9680ee78c3760c3d2'
receipt = {'kind': 'INDEPENDENT_COMPLETE_AMENDED_READER_CPU_REVIEW_002', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'actual_fixture_dependency_path': str(dependency), 'actual_fixture_dependency_sha256': hashlib.sha256(dependency.read_bytes()).hexdigest(), 'public_tensor_contract_sha256': public_sha, 'completed_positive_coverage': COVERAGE, 'actual_validation_functions_executed': True, 'validator_return_values_replaced_by_true': False, 'Git_subprocess_output_injected': False, 'injected_provider_seams': ['Historical65/stdlib779 exact descriptors and roles: metadata obligations only, bytes not read', 'Authored finite SITE and interpreter layout', 'Preserved exposed diagnostic input and source metadata copied into fixture; supplied metadata retained'], 'site_disabled': sys.flags.no_site == 1, 'actual_runtime_startup_modules_imported': False, 'native_API_DLL_model_called': False, 'actual_host_runtime_qualified': False, 'current_module_execution_proven': False, 'native_entry_implemented': False, 'native_PASS': False, 'native_authority_issued': False, 'scope': 'Complete CPU amended reader on inert runtime files, copied supplied metadata, actual source/roster checks and real temporary Git commits only.'}
receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
out = HERE / 'INDEPENDENT_REVIEW_002.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
