"""Independent authored-only observer startup V2 negatives. Run -I -S -B."""
import runpy, pathlib, ast, copy, unittest, hashlib, json, subprocess, sys
P = pathlib.Path(__file__).resolve().parents[3]
PUBLIC = P / 'observer_startup_measurement_v1'
NAMES = ('contract.py', 'worker.py', 'audit.py', 'prefix_tensor_contract.py')

def sources():
    return {name: hashlib.sha256(((P / name) if name == 'prefix_tensor_contract.py' else (PUBLIC / name)).read_bytes()).hexdigest() for name in NAMES}

before = sources()
ns = runpy.run_path(str(P / 'artifacts/observer_bootstrap_retry_preparation_work/test_bootstrap_v2.py'), run_name='independent_loader')
C, W, A, Base = ns['C'], ns['W'], ns['A'], ns['Controls']

class IndependentControls(Base):
    def test_location_flag_and_missing_path_refused(self):
        m = self.namespace(); m.__spec__.has_location = True; self.assert_denied_measure(m)
        m = self.namespace(); del m.__path__; self.assert_denied_measure(m)
        m = self.namespace(); m.__spec__.submodule_search_locations = None; self.assert_denied_measure(m)

    def test_namespace_origin_metadata_refused(self):
        m = self.namespace(); m.__spec__.origin = str(self.ns / 'pythoncom311.dll'); self.assert_denied_measure(m)
        m = self.namespace(); m.__file__ = ''; self.assert_denied_measure(m)

    def test_missing_dll_and_missing_pth_refused(self):
        pathlib.Path(next(iter(self.nsfiles))).unlink(); self.assert_denied_measure()
        pathlib.Path(next(iter(self.hooks))).unlink()
        with self.assertRaises((OSError, ValueError)):
            C._directory_roster(self.site, self.hooks, suffix='.pth')

    def test_finite_roles_refuse_extra_namespace_and_widened_directory(self):
        self.cache['allowed_namespace_packages']['other'] = copy.deepcopy(self.entry)
        with self.assertRaises(ValueError): C._startup_admitted_roles(self.config, self.runtime)
        self.cache['allowed_namespace_packages'].pop('other')
        self.cache['startup_hook_sources']['directory'] = str(self.base)
        with self.assertRaises(ValueError): C._startup_admitted_roles(self.config, self.runtime)

    def test_auditor_refuses_path_disagreement_duplicate_name_and_authority(self):
        good = self.measure()
        bad = copy.deepcopy(good); bad['modules'][0]['module_search_locations'] = [str(self.base)]
        with self.assertRaises(ValueError): A.replay_bootstrap_modules(bad, self.runtime, fixture_only=True)
        bad = copy.deepcopy(good); bad['modules'].append(copy.deepcopy(bad['modules'][0]))
        with self.assertRaises(ValueError): A.replay_bootstrap_modules(bad, self.runtime, fixture_only=True)
        bad = copy.deepcopy(good); bad['namespace_execution_proven'] = True
        with self.assertRaises(ValueError): A.replay_bootstrap_modules(bad, self.runtime, fixture_only=True)

    def test_frozen_constants_unchanged(self):
        old = subprocess.run(['git', 'show', 'a9e62a0342c80c0a1b14cdd397a8369a582d93b9:experiments/rineya_synth_centralized_reasoning_v1/observer_startup_measurement_v1/contract.py'], capture_output=True, check=True).stdout
        constants = {}
        for node in ast.parse(old).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {'SCOPE', 'PERMISSION', 'RESOURCES', 'OPS', 'ABI'}:
                        constants[target.id] = ast.literal_eval(node.value)
        for name, value in constants.items(): self.assertEqual(value, getattr(C, name))

METHODS = ('test_location_flag_and_missing_path_refused', 'test_namespace_origin_metadata_refused', 'test_missing_dll_and_missing_pth_refused', 'test_finite_roles_refuse_extra_namespace_and_widened_directory', 'test_auditor_refuses_path_disagreement_duplicate_name_and_authority', 'test_frozen_constants_unchanged')
r = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = sources()
receipt = {'kind': 'INDEPENDENT_CPU_OBSERVER_BOOTSTRAP_V2_FOCUSED_REVIEW_002', 'tests_run': r.testsRun, 'failures': len(r.failures), 'errors': len(r.errors), 'source_before': before, 'source_after': after, 'source_unchanged_during_review': before == after, 'site_disabled': sys.flags.no_site == 1, 'explicit_native_imports_trapped_before_candidate_import': True, 'scope': C.SCOPE, 'permission': C.PERMISSION, 'resources': C.RESOURCES, 'ABI': C.ABI, 'operations': C.OPS, 'envelope_AST_equal_to_original': 'test_frozen_constants_unchanged' in METHODS and not r.failures and not r.errors, 'shared_prefix_tensor_sha256': after['prefix_tensor_contract.py'], 'native_run': False, 'native_PASS': False, 'native_authority_issued': False, 'namespace_execution_proven': False, 'prior_unaccepted_review': {'tool_chunk_id': '4a5a10', 'tests_run': 6, 'failures': 0, 'errors': 1, 'safe_refusal': 'FileNotFoundError for missing .pth', 'review_harness_error': 'Expected only ValueError; corrected to (OSError, ValueError)', 'implementation_repaired_for_harness_error': False}}
receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
if len(sys.argv) > 1:
    with pathlib.Path(sys.argv[1]).open('x', encoding='utf-8') as f:
        json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
else: print(json.dumps(receipt, sort_keys=True))
raise SystemExit(0 if r.wasSuccessful() and before == after else 1)
