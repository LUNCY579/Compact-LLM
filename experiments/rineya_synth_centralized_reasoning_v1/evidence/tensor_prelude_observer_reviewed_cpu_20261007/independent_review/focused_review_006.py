"""Independent CPU selective-read/definition/record validation controls."""
from pathlib import Path
import ast, copy, hashlib, json, marshal, runpy, sys, tempfile, types, unittest
from unittest.mock import patch
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
NAMES = ('contract.py', 'worker.py', 'prelude.py', 'audit.py', 'supervisor.py', 'launch_driver.py', 'test_prelude.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in NAMES}
assert before['worker.py'] == '46af13e71898baa989f6f64ab4dc9fd5cc4320ca9e064c8f6d5bc8101a8d33ad'
assert before['prelude.py'] == 'ae1082ce12a3413e33271d301b584473f050ff93bdb29d758707445897832dc3'
assert before['audit.py'] == '95b3ab6a450fffa3f21635dfe5a24ddd7635dc5ee963b4c810c22fbff1cf7a53'
ns = runpy.run_path(str(WORK / 'test_prelude.py'), run_name='independent_prelude_fixture')
C, P, W, Base = ns['C'], ns['P'], ns['W'], ns['Controls']
OBSERVED = []
class IndependentControls(Base):
    def test_actual_selected_worker_contract_Git_loader_definitions_and_root_restore(self):
        original = sys.path; baseline = tuple(original)
        record, transition, modules = P.source_prelude(self.plan, C, self.value, cpu_only=True)
        try:
            expected = [self.plan['sources']['prefix_input_owned_worker.py']['path'], next(iter(self.plan['helper_sources'])), self.plan['sources']['prefix_registered_loader.py']['path']]
            self.assertEqual([row['file'] for row in record['compiled_helpers']], expected)
            self.assertEqual([row['source'] for row in record['source_costs']], expected)
            self.assertEqual(modules['prefix_input_owned_worker.py'].C.__file__, self.plan['sources']['prefix_tensor_contract.py']['path'])
            with self.assertRaises(ValueError): modules['prefix_input_owned_worker.py'].C.stage_native_gate()
            self.assertFalse(record['tensor_main_executed']); self.assertFalse(record['Torch_called'])
            self.assertIn('importlib.abc', [row['name'] for row in record['modules_after_prelude']])
            OBSERVED.append({'helpers': record['compiled_helpers'], 'costs': record['source_costs'], 'module_rows': len(record['modules_after_prelude']), 'root_transition': record['registered_root_transition']})
        finally: P.restore_roots(transition)
        self.assertIs(sys.path, original); self.assertEqual(tuple(sys.path), baseline)
    def test_unselected_origin_cache_and_transitive_contract_never_opened(self):
        fake = types.ModuleType('CPU_unknown'); fake.__file__ = 'C:/UNSELECTED/private.py'; fake.__cached__ = 'C:/UNSELECTED/private.pyc'
        with patch.object(P.sys, 'modules', {'CPU_unknown': fake}), patch.object(P, 'bounded_bytes', side_effect=AssertionError('unselected bytes opened')):
            rows = P.module_snapshot(self.value['runtime']['stdlib_layout'], {}, {})
            self.assertIsNotNone(rows[0]['error']); self.assertIsNone(rows[0]['source_sha256'])
            cached = P.cache_metadata(fake, fake.__file__, self.value['runtime']['stdlib_layout'], {}, {})
            self.assertIsNotNone(cached['error']); self.assertIsNone(cached['sha256'])
        bad = copy.deepcopy(self.plan); bad['sources']['prefix_tensor_contract.py']['descriptor']['sha256'] = '0' * 64
        with patch.object(P, 'bounded_bytes', side_effect=AssertionError('compile read before transitive admission')):
            with self.assertRaises(ValueError): P.source_prelude(bad, C, self.value, cpu_only=True)
    def test_real_authored_cache_equivalence_and_bad_descriptor_refusal(self):
        with tempfile.TemporaryDirectory(prefix='CPU_prelude_cache_') as folder:
            root = Path(folder).resolve(); source = root / 'inert.py'; source.write_bytes(b'VALUE=7\n'); cache = root / '__pycache__/inert.cpython-311.pyc'; cache.parent.mkdir()
            code = compile(source.read_bytes(), str(source), 'exec'); raw = P.importlib.util.MAGIC_NUMBER + bytes(12) + marshal.dumps(code); cache.write_bytes(raw)
            desc = lambda q, role: {'bytes': q.stat().st_size, 'sha256': hashlib.sha256(q.read_bytes()).hexdigest(), 'role': role}
            files = {str(source): desc(source, 'python_source'), str(cache): desc(cache, 'qualified_bytecode_cache')}
            module = types.ModuleType('CPU_inert'); module.__file__ = str(source); module.__cached__ = str(cache)
            layout = {'logical_base_prefix': str(root / 'logical'), 'physical_base_prefix': str(root / 'physical')}
            result = P.cache_metadata(module, str(source), layout, files, {str(cache): files[str(cache)]})
            self.assertTrue(result['source_equivalence']); self.assertIsNone(result['error'])
            wrong = {str(cache): {**files[str(cache)], 'sha256': '0' * 64}}
            result = P.cache_metadata(module, str(source), layout, files, wrong)
            self.assertIsNotNone(result['error']); self.assertIsNot(result['source_equivalence'], True)
    def test_snapshot_failure_rolls_back_and_primary_cleanup_chain_retains_foreign(self):
        baseline = tuple(sys.path); failure = RuntimeError('independent snapshot failure')
        with patch.object(P, 'module_snapshot', side_effect=failure):
            with self.assertRaises(RuntimeError) as caught: P.source_prelude(self.plan, C, self.value, cpu_only=True)
        self.assertIs(caught.exception, failure); self.assertEqual(tuple(sys.path), baseline)
        def mutate_and_fail(*args, **kw): sys.path.append('CPU_FOREIGN'); raise failure
        try:
            with patch.object(P, 'module_snapshot', side_effect=mutate_and_fail):
                with self.assertRaises(RuntimeError) as caught: P.source_prelude(self.plan, C, self.value, cpu_only=True)
            self.assertIs(caught.exception, failure); self.assertIsInstance(caught.exception.__cause__, ValueError); self.assertEqual(sys.path[-1], 'CPU_FOREIGN')
        finally: sys.path[:] = baseline
    def test_shared_record_replay_rejects_empty_populations_costs_and_cache_errors(self):
        record, transition, modules = P.source_prelude(self.plan, C, self.value, cpu_only=True)
        try:
            # Deliberately authored schema rows; these are not native snapshots.
            minimal = [{'name': name, 'origin': 'built-in', 'source_sha256': None, 'source_bytes': None, 'cached': None, 'cached_attribute_present': False, 'cached_value_kind': 'missing', 'error': None, 'current_execution_history_proven': False} for name in ('sys', '__main__', 'builtins')]
            record['modules_after_prelude'] = copy.deepcopy(minimal); record['modules_after_ctypes'] = copy.deepcopy(minimal); record['modules_at_guard_boundary'] = copy.deepcopy(minimal)
            record['importers_after_ctypes'] = record['after_prelude_importers']; record['importers_at_guard_boundary'] = record['after_prelude_importers']
            ordinary = [{'name': row['name'], 'origin': row['origin'], 'source_sha256': None} for row in minimal]
            files = {**self.value['runtime']['files'], **self.value['config']['source_files']}; caches = {path: d for path, d in files.items() if d['role'] == 'qualified_bytecode_cache'}
            self.assertTrue(P.validate_record(record, self.plan, files, caches, guard_boundary=True, ordinary_modules=ordinary))
            for field in ('modules_after_prelude', 'modules_after_ctypes', 'modules_at_guard_boundary'):
                bad = copy.deepcopy(record); bad[field] = []
                with self.assertRaises(ValueError): P.validate_record(bad, self.plan, files, caches, guard_boundary=True, ordinary_modules=ordinary)
            for field, value in (('compile_returned', 0), ('elapsed_seconds', float('nan'))):
                bad = copy.deepcopy(record); bad['source_costs'][0][field] = value
                with self.assertRaises(ValueError): P.validate_record(bad, self.plan, files, caches, guard_boundary=True, ordinary_modules=ordinary)
            extra = ordinary + [{'name': 'unobserved', 'origin': 'built-in', 'source_sha256': None}]
            with self.assertRaises(ValueError): P.validate_record(record, self.plan, files, caches, guard_boundary=True, ordinary_modules=extra)
            bad = copy.deepcopy(record); bad['modules_at_guard_boundary'][0]['error'] = {'type': 'ValueError'}
            with self.assertRaises(ValueError): P.validate_record(bad, self.plan, files, caches, guard_boundary=True, ordinary_modules=ordinary)
        finally: P.restore_roots(transition)

METHODS = ('test_actual_selected_worker_contract_Git_loader_definitions_and_root_restore', 'test_unselected_origin_cache_and_transitive_contract_never_opened', 'test_real_authored_cache_equivalence_and_bad_descriptor_refusal', 'test_snapshot_failure_rolls_back_and_primary_cleanup_chain_retains_foreign', 'test_shared_record_replay_rejects_empty_populations_costs_and_cache_errors')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in NAMES}; assert before == after
receipt = {'kind': 'INDEPENDENT_CPU_FROZEN_TENSOR_PRELUDE_REVIEW_006', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'actual_definition_execution_observations': OBSERVED, 'source_and_cache_admitted_before_open_read_traps_exercised': True, 'transitive_contract_before_eager_compile_checked': True, 'authored_cache_code_equivalence_exercised_without_execution': True, 'snapshot_failure_root_rollback_and_cleanup_chain_exercised': True, 'shared_validator_replay_controls': True, 'independent_native_cache_or_population_proof': False, 'minimal_population_positive_authored_only': True, 'complete_native_snapshots_claimed': False, 'native_API_ctypes_DLL_framework_tensor_called': False, 'site_disabled': sys.flags.no_site == 1, 'native_authority_issued': False, 'native_PASS': False, 'scope': 'Actual frozen definition execution, selective metadata reads and authored shared-validator envelopes on CPU only.'}
receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
out = HERE / 'INDEPENDENT_REVIEW_006.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
