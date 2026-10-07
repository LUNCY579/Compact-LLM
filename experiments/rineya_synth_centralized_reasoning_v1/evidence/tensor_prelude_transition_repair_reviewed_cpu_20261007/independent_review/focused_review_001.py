"""Independent strict initial/owned transition metadata controls; no native calls."""
from pathlib import Path
import ast, copy, hashlib, json, runpy, sys, types, unittest
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
FILES = ('contract.py', 'worker.py', 'prelude.py', 'audit.py', 'supervisor.py', 'launch_driver.py', 'test_transition.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before['worker.py'] == '0b4b3aae9182e2f8b1f820ad3a4e39fafc609d1a5cb4c9648069e1fd1b8edfa0'
assert before['test_transition.py'] == 'f57d66a6db955ce10eda5693f13ca0c3710c9c4cbfd5e26c57f58c036679c90d'
ns = runpy.run_path(str(WORK / 'test_transition.py'), run_name='independent_transition_fixture')
C, W, P, Base = ns['C'], ns['W'], ns['P'], ns['Controls']
TRACE = []
class IndependentControls(Base):
    def test_real_definition_prelude_owned_six_path_profile_and_strict_initial_restore(self):
        original = sys.path; baseline = tuple(original)
        initial, record, transition = self.prelude()
        try:
            observed = W._measure_registered_root_profile(C, self.value, sys, transition, self.plan, initial)
            self.assertEqual(observed['path'], list(baseline) + self.plan['registered_roots'])
            self.assertEqual(initial, self.value['runtime']['cache_policy']['interpreter_profile'])
            self.assertEqual(len(observed['path']), 6)
            with self.assertRaises(ValueError): W._measure_no_site_profile(C, self.value, sys)
            self.assertFalse(observed['source_execution_history_proven'])
            TRACE.append({'initial': initial, 'after': observed, 'roots': self.plan['registered_roots'], 'actual_definition_count': len(record['compiled_helpers'])})
        finally: P.restore_roots(transition)
        self.assertIs(sys.path, original); self.assertEqual(tuple(sys.path), baseline)
    def test_foreign_path_identity_changed_profile_and_root_order_refused(self):
        initial, record, transition = self.prelude()
        try:
            actor = types.SimpleNamespace(**{name: getattr(sys, name) for name in ('executable', 'prefix', 'exec_prefix', 'base_prefix', 'base_exec_prefix', 'meta_path', 'path_hooks')}, path=list(sys.path), flags=types.SimpleNamespace(**initial['flags']))
            with self.assertRaises(ValueError): W._measure_registered_root_profile(C, self.value, actor, transition, self.plan, initial)
            actor.path = sys.path; actor.flags.no_site = 0
            with self.assertRaises(ValueError): W._measure_registered_root_profile(C, self.value, actor, transition, self.plan, initial)
            changed = copy.deepcopy(initial); changed['path'] = list(reversed(changed['path']))
            with self.assertRaises(ValueError): W._measure_registered_root_profile(C, self.value, sys, transition, self.plan, changed)
            changed = copy.deepcopy(self.plan); changed['registered_roots'].reverse()
            with self.assertRaises(ValueError): W._measure_registered_root_profile(C, self.value, sys, transition, changed, initial)
            sys.path.append('CPU_FOREIGN')
            with self.assertRaises(ValueError): W._measure_registered_root_profile(C, self.value, sys, transition, self.plan, initial)
            with self.assertRaises(ValueError): P.restore_roots(transition)
            self.assertEqual(sys.path[-1], 'CPU_FOREIGN'); sys.path.pop()
        finally: P.restore_roots(transition)
    def test_single_worker_delta_shared_identity_checks_and_before_ctypes_route(self):
        for name in FILES[:-1]:
            prior = WORK / 'pre_transition_sources' / name
            if name != 'worker.py': self.assertEqual((WORK / name).read_bytes(), prior.read_bytes())
        current = ast.parse((WORK / 'worker.py').read_bytes())
        initial = next(node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == '_measure_no_site_profile')
        self.assertIn('len(interpreter.path) == 4', ast.unparse(initial))
        bootstrap = next(node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == '_native_bootstrap')
        calls = [node for node in ast.walk(bootstrap) if isinstance(node, ast.Call)]
        second = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == '_measure_registered_root_profile']
        imports = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == '__import__']
        self.assertEqual(len(second), 1); self.assertLess(second[0].lineno, min(node.lineno for node in imports))
        self.assertEqual(len(C.OPS), 12)
        self.assertFalse({'ctypes', '_ctypes', 'torch', 'psutil', 'native_worker'} & set(sys.modules))

METHODS = ('test_real_definition_prelude_owned_six_path_profile_and_strict_initial_restore', 'test_foreign_path_identity_changed_profile_and_root_order_refused', 'test_single_worker_delta_shared_identity_checks_and_before_ctypes_route')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}; assert before == after
receipt = {'kind': 'INDEPENDENT_CPU_PRELUDE_OWNED_ROOT_TRANSITION_REPAIR_REVIEW_V1', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'actual_definition_prelude_and_metadata_transition_trace': TRACE, 'strict_initial_four_path_check_preserved': True, 'owned_six_path_check_separate': True, 'only_worker_changed_from_original007': True, 'native_second_check_before_ctypes_source_order_verified': True, 'site_disabled': sys.flags.no_site == 1, 'ctypes_observer_framework_tensor_native_API_called': False, 'native_PASS': False, 'native_authority_issued': False, 'fresh_native_measurement_pending': True}
receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
out = HERE / 'INDEPENDENT_REVIEW001.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
