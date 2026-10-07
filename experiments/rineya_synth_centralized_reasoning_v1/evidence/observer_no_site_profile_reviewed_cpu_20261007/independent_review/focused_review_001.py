"""Independent real no-site metadata and inert source-derived supervisor checks."""
from pathlib import Path
import ast, copy, hashlib, json, runpy, sys, types, unittest
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
FILES = ('contract.py', 'worker.py', 'supervisor.py', 'audit.py', 'launch_driver.py', 'test_no_site_profile.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
expected = {'contract.py': '007f29152f991a00e9cae15befc5d392e5c86794ab70477082f9ec7cfa305d44', 'worker.py': '0e85e80e20ee32e1dc54bdb9c177f38c30f16af8dac09e68906005081030f27e', 'supervisor.py': 'b9dc99119c5d882740d8c2d1286fb80510e59a13bdaa28cb5251caf5cadddcce', 'audit.py': '7f3e78570a2587fa02716c7c43f55806c2d2ef99d717dd034e72873d3f888845', 'launch_driver.py': 'b6713151d2438b4ebd9fab3edba20b9db705205f1ca973bae8b193044e7448b4', 'test_no_site_profile.py': '721c824a1bf941502b9167f5415ef5417027e26434df78058bc702b02b26c53e'}
assert before == expected
ns = runpy.run_path(str(WORK / 'test_no_site_profile.py'), run_name='independent_no_site_fixture')
C, W, S, A, Base = ns['C'], ns['W'], ns['S'], ns['A'], ns['Controls']
OBSERVATIONS = []
class IndependentControls(Base):
    def test_actual_sys_profile_and_frozen_permission_resource_ABI_operations(self):
        observed = W._measure_no_site_profile(C, self.value, sys)
        self.assertEqual(observed, C.expected_no_site_profile())
        self.assertFalse(observed['source_execution_history_proven'])
        self.assertFalse({'ctypes', '_ctypes', 'torch', 'psutil', 'pywin32_bootstrap', 'pywin32_system32'} & set(sys.modules))
        old = ast.parse((WORK / 'pre_no_site_sources/contract.py').read_bytes())
        constants = {}
        for node in old.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {'SCOPE', 'PERMISSION', 'RESOURCES', 'ABI', 'OPS'}: constants[target.id] = ast.literal_eval(node.value)
        for name, value in constants.items(): self.assertEqual(getattr(C, name), value)
        self.assertEqual(C.OPS, W.OPS); self.assertEqual(C.OPS, A.OPS)
        OBSERVATIONS.append(observed)
    def test_changed_typed_flags_prefixes_and_ordered_path_refused(self):
        for flag in self.profile['flags']:
            actor = self.interpreter(); setattr(actor.flags, flag, 0)
            with self.assertRaises(ValueError): W._measure_no_site_profile(C, self.value, actor)
        actor = self.interpreter(); actor.flags.no_site = True
        with self.assertRaises(ValueError): W._measure_no_site_profile(C, self.value, actor)
        for key in ('executable', 'prefix', 'exec_prefix', 'base_prefix', 'base_exec_prefix'):
            actor = self.interpreter(); setattr(actor, key, 'different')
            with self.assertRaises(ValueError): W._measure_no_site_profile(C, self.value, actor)
        actor = self.interpreter(); actor.path[0], actor.path[1] = actor.path[1], actor.path[0]
        with self.assertRaises(ValueError): W._measure_no_site_profile(C, self.value, actor)
    def test_metadata_name_spoof_and_standard_hook_closure_change_refused(self):
        actor = self.interpreter(); actor.meta_path[0] = types.SimpleNamespace(__module__='_frozen_importlib', __name__='BuiltinImporter')
        with self.assertRaises(ValueError): W._measure_no_site_profile(C, self.value, actor)
        actor = self.interpreter(); original = actor.path_hooks[1]
        def cell(value): return (lambda: value).__closure__[0]
        details = list(original.__closure__[1].cell_contents)
        details[1] = (details[1][0], ['.counterfeit'])
        actor.path_hooks[1] = types.FunctionType(original.__code__, original.__globals__, original.__name__, closure=(original.__closure__[0], cell(tuple(details))))
        with self.assertRaises(ValueError): W._measure_no_site_profile(C, self.value, actor)
    def test_real_original_body_no_site_argv_indices_cleanup_and_legacy(self):
        result, host, value, output = self.run_supervisor(C.NO_SITE_MODE)
        self.assertEqual(result['source_adaptation']['mutation_count'], 6)
        self.assertEqual(result['source_adaptation']['no_site_argv_insertions'], 1)
        self.assertEqual(result['source_adaptation']['argv_index_shifts'], 2)
        self.assertEqual(host.argv, [self.profile['executable'], '-I', '-S', '-B', value['config']['worker_path'], '--config', value['config_path'], '--output-dir', str(output)])
        self.assertEqual(result['receipt']['bounds'], S.BOUNDS)
        self.assertTrue(result['cleanup_verified'])
        self.assertLess(host.calls.index('Job_assigned'), host.calls.index('resume'))
        A.replay_worker_command(result['receipt'], value, output)
        legacy, old_host, old_value, old_output = self.run_supervisor('MEASURE_BYTE_VERIFIED_STARTUP_V1')
        self.assertEqual(legacy['source_adaptation']['mutation_count'], 3)
        self.assertEqual(old_host.argv[1:3], ['-I', '-B'])
        self.assertEqual(old_host.argv[3], old_value['config']['worker_path'])
        self.assertEqual(old_host.argv[5], old_value['config_path'])
    def test_auditor_missing_or_rebound_profile_and_wrong_worker_command_refused(self):
        runtime = {'cache_policy': self.cache, 'files': {}, 'stdlib_layout': {}}
        bootstrap = {'kind': 'MEASURED_BOOTSTRAP_IMPORTS_V1', 'imports': self.cache['imports'], 'modules': [], 'unknown_modules': [], 'fixture_only': True, 'namespace_execution_proven': False, 'interpreter_profile': self.profile}
        A.replay_bootstrap_modules(bootstrap, runtime, fixture_only=True)
        for altered in ('missing', 'execution', 'flags'):
            bad = copy.deepcopy(bootstrap)
            if altered == 'missing': bad.pop('interpreter_profile')
            elif altered == 'execution': bad['interpreter_profile']['source_execution_history_proven'] = True
            else: bad['interpreter_profile']['flags']['no_site'] = 0
            with self.assertRaises(ValueError): A.replay_bootstrap_modules(bad, runtime, fixture_only=True)
        result, host, value, output = self.run_supervisor(C.NO_SITE_MODE)
        for index, replacement in ((2, '-B'), (4, 'foreign_worker.py'), (6, 'foreign_config.json')):
            bad = copy.deepcopy(result['receipt']); bad['attempted_worker_argv'][index] = replacement
            with self.assertRaises(ValueError): A.replay_worker_command(bad, value, output)
    def test_actual_function_order_precedes_ctypes_and_driver_native_helpers(self):
        def function_tree(name, function):
            tree = ast.parse((WORK / name).read_bytes()); return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
        tree = function_tree('worker.py', '_native_bootstrap')
        profile_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_measure_no_site_profile']
        import_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '__import__' and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in ('ctypes', 'ctypes.wintypes', '_ctypes')]
        self.assertEqual(len(profile_lines), 1); self.assertEqual(len(import_lines), 3); self.assertLess(profile_lines[0], min(import_lines))
        tree = function_tree('launch_driver.py', 'run_registered')
        profile_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == '_measure_no_site_profile']
        begin_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'begin']
        self.assertEqual(len(profile_lines), 1); self.assertEqual(len(begin_lines), 1); self.assertLess(profile_lines[0], begin_lines[0])

METHODS = ('test_actual_sys_profile_and_frozen_permission_resource_ABI_operations', 'test_changed_typed_flags_prefixes_and_ordered_path_refused', 'test_metadata_name_spoof_and_standard_hook_closure_change_refused', 'test_real_original_body_no_site_argv_indices_cleanup_and_legacy', 'test_auditor_missing_or_rebound_profile_and_wrong_worker_command_refused', 'test_actual_function_order_precedes_ctypes_and_driver_native_helpers')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}; assert before == after
receipt = {'kind': 'INDEPENDENT_CPU_OBSERVER_NO_SITE_PROFILE_REVIEW_001', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'actual_metadata_only_profile': OBSERVATIONS, 'original_supervisor_body_executed_with_inert_host': True, 'one_no_site_flag_two_argv_index_shifts_verified': True, 'legacy_three_changes_retained': True, 'permission_resources_ABI_operations_equal_predecessor': True, 'profile_before_ctypes_and_driver_native_helpers_source_order_verified': True, 'site_disabled': sys.flags.no_site == 1, 'ctypes_or_path_hook_callbacks_invoked': False, 'native_API_DLL_model_called': False, 'native_runtime_qualified': False, 'native_PASS': False, 'native_authority_issued': False, 'source_execution_history_proven': False, 'scope': 'Actual no-site interpreter metadata and original lifecycle on inert CPU host; prospective observer profile only.'}
receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
out = HERE / 'INDEPENDENT_REVIEW_001.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
