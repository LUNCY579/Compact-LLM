"""Independent exact typing-alias CPU controls; no native calls or hook dispatch."""
from pathlib import Path
import copy, hashlib, json, runpy, sys, types, unittest, warnings
from unittest.mock import patch
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
NAMES = ('worker.py', 'prelude.py', 'audit.py', 'contract.py', 'test_typing_alias.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in NAMES}
assert before['worker.py'] == '0bec0930d36972e3568816dbe7dea874595b993006956995ac53a6af92956797'
assert before['prelude.py'] == '458764f327d861319f3502774f39d48c58d10fe8e7a476a387f65c3dd1e73742'
assert before['audit.py'] == 'aa91aff76636fbd36d846302b5810d680b93ac56f00a134abcec5650f7d3f92e'
ns = runpy.run_path(str(WORK / 'test_typing_alias.py'), run_name='independent_alias_fixture')
P, C, W, A, Base = ns['P'], ns['C'], ns['W'], ns['A'], ns['Controls']
RECORDS = []
class IndependentControls(Base):
    def test_actual_owner_aliases_ordinary_and_snapshot_rows_no_origin_or_cache(self):
        with warnings.catch_warnings(record=True) as warnings_seen:
            warnings.simplefilter('error', DeprecationWarning)
            ordinary = W._measure_bootstrap_modules(self.value, C, tuple(self.population.items()))
            self.assertEqual(ordinary['unknown_modules'], [])
            A.replay_bootstrap_modules(ordinary, self.value['runtime'], fixture_only=True)
            with patch.object(P.sys, 'modules', dict(self.population)):
                rows = P.module_snapshot(self.layout, self.files, {})
            for row in rows:
                if row['name'] in ('typing.io', 'typing.re'):
                    self.assertEqual(row['classification'], 'stdlib_class_alias')
                    self.assertIsNone(row['origin']); self.assertIsNone(row['source_sha256']); self.assertIsNone(row['cached']); self.assertIsNone(row['error'])
                    self.assertTrue(P.validate_typing_alias_row(row['name'], row['alias_metadata'], self.files))
                    self.assertFalse(row['alias_metadata']['source_execution_history_proven'])
                    RECORDS.append(row['alias_metadata'])
            self.assertEqual(warnings_seen, [])
    def test_counterfeit_alias_or_unknown_class_never_invokes_metaclass_hook(self):
        called = []
        class ForeignMeta(type):
            def __getattribute__(cls, name): called.append(name); raise AssertionError('forbidden metaclass hook')
        class Foreign(metaclass=ForeignMeta): pass
        population = {**self.population, 'typing.io': Foreign, 'other_fileless': Foreign}
        ordinary = W._measure_bootstrap_modules(self.value, C, tuple(population.items()))
        self.assertEqual({row['name'] for row in ordinary['unknown_modules']}, {'typing.io', 'other_fileless'})
        with patch.object(P.sys, 'modules', population): rows = P.module_snapshot(self.layout, self.files, {})
        self.assertTrue(all(next(row for row in rows if row['name'] == name)['error'] for name in ('typing.io', 'other_fileless')))
        self.assertEqual(called, [])
    def test_owner_or_method_export_replacement_and_unselected_bytes_refused(self):
        owner = types.ModuleType('typing'); owner.__dict__.update(vars(ns['typing']))
        population = {**self.population, 'typing': owner}
        with self.assertRaises(ValueError): P.typing_alias_metadata('typing.io', population['typing.io'], population, self.files, self.layout)
        with patch.object(P, 'bounded_bytes', side_effect=AssertionError('unselected owner opened')):
            with self.assertRaises(ValueError): P.typing_alias_metadata('typing.io', self.population['typing.io'], self.population, {}, self.layout)
        alias = self.population['typing.io']; exports = type.__getattribute__(alias, '__dict__'); original = exports['IO']; type.__setattr__(alias, 'IO', object())
        try:
            with self.assertRaises(ValueError): P.typing_alias_metadata('typing.io', alias, self.population, self.files, self.layout)
        finally: type.__setattr__(alias, 'IO', original)
        meta = vars(ns['typing'])['_DeprecatedType']; method = type.__getattribute__(meta, '__dict__')['__getattribute__']; code = method.__code__; method.__code__ = code.replace(co_names=('counterfeit',))
        try:
            with self.assertRaises(ValueError): P.typing_alias_metadata('typing.io', alias, self.population, self.files, self.layout)
        finally: method.__code__ = code
    def test_unregistered_fileless_and_alias_audit_owner_or_exports_rebinding_refused(self):
        unknown = types.ModuleType('unregistered_fileless')
        ordinary = W._measure_bootstrap_modules(self.value, C, tuple({**self.population, 'unregistered_fileless': unknown}.items()))
        self.assertEqual([row['name'] for row in ordinary['unknown_modules']], ['unregistered_fileless'])
        good = W._measure_bootstrap_modules(self.value, C, tuple(self.population.items()))
        for field, value in (('owner_sha256', '0' * 64), ('exports', ['different']), ('source_execution_history_proven', True)):
            bad = copy.deepcopy(good); row = next(row for row in bad['modules'] if row['name'] == 'typing.re'); row['alias_metadata'][field] = value
            with self.assertRaises(ValueError): A.replay_bootstrap_modules(bad, self.value['runtime'], fixture_only=True)

METHODS = ('test_actual_owner_aliases_ordinary_and_snapshot_rows_no_origin_or_cache', 'test_counterfeit_alias_or_unknown_class_never_invokes_metaclass_hook', 'test_owner_or_method_export_replacement_and_unselected_bytes_refused', 'test_unregistered_fileless_and_alias_audit_owner_or_exports_rebinding_refused')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in NAMES}; assert before == after
receipt = {'kind': 'INDEPENDENT_CPU_EXACT_TYPING_ALIAS_REPAIR_REVIEW_V2', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'actual_alias_metadata': RECORDS, 'actual_CPythontyping_alias_objects_used': True, 'counterfeit_metaclass_and_unknown_object_hooks_not_dispatched': True, 'selected_owner_before_open_readtrap_exercised': True, 'source_metaclass_method_and_export_identity_checks_exercised': True, 'ordinary_and_alias_snapshot_null_origin_cache_preserved': True, 'shared_alias_validator_used': True, 'independent_execution_history_proof_claimed': False, 'other_fileless_objects_still_refused': True, 'site_disabled': sys.flags.no_site == 1, 'native_API_ctypes_DLL_framework_tensor_called': False, 'native_PASS': False, 'native_authority_issued': False, 'actual009_measurement_pending': True}
receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
out = HERE / 'INDEPENDENT_REVIEW002.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
