"""Independent CPU review; source/roster seams are fixtures, never native evidence."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import ast, hashlib, json, sys, time, unittest

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
PACKAGE = WORK.parents[1]
QUALIFIER = WORK / 'import_runtime_qualification.py'
AUTHORED = WORK / 'test_extended_overlay.py'
PUBLIC = PACKAGE / 'import_runtime_qualification.py'
PRIOR = PACKAGE / 'evidence/psutil_bootstrap_assumptions_reviewed_cpu_20261007/PASSIVE_PSUTIL_BOOTSTRAP_ASSUMPTIONS001.json'
V2 = PACKAGE / 'evidence/import_launch_corrections_reviewed_cpu_20261007/frozen_ram_source_fix002/PASSIVE_IMPORT_BOOTSTRAP_ASSUMPTIONS002.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def module(path, name):
    value = ModuleType(name)
    value.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), value.__dict__)
    return value

Q = module(QUALIFIER, 'independent_extended_qualifier')
O = module(PUBLIC, 'independent_original_qualifier')
F = module(AUTHORED, 'independent_declared_extended_fixture')
pins = {str(p): sha(p) for p in (QUALIFIER, AUTHORED, PUBLIC, PRIOR, V2,
    Q.SOURCE, PACKAGE / 'artifacts/torch_import_observation_work/native_candidate/test_import_runtime_qualification.py',
    PACKAGE / 'artifacts/torch_import_observation_work/native_candidate/import_runtime_qualification.py',
    PACKAGE / 'artifacts/import_exact_registration_work/ACTUAL_SOURCE_RUNTIME_SELECTION003.json')}

class Review(unittest.TestCase):
    def setUp(self):
        self.old = json.loads(PRIOR.read_bytes())
        self.new = json.loads(V2.read_bytes())

    def test_exact_diff_and_original_output(self):
        a, b = ast.parse(PUBLIC.read_bytes()), ast.parse(QUALIFIER.read_bytes())
        changed = {'passive_preload_overlay', '_cache'}
        retained = lambda t: [ast.dump(n, include_attributes=False) for n in t.body
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name not in changed]
        self.assertEqual(retained(a), retained(b))
        old_cache = next(n for n in a.body if isinstance(n, ast.FunctionDef) and n.name == '_cache')
        new_cache = next(n for n in b.body if isinstance(n, ast.FunctionDef) and n.name == '_cache')
        calls = [n for n in ast.walk(new_cache) if isinstance(n, ast.Call) and
            isinstance(n.func, ast.Name) and n.func.id == 'passive_preload_overlay']
        self.assertEqual(len(calls), 1)
        self.assertEqual([k.arg for k in calls[0].keywords], ['helpers', 'budget'])
        calls[0].keywords = []
        self.assertEqual(ast.dump(old_cache, include_attributes=False), ast.dump(new_cache, include_attributes=False))
        self.assertEqual(O.passive_preload_overlay(self.old), Q.passive_preload_overlay(self.old))
        self.assertEqual(len(Q.passive_preload_overlay(self.old)), 10)

    def test_version2_reads_actual_sealed_predecessor_and_only_adds_copy(self):
        self.assertEqual(pins[str(PRIOR)], '4a47c0cf98946ce5f31ace0f358dcf5e39652b2293ea7b97785f51635cebf3ae')
        self.assertEqual(self.old['sha256'], '4e5b197de6a7bca410574991274b238c3679d3d8a572541c026176a6eb6e4aaf')
        helper = Q.original_helpers()
        original = helper.sealed_json
        calls = []
        def recording(path, file_hash, body_hash, **kwargs):
            calls.append((str(path), file_hash, body_hash, kwargs.get('budget')))
            return original(path, file_hash, body_hash, **kwargs)
        helper.sealed_json = recording
        budget = Q.QualificationBudget()
        result = Q.passive_preload_overlay(self.new, helpers=helper, budget=budget)
        self.assertEqual(calls, [(str(PRIOR), pins[str(PRIOR)], self.old['sha256'], budget)])
        base = Q.passive_preload_overlay(self.old, helpers=helper)
        self.assertEqual({k:v for k,v in result.items() if k != 'copy'}, base)
        self.assertEqual(len(result), 11)
        self.assertEqual(result['copy']['descriptor']['sha256'], '27dcfc53a4b9d4fbc3d90c74e549eb6eca9301524d6d2fbff9a6589cf51b6fd5')
        self.assertEqual(result['copy']['cache_descriptor']['sha256'], '9753f133122a6321e83e87753f5f81900beadeb5ffa5c8f0a0883a0d699f18ff')
        self.assertFalse(result['copy']['source_execution_history_proven'])
        self.assertEqual(self.new['source_derived_baseline_expectation'], self.old['source_derived_baseline_expectation'])
        self.assertEqual(len(self.new['source_derived_baseline_expectation']), 39)

    def test_semantic_rebindings_refuse(self):
        cases = []
        def case(name, change):
            value = deepcopy(self.new); change(value); cases.append((name, value))
        case('version bool', lambda v: v.__setitem__('schema_version', True))
        case('kind rebound', lambda v: v.__setitem__('kind', self.old['kind']))
        case('predecessor file', lambda v: v['predecessor_reference'].__setitem__('file_sha256', '0'*64))
        case('predecessor body', lambda v: v['predecessor_reference'].__setitem__('body_sha256', '0'*64))
        case('predecessor path', lambda v: v['predecessor_reference'].__setitem__('path', str(V2)))
        case('source role', lambda v: v['source_modules']['copy']['descriptor'].__setitem__('role', 'extension_binary'))
        case('source history', lambda v: v['source_modules']['copy'].__setitem__('source_execution_history_proven', True))
        case('original source altered', lambda v: v['source_modules']['psutil']['descriptor'].__setitem__('bytes', 1))
        case('original extension altered', lambda v: next(iter(v['extension_modules'].values()))['descriptor'].__setitem__('bytes', 1))
        cp = next(p for p,c in self.new['additional_cache_candidates'].items() if c['source_path'].endswith('copy.py'))
        case('copy cache executed', lambda v: v['additional_cache_candidates'][cp].__setitem__('cache_executed', True))
        case('copy cache hash', lambda v: v['additional_cache_candidates'][cp]['descriptor'].__setitem__('sha256', '0'*64))
        case('copy cache key', lambda v: v['additional_cache_candidates'].__setitem__(cp+'x', v['additional_cache_candidates'].pop(cp)))
        case('copy name', lambda v: v['source_modules'].__setitem__('foreign', v['source_modules'].pop('copy')))
        case('baseline addition', lambda v: v['source_derived_baseline_expectation'].append('C:\\foreign.dll'))
        case('pdh alteration', lambda v: v['new_exact_System32_candidate']['descriptor'].__setitem__('bytes', 1))
        for name, value in cases:
            with self.subTest(name=name), self.assertRaises((ValueError, KeyError)):
                Q.passive_preload_overlay(value)

    def test_actual_full_qualifier_then_historical_and_native_snapshots_refuse(self):
        fixture = F.Controls('test_original_V1_and_full_V2_8725_native2_qualifier')
        fixture.setUp()
        try:
            fixture.extended()
            result = Q._qualify(fixture.value, Q.QualificationBudget(), fixture.B)
            self.assertEqual(result['kind'], 'IMPORT_RUNTIME_BYTE_ROSTER_QUALIFICATION_ONLY')
            self.assertFalse(result['native_effects_or_authority_issued'])
            runtime = fixture.value['runtime']
            self.assertEqual(len(runtime['files']), 8725)
            self.assertEqual(len(fixture.value['config']['source_files']), 196)
            self.assertEqual(len(fixture.value['registration']['git_output_exceptions']), 214)
            self.assertEqual(len(runtime['cache_policy']['preloaded_bootstrap_modules']), 63)
            self.assertEqual(len(runtime['cache_policy']['preloaded_module_cache_pins']), 59)
            self.assertEqual(len(runtime['cache_policy']['additional_preload_sources']), 11)
            original = deepcopy(fixture.value)
            fixture.value['runtime']['cache_policy']['preloaded_bootstrap_modules']['ctypes']['source_sha256'] = '0'*64
            fixture.reseal()
            with self.assertRaisesRegex(ValueError, 'cannot be rebound'):
                Q._qualify(fixture.value, Q.QualificationBudget(), fixture.B)
            fixture.value = deepcopy(original)
            exceptions = fixture.value['registration']['git_output_exceptions']
            key = next(p for p,e in exceptions.items() if e.get('source_path','').endswith('.c'))
            exceptions[key]['source_path'] = str(PACKAGE/'foreign.c')
            fixture.reseal()
            with self.assertRaises(ValueError):
                Q._qualify(fixture.value, Q.QualificationBudget(), fixture.B)
        finally:
            fixture.doCleanups()

if __name__ == '__main__':
    start = time.perf_counter()
    results = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Review))
    after = {path:sha(Path(path)) for path in pins}
    stable = pins == after
    record = {'kind':'INDEPENDENT_CPU_VERSION2_COPY_OVERLAY_REVIEW001', 'tests_run':results.testsRun,
        'failures':len(results.failures), 'errors':len(results.errors), 'elapsed_seconds':time.perf_counter()-start,
        'pins_before':pins, 'pins_after':after, 'pins_stable':stable,
        'candidate_sha256':pins[str(QUALIFIER)], 'public_predecessor_sha256':pins[str(PUBLIC)],
        'independent_review_source_sha256':sha(Path(__file__)), 'semantic_mutations':15,
        'authored_full_qualification_fixture_reused':True,
        'physical_descriptor_and_roster_reads_in_full_qualifier_are_CPU_fixture_seams':True,
        'actual_predecessor_receipt_file_and_body_read':True,
        'native_context_or_authority_issued':False, 'framework_native_API_called':False,
        'current_runtime_bytes_fully_rehashed':False, 'actual_module_execution_history_proven':False}
    record['sha256'] = hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8',newline='\n') as out:
            json.dump(record,out,sort_keys=True,indent=2); out.write('\n')
    raise SystemExit(0 if results.wasSuccessful() and stable else 1)
