from pathlib import Path
from types import ModuleType
from copy import deepcopy
import ast,hashlib,json,unittest
HERE=Path(__file__).resolve().parent
PINNED={'measurement_guard.py':'55bcdff206a8574b6246e5f2c0c047ded4cac3735e1898dc1ec85c32fd01d8d1','measurement_replay.py':'ce8ea36cd113310d35bd09ee223f279bdbf5703b8c7d93ae143325bd9f95124f','test_measurement.py':'09e52ce01f4946beb808418ca1cf5c0e0ca0f20261afa52fe9352a283a7afcf0'}
before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in PINNED};assert before==PINNED
F=ModuleType('root_inert_measurement_fixture');F.__file__=str(HERE/'test_measurement.py')
tree=ast.parse((HERE/'test_measurement.py').read_bytes());tree.body=[n for n in tree.body if not isinstance(n,ast.If)]
exec(compile(tree,F.__file__,'exec'),F.__dict__)
details=[]
class Review(unittest.TestCase):
    def setUp(self):self.fixture=F.Controls();self.fixture.setUp()
    def tearDown(self):self.fixture.tearDown()
    def test_raw_derived_nested_effects_and_exact_map_conservation(self):
        x=self.fixture;x.execute(reverse=True);v=x.replay();framework=next(r for r in v['effects'] if r['boundary']['name']=='framework_import');self.assertEqual(set(framework['own_new_libraries']),{x.paths['pre.dll'],x.paths['direct.dll']});self.assertEqual(set(framework['nested_new_libraries']),{x.paths['one.pyd'],x.paths['two.pyd'],x.paths['nested.dll']});self.assertEqual(v['verified_events'],5);self.assertFalse(v['native_qualification']);details.append({'observed_effect_rows':len(v['effects']),'events':v['verified_events']})
    def test_external_file_and_source_maps_cannot_be_resealed_away(self):
        x=self.fixture;x.execute();raw=x.raw();sources=deepcopy(x.sources);sources[str(HERE/'measurement_guard.py')]['sha256']='0'*64
        with self.assertRaises(ValueError):F.R.validate_measurement(raw,policy=x.policy,grammar=x.grammar,expected_identity=x.identity,selected_file_map=x.files,selected_source_map=sources,expected_file_sha256=hashlib.sha256(raw).hexdigest())
        files=deepcopy(x.files);files.pop(x.paths['nested.dll'])
        with self.assertRaises(ValueError):F.R.validate_measurement(raw,policy=x.policy,grammar=x.grammar,expected_identity=x.identity,selected_file_map=files,selected_source_map=x.sources,expected_file_sha256=hashlib.sha256(raw).hexdigest())
    def test_framework_entry_outside_load_is_rejected(self):
        x=self.fixture;x.P.add(x.paths['pre.dll'])
        with self.assertRaises(ValueError):
            with x.guard.framework_import():pass
        self.assertTrue(x.guard.failed);self.assertFalse(x.guard.export_raw_packet(x.identity)['native_PASS'])
    def test_native_claim_and_producer_effect_forgery_are_rejected(self):
        x=self.fixture;x.execute();original=json.loads(x.raw())
        for k in ('native_PASS','native_authority_issued'):
            v=deepcopy(original);v[k]=True;v.pop('sha256');v['sha256']=F.digest(v)
            with self.assertRaises(ValueError):x.replay((F.canonical(v)+'\n').encode())
        v=deepcopy(original);v['measured_boundary_effects'][-1]['nested_new_libraries']=[];v.pop('sha256');v['sha256']=F.digest(v)
        with self.assertRaises(ValueError):x.replay((F.canonical(v)+'\n').encode())
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Review))
after={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in PINNED};assert after==before
receipt={'kind':'ROOT_INDEPENDENT_MEASUREMENT_RAW_REVIEW001','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_before':before,'source_after':after,'actual_original_inert_guard_and_raw_replayer_executed':True,'fixture_definitions_AST_dispatch_only_excluded':True,'native_API_or_framework_called':False,'native_authority_issued':False,'details':details}
receipt['sha256']=F.digest(receipt);(HERE/'ROOT_INDEPENDENT_MEASUREMENT_REVIEW001.json').write_bytes((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
raise SystemExit(0 if result.wasSuccessful() else 1)
