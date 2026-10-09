"""CPU metadata fixture from actual selected parent observation003; no extension execution."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import copy,hashlib,importlib.machinery,importlib.util,json,sys,unittest
HERE=Path(__file__).resolve().parent;P=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1');OBS=P/'artifacts/import_parent_metadata_work/ACTUAL_PARENT_METADATA003.json';EXPECTED='22f6c6bc2dd1869c66b89cd03a39ab17862c6c1ed7d65e2adcf50aabccaa6b91'
raw=OBS.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED;observed=json.loads(raw);seal=observed.pop('sha256');assert hashlib.sha256(json.dumps(observed,sort_keys=True,separators=(',',':')).encode()).hexdigest()==seal
record=observed['module_rows']['psutil._psutil_windows'];assert record['module_name']=='psutil_windows' and record['spec_name']=='psutil._psutil_windows' and record['verification_passed'] is False
B=ModuleType('CPU_psutil_declared_name_fix');B.__file__=str(HERE/'import_bootstrap.py');exec(compile(Path(B.__file__).read_bytes(),B.__file__,'exec'),B.__dict__)
class Controls(unittest.TestCase):
    def setUp(self):
        self.name='psutil._psutil_windows';self.row=copy.deepcopy(record['selected_row']);self.path=Path(self.row['path']);self.module=ModuleType(record['module_name']);loader=importlib.machinery.ExtensionFileLoader(record['loader_name'],record['loader_path']);spec=importlib.util.spec_from_file_location(record['spec_name'],record['spec_origin'],loader=loader);self.module.__file__=record['module_file'];self.module.__loader__=loader;self.module.__spec__=spec;self.files={self.row['path']:self.row['descriptor']};self.seen=[]
        def fd(p,d,**kw):
            raw=Path(p).read_bytes();self.assertEqual(d,{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'extension_binary'});self.seen.append(str(p))
        self.C=SimpleNamespace(SITE=self.path.parents[1],require=B.__dict__.get('require',lambda v,m: (_ for _ in ()).throw(ValueError(m)) if not v else None),qualified_path=lambda p,**kw:Path(p).resolve(),file_descriptor=fd)
    def check(self):return B.verify_current_module(self.name,self.module,self.row,self.C,self.files,{})
    def test_actual_observed_alias_positive_keeps_full_spec_loader_and_pin(self):
        result=self.check();self.assertEqual(result['path'],record['module_file']);self.assertEqual(result['source_sha256'],self.row['descriptor']['sha256']);self.assertEqual(self.seen,[record['module_file']]);self.assertFalse(result['source_execution_history_proven']);self.assertEqual(self.module.__name__,'psutil_windows');self.assertEqual(self.module.__spec__.name,'psutil._psutil_windows')
    def test_wrong_declared_name_and_namespace_rejected(self):
        for name in ('_psutil_windows','psutil._psutil_windows','evil'):
            self.module.__name__=name
            with self.assertRaisesRegex(ValueError,'module/spec'):self.check()
        self.module.__name__=record['module_name'];self.module.__spec__.name='other._psutil_windows'
        with self.assertRaisesRegex(ValueError,'module/spec'):self.check()
    def test_wrong_descriptor_path_or_role_rejected_before_byte_read(self):
        for field,value in (('sha256','0'*64),('bytes',70655),('role','python_source')):
            row=copy.deepcopy(self.row);self.row['descriptor'][field]=value
            with self.assertRaisesRegex(ValueError,'declared-name binding'):self.check()
            self.row=row
        self.row['path']=str(self.path.parent/'wrong.pyd')
        with self.assertRaisesRegex(ValueError,'declared-name binding'):self.check()
        self.assertEqual(self.seen,[])
    def test_wrong_origin_loader_and_unrelated_alias_rejected(self):
        self.module.__spec__.origin=str(self.path.parent/'wrong.pyd')
        with self.assertRaisesRegex(ValueError,'source/spec'):self.check()
        self.module.__spec__.origin=record['spec_origin'];self.module.__loader__.name='wrong'
        with self.assertRaisesRegex(ValueError,'loader'):self.check()
        self.module.__loader__.name=record['loader_name'];self.name='unrelated._psutil_windows';self.module.__spec__.name=self.name;self.module.__loader__.name=self.name
        with self.assertRaisesRegex(ValueError,'module/spec'):self.check()
    def test_default_module_name_rule_unchanged_for_other_extension(self):
        self.name='_socket';self.module.__name__=self.name;self.module.__spec__.name=self.name;self.module.__loader__.name=self.name;result=self.check();self.assertEqual(result['path'],self.row['path']);self.module.__name__='socket'
        with self.assertRaisesRegex(ValueError,'module/spec'):self.check()
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));record={'kind':'CPU_EXACT_OBSERVED_PSUTIL_DECLARED_NAME_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_observation_full_file_sha256':EXPECTED,'inert_metadata_module_only':True,'extension_framework_API_or_process_called':False,'fixture_only':True,'native_authority_issued':False,'source_pins':{str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (HERE/'import_bootstrap.py',Path(__file__).resolve())}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
