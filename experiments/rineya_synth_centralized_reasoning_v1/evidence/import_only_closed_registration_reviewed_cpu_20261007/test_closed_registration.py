"""Closed packet construction on selected byte maps; no native permission."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import hashlib,json,tempfile,unittest,sys
class NativeTrap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker','native_process','native_nvml'):
            raise AssertionError('native/framework import forbidden '+fullname)
sys.meta_path.insert(0,NativeTrap())
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
path=HERE/'closed_registration.py';M=ModuleType('CPU_import_registration');M.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),M.__dict__)
inputs_module=ModuleType('CPU_program002_registration_inputs');inputs_module.__file__=str(HERE/'program002_inputs.py');exec(compile((HERE/'program002_inputs.py').read_bytes(),inputs_module.__file__,'exec'),inputs_module.__dict__)
C=inputs_module.module(inputs_module.PROGRAM/'import_observation_contract.py','CPU_exact_program002_contract')
def selected(relative):return json.loads((PACKAGE/relative).read_bytes())
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_import_closed_');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.inputs=inputs_module.inputs();self.runtime=self.inputs['runtime_candidate'];self.rosters=self.inputs['roster_candidate'];self.observation=self.inputs['observation'];self.initial=self.inputs['initial35_expectation'];self.sources=self.inputs['source_files'];self.main=self.inputs['main_source'];self.resources=self.inputs['resources']
    def build(self):return M.build(**self.inputs,implementation_commit=None,dispatch_head=None,packet_paths={n:str(self.root/(n+'.json')) for n in ('config','runtime','registration')})
    def validate_actual(self,bundle):
        value=C.validate_import_metadata(bundle['config'],bundle['runtime'],bundle['registration'],bundle['external_dispatch_head']);M.validate(bundle);return value
    def test_closed_source_runtime_cache_and_nine_external_anchor_joins(self):
        bundle=self.build();v=self.validate_actual(bundle);self.assertFalse(v['native_authority_issued']);self.assertEqual(len(bundle['runtime']['files']),8716);self.assertEqual(len(bundle['runtime']['cache_policy']['preloaded_module_cache_pins']),59);self.assertEqual(len(bundle['runtime']['initial35_expectation']),35);self.assertEqual(C.OPERATIONS,M.OPERATIONS)
        raw={name:(M.canonical(bundle[name])+'\n').encode() for name in ('config','runtime','registration')}
        with self.assertRaisesRegex(ValueError,'pending'):M.anchors(bundle,raw)
    def test_resealed_source_main_rebinding_refused(self):
        bundle=self.build();bundle['config']['main_source']=str(self.root/'other.py');bundle['config']=M.seal({k:v for k,v in bundle['config'].items() if k!='sha256'});bundle['registration']['config_body_sha256']=bundle['config']['sha256'];bundle['registration']=M.seal({k:v for k,v in bundle['registration'].items() if k!='sha256'})
        with self.assertRaises((ValueError,KeyError)):self.validate_actual(bundle)
    def test_runtime_cache_omission_and_forged_permission_refused(self):
        self.runtime['files'].pop(next(iter(self.observation['maps']['preloaded_module_cache_pins'])));self.runtime=M.seal({k:v for k,v in self.runtime.items() if k!='sha256'})
        with self.assertRaises(ValueError):self.build()
        self.setUp();bundle=self.build();bundle['config']['native_launch_authorized']=True;bundle['config']=M.seal({k:v for k,v in bundle['config'].items() if k!='sha256'})
        with self.assertRaises(ValueError):self.validate_actual(bundle)
        bundle=self.build();selection=deepcopy(bundle['selected_inputs']);bundle['runtime']['cache_policy']['preloaded_bootstrap_modules'].pop(next(iter(bundle['runtime']['cache_policy']['preloaded_bootstrap_modules'])));bundle['runtime']=M.seal({k:v for k,v in bundle['runtime'].items() if k!='sha256'});bundle['registration']['runtime_body_sha256']=bundle['runtime']['sha256'];bundle['registration']=M.seal({k:v for k,v in bundle['registration'].items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'external selected'):M.validate(bundle,expected_selection=selection)
        with self.assertRaises(ValueError):C.validate_import_metadata(bundle['config'],bundle['runtime'],bundle['registration'],'invalid')
    def test_changed_packet_bytes_dispatch_cycle_and_native_factory_refuse(self):
        bundle=self.build();raw={n:(M.canonical(bundle[n])+'\n').encode() for n in ('config','runtime','registration')};raw['config']+=b' '
        with self.assertRaises(ValueError):M.anchors(bundle,raw)
        bundle['external_dispatch_head']='a'*40
        with self.assertRaises(ValueError):self.validate_actual(bundle)
        with self.assertRaises(ValueError):M.native_context(bundle)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_CLOSED_IMPORT_OBSERVATION_REGISTRATION_CONSTRUCTION_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_program_source_graph_not_yet_frozen':False,'inert_main_source_fixture':False,'actual_program002_metadata_validator_called':True,'actual_commits_pending_None':True,'native_API_framework_called':False,'native_launch_authorized':False,'native_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'closed_registration.py',Path(__file__).resolve(),HERE/'program002_inputs.py',inputs_module.PROGRAM/'import_observation_contract.py')}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
