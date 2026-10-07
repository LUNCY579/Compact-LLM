"""Closed packet construction on selected byte maps; no native permission."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import hashlib,json,tempfile,unittest,sys
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
path=HERE/'closed_registration.py';M=ModuleType('CPU_import_registration');M.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),M.__dict__)
def selected(relative):return json.loads((PACKAGE/relative).read_bytes())
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_import_closed_');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();source=self.root/'import_observation_worker.py';raw=b'# inert future scope fixture\n';source.write_bytes(raw);self.main=str(source);self.sources={self.main:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}}
        self.runtime=selected('artifacts/library_expectation_registration_work/ADDITIVE_WINDOWS_RUNTIME_CANDIDATE001.json');self.rosters=selected('artifacts/runtime_refresh_work/TENSOR_RUNTIME_BYTE_CANDIDATE001.json');proposal=selected('artifacts/tensor_two_tensor_proposal_work/TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2_001.json');self.observation=M.seal(deepcopy(proposal['observed009_selection']));self.initial=selected('artifacts/library_expectation_registration_work/LIBRARY_EXPECTATION_INPUTS001.json')['expected_initial_loaded_libraries'];self.resources=selected('configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json')['resources']
    def build(self):return M.build(runtime_candidate=self.runtime,roster_candidate=self.rosters,observation=self.observation,initial35_expectation=self.initial,source_files=self.sources,main_source=self.main,resources=self.resources,implementation_commit='a'*40,dispatch_head='b'*40,packet_paths={n:str(self.root/(n+'.json')) for n in ('config','runtime','registration')},metadata_references={'runtime_basis':self.runtime['basis']})
    def test_closed_source_runtime_cache_and_nine_external_anchor_joins(self):
        bundle=self.build();v=M.validate(bundle);self.assertFalse(v['native_launch_authorized']);self.assertEqual(len(bundle['runtime']['files']),8716);self.assertEqual(len(bundle['runtime']['cache_policy']['preloaded_module_cache_pins']),59);self.assertEqual(len(bundle['runtime']['initial35_expectation']),35)
        raw={name:(M.canonical(bundle[name])+'\n').encode() for name in ('config','runtime','registration')};anchors=M.anchors(bundle,raw);self.assertEqual(len(anchors),9);self.assertEqual(anchors['expected_head'],'b'*40)
    def test_resealed_source_main_rebinding_refused(self):
        bundle=self.build();bundle['config']['main_source']=str(self.root/'other.py');bundle['config']=M.seal({k:v for k,v in bundle['config'].items() if k!='sha256'});bundle['registration']['config_body_sha256']=bundle['config']['sha256'];bundle['registration']=M.seal({k:v for k,v in bundle['registration'].items() if k!='sha256'})
        with self.assertRaises((ValueError,KeyError)):M.validate(bundle)
    def test_runtime_cache_omission_and_forged_permission_refused(self):
        self.runtime['files'].pop(next(iter(self.observation['maps']['preloaded_module_cache_pins'])));self.runtime=M.seal({k:v for k,v in self.runtime.items() if k!='sha256'})
        with self.assertRaises(ValueError):self.build()
        self.setUp();bundle=self.build();bundle['config']['native_launch_authorized']=True;bundle['config']=M.seal({k:v for k,v in bundle['config'].items() if k!='sha256'})
        with self.assertRaises(ValueError):M.validate(bundle)
        bundle=self.build();selection=deepcopy(bundle['selected_inputs']);bundle['runtime']['cache_policy']['preloaded_bootstrap_modules'].pop(next(iter(bundle['runtime']['cache_policy']['preloaded_bootstrap_modules'])));bundle['runtime']=M.seal({k:v for k,v in bundle['runtime'].items() if k!='sha256'});bundle['registration']['runtime_body_sha256']=bundle['runtime']['sha256'];bundle['registration']=M.seal({k:v for k,v in bundle['registration'].items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'external selected'):M.validate(bundle,expected_selection=selection)
    def test_changed_packet_bytes_dispatch_cycle_and_native_factory_refuse(self):
        bundle=self.build();raw={n:(M.canonical(bundle[n])+'\n').encode() for n in ('config','runtime','registration')};raw['config']+=b' '
        with self.assertRaises(ValueError):M.anchors(bundle,raw)
        bundle['external_dispatch_head']='a'*40
        with self.assertRaises(ValueError):M.validate(bundle)
        with self.assertRaises(ValueError):M.native_context(bundle)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_CLOSED_IMPORT_OBSERVATION_REGISTRATION_CONSTRUCTION_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_program_source_graph_not_yet_frozen':True,'inert_main_source_fixture':True,'native_API_framework_called':False,'native_launch_authorized':False,'native_authority_issued':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'closed_registration.py',Path(__file__).resolve())}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
