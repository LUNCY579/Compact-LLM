"""Actual shared worker-main, finder and selected pipeline bodies on CPU.

Already-loaded standard-library exports are explicit CPU bootstrap proxies.
Their inert origin metadata is an injected assumption, never an execution proof
or qualification of the real host's bootstrap/runtime bytes.
"""
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
import importlib,importlib.abc,importlib.machinery,sys,tempfile,time,unittest
from unittest.mock import patch
from outer_context_fixture import OuterContextFixture,HERE
from test_full_native_cpu_workflow import compiled
from test_worker_bindings import InertSuppliers

BOOTSTRAP_NAMES=('pathlib','ast','hashlib','importlib','importlib.util','importlib.abc',
    'importlib.machinery','marshal','types','stat','time','json','math','re','sys',
    'subprocess','copy','datetime','os')

def bootstrap_export_provenance():
    return {'kind':'CPU_ALREADY_LOADED_STDLIB_EXPORT_SUPPLIER_ASSUMPTIONS','fixture_only':True,
        'current_execution_proven':False,'actual_fresh_bootstrap_qualified':False,
        'inert_origin_represents_executing_namespace':False,
        'modules':[{'name':name,'actual_export_module_name':sys.modules[name].__name__,
            'actual_module_origin':getattr(sys.modules[name],'__file__',None),
            'actual_spec_origin':getattr(getattr(sys.modules[name],'__spec__',None),'origin',None),
            'export_supplier':'already_loaded_standard_library_module_dictionary',
            'current_execution_proven':False} for name in BOOTSTRAP_NAMES]}

def bootstrap_documents(fixture):
    C=fixture.C;path=str(fixture.site/'typing_extensions.py');pin=fixture.inert_files[path]['sha256']
    modules={name:{'path':path,'source_sha256':pin,'cache_path':None,'cache_sha256':None} for name in BOOTSTRAP_NAMES}
    fixture.bootstrap=C.parse(C.canonical(fixture.bootstrap));fixture.bootstrap['preloaded_bootstrap_modules_sha256']=C.digest(modules);fixture.bootstrap=fixture._seal(fixture.bootstrap)
    fixture.runtime=C.parse(C.canonical(fixture.runtime));fixture.runtime['cache_policy']['preloaded_bootstrap_modules']=modules
    fixture.runtime['cache_policy']['fresh_bootstrap_evidence']=fixture.anchor(fixture.bootstrap_path,fixture.bootstrap);fixture.runtime=fixture._seal(fixture.runtime)
    fixture.review=C.parse(C.canonical(fixture.review));fixture.review['cache_policy_sha256']=C.digest(fixture.runtime['cache_policy']);fixture.review=fixture._seal(fixture.review)
    fixture.registration=C.parse(C.canonical(fixture.registration));fixture.registration['runtime_body_sha256']=fixture.runtime['sha256']
    fixture.registration['qualification_review_file_sha256']=fixture.anchor(fixture.review_path,fixture.review)['file_sha256'];fixture.registration['qualification_review_body_sha256']=fixture.review['sha256'];fixture.registration=fixture._seal(fixture.registration)
    for p,v in ((fixture.bootstrap_path,fixture.bootstrap),(fixture.runtime_path,fixture.runtime),(fixture.review_path,fixture.review),(fixture.registration_path,fixture.registration)):
        Path(p).write_bytes(fixture._raw(v));fixture.documents[str(p)]=v
    fixture.anchors.update(expected_runtime_file=fixture.anchor(fixture.runtime_path,fixture.runtime)['file_sha256'],expected_runtime_body=fixture.runtime['sha256'],expected_registration_file=fixture.anchor(fixture.registration_path,fixture.registration)['file_sha256'],expected_registration_body=fixture.registration['sha256'])

class RegisteredWorkerOrchestrationControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_actual_registered_main_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.fixture=OuterContextFixture(self.temp.name);bootstrap_documents(self.fixture)
        self.C=self.fixture.C;self.context=self.fixture.context();self.cap=self.C._mint_CPU_native_context_fixture()
        self.worker=compiled(self.fixture.package/'prefix_input_owned_worker.py');self.worker.C=self.C
        self.loader=compiled(self.fixture.package/'prefix_registered_loader.py')
        self.output=self.fixture.package/'artifacts/prefix_input_owned_run_CPU';self.output.mkdir()
        self.owner={'worker_pid':12345,'root_worker_pid':12345,'supervisor_pid':12344,'job_name':'Local\\SYNTH_NQ_JOB_'+'1'*32,'supervisor_create_time':1000.25,'config_sha256':self.fixture.anchors['expected_config_file']}
        (self.output/'supervisor_ownership.json').write_bytes((self.C.canonical(self.owner)+'\n').encode('ascii'))
        (self.output/'OWNED_EXTERNAL_ANCHORS.json').write_bytes((self.C.canonical({'config_path':str(self.fixture.config_path),'anchors':self.fixture.anchors})+'\n').encode('ascii'))
        F=compiled(HERE/'fixture_support.py');self.inert=InertSuppliers(self.fixture,self.owner)
        fake=F.F.FakeTorch(F.F.Recorder());fake.__version__=self.fixture.config['torch_version'];fake.backends=self.inert.torch.backends
        for name in ('set_float32_matmul_precision','use_deterministic_algorithms','set_num_threads'):setattr(fake,name,getattr(self.inert.torch,name))
        self.inert.torch=fake;self.host=F.FakeHost(self.output);self.host.offset=0;self.host.started=time.monotonic();self.host.assigned=self.host.resumed=True
        self.bootstrap_provenance=bootstrap_export_provenance()
        self.host.job_name=self.owner['job_name'];self.host.supervisor_pid=self.owner['supervisor_pid'];self.environment={}
        original_framework=self.inert.framework
        def framework():
            self.inert_source=importlib.import_module('psutil')
            return original_framework()
        self.inert.framework=framework
    @contextmanager
    def isolated(self):
        # Export functions come only from libraries already loaded for this
        # CPU test. No proxy's namespace is attributed to current source exec.
        with self.fixture.activate():
            modules={}
            for name in BOOTSTRAP_NAMES:
                original=sys.modules[name];proxy=ModuleType(name);proxy.__dict__.update(original.__dict__)
                proxy.__file__=str(self.fixture.site/'typing_extensions.py');proxy.__cached__=None;proxy.__spec__=None;modules[name]=proxy
            allowed=[importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder]
            with patch.dict(sys.modules,modules,clear=True),patch.object(sys,'meta_path',allowed),patch.object(sys,'path',[str(self.fixture.site)]):yield
    def run_main(self):
        return self.worker._CPU_registered_main_fixture(self.context,self.owner,self.output,_capability=self.cap,loader=self.loader,suppliers=self.inert.mapping(),host=self.host,environment=self.environment)
    def test_actual_main_context_finder_pipeline_core_and_writer(self):
        with self.isolated():
            baseline=list(sys.meta_path);proof=self.run_main();self.assertEqual(sys.meta_path,baseline)
        self.assertEqual(proof['status'],'CPU_REGISTERED_MAIN_RETURNED');self.assertEqual(proof['returncode'],0)
        self.assertTrue(proof['fixture_only']);self.assertFalse(proof['native_PASS']);self.assertFalse(proof['native_authority_issued'])
        self.assertTrue(proof['finder'].fixture_only);self.assertNotIn(proof['finder'],sys.meta_path)
        self.assertEqual(proof['report']['status'],'NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT')
        actual=self.C.parse((self.output/'OWNED_WORKER_RETURN.json').read_bytes());self.assertEqual(actual,proof['container'])
        self.assertEqual(actual['report'],proof['report']);self.assertFalse(actual['native_PASS'])
        self.assertEqual(self.environment,{**self.fixture.config['offline_environment'],'CUDA_VISIBLE_DEVICES':'0'})
        self.assertEqual(self.inert.events[:2],['native_worker_supplier','ownership_verify'])
        self.assertEqual(self.inert.events.count('framework_supplier'),1)
        self.assertEqual(len(proof['pipeline_proof']['binding']['topology_observations']),1)
        self.assertTrue(all(not row['current_module_execution_observed'] for key,row in proof['finder'].observed.items() if key in BOOTSTRAP_NAMES))
        observed=proof['finder'].observed['psutil'];self.assertEqual(observed['path'],str(self.fixture.site/'psutil/__init__.py'))
        self.assertTrue(observed['current_module_execution_observed']);self.assertEqual(observed['loader_kind'],'registered_source')
        self.assertEqual(actual['runtime_execution_modules']['psutil'],observed)
        self.assertEqual([x['name'] for x in self.bootstrap_provenance['modules']],list(BOOTSTRAP_NAMES))
        self.assertFalse(self.bootstrap_provenance['current_execution_proven']);self.assertFalse(self.bootstrap_provenance['actual_fresh_bootstrap_qualified'])
        self.assertTrue(proof['input_prelude']['actual_selected_files_reconsumed']);self.assertFalse(proof['input_prelude']['native_authority_issued'])
        self.assertEqual(proof['input_prelude']['external'],{'config_path':str(self.fixture.config_path),'anchors':self.fixture.anchors})
        self.assertEqual(proof['input_prelude']['owner_sha256'],self.C.digest(self.owner))
    def prelude_failure(self,message):
        with self.fixture.activate(),patch.object(self.loader,'_CPU_install_fixture',wraps=self.loader._CPU_install_fixture) as install:
            with self.assertRaisesRegex(ValueError,message):self.run_main()
            self.assertEqual(install.call_count,0)
        self.assertEqual(self.inert.events,[]);self.assertEqual(self.environment,{})
        self.assertFalse((self.output/'PHASE_OBSERVATIONS.jsonl').exists());self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists())
    def test_actual_external_config_path_drift_refused_before_loader_or_framework(self):
        path=self.output/'OWNED_EXTERNAL_ANCHORS.json';value=self.C.parse(path.read_bytes());value['config_path']=str(self.fixture.package/'configs/CPU_DIFFERENT_CONFIG.json');path.write_bytes(self.fixture._raw(value))
        self.prelude_failure('owned external anchor config join')
    def test_actual_external_anchor_drift_fails_real_factory_before_loader_or_framework(self):
        path=self.output/'OWNED_EXTERNAL_ANCHORS.json';value=self.C.parse(path.read_bytes());value['anchors']['expected_config_file']='0'*64;path.write_bytes(self.fixture._raw(value))
        self.prelude_failure('external file anchor')
    def test_actual_owner_receipt_drift_refused_before_loader_or_framework(self):
        path=self.output/'supervisor_ownership.json';value=self.C.parse(path.read_bytes());value['worker_pid']=value['root_worker_pid']=999;path.write_bytes(self.fixture._raw(value))
        self.prelude_failure('same original CPU main owned receipt')
    def test_owned_membership_failure_removes_actual_finder_before_environment_or_core(self):
        self.inert.membership_fault=lambda result:result.update(owned_job_membership_verified=False)
        with self.isolated():
            baseline=list(sys.meta_path)
            with self.assertRaisesRegex(ValueError,'same immutable main launcher'):self.run_main()
            self.assertEqual(sys.meta_path,baseline)
        self.assertEqual(self.inert.events,['native_worker_supplier','ownership_verify']);self.assertEqual(self.environment,{})
        self.assertFalse((self.output/'PHASE_OBSERVATIONS.jsonl').exists());self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists())
    def test_selected_pipeline_drift_after_install_refuses_before_framework_and_removes(self):
        original=self.inert.verify
        def verify(owner,config_path):
            result=original(owner,config_path);path=self.fixture.package/'prefix_native_pipeline.py';raw=path.read_bytes();path.write_bytes(b'#'+raw[1:]);return result
        self.inert.verify=verify
        with self.isolated():
            baseline=list(sys.meta_path)
            with self.assertRaisesRegex(ValueError,'exact stable source module'):self.run_main()
            self.assertEqual(sys.meta_path,baseline)
        self.assertEqual(self.inert.events,['native_worker_supplier','ownership_verify']);self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists())
    def test_exclusive_return_write_failure_keeps_prior_bytes_and_removes_finder(self):
        path=self.output/'OWNED_WORKER_RETURN.json';prior=b'CPU explicit preexisting return bytes\n';path.write_bytes(prior)
        with self.isolated():
            baseline=list(sys.meta_path)
            with self.assertRaises(FileExistsError):self.run_main()
            self.assertEqual(sys.meta_path,baseline)
        self.assertEqual(path.read_bytes(),prior);self.assertTrue((self.output/'worker_result.json').exists());self.assertEqual(self.inert.events.count('framework_supplier'),1)
    def test_public_main_and_native_binding_remain_stage_refused_for_CPU_context(self):
        with self.fixture.activate():
            with self.assertRaises(ValueError):self.worker.main()
            with self.assertRaises(ValueError):self.worker.real_torch_bindings(self.context,self.owner,self.output)
        self.assertEqual(self.inert.events,[])

if __name__=='__main__':unittest.main()
