"""Full metadata/roster/bootstrap reader on inert filesystem and real Git."""
import builtins,contextlib,copy,hashlib,json,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ORIGINAL_IMPORT=builtins.__import__
def trap(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','ctypes','_ctypes','transformers','safetensors','pywin32_bootstrap','pywin32_system32','_virtualenv','_distutils_hack'} or name.startswith('__editable__'):raise AssertionError('CPU reader must not import runtime/startup modules')
    return ORIGINAL_IMPORT(name,*args,**kwargs)
builtins.__import__=trap
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'prefix_input_owned_execution_work'
REAL_RUN=subprocess.run
def load(path,name):
    module=types.ModuleType(name);module.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__);return module
O=load(OLD/'outer_context_fixture.py','CPU_original_outer_fixture')
class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_full_amended_');self.fixture=O.OuterContextFixture(self.temp.name);f=self.fixture;f.C=load(HERE/'prefix_tensor_contract.py','CPU_amended_contract');self.C=f.C;self.cap=self.C._mint_CPU_native_context_fixture();self.repo=f.package.parents[1]
        for name in ['prefix_tensor_contract.py','registration_git_join.py']:
            path=f.package/name;raw=(HERE/name).read_bytes();path.write_bytes(raw);f.sources[str(path)]=O.descriptor(raw)
        self.git('init','-q');self.git('config','user.email','cpu@example.invalid');self.git('config','user.name','CPU Fixture');self.git('add','.');self.git('commit','-qm','source and inert fixture implementation');f.head=self.git('rev-parse','HEAD').decode().strip();self.implementation=f.head
        f.config_path=f.package/'configs/CPU_AMENDED_CONFIG.json';f.registration_path=f.package/'artifacts/CPU_AMENDED_REGISTRATION.json';f.runtime_path=f.package/'artifacts/CPU_AMENDED_RUNTIME.json';f.review_path=f.package/'artifacts/CPU_AMENDED_REVIEW.json';f.bootstrap_path=f.package/'artifacts/CPU_AMENDED_BOOTSTRAP.json'
        f._build_documents();self.git('add','.');self.git('commit','-qm','registered dispatch and reviews');self.head=self.git('rev-parse','HEAD').decode().strip();f.anchors['expected_head']=self.head;self.actual_functions={}
    def tearDown(self):self.temp.cleanup()
    def git(self,*args,input=None):return REAL_RUN(['git','-c','maintenance.auto=false','-c','gc.auto=0',*args],cwd=self.repo,input=input,capture_output=True,check=True,timeout=30).stdout
    @contextlib.contextmanager
    def active(self):
        with self.fixture.activate(),patch('subprocess.run',new=REAL_RUN),contextlib.ExitStack() as stack:
            for name in ['native_metadata','_CPU_amended_metadata','qualification_rosters','physical_roster','_bootstrap_policy','load_input']:
                original=getattr(self.C,name)
                def wrapper(*args,_name=name,_original=original,**kwargs):self.actual_functions[_name]=self.actual_functions.get(_name,0)+1;return _original(*args,**kwargs)
                stack.enter_context(patch.object(self.C,name,new=wrapper))
            yield
    def read(self,**change):
        anchors=dict(self.fixture.anchors);anchors.update(change)
        with self.active():return self.C._CPU_read_amended_manifest(self.fixture.config_path,_capability=self.cap,**anchors)
    def rewrite_dispatch(self):
        f=self.fixture
        for path,value in [(f.config_path,f.config),(f.runtime_path,f.runtime),(f.review_path,f.review),(f.bootstrap_path,f.bootstrap),(f.registration_path,f.registration)]:path.write_bytes(f._raw(value))
        self.git('add','.');self.git('commit','-qm','rebound negative dispatch');self.head=self.git('rev-parse','HEAD').decode().strip()
        f.anchors={'expected_config_file':f.anchor(f.config_path,f.config)['file_sha256'],'expected_config_body':f.config['sha256'],'registration_path':str(f.registration_path),'expected_registration_file':f.anchor(f.registration_path,f.registration)['file_sha256'],'expected_registration_body':f.registration['sha256'],'runtime_path':str(f.runtime_path),'expected_runtime_file':f.anchor(f.runtime_path,f.runtime)['file_sha256'],'expected_runtime_body':f.runtime['sha256'],'expected_head':self.head}
    def test_full_original_metadata_rosters_bootstrap_and_actual_git_execute(self):
        context=self.read();value=self.C.context_value(context);self.assertEqual(value['implementation_commit'],self.implementation);self.assertEqual(value['expected_head'],self.head);self.assertNotEqual(self.implementation,self.head);self.assertTrue(value['fixture_only']);self.assertFalse(value['native_authority_issued'])
        for name in ['native_metadata','_CPU_amended_metadata','qualification_rosters','physical_roster','_bootstrap_policy','load_input']:self.assertGreater(self.actual_functions.get(name,0),0)
        self.assertTrue(self.fixture.actual_file_checks);self.assertTrue(self.fixture.historical_runtime_checks);self.assertEqual(self.fixture.git_calls,[])
        self.assertEqual(value['CPU_git_join']['dispatch_head'],self.head);self.assertFalse(value['CPU_git_join']['native_runtime_qualified'])
    def test_dirty_source_wrong_anchor_and_extra_anchor_refuse(self):
        f=self.fixture;path=f.package/'registration_git_join.py';raw=path.read_bytes();path.write_bytes(raw+b'\n')
        with self.assertRaises(ValueError):self.read()
        path.write_bytes(raw)
        with self.assertRaises(ValueError):self.read(expected_runtime_file='0'*64)
        with self.assertRaises(ValueError):self.read(extra='unregistered')
    def test_runtime_byte_drift_and_cache_source_review_rebound_refuse(self):
        f=self.fixture;path=f.site/'typing_extensions.py';raw=path.read_bytes();path.write_bytes(b'# altered inert runtime\n')
        with self.assertRaises(ValueError):self.read()
        path.write_bytes(raw);f.bootstrap=f._seal({**f.bootstrap,'preloaded_bootstrap_modules_sha256':'0'*64});f.runtime['cache_policy']['fresh_bootstrap_evidence']=f.anchor(f.bootstrap_path,f.bootstrap);f.runtime=f._seal(f.runtime);f.config['runtime_files']=f.runtime['files'];f.config=f._seal(f.config);f.review=f._seal({**f.review,'cache_policy_sha256':self.C.digest(f.runtime['cache_policy'])});f.registration=f._seal({**f.registration,'config_body_sha256':f.config['sha256'],'runtime_body_sha256':f.runtime['sha256'],'qualification_review_file_sha256':f.anchor(f.review_path,f.review)['file_sha256'],'qualification_review_body_sha256':f.review['sha256']});self.rewrite_dispatch()
        with self.assertRaises(ValueError):self.read()
    def test_nonancestor_implementation_refuses_real_git(self):
        f=self.fixture;tree=self.git('mktree',input=b'').decode().strip();other=self.git('commit-tree',tree,input=b'unrelated implementation\n').decode().strip()
        f.config=f._seal({**f.config,'source_commit':other});f.runtime=f._seal({**f.runtime,'source_commit':other});f.bootstrap=f._seal({**f.bootstrap,'source_commit':other});f.runtime['cache_policy']['fresh_bootstrap_evidence']=f.anchor(f.bootstrap_path,f.bootstrap);f.runtime=f._seal(f.runtime);f.review=f._seal({**f.review,'source_commit':other,'cache_policy_sha256':self.C.digest(f.runtime['cache_policy'])});f.registration=f._seal({**f.registration,'source_commit':other,'config_body_sha256':f.config['sha256'],'runtime_body_sha256':f.runtime['sha256'],'qualification_review_file_sha256':f.anchor(f.review_path,f.review)['file_sha256'],'qualification_review_body_sha256':f.review['sha256']});self.rewrite_dispatch()
        with self.assertRaises(subprocess.CalledProcessError):self.read()
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));r={'kind':'CPU_FULL_AMENDED_MANIFEST_READER_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'real_temporary_git':True,'native_metadata_qualification_rosters_bootstrap_executed':True,'injections':['exact historical65/stdlib779 descriptor and role checks','finite authored SITE/interpreter layout','preserved diagnostic input/source metadata copied by originalfixture'],'Git_output_injected':False,'real_runtime_bytes_qualified':False,'current_module_execution_proven':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['prefix_tensor_contract.py','registration_git_join.py','test_full_amended_reader.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
