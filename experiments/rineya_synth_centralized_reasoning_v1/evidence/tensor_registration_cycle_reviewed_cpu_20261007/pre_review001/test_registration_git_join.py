"""Real temporary two-commit Git construction, no runtime/model access."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,types,unittest
HERE=Path(__file__).resolve().parent
J=types.ModuleType('CPU_git_join');J.__file__=str(HERE/'registration_git_join.py');exec(compile(Path(J.__file__).read_bytes(),J.__file__,'exec'),J.__dict__)
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_registration_git_');self.root=Path(self.tmp.name);self.cap=J._CPU_capability();self.exceptions={}
        self.git('init','-q');self.git('config','user.email','cpu@example.invalid');self.git('config','user.name','CPU Fixture')
        self.source=self.root/'implementation.py';self.source.write_bytes(b'# authored source, never executed\n');self.git('add','implementation.py');self.git('commit','-qm','implementation')
        self.implementation=self.git('rev-parse','HEAD').decode().strip();raw=self.source.read_bytes();self.sources={str(self.source):{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}}
        self.config=self.seal({'kind':'CPU_AUTHORED_CONFIG','source_commit':self.implementation,'source_files':self.sources});self.runtime=self.seal({'kind':'CPU_AUTHORED_RUNTIME','source_commit':self.implementation,'files':{}})
        self.registration=self.seal({'kind':'CPU_AUTHORED_REGISTRATION','source_commit':self.implementation,'source_files':self.sources,'config_body_sha256':self.config['sha256'],'runtime_body_sha256':self.runtime['sha256']})
        self.packets={}
        for name,value in [('config',self.config),('runtime',self.runtime),('registration',self.registration)]:
            p=self.root/(name+'.json');raw=(json.dumps(value,sort_keys=True)+'\n').encode('utf-8');p.write_bytes(raw);self.packets[name]={'path':str(p),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256']}
        self.git('add','config.json','runtime.json','registration.json');self.git('commit','-qm','dispatch packets');self.head=self.git('rev-parse','HEAD').decode().strip()
    def tearDown(self):self.tmp.cleanup()
    def git(self,*args,input=None):return subprocess.run(['git','-c','maintenance.auto=false','-c','gc.auto=0',*args],cwd=self.root,input=input,capture_output=True,check=True,timeout=30).stdout
    def seal(self,x):return {**x,'sha256':J.digest(x)}
    def join(self,**change):
        args={'packets':self.packets,'source_files':self.sources,'implementation_commit':self.implementation,'expected_head':self.head,'git_output_exceptions':self.exceptions,'_capability':self.cap};args.update(change);return J._CPU_join(self.root,**args)
    def test_real_two_commit_packets_have_no_commit_fixed_point(self):
        result=self.join();self.assertNotEqual(self.head,self.implementation);self.assertTrue(result['fixture_only']);self.assertFalse(result['native_authority_issued']);self.assertEqual(result['implementation_commit'],self.implementation)
    def test_dirty_source_and_wrong_external_head_refused(self):
        self.source.write_bytes(b'# changed current source\n')
        with self.assertRaises(ValueError):self.join()
        self.git('checkout','--','implementation.py')
        with self.assertRaises(ValueError):self.join(expected_head=self.implementation)
    def test_rebound_or_uncommitted_packet_and_anchor_refused(self):
        p=self.root/'runtime.json';raw=p.read_bytes();p.write_bytes(raw+b' ')
        with self.assertRaises(ValueError):self.join()
        p.write_bytes(raw);self.packets['runtime']['file_sha256']='0'*64
        with self.assertRaises(ValueError):self.join()
    def test_nonancestor_implementation_and_wrong_source_blob_refused(self):
        tree=self.git('mktree',input=b'').decode().strip();other=self.git('commit-tree',tree,input=b'unrelated implementation\n').decode().strip()
        with self.assertRaises(subprocess.CalledProcessError):self.join(implementation_commit=other)
        old=self.sources[str(self.source)]['sha256'];self.sources[str(self.source)]['sha256']='0'*64
        with self.assertRaises(ValueError):self.join()
        self.sources[str(self.source)]['sha256']=old
    def test_only_exact_output_leaf_exception_and_native_capability_refuse(self):
        p=self.root/'output.log';p.write_bytes(b'CPU fixture output');self.exceptions[str(p)]={'role':'run_output'};self.join();q=self.root/'foreign.log';q.write_bytes(b'foreign')
        with self.assertRaises(ValueError):self.join()
        q.unlink()
        with self.assertRaises(ValueError):self.join(_capability=object())
        with self.assertRaises(ValueError):J.native_join()
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));r={'kind':'CPU_REAL_GIT_REGISTRATION_CYCLE_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'real_temporary_git_repositories':True,'model_or_runtime_read':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['registration_git_join.py','prefix_tensor_contract.py','test_registration_git_join.py']}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
