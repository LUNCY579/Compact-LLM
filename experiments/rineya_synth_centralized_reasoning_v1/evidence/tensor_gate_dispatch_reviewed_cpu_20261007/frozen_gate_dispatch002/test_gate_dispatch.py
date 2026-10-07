"""Externally pinned owner/dispatch enrollment; CPU tokens never native permits."""
from pathlib import Path
import ast,copy,hashlib,json,shutil,subprocess,tempfile,types,unittest
HERE=Path(__file__).resolve().parent;CANDIDATE=HERE/'gate_dispatch_candidate'
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_gate_dispatch_');self.repo=Path(self.tmp.name)/'repo';self.repo.mkdir();self.pkg=self.repo/'experiments/rineya_synth_centralized_reasoning_v1';self.pkg.mkdir(parents=True);self.configs=self.pkg/'configs';self.configs.mkdir();self.outside=Path(self.tmp.name)/'owner';self.outside.mkdir()
        for n in ('prefix_tensor_contract.py','registration_git_join.py'):shutil.copy2(CANDIDATE/n,self.pkg/n)
        self.git('init','-q');self.git('add','.');self.git('commit','-qm','CPU implementation');self.impl=self.git('rev-parse','HEAD').decode().strip();self.C=load(self.pkg/'prefix_tensor_contract.py','CPU_gate_C');self.cap=self.C._mint_CPU_native_context_fixture();self.sources={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'role':'python_source'} for p in self.pkg.glob('*.py')}
        runtime={'source_commit':self.impl,'files':{}};self.runtime=self.write(self.configs/'runtime.json',runtime);config={'source_commit':self.impl,'scope':self.C.NATIVE_SCOPE,'permission':self.C.PERMISSION,'resources':self.C.RESOURCES,'source_files':self.sources,'runtime_files':{}};self.config=self.write(self.configs/'config.json',config);registration={'source_commit':self.impl,'config_body_sha256':self.config['body_sha256'],'runtime_body_sha256':self.runtime['body_sha256'],'source_files':self.sources,'runtime_files':{},'git_output_exceptions':{}};self.registration=self.write(self.configs/'registration.json',registration);self.git('add','.');self.git('commit','-qm','CPU dispatch');self.head=self.git('rev-parse','HEAD').decode().strip()
        self.binding={'config_path':self.config['path'],'anchors':{'expected_config_file':self.config['file_sha256'],'expected_config_body':self.config['body_sha256'],'runtime_path':self.runtime['path'],'expected_runtime_file':self.runtime['file_sha256'],'expected_runtime_body':self.runtime['body_sha256'],'registration_path':self.registration['path'],'expected_registration_file':self.registration['file_sha256'],'expected_registration_body':self.registration['body_sha256'],'expected_head':self.head}}
        self.receipt={'kind':'CPU_OWNER_REPLY_SOURCE_FIXTURE','scope':self.C.NATIVE_SCOPE,'permission':self.C.PERMISSION,'approved':True,'direct_human_text':'CPU inert text, not human permission','source_thread_id':'CPU_thread','source_turn_id':'CPU_turn','source_user_message_id':'CPU_message','verified_by':'CPU_INERT_ORIGIN_FIXTURE','fixture_only':True};self.receipt_reference=self.write(self.outside/'receipt.json',self.receipt);self.receipt['sha256']=self.receipt_reference['body_sha256']
        self.grant={'kind':'SEPARATE_TWO_TENSOR_OWNER_SCOPE_GRANT','scope':self.C.NATIVE_SCOPE,'permission':self.C.PERMISSION,'resources':self.C.RESOURCES,'approved':True,'trusted_owner_origin':{'receipt_body_sha256':self.receipt['sha256'],**{k:self.receipt[k] for k in ('source_thread_id','source_turn_id','source_user_message_id')}},'dispatch_binding_sha256':self.C.digest(self.binding),'source_inventory_sha256':self.C.digest(self.sources)};self.grant_reference=self.write(self.outside/'grant.json',self.grant);self.grant['sha256']=self.grant_reference['body_sha256']
    def tearDown(self):self.tmp.cleanup()
    def git(self,*args):return subprocess.run(['git','-c','user.name=CPU Fixture','-c','user.email=cpu@example.invalid','-c','maintenance.auto=false','-c','gc.auto=0',*args],cwd=self.repo,capture_output=True,check=True).stdout
    def write(self,path,value):
        value=copy.deepcopy(value);value['sha256']=self.C.digest(value) if hasattr(self,'C') else hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest();raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();path.write_bytes(raw);return {'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':value['sha256']}
    def seal(self,value):value['sha256']=self.C.digest({k:v for k,v in value.items() if k!='sha256'});return value
    def enroll(self):return self.C._enroll_native_gate_impl(self.grant_reference,self.receipt_reference,self.binding,fixture_only=True,_capability=self.cap)
    def test_actual_two_commit_source_Git_enrollment_CPU_token_native_gate_refused(self):
        proof=self.enroll();self.assertTrue(proof['fixture_only']);self.assertFalse(proof['native_authority_issued']);self.assertFalse(proof['native_runtime_qualified'])
        with self.assertRaisesRegex(ValueError,'CPU fixture refused'):self.C.stage_native_gate()
    def test_absent_unapproved_NoModel_or_forged_origin_refused(self):
        with self.assertRaises(ValueError):self.C._enroll_native_gate_impl(None,None,self.binding,fixture_only=True,_capability=self.cap)
        for change in ('approved','NoModel','origin'):
            grant=copy.deepcopy(self.grant)
            if change=='approved':grant['approved']=False
            elif change=='NoModel':grant['scope']='NO_MODEL_NATIVE_LIBRARY_OBSERVER_LOAD_UNLOAD'
            else:grant['trusted_owner_origin']['source_user_message_id']='forged_rebind'
            with self.assertRaises(ValueError):self.C._gate_document_join(self.seal(grant),self.receipt,self.binding,self.sources,fixture_only=True)
        with self.assertRaises(ValueError):self.C._gate_document_join(self.grant,{**self.receipt,'fixture_only':False},self.binding,self.sources,fixture_only=False)
    def test_dispatch_source_rebinding_dirty_repo_and_bad_anchor_refused(self):
        bad=copy.deepcopy(self.binding);bad['anchors']['expected_head']='0'*40
        with self.assertRaises(ValueError):self.C._enroll_native_gate_impl(self.grant_reference,self.receipt_reference,bad,fixture_only=True,_capability=self.cap)
        path=self.pkg/'registration_git_join.py';path.write_bytes(path.read_bytes()+b'\n# drift\n')
        with self.assertRaises(ValueError):self.enroll()
        with self.assertRaises(ValueError):self.C._enroll_native_gate_impl(self.grant_reference,self.receipt_reference,{'config_path':'x','anchors':{}},fixture_only=True,_capability=self.cap)
    def test_real_route_does_not_accept_CPU_origin_or_bool_token(self):
        with self.assertRaises(ValueError):self.C.enroll_native_gate(self.grant_reference,self.receipt_reference,self.binding)
        with self.assertRaises(ValueError):self.C._CPU_enroll_gate_fixture(self.grant,self.receipt,self.binding,self.sources,_capability=True)
        with self.assertRaises(ValueError):self.C.stage_native_gate()
    def test_concrete_dispatcher_V2_child_propagation_closed_templates(self):
        D=load(CANDIDATE/'prefix_tensor_dispatcher.py','CPU_dispatcher');self.assertIsNotNone(D.CONTRACT_DESCRIPTOR);self.assertIsNotNone(D.PIPELINE_DESCRIPTOR)
        with self.assertRaises(ValueError):D.dispatch(self.configs/'config.json',None,grant_reference=None,owner_receipt_reference=None,anchors=self.binding['anchors'])
        W=ast.parse((CANDIDATE/'prefix_input_owned_worker.py').read_bytes());text=ast.unparse(W);self.assertIn('C.enroll_native_gate_from_owned_output(args.output_dir)',text);P=ast.parse((CANDIDATE/'prefix_native_pipeline.py').read_bytes());self.assertIn("'schema_version': 2",ast.unparse(P));self.assertIn("'owner_receipt_reference': owner_receipt_reference",ast.unparse(P));S=ast.parse((CANDIDATE/'prefix_input_owned_supervisor.py').read_bytes());A=ast.parse((CANDIDATE/'prefix_input_owned_auditor.py').read_bytes());self.assertIn('C.enroll_native_gate_from_owned_output(output',ast.unparse(S));self.assertIn('contract.enroll_native_gate_from_owned_output(original_directory',ast.unparse(A))
        template=json.loads((HERE/'OWNER_GRANT_UNAPPROVED_TEMPLATE001.json').read_text());self.assertFalse(template['approved']);self.assertIsNone(template['trusted_owner_origin']);dispatch=json.loads((HERE/'DISPATCH_UNAPPROVED_TEMPLATE001.json').read_text());self.assertFalse(dispatch['native_launch_authorized']);self.assertIsNone(dispatch['grant_reference'])
    def test_all_candidate_definitions_compile_and_budget_caps_retained(self):
        for p in CANDIDATE.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
        self.assertEqual(self.C.RESOURCES['whole_scope_seconds'],600);budget=self.C.QualificationBudget()
        with self.assertRaises(ValueError):budget.check(16*1024**3+1)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_GATE_DISPATCH_EXTERNAL_OWNER_SOURCE_ENROLLMENT_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'actual_two_commit_Git_and_source_checks':True,'owner_origin':'explicit_CPU_fixture_not_human','native_permission_granted':False,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*CANDIDATE.glob('*.py')]}}
    if len(__import__('sys').argv)>1:
        with Path(__import__('sys').argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
