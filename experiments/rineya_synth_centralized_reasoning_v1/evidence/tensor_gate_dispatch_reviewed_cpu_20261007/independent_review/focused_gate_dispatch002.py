"""Independent frozen gate enrollment tests; CPU origins never native grants."""
from pathlib import Path
import ast, copy, hashlib, json, sys, types, unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_gate_dispatch002';CANDIDATE=FROZEN/'candidate';TEST=FROZEN/'test_gate_dispatch.py';PLAN=FROZEN/'GATE_DISPATCH_SOURCE_PLAN002.json';FILES=[p for p in FROZEN.rglob('*') if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(PLAN)]['sha256']=='2ad7bbaba2a304a9b36c6f269e045bc4e50170ce5d5525292de41bb58d55ddf7';assert BEFORE[str(TEST)]['sha256']=='9e3371a5fce311ff788d12ec91792395090a60ee60dc84badd22fcebe10b5c2f';assert BEFORE[str(CANDIDATE/'prefix_tensor_contract.py')]['sha256']=='80a4650ab7c4b386f560de7c18f08d5c8d317f948fc0212afdf5cb5be4d9a577'
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker','pynvml'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap)
# Only the authored fixture source-location assignment is replaced. The actual
# candidate enrollment/Git/helper/function implementations are unmodified.
tree=ast.parse(TEST.read_bytes());changed=0
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CANDIDATE' for t in n.targets):n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(CANDIDATE))],keywords=[]);changed+=1
assert changed==1
namespace={'__name__':'independent_frozen_gate_fixture','__file__':str(TEST)};exec(compile(ast.fix_missing_locations(tree),str(TEST),'exec'),namespace);Base=namespace['Controls'];load=namespace['load'];DETAILS={}
class Independent(Base):
    def test_real_sealed_two_commit_enrollment_and_CPU_record_refusal(self):
        proof=self.enroll();self.assertTrue(proof['fixture_only']);self.assertFalse(proof['native_runtime_qualified']);self.assertFalse(proof['native_authority_issued']);self.assertEqual(len(self.impl),40);self.assertEqual(len(self.head),40);self.assertNotEqual(self.impl,self.head)
        with self.assertRaisesRegex(ValueError,'CPU fixture refused'):self.C.stage_native_gate()
        self.assertTrue(self.git('merge-base','--is-ancestor',self.impl,self.head)==b'');DETAILS['actual_two_commit_sealed_Git_enrollment']=True
    def test_absent_unapproved_nomodel_forged_origin_and_real_route_refuse(self):
        self.test_absent_unapproved_NoModel_or_forged_origin_refused();self.test_real_route_does_not_accept_CPU_origin_or_bool_token()
        grant=copy.deepcopy(self.grant);grant['trusted_owner_origin']['receipt_body_sha256']='0'*64
        with self.assertRaises(ValueError):self.C._gate_document_join(self.seal(grant),self.receipt,self.binding,self.sources,fixture_only=True)
    def test_exact_anchor_and_immutable_blob_rebinding_refuse(self):
        for name in ('expected_config_file','expected_config_body','expected_runtime_file','expected_registration_file'):
            binding=copy.deepcopy(self.binding);binding['anchors'][name]='0'*64
            with self.subTest(anchor=name),self.assertRaises(ValueError):self.C._enroll_native_gate_impl(self.grant_reference,self.receipt_reference,binding,fixture_only=True,_capability=self.cap)
        path=self.pkg/'registration_git_join.py';path.write_bytes(path.read_bytes()+b'\n# independent drift\n')
        with self.assertRaises(ValueError):self.enroll()
    def test_actual_dispatcher_missing_references_and_original_V2_child_schema(self):
        D=load(CANDIDATE/'prefix_tensor_dispatcher.py','independent_dispatcher');self.assertEqual(D.CONTRACT_DESCRIPTOR['sha256'],BEFORE[str(CANDIDATE/'prefix_tensor_contract.py')]['sha256']);self.assertEqual(D.PIPELINE_DESCRIPTOR['sha256'],BEFORE[str(CANDIDATE/'prefix_native_pipeline.py')]['sha256'])
        with self.assertRaises(ValueError):D.dispatch(self.configs/'config.json',None,grant_reference=None,owner_receipt_reference=None,anchors=self.binding['anchors'])
        w=ast.parse((CANDIDATE/'prefix_input_owned_worker.py').read_bytes());self.assertIn('C.enroll_native_gate_from_owned_output(args.output_dir)',ast.unparse(w));p=ast.parse((CANDIDATE/'prefix_native_pipeline.py').read_bytes());text=ast.unparse(p);self.assertIn("'schema_version': 2",text);self.assertIn("'grant_reference': grant_reference",text);self.assertIn("'owner_receipt_reference': owner_receipt_reference",text)
        s=ast.unparse(ast.parse((CANDIDATE/'prefix_input_owned_supervisor.py').read_bytes()));a=ast.unparse(ast.parse((CANDIDATE/'prefix_input_owned_auditor.py').read_bytes()));self.assertIn('C.enroll_native_gate_from_owned_output(output',s);self.assertIn('contract.enroll_native_gate_from_owned_output(original_directory',a);DETAILS['parent_child_auditor_propagation_source_only']=True
    def test_actual_owned_anchor_reader_refuses_rebinding_before_enrollment(self):
        output=self.pkg/'artifacts';output.mkdir();binding=copy.deepcopy(self.binding);binding['anchors']['expected_head']='0'*40;external={'schema_version':2,'config_path':binding['config_path'],'anchors':binding['anchors'],'grant_reference':self.grant_reference,'owner_receipt_reference':self.receipt_reference};(output/'OWNED_EXTERNAL_ANCHORS.json').write_text(self.C.canonical(external),encoding='utf-8')
        # The native interface refuses at its exact owned-output leaf predicate
        # or independently selected binding comparison; no CPU enrollment route.
        with self.assertRaises(ValueError):self.C.enroll_native_gate_from_owned_output(output,expected_binding=self.binding)
        leaf=output/'leaf';leaf.mkdir();(leaf/'OWNED_EXTERNAL_ANCHORS.json').write_text(self.C.canonical(external),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'independently selected caller binding'):self.C.enroll_native_gate_from_owned_output(leaf,expected_binding=self.binding)
    def test_compile_budget_caps_and_exact_unapproved_templates(self):
        self.assertEqual(len(list(CANDIDATE.glob('*.py'))),12)
        for p in CANDIDATE.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
        with self.assertRaises(ValueError):self.C.QualificationBudget().check(16*1024**3+1)
        clock=[0];budget=self.C.QualificationBudget(clock=lambda:clock[0]);clock[0]=601
        with self.assertRaises(ValueError):budget.check()
        owner=json.loads((FROZEN/'OWNER_GRANT_UNAPPROVED_TEMPLATE001.json').read_text(encoding='utf-8'));dispatch=json.loads((FROZEN/'DISPATCH_UNAPPROVED_TEMPLATE001.json').read_text(encoding='utf-8'));self.assertIs(owner['approved'],False);self.assertIsNone(owner['trusted_owner_origin']);self.assertIs(dispatch['native_launch_authorized'],False);self.assertIsNone(dispatch['grant_reference']);self.assertEqual(self.C.RESOURCES['whole_scope_seconds'],600)
if __name__=='__main__':
    names=('test_real_sealed_two_commit_enrollment_and_CPU_record_refusal','test_absent_unapproved_nomodel_forged_origin_and_real_route_refuse','test_exact_anchor_and_immutable_blob_rebinding_refuse','test_actual_dispatcher_missing_references_and_original_V2_child_schema','test_actual_owned_anchor_reader_refuses_rebinding_before_enrollment','test_compile_budget_caps_and_exact_unapproved_templates');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);after=pins();receipt={'kind':'INDEPENDENT_FROZEN_GATE_DISPATCH002_CPU_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_sources_unchanged':BEFORE==after,'details':DETAILS,'fixture_one_CANDIDATE_location_AST_adaptation':str(CANDIDATE),'owner_origin_explicit_inert_CPU_fixture':True,'human_origin_authenticated_by_code':False,'cooperative_external_Root_verified_receipt_required':True,'real_owner_permission_present':False,'native_API_or_framework_called':False,'native_gate_or_public_source_promoted':False,'native_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_GATE_DISPATCH002.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
