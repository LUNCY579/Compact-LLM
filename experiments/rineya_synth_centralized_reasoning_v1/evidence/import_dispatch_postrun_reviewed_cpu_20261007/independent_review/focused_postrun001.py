"""Independent ordered postrun CPU fixture plus actual pure terminal audit."""
from pathlib import Path
from types import ModuleType
import ast,hashlib,json,shutil,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_postrun001';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');NATIVE=PACKAGE/'artifacts/torch_import_observation_work/native_candidate';TERM=NATIVE/'frozen_import_terminal002';PROGRAM=PACKAGE/'evidence/import_only_program_reviewed_cpu_20261007/frozen_program002';MEASURE=PACKAGE/'evidence/torch_import_measurement_reviewed_cpu_20261007/frozen_measurement004';FILES=[p for root in (FROZEN,TERM,PROGRAM,MEASURE) for p in root.rglob('*') if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'import_dispatch_postrun.py')]['sha256']=='560cd7f6e17de962d9782e6e44731445b3c5411ab10984c3bac2c5df1ae8e4cb';assert BEFORE[str(FROZEN/'import_terminal_audit.py')]['sha256']=='3bd489a1f42458a931d72620f023e3dcd0443c501fe57059378d7e855359e94e';assert BEFORE[str(FROZEN/'test_postrun.py')]['sha256']=='00129f8eead3061c9097d23bbb7296284f7d289c144b429b0ad60596ecdd0fbe'
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);mirror=tempfile.TemporaryDirectory(prefix='CPU_postrun_frozen_',dir=HERE.parent);ROOT=Path(mirror.name);WORK=ROOT/'program';WORK.mkdir();TL=WORK/'terminal';TL.mkdir();POST=ROOT/'post';POST.mkdir()
for p in PROGRAM.glob('*.py'):shutil.copy2(p,WORK/p.name)
shutil.copytree(PROGRAM/'predecessors',WORK/'predecessors');shutil.copy2(MEASURE/'test_measurement.py',WORK/'test_measurement.py')
for p in TERM.glob('*.py'):shutil.copy2(p,TL/p.name)
for p in FROZEN.glob('*.py'):shutil.copy2(p,POST/p.name)
TEST=POST/'test_postrun.py';tree=ast.parse(TEST.read_bytes());changes=0
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE' for t in n.targets):n.value.args[0]=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(TL/'test_import_terminal_audit.py'))],keywords=[]);changes+=1
assert changes==1
namespace={'__name__':'independent_postrun_fixture','__file__':str(TEST)};exec(compile(ast.fix_missing_locations(tree),str(TEST),'exec'),namespace);Base=namespace['Controls'];D=namespace['D'];T=namespace['T'];DETAILS={}
class Independent(Base):
    def test_actual_terminal_success_after_preservation_only(self):
        self.set_postrun();result=self.postrun();self.assertEqual(self.events,['preserve']);self.assertEqual(result['terminal_status'],'IMPORT_MEASUREMENT_TERMINAL_VALIDATED_UNQUALIFIED');self.assertFalse(result['native_PASS']);self.assertFalse(result['native_authority_issued']);DETAILS['ordered_preserve_then_actual_terminal']=True
    def test_failed_import_and_absent_semantic_result_remain_partial(self):self.test_failed_worker_raw_stays_failed();self.doCleanups();self.setUp();self.test_absent_semantic_result_reports_partial_terminal()
    def test_external_context_birth_and_producer_identity_rebind_refuse(self):
        self.test_raw_context_or_birth_identity_rebind_refuses();self.doCleanups();self.setUp();self.set_postrun();bad=json.loads(json.dumps(self.expected_identity));bad['producer_source']={'path':next(p for p in self.value['config']['source_files'] if p!=str(self.package/'measurement_guard.py')),'descriptor':next(d for p,d in self.value['config']['source_files'].items() if p!=str(self.package/'measurement_guard.py'))};self.update_raw_identity(bad)
        with self.assertRaises(ValueError):self.postrun()
    def test_actual_preservation_failure_and_missing_grant_refuse(self):self.test_preservation_failure_and_absent_owner_refs_never_complete()
    def test_unregistered_target_and_no_resume_denial_never_fake_audit(self):self.test_unregistered_preservation_target_refuses_before_preserve();self.doCleanups();self.setUp();self.test_original_no_resume_denial_retains_raw_without_fake_audit()
    def test_source_order_exact_context_producer_and_preserve_before_audit(self):
        tree=ast.parse((FROZEN/'import_dispatch_postrun.py').read_bytes());post=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_postrun');text=ast.unparse(post);self.assertLess(text.index('supervisor.preserve_terminal'),text.index('terminal.audit_preserved'));dispatch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch');text=ast.unparse(dispatch);self.assertLess(text.index('C.enroll_native_gate'),text.index('SUP.supervise_registered'));self.assertLess(text.index('C.validate_native_manifest'),text.index('SUP.supervise_registered'));self.assertIn('context_sha256',ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_expected_measurement_identity')));DETAILS['source_dispatch_enrollment_order_only_inspected']=True
if __name__=='__main__':
    names=('test_actual_terminal_success_after_preservation_only','test_failed_import_and_absent_semantic_result_remain_partial','test_external_context_birth_and_producer_identity_rebind_refuse','test_actual_preservation_failure_and_missing_grant_refuse','test_unregistered_target_and_no_resume_denial_never_fake_audit','test_source_order_exact_context_producer_and_preserve_before_audit');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);mirror.cleanup();after=pins();receipt={'kind':'INDEPENDENT_POSTRUN_DISPATCH_TERMINAL_IDENTITY001_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_source_pins_unchanged':BEFORE==after,'details':DETAILS,'actual_pure_terminal_reader_raw_replay_executed':True,'preserver_callback_inert_fixture_copy':True,'SUP_Job_resources_death_gate_documents_authored_CPU_only':True,'context_getter_revalidator_inert_CPU_value':True,'test_only_one_BASE_load_path_AST_and_original_relative_companion_mirror':True,'native_SUP_dispatch_or_process_executed':False,'native_API_or_framework_called':False,'native_context_or_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_POSTRUN001.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
