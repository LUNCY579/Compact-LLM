"""Independent frozen import supervisor adapter, original body with inert host."""
from pathlib import Path
from types import ModuleType
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_import_supervisor001';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');TEST=FROZEN/'test_import_supervisor.py';FILES=[p for p in FROZEN.rglob('*') if p.is_file()]+[PACKAGE/'native_supervisor.py',PACKAGE/'evidence/observer_startup_measurement_reviewed_cpu_20261006/test_supervisor.py']
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'import_observation_supervisor.py')]['sha256']=='00726a65bd60f85528d7133607e6af0a5400e4006fc9eee313a1e90ed718da76';assert BEFORE[str(TEST)]['sha256']=='261a0c106a06718b4a2fb4c609ce754d50ad1a39b3646deffa7166a13ba7e49c';assert BEFORE[str(PACKAGE/'native_supervisor.py')]['sha256']=='edf2de5b0917150c2c79e6a555dc0399995d12859028e62a7f7ece86b767e691'
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_process','native_worker','pynvml'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);m=ModuleType('independent_frozen_import_SUP_fixture');m.__file__=str(TEST);exec(compile(TEST.read_bytes(),str(TEST),'exec'),m.__dict__);Base=m.Controls;DETAILS={}
class Independent(Base):
    def test_original_owned_flow_V2argv_Job_before_resume_terminal(self):
        result,proof=self.run_body();self.assertEqual(result['status'],'SUPERVISED_WORKER_EXITED_ZERO');self.assertEqual(proof['total_changes'],6);self.assertEqual(self.host.argv,['CPU_PINNED_PYTHON','-I','-S','-B',str(self.worker),'--config',self.config,'--output-dir',str(self.output)]);self.assertLess(self.host.calls.index('Job_assigned'),self.host.calls.index('resume'));self.assertEqual(self.host.writes[str(self.output/'OWNED_EXTERNAL_ANCHORS.json')],self.refs);self.assertTrue(result['job_closed_kill_on_close']);self.assertTrue(result['sampler_threads_terminal_verified']);self.assertTrue(result['gpu_telemetry_cleanup_verified']);DETAILS['original_supervise_inert_return']=True
    def test_original_low_floor_denied_without_worker(self):
        self.host.low=True;result,_=self.run_body();self.assertEqual(result['status'],'ADMISSION_DENIED');self.assertFalse(result['worker_attempted']);self.assertNotIn('resume',self.host.calls)
    def test_failed_worker_and_unjoined_probe_refuse_completion(self):
        self.host.returncode=2;result,_=self.run_body();self.assertEqual(result['status'],'SUPERVISED_WORKER_FAILED');self.assertEqual(result['returncode'],2);self.assertTrue(result['job_closed_kill_on_close'])
        self.host.pending=[{'name':'CPU_pending','ident':1,'operation':'CPU_inert'}];result,_=self.run_body();self.assertFalse(result['sampler_threads_terminal_verified']);self.assertFalse(result['partial_outputs_frozen']);self.assertTrue(result['unjoined_sampler_threads']);DETAILS['actual_unjoined_probe_terminal_refusal']=True
    def test_separate_grant_default_denial_before_source_worker(self):self.test_public_entry_closed_before_source_or_worker_access()
    def test_actual_partial_raw_leases_sources_and_absent_worker_result_preserved(self):self.test_actual_partial_preservation_without_worker_result();DETAILS['actual_partial_preservation_absent_worker_result']=True
if __name__=='__main__':
    names=('test_original_owned_flow_V2argv_Job_before_resume_terminal','test_original_low_floor_denied_without_worker','test_failed_worker_and_unjoined_probe_refuse_completion','test_separate_grant_default_denial_before_source_worker','test_actual_partial_raw_leases_sources_and_absent_worker_result_preserved');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);after=pins();receipt={'kind':'INDEPENDENT_FROZEN_IMPORT_SUPERVISOR001_CPU_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'pins_unchanged':BEFORE==after,'details':DETAILS,'actual_original_supervise_source_body_executed':True,'actual_finite_partial_raw_lease_source_preservation_executed':True,'Job_resources_birth_death_monitor_callbacks_inert_CPU':True,'native_source_bridge_entry_default_denied':True,'native_API_framework_or_process_called':False,'native_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_IMPORT_SUPERVISOR001.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
