"""ONE stdlib controlled-interface fixture; complete integration stays NOT_READY."""
import time
START=time.monotonic()
import ctypes as c
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

P=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
lib=load('fixed_library3',P/'cpu_only_tool_version.py')
ops=load('fixed_operations',P/'operations.py')
m=load('round4_components',P/'tool_complete_observer.py')

class Clock:
    def __init__(self):self.value=0.0
    def __call__(self):return self.value
    def wall(self):return 1800000000+self.value
    def tick(self):self.value+=.01
class Fn:
    def __init__(self,k,fn):self.k=k;self.fn=fn
    def __call__(self,*args):self.k.clock.tick();return self.fn(*args)
class Kernel:
    OUTER=0x100000040;INNER=0x100000050
    O=0x100000010;C=0x100000020;U=0x100000030;TH=0x100000031
    def __init__(self,clock):
        self.clock=clock;self.error=122;self.jobs=None;self.handles=None
        self.attributes={};self.open={self.OUTER,self.INNER,self.O,self.C}
        self.closed=[];self.creates=0;self.resumes=0;self.exited=False
        self.fail=None;self.membership=True;self.terminate_fail=False
        self.no_signal=False;self.wait_error=False;self.job_query_fail=False
        self.copy_failure=False;self.caller_lost=False;self.peak={self.O:100,self.C:100,self.U:100}
        methods={'InitializeProcThreadAttributeList':self.init,'UpdateProcThreadAttribute':self.update,
          'DeleteProcThreadAttributeList':lambda *a:None,'CreateProcessW':self.create,
          'GetProcessTimes':self.times,'QueryFullProcessImageNameW':self.image,
          'IsProcessInJob':self.injob,'WaitForSingleObject':self.wait,'TerminateProcess':self.kill,
          'ResumeThread':self.resume,'CloseHandle':self.close,'GetCurrentProcess':lambda:self.O,
          'DuplicateHandle':self.duplicate,'GetProcessMemoryInfo':self.memory,
          'QueryInformationJobObject':self.query_job}
        for name,fn in methods.items():setattr(self,name,Fn(self,fn))
    def init(self,attribute,count,flags,size):
        assert count==2 and flags==0
        if attribute is None:c.cast(size,c.POINTER(c.c_size_t)).contents.value=256;return False
        return self.fail!='init'
    def update(self,attribute,flags,key,value,size,*rest):
        assert flags==0
        if self.fail=='attribute':return False
        count=2 if key==m.JOB_LIST else 3
        assert key in (m.JOB_LIST,m.HANDLE_LIST) and size==count*c.sizeof(lib.w.HANDLE)
        self.attributes[key]=list(c.cast(value,c.POINTER(lib.w.HANDLE*count)).contents)
        return True
    def create(self,app,command,_,__,inherit,flags,env,cwd,startup,info):
        self.creates+=1
        assert app==m.UV and command.value=='"'+m.UV+'" --version'
        assert inherit and flags & 4 and flags & 0x80000 and not flags & 0x1000000
        assert self.attributes[m.JOB_LIST]==[self.OUTER,self.INNER]
        assert self.attributes[m.HANDLE_LIST]==[0x100000061,0x100000062,0x100000063]
        start=c.cast(startup,c.POINTER(lib.StartupEx)).contents
        assert start.StartupInfo.cb==c.sizeof(lib.StartupEx) and start.lpAttributeList
        self.jobs=list(self.attributes[m.JOB_LIST]);self.handles=list(self.attributes[m.HANDLE_LIST])
        p=c.cast(info,c.POINTER(lib.ProcessInfo)).contents
        if self.fail=='create':return False  # attempted, birth outcome ambiguous
        p.hProcess=self.U;p.dwProcessId=7;self.open.add(self.U)
        if self.fail=='partial':return False
        p.hThread=self.TH;p.dwThreadId=17;self.open.add(self.TH)
        if self.fail=='caller_loss':self.caller_lost=True
        return True
    def times(self,h,*args):
        assert h==self.U
        if self.fail=='times':return False
        f=c.cast(args[0],c.POINTER(lib.w.FILETIME)).contents;f.dwHighDateTime=1;f.dwLowDateTime=42
        return True
    def image(self,h,flags,buf,size):
        assert h==self.U
        buf.value=m.UV if self.fail!='image' else 'C:/wrong.exe';return True
    def injob(self,h,job,answer):
        assert h==self.U and job in (self.OUTER,self.INNER)
        c.cast(answer,c.POINTER(lib.w.BOOL)).contents.value=int(self.membership);return True
    def wait(self,h,timeout):
        assert h==self.U and timeout==0
        if self.wait_error:return 0xffffffff
        return 0 if self.exited else 258
    def kill(self,h,code):
        assert h==self.U
        if self.terminate_fail:return False
        if not self.no_signal:self.exited=True
        return True
    def resume(self,h):
        assert h==self.TH and h!=7
        self.resumes+=1;self.exited=True;return 1
    def close(self,h):self.closed.append(h);self.open.discard(h);return True
    def duplicate(self,own,h,caller,out,access,inherit,flags):
        assert own==self.O and caller==self.C and h in (self.U,self.TH) and flags==2 and not inherit
        if self.copy_failure:return False
        c.cast(out,c.POINTER(lib.w.HANDLE)).contents.value=h+0x1000;return True
    def memory(self,h,buf,size):
        if self.fail=='memory':return False
        x=c.cast(buf,c.POINTER(lib.Memory)).contents;x.WorkingSetSize=50;x.PeakWorkingSetSize=self.peak[h]
        return True
    def query_job(self,job,kind,buf,size,unused):
        assert job==self.INNER and kind==3
        if self.job_query_fail:return False
        x=c.cast(buf,c.POINTER(lib.Pids)).contents
        x.assigned=x.returned=0 if self.exited else 1;x.pids[0]=0 if self.exited else 7;return True

class Timer:
    def __init__(self,*args):self.daemon=False
    def start(self):pass
    def cancel(self):pass
    def is_alive(self):return False
    def join(self,timeout=None):pass

class Fixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.clock=Clock();self.k=Kernel(self.clock)
        self.birth=m.AtomicUVBirth(self.k,lib,self.k.OUTER,self.k.INNER,lambda:self.k.error,self.clock)
        self.store=ops.OperationsStore(Path(self.tmp.name)/'fixture.sqlite3',Path(self.tmp.name)/'receipts',clock=self.clock.wall)
        row=self.store.register_job('fixture-only-round4',{'synthetic':True,'cumulative_round':4},'fixture-epoch')
        self.store.ready(row['key']);lease=self.store.acquire_lease('fixture-owner',600)
        self.gate={'key':row['key'],'request_hash':row['request_hash'],'epoch':row['epoch'],'owner':'fixture-owner','fence':lease['fence']}
        self.store.reserve(self.gate['key'],self.gate['owner'],self.gate['fence'],10)
        self.delegate=m.TruthfulStoreDelegate(self.store,self.birth,self.gate)
    def tearDown(self):self.store.close();self.tmp.cleanup()
    def create(self):return self.delegate.create_and_record('C:/fixture','C:/fixture/tmp',[0x100000061,0x100000062,0x100000063])
    def graph(self):return m.combined_peak_witness(self.k,lib,{'O':self.k.O,'C':self.k.C,'U':self.k.U})
    def unknown10(self):
        self.assertEqual(self.store.budget_snapshot('fixture-epoch')['epoch_charged_seconds'],10)
    def native_success(self):
        info,identity=self.create();self.k.exited=True;self.clock.value+=.25
        proof={'observer':self.gate['owner'],'request_hash':self.gate['request_hash'],'owner':self.gate['owner'],
          'fence':self.gate['fence'],'worker_identity':identity,'process_exited':True,
          'exit_observed_at':self.clock.wall(),'elapsed_native_seconds':.25,'never_dispatched':False}
        ref=self.store.write_receipt('native_outcome_observation',self.gate['key'],proof)
        row=self.store.reconcile(self.gate['key'],'SUCCEEDED',.25,{'native_outcome_proof_ref':ref},owner=self.gate['owner'],fence=self.gate['fence'])
        return row,ref
    def test_typed_atomic_job_list_and_original_handles_before_return(self):
        info,identity=self.create();self.assertEqual(self.k.jobs,[self.k.OUTER,self.k.INNER])
        self.assertEqual(self.birth.partial_handles,{'process':self.k.U,'thread':self.k.TH})
        self.assertEqual(self.k.CreateProcessW.argtypes[-1],c.POINTER(lib.ProcessInfo))
        self.assertEqual(self.store.get_job(self.gate['key'])['worker'],identity)
        self.assertEqual(self.k.resumes,0)
    def test_idempotent_library_mark_running_same_actual_identity(self):
        _,identity=self.create()
        self.assertEqual(self.delegate.mark_running(self.gate['key'],self.gate['owner'],self.gate['fence'],identity)['worker'],identity)
        self.assertEqual(self.k.creates,1)
    def test_copy_ack_required_and_originals_survive_ack_loss(self):
        self.create();copies=self.birth.duplicate_for_c(self.k.C)
        with self.assertRaises(m.Refused):self.delegate.resume()
        self.assertIn(self.k.U,self.k.open);self.assertIn(self.k.TH,self.k.open)
        frame=m.FixedFrame('a'*64).encode('ACK_COPIES',copies)
        self.birth.acknowledge(m.hashlib.sha256(frame).hexdigest(),copies);self.delegate.resume()
        self.assertEqual(self.k.resumes,1)
    def test_no_second_U_creation(self):
        self.create()
        with self.assertRaises(m.Refused):self.birth.create('x','x',[1,2,3])
        self.assertEqual(self.k.creates,1)
    def test_caller_loss_at_birth_keeps_originals(self):
        self.k.fail='caller_loss';self.create()
        self.assertTrue(self.k.caller_lost);self.assertIn(self.k.U,self.k.open);self.assertNotIn(self.k.U,self.k.closed)
    def test_pristine_library_unknown_refusal_keeps_real_RESERVED(self):
        with self.assertRaises(m.Refused):self.delegate.mark_unknown(self.gate['key'],'library finally')
        self.assertEqual(self.store.get_job(self.gate['key'])['state'],'RESERVED')
        self.assertFalse(self.birth.attempted);self.assertEqual(self.k.creates,0)
        self.assertEqual(self.delegate.reconcile_pristine_failure()['actual_seconds'],0)
    def test_actual_immutable_library_preU_finally_records_refusal_not_fake_ack(self):
        # Fresh READY job on a separate temporary store: actual library path,
        # constructor refusal before creation; no OS kernel or child invocation.
        with tempfile.TemporaryDirectory() as d:
            s=ops.OperationsStore(Path(d)/'only.sqlite3',Path(d)/'r',clock=self.clock.wall)
            try:
                row=s.register_job('fixture-only-library',{'synthetic':True},'e');s.ready(row['key']);lease=s.acquire_lease('o',600)
                g=dict(separately_registered=True,source_roles_commit_accepted=True,loader_documentary_accepted=True,
                  trusted_metadata_boundary_accepted=True,key=row['key'],owner='o',epoch='e',request_hash=row['request_hash'],
                  source_commit='SYNTHETIC',contract_sha256='SYNTHETIC',runtime_pin='SYNTHETIC',admission_pin='SYNTHETIC',
                  fence=lease['fence'],telemetry_age_seconds=0,preworker_available_bytes=4*m.GIB,disk_free_bytes=5*m.GIB,
                  host_reserve_bytes=2*m.GIB,gpu_status='UNMEASURED_NOT_QUALIFIED',argv=['--version'],tool_path=lib.UV_PATH,
                  tool_sha256=lib.UV_SHA,canonical_operations_sha256=lib.OPS_SHA,physical_paths_verified=True,
                  empty_scoped_paths_verified=True,cwd=str(P/'.venv/tool-version-007/work'),temp=str(P/'.venv/tool-version-007/temp'))
                delegation=m.TruthfulStoreDelegate(s,self.birth,g)
                def refused_factory(guard):raise m.Refused('component has no C kernel-RPC bootstrap')
                r=lib.observe_version(g,delegation,refused_factory,lambda:{},clock=self.clock,
                  guard_factory=lambda:lib.Deadline(self.clock,Timer))
                self.assertEqual(s.get_job(row['key'])['state'],'RESERVED');self.assertTrue(r['ledger_errors'])
                self.assertTrue(delegation.journal);self.assertFalse(self.birth.attempted)
            finally:s.close()
    def test_attribute_failure_is_pre_attempt_pristine(self):
        self.k.fail='attribute'
        with self.assertRaises(m.Refused):self.create()
        self.assertFalse(self.birth.attempted);self.assertEqual(self.k.creates,0)
        with self.assertRaises(m.Refused):self.delegate.mark_unknown(self.gate['key'],'refused')
        self.assertEqual(self.store.get_job(self.gate['key'])['state'],'RESERVED')
    def test_CreateProcess_false_no_handles_is_attempt_ambiguity_not_zero(self):
        self.k.fail='create'
        with self.assertRaises(m.Refused):self.create()
        self.assertTrue(self.birth.attempted);self.assertIsNone(self.birth.info)
        self.delegate.mark_unknown(self.gate['key'],'attempt unknown');self.unknown10()
        with self.assertRaises(m.Refused):self.delegate.reconcile_pristine_failure()
    def test_partial_handles_preserved_on_creation_failure(self):
        self.k.fail='partial';self.k.terminate_fail=True
        with self.assertRaises(m.Refused):self.create()
        self.assertEqual(self.birth.partial_handles['process'],self.k.U)
        self.assertFalse(self.birth.retained_cleanup()['signal_proved']);self.assertIn(self.k.U,self.k.open)
        self.delegate.mark_unknown(self.gate['key'],'partial');self.unknown10()
    def test_created_identity_failure_workerNone_UNKNOWN_has_no_false_terminal(self):
        self.k.fail='times'
        with self.assertRaises(m.Refused):self.create()
        self.delegate.mark_unknown(self.gate['key'],'created identity unavailable')
        row=self.store.get_job(self.gate['key']);self.assertEqual(row['state'],'OUTCOME_UNKNOWN');self.assertIsNone(row['worker'])
        with self.assertRaises(m.Refused):self.delegate.reconcile_pristine_failure()
        with self.assertRaises(m.Refused):self.delegate.record_same_created_identity()
        self.unknown10();self.assertEqual(self.k.resumes,0)
    def test_registration_response_unknown_after_actual_commit_rereads_same_worker(self):
        original=self.store.mark_running
        def after_commit(*a):original(*a);raise RuntimeError('response lost')
        self.store.mark_running=after_commit
        _,identity=self.create();self.assertEqual(self.store.get_job(self.gate['key'])['worker'],identity)
        self.assertEqual(self.k.creates,1);self.assertTrue(self.delegate.journal)
    def test_registration_before_commit_failure_same_RESERVED_transition_no_relaunch(self):
        original=self.store.mark_running
        def before_commit(*a):raise RuntimeError('blocked before commit')
        self.store.mark_running=before_commit
        with self.assertRaises(RuntimeError):self.create()
        self.assertEqual(self.store.get_job(self.gate['key'])['state'],'RESERVED')
        self.store.mark_running=original;self.delegate.record_same_created_identity()
        self.assertEqual(self.k.creates,1);self.assertEqual(self.store.get_job(self.gate['key'])['state'],'RUNNING')
    def test_held_termination_failure_and_unknown_signal_keep_live_originals(self):
        self.create();self.k.terminate_fail=True
        result=self.birth.retained_cleanup();self.assertFalse(result['signal_proved']);self.assertFalse(self.k.exited)
        self.assertIn(self.k.U,self.k.open);self.assertIs(result['owner'],self.birth)
    def test_successful_termination_without_signal_and_unknown_job_keep_originals(self):
        self.create();self.k.no_signal=True
        self.assertFalse(self.birth.retained_cleanup()['signal_proved']);self.assertIn(self.k.U,self.k.open)
        self.k.exited=True;self.k.job_query_fail=True
        self.assertFalse(self.birth.retained_cleanup()['signal_proved']);self.assertIn(self.k.U,self.k.open)
    def test_unknown_wait_and_observer_loss_are_not_orphan_clearance(self):
        self.create();self.k.wait_error=True
        result=self.birth.retained_cleanup();self.assertFalse(result['signal_proved']);self.assertFalse(result['O_loss_recovery_implemented'])
        with self.assertRaises(m.Refused):m.actual_dispatch()
    def test_membership_and_graph_peak_unavailable_or_overcap_refuse(self):
        self.k.membership=False
        with self.assertRaises(m.Refused):self.create()
        self.assertIn(self.k.U,self.k.open)
        self.k.peak[self.k.O]=2*m.GIB
        with self.assertRaises(m.Refused):self.graph()
        self.k.fail='memory'
        with self.assertRaises(m.Refused):self.graph()
    def test_missing_O_or_C_not_exempted_from_combined_graph(self):
        with self.assertRaises(m.Refused):m.combined_peak_witness(self.k,lib,{'U':self.k.U})
        self.assertEqual(self.graph()['conservative_sum_of_exact_process_peaks'],300)
    def test_durable_leaf_completion_after_write_and_late_caller_publication(self):
        self.create();self.k.exited=True;original=self.store.write_receipt
        def slow(*a):r=original(*a);self.clock.value+=11;return r
        self.store.write_receipt=slow
        leaf=m.publish_completed_leaf(self.store,'tool_caller_leaf',self.gate['key'],{'synthetic':True},self.clock)
        self.assertGreaterEqual(leaf['completed_after_publication'],11)
        with self.assertRaises(m.Refused):m.witness_caller_completion(self.birth,leaf,{'stdout':True,'stderr':True},self.graph(),0,self.clock)
    def test_completed_leaf_component_is_not_actual_TC_or_OE(self):
        self.create();self.k.exited=True
        leaf=m.publish_completed_leaf(self.store,'tool_caller_leaf',self.gate['key'],{'synthetic':True},self.clock)
        witness=m.witness_caller_completion(self.birth,leaf,{'stdout':True,'stderr':True},self.graph(),0,self.clock)
        self.assertFalse(witness['actual_TC_qualified'])
        observer=m.assess_observer_interval(0,0,self.clock,lambda:.1)
        self.assertEqual(observer['whole_status'],'NOT_READY');self.assertFalse(observer['external_OE_producer_implemented'])
    def test_missing_EOF_and_live_exit_refuse_caller_component(self):
        self.create();leaf=m.publish_completed_leaf(self.store,'tool_caller_leaf',self.gate['key'],{'synthetic':True},self.clock)
        with self.assertRaises(m.Refused):m.witness_caller_completion(self.birth,leaf,{'stdout':True,'stderr':True},self.graph(),0,self.clock)
        self.k.exited=True
        with self.assertRaises(m.Refused):m.witness_caller_completion(self.birth,leaf,{'stdout':True,'stderr':False},self.graph(),0,self.clock)
    def test_native_SUCCESS_then_late_observer_never_whole_ACCEPTED(self):
        row,_=self.native_success();self.assertEqual(row['state'],'SUCCEEDED');self.clock.value=3
        result=m.assess_observer_interval(0,0,self.clock,lambda:.1)
        self.assertFalse(result['phase_within_2s']);self.assertEqual(result['whole_status'],'NOT_READY')
        self.assertEqual(self.store.get_job(self.gate['key'])['state'],'SUCCEEDED')
    def test_native_receipt_duration_and_terminal_conflict_immutable(self):
        row,ref=self.native_success()
        with self.assertRaises(ops.OperationsError):
            self.store.reconcile(self.gate['key'],'FAILED',.3,{'native_outcome_proof_ref':ref},owner=self.gate['owner'],fence=self.gate['fence'])
        with self.assertRaises(ops.OperationsError):self.store.write_receipt('native_outcome_observation',self.gate['key'],{'changed':True})
    def test_RPC_nonce_sequence_bounds_and_file_pin_drift(self):
        codec=m.FixedFrame('a'*64);frame=codec.encode('IDENTITY',{})
        codec.decode(frame)
        with self.assertRaises(m.Refused):codec.decode(frame)
        with self.assertRaises(m.Refused):m.FixedFrame('b'*64).decode(frame)
        with self.assertRaises(m.Refused):m.FixedFrame('a'*64).encode('CREATE_ANOTHER_U',{})
        with self.assertRaises(m.Refused):m.FixedFrame('a'*64).encode('CALLER_LEAF',{'x':'x'*9000})
        f=Path(self.tmp.name)/'pin.txt';f.write_bytes(b'ab')
        pin={'path':str(f),'bytes':2,'sha256':m.hashlib.sha256(b'ab').hexdigest()};m.verify_file_pins([pin])
        f.write_bytes(b'cd')
        with self.assertRaises(m.Refused):m.verify_file_pins([pin])
    def test_stale_fence_and_additive_reservation_refuse_before_new_birth(self):
        self.gate['fence']+=1
        with self.assertRaises(ops.OperationsError):self.create()
        self.assertFalse(self.birth.attempted)
        r=self.store.register_job('fixture-second',{'synthetic':True},'fixture-epoch');self.store.ready(r['key'])
        with self.assertRaises(ops.OperationsError):self.store.reserve(r['key'],self.gate['owner'],self.gate['fence']-1,10)
    def test_actual_dispatch_always_refuses_missing_mandatory_boundaries(self):
        with self.assertRaisesRegex(m.Refused,'C kernel-RPC'):m.actual_dispatch()
        self.assertGreaterEqual(len(m.NOT_READY_GAPS),5)

if __name__=='__main__':
    OUT=P/'evidence/tool_complete_integration_round4_013'
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures))
    with (OUT/'RAW_TEST_OUTPUT.txt').open('x',encoding='utf-8',newline='\n') as f:f.write(stream.getvalue())
    summary={'status':'COMPONENT_FIXTURES_PASS_COMPLETE_INTEGRATION_NOT_READY' if result.wasSuccessful() else 'FIXTURE_FAILURE_FROZEN_NO_REPAIR_RETEST_FIFTH',
      'cumulative_same_milestone_source_round':4,'additional_candidate':1,'additional_candidates_max':1,'invocations':1,
      'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'internal_import_suite_seconds':time.monotonic()-START,
      'actual_typed_APIs_with_controlled_kernel_only':True,'actual_OperationsStore_temporary_DB_only':True,
      'complete_contract_implemented':False,'actual_dispatch_qualified':False,'mandatory_missing_boundaries':list(m.NOT_READY_GAPS),
      'actual_OS_child_UV_framework_model_installer_hostDB_GPU_network':0,'repair_retest_fifth_or_renamed_route':False}
    with (OUT/'TEST_RESULT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(summary,f,indent=2);f.write('\n')
    print(json.dumps({'status':summary['status'],'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'internal_before_terminal_write_seconds':time.monotonic()-START}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
