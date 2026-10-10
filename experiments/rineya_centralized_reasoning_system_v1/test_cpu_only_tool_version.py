"""Candidate3: one stdlib synthetic retained-ownership and temporary-DB fixture."""
import time
START=time.monotonic()
import ctypes as c
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
P=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
m=load('tool_version_candidate3',P/'cpu_only_tool_version.py')
ops=load('unchanged_canonical_operations',P/'operations.py')

class Clock:
    def __init__(self):self.value=0.
    def __call__(self):return self.value
    def tick(self):self.value+=.01
    def sleep(self,seconds):self.value+=seconds
    def wall(self):return 1800000000.+self.value
class Timer:
    def __init__(self,seconds,fn):assert seconds==10;self.fn=fn;self.daemon=False
    def start(self):pass
    def cancel(self):pass
    def is_alive(self):return False
    def join(self,timeout=None):pass
class Fn:
    def __init__(self,name,k,call):self.name=name;self.k=k;self.call=call
    def __call__(self,*a):
        self.k.clock.tick();self.k.calls.append((self.name,a));return self.call(*a)
class Kernel:
    PH=0x100000077;TH=0x100000033;JH=0x100000088
    def __init__(self,clock):
        self.clock=clock;self.calls=[];self.open=set();self.closed=[];self.limits=None
        self.error=0;self.created=False;self.exited=False;self.job_pid=None
        self.pipe_count=0;self.data={};self.peak=100;self.current=0
        self.fail=None;self.resume_count=1;self.close_fail=None;self.killed=threading.Event()
        self.block_read=False;self.read_entered=threading.Event();self.complete_on_resume=True;self.claim=b'uv 0.0.1 (fixture)\n'
        self.stderr_claim=b'';self.code=0;self.image_claim=m.UV_PATH
        self.terminate_fail=False;self.job_kill_fail=False
        self.terminate_without_signal=False;self.wait_override=None;self.partial_assignment=False
        methods={'CreateJobObjectW':self.create_job,'SetInformationJobObject':self.set_job,
          'QueryInformationJobObject':self.query_job,'AssignProcessToJobObject':self.assign,
          'TerminateJobObject':self.kill_job,'CloseHandle':self.close,
          'CreatePipe':self.pipe,'SetHandleInformation':lambda *a:self.fail!='handleinfo',
          'InitializeProcThreadAttributeList':self.init_attribute,'UpdateProcThreadAttribute':self.update_attribute,
          'DeleteProcThreadAttributeList':lambda *a:None,'CreateProcessW':self.process,
          'GetProcessTimes':self.times,'QueryFullProcessImageNameW':self.image,'ResumeThread':self.resume,
          'WaitForSingleObject':self.wait,'GetExitCodeProcess':self.exitcode,'TerminateProcess':self.terminate,
          'PeekNamedPipe':self.peek,'ReadFile':self.read,'GetProcessMemoryInfo':self.memory}
        for name,fn in methods.items():setattr(self,name,Fn(name,self,fn))
    def create_job(self,*a):self.open.add(self.JH);return self.JH
    def set_job(self,h,kind,p,size):
        assert h==self.JH and kind==9
        self.limits=m.Extended.from_buffer_copy(c.string_at(p,size));return self.fail!='setjob'
    def query_job(self,h,kind,p,size,_):
        if self.fail=='queryjob':return False
        if kind==9:c.memmove(p,c.byref(self.limits),size)
        else:
            q=c.cast(p,c.POINTER(m.Pids)).contents
            q.assigned=q.returned=int(self.job_pid is not None);q.pids[0]=self.job_pid or 0
        return True
    def assign(self,h,p):
        assert h==self.JH and p==self.PH
        if self.fail=='assign':
            if self.partial_assignment:self.job_pid=7
            return False
        self.job_pid=7;return True
    def kill_job(self,*a):
        if self.fail=='jobkill' or self.job_kill_fail:return False
        if self.job_pid is not None:self.exited=True;self.job_pid=None;self.killed.set()
        return True
    def close(self,h):
        if h==self.close_fail:return False
        self.open.discard(h);self.closed.append(h);return True
    def pipe(self,r,w,sa,size):
        assert c.cast(sa,c.POINTER(m.Security)).contents.bInheritHandle
        self.pipe_count+=1
        read=0x100000100+self.pipe_count*2;write=read+1
        c.cast(r,c.POINTER(m.w.HANDLE)).contents.value=read;self.open.add(read)
        if self.fail=='partialpipe' and self.pipe_count==2:return False
        c.cast(w,c.POINTER(m.w.HANDLE)).contents.value=write;self.open.add(write)
        self.data[read]=bytearray();return True
    def init_attribute(self,p,count,flags,size):
        if p is None:c.cast(size,c.POINTER(c.c_size_t)).contents.value=128;self.error=122;return False
        return self.fail!='attributeinit'
    def update_attribute(self,*a):
        assert a[2]==0x20002 and c.sizeof(a[3])==3*c.sizeof(m.w.HANDLE)
        return self.fail!='attributeupdate'
    def process(self,app,command,_,__,inherit,flags,env,cwd,start,info):
        assert app==m.UV_PATH and command.value=='"'+m.UV_PATH+'" --version'
        assert flags & 4 and flags & 0x80000 and flags & 0x400 and inherit
        s=c.cast(start,c.POINTER(m.StartupEx)).contents
        assert s.StartupInfo.cb==c.sizeof(m.StartupEx) and s.StartupInfo.dwFlags==0x100
        self.env=''.join(env);self.child_handles=(s.StartupInfo.hStdInput,s.StartupInfo.hStdOutput,s.StartupInfo.hStdError)
        p=c.cast(info,c.POINTER(m.ProcessInfo)).contents
        assert not p.hProcess and not p.hThread and p.dwProcessId==p.dwThreadId==0
        if self.fail=='create':return False
        if self.fail=='partialcreate':
            p.hProcess=self.PH;p.dwProcessId=7;self.open.add(self.PH);self.created=True;return False
        p.hProcess=self.PH;p.hThread=self.TH;p.dwProcessId=7;p.dwThreadId=17
        self.open.update((self.PH,self.TH));self.created=True
        reads=sorted(self.data);self.stdout=reads[1];self.stderr=reads[2]
        self.data[self.stdout]=bytearray(self.claim)
        self.data[self.stderr]=bytearray(self.stderr_claim)
        return True
    def times(self,h,*times):
        assert h==self.PH
        if self.fail=='times':return False
        f=c.cast(times[0],c.POINTER(m.w.FILETIME)).contents;f.dwHighDateTime=1;f.dwLowDateTime=42
        return True
    def image(self,h,flags,buf,size):
        assert h==self.PH
        if self.fail=='image':return False
        buf.value=self.image_claim;return True
    def resume(self,h):
        assert h==self.TH and h!=7
        if self.resume_count==1 and self.complete_on_resume:
            self.exited=True;self.job_pid=None
        return self.resume_count
    def wait(self,h,timeout):
        assert h==self.PH and timeout==0
        if self.fail=='wait':return 0xffffffff
        if self.wait_override is not None:return self.wait_override
        return 0 if self.exited else 258
    def exitcode(self,h,code):
        if self.fail=='exitcode':return False
        c.cast(code,c.POINTER(m.w.DWORD)).contents.value=self.code;return True
    def terminate(self,h,code):
        assert h==self.PH
        # A failed request has NO successful exit side effects. A successful
        # asynchronous request may also leave signal proof unavailable.
        if self.fail=='terminate' or self.terminate_fail:return False
        if not self.terminate_without_signal:
            self.exited=True;self.job_pid=None;self.killed.set()
        return True
    def peek(self,h,_,__,___,available,____):
        if self.fail=='peek':self.error=5;return False
        data=self.data[h]
        if data:c.cast(available,c.POINTER(m.w.DWORD)).contents.value=len(data);return True
        if self.exited:self.error=109;return False
        return True
    def read(self,h,buf,count,read,_):
        if self.block_read:self.read_entered.set();self.killed.wait(.5)
        if self.fail=='read':return False
        data=bytes(self.data[h][:count]);del self.data[h][:count]
        c.memmove(buf,data,len(data));c.cast(read,c.POINTER(m.w.DWORD)).contents.value=len(data);return True
    def memory(self,h,buf,size):
        assert h==self.PH
        if self.fail=='memory':return False
        value=c.cast(buf,c.POINTER(m.Memory)).contents
        value.WorkingSetSize=self.current;value.PeakWorkingSetSize=self.peak;return True

def sample():return {'age_seconds':0,'host_available_bytes':4*m.GIB,'disk_free_bytes':5*m.GIB}
class Fixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.clock=Clock()
        self.store=ops.OperationsStore(Path(self.tmp.name)/'fixture.sqlite3',Path(self.tmp.name)/'receipts',clock=self.clock.wall)
        row=self.store.register_job('fixture-only',{'synthetic':True,'source_candidate':3},'fixture-epoch')
        self.store.ready('fixture-only');lease=self.store.acquire_lease('fixture-owner',600)
        self.gate=dict(separately_registered=True,source_roles_commit_accepted=True,
          loader_documentary_accepted=True,trusted_metadata_boundary_accepted=True,
          key='fixture-only',owner='fixture-owner',epoch='fixture-epoch',request_hash=row['request_hash'],
          source_commit='SYNTHETIC_NOT_ATTESTATION',contract_sha256='SYNTHETIC',runtime_pin='SYNTHETIC',
          admission_pin='SYNTHETIC',fence=lease['fence'],telemetry_age_seconds=0,
          preworker_available_bytes=4*m.GIB,disk_free_bytes=5*m.GIB,host_reserve_bytes=2*m.GIB,
          gpu_status='UNMEASURED_NOT_QUALIFIED',argv=['--version'],tool_path=m.UV_PATH,tool_sha256=m.UV_SHA,
          canonical_operations_sha256=m.OPS_SHA,physical_paths_verified=True,empty_scoped_paths_verified=True,
          cwd=str(P/'.venv/tool-version-007/work'),temp=str(P/'.venv/tool-version-007/temp'))
        self.kernel=Kernel(self.clock)
        self.guard=None;self.port=None
    def tearDown(self):self.store.close();self.tmp.cleanup()
    def observe(self,telemetry=sample,timer_factory=Timer):
        self.guard=m.Deadline(self.clock,timer_factory)
        def factory(guard):
            self.port=m.WindowsPort(guard,self.kernel,self.kernel,last_error=lambda:self.kernel.error)
            return self.port
        return m.observe_version(self.gate,self.store,factory,telemetry,clock=self.clock,sleep=self.clock.sleep,guard_factory=lambda:self.guard)
    def assert_unknown10(self):
        row=self.store.get_job(self.gate['key']);self.assertEqual(row['state'],'OUTCOME_UNKNOWN')
        self.assertEqual(self.store.budget_snapshot('fixture-epoch')['epoch_charged_seconds'],10)
    def test_typed_success_EOF_early_exit_peak_and_exact_handles(self):
        r=self.observe()
        self.assertEqual(r['status'],'METADATA_OBSERVED_PENDING_EXTERNAL_FINALIZATION')
        self.assertEqual(r['stream_EOF'],{'stdout':True,'stderr':True});self.assertGreater(r['native_elapsed_seconds'],0)
        self.assertEqual(r['resource_samples'][-1]['memory']['peak_rss_bytes'],100)
        self.assertFalse(r['source_or_launch_qualified']);self.assertFalse(r['canonical_terminal_published'])
        self.assert_unknown10();self.assertFalse(self.kernel.open)
        self.assertEqual(c.sizeof(m.w.HANDLE),c.sizeof(c.c_void_p))
        self.assertEqual(m.ProcessInfo.hThread.offset,c.sizeof(c.c_void_p))
        self.assertEqual(set(s.split('=')[0] for s in self.kernel.env.split('\0') if s),{'PATH','SystemRoot','WINDIR','TEMP','TMP'})
        self.assertTrue(all(h>2**32 for h in self.kernel.child_handles))
        self.assertEqual(self.kernel.CreateProcessW.restype,m.w.BOOL)
        self.assertEqual(self.kernel.CreateProcessW.argtypes[-1],c.POINTER(m.ProcessInfo))
        self.assertEqual(self.kernel.ResumeThread.argtypes,[m.w.HANDLE])
        self.assertEqual(self.kernel.GetProcessMemoryInfo.argtypes,[m.w.HANDLE,c.c_void_p,m.w.DWORD])
    def test_canonical_external_observer_nonzero_transition_and_terminal_immutability(self):
        r=self.observe();self.assert_unknown10()
        actual=r['native_elapsed_seconds'];ref=r['outcome_observation_ref']
        row=self.store.reconcile(self.gate['key'],'SUCCEEDED',actual,{'native_outcome_proof_ref':ref,'synthetic_external_complete_observer':True},
                                 owner=self.gate['owner'],fence=self.gate['fence'])
        self.assertEqual(row['state'],'SUCCEEDED');self.assertEqual(row['actual_seconds'],actual)
        with self.assertRaises(ops.OperationsError):
            self.store.reconcile(self.gate['key'],'FAILED',actual,{'native_outcome_proof_ref':ref},owner=self.gate['owner'],fence=self.gate['fence'])
    def test_canonical_zero_wrong_fence_checksum_and_request_refused(self):
        r=self.observe();ref=r['outcome_observation_ref'];actual=r['native_elapsed_seconds']
        for seconds,owner,fence,bound in [(0,self.gate['owner'],self.gate['fence'],ref),
            (actual,self.gate['owner'],self.gate['fence']+1,ref),(actual,'wrong-owner',self.gate['fence'],ref),
            (actual,self.gate['owner'],self.gate['fence'],{**ref,'sha256':'0'*64})]:
            with self.assertRaises(ops.OperationsError):
                self.store.reconcile(self.gate['key'],'SUCCEEDED',seconds,{'native_outcome_proof_ref':bound},owner=owner,fence=fence)
        path=Path(ref['path']);data=json.loads(path.read_bytes());data['payload']['request_hash']='0'*64
        raw=ops.canonical(data);path.write_bytes(raw)
        wrong={'path':str(path),'sha256':m.hashlib.sha256(raw).hexdigest()}
        with self.assertRaises(ops.OperationsError):
            self.store.reconcile(self.gate['key'],'SUCCEEDED',actual,{'native_outcome_proof_ref':wrong},owner=self.gate['owner'],fence=self.gate['fence'])
        self.assert_unknown10()
    def test_canonical_wrong_worker_exit_and_elapsed_proofs_refused(self):
        r=self.observe();ref=r['outcome_observation_ref'];actual=r['native_elapsed_seconds']
        original=json.loads(Path(ref['path']).read_bytes())['payload']
        for index,changes in enumerate(({'worker_identity':{**original['worker_identity'],'pid':8}},
                                       {'process_exited':False},{'elapsed_native_seconds':actual+1})):
            bound=self.store.write_receipt('wrong-proof-'+str(index),self.gate['key'],{**original,**changes})
            path=Path(bound['path']);data=json.loads(path.read_bytes());data['kind']='native_outcome_observation'
            raw=ops.canonical(data);path.write_bytes(raw);bound['sha256']=m.hashlib.sha256(raw).hexdigest()
            with self.assertRaises(ops.OperationsError):
                self.store.reconcile(self.gate['key'],'SUCCEEDED',actual,{'native_outcome_proof_ref':bound},
                                     owner=self.gate['owner'],fence=self.gate['fence'])
        self.assert_unknown10()
    def test_canonical_unknown_native_cost_preserves_full10(self):
        r=self.observe();ref=r['outcome_observation_ref']
        path=Path(ref['path']);data=json.loads(path.read_bytes());data['payload']['elapsed_native_seconds']=None
        raw=ops.canonical(data);path.write_bytes(raw);ref={**ref,'sha256':m.hashlib.sha256(raw).hexdigest()}
        with self.assertRaises(ops.OperationsError):
            self.store.reconcile(self.gate['key'],'SUCCEEDED',None,{'native_outcome_proof_ref':ref},
                                 owner=self.gate['owner'],fence=self.gate['fence'])
        row=self.store.reconcile(self.gate['key'],'ABORTED',None,{'native_outcome_proof_ref':ref},
                                 owner=self.gate['owner'],fence=self.gate['fence'])
        self.assertEqual(row['state'],'ABORTED');self.assertIsNone(row['actual_seconds'])
        self.assertEqual(self.store.budget_snapshot('fixture-epoch')['epoch_charged_seconds'],10)
    def test_first_identity_failure_terminates_created_handle_and_closes_everything(self):
        self.kernel.fail='times';r=self.observe()
        self.assertNotEqual(r['status'],'METADATA_OBSERVED_PENDING_EXTERNAL_FINALIZATION')
        self.assertTrue(self.kernel.killed.is_set());self.assertFalse(self.kernel.open);self.assert_unknown10()
        self.assertFalse(any(name=='ResumeThread' for name,_ in self.kernel.calls))
    def test_constructor_Job_failure_preserves_cleanup_ownership(self):
        self.kernel.fail='setjob';r=self.observe()
        self.assertFalse(self.kernel.open);self.assertIn(self.kernel.JH,self.kernel.closed);self.assert_unknown10()
        self.assertFalse(self.kernel.created)
    def test_constructor_signature_failure_before_Job_creates_no_unowned_handle(self):
        del self.kernel.ReadFile
        r=self.observe()
        self.assertFalse(self.kernel.open);self.assertFalse(self.kernel.created);self.assert_unknown10()
        self.assertNotEqual(r['status'],'METADATA_OBSERVED_PENDING_EXTERNAL_FINALIZATION')
    def test_first_image_failure_terminates_created_handle_without_identity(self):
        self.kernel.fail='image';r=self.observe()
        self.assertTrue(self.kernel.killed.is_set());self.assertFalse(self.kernel.open);self.assert_unknown10()
        self.assertFalse(any(name=='ResumeThread' for name,_ in self.kernel.calls))
    def test_wrong_image_identity_never_assigns_or_resumes(self):
        self.kernel.image_claim='C:/fixture/wrong.exe';r=self.observe()
        self.assertIn('image drift',r['reason']);self.assertTrue(self.kernel.killed.is_set());self.assert_unknown10()
        self.assertFalse(any(name in ('AssignProcessToJobObject','ResumeThread') for name,_ in self.kernel.calls))
    def test_partial_pipe_failure_closes_returned_handles_independently(self):
        self.kernel.fail='partialpipe';r=self.observe()
        self.assertFalse(self.kernel.open);self.assertFalse(self.kernel.created);self.assert_unknown10()
    def test_attribute_failure_closes_all_pipe_and_Job_handles(self):
        self.kernel.fail='attributeupdate';r=self.observe()
        self.assertFalse(self.kernel.open);self.assertFalse(self.kernel.created);self.assert_unknown10()
        self.assertTrue(any(name=='DeleteProcThreadAttributeList' for name,_ in self.kernel.calls))
    def test_attribute_initialization_failure_closes_pipes_without_delete(self):
        self.kernel.fail='attributeinit';r=self.observe()
        self.assertFalse(self.kernel.open);self.assertFalse(self.kernel.created);self.assert_unknown10()
        self.assertFalse(any(name=='DeleteProcThreadAttributeList' for name,_ in self.kernel.calls))
    def test_CreateProcess_failure_closes_attribute_pipes_and_Job(self):
        self.kernel.fail='create';r=self.observe()
        self.assertFalse(self.kernel.open);self.assertFalse(self.kernel.created);self.assert_unknown10()
        self.assertTrue(any(name=='DeleteProcThreadAttributeList' for name,_ in self.kernel.calls))
    def test_partial_CreateProcess_failure_preserves_exact_returned_handle(self):
        self.kernel.fail='partialcreate';r=self.observe()
        self.assertTrue(self.kernel.killed.is_set());self.assertIn(self.kernel.PH,self.kernel.closed)
        self.assertFalse(self.kernel.open);self.assert_unknown10()
        self.assertFalse(any(name=='ResumeThread' for name,_ in self.kernel.calls))
    def test_handle_inheritance_failure_closes_every_owned_pipe(self):
        self.kernel.fail='handleinfo';r=self.observe()
        self.assertFalse(self.kernel.open);self.assertFalse(self.kernel.created);self.assert_unknown10()
    def test_assignment_refusal_does_not_assign_or_resume_and_kills_known_handle(self):
        self.kernel.fail='assign';r=self.observe()
        self.assertTrue(self.kernel.killed.is_set());self.assertFalse(self.kernel.open);self.assert_unknown10()
        self.assertFalse(any(name=='ResumeThread' for name,_ in self.kernel.calls))
    def test_resume_count_failure_closes_exact_primary_thread(self):
        self.kernel.resume_count=0;r=self.observe()
        self.assertIn(self.kernel.TH,self.kernel.closed);self.assertTrue(self.kernel.killed.is_set());self.assert_unknown10()
    def test_resume_error_sentinel_closes_exact_primary_thread(self):
        self.kernel.resume_count=0xffffffff;r=self.observe()
        self.assertIn(self.kernel.TH,self.kernel.closed);self.assertTrue(self.kernel.killed.is_set());self.assert_unknown10()
    def test_unknown_pipe_error_is_not_EOF_or_complete_output(self):
        self.kernel.fail='peek';r=self.observe()
        self.assertIn('unknown EOF',r['reason']);self.assertNotEqual(r['stream_EOF'],{'stdout':True,'stderr':True})
        self.assertFalse(self.kernel.open);self.assert_unknown10()
    def test_missing_pipe_EOF_hits_deadline_without_metadata_success(self):
        original=self.kernel.peek
        def no_EOF(h,*args):
            if self.kernel.data[h]:return original(h,*args)
            return True
        self.kernel.PeekNamedPipe.call=no_EOF
        r=self.observe()
        self.assertTrue(r['combined_deadline_exceeded']);self.assertNotEqual(r['stream_EOF'],{'stdout':True,'stderr':True})
        self.assertNotEqual(r['status'],'METADATA_OBSERVED_PENDING_EXTERNAL_FINALIZATION');self.assert_unknown10()
    def test_pipe_read_error_and_cleanup_close_error_do_not_skip_other_handles(self):
        self.kernel.fail='read';self.kernel.close_fail=0x100000104
        r=self.observe()
        self.assertTrue(r['cleanup_errors']);self.assertEqual(self.kernel.open,{self.kernel.close_fail})
        self.assertIn(self.kernel.PH,self.kernel.closed);self.assertIn(self.kernel.TH,self.kernel.closed)
        self.assert_unknown10()
    def test_early_exit_missing_peak_truthfully_refuses(self):
        self.kernel.fail='memory';r=self.observe()
        self.assertEqual(r['resource_samples'],[]);self.assertNotEqual(r['status'],'METADATA_OBSERVED_PENDING_EXTERNAL_FINALIZATION');self.assert_unknown10()
    def test_peak_over_cap_refuses(self):
        self.kernel.peak=2*m.GIB+1;r=self.observe();self.assertIn('RSS cap',r['reason']);self.assert_unknown10()
    def test_stale_telemetry_refuses_and_keeps_full_reservation(self):
        r=self.observe(telemetry=lambda:{**sample(),'age_seconds':6})
        self.assertIn('stale/host/disk',r['reason']);self.assert_unknown10()
    def test_combined_output_overflow_has_only_one_sentinel(self):
        self.kernel.claim=b'x'*8192;self.kernel.stderr_claim=b'y'*8192
        r=self.observe();self.assertIn('combined output overflow',r['reason']);self.assert_unknown10()
        self.assertEqual(len(bytes.fromhex(r['raw_stdout_hex']))+len(bytes.fromhex(r['raw_stderr_hex'])),8193)
    def test_malformed_extra_line_is_not_release_key(self):
        self.kernel.claim=b'uv 0.0.1\nextra\n';r=self.observe()
        self.assertIn('malformed',r['reason']);self.assertNotIn('release_lookup_key',r);self.assert_unknown10()
    def test_stderr_refuses_metadata_success(self):
        self.kernel.stderr_claim=b'warning\n';r=self.observe()
        self.assertIn('unexpected stderr',r['reason']);self.assert_unknown10()
    def test_nonzero_exit_refuses_metadata_success(self):
        self.kernel.code=3;r=self.observe()
        self.assertIn('nonzero exit',r['reason']);self.assert_unknown10()
    def test_independent_timer_fires_while_typed_ReadFile_mainline_blocked(self):
        self.kernel.block_read=True;self.kernel.complete_on_resume=False
        def factory(seconds,fn):
            self.assertEqual(seconds,10)
            def trip_after_read_enters():
                self.kernel.read_entered.wait(.5);fn()
            # Deterministic controlled firing while ReadFile is blocked. The
            # production Deadline still requests exactly10s from its factory.
            return threading.Timer(.01,trip_after_read_enters)
        r=self.observe(timer_factory=factory)
        self.assertTrue(self.guard.expired.is_set());self.assertFalse(r['canonical_terminal_published']);self.assert_unknown10()
        self.assertTrue(self.kernel.killed.is_set())
        self.assertTrue(any(name=='TerminateJobObject' for name,_ in self.kernel.calls))
        self.assertFalse(self.kernel.open)
    def test_late_durable_receipt_is_unknown_with_distinct_native_and_wall_intervals(self):
        original=self.store.write_receipt
        def slow(*a,**kw):
            ref=original(*a,**kw);self.clock.value+=12;return ref
        self.store.write_receipt=slow
        r=self.observe()
        self.assertEqual(r['status'],'LATE_OR_UNKNOWN_PENDING_EXTERNAL_OBSERVER')
        self.assertGreater(r['caller_wall_through_internal_finalization_seconds'],10)
        self.assertLess(r['native_elapsed_seconds'],10);self.assertFalse(r['canonical_terminal_published']);self.assert_unknown10()
    def test_wrong_admission_or_fence_never_reserves(self):
        self.gate['argv']=['--help'];r=self.observe()
        self.assertEqual(self.store.get_job(self.gate['key'])['state'],'READY')
        self.assertEqual(self.store.budget_snapshot('fixture-epoch')['epoch_charged_seconds'],0)
    def assert_reachable_exact_recovery(self,r,*,live=True):
        owner=r['recovery_owner'];self.assertIs(owner,self.guard)
        self.assertIs(owner.port,self.port or self.guard.port)
        self.assertEqual(owner.port.handles['process'],self.kernel.PH)
        self.assertEqual(r['retained_ownership']['process_handle'],self.kernel.PH)
        self.assertIn(self.kernel.PH,self.kernel.open);self.assertNotIn(self.kernel.PH,self.kernel.closed)
        self.assertTrue(callable(owner.port.terminate_created))
        self.assertTrue(callable(owner.port.cleanup))
        self.assertTrue(r['retained_ownership']['in_memory_only_not_durable_or_transferred'])
        if live:self.assertFalse(self.kernel.exited)
        self.assertFalse(r['canonical_terminal_published']);self.assert_unknown10()
        self.assertFalse(any(name=='ResumeThread' for name,_ in self.kernel.calls))
        self.assertFalse(any(name=='TerminateProcess' and args[0]!=self.kernel.PH for name,args in self.kernel.calls))
        self.assertTrue(all(not v for k,v in owner.port.handles.items() if k!='process'))
        self.assertIsNone(owner.port.attribute)
    def test_preJob_first_identity_and_failed_termination_keep_live_exact_owner(self):
        self.kernel.fail='times';self.kernel.terminate_fail=True;r=self.observe()
        self.assert_reachable_exact_recovery(r);self.assertIsNone(self.kernel.job_pid)
        self.assertFalse(self.kernel.killed.is_set());self.assertTrue(r['timer_quiescent_at_return'])
        self.assertEqual(sum(name=='TerminateProcess' for name,_ in self.kernel.calls),1)
        self.assertIn(self.kernel.JH,self.kernel.open)
    def test_preJob_assignment_refusal_failed_termination_keep_live_owner(self):
        self.kernel.fail='assign';self.kernel.terminate_fail=True;r=self.observe()
        self.assert_reachable_exact_recovery(r);self.assertIsNone(self.kernel.job_pid)
        self.assertFalse(any(name=='CloseHandle' and args[0] in (self.kernel.PH,self.kernel.JH) for name,args in self.kernel.calls))
    def test_partial_assignment_failed_Job_and_process_termination_keep_both_owners(self):
        self.kernel.fail='assign';self.kernel.partial_assignment=True
        self.kernel.job_kill_fail=True;self.kernel.terminate_fail=True;r=self.observe()
        self.assert_reachable_exact_recovery(r);self.assertEqual(self.kernel.job_pid,7)
        self.assertEqual(r['retained_ownership']['Job_handle'],self.kernel.JH)
        self.assertFalse(r['retained_ownership']['owned_Job_empty_proved'])
    def test_partial_creation_failed_termination_keeps_exact_process_owner(self):
        self.kernel.fail='partialcreate';self.kernel.terminate_fail=True;r=self.observe()
        self.assert_reachable_exact_recovery(r)
    def test_unknown_exact_wait_keeps_live_owner_without_termination_guess(self):
        self.kernel.fail='times';self.kernel.wait_override=0xffffffff;r=self.observe()
        self.assert_reachable_exact_recovery(r)
        self.assertFalse(any(name=='TerminateProcess' for name,_ in self.kernel.calls))
    def test_successful_termination_request_without_signal_retains_live_owner(self):
        self.kernel.fail='times';self.kernel.terminate_without_signal=True;r=self.observe()
        self.assert_reachable_exact_recovery(r)
        self.assertEqual(sum(name=='TerminateProcess' for name,_ in self.kernel.calls),1)
        self.assertFalse(r['retained_ownership']['exact_process_signal_proved'])
    def test_deadline_crosses_CreateProcess_and_failed_termination_retains_owner(self):
        original=self.kernel.process
        def late_create(*args):
            ok=original(*args);self.clock.value=11;return ok
        self.kernel.CreateProcessW.call=late_create;self.kernel.terminate_fail=True
        r=self.observe();self.assert_reachable_exact_recovery(r)
        self.assertTrue(r['combined_deadline_exceeded']);self.assertIsNone(r['outcome_observation_ref'])
    def test_Job_query_failure_after_signaled_process_keeps_recovery_handles(self):
        original=self.kernel.times
        def failed_identity(*args):
            self.kernel.fail='queryjob';return False
        self.kernel.GetProcessTimes.call=failed_identity
        r=self.observe();self.assert_reachable_exact_recovery(r,live=False)
        self.assertTrue(r['retained_ownership']['exact_process_signal_proved'])
        self.assertFalse(r['retained_ownership']['owned_Job_empty_proved'])
    def test_unquiescent_timer_keeps_targets_reachable_and_closes_safe_resources(self):
        class BusyTimer(Timer):
            def is_alive(self):return True
        self.kernel.fail='times';r=self.observe(timer_factory=BusyTimer)
        self.assert_reachable_exact_recovery(r);self.assertFalse(r['timer_quiescent_at_return'])
        self.assertIn(self.kernel.JH,self.kernel.open)
        self.assertFalse(any(name in ('TerminateProcess','TerminateJobObject') for name,_ in self.kernel.calls))
        self.assertIs(r['recovery_owner'].timer.fn.__self__,self.guard)
    def test_no_port_return_reports_timer_quiescence_and_reachable_timer_owner(self):
        class BusyTimer(Timer):
            def is_alive(self):return True
        self.gate['argv']=['--help'];r=self.observe(timer_factory=BusyTimer)
        self.assertFalse(r['timer_quiescent_at_return']);self.assertIs(r['recovery_owner'],self.guard)
        self.assertIsNone(self.guard.port);self.assertFalse(self.kernel.created)
        self.assertEqual(self.store.budget_snapshot('fixture-epoch')['epoch_charged_seconds'],0)
    def test_timer_failed_preJob_termination_is_not_retried_by_cleanup(self):
        entered=threading.Event();finished=threading.Event()
        def blocked_identity(*args):
            entered.set();finished.wait(.5);return False
        self.kernel.GetProcessTimes.call=blocked_identity;self.kernel.terminate_fail=True
        def factory(seconds,fn):
            self.assertEqual(seconds,10)
            def trip_after_identity_enters():
                entered.wait(.5)
                try:fn()
                finally:finished.set()
            return threading.Timer(.01,trip_after_identity_enters)
        r=self.observe(timer_factory=factory);self.assert_reachable_exact_recovery(r)
        self.assertTrue(self.guard.expired.is_set());self.assertTrue(r['timer_quiescent_at_return'])
        self.assertEqual(sum(name=='TerminateProcess' for name,_ in self.kernel.calls),1)

if __name__=='__main__':
    OUT=P/'evidence/cpu_only_tool_version_source_009'
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures))
    raw=stream.getvalue()
    with (OUT/'RAW_TEST_OUTPUT.txt').open('x',encoding='utf-8',newline='\n') as f:f.write(raw)
    summary={'status':'PASS_SYNTHETIC_FINAL_CANDIDATE3_ONLY' if result.wasSuccessful() else 'FAIL_FROZEN_QUARANTINE3_OF3_NO_REPAIR_OR_RETEST',
      'candidate':3,'default_candidates_max':3,'invocations':1,'tests':result.testsRun,
      'failures':len(result.failures),'errors':len(result.errors),'suite_and_import_elapsed_seconds':time.monotonic()-START,
      'typed_actual_adapter_with_controlled_kernel_psapi_timer_streams':True,
      'OperationsStore_actual_interface_on_temporary_fixture_DB_only':True,
      'synthetic_external_observer_transition_not_actual_provenance':True,
      'actual_UV_Python_child_OS_process_framework_model_installer_hostDB_GPU_network':0,
      'launch_ready':False,'irreducible_external_complete_observer_and_caller_trust_not_qualified':True}
    with (OUT/'TEST_RESULT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(summary,f,indent=2);f.write('\n')
    print(json.dumps({'status':summary['status'],'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                      'elapsed_before_complete_terminal_write_seconds':time.monotonic()-START}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
