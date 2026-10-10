"""Exactly one stdlib synthetic fixture invocation; never construct live port."""
import time
START=time.monotonic()
import ctypes as c
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate_tool_version',P/'cpu_only_tool_version.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=P/'evidence/cpu_only_tool_version_source_007'

class Fn:
    def __init__(self,call):self.call=call
    def __call__(self,*a):return self.call(*a)
class Kernel:
    def __init__(self):
        self.limits=None;self.pid=None;self.events=[];self.fail_set=False;self.fail_assign=False;self.child=False
        for n,f in [('CreateJobObjectW',lambda *a:88),('SetInformationJobObject',self.set),
                    ('QueryInformationJobObject',self.query),('AssignProcessToJobObject',self.assign),
                    ('TerminateJobObject',self.kill),('CloseHandle',self.close)]:setattr(self,n,Fn(f))
    def set(self,h,kind,p,size):
        self.events.append('set');self.limits=m.Extended.from_buffer_copy(c.string_at(p,size));return not self.fail_set
    def query(self,h,kind,p,size,_):
        if kind==9:c.memmove(p,c.byref(self.limits),size)
        else:
            q=c.cast(p,c.POINTER(m.Pids)).contents
            q.assigned=q.returned=(2 if self.child else int(self.pid is not None))
            q.pids[0]=self.pid or 0
            if self.child:q.pids[1]=99
        return True
    def assign(self,h,process):self.events.append('assign');self.pid=7;return not self.fail_assign
    def kill(self,*a):self.events.append('kill');self.pid=None;return True
    def close(self,*a):self.events.append('close');return True
class Process:
    pid=7;_handle=77
    def __init__(self):self.returncode=None;self.stdout=self.stderr=None
    def poll(self):return self.returncode
    def kill(self):self.returncode=2
class Clock:
    def __init__(self):self.value=0
    def __call__(self):return self.value
    def sleep(self,s):self.value+=s
class Store:
    def __init__(self):self.events=[];self.clock=lambda:100.;self.drift=False
    def _require_lease(self,*a):
        if self.drift:raise m.Refused('lease drift')
    def get_job(self,key):return {'request_hash':'request','epoch':'epoch','state':'READY'}
    def reserve(self,*a):self.events.append(('reserve',a[-1]))
    def mark_running(self,*a):self.events.append(('running',a[-1]))
    def mark_unknown(self,*a):self.events.append(('unknown',a[-1]))
    def write_receipt(self,*a):self.events.append(('receipt',a));return {'path':'synthetic','sha256':'0'*64}
    def reconcile(self,*a,**kw):self.events.append(('reconcile',a[1],a[2]))
class Port:
    def __init__(self,clock):
        self.kernel=Kernel();self.job=m.OwnedJob(self.kernel);self.process=Process();self.clock=clock
        self.events=[];self.payload=b'uv 0.1.2 (synthetic)\n';self.sent=False;self.bad_image=False;self.delay=0;self.rss_value=0
    def spawn_suspended(self,cwd,env):self.events.append(('suspended',dict(env)));return self.process
    def identity(self,p):return {'pid':7,'creation_time':'1','image':'wrong.exe' if self.bad_image else m.UV_PATH,'command_hash':'a'*64}
    def resume(self,p):
        assert self.kernel.pid==7 and self.kernel.events.index('assign')>self.kernel.events.index('set')
        self.events.append(('resume',));self.clock.value+=self.delay;p.returncode=0;self.kernel.pid=None
    def output(self,p,left):
        if self.sent:return []
        self.sent=True;return [('stdout',self.payload)]
    def rss(self,p):return self.rss_value
    def kill_prejob(self,p,i):self.events.append(('prejobkill',i));p.kill()
    def close_pipes(self,p):self.events.append(('pipesclosed',))

def gate():
    return dict(separately_registered=True,source_roles_commit_accepted=True,loader_documentary_accepted=True,
      trusted_metadata_boundary_accepted=True,key='synthetic',owner='owner',epoch='epoch',request_hash='request',
      source_commit='synthetic',contract_sha256='synthetic',runtime_pin='synthetic',admission_pin='synthetic',fence=1,
      telemetry_age_seconds=0,preworker_available_bytes=4*m.GIB,disk_free_bytes=5*m.GIB,host_reserve_bytes=2*m.GIB,
      gpu_status='UNMEASURED_NOT_QUALIFIED',argv=['--version'],tool_path=m.UV_PATH,tool_sha256=m.UV_SHA,
      canonical_operations_sha256=m.OPS_SHA,physical_paths_verified=True,empty_scoped_paths_verified=True,
      cwd=str(P/'.venv/tool-version-007/work'),temp=str(P/'.venv/tool-version-007/temp'))
def sample():return dict(age_seconds=0,host_available_bytes=4*m.GIB,disk_free_bytes=5*m.GIB)

class Fixtures(unittest.TestCase):
    def run_case(self,change=None,modify_gate=None,telemetry=sample):
        clock=Clock();port=Port(clock);store=Store();g=gate()
        if change:change(port,store)
        if modify_gate:modify_gate(g)
        result=m.observe_version(g,store,lambda:port,telemetry,clock=clock,sleep=clock.sleep)
        return result,port,store
    def test_typed_Job_limit_assignment_query_and_close(self):
        k=Kernel();job=m.OwnedJob(k);job.assign(Process());proof=job.verify_limits()
        self.assertEqual(proof['ActiveProcessLimit'],1);self.assertEqual(k.limits.BasicLimitInformation.LimitFlags,m.FLAGS)
        self.assertEqual(k.limits.JobMemoryLimit,2*m.GIB)
        k.child=True
        with self.assertRaises(m.Refused):job.pids()
        job.close();self.assertEqual(k.events[-1],'close')
        k=Kernel();k.fail_set=True
        with self.assertRaises(m.Refused):m.OwnedJob(k)
        self.assertEqual(k.events[-1],'close')
    def test_actual_spawn_adapter_arguments_are_suspended_and_literal(self):
        port=object.__new__(m.WindowsPort)
        with patch.object(m.subprocess,'Popen',return_value=Process()) as fake:
            env={'SystemRoot':'C:/Windows'};port.spawn_suspended('synthetic-empty',env)
            a=fake.call_args;self.assertEqual(a.args[0],[m.UV_PATH,'--version']);self.assertFalse(a.kwargs['shell'])
            self.assertEqual(a.kwargs['env'],env);self.assertEqual(a.kwargs['creationflags']&4,4);self.assertTrue(a.kwargs['close_fds'])
    def test_success_metadata_is_not_publisher_proof_accounted(self):
        r,p,s=self.run_case();self.assertEqual(r['status'],'SELF_REPORTED_VERSION_ONLY');self.assertFalse(r['publisher_source_or_setup_qualified'])
        self.assertEqual(s.events[0],('reserve',10));self.assertTrue(any(e[0]=='reconcile' for e in s.events));self.assertEqual(p.events[1][0],'resume')
    def test_wrong_boundary_tool_argv_admission_never_reserves(self):
        for key,value in [('argv',['--help']),('tool_sha256','wrong'),('trusted_metadata_boundary_accepted',False),('telemetry_age_seconds',6),('disk_free_bytes',0)]:
            clock=Clock();s=Store();g=gate();g[key]=value
            with self.assertRaises(m.Refused):m.observe_version(g,s,lambda:None,sample,clock=clock,sleep=clock.sleep)
            self.assertEqual(s.events,[])
    def test_assignment_refusal_and_wrong_image_prejob_cleanup_unknown(self):
        for fn in [lambda p,s:setattr(p.kernel,'fail_assign',True),lambda p,s:setattr(p,'bad_image',True)]:
            r,p,s=self.run_case(fn);self.assertEqual(r['status'],'FAILED_OR_UNKNOWN');self.assertTrue(any(e[0]=='prejobkill' for e in p.events))
            self.assertTrue(any(e[0]=='unknown' for e in s.events));self.assertFalse(any(e[0]=='resume' for e in p.events))
    def test_output_late_malformed_stale_and_lease_refuse(self):
        changes=[lambda p,s:setattr(p,'payload',b'x'*8193),lambda p,s:setattr(p,'payload',b'uv 1.2.3\nuv 1.2.4\n'),lambda p,s:setattr(p,'delay',10)]
        for change in changes:
            r,p,s=self.run_case(change);self.assertEqual(r['status'],'FAILED_OR_UNKNOWN');self.assertFalse(any(e[0]=='reconcile' and e[1]=='SUCCEEDED' for e in s.events))
        r,p,s=self.run_case(telemetry=lambda:{**sample(),'age_seconds':6});self.assertEqual(r['status'],'FAILED_OR_UNKNOWN')
        clock=Clock();s=Store();s.drift=True
        with self.assertRaises(m.Refused):m.observe_version(gate(),s,lambda:None,sample,clock=clock,sleep=clock.sleep)
        self.assertEqual(s.events,[])
    def test_unknown_exit_retains_full_reservation(self):
        def change(p,s):
            p.job.pids=lambda:[7];p.delay=10
        r,p,s=self.run_case(change);self.assertFalse(r['process_exited']);self.assertEqual(r['reservation_retained_seconds'],10)
        self.assertFalse(any(e[0]=='reconcile' for e in s.events))

if __name__=='__main__':
    import hashlib,io
    stream=io.StringIO();runner=unittest.TextTestRunner(stream=stream,verbosity=2)
    result=runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures))
    raw=stream.getvalue();(OUT/'RAW_FIXTURE_OUTPUT.txt').write_text(raw,encoding='utf-8',newline='\n')
    summary={'status':'PASS_SYNTHETIC_ONLY' if result.wasSuccessful() else 'FAIL_FROZEN_NO_RERUN',
      'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
      'tool_framework_model_hostDB_GPU_process_actions':0,'source_candidate':1,
      'elapsed_before_receipt_seconds':time.monotonic()-START,'whole_stage_monotonic_start':START,
      'real_Windows_interface':'OwnedJob typed calls and WindowsPort suspended Popen argument adapter, controlled fake kernel/process; no actual Win32/UV invocation',
      'known_gaps':['WindowsPort identity/resume/pipe/RSS actual calls were not exercised against an OS or complete fake-kernel fixture; source remains NOT_READY.', 'Caller receipt/clean-commit gates and trusted side-effect scope remain independently unresolved.']}
    (OUT/'FIXTURE_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    entries=[]
    for p in [P/'cpu_only_tool_version.py',P/'test_cpu_only_tool_version.py',P/'contracts/CPU_ONLY_TOOL_VERSION_V1.json',*sorted(OUT.iterdir())]:
        if p.is_file():entries.append({'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    freeze={'status':'FROZEN_NOT_READY_SOURCE_CANDIDATE1_STOP_FOR_REVIEW','files':entries,
      'fixture_status':summary['status'],'elapsed_through_raw_result_and_source_hashing_seconds':time.monotonic()-START,
      'no_actual_launch_or_source_qualification':True}
    (OUT/'FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'fixture_status':summary['status'],'tests':result.testsRun,
                      'whole_stage_elapsed_through_final_freeze_seconds':time.monotonic()-START}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
