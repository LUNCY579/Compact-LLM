"""Finite inert ownership/resource fixtures; no process/device APIs."""
from pathlib import Path
import hashlib,importlib.util,time
HERE=Path(__file__).resolve().parent
def module(path):
    path=Path(path);spec=importlib.util.spec_from_file_location('owned_fixture_'+path.stem,path);value=importlib.util.module_from_spec(spec);exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),value.__dict__);return value
S=module(HERE/'prefix_input_owned_supervisor.py');C=S.C
F=module(C.CORE/'fixture_support.py')
def source_pins():
    paths=list(HERE.glob('*.py'))+[C.CORE/name for name in C.CORE_PINS]+[C.PACKAGE/'inference_qual_worker.py']
    return {str(p.absolute()):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(C.read_bytes(p)).hexdigest(),'role':'python_source'} for p in sorted(paths)}
class FakeHost:
    def __init__(self,output,faults=None):
        self.output=Path(output);self.faults={} if faults is None else dict(faults);self.started=time.monotonic()-2.01;self.offset=-2.01
        self.owner={'pid':12345,'creation_filetime_100ns':134000000000000001};self.assigned=False;self.resumed=False;self.calls=[];self.job_name='CPU_FAKE_OWNED_JOB';self.supervisor_pid=12344;self.cpu=[];self.gpu=[]
    def clock(self):return time.monotonic()+self.offset
    def wall(self):return time.time()
    def result(self,name,default):
        self.calls.append(name);value=self.faults.get(name,default)
        if isinstance(value,BaseException):raise value
        return value
    def observation(self):
        now=self.clock();default={'monotonic':now,'available_bytes':8*1024**3,'worker_RSS_bytes':512*1024**2,'GPU_monotonic':now,'free_bytes':16*1024**3,'total_bytes':24*1024**3,'GPU_identity':'GPU-00000000-0000-0000-0000-000000000001','disk_free_bytes':64*1024**3}
        return self.result('observation',default)
    def wait_admission(self):self.offset=0
    def create_suspended(self):return self.result('create_suspended',dict(self.owner))
    def assign(self,owner):
        result=self.result('assign',True);self.assigned=result is True;return result
    def membership(self,owner):return self.result('membership',self.assigned and self.resumed and owner==self.owner)
    def resume(self,owner):
        if not self.assigned:raise ValueError('fixture cannot resume before Job assignment')
        result=self.result('resume',True);self.resumed=result is True;return result
    def owned_resources(self,name):return {'tree_rss_bytes':self.result('tree_RSS',512*1024**2)}
    def available(self):return self.result('available',8*1024**3)
    def heartbeat(self,path):return {'unix_seconds':self.wall(),'supervisor_pid':self.supervisor_pid}
    def guard_sample(self,stage):
        now=self.clock();self.cpu.append({'monotonic':now,'available_bytes':self.available(),'worker_RSS_bytes':self.owned_resources(self.job_name)['tree_rss_bytes']});self.gpu.append({'monotonic':now,'free_bytes':16*1024**3,'total_bytes':24*1024**3})
    def worker_dead(self,owner):return self.result('worker_dead',True)
    def join_samplers(self,owner):return self.result('join_samplers',True)
    def close_NVML(self,owner):return self.result('close_NVML',True)
    def close_Job(self,owner):return self.result('close_Job',True)
    def release_reservation(self,owner):return self.result('release_reservation',True)
    def close_lease(self,owner):return self.result('close_lease',True)
    def current_death(self,owner):return self.result('current_death',[{'identity':dict(owner),'currently_alive':False}])
    def resource_samples(self):
        if not self.cpu:self.guard_sample('no_worker_terminal')
        return self.result('resource_samples',{'CPU':self.cpu,'GPU':self.gpu})
def terminal_frame(root,*,api_faults=None,host_faults=None):
    root=Path(root);manifest=C.fixture_manifest(source_pins());api=F.InjectedAPI(api_faults);host=FakeHost(root/'raw',host_faults)
    terminal=S.supervise_injected(manifest,api,host,root/'raw');preservation=S.preserve_injected(root/'raw',root/'preserved',manifest)
    return {'manifest':manifest,'api':api,'host':host,'terminal':terminal,'preservation':preservation,'raw':root/'raw','preserved':root/'preserved'}
