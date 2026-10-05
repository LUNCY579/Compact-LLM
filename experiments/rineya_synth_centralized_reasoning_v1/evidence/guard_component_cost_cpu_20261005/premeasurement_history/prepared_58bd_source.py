"""512 current CPU guard calls, timing wrappers only; no native/GPU authority."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

WORK=Path(__file__).resolve().parent;PACKAGE=WORK.parents[1]
sys.path.insert(0,str(PACKAGE))
from inference_qual_worker import make_guard
from native_worker import verify_worker_ownership
import native_supervisor as supervisor

SCOPE='CPU_ONLY_512_CURRENT_GUARD_CALLBACK_TIMINGS_NO_NATIVE_FIT'
COUNT=512


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def no_framework():require(not any(n in sys.modules for n in ('torch','transformers','jax','tensorflow','flax')),'no framework')
def source_pins():
    paths=[PACKAGE/n for n in ('inference_qual_worker.py','native_worker.py','native_supervisor.py','native_process.py','native_contract.py','rs_contract.py')]
    paths += [Path(__file__),WORK/'test_guard_component_benchmark.py']
    paths += [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
              and Path(m.__file__).suffix=='.py' and PACKAGE in Path(m.__file__).resolve().parents]
    return {str(p):sha(p) for p in sorted(set(paths))}
def new_json(path,value):supervisor._json_new(path,value)
def constructor_owner(output,worker):
    owner=json.loads((output/'supervisor_ownership.json').read_bytes())
    expected={'worker_pid':worker.process.pid,'root_worker_pid':worker.process.pid,'job_name':worker.job.name,
        'supervisor_pid':os.getpid(),'supervisor_create_time':supervisor._pid_identity(os.getpid())['create_time'],
        'config_sha256':sha(output/'CPU_SCOPE.json')}
    require(owner==expected and all(type(owner[k]) is int and owner[k]>0 for k in ('worker_pid','root_worker_pid','supervisor_pid')),'actual immutable constructor owner')
    return owner


class Timer:
    def __init__(self,function,clock=time.perf_counter):
        self.function,self.clock=function,clock;self.counts=dict.fromkeys(('attempted','returned','failed','pending'),0);self.intervals=[]
    def __call__(self,*args,**kwargs):
        self.counts['attempted']+=1;self.counts['pending']+=1;start=self.clock()
        try:value=self.function(*args,**kwargs)
        except BaseException:self.counts['failed']+=1;self.counts['pending']-=1;raise
        else:self.counts['returned']+=1;self.counts['pending']-=1;return value
        finally:self.intervals.append(self.clock()-start)
    def snapshot(self):
        values=sorted(self.intervals)
        return {'counts':deepcopy(self.counts),'total_seconds':sum(values),'intervals_seconds':list(self.intervals),
                'min_seconds':values[0] if values else None,'median_seconds':statistics.median(values) if values else None,
                'p95_seconds':values[int((len(values)-1)*.95)] if values else None,'max_seconds':values[-1] if values else None}


def instrument_guard(output,owner,start,*,factory=make_guard):
    # Extract actual default closures, including make_guard's own nested
    # psutil.virtual_memory().available callback; do not substitute a rewritten
    # RAM API/body or alter clocks, resource floors, heartbeat rules or frequency.
    original=factory(output,owner,start,lambda:None)
    closure={name:cell.cell_contents for name,cell in zip(original.__code__.co_freevars,original.__closure__)}
    timers={name:Timer(closure[name]) for name in ('owned_resources','available','heartbeat_reader')}
    guard=factory(output,owner,start,lambda:None,**timers)
    return Timer(guard),timers,{name:closure[name].__module__+'.'+closure[name].__qualname__ for name in timers}


def cpu_runtime_pins():
    import psutil
    paths={Path(sys.executable),Path(sys._base_executable),Path(sys.base_prefix)/'python311.dll',Path(os.environ['SystemRoot'])/'System32/kernel32.dll'}
    psroot=Path(psutil.__file__).parent;paths.update(psroot.rglob('*.py'));paths.update(psroot.rglob('*.pyd'))
    # Executed stdlib/native-extension closure (ctypes/pathlib/json/etc), plus
    # the interpreter DLL and conservative complete psutil package inventory.
    stdroot=Path(os.__file__).resolve().parent
    paths.update(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
                 and Path(m.__file__).suffix in ('.py','.pyd') and stdroot in Path(m.__file__).resolve().parents)
    require(importlib.metadata.version('psutil')=='7.2.2','current pinned psutil7.2.2')
    return {'python_executable':sys.executable,'python_base_executable':sys._base_executable,'python_version':sys.version,
        'psutil_version':'7.2.2','file_pins':{str(p):sha(p) for p in sorted(paths) if p.is_file()}}


def publish_counts_before_metadata(output,measured,metadata_loader):
    """Measured counts/timings are immutable even if postmetadata fails."""
    partial=output/'CPU_measured_counts.json'
    new_json(partial,{**measured,'metadata_status':'NOT_YET_CHECKED'})
    try:metadata=metadata_loader()
    except BaseException as error:
        new_json(output/'CPU_metadata_failure.json',{'status':'FAILED_POST_MEASUREMENT_METADATA',
            'error':{'type':type(error).__name__,'message':str(error)},'measured_counts_sha256':sha(partial),
            'native_qualification_established':False})
        raise
    result={**measured,**metadata,'measured_counts_sha256':sha(partial),'metadata_status':'CHECKED'}
    new_json(output/'CPU_worker_result.json',result)
    return result


def child(output):
    output=Path(output).absolute()
    require(output.is_dir() and output.resolve(strict=True)==output and output.parent==WORK and not output.is_symlink(),'one staging child only')
    scope=json.loads((output/'CPU_SCOPE.json').read_bytes())
    require(scope=={'scope':SCOPE,'guard_calls':COUNT,'GPU':False,'native_qualification':False,'training':False},'exact512 CPU scope')
    owner=json.loads((output/'supervisor_ownership.json').read_bytes());verify_worker_ownership(owner,output/'CPU_SCOPE.json')
    before=source_pins();runtime_before=cpu_runtime_pins();start=time.monotonic()
    guard,timers,origins=instrument_guard(output,owner,start)
    status='PASS_512_CURRENT_CPU_GUARDS_CALLBACK_COSTS_ONLY';failure=None
    started=time.perf_counter()
    try:
        for ordinal in range(COUNT):guard('CPU_component_guard_%04d'%ordinal)
        require(guard.counts==dict(attempted=COUNT,returned=COUNT,failed=0,pending=0)
            and all(t.counts==guard.counts for t in timers.values()),'all512 complete callbacks/guards')
    except BaseException as error:
        status='FAILED_CURRENT_CPU_GUARD_CALLBACK_MEASUREMENT';failure={'type':type(error).__name__,'message':str(error)}
    wall=time.perf_counter()-started
    # This first publication has no postmetadata function calls; no retry.
    measured={'scope':SCOPE,'status':status,'failure':failure,'timed_loop_wall_seconds':wall,
        'guard':guard.snapshot(),'callbacks':{name:t.snapshot() for name,t in timers.items()},'callback_origins':origins,
        'GPU_guard_measured':False,'Torch_NVML_called':False,'native_qualification_established':False,
        'source_before':before,'CPU_runtime_before':runtime_before,
        'semantics':'actual default callback functions retained; only timing wrappers injected,1call each perguard; all current checks unchanged',
        'limits':'512CPU calls in fresh ownedJob; no model/body/native GPU cost or native/fullrun prediction'}
    def postmetadata():
        after,runtime_after=source_pins(),cpu_runtime_pins()
        require(before==after and runtime_before==runtime_after,'source/runtime drift')
        return {'source_after':after,'CPU_runtime_after':runtime_after}
    result=publish_counts_before_metadata(output,measured,postmetadata);no_framework()
    require(before==source_pins() and result['CPU_runtime_before']==result['CPU_runtime_after'],'source/runtime drift')
    if failure:raise RuntimeError(failure)


def parent():
    no_framework();before=source_pins();output=WORK/('measurement_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir(exist_ok=False)
    new_json(output/'CPU_SCOPE.json',{'scope':SCOPE,'guard_calls':COUNT,'GPU':False,'native_qualification':False,'training':False})
    worker=None;failure=None;observations=[];start=time.monotonic()
    def cpu_sample():
        sample=supervisor._cpu_snapshot(output,worker);observations.append(sample)
        require(supervisor._fresh_cpu(sample,time.monotonic()),'externalCPU200ms freshness')
        require(sample['host_available_bytes']>=1.25*1024**3 and sample['worker_tree_rss_bytes']<=2*1024**3 and sample['disk_free_bytes']>=4*1024**3,'unchanged CPU/RAM/disk floors')
        require(time.monotonic()-start<600,'unchanged600s ceiling')
    try:
        worker=supervisor._WindowsOwnedWorker([sys.executable,'-I','-B',str(Path(__file__)),'--cpu-child',str(output)],output,
            config_sha256=sha(output/'CPU_SCOPE.json'),deadline_monotonic=start+600)
        constructor_owner(output,worker);cpu_sample();supervisor._heartbeat(output,os.getpid());worker.resume()
        while worker.poll() is None:
            cpu_sample();supervisor._heartbeat(output,os.getpid());time.sleep(.1)
        require(worker.poll()==0,'CPU guard component worker failed')
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
        if worker is not None:worker.terminate()
    finally:
        if worker is not None:worker.close()
        new_json(output/'CPU_SUPERVISOR_RECEIPT.json',{'scope':SCOPE,'status':'PASS_CPU_GUARD_COMPONENT_MEASUREMENT' if failure is None else 'FAILED_CPU_GUARD_COMPONENT_MEASUREMENT',
            'failure':failure,'source_before':before,'source_after':source_pins(),'returncode':worker.poll() if worker else None,
            'owned_CPU_Job_closed':worker is not None,'wall_seconds':time.monotonic()-start,'observations':observations,
            'limits':{'CPU_freshness_seconds':.2,'CPU_cadence_seconds':.1,'disk_GiB':4.,'whole_scope_seconds':600},
            'GPU_NVML_called':False,'framework_imported':False,'native_launch_authorized':False,'training_authorized':False,
            'inventory':supervisor._inventory(output)})
    print(output)
    require(before==source_pins(),'source drift')
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-child',type=Path);parser.add_argument('--measure-reviewed-CPU-only',action='store_true')
    args=parser.parse_args()
    if args.cpu_child:child(args.cpu_child)
    else:
        require(args.measure_reviewed_CPU_only,'explicit reviewed CPU measurement flag required');parent()
