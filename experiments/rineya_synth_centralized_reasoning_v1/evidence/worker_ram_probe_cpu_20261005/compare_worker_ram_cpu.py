"""DRAFT matched512 default-vs-candidate CPU guards; hold until root terminal.

No root/helper edit, framework, GPU/NVML, numeric cache, forecast or native
qualification. Uses the already-reviewed CPU ownership/publication utilities.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

WORK=Path(__file__).resolve().parent;PACKAGE=WORK.parents[1]
sys.path.insert(0,str(WORK));sys.path.insert(0,str(PACKAGE/'artifacts/guard_component_cost_work'))
import guard_component_benchmark as utilities
import worker_ram_probe as candidate
from inference_qual_worker import make_guard
from native_worker import verify_worker_ownership

SCOPE='CPU_ONLY_DEFAULT_VS_DIRECT_PHYSICAL_AVAILABLE_CALLBACK_512_EACH'
COUNT=512
require,sha,new_json=utilities.require,utilities.sha,utilities.new_json
supervisor=utilities.supervisor


def source_pins():
    files=[Path(__file__),WORK/'worker_ram_probe.py',WORK/'test_worker_ram_probe.py',Path(utilities.__file__)]
    files += [PACKAGE/n for n in ('inference_qual_worker.py','native_worker.py','native_process.py','native_supervisor.py','native_contract.py','rs_contract.py')]
    files += [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
              and Path(m.__file__).suffix=='.py' and PACKAGE in Path(m.__file__).resolve().parents]
    return {str(p):sha(p) for p in sorted(set(files))}


def timed_guard(output,owner,start,factor):
    original=make_guard(output,owner,start,lambda:None)
    closure={name:cell.cell_contents for name,cell in zip(original.__code__.co_freevars,original.__closure__)}
    available=closure['available'] if factor=='default_psutil' else candidate.available_bytes
    observed=[]
    def record_available():
        value=available();observed.append(value);return value
    timers={name:utilities.Timer(record_available if name=='available' else closure[name])
            for name in ('owned_resources','available','heartbeat_reader')}
    guard=utilities.Timer(make_guard(output,owner,start,lambda:None,**timers))
    return guard,timers,observed


def child(output):
    output=Path(output).absolute()
    require(output.is_dir() and output.resolve(strict=True)==output and output.parent==WORK and not output.is_symlink(),'one new staging output')
    scope=json.loads((output/'CPU_SCOPE.json').read_bytes())
    require(scope=={'scope':SCOPE,'guard_calls_per_factor':COUNT,'factors':['default_psutil','candidate_direct'],'GPU':False,'native_qualification':False,'training':False},'exact matched512 CPU scope')
    owner=json.loads((output/'supervisor_ownership.json').read_bytes());verify_worker_ownership(owner,output/'CPU_SCOPE.json')
    before=source_pins();runtime_before=utilities.cpu_runtime_pins();results=[];start=time.monotonic()
    for factor in scope['factors']:
        leaf=output/factor;leaf.mkdir(exist_ok=False)
        guard,timers,values=timed_guard(output,owner,start,factor);status='PASS_CPU_512_CALLBACK_FACTOR';failure=None
        began=time.perf_counter()
        try:
            for ordinal in range(COUNT):guard('CPU_RAM_probe_%s_%04d'%(factor,ordinal))
            expected=dict(attempted=COUNT,returned=COUNT,failed=0,pending=0)
            require(guard.counts==expected and all(t.counts==expected for t in timers.values()),'all512 unchanged guard/callback returns')
        except BaseException as error:
            status='FAILED_CPU_CALLBACK_FACTOR';failure={'type':type(error).__name__,'message':str(error)}
        measured={'scope':SCOPE,'factor':factor,'status':status,'failure':failure,'timed_wall_seconds':time.perf_counter()-began,
            'guard':guard.snapshot(),'callbacks':{name:t.snapshot() for name,t in timers.items()},
            'available_bytes_observed':values,'source_before':before,'CPU_runtime_before':runtime_before,
            'GPU_guard_measured':False,'native_qualification_established':False,'training_authorized':False,
            'volatile_reading_policy':'fresh queries; values recorded but never reused; independent readings need not equal',
            'guard_policy':'only available callback differs; same current ownedRSS/RAMfloors/deadline/heartbeat/frequency'}
        def metadata():
            after,runtime_after=source_pins(),utilities.cpu_runtime_pins()
            require(before==after and runtime_before==runtime_after,'source/runtime drift')
            return {'source_after':after,'CPU_runtime_after':runtime_after}
        final=utilities.publish_counts_before_metadata(leaf,measured,metadata);results.append(final)
        if failure:raise RuntimeError(failure)
    utilities.no_framework()
    new_json(output/'CPU_worker_result.json',{'status':'PASS_MATCHED512_RAM_CALLBACK_CPU_FACTORS_ONLY','scope':SCOPE,
        'source_before':before,'source_after':source_pins(),'CPU_runtime_before':runtime_before,'CPU_runtime_after':utilities.cpu_runtime_pins(),
        'factors':results,'GPU_NVML_called':False,'native_launch_authorized':False,'training_authorized':False,
        'native_body_and_GPUguard_costs_unmeasured':True,'fullrun_forecast':False})


def parent():
    utilities.no_framework();before=source_pins();output=WORK/('measurement_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir(exist_ok=False)
    new_json(output/'CPU_SCOPE.json',{'scope':SCOPE,'guard_calls_per_factor':COUNT,'factors':['default_psutil','candidate_direct'],
                                   'GPU':False,'native_qualification':False,'training':False})
    worker=None;failure=None;observations=[];started=time.monotonic()
    def sample():
        value=supervisor._cpu_snapshot(output,worker);observations.append(value)
        require(supervisor._fresh_cpu(value,time.monotonic()),'unchanged externalCPU200ms freshness')
        require(value['host_available_bytes']>=1.25*1024**3 and value['worker_tree_rss_bytes']<=2*1024**3
                and value['disk_free_bytes']>=4*1024**3,'unchanged CPU/RAM/disk guards')
        require(time.monotonic()-started<600,'unchanged600s maximum')
    try:
        worker=supervisor._WindowsOwnedWorker([sys.executable,'-I','-B',str(Path(__file__)),'--cpu-child',str(output)],output,
            config_sha256=sha(output/'CPU_SCOPE.json'),deadline_monotonic=started+600)
        utilities.constructor_owner(output,worker);sample();supervisor._heartbeat(output,os.getpid());worker.resume()
        while worker.poll() is None:sample();supervisor._heartbeat(output,os.getpid());time.sleep(.1)
        require(worker.poll()==0,'CPU matched callback worker failure')
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
        if worker is not None:worker.terminate()
    finally:
        if worker is not None:worker.close()
        new_json(output/'CPU_SUPERVISOR_RECEIPT.json',{'scope':SCOPE,'status':'PASS_CPU_RAM_CALLBACK_COMPARISON' if failure is None else 'FAILED_CPU_RAM_CALLBACK_COMPARISON',
            'failure':failure,'source_before':before,'source_after':source_pins(),'returncode':worker.poll() if worker else None,
            'CPU_Job_closed':worker is not None,'wall_seconds':time.monotonic()-started,'observations':observations,
            'limits':{'CPU_freshness_seconds':.2,'CPU_cadence_seconds':.1,'disk_GiB':4.,'whole_scope_seconds':600},
            'GPU_NVML_called':False,'framework_imported':False,'native_launch_authorized':False,'training_authorized':False,
            'inventory':supervisor._inventory(output)})
    print(output);require(before==source_pins(),'source drift')
    if failure:raise RuntimeError(failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-child',type=Path);parser.add_argument('--measure-after-root-native-and-audit-terminal',action='store_true')
    args=parser.parse_args()
    if args.cpu_child:child(args.cpu_child)
    else:
        require(args.measure_after_root_native_and_audit_terminal,'hold measurement until root native/audit terminal approval');parent()
