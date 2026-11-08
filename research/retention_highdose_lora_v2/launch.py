"""Bounded wait for unchanged host/disk guards, then dispatch one fixed trainer."""
from pathlib import Path
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from prepare import BASE,OUT,HERE,PILOT,sha,read,write

WAIT_SECONDS=900
RESERVATION_NAME='NATIVE_DISPATCH_RESERVATION.json'


def occupied(path):
    return path.exists() or path.is_symlink()


def native_reserve_seconds(envelope):
    seconds=envelope['native_limits']['total_owned_work_wall_seconds']
    if type(seconds) is not int or seconds!=2700:
        raise ValueError('Expected the frozen 2700-second native owned-work envelope')
    deadline=envelope['deadline_unix']
    if type(deadline) not in (int,float) or not math.isfinite(deadline):
        raise ValueError('Invalid frozen session deadline')
    return seconds


def admission_reason(envelope,elapsed,now):
    """Readiness never overrides an elapsed wait or insufficient native reserve."""
    if elapsed>=WAIT_SECONDS:
        return 'PRELOAD_WAIT_LIMIT_REACHED'
    if now+native_reserve_seconds(envelope)>envelope['deadline_unix']:
        return 'SESSION_CANNOT_FIT_NATIVE_OWNED_WORK'
    return None


def resource_snapshot():
    import psutil
    lock=Path(tempfile.gettempdir())/'rineya-gpu-0.lock'
    return {'available_host_bytes':psutil.virtual_memory().available,
            'free_disk_bytes':shutil.disk_usage(OUT).free,'lease_present':occupied(lock)}


def resources_ready(snapshot,envelope):
    return (snapshot['available_host_bytes']>=envelope['pretrain_host_bytes']
            and snapshot['free_disk_bytes']>=envelope['pretrain_free_disk_bytes']
            and not snapshot['lease_present'])


def verify_frozen():
    frozen=read(OUT/'PRELAUNCH_BINDING.json')
    if frozen.get('status')!='PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION':
        raise ValueError('Prelaunch binding is not passed')
    for name,expected in frozen['code_files_sha256'].items():
        if sha(HERE/name)!=expected:
            raise ValueError('Executable changed after prelaunch freeze: '+name)
    for name,key in (('SESSION_ENVELOPE_V2.json','session_envelope_sha256'),
                     ('PLAN.json','plan_sha256'),('GRANT.json','grant_sha256'),
                     ('PREREG.md','registration_sha256'),('BUILD_GATE_LORA_FINAL.json','build_gate_sha256')):
        if sha(OUT/name)!=frozen[key]:
            raise ValueError('Frozen prelaunch input changed: '+name)
    return frozen


def record_no_worker(waitroot,reason,started,snapshot=None,*,error=None):
    value={'status':'WAIT_EXPIRED_NO_WORKER' if reason in ('PRELOAD_WAIT_LIMIT_REACHED','SESSION_CANNOT_FIT_NATIVE_OWNED_WORK')
           else 'DISPATCH_REFUSED_NO_WORKER','worker_launched':False,'controller_launched':False,
           'reason':reason,'wall_seconds':time.monotonic()-started,
           'dispatch_reservation_present':occupied(OUT/RESERVATION_NAME)}
    if snapshot is not None:
        value.update(snapshot)
    if error is not None:
        value['error']={'type':type(error).__name__,'message':str(error)}
    write(waitroot/'PRELOAD_TERMINAL.json',value)
    return 2


def reserve_dispatch(wait_id,command,envelope,snapshot):
    """Atomic, persistent one-use reservation shared by every preload wait ID.

    Kept after refusal or launch errors for manual inspection. Never automatically
    removed, reused, or treated as proof that a model worker actually started.
    """
    if occupied(OUT/'journal'):
        raise RuntimeError('A native journal already exists; duplicate dispatch refused')
    path=OUT/RESERVATION_NAME
    write(path,{'schema':1,'status':'RESERVED_FOR_ONE_NATIVE_DISPATCH','worker_launched':False,
                'owner_pid':os.getpid(),'wait_id':wait_id,'created_unix':time.time(),
                'deadline_unix':envelope['deadline_unix'],
                'native_owned_seconds_reserved':native_reserve_seconds(envelope),
                'prelaunch_binding_sha256':sha(OUT/'PRELAUNCH_BINDING.json'),'argv':command,
                'resources_at_reservation':snapshot,
                'full_battery_completion_guaranteed':False,
                'downstream_policy':'Each export, compression and probe requires its own remaining-time admission; preserve partial outputs if a later stage cannot fit.'})
    return path


def dispatch_once(waitroot,wait_id,command,envelope,started):
    """Revalidate before and after the atomic reservation, immediately before spawn."""
    snapshot=None
    try:
        verify_frozen()
        snapshot=resource_snapshot()
        reason=admission_reason(envelope,time.monotonic()-started,time.time())
        if reason:
            return record_no_worker(waitroot,reason,started,snapshot)
        if not resources_ready(snapshot,envelope):
            return record_no_worker(waitroot,'RESOURCES_CHANGED_BEFORE_DISPATCH',started,snapshot)
        reserve_dispatch(wait_id,command,envelope,snapshot)
        # Other wait IDs cannot pass the exclusive reservation. Recheck even if
        # a former launcher created a journal before adopting this guard.
        if occupied(OUT/'journal'):
            raise RuntimeError('A native journal appeared after dispatch reservation')
        verify_frozen()
        snapshot=resource_snapshot()
        reason=admission_reason(envelope,time.monotonic()-started,time.time())
        if reason:
            return record_no_worker(waitroot,reason,started,snapshot)
        if not resources_ready(snapshot,envelope):
            return record_no_worker(waitroot,'RESOURCES_CHANGED_AFTER_RESERVATION',started,snapshot)
        options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        child=subprocess.Popen(command,cwd=PILOT,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),**options)
    except (OSError,ValueError,RuntimeError) as exc:
        return record_no_worker(waitroot,'DISPATCH_RESERVATION_OR_LAUNCH_FAILED',started,snapshot,error=exc)
    try:
        write(waitroot/'PRELOAD_TERMINAL.json',{'status':'DISPATCHED_ONE_CONTROLLER',
            'worker_launched':None,'controller_launched':True,'controller_pid':child.pid,
            'worker_status_source':'Inspect the authoritative native journal and supervisor receipts',
            'wall_seconds':time.monotonic()-started,**snapshot,'argv':command,
            'dispatch_reservation_sha256':sha(OUT/RESERVATION_NAME)})
    finally:
        returncode=child.wait()
    write(waitroot/'TRAIN_COMMAND_TERMINAL.json',{'returncode':returncode,'status':'CHECK_NATIVE_RECEIPTS',
                                               'controller_pid':child.pid})
    return returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-id',default='initial')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,31}',args.wait_id):
        raise ValueError('Wait identifier must be one bounded filename component')
    if occupied(OUT/'journal') or occupied(OUT/RESERVATION_NAME):
        raise RuntimeError('A native journal already exists; inspect its authoritative state instead of dispatching again')
    waitroot=OUT
    if args.wait_id!='initial':
        prior=read(OUT/'PRELOAD_TERMINAL.json')
        if prior['status']!='WAIT_EXPIRED_NO_WORKER' or prior['worker_launched'] is not False:
            raise RuntimeError('Another wait requires a verified terminal with no worker')
        waitroot=OUT/'preload-attempts'/args.wait_id
        waitroot.mkdir(parents=True,exist_ok=False)
    verify_frozen()
    envelope=read(OUT/'SESSION_ENVELOPE_V2.json')
    native_reserve_seconds(envelope)
    gate=read(OUT/'BUILD_GATE_LORA_FINAL.json')
    started=time.monotonic()
    with (waitroot/'preload_wait.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        while True:
            snapshot=resource_snapshot()
            elapsed=time.monotonic()-started
            now=time.time()
            f.write(json.dumps({'elapsed_seconds':elapsed,'sampled_unix':now,
                               'native_owned_seconds_required':native_reserve_seconds(envelope),**snapshot})+'\n');f.flush()
            reason=admission_reason(envelope,elapsed,now)
            if reason:
                return record_no_worker(waitroot,reason,started,snapshot)
            if occupied(OUT/'journal') or occupied(OUT/RESERVATION_NAME):
                return record_no_worker(waitroot,'ANOTHER_WAIT_RESERVED_OR_DISPATCHED',started,snapshot)
            if resources_ready(snapshot,envelope):
                break
            print(f"WAIT_RESOURCE RAM={snapshot['available_host_bytes']/2**30:.2f}GiB disk={snapshot['free_disk_bytes']/2**30:.2f}GiB lease={snapshot['lease_present']}",flush=True)
            remaining=envelope['deadline_unix']-native_reserve_seconds(envelope)-time.time()
            time.sleep(max(0,min(30,WAIT_SECONDS-elapsed,remaining)))
    command=[str(BASE/'.venv-mq001/Scripts/python.exe'),str(HERE/'train.py'),
             '--source',gate['source'],'--prepared',str(OUT/'prepared'),
             '--plan',str(OUT/'PLAN.json'),'--grant',str(OUT/'GRANT.json'),
             '--journal',str(OUT/'journal'),'--run-id','train-001','--confirm-model-execution']
    return dispatch_once(waitroot,args.wait_id,command,envelope,started)

if __name__=='__main__':
    raise SystemExit(main())
