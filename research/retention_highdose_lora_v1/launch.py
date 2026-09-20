"""Bounded wait for unchanged host/disk guards, then dispatch one fixed trainer."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from prepare import BASE,OUT,HERE,PILOT,sha,read,write

def main():
    import psutil
    frozen=read(OUT/'PRELAUNCH_BINDING.json')
    for name,expected in frozen['code_files_sha256'].items():
        if sha(HERE/name)!=expected:
            raise ValueError('Executable changed after prelaunch freeze: '+name)
    if sha(OUT/'SESSION_ENVELOPE_V2.json')!=frozen['session_envelope_sha256']:
        raise ValueError('Resource envelope changed')
    envelope=read(OUT/'SESSION_ENVELOPE_V2.json')
    gate=read(OUT/'BUILD_GATE_LORA_FINAL.json')
    started=time.monotonic()
    with (OUT/'preload_wait.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        while True:
            free=psutil.virtual_memory().available
            disk=shutil.disk_usage(OUT).free
            locked=(Path(tempfile.gettempdir())/'rineya-gpu-0.lock').exists()
            elapsed=time.monotonic()-started
            f.write(json.dumps({'elapsed_seconds':elapsed,'available_host_bytes':free,
                               'free_disk_bytes':disk,'lease_present':locked})+'\n');f.flush()
            if free>=envelope['pretrain_host_bytes'] and disk>=envelope['pretrain_free_disk_bytes'] and not locked:
                break
            if elapsed>=900 or time.time()+2540>envelope['deadline_unix']:
                write(OUT/'PRELOAD_TERMINAL.json',{'status':'WAIT_EXPIRED_NO_WORKER','worker_launched':False,
                    'wall_seconds':elapsed,'available_host_bytes':free,'free_disk_bytes':disk})
                return 2
            print(f'WAIT_RESOURCE RAM={free/2**30:.2f}GiB disk={disk/2**30:.2f}GiB lease={locked}',flush=True)
            time.sleep(min(30,900-elapsed))
    command=[str(BASE/'.venv-mq001/Scripts/python.exe'),str(HERE/'train.py'),
             '--source',gate['source'],'--prepared',str(OUT/'prepared'),
             '--plan',str(OUT/'PLAN.json'),'--grant',str(OUT/'GRANT.json'),
             '--journal',str(OUT/'journal'),'--run-id','train-001','--confirm-model-execution']
    write(OUT/'PRELOAD_TERMINAL.json',{'status':'DISPATCH_ONE_WORKER','worker_launched':True,
        'wall_seconds':time.monotonic()-started,'available_host_bytes':free,'free_disk_bytes':disk,'argv':command})
    result=subprocess.run(command,cwd=PILOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
    write(OUT/'TRAIN_COMMAND_TERMINAL.json',{'returncode':result.returncode,'status':'CHECK_NATIVE_RECEIPTS'})
    return result.returncode

if __name__=='__main__':
    raise SystemExit(main())
