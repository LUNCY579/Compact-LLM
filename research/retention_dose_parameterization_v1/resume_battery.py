"""Continue fixed probe legs after a preserved failure using an explicit retry map."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.retention_dose_parameterization_v1 import paired
from research.retention_dose_parameterization_v1.run_battery import (
    ARMS,LEGS,PILOT,PYTHON,HERE,DEFAULT_ROOT,read,write,sha,size,validate_native)

def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    a=ap.parse_args()
    root=a.root.resolve()
    validate_native(root,root/'journal-dual899-recovery/runs/train-001')
    old=read(root/'battery-launch/TERMINAL.json')
    assert old['status']=='FAILED' and old['failed_leg']['label']=='probe-a025-long_context'
    failure_dir=root/'eval/a025-long_context'
    failed_supervisor=read(failure_dir/'worker/supervisor.json')
    failed_terminal=read(failure_dir/'worker/probe_terminal.json')
    assert failed_supervisor['status']=='EXITED' and failed_supervisor['returncode']!=0
    assert failed_terminal['complete'] is False and 'CUDA error: unknown error' in failed_terminal['failure']['message']
    assert not (failure_dir/'PROBE_WEIGHT_BINDING.json').exists() and not (failure_dir/'result.json').exists()
    mapping={'schema':1,'replacements':{'a025-long_context':'eval/a025-long_context-retry1'},
        'reason':'One registered inference-only retry after CUDA unknown error; identical weights/data/scripts/settings; no completed endpoint existed.',
        'prior_failure_files_sha256':{str(p.relative_to(root)):sha(p) for p in (
            failure_dir/'FAILURE.json',failure_dir/'PRECHECK.json',
            failure_dir/'worker/supervisor.json',failure_dir/'worker/probe_terminal.json')},
        'quality_claim':False}
    write(root/'ATTEMPT_MAP.json',mapping)
    logroot=root/'battery-resume-001'
    logroot.mkdir(exist_ok=False)
    envelope=read(root/'SESSION_ENVELOPE.json')
    seconds=old['probe_worker_seconds']
    rows_reserved=old['probe_rows_reserved']
    tokens_reserved=old['probe_generation_tokens_reserved']
    actions=[]
    already=[]
    for name,_ in ARMS:
        for suffix,kind,instrument in LEGS:
            key=f'{name}-{suffix}'
            directory=root/mapping['replacements'].get(key,f'eval/{key}')
            if (directory/'PROBE_WEIGHT_BINDING.json').exists():
                paired.validate_probe_binding(directory/'result.json',directory/'PROBE_WEIGHT_BINDING.json',kind)
                already.append({'leg':key,'binding_sha256':sha(directory/'PROBE_WEIGHT_BINDING.json')})
                continue
            assert not directory.exists(), 'unmapped incomplete attempt must be preserved and reviewed'
            label=f'doseint001_dual899_{name}_{suffix}'+('_retry1' if key in mapping['replacements'] else '')
            command=[str(PYTHON),str(HERE/'run_probe.py'),'--arm',str(root/'arms'/name),
                     '--label',label,'--kind',kind,'--instrument',instrument,'--out',str(directory),
                     '--build-gate',str(root/'BUILD_GATE_DUAL899.json')]
            actions.append((key,kind,directory,command))
    assert len(actions)==9 and len(already)==7
    write(logroot/'PLAN.json',{'actions':[{'leg':key,'kind':kind,'out':str(out),'argv':cmd} for key,kind,out,cmd in actions],
                            'already_verified':already,'prior_terminal_sha256':sha(root/'battery-launch/TERMINAL.json'),
                            'attempt_map_sha256':sha(root/'ATTEMPT_MAP.json'),'prior_probe_seconds':seconds,
                            'prior_rows_reserved':rows_reserved,'prior_generation_tokens_reserved':tokens_reserved})
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    completed=[]
    for key,kind,out,command in actions:
        assert time.time()+900<=envelope['deadline_unix'] and seconds+900<=envelope['probe_worker_seconds_all_attempts']
        assert size(root)<=envelope['new_records_disk_bytes'] and shutil.disk_usage(root).free>=envelope['free_disk_floor_bytes']
        count,max_new={'process':(384,128),'retention':(150,24),'long_context':(256,24)}[kind]
        assert rows_reserved+count<=envelope['probe_rows_reserved_all_attempts']
        assert tokens_reserved+count*max_new<=envelope['probe_generation_tokens_reserved_all_attempts']
        rows_reserved+=count;tokens_reserved+=count*max_new
        print('START '+key+(' retry1' if key in mapping['replacements'] else ''),flush=True)
        started=time.monotonic()
        with (logroot/(key+'.log')).open('xb') as log:
            child=subprocess.run(command,cwd=PILOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        supervisor=out/'worker/supervisor.json'
        charged=900.0
        accounting='FULL_RESERVATION_NO_VALID_TERMINAL'
        if supervisor.is_file():
            receipt=read(supervisor)
            value=receipt.get('wall_seconds')
            if type(value) in (int,float) and value>=0 and receipt.get('returncode') is not None:
                charged=float(value);accounting='OBSERVED_SUPERVISOR_WALL_INCLUDING_FAILURE'
        seconds+=charged
        leg={'leg':key,'out':str(out),'returncode':child.returncode,'wall_seconds':time.monotonic()-started,
             'probe_seconds_charged':charged,'probe_time_accounting':accounting,
             'probe_rows_reserved':count,'probe_generation_tokens_reserved':count*max_new}
        write(logroot/(key+'.json'),leg)
        if child.returncode:
            write(logroot/'TERMINAL.json',{'status':'FAILED','failed_leg':leg,'completed_legs':completed,
                'probe_worker_seconds_all_attempts':seconds,'probe_rows_reserved_all_attempts':rows_reserved,
                'probe_generation_tokens_reserved_all_attempts':tokens_reserved})
            return child.returncode
        paired.validate_probe_binding(out/'result.json',out/'PROBE_WEIGHT_BINDING.json',kind)
        completed.append(leg)
        print('VERIFIED '+key,flush=True)
    report=subprocess.run([str(PYTHON),str(HERE/'report_battery.py'),'--root',str(root)],cwd=PILOT,env=env)
    assert time.time()<=envelope['deadline_unix']
    assert size(root)<=envelope['new_records_disk_bytes'] and shutil.disk_usage(root).free>=envelope['free_disk_floor_bytes']
    write(logroot/'TERMINAL.json',{'status':'COMPLETE' if report.returncode==0 else 'REPORT_FAILED',
        'completed_legs':completed,'report_returncode':report.returncode,
        'probe_worker_seconds_all_attempts':seconds,'probe_rows_reserved_all_attempts':rows_reserved,
        'probe_generation_tokens_reserved_all_attempts':tokens_reserved,
        'session_elapsed_seconds':time.time()-envelope['session_start_unix'],'new_disk_bytes':size(root)})
    return report.returncode

if __name__=='__main__':
    raise SystemExit(main())
