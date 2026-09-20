"""Two fixed exports and eight fixed probes, with no retry or deletion in this runner."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
from prepare import BASE,OUT,HERE,PILOT,sha,read,write

PYTHON=BASE/'.venv-mq001/Scripts/python.exe'
OLD=BASE/'rineya-dataset-retention-records/doseint001-20260919'
CONTROL=OLD/'journal-dual899-recovery/runs/train-001/microfit/checkpoint-00193'
LEGS=(('dose','process','process-transition-probe-001'),('select','process','select-views-001'),
      ('retention','retention','process-transition-probe-001'),('long_context','long_context','process-transition-probe-001'))

def compress_verified(arm,logdir):
    path=arm/'model.safetensors'
    expected=read(arm/'ARM.json')['arm_weight_sha256']
    if sha(path)!=expected:
        raise ValueError('Model differs before transparent compression')
    started=time.monotonic()
    with (logdir/'compact.log').open('xb') as log:
        rc=subprocess.run(['compact.exe','/C','/EXE:LZX','/Q',str(path)],stdout=log,stderr=subprocess.STDOUT,
                          creationflags=subprocess.CREATE_NO_WINDOW,timeout=600).returncode
    after=sha(path)
    write(logdir/'COMPRESSION.json',{'returncode':rc,'before_sha256':expected,'after_sha256':after,
        'logical_bytes':path.stat().st_size,'wall_seconds':time.monotonic()-started,
        'free_disk_bytes':shutil.disk_usage(OUT).free,'model_execution':False})
    if rc or after!=expected:
        raise ValueError('Compression did not verify')

def main():
    sys.path.insert(0,str(BASE/'ChatBubble-dose-param'))
    from research.retention_dose_parameterization_v1.paired import validate_probe_binding
    from experiments.model_quality_gpu_v1.integrity import unseal
    run=OUT/'journal/runs/train-001'
    summary=read(run/'run_summary.json')
    if summary['returncode'] or summary['worker_failed'] or summary['budget_after']['budget_breached']:
        raise ValueError('Training worker did not complete successfully')
    result=read(run/'result.json')
    if result['status']!='COMPLETE' or result['complete'] is not True:
        raise ValueError('Complete qualified training result required')
    envelope=read(OUT/'SESSION_ENVELOPE_V2.json')
    gate=read(OUT/'BUILD_GATE_LORA_FINAL.json')
    logs=OUT/'posttrain';logs.mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    attempts=[];seconds=0.0;rows_reserved=tokens_reserved=0
    def guard(seconds_needed):
        if time.time()+seconds_needed>envelope['deadline_unix']:
            raise RuntimeError('Registered session deadline cannot fit this leg')
        if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>envelope['new_records_disk_bytes']:
            raise RuntimeError('New experiment disk allowance exhausted')
        if shutil.disk_usage(OUT).free<envelope['free_disk_floor_bytes']:
            raise RuntimeError('Unchanged free-space floor refused')
    try:
        for name in ('lora','norm'):
            guard(600)
            if name=='norm' and shutil.disk_usage(OUT).free<envelope['pre_second_export_free_disk_bytes']:
                raise RuntimeError('BLOCKED_ON_STORAGE before matched scalar export')
            arm=OUT/'arms'/name
            cpu=logs/f'build-{name}'
            args=['lora' if name=='lora' else 'scalar','--source',gate['source'],
                  '--native-checkpoint',str(CONTROL),'--out',str(arm),'--norm-record',str(OUT/'NORM_MATCH.json')]
            if name=='lora':
                args+=['--checkpoint',result['checkpoint'],'--training-result',str(run/'result.json')]
            command=[str(PYTHON),str(HERE/'cpu_stage.py'),'--out',str(cpu),'--',*args]
            print('START build-'+name,flush=True)
            rc=subprocess.run(command,cwd=PILOT,env=env).returncode
            if rc:
                raise RuntimeError(f'Materialization {name} failed; preserved rc{rc}')
            if unseal(read(OUT/'NORM_MATCH.json'))['status']!='PASS':
                raise RuntimeError('BLOCKED_ON_NORM_COMPARABILITY')
            if read(arm/'ARM.json')['arm_weight_sha256']!=sha(arm/'model.safetensors'):
                raise ValueError('Materialized model bytes differ')
            guard(600)
            compress_verified(arm,cpu)
            print('VERIFIED build-'+name,flush=True)
        # Both new arms must exist and be norm-qualified before any endpoint runs.
        norm=read(OUT/'arms/norm/ARM.json')
        if norm['relative_norm_discrepancy']>0.005:
            raise ValueError('Matched norm tolerance failed')
        for name in ('lora','norm'):
            for suffix,kind,instrument in LEGS:
                guard(900)
                if seconds+900>envelope['probe_worker_seconds_all_attempts']:
                    raise RuntimeError('Probe wall budget exhausted')
                n,max_new={'process':(384,128),'retention':(150,24),'long_context':(256,24)}[kind]
                if rows_reserved+n>envelope['probe_rows_reserved_all_attempts'] or tokens_reserved+n*max_new>envelope['probe_generation_tokens_reserved_all_attempts']:
                    raise RuntimeError('Probe row/token allowance exhausted')
                rows_reserved+=n;tokens_reserved+=n*max_new
                attempt=OUT/'eval'/f'{name}-{suffix}'
                command=[str(PYTHON),str(BASE/'ChatBubble-dose-param/research/retention_dose_parameterization_v1/run_probe.py'),
                         '--arm',str(OUT/'arms'/name),'--label',f'doselora001_{name}_{suffix}',
                         '--kind',kind,'--instrument',instrument,'--out',str(attempt),'--build-gate',
                         str(OUT/('BUILD_GATE_LORA_FINAL.json' if name=='lora' else 'BUILD_GATE_NATIVE_NORM.json'))]
                print('START '+name+'-'+suffix,flush=True)
                started=time.monotonic()
                with (logs/f'{name}-{suffix}.log').open('xb') as log:
                    rc=subprocess.run(command,cwd=PILOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                supervisor=attempt/'worker/supervisor.json'
                charged=900.0
                if supervisor.is_file():
                    record=read(supervisor)
                    if record.get('returncode') is not None and type(record.get('wall_seconds')) in (int,float):
                        charged=record['wall_seconds']
                seconds+=charged
                record={'leg':name+'-'+suffix,'out':str(attempt),'returncode':rc,'wall_seconds':time.monotonic()-started,
                        'worker_seconds_charged':charged,'reserved_rows':n,'reserved_generation_tokens':n*max_new}
                write(logs/f'{name}-{suffix}.json',record);attempts.append(record)
                if rc:
                    raise RuntimeError('Probe failed; any registered retry needs a new explicit attempt')
                validate_probe_binding(attempt/'result.json',attempt/'PROBE_WEIGHT_BINDING.json',kind)
                print('VERIFIED '+name+'-'+suffix,flush=True)
        guard(0)
        write(logs/'TERMINAL.json',{'status':'COMPLETE_EIGHT_PROBES','attempts':attempts,
            'probe_worker_seconds':seconds,'probe_rows_reserved':rows_reserved,
            'probe_generation_tokens_reserved':tokens_reserved,'session_elapsed_seconds':time.time()-envelope['session_start_unix']})
        return 0
    except Exception as exc:
        write(logs/'TERMINAL.json',{'status':'FAILED_OR_BLOCKED','failure':{'type':type(exc).__name__,'message':str(exc)},
            'attempts':attempts,'probe_worker_seconds':seconds,'probe_rows_reserved':rows_reserved,
            'probe_generation_tokens_reserved':tokens_reserved,'quality_claim':False})
        raise

if __name__=='__main__':
    raise SystemExit(main())
