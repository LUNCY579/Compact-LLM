"""Run the fixed high-dose alpha battery only after verified native completion.

One child at a time. Failed legs and partial outputs are never removed or retried
in place. This command is deliberately non-adaptive and must be explicitly rerun
after reviewing any preserved failure.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import hashlib
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = Path('C:/Users/clclc/BusinessProject')
DEFAULT_ROOT = BASE / 'rineya-dataset-retention-records/doseint001-20260919'
PYTHON = BASE / '.venv-mq001/Scripts/python.exe'
PILOT = BASE / 'ChatBubble-retention-pilot-20260917'
SOURCE = BASE / 'Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze'
ARMS = (('a100',1.0),('a025',0.25),('a050',0.5),('a075',0.75))
LEGS = (('dose','process','process-transition-probe-001'),
        ('select','process','select-views-001'),
        ('retention','retention','process-transition-probe-001'),
        ('long_context','long_context','process-transition-probe-001'))

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')

def size(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def validate_native(root,native):
    plan=read(root/'PLAN_DUAL899.json')
    gate=read(root/'BUILD_GATE_DUAL899.json')
    assert sha(root/'PLAN_DUAL899.json')==gate['plan_file_sha256']
    contract=Path(gate['prepared'])/'DATA_CONTRACT.json'
    for name,expected in read(contract)['files'].items():
        assert sha(contract.parent/name)==expected, 'prepared input changed'
    summary=read(native/'run_summary.json')
    result=read(native/'result.json')
    qualification=read(native/'qualification.json')
    assert summary['returncode']==0 and summary['supervisor']['status']=='EXITED'
    assert summary['supervisor']['returncode']==0 and not summary['worker_failed']
    assert summary['budget_after']['budget_breached'] is False
    assert result['complete'] is True
    assert result['progress']['updates']==len(plan['update_groups'])==193
    assert result['progress']['processed_tokens']==plan['processed_tokens']==380921
    binding=result['binding']
    assert binding['plan_sha256']==digest(plan), 'native run used another plan'
    assert binding['data_contract_sha256']==sha(contract), 'native run used another data contract'
    assert binding['source']['assets']==gate['source_assets_sha256'], 'native source differs'
    assert binding['checkout']==gate['pilot_checkout'], 'native executable checkout differs'
    assert binding['grant_sha256']==digest(read(native.parent.parent/'grant.json'))
    assert qualification['status']=='PASS' and qualification['optimizer_steps']==0
    assert qualification['binding']==binding
    assert qualification['qualified_training_shape_tokens']==2048
    assert qualification['trainable_tensors']==220 and qualification['unique_parameters']==596049920
    assert qualification['source_state_before']==qualification['source_state_after']
    checkpoint=native/'microfit'/f"checkpoint-{result['progress']['updates']:05d}"
    manifest=read(checkpoint/'manifest.json')
    seal=manifest.pop('sha256')
    assert seal==digest(manifest) and manifest['status']=='COMPLETE'
    assert manifest['binding']==binding and manifest['progress']==result['progress']
    return checkpoint

def main():
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--native-run',type=Path,required=True)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args()
    root=a.root.resolve()
    native=a.native_run.resolve()
    checkpoint=validate_native(root,native)
    gate=root/'BUILD_GATE_DUAL899.json'
    actions=[]
    for name,alpha in ARMS:
        arm=root/'arms'/name
        actions.append(('build-'+name,[str(PYTHON),str(HERE/'build_arm.py'),
            '--source',str(SOURCE),'--checkpoint',str(checkpoint),'--alpha',str(alpha),'--out',str(arm)]))
    for name,alpha in ARMS:
        arm=root/'arms'/name
        for suffix,kind,instrument in LEGS:
            label=f'doseint001_dual899_{name}_{suffix}'
            out=root/'eval'/f'{name}-{suffix}'
            actions.append((f'probe-{name}-{suffix}',[str(PYTHON),str(HERE/'run_probe.py'),
                '--arm',str(arm),'--label',label,'--kind',kind,'--instrument',instrument,
                '--out',str(out),'--build-gate',str(gate)]))
    if not a.execute:
        print(json.dumps({'status':'READY_NOT_EXECUTED','native_run':str(native),'actions':actions},indent=2))
        return 0
    logroot=root/'battery-launch'
    logroot.mkdir(exist_ok=False)
    write(logroot/'PLAN.json',{'actions':actions,'native_run':str(native),'max_probe_seconds':14400,
                            'max_new_disk_bytes':17*2**30,'retries_in_this_runner':0})
    start=time.monotonic()
    total_probe_seconds=0.0
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    completed=[]
    for label,cmd in actions:
        assert size(root)<=17*2**30, 'registered new-disk ceiling exhausted'
        assert shutil.disk_usage(root).free>=14*2**30, 'native free-disk floor reached'
        assert time.monotonic()-start < 14400, 'battery wall envelope exhausted'
        if label.startswith('probe-'):
            assert total_probe_seconds+900 <=14400, 'probe worker reservation exhausted'
        print('START '+label,flush=True)
        began=time.monotonic()
        with (logroot/(label+'.log')).open('xb') as log:
            child=subprocess.run(cmd,cwd=PILOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.monotonic()-began
        leg={'label':label,'returncode':child.returncode,'wall_seconds':elapsed}
        write(logroot/(label+'.json'),leg)
        if child.returncode:
            write(logroot/'TERMINAL.json',{'status':'FAILED','failed_leg':label,'completed_legs':completed})
            return child.returncode
        if label.startswith('probe-'):
            out=Path(cmd[cmd.index('--out')+1])
            binding=read(out/'PROBE_WEIGHT_BINDING.json')
            assert binding['complete'] is True and binding['returncode']==0
            total_probe_seconds+=read(out/'worker/supervisor.json')['wall_seconds']
        else:
            out=Path(cmd[cmd.index('--out')+1])
            assert read(out/'ARM.json')['stored_dtype']=='float32'
        completed.append(leg)
        print('VERIFIED '+label,flush=True)
    report=subprocess.run([str(PYTHON),str(HERE/'report_battery.py'),'--root',str(root)],cwd=PILOT,env=env)
    write(logroot/'TERMINAL.json',{'status':'COMPLETE' if report.returncode==0 else 'REPORT_FAILED',
                                  'completed_legs':completed,'report_returncode':report.returncode,
                                  'probe_worker_seconds':total_probe_seconds,'wall_seconds':time.monotonic()-start,
                                  'new_disk_bytes':size(root)})
    return report.returncode

if __name__=='__main__':
    raise SystemExit(main())
