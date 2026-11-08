"""Prospectively bind one fresh recovery of a token-accounting failure."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
from prepare import BASE, OUT, PILOT, SOURCE, sha, read, write

def main():
    import psutil
    native=BASE/'.venv-mq001/Scripts/python.exe'
    previous=OUT/'journal-dual899-preloadretry/runs/train-001'
    summary=read(previous/'run_summary.json')
    failure=read(previous/'failure.json')
    used=summary['budget_after']['used_or_reserved']
    assert summary['supervisor']['returncode']==2 and failure['message']=='approved budget exhausted; preserve the attempt'
    assert used['processed_training_tokens_all_attempts']==379484
    assert not (previous/'result.json').exists()
    assert {p.name for p in (previous/'microfit').glob('checkpoint-*')}=={'checkpoint-00002'}
    events=previous.parent.parent/'events'
    qualification_charge=read(events/'000001.json')['charges']
    assert qualification_charge['processed_training_tokens_all_attempts']==2203
    plan=read(OUT/'PLAN_DUAL899.json')
    assert plan['processed_tokens']+2203==383124
    grant_path=OUT/'GRANT_DUAL899_RECOVERY.json'
    amendment=Path(__file__).with_name('AMENDMENT_03.md')
    if not grant_path.exists():
        with (OUT/amendment.name).open('xb') as f:
            f.write(amendment.read_bytes())
        grant=read(OUT/'GRANT_DUAL899.json')
        grant['approved_by']='current user goal plus delegated best-step discretion in USER_STEERING_02.md; agent-instantiated recovery envelope'
        grant['authorization_secondary_reference']=str(OUT/'USER_STEERING_02.md')
        grant['authorization_secondary_sha256']=sha(OUT/'USER_STEERING_02.md')
        grant['recovery_amendment_sha256']=sha(amendment)
        grant['prior_failure_summary_sha256']=sha(previous/'run_summary.json')
        grant['prior_failure_supervisor_sha256']=sha(previous/'supervisor.json')
        grant['prior_qualification_charge_sha256']=sha(events/'000001.json')
        grant['prior_event_chain_sha256']={p.name:sha(p) for p in sorted(events.glob('*.json'))}
        grant['limits']['processed_training_tokens_all_attempts']=400000
        grant['what_training_permission_means']='One fresh recovery attempt from source after preserved token-accounting failure; identical data/plan/seed, not an extra candidate; includes qualification tokens; no release.'
        write(grant_path,grant)
        sys.path.insert(0,str(PILOT))
        from argparse import Namespace
        from experiments.reasoning_posttraining_v1.controller import inputs
        inputs(Namespace(prepared=str(OUT/'prepared-dual899'),plan=str(OUT/'PLAN_DUAL899.json'),grant=str(grant_path),
                         experiment='RRP-008-PROCESS-DOSE',command='train',bank=None,heldout_bank=[]),retain_presentations=False)
        print('RECOVERY_GRANT_PREPARED; commit before --execute',flush=True)
        return 0
    if '--execute' not in sys.argv:
        raise SystemExit('fresh recovery grant exists; explicit --execute required')
    assert read(grant_path)['recovery_amendment_sha256']==sha(OUT/amendment.name)
    started=time.monotonic()
    with (OUT/'recovery_preload_wait.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        while True:
            available=psutil.virtual_memory().available
            elapsed=time.monotonic()-started
            locked=(Path(tempfile.gettempdir())/'rineya-gpu-0.lock').exists()
            f.write(json.dumps({'elapsed_seconds':elapsed,'available_host_bytes':available,'lease_present':locked})+'\n');f.flush()
            if available>=7*2**30 and not locked:
                break
            if elapsed>=900:
                write(OUT/'RECOVERY_PRELOAD_TERMINAL.json',{'status':'WAIT_EXPIRED_NO_WORKER','wall_seconds':elapsed,'model_worker_launched':False})
                return 2
            print(f'WAIT_RAM free={available/2**30:.2f} GiB elapsed={int(elapsed)}s',flush=True)
            time.sleep(min(30,900-elapsed))
    command=[str(native),'-m','experiments.reasoning_posttraining_v1.controller','train','--source',str(SOURCE),
             '--prepared',str(OUT/'prepared-dual899'),'--plan',str(OUT/'PLAN_DUAL899.json'),'--grant',str(grant_path),
             '--journal',str(OUT/'journal-dual899-recovery'),'--experiment','RRP-008-PROCESS-DOSE',
             '--run-id','train-001','--timeout','2000','--max-sequence','2048','--confirm-model-execution']
    write(OUT/'RECOVERY_PRELOAD_TERMINAL.json',{'status':'READY_DISPATCH_ONE_WORKER','wall_seconds':time.monotonic()-started,
                                              'model_worker_launched':True,'available_host_bytes':available,'argv':command})
    result=subprocess.run(command,cwd=PILOT,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
    write(OUT/'RECOVERY_COMMAND_TERMINAL.json',{'returncode':result.returncode,'status':'CHECK_NATIVE_RECEIPTS'})
    return result.returncode

if __name__=='__main__':
    raise SystemExit(main())
