"""Bounded RAM wait then a single unchanged native training command."""
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
    native = BASE / '.venv-mq001/Scripts/python.exe'
    previous = OUT / 'journal-dual899/runs/train-001'
    failure = read(previous / 'failure.json')
    summary = read(previous / 'run_summary.json')
    used = summary['budget_after']['used_or_reserved']
    assert failure['type'] == 'RuntimeError' and failure['message'].startswith('host RAM guard refused')
    assert summary['supervisor']['returncode'] == 2
    assert used['processed_training_tokens_all_attempts'] == used['evaluation_and_qualification_model_calls'] == 0
    assert not (previous / 'binding.json').exists() and not (previous / 'microfit').exists()
    amendment = Path(__file__).with_name('AMENDMENT_02.md')
    grant_path = OUT / 'GRANT_DUAL899_PRELOAD_RETRY.json'
    if not grant_path.exists():
        with (OUT / amendment.name).open('xb') as f:
            f.write(amendment.read_bytes())
        grant = read(OUT / 'GRANT_DUAL899.json')
        grant['preload_retry_amendment_sha256'] = sha(amendment)
        grant['prior_refusal_summary_sha256'] = sha(previous / 'run_summary.json')
        grant['prior_refusal_journal'] = str(previous.parent.parent)
        for key in ('model_worker_wall_seconds_and_conservative_gpu_seconds','total_owned_work_wall_seconds','new_disk_bytes'):
            grant['limits'][key] -= used[key]
        write(grant_path, grant)
        print('PREPARED_GRANT_ONLY; commit amendment before --execute', flush=True)
        return 0
    if '--execute' not in sys.argv:
        raise SystemExit('grant already exists; explicit --execute required')
    assert sha(OUT / amendment.name) == read(grant_path)['preload_retry_amendment_sha256']
    logfile = OUT / 'preload_wait.jsonl'
    started = time.monotonic()
    with logfile.open('x', encoding='utf-8', newline='\n') as log:
        while True:
            available = psutil.virtual_memory().available
            elapsed = time.monotonic() - started
            locked = (Path(tempfile.gettempdir()) / 'rineya-gpu-0.lock').exists()
            log.write(json.dumps({'elapsed_seconds':elapsed,'available_host_bytes':available,'lease_present':locked})+'\n')
            log.flush()
            if available >= 7 * 2**30 and not locked:
                break
            if elapsed >= 900:
                write(OUT / 'PRELOAD_WAIT_TERMINAL.json', {'status':'WAIT_EXPIRED_NO_WORKER', 'model_worker_launched':False,'wall_seconds':elapsed,'available_host_bytes':available})
                return 2
            print(f'WAIT_RAM free={available / 2**30:.2f} GiB elapsed={int(elapsed)}s lease={locked}', flush=True)
            time.sleep(min(30, 900-elapsed))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    cmd = [str(native), '-m','experiments.reasoning_posttraining_v1.controller','train',
           '--source',str(SOURCE),'--prepared',str(OUT/'prepared-dual899'),
           '--plan',str(OUT/'PLAN_DUAL899.json'),'--grant',str(grant_path),
           '--journal',str(OUT/'journal-dual899-preloadretry'),'--experiment','RRP-008-PROCESS-DOSE',
           '--run-id','train-001','--timeout','2000','--max-sequence','2048','--confirm-model-execution']
    write(OUT / 'PRELOAD_WAIT_TERMINAL.json', {'status':'READY_DISPATCH_ONE_WORKER','model_worker_launched':True,
                                             'wall_seconds':time.monotonic()-started,'available_host_bytes':available,'argv':cmd})
    result = subprocess.run(cmd,cwd=PILOT,env=env)
    write(OUT / 'TRAIN_COMMAND_TERMINAL.json', {'returncode':result.returncode,'status':'CHECK_NATIVE_RECEIPTS'})
    return result.returncode

if __name__ == '__main__':
    raise SystemExit(main())
