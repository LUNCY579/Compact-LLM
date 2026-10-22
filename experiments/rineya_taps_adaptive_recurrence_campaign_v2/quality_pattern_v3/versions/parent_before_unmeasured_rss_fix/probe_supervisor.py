"""Parent-owned, one-attempt passive probe with unchanged GPU resource floors."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import psutil
from .authority import PACKAGE, CAMPAIGN, ROOT, GIB, sha, validate_probe_unused
from ..bulk_restart_supervisor import gpu_inventory
from ..native_supervisor import append
from ..supervision import WindowsJob, resource_sample


def check_parent(source,out,admission_path):
    admission=json.loads(admission_path.read_text(encoding='utf-8'))
    validate_probe_unused(admission)
    if PACKAGE not in out.parents or out.exists():
        raise PermissionError('exclusive new V3 output required')
    if str(source)!=admission['source'] or sys.executable!=admission['interpreter']:
        raise PermissionError('canonical source/interpreter binding differs')
    for relative,digest in admission['dispatch_hashes'].items():
        if sha(ROOT/relative)!=digest:raise PermissionError('frozen source changed:'+relative)
    for relative,digest in admission['checkpoint_files'].items():
        if sha(source/relative)!=digest:raise PermissionError('canonical asset changed:'+relative)
    review_path=(CAMPAIGN/admission['review_receipt']).resolve(strict=True)
    if PACKAGE not in review_path.parents:
        raise PermissionError('probe review must be V3-local')
    review=json.loads(review_path.read_text(encoding='utf-8'))
    if review['status']!='PASS_PASSIVE_PROBE_FP32_DISPATCH' or review['admission_sha256']!=sha(admission_path):
        raise PermissionError('independent exact admission review required')
    events=[json.loads(x) for x in (CAMPAIGN/'BUDGET_LEDGER.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    spent=sum(e.get('native_worker_seconds',0) for e in events if e.get('event')=='native_completed')
    if spent+900+1800>21600:raise RuntimeError('complete study and final reserve do not fit')
    inventory=gpu_inventory()
    if inventory['uuid']!=admission['gpu_uuid'] or inventory['driver_version']!=admission['gpu_driver_version'] or inventory['free_bytes']<6*GIB:
        raise MemoryError('GPU identity or original six-GiB free floor unavailable')
    if psutil.virtual_memory().available<7*GIB:raise MemoryError('original parent seven-GiB host floor unavailable')
    if shutil.disk_usage(CAMPAIGN).free<15*GIB or sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file())>1536*1024**2:
        raise MemoryError('evidence disk reserve unavailable')
    if (CAMPAIGN/'native_worker.lock').exists():raise RuntimeError('live or unknown worker lock')
    return admission,inventory


def run(source,out,admission_path):
    source,out,admission_path=source.resolve(),out.resolve(),admission_path.resolve()
    admission,inventory=check_parent(source,out,admission_path)
    lock=CAMPAIGN/'native_worker.lock';token=uuid.uuid4().hex
    with lock.open('x',encoding='utf-8') as handle:json.dump({'token':token,'parent_pid':os.getpid(),'out':str(out),'lane':admission['lane']},handle)
    start=time.perf_counter();child=None;job=None;assigned=False;created=False;child_start=None
    receipt={'status':'FAIL_BEFORE_WORKER','study_id':admission['study_id'],'lane':admission['lane'],
        'run':str(out),'admission_sha256':sha(admission_path),'native_worker_seconds':0,
        'started_utc':datetime.now(timezone.utc).isoformat(),'peak_rss_bytes':0,'gpu_initial_inventory':inventory}
    try:
        out.mkdir(parents=True,exist_ok=False);created=True
        (out/'admission.json').write_bytes(admission_path.read_bytes())
        (out/'review.json').write_bytes((CAMPAIGN/admission['review_receipt']).read_bytes())
        (out/'source_manifest.json').write_text(json.dumps({'sha256':admission['dispatch_hashes'],'head':admission['head']},indent=2)+'\n',encoding='utf-8')
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'HF_HOME':str(out/'result/hf_cache'),'HF_MODULES_CACHE':str(out/'result/hf_cache/modules'),
            'TOKENIZERS_PARALLELISM':'false','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        release=out/'release.json'
        command=[sys.executable,'-B','-m','experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.probe_entry',
            '--release',str(release),'--source',str(source),'--out',str(out/'result'),'--admission',str(admission_path)]
        receipt['command']=command;job=WindowsJob()
        with (out/'stdout.log').open('w',encoding='utf-8') as stdout,(out/'stderr.log').open('w',encoding='utf-8') as stderr:
            child_start=time.perf_counter();env['PASSIVE_PROBE_PARENT_STARTED_PERF_COUNTER']=repr(child_start)
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|0x4)
            receipt['worker_pid']=child.pid;job.assign(child);assigned=True;job.resume_suspended(child)
            append(CAMPAIGN/'BUDGET_LEDGER.jsonl',{'event':'native_started','study_id':admission['study_id'],'lane':admission['lane'],'run':str(out),'reserved_seconds':900,'worker_pid':child.pid,'timestamp_utc':datetime.now(timezone.utc).isoformat()})
            release.write_text(json.dumps({'contained':True,'token':token}),encoding='utf-8')
            last_progress=time.monotonic();last_sizes=None;last_gpu=0
            while child.poll() is None:
                now=time.monotonic()
                try:sample=resource_sample(child)
                except psutil.NoSuchProcess:
                    if child.poll() is not None:break
                    raise
                sample['worker_seconds']=time.perf_counter()-child_start
                if now-last_gpu>=10:inventory=gpu_inventory();last_gpu=now;sample['gpu_global_inventory']=inventory
                append(out/'resources.jsonl',sample);receipt['peak_rss_bytes']=max(receipt['peak_rss_bytes'],sample['rss_bytes'])
                paths=[out/'result'/n for n in ('registration.json','qualification.json','forwards.jsonl','feature_events.jsonl','readouts.jsonl','receipt.json')]
                sizes=tuple((p.stat().st_size,p.stat().st_mtime_ns) if p.exists() else None for p in paths)
                if sizes!=last_sizes:last_sizes=sizes;last_progress=now
                reason=('HARD_DEADLINE' if sample['worker_seconds']>900 else 'HUNG_FORWARD' if now-last_progress>180 else
                    'RSS_CEILING' if sample['rss_bytes']>6*GIB else 'HOST_RESERVE' if sample['host_available_bytes']<GIB else
                    'GPU_FREE_RESERVE' if inventory['free_bytes']<GIB else 'GPU_IDENTITY' if inventory['uuid']!=admission['gpu_uuid'] else
                    'DISK_RESERVE' if shutil.disk_usage(CAMPAIGN).free<15*GIB or sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file())>1536*1024**2 else None)
                if reason:receipt.update(status='SUPERVISOR_SAFETY_STOP',stop_reason=reason);job.terminate();break
                time.sleep(1)
            child.wait(timeout=15);receipt.update(exit_code=child.returncode,native_worker_seconds=time.perf_counter()-child_start)
            if receipt['status']!='SUPERVISOR_SAFETY_STOP':receipt['status']='WORKER_TERMINAL' if child.returncode==0 else 'WORKER_NONZERO'
    except BaseException as error:
        receipt['error']=type(error).__name__+': '+str(error)
        if child is not None:
            job.terminate() if assigned else child.kill();child.wait(timeout=15)
            receipt['native_worker_seconds']=time.perf_counter()-child_start
    finally:
        if job is not None:job.close()
        receipt.update(supervisor_wall_seconds=time.perf_counter()-start,ended_utc=datetime.now(timezone.utc).isoformat())
        if created:(out/'terminal_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        append(CAMPAIGN/'BUDGET_LEDGER.jsonl',{'event':'native_completed',**receipt})
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))['token']==token:lock.unlink()
    print(json.dumps(receipt));return 0 if receipt['status']=='WORKER_TERMINAL' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out','admission'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(args.source,args.out,args.admission))
