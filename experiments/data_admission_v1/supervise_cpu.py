"""Independent CPU-only experiment supervisor; never changes native guards."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

GIB = 2**30
ROOT = Path(__file__).resolve().parents[2]

def dump_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')

def gpu_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return {'returncode': result.returncode, 'raw': result.stdout.strip(), 'error': result.stderr.strip()}
    except Exception as e:
        return {'error': type(e).__name__+': '+str(e)}

def tree_snapshot(pid):
    rss, cpu_seconds = 0, 0.0
    try:
        root = psutil.Process(pid)
        tree = [root]+root.children(recursive=True)
        for item in tree:
            try:
                rss += item.memory_info().rss
                t = item.cpu_times()
                cpu_seconds += t.user+t.system
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    return rss, cpu_seconds

def output_bytes(path):
    total = 0
    if path.exists():
        for entry in path.rglob('*'):
            if entry.is_symlink():
                raise RuntimeError('Linked worker output prohibited')
            if entry.is_file():
                total += entry.stat().st_size
    return total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--worker-out', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--rss-mib', type=int, default=512)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command or not 1 <= args.seconds <= 900 or not 16 <= args.rss_mib <= 512:
        raise SystemExit('Invalid bounded CPU command')
    assert not args.receipt.exists(), 'new receipt directory required'
    out = args.receipt.resolve()
    worker_out = args.worker_out.resolve()
    assert out.is_relative_to(ROOT) and worker_out.is_relative_to(ROOT), 'outputs must stay in experiment worktree'
    out.mkdir(parents=True)
    tmp = out/'tmp'
    tmp.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        USE_TORCH='0', USE_TF='0', USE_FLAX='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        TEMP=str(tmp), TMP=str(tmp), TMPDIR=str(tmp))
    before = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host_available_bytes':psutil.virtual_memory().available,
        'disk_free_bytes':psutil.disk_usage(str(ROOT)).free, 'gpu':gpu_snapshot()}
    limits = {'cpu_host_admission_gib':2.0,'cpu_host_ongoing_gib':1.5,
        'worker_tree_rss_mib':args.rss_mib,'wall_seconds':args.seconds,'disk_free_gib':4.0,
        'worker_output_bytes':50*2**20,'stdout_stderr_bytes':8*2**20,'poll_seconds':0.1,
        'native_guards':'UNCHANGED; no pretrained/native model worker in this study'}
    dump_new(out/'REQUEST.json', {'command':command,'cwd':str(ROOT),'limits':limits,
        'before':before,'python':sys.version,'psutil':psutil.__version__,
        'supervisor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    admission = before['host_available_bytes'] >= 2*GIB and before['disk_free_bytes'] >= 4*GIB
    if not admission:
        dump_new(out/'SUPERVISOR.json', {'status':'BLOCKED_CPU_ADMISSION','before':before,
            'limits':limits,'worker_started':False,'no_automatic_retry':True})
        print('BLOCKED_CPU_ADMISSION')
        return 2
    process = job = None
    reason = None
    code = None
    started = time.monotonic()
    peak_rss = 0
    min_host = before['host_available_bytes']
    max_cpu = 0.0
    try:
        # Reuse containment only; the inherited native run_bounded and its floors
        # are never called or changed by this independent CPU supervisor.
        from experiments.opallo_native_collector_v1.supervisor import WindowsJob
        with (out/'stdout.log').open('xb') as stdout, (out/'stderr.log').open('xb') as stderr, (out/'HARDWARE.jsonl').open('x',encoding='utf-8') as journal:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            job = WindowsJob(process)
            last_disk = last_written = started-10
            last_gpu = started
            disk = before['disk_free_bytes']
            gpu = before['gpu']
            while True:
                now = time.monotonic()
                rss,cpu = tree_snapshot(process.pid)
                available = psutil.virtual_memory().available
                peak_rss = max(peak_rss,rss)
                max_cpu = max(max_cpu,cpu)
                min_host = min(min_host,available)
                if now-last_disk >= 1:
                    disk = psutil.disk_usage(str(ROOT)).free
                    size = output_bytes(worker_out)
                    last_disk = now
                if now-last_gpu >= 5:
                    gpu = gpu_snapshot()
                    last_gpu = now
                if now-last_written >= 0.2 or process.poll() is not None:
                    journal.write(json.dumps({'seconds':now-started,'host_available_bytes':available,
                        'worker_tree_rss_bytes':rss,'observed_cpu_seconds':cpu,'disk_free_bytes':disk,'gpu':gpu})+'\n')
                    journal.flush()
                    last_written = now
                if now-started >= args.seconds: reason='CPU_WALL_LIMIT'
                elif rss > args.rss_mib*2**20: reason='CPU_TREE_RSS_LIMIT'
                elif available < 1.5*GIB: reason='CPU_HOST_HEADROOM'
                elif disk < 4*GIB: reason='DISK_FLOOR'
                elif size > 50*2**20: reason='OUTPUT_LIMIT'
                elif stdout.tell()+stderr.tell() > 8*2**20: reason='LOG_LIMIT'
                if reason or process.poll() is not None:
                    break
                time.sleep(0.1)
            if reason:
                job.kill()
            code = process.wait(timeout=10)
            if code and reason is None: reason='WORKER_NONZERO_EXIT'
    except BaseException as exc:
        reason = reason or type(exc).__name__+': '+str(exc)
    finally:
        if job:
            job.close()
        if process and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        after = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'host_available_bytes':psutil.virtual_memory().available,
            'disk_free_bytes':psutil.disk_usage(str(ROOT)).free,'gpu':gpu_snapshot()}
        receipt = {'status':'PASS_CPU_PROCESS' if reason is None and code == 0 else 'BLOCKED',
            'reason':reason,'exit_code':code,'worker_started':process is not None,
            'elapsed_seconds':time.monotonic()-started,'peak_observed_tree_rss_bytes':peak_rss,
            'minimum_observed_host_available_bytes':min_host,'peak_observed_cpu_seconds':max_cpu,
            'worker_output_bytes':output_bytes(worker_out),'limits':limits,'before':before,'after':after,
            'native_model_loads':0,'gpu_worker_launches':0,'paid_calls':0,'no_automatic_retry':True,
            'measurement_note':'0.1-second observed peaks are sampled, not instantaneous OS hard RSS guarantees'}
        dump_new(out/'SUPERVISOR.json',receipt)
        print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='PASS_CPU_PROCESS' else 2

if __name__ == '__main__':
    raise SystemExit(main())
