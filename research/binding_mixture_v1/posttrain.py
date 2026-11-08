"""One qualified export and five fixed probes; no automatic retries or deletion."""
from pathlib import Path
import os
import math
import shutil
import subprocess
import sys
import time

from paths import BASE, OUT, HERE, PILOT, sha, read, write

PYTHON = BASE/'.venv-mq001/Scripts/python.exe'
BASELINE_ROOT = BASE/'rineya-dataset-retention-records/binding-dev-baseline001-20260920'
LEGS = (('dose', 'process', 'process-transition-probe-001', 384, 128),
        ('select', 'process', 'select-views-001', 384, 128),
        ('retention', 'retention', 'process-transition-probe-001', 150, 24),
        ('long_context', 'long_context', 'process-transition-probe-001', 256, 24),
        ('binding', 'binding', None, 384, 24))


def guard(envelope, seconds_needed=0):
    if time.time()+seconds_needed > envelope['deadline_unix']:
        raise RuntimeError('Registered session cannot fit this stage')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) > envelope['new_records_disk_bytes']:
        raise RuntimeError('New experiment logical disk budget exhausted')
    if shutil.disk_usage(OUT).free < envelope['free_disk_floor_bytes']:
        raise RuntimeError('Unchanged physical free-space floor refused')


def main():
    sys.path.insert(0, str(BASE/'ChatBubble-dose-param'))
    from research.retention_highdose_lora_resume_v1.paired import validate_probe_binding
    from research.binding_mixture_v1 import binding_common
    from launch import verify_frozen
    verify_frozen()
    run = OUT/'journal/runs/train-001'
    summary, result = read(run/'run_summary.json'), read(run/'result.json')
    if (summary['returncode'] or summary['worker_failed'] or summary['budget_after']['budget_breached']
            or result['status'] != 'COMPLETE' or result['complete'] is not True):
        raise ValueError('A successful qualified and settled native run is required')
    envelope, gate = read(OUT/'SESSION_ENVELOPE.json'), read(OUT/'BUILD_GATE.json')
    logs = OUT/'posttrain'
    logs.mkdir(exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    attempts, seconds, rows_reserved, tokens_reserved = [], 0.0, 0, 0
    try:
        guard(envelope, 600)
        arm, cpu = OUT/'arms/mixed', logs/'build-mixed'
        command = [str(PYTHON), str(HERE/'cpu_stage.py'), '--out', str(cpu), '--',
                   '--source', gate['source'], '--checkpoint', result['checkpoint'],
                   '--training-result', str(run/'result.json'), '--out', str(arm)]
        print('START build-mixed', flush=True)
        rc = subprocess.run(command, cwd=PILOT, env=env).returncode
        if rc:
            raise RuntimeError(f'Qualified candidate export failed; preserved rc{rc}')
        receipt = read(arm/'ARM.json')
        weight = arm/'model.safetensors'
        if sha(weight) != receipt['arm_weight_sha256']:
            raise ValueError('Exported candidate bytes differ from receipt')
        guard(envelope, 600)
        started = time.monotonic()
        with (cpu/'compact.log').open('xb') as log:
            rc = subprocess.run(['compact.exe', '/C', '/EXE:LZX', '/Q', str(weight)],
                                stdout=log, stderr=subprocess.STDOUT, timeout=600,
                                creationflags=subprocess.CREATE_NO_WINDOW).returncode
        after = sha(weight)
        write(cpu/'COMPRESSION.json', {'returncode': rc, 'before_sha256': receipt['arm_weight_sha256'],
              'after_sha256': after, 'logical_bytes': weight.stat().st_size,
              'wall_seconds': time.monotonic()-started, 'free_disk_bytes': shutil.disk_usage(OUT).free,
              'model_execution': False})
        if rc or after != receipt['arm_weight_sha256']:
            raise ValueError('Transparent candidate compression did not verify')
        print('VERIFIED build-mixed', flush=True)
        baseline_path = BASELINE_ROOT/'BASELINE_PLAN.json'
        baseline = read(baseline_path)
        binding_grant = {'schema': 1, 'record': 'BINDING_MIXTURE_DEV_GRANT',
            'experiment': 'BINDING-MIXTURE-001', 'decision': 'APPROVED',
            'authorization_reference': 'Registered five-leg mixed-data experiment; local retry is a subcap of the one total inference retry in SESSION_ENVELOPE.',
            'build_gate_path': str(OUT/'BUILD_GATE.json'), 'build_gate_sha256': sha(OUT/'BUILD_GATE.json'),
            'baseline_plan_path': str(baseline_path), 'baseline_plan_sha256': sha(baseline_path),
            'arm_path': str(arm), 'arm_weight_sha256': receipt['arm_weight_sha256'],
            'limits': binding_common.LIMITS, 'deadline_unix': envelope['deadline_unix'],
            'code_sha256': {name: sha(HERE/name) for name in ('binding_common.py', 'binding_probe.py', 'run_probe.py')}}
        write(OUT/'BIND_PROBE_GRANT.json', binding_grant)
        for suffix, kind, instrument, count, max_new in LEGS:
            guard(envelope, 900)
            if (len(attempts)+1 > envelope['probe_workers_all_attempts']
                    or seconds+900 > envelope['probe_worker_seconds_all_attempts']
                    or rows_reserved+count > envelope['probe_rows_reserved_all_attempts']
                    or tokens_reserved+count*max_new > envelope['probe_generation_tokens_reserved_all_attempts']):
                raise RuntimeError('All-attempt probe envelope cannot fit this leg')
            rows_reserved += count
            tokens_reserved += count*max_new
            label = 'mixed-'+suffix
            attempt = OUT/'eval'/label
            if kind == 'binding':
                command = [baseline['python_path'], str(HERE/'binding_probe.py'), '--arm', str(arm),
                    '--out', str(attempt), '--build-gate', str(OUT/'BUILD_GATE.json'),
                    '--baseline-plan', str(baseline_path), '--baseline-plan-sha256', sha(baseline_path),
                    '--grant', str(OUT/'BIND_PROBE_GRANT.json'), '--confirm-model-execution']
            else:
                command = [str(PYTHON), str(HERE/'run_probe.py'), '--arm', str(arm),
                    '--label', 'bindingmixture001_'+suffix, '--kind', kind, '--instrument', instrument,
                    '--out', str(attempt), '--build-gate', str(OUT/'BUILD_GATE.json')]
            print('START '+label, flush=True)
            started = time.monotonic()
            with (logs/(label+'.log')).open('xb') as log:
                rc = subprocess.run(command, cwd=PILOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            supervisor = attempt/('supervisor.json' if kind == 'binding' else 'worker/supervisor.json')
            charged = 900.0
            if supervisor.is_file():
                observed = read(supervisor)
                if (type(observed.get('returncode')) is int
                        and observed.get('status') in ('EXITED', 'TIMEOUT_OR_INTERRUPTED')
                        and type(observed.get('wall_seconds')) in (int, float)
                        and math.isfinite(observed['wall_seconds']) and observed['wall_seconds'] >= 0):
                    charged = observed['wall_seconds']
            seconds += charged
            record = {'leg': label, 'out': str(attempt), 'returncode': rc,
                'wall_seconds': time.monotonic()-started, 'worker_seconds_charged': charged,
                'reserved_rows': count, 'reserved_generation_tokens': count*max_new}
            write(logs/(label+'.json'), record)
            attempts.append(record)
            if rc:
                raise RuntimeError('Probe failed; any registered retry requires a fresh explicit recovery')
            if kind == 'binding':
                bound = read(attempt/'BINDING.json')
                if bound['status'] != 'COMPLETE' or bound['complete'] is not True:
                    raise ValueError('Binding probe has no completed custody receipt')
            else:
                validate_probe_binding(attempt/'result.json', attempt/'PROBE_WEIGHT_BINDING.json', kind)
            print('VERIFIED '+label, flush=True)
        guard(envelope)
        write(logs/'TERMINAL.json', {'status': 'COMPLETE_FIVE_PROBES', 'attempts': attempts,
            'probe_worker_seconds': seconds, 'probe_rows_reserved': rows_reserved,
            'probe_generation_tokens_reserved': tokens_reserved,
            'session_elapsed_seconds': time.time()-envelope['session_start_unix']})
        return 0
    except Exception as exc:
        write(logs/'TERMINAL.json', {'status': 'FAILED_OR_BLOCKED', 'attempts': attempts,
            'failure': {'type': type(exc).__name__, 'message': str(exc)},
            'probe_worker_seconds': seconds, 'probe_rows_reserved': rows_reserved,
            'probe_generation_tokens_reserved': tokens_reserved, 'quality_claim': False})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
