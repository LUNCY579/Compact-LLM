"""One verified export and six fixed probes; no automatic retry or deletion."""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from paths import BASE, HERE, OUT, PILOT, MQPY, RABPY, BASELINE, read, sha, write
import eval_common as auxiliary
from launch import verify_frozen

LEGS = (('dose', 'process', 'process-transition-probe-001', 384, 128),
        ('select', 'process', 'select-views-001', 384, 128),
        ('retention', 'retention', 'process-transition-probe-001', 150, 24),
        ('long_context', 'long_context', 'process-transition-probe-001', 256, 24),
        ('binding', 'auxiliary', None, 384, 24), ('diagnostic', 'auxiliary', None, 896, 24))


def guard(envelope, reserve=0):
    if time.time() + reserve > envelope['deadline_unix']:
        raise RuntimeError('remaining registered session cannot fit this stage')
    if shutil.disk_usage(OUT).free < envelope['free_disk_floor_bytes']:
        raise RuntimeError('unchanged free-disk floor refused')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) > envelope['new_records_disk_bytes']:
        raise RuntimeError('logical record budget exhausted')


@contextmanager
def parent_lease():
    path = OUT / 'POSTTRAIN_CONTROLLER_LEASE.json'
    owner = {'owner_pid': os.getpid(), 'nonce': str(uuid.uuid4()), 'prelaunch_binding_sha256': sha(OUT / 'PRELAUNCH_BINDING.json')}
    write(path, owner)
    try:
        yield owner
    finally:
        if read(path) != owner:
            raise RuntimeError('posttrain controller ownership changed; preserve lease')
        path.unlink()


def auxiliary_grant(gate, arm, envelope):
    baseline = read(BASELINE / 'BASELINE_PLAN.json')
    admission = read(OUT / 'DATA_USE_ADMISSION.json')['evaluation_scope']
    binding = admission['legacy_binding384']['input_gold_and_baseline_report_pins']['DEV_INPUTS.jsonl']
    diagnostic = admission['controlled_diagnostic896']['input_gold_and_report_pins']['candidate-v1/DIAGNOSTIC_INPUTS.jsonl']
    inputs = {'binding': binding, 'diagnostic': diagnostic}
    receipt = read(arm / 'ARM.json')
    value = {'schema': 1, 'record': 'BINDING_INTEGRATION_AUX_GRANT', 'experiment': 'BINDING-INTEGRATION-001',
        'decision': 'APPROVED', 'authorization_reference': 'Two pre-registered auxiliary legs; local retry is only a subcap of the one global explicit inference retry.',
        'arm_path': str(arm), 'arm_weight_sha256': receipt['arm_weight_sha256'],
        'arm_receipts_sha256': {name: sha(arm / name) for name in ('ARM.json', 'MATERIALIZATION.json')},
        'source_dir': gate['source'], 'sidecars_sha256': baseline['sidecars_sha256'],
        'build_gate_path': str(OUT / 'BUILD_GATE.json'), 'build_gate_sha256': sha(OUT / 'BUILD_GATE.json'),
        'data_use_admission_sha256': sha(OUT / 'DATA_USE_ADMISSION.json'),
        'datasets': {kind: {'input_path': value['path'], 'input_sha256': value['sha256'],
            'rows': auxiliary.KINDS[kind][0], 'parent_counts': {str(k): v for k, v in auxiliary.KINDS[kind][1].items()}}
            for kind, value in inputs.items()},
        'code_sha256': {name: sha(HERE / name) for name in auxiliary.WORKER_FILES},
        'core_code_sha256': {name: sha(path) for name, path in auxiliary.CORE_FILES.items()},
        'pilot_code_sha256': auxiliary.original.pilot_code_files(PILOT), 'runtime_versions': baseline['runtime_versions'],
        'python_path': str(RABPY), 'output_root': str(OUT / 'eval'), 'limits': auxiliary.LIMITS,
        'deadline_unix': envelope['deadline_unix']}
    write(OUT / 'AUX_PROBE_GRANT.json', value)
    auxiliary.context(OUT / 'AUX_PROBE_GRANT.json', OUT / 'BUILD_GATE.json')


def main():
    verify_frozen()
    run = OUT / 'journal/runs/train-001'
    summary, result = read(run / 'run_summary.json'), read(run / 'result.json')
    if (summary['returncode'] or summary['worker_failed'] or summary['budget_after']['budget_breached'] or
            summary['budget_after']['active_reservations'] or result['status'] != 'COMPLETE' or not result['complete']):
        raise ValueError('successful settled integration training is required')
    envelope, gate = read(OUT / 'SESSION_ENVELOPE.json'), read(OUT / 'BUILD_GATE.json')
    logs = OUT / 'posttrain'; logs.mkdir(exist_ok=False)
    (logs / 'reservations').mkdir()
    attempts, final_bindings = [], {}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    try:
        with parent_lease() as owner:
            write(logs / 'DISPATCH_RESERVATION.json', {'owner': owner, 'legs': [row[0] for row in LEGS], 'automatic_retries': 0})
            guard(envelope, 600)
            arm, cpu = OUT / 'arms/integrated', logs / 'build-integrated'
            command = [str(MQPY), str(HERE / 'cpu_stage.py'), '--out', str(cpu), '--', '--source', gate['source'],
                '--checkpoint', result['checkpoint'], '--training-result', str(run / 'result.json'), '--out', str(arm)]
            print('START build-integrated', flush=True)
            rc = subprocess.run(command, cwd=PILOT, env=env).returncode
            if rc:
                raise RuntimeError('qualified export failed; preserve its evidence')
            weight = arm / 'model.safetensors'; arm_receipt = read(arm / 'ARM.json')
            if sha(weight) != arm_receipt['arm_weight_sha256']:
                raise ValueError('exported bytes differ from the qualified receipt')
            guard(envelope, 600); started = time.monotonic()
            with (cpu / 'compact.log').open('xb') as stream:
                rc = subprocess.run(['compact.exe', '/C', '/EXE:LZX', '/Q', str(weight)], stdout=stream,
                    stderr=subprocess.STDOUT, timeout=600, creationflags=subprocess.CREATE_NO_WINDOW).returncode
            after = sha(weight)
            write(cpu / 'COMPRESSION.json', {'returncode': rc, 'before_sha256': arm_receipt['arm_weight_sha256'],
                'after_sha256': after, 'logical_bytes': weight.stat().st_size, 'wall_seconds': time.monotonic() - started,
                'free_disk_bytes': shutil.disk_usage(OUT).free, 'model_execution': False})
            if rc or after != arm_receipt['arm_weight_sha256']:
                raise ValueError('lossless export compression did not verify')
            print('VERIFIED build-integrated', flush=True)
            auxiliary_grant(gate, arm, envelope)
            for suffix, kind, instrument, n, maximum in LEGS:
                verify_frozen(); guard(envelope, 900)
                current = auxiliary.global_costs()
                if (current['attempts'] + 1 > envelope['probe_workers_all_attempts'] or
                    current['worker_seconds'] + 900 > envelope['probe_worker_seconds_all_attempts'] or
                    current['rows'] + n > envelope['probe_rows_reserved_all_attempts'] or
                    current['generation_tokens'] + n * maximum > envelope['probe_generation_tokens_reserved_all_attempts']):
                    raise RuntimeError('global probe budget cannot fit the next full reservation')
                label = 'integrated-' + suffix; out = OUT / 'eval' / label
                write(logs / 'reservations' / (label + '.json'), {'suffix': suffix, 'attempt': 1, 'out': str(out),
                    'reserved_rows': n, 'reserved_generation_tokens': n * maximum, 'reserved_worker_seconds': 900,
                    'session_envelope_sha256': sha(OUT / 'SESSION_ENVELOPE.json'), 'parent_owner': owner})
                auxiliary.global_costs()
                if kind == 'auxiliary':
                    command = [str(RABPY), str(HERE / 'eval_probe.py'), '--kind', suffix,
                        '--grant', str(OUT / 'AUX_PROBE_GRANT.json'), '--build-gate', str(OUT / 'BUILD_GATE.json'), '--confirm-model-execution']
                else:
                    command = [str(MQPY), str(HERE / 'run_probe.py'), '--arm', str(arm), '--label', 'bindingintegration001_' + suffix,
                        '--kind', kind, '--instrument', instrument, '--out', str(out), '--build-gate', str(OUT / 'BUILD_GATE.json')]
                print('START ' + label, flush=True); started = time.monotonic()
                with (logs / (label + '.log')).open('xb') as stream:
                    rc = subprocess.run(command, cwd=PILOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
                supervisor_path = out / ('supervisor.json' if kind == 'auxiliary' else 'worker/supervisor.json')
                charged = 900.0
                if supervisor_path.exists():
                    supervisor = read(supervisor_path); wall = supervisor.get('wall_seconds')
                    if type(supervisor.get('returncode')) is int and supervisor.get('status') in ('EXITED', 'TIMEOUT_OR_INTERRUPTED') and type(wall) in (int, float) and math.isfinite(wall) and wall >= 0:
                        charged = float(wall)
                record = {'leg': label, 'out': str(out), 'returncode': rc, 'wall_seconds': time.monotonic() - started,
                          'worker_seconds_charged': charged, 'reserved_rows': n, 'reserved_generation_tokens': n * maximum}
                write(logs / (label + '.json'), record); attempts.append(record)
                if rc:
                    raise RuntimeError('probe failed; any one permitted retry requires explicit recovery')
                binding_path = out / ('AUX_BINDING.json' if kind == 'auxiliary' else 'PROBE_WEIGHT_BINDING.json')
                bound = read(binding_path)
                if bound['status'] != 'COMPLETE' or not bound['complete']:
                    raise ValueError('probe has no successful custody receipt')
                final_bindings[label] = sha(binding_path)
                print('VERIFIED ' + label, flush=True)
            # Cost-only helper is stdlib; scientific scoring remains a separate RAB process.
            import report
            cost = report.probe_cost_accounting(OUT, envelope)
            guard(envelope)
            completed = {'status': 'COMPLETE_SIX_PROBES', 'attempts': attempts, 'final_binding_sha256': final_bindings,
                'cost_accounting': cost, 'probe_worker_seconds': cost['probe_worker_seconds'],
                'probe_rows_reserved': cost['probe_rows_reserved'], 'probe_generation_tokens_reserved': cost['probe_generation_tokens_reserved'],
                'session_elapsed_seconds': time.time() - envelope['session_start_unix']}
        # Publish completion only after the parent admission lease is released.
        guard(envelope)
        write(logs / 'TERMINAL.json', completed)
        return 0
    except Exception as exc:
        write(logs / 'TERMINAL.json', {'status': 'FAILED_OR_BLOCKED', 'attempts': attempts,
            'failure': {'type': type(exc).__name__, 'message': str(exc)},
            'global_reservation_accounting': auxiliary.global_costs(), 'quality_claim': False})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
