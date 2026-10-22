"""Two gold-free probes using the exact previously qualified greedy decoder."""
import argparse
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if __package__:
    from . import eval_common as c
else:
    import eval_common as c


def setup():
    head = subprocess.check_output(['git', '-C', str(c.PILOT), 'rev-parse', 'HEAD'], text=True).strip()
    c.require(head == '5b4c22dae78f587c41575269f419087b0c7997f6', 'qualified inference checkout changed')
    sys.path.insert(0, str(c.PILOT))


def reservation_path(kind, attempt):
    return c.OUT / 'eval' / ('AUX_RESERVATION_' + c.slot(kind, attempt) + '.json')


def verify_reservation(args, grant, rows, out):
    reservation = c.read(reservation_path(args.kind, args.attempt))
    owner = c.read(c.OUT / 'eval/AUX_CONTROLLER_LEASE.json')
    launch, claim = c.read(out / 'launch.json'), c.read(out / 'worker_claim.json')
    n = len(rows)
    c.require(reservation['kind'] == args.kind and reservation['attempt'] == args.attempt and
        reservation['grant_sha256'] == c.sha(args.grant) and reservation['rows'] == n and
        reservation['forward_calls'] == reservation['reserved_generated_tokens'] == n * 24 and
        reservation['processed_input_tokens'] == sum(len(row['prompt_ids']) for row in rows) + n * 24 and
        reservation['controller_owner'] == owner and
        owner['grant_sha256'] == c.sha(args.grant) and
        reservation['controller_lease_sha256'] == c.sha(c.OUT / 'eval/AUX_CONTROLLER_LEASE.json') and
        owner['owner_pid'] == launch['supervisor_pid'] == claim['supervisor_pid'], 'private auxiliary worker lacks exact ownership/reservation')
    parent_path = c.OUT / 'posttrain/reservations' / (c.slot(args.kind, args.attempt) + '.json')
    c.require(reservation['parent_reservation_sha256'] == c.sha(parent_path), 'parent all-attempt reservation changed')
    c.costs(grant, args.grant); c.global_costs()


def worker(args):
    setup()
    from experiments.model_quality_gpu_v1.supervisor import claim_worker
    out = Path(args.out).resolve(); claim_worker(out)
    before = after = None; started = time.monotonic()
    terminal = {'status': 'FAILED', 'complete': False, 'failure': 'worker did not finish',
                'kind': args.kind, 'grant_sha256': c.sha(args.grant), 'build_gate_sha256': c.sha(args.build_gate)}
    try:
        grant, gate = c.context(args.grant, args.build_gate)
        c.require(Path(sys.executable).resolve() == c.RABPY.resolve() and
                  out == c.OUT / 'eval' / c.slot(args.kind, args.attempt), 'private worker interpreter/output differs')
        rows = c.load_inputs(grant, args.kind)
        verify_reservation(args, grant, rows, out)
        from experiments.model_quality_gpu_v1.resources import guard, lease
        with lease(Path(tempfile.gettempdir()) / 'rineya-gpu-0.lock'):
            c.phase_guard(grant, 900)
            import psutil
            c.require(psutil.virtual_memory().available >= 5 * 2**30, 'unchanged 5GiB pre-load host guard refused')
            resources = guard(before_load=True, phase='qualify')
            before = c.snapshot(grant, args.kind)
            import torch
            from experiments.architecture_benchmark_001 import run
            from research.binding_signal_eval_v1.probe import generate
            torch.set_num_threads(1); torch.manual_seed(17); torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            c.require(torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False and
                      torch.get_float32_matmul_precision() == 'highest', 'qualified numeric controls failed')
            c.write(out / 'PRECHECK.json', {'status': 'PRECHECK_PASSED', 'kind': args.kind, 'before': before,
                'resources_before': resources, 'numerics': c.NUMERICS, 'gold_file_read': False,
                'model_input_fields': ['prompt_ids'], 'worker_input_fields': ['id', 'parent_id', 'prompt', 'prompt_ids']})
            model, tokenizer, unused = run._load(Path(grant['arm_path']), 'cuda:0')
            c.require(unused is None and all(parameter.dtype == torch.float32 for parameter in model.parameters()), 'one FP32 reference model required')
            eos = run._eos_ids(Path(grant['source_dir']), tokenizer)
            # Reuse the function verbatim. Only its historical384-row progress
            # text is suppressed; numerical computation and output are untouched.
            with redirect_stdout(io.StringIO()):
                predictions = generate(model, tokenizer, rows, eos, torch)
            print(f'Generated {len(predictions)}/{len(rows)} rows with the frozen decoder; no scoring', flush=True)
            del model, tokenizer; torch.cuda.empty_cache()
            c.phase_guard(grant); guard(phase='qualify')
            after = c.snapshot(grant, args.kind)
            c.require(before == after and c.sha(args.grant) == terminal['grant_sha256'] and
                      c.sha(args.build_gate) == terminal['build_gate_sha256'], 'auxiliary custody changed during generation')
            bundle = {'schema': 1, 'record': 'BINDING_INTEGRATION_AUX_GENERATIONS', 'kind': args.kind,
                'rows': len(predictions), 'predictions': predictions, 'eos_ids': sorted(eos), 'numerics': c.NUMERICS,
                'before': before, 'after': after, 'gold_file_read': False, 'grant_sha256': c.sha(args.grant),
                'build_gate_sha256': c.sha(args.build_gate), 'extra_warmup_or_qualification_calls': 0,
                'actual_budget_charges': {'rows': len(predictions),
                    'generated_tokens_including_eos': sum(p['generated_tokens'] for p in predictions),
                    'forward_calls': sum(p['forward_calls'] for p in predictions),
                    'processed_input_tokens': sum(p['processed_input_tokens'] for p in predictions)},
                'quality_claim': False, 'training_permission': False}
            c.write(out / 'GENERATIONS.json', bundle)
            terminal.update(status='COMPLETE', complete=True, failure=None, generations_sha256=c.sha(out / 'GENERATIONS.json'))
        return 0
    except BaseException as exc:
        terminal.update(status='FAILED', complete=False, failure={'type': type(exc).__name__, 'message': str(exc)})
        c.write(out / 'FAILURE.json', terminal)
        raise
    finally:
        terminal.update(before=before, after=after, wall_seconds=time.monotonic() - started)
        c.write(out / 'worker_terminal.json', terminal)


def controller(args):
    setup()
    grant, gate = c.context(args.grant, args.build_gate)
    c.require(Path(sys.executable).resolve() == c.RABPY.resolve(), 'pinned RAB interpreter required')
    if __package__:
        from . import run_probe
    else:
        import run_probe
    out = c.OUT / 'eval' / c.slot(args.kind, args.attempt)
    with c.controller_lease(args.grant) as owner:
        c.require(not out.exists() and not (Path(tempfile.gettempdir()) / 'rineya-gpu-0.lock').exists(), 'output or shared GPU lease already exists')
        rows = c.load_inputs(grant, args.kind); c.phase_guard(grant, 900)
        totals = c.costs(grant, args.grant)['totals']; c.global_costs()
        if args.attempt == 2:
            original = c.OUT / 'eval' / c.slot(args.kind, 1)
            prior, supervisor = c.read(original / 'AUX_BINDING.json'), c.read(original / 'supervisor.json')
            c.require(prior['status'] == 'FAILED' and prior['complete'] is False and prior['kind'] == args.kind and
                prior['grant_sha256'] == c.sha(args.grant) and prior['supervisor_sha256'] == c.sha(original / 'supervisor.json') and
                type(supervisor.get('returncode')) is int and supervisor['returncode'] != 0 and
                supervisor['status'] in ('EXITED', 'TIMEOUT_OR_INTERRUPTED') and not (original / 'GENERATIONS.json').exists() and
                totals['retries'] == 0, 'retry requires one verified original failure without completed generations')
        c.require(not (c.OUT / 'eval' / c.slot(args.kind, 2)).exists(), 'replacement of successful or earlier retry output forbidden')
        n = len(rows)
        reservation = {'kind': args.kind, 'attempt': args.attempt, 'grant_sha256': c.sha(args.grant),
            'rows': n, 'reserved_generated_tokens': n * 24, 'forward_calls': n * 24,
            'processed_input_tokens': sum(len(row['prompt_ids']) for row in rows) + n * 24,
            'controller_owner': owner, 'controller_lease_sha256': c.sha(c.OUT / 'eval/AUX_CONTROLLER_LEASE.json'),
            'parent_reservation_sha256': c.sha(c.OUT / 'posttrain/reservations' / (out.name + '.json'))}
        added = {'workers': 1, 'total_worker_seconds': 900, 'retries': args.attempt - 1,
                 **{k: reservation[k] for k in ('rows', 'reserved_generated_tokens', 'forward_calls', 'processed_input_tokens')}}
        c.require(all(totals[k] + value <= c.LIMITS[k] for k, value in added.items()), 'auxiliary subcap cannot fit this attempt')
        before = c.snapshot(grant, args.kind)
        training_before = run_probe.arm_snapshot(Path(grant['arm_path']), gate)
        c.write(reservation_path(args.kind, args.attempt), reservation)
        from experiments.model_quality_gpu_v1.supervisor import bounded_child
        command = [str(c.RABPY), str(Path(__file__).resolve()), '--grant', str(Path(args.grant).resolve()),
            '--build-gate', str(Path(args.build_gate).resolve()), '--kind', args.kind, '--attempt', str(args.attempt),
            '--out', str(out), '--confirm-model-execution', '--_worker']
        rc = bounded_child(command, out, 900)
        supervisor = c.read(out / 'supervisor.json'); success = False; reason = None; training_after = None; identities = {}
        try:
            c.require(rc == 0 and not (out / 'FAILURE.json').exists(), 'supervised auxiliary worker failed')
            terminal = c.read(out / 'worker_terminal.json'); c.original.verify_terminal(supervisor, terminal)
            c.require(terminal['generations_sha256'] == c.sha(out / 'GENERATIONS.json') and terminal['before'] == before,
                      'worker generation/custody identity differs')
            identities = c.verify_launch(out, args.grant, args.build_gate, args.kind, args.attempt, supervisor)
            training_after = run_probe.arm_snapshot(Path(grant['arm_path']), gate)
            c.require(training_before == training_after and before == c.snapshot(grant, args.kind), 'training/model provenance changed')
            c.phase_guard(grant); c.costs(grant, args.grant); c.global_costs(); success = True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reason = str(exc)
        receipt = {'schema': 1, 'status': 'COMPLETE' if success else 'FAILED', 'complete': success,
            'kind': args.kind, 'attempt': args.attempt, 'controller_returncode': 0 if success else 2, 'worker_returncode': rc,
            'grant_sha256': c.sha(args.grant), 'build_gate_sha256': c.sha(args.build_gate),
            'arm_receipt_sha256': c.sha(Path(grant['arm_path']) / 'ARM.json'), 'before': before,
            'after': terminal['after'] if success else None, 'training_before': training_before, 'training_after': training_after,
            'supervisor_sha256': c.sha(out / 'supervisor.json'),
            'worker_terminal_sha256': c.sha(out / 'worker_terminal.json') if (out / 'worker_terminal.json').exists() else None,
            'generations_sha256': c.sha(out / 'GENERATIONS.json') if (out / 'GENERATIONS.json').exists() else None,
            'reservation_sha256': c.sha(reservation_path(args.kind, args.attempt)), 'failure': reason, 'quality_claim': False, **identities}
    c.phase_guard(grant)
    c.write(out / 'AUX_BINDING.json', receipt)
    print({'status': receipt['status'], 'kind': args.kind, 'out': str(out)})
    return receipt['controller_returncode']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('grant', 'build-gate'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--kind', choices=tuple(c.KINDS), required=True)
    parser.add_argument('--attempt', type=int, choices=(1, 2), default=1)
    parser.add_argument('--confirm-model-execution', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--out', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(); c.require(args.confirm_model_execution, 'explicit bounded auxiliary dispatch required')
    if args._worker:
        c.require(args.out is not None, 'private worker output missing')
        return worker(args)
    c.require(args.out is None, 'public output override forbidden')
    return controller(args)


if __name__ == '__main__':
    raise SystemExit(main())
