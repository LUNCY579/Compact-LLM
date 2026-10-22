"""Run the three fixed diagnostic models once; never retry automatically."""
import argparse
import os
import subprocess
import time

from paths import ROOT, HERE, PILOT, PYTHON
import common as c
from prepare_runtime import verify_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-model-execution', action='store_true')
    args = parser.parse_args()
    c.require(args.confirm_model_execution, 'explicit bounded inference dispatch required')
    verify_frozen()
    plan, grant = c.read(ROOT / 'PLAN.json'), c.read(ROOT / 'GRANT.json')
    c.validate_plan(plan); c.validate_grant(ROOT / 'PLAN.json', grant)
    c.phase_guard(plan, grant, seconds_needed=2700)
    logs = ROOT / 'battery'
    logs.mkdir(exist_ok=False)
    c.write_new(logs / 'DISPATCH_RESERVATION.json', {'plan_sha256': c.sha(ROOT / 'PLAN.json'),
        'grant_sha256': c.sha(ROOT / 'GRANT.json'), 'owner_pid': os.getpid(),
        'models': ['source', 'lora', 'mixed'], 'automatic_retries': 0})
    attempts = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    try:
        for index, role in enumerate(('source', 'lora', 'mixed')):
            verify_frozen()
            c.phase_guard(plan, grant, seconds_needed=(3 - index) * 900)
            print('START ' + role, flush=True)
            command = [str(PYTHON), str(HERE / 'probe.py'), '--plan', str(ROOT / 'PLAN.json'),
                       '--grant', str(ROOT / 'GRANT.json'), '--model', role, '--confirm-model-execution']
            started = time.monotonic()
            with (logs / (role + '.log')).open('xb') as stream:
                rc = subprocess.run(command, cwd=PILOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            record = {'model_role': role, 'attempt': 1, 'returncode': rc,
                      'controller_wall_seconds': time.monotonic() - started}
            c.write_new(logs / (role + '.json'), record)
            attempts.append(record)
            c.require(rc == 0, 'diagnostic worker failed; preserve it and inspect before an explicit retry')
            binding = c.read(ROOT / 'eval' / role / 'DIAGNOSTIC_BINDING.json')
            c.require(binding['status'] == 'COMPLETE' and binding['complete'] is True and
                      binding['controller_returncode'] == 0, 'successful diagnostic custody absent')
            print('VERIFIED ' + role, flush=True)
        verify_frozen()
        c.phase_guard(plan, grant)
        c.write_new(logs / 'TERMINAL.json', {'status': 'COMPLETE_THREE_MODELS', 'attempts': attempts,
            'final_binding_sha256': {role: c.sha(ROOT / 'eval' / role / 'DIAGNOSTIC_BINDING.json') for role in c.MODEL_SHAS},
            'cost_accounting': c.attempt_costs(plan, ROOT / 'GRANT.json'),
            'session_elapsed_seconds': time.time() - grant['session_start_unix'], 'quality_claim': False})
        return 0
    except Exception as exc:
        c.write_new(logs / 'TERMINAL.json', {'status': 'FAILED_OR_BLOCKED', 'attempts': attempts,
            'failure': {'type': type(exc).__name__, 'message': str(exc)},
            'cost_accounting': c.attempt_costs(plan, ROOT / 'GRANT.json'), 'quality_claim': False})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
