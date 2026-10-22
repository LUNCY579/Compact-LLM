"""Preserve exact registered preflight/qualification attempts and fresh guards."""
import json
from pathlib import Path
import subprocess
import sys
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sbhm.protocol import environment, now, write_json


def main():
    evidence = Path(__file__).resolve().parent
    config = ROOT / 'configs/titan_transition.v2.json'
    output = ROOT / 'runs/titan_transition_v2_20261003_admission/A-qualification'
    record = {'created_utc': now(), 'environment': environment(),
              'host_available_ram_bytes_before_attempts': psutil.virtual_memory().available,
              'commands': [], 'qualification_directory_created_before': output.exists()}
    stages = [('preflight', []),
              ('qualify', ['--arm', 'A', '--output', str(output), '--allow-model-load'])]
    for stage, additional in stages:
        command = [sys.executable, '-m', 'sbhm', stage, '--config', str(config), *additional]
        result = subprocess.run(command, cwd=ROOT, text=True, encoding='utf-8', capture_output=True)
        with (evidence / f'native-v2-{stage}.txt').open('x', encoding='utf-8') as stream:
            stream.write(result.stdout)
            if result.stderr:
                stream.write('\nSTDERR\n'+result.stderr)
        record['commands'].append({'argv': command, 'exit_code': result.returncode,
                                   'reported': json.loads(result.stdout)})
    plan_output = ROOT / 'runs/titan_transition_v2_20261003_matrix'
    plan = subprocess.run([sys.executable, '-m', 'sbhm.matrix', '--config', str(config),
                           '--output', str(plan_output)], cwd=ROOT, text=True,
                          encoding='utf-8', capture_output=True)
    plan_data = json.loads(plan.stdout)
    write_json(evidence / 'matrix-v2-plan.json', plan_data)
    record.update(host_available_ram_bytes_after_attempts=psutil.virtual_memory().available,
                  qualification_directory_created_after=output.exists(),
                  qualification_receipt_present=(output / 'receipt.json').exists(),
                  matrix_plan_commands=len(plan_data['commands']),
                  matrix_plan_directory_created=plan_output.exists(),
                  native_training_executed=False, behavioral_question_answered=False)
    write_json(evidence / 'native-v2-admission.json', record)
    print(json.dumps({key: record[key] for key in ('host_available_ram_bytes_after_attempts',
                      'qualification_directory_created_after', 'qualification_receipt_present',
                      'matrix_plan_commands', 'matrix_plan_directory_created',
                      'native_training_executed', 'behavioral_question_answered')}))
    for command in record['commands']:
        print(json.dumps({'stage': command['argv'][3], 'exit_code': command['exit_code'],
                          'reported': command['reported']}))


if __name__ == '__main__':
    main()
