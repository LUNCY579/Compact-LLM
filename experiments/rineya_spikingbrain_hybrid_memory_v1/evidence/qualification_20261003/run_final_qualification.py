"""Final offline suite, blocked preflight, and nonexecuting four-arm plan."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import psutil
import torch

experiment = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(experiment))
from sbhm.protocol import code_hash

output = Path(__file__).resolve().parent
config = experiment/'configs/titan_transition.proposed.json'
plan_root = output/'PROPOSED_MATRIX_RUN_NOT_CREATED'
assert not plan_root.exists()
assert torch.version.cuda is None
before_code = code_hash()
before_tests = {str(path.relative_to(experiment)): sha256(path.read_bytes()).hexdigest()
                for path in sorted((experiment/'tests').glob('*.py'))}
started = datetime.now(timezone.utc).isoformat()


def run(arguments, filename):
    command = [sys.executable, *arguments]
    result = subprocess.run(command, cwd=experiment, env=os.environ.copy(),
                            text=True, encoding='utf-8', stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    with (output/filename).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(result.stdout)
    print(f'{filename}: exit {result.returncode}\n{result.stdout[:3000]}', flush=True)
    return command, result


commands = []
command, tests = run(['-m', 'pytest', '-q', 'tests', '--disable-warnings'], 'pytest-final.txt')
commands.append({'command': command, 'exit_code': tests.returncode})
assert tests.returncode == 0 and '98 passed' in tests.stdout and 'skipped' not in tests.stdout
command, compile_result = run(['-m', 'compileall', '-q', 'sbhm', 'tests', 'scripts'], 'compileall-final.txt')
commands.append({'command': command, 'exit_code': compile_result.returncode})
assert compile_result.returncode == 0
command, preflight = run(['-m', 'sbhm', 'preflight', '--config', str(config)], 'preflight-final.txt')
commands.append({'command': command, 'exit_code': preflight.returncode})
preflight_json = json.loads(preflight.stdout)
assert preflight.returncode == 2
assert preflight_json['status'] == 'BLOCKED' and preflight_json['model_loaded'] is False
with (output/'preflight-final.json').open('x', encoding='utf-8') as stream:
    json.dump(preflight_json, stream, indent=2)
    stream.write('\n')
command, matrix = run(['-m', 'sbhm.matrix', '--config', str(config), '--output', str(plan_root)], 'matrix-plan-final.txt')
commands.append({'command': command, 'exit_code': matrix.returncode})
matrix_json = json.loads(matrix.stdout)
assert matrix.returncode == 0 and matrix_json['status'] == 'PLAN_ONLY'
assert len(matrix_json['commands']) == 22 and not plan_root.exists()
assert code_hash() == before_code
assert before_tests == {str(path.relative_to(experiment)): sha256(path.read_bytes()).hexdigest()
                        for path in sorted((experiment/'tests').glob('*.py'))}
receipt = {
    'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS_OFFLINE_PINNED_MECHANICS_NATIVE_BLOCKED',
    'python_executable': sys.executable, 'python_version': platform.python_version(),
    'platform': platform.platform(),
    'versions': {name: version(name) for name in ('torch', 'transformers', 'tokenizers',
                  'safetensors', 'psutil', 'pytest', 'accelerate')},
    'compiled_cuda': torch.version.cuda,
    'host_available_ram_bytes_at_receipt': psutil.virtual_memory().available,
    'worker_rss_bytes_at_receipt': psutil.Process().memory_info().rss,
    'code_sha256_before_and_after_checks': before_code, 'tests_sha256': before_tests,
    'configuration_file_sha256': sha256(config.read_bytes()).hexdigest(),
    'commands': commands, 'full_suite_passed': 98, 'full_suite_skipped': 0,
    'official_library_random_cpu_integration_executed': True,
    'preflight_exit_code': preflight.returncode, 'preflight_blockers': preflight_json['blockers'],
    'preflight_source_sha256': preflight_json['source_sha256'],
    'plan_commands': len(matrix_json['commands']), 'plan_root_created': plan_root.exists(),
    'model_loaded': False, 'pretrained_inference': False, 'gpu_execution': False,
    'native_training': False, 'native_training_fit_established': False,
    'corpus_reviewed': False, 'operator_grant_created': False,
}
with (output/'environment-final.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps({'status': receipt['status'], 'passed': 98, 'skipped': 0,
                  'native_preflight': preflight_json['blockers'],
                  'plan_commands': len(matrix_json['commands']),
                  'plan_root_created': plan_root.exists(), 'code_sha256': before_code}, indent=2))
