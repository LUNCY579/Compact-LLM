"""Restore only registered dependency versions; never import frameworks."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / 'PLAN.json').read_text(encoding='utf-8'))
target = Path(plan['target']) / 'Scripts/python.exe'
assert hashlib.sha256(target.read_bytes()).hexdigest() == plan['stdlib_venv_launcher_sha256']
with (HERE / 'ATTEMPT.json').open('x', encoding='utf-8') as stream:
    json.dump({'status': 'DEPENDENCY_INSTALLATION_ONLY', 'framework_imports': 0, 'started_unix': time.time()}, stream)
command = ['uv', 'pip', 'install', '--python', str(target),
           'torch==2.10.0+cu126', 'transformers==4.57.1', 'safetensors==0.7.0',
           'tokenizers==0.22.2', 'psutil==7.2.2',
           '--index', 'https://download.pytorch.org/whl/cu126',
           '--index', 'https://pypi.org/simple', '--only-binary', ':all:']
start = time.perf_counter()
with (HERE / 'INSTALL_STDOUT.txt').open('x', encoding='utf-8') as stdout, (HERE / 'INSTALL_STDERR.txt').open('x', encoding='utf-8') as stderr:
    try:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=600, check=False)
        exit_code, failure = result.returncode, None
    except subprocess.TimeoutExpired:
        exit_code, failure = None, 'INSTALLATION_DEADLINE_EXCEEDED'
record = {'command': command, 'exit_code': exit_code, 'failure': failure,
          'wall_seconds': time.perf_counter() - start, 'framework_imports': 0,
          'model_loads': 0, 'probe_attempts': 0, 'native_budget_reset': False,
          'qualification': 'DEPENDENCY_SETUP_ONLY; exact old runtime/file pins must still pass'}
with (HERE / 'INSTALL_RESULT.json').open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
print(json.dumps(record, sort_keys=True))
raise SystemExit(0 if exit_code == 0 else 1)
