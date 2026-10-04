"""One fail-fast run of the two approved CPU modules; never native qualification."""
import contextlib
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path.cwd()
PACKAGE = ROOT / 'experiments/rineya_centralized_reasoning_system_v1'
OUTPUT = Path(__file__).resolve().parent
PSUTIL_SITE = Path('C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Lib/site-packages')
BASE_PYTHON = Path('C:/Users/clclc/AppData/Roaming/uv/python/cpython-3.11.15-windows-x86_64-none/python.exe')
BASE_SHA = '28d2965d061656abbe3a46100bd3342d1152185289445d489782a21a8cfe9b57'
FORBIDDEN = {'torch', 'transformers', 'tokenizers', 'safetensors'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


assert not (OUTPUT / 'TEST_RESULT.json').exists(), 'single integration run already recorded'
assert Path(sys.executable).resolve() == BASE_PYTHON.resolve() and sha(BASE_PYTHON) == BASE_SHA
freeze = json.loads((OUTPUT / 'SOURCE_FREEZE.json').read_bytes())
for name, pin in {**freeze['source_sha256'], **freeze['preserved_pins']}.items():
    assert sha(PACKAGE / name) == pin, name
framework_attempts = []


class BlockFrameworkImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            framework_attempts.append(fullname)
            raise RuntimeError('CPU-only integration forbids framework import: ' + fullname)


sys.meta_path.insert(0, BlockFrameworkImports())
sys.path.insert(0, str(ROOT))
sys.path.append(str(PSUTIL_SITE))  # Only psutil is imported; framework imports fail above.
import psutil
assert psutil.__version__ == '7.2.2' and Path(psutil.__file__).resolve().is_relative_to(PSUTIL_SITE.resolve())
started = time.perf_counter()
records = []
modules = [('probe', 'experiments.rineya_centralized_reasoning_system_v1.tests.test_owned_runtime_identity_probe'),
           ('reader', 'experiments.rineya_centralized_reasoning_system_v1.tests.test_reader')]
for name, module in modules:
    transcript = io.StringIO()
    stage_started = time.perf_counter()
    with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
        suite = unittest.defaultTestLoader.loadTestsFromName(module)
        result = unittest.TextTestRunner(stream=transcript, verbosity=2, failfast=True).run(suite)
    seconds = time.perf_counter() - stage_started
    path = OUTPUT / (name.upper() + '_CPU_STDOUT.txt')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(transcript.getvalue())
    records.append({'module': module, 'tests': result.testsRun, 'passed': result.wasSuccessful(),
                    'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
                    'seconds': seconds, 'stdout_ref': {'path': str(path), 'sha256': sha(path)},
                    'fail_fast': True})
    print(transcript.getvalue(), end='')
    if not result.wasSuccessful():
        break
seconds = time.perf_counter() - started
pins_unchanged = all(sha(PACKAGE / name) == pin for name, pin in
                     {**freeze['source_sha256'], **freeze['preserved_pins']}.items())
passed = len(records) == len(modules) and all(record['passed'] for record in records)
assert not (FORBIDDEN & set(sys.modules)), 'forbidden framework imported'
payload = {
    'status': 'CPU_INTEGRATION_PASSED_PENDING_INDEPENDENT_SOURCE_REVIEW' if passed else 'CPU_INTEGRATION_FAILED_FROZEN_NO_REPAIR_OR_RERUN',
    'passed': passed, 'tests': sum(record['tests'] for record in records), 'seconds': seconds, 'modules': records,
    'source_freeze_sha256': sha(OUTPUT / 'SOURCE_FREEZE.json'), 'source_pins_unchanged': pins_unchanged,
    'builder': '/root/exception_builder', 'independent_approval': False,
    'cpu_environment': {'executable': str(BASE_PYTHON), 'executable_sha256': BASE_SHA,
                        'python': sys.version, 'psutil': psutil.__version__, 'psutil_site': str(PSUTIL_SITE),
                        'psutil_module_path': str(Path(psutil.__file__).resolve()), 'psutil_module_sha256': sha(psutil.__file__),
                        'registered_native_environment': False,
                        'registered_native_interpreter_exists': Path('C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe').exists(),
                        'native_environment_gates_retargeted_or_lowered': False,
                        'imported_third_party_package_scope': 'psutil7.2.2 only; frameworks blocked by fatal import refusal'},
    'exact_job_cpu_fixture_adaptation': 'Explicit base interpreter plus one owned stdlib sleeping child; preserves >=2-process assertion, absolute deadline and empty-tree exit; original venv fixture/outcome retained in V1/history/base',
    'framework_import_attempts': framework_attempts, 'framework_imports': 0, 'model_calls': 0,
    'actual_probe_runs': 0, 'native_reader_runs': 0, 'native_qualification': False,
    'canonical_host_database_access': False, 'CPU_temporary_SQLite_only': True,
    'stop_at_first_material_failure': True, 'automatic_retry': False,
}
new_json(OUTPUT / 'TEST_RESULT.json', payload)
new_json(OUTPUT / 'FREEZE.json', {
    'status': payload['status'], 'source_freeze_ref': {'path': str(OUTPUT / 'SOURCE_FREEZE.json'), 'sha256': payload['source_freeze_sha256']},
    'test_result_ref': {'path': str(OUTPUT / 'TEST_RESULT.json'), 'sha256': sha(OUTPUT / 'TEST_RESULT.json')},
    'source_sha256': freeze['source_sha256'], 'preserved_sha256': freeze['preserved_pins'],
    'evidence_source_sha256': {'REGISTRATION_BUILD.py': sha(OUTPUT / 'REGISTRATION_BUILD.py'),
                              'RUN_CPU_TESTS_ONCE.py': sha(__file__)},
    'builder': '/root/exception_builder', 'builder_self_approval': False, 'source_edits_frozen': True,
    'new_probe_key': 'CRS-OWNED-IDENTITY-002', 'new_probe_runs': 0, 'committed': False,
})
print(json.dumps({'passed': passed, 'tests': payload['tests'], 'seconds': seconds,
                  'test_result_sha256': sha(OUTPUT / 'TEST_RESULT.json'), 'freeze_sha256': sha(OUTPUT / 'FREEZE.json')}))
raise SystemExit(0 if passed and pins_unchanged and not framework_attempts else 1)
