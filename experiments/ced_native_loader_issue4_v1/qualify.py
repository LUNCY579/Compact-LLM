"""Fail-closed, offline, CPU-only qualification of the inherited real RAB-001 loader.

Only three explicitly named test modules run. No official checkpoint path, G0/G1,
optimizer, recovery, DEV, TEST, or deployment entry point is accepted. Outputs must
be new directories outside the checkout. A skip, omitted test, or dirty source is
not a PASS. This wrapper does not modify the inherited experiment or its receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE_SHA = 'e7285e5f68774bc2cc340a92cae7c7963bb569f2'
SUBJECT = 'experiments/architecture_benchmark_001'
PACKAGE = 'experiments/ced_native_loader_issue4_v1'
ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_SECONDS = 180
PINS = {'torch': '2.10.0', 'numpy': '2.3.5', 'safetensors': '0.7.0',
        'pytest': '9.0.2', 'transformers': '4.51.3'}
SCOPES = (f'{SUBJECT}/tests/test_native.py', f'{SUBJECT}/tests/test_native_loader.py',
          f'{PACKAGE}/test_native_issue4.py')
PERMISSIONS = dict.fromkeys(('official_weights_loaded', 'licenses_G0', 'licenses_G1',
    'licenses_optimizer_execution', 'licenses_recovery', 'licenses_DEV',
    'licenses_sealed_evaluation', 'licenses_efficacy_claim', 'licenses_gpu_fit_claim',
    'licenses_deployment', 'licenses_paid_execution'), False)

NATIVE_CASES = {
    'test_native_random_qwen3_logits[False]', 'test_native_random_qwen3_logits[True]',
    'test_native_rope_order_at_registered_length_and_long_absolute_positions[0]',
    'test_native_rope_order_at_registered_length_and_long_absolute_positions[10000]',
}
LOADER_CASES = {
    'test_load_materializes_required_buffers_not_only_parameters',
    'test_rebuilt_rotary_state_is_exactly_the_native_state',
    'test_loaded_weights_and_logits_match_ordinary_native_construction',
    'test_loaded_native_matches_the_dense_reference_unpadded_and_padded',
    'test_native_reference_parity_in_the_exact_g0_call_shape',
    'test_loaded_cached_continuation_matches_full_reforward',
    'test_load_ties_embeddings_and_never_writes_the_source',
    'test_reference_only_load_needs_no_buffer_reconstruction',
    'test_unknown_unmaterialized_buffer_fails_closed_instead_of_being_filled',
    'test_materialization_is_a_noop_when_nothing_is_left_on_meta',
    'test_materialization_reports_every_rebuilt_buffer',
    'test_load_upcasts_a_bfloat16_checkpoint_like_the_registered_source',
    'test_reference_and_native_rope_agree_across_the_registered_context',
    'test_require_materialized_rejects_meta_and_foreign_devices[meta]',
    'test_require_materialized_rejects_meta_and_foreign_devices[device]',
}
NEW_CASES = {
    'test_original_meta_failure_is_reproduced_through_actual_load',
    'test_save_reload_uses_actual_loader[float32]',
    'test_save_reload_uses_actual_loader[bfloat16]',
    'test_padded_batch_cache_and_chunk_parity[left-one]',
    'test_padded_batch_cache_and_chunk_parity[left-two]',
    'test_padded_batch_cache_and_chunk_parity[right-one]',
    'test_padded_batch_cache_and_chunk_parity[right-two]',
    'test_corrupt_checkpoint_fails_closed[missing]',
    'test_corrupt_checkpoint_fails_closed[extra]',
    'test_corrupt_checkpoint_fails_closed[shape]',
    'test_official_shape_validator_remains_closed_to_tiny_source',
    'test_wrong_transformers_version_blocks_actual_load',
    'test_loading_and_forward_do_not_mutate_assigned_source_tensors',
}
EXPECTED = {
    ('experiments.architecture_benchmark_001.tests.test_native', n) for n in NATIVE_CASES
} | {('experiments.architecture_benchmark_001.tests.test_native_loader', n) for n in LOADER_CASES
} | {('experiments.ced_native_loader_issue4_v1.test_native_issue4', n) for n in NEW_CASES}


class Blocked(RuntimeError):
    """An execution prerequisite is absent; never convert this into native evidence."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, body: dict) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(body, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def runtime() -> dict:
    versions = {}
    for name in (*PINS, 'tokenizers', 'huggingface-hub', 'packaging', 'regex', 'requests', 'tqdm'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    native_sources = {}
    if versions['transformers'] is not None:
        distribution = metadata.distribution('transformers')
        for relative in ('transformers/models/qwen3/modeling_qwen3.py',
                         'transformers/models/qwen3/configuration_qwen3.py',
                         'transformers/modeling_rope_utils.py'):
            path = Path(distribution.locate_file(relative))
            native_sources[relative] = sha256(path) if path.is_file() else None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions, 'native_sources_sha256': native_sources,
            'execution_policy': {'device': 'cpu', 'dtype': 'float32',
                                 'attention_backend': 'eager', 'threads': 1}}


def require_runtime(observed: dict) -> None:
    versions = observed['packages']
    mismatches = {name: {'expected': pin, 'observed': versions.get(name)}
                  for name, pin in PINS.items()
                  if (versions.get(name) or '').split('+')[0] != pin}
    if mismatches:
        raise Blocked('pinned native runtime unavailable: ' + json.dumps(mismatches, sort_keys=True))
    if versions['torch'] not in ('2.10.0', '2.10.0+cpu'):
        raise Blocked('this bounded receipt requires a CPU PyTorch build')


def inspect_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    for suite in root.iter('testsuite'):
        if any(int(suite.get(tag, '0')) != 0 for tag in ('skipped', 'failures', 'errors')):
            raise RuntimeError('JUnit suite reports skips, failures, or errors')
        if int(suite.get('tests', '-1')) != len(suite.findall('.//testcase')):
            raise RuntimeError('JUnit suite count is inconsistent')
    cases = root.findall('.//testcase')
    seen = set()
    for case in cases:
        key = (case.get('classname', ''), case.get('name', ''))
        if key in seen:
            raise RuntimeError(f'duplicate test case: {key}')
        seen.add(key)
        if any(case.find(tag) is not None for tag in ('skipped', 'failure', 'error')):
            raise RuntimeError(f'nonpassing or skipped native test: {key}')
    if seen != EXPECTED:
        raise RuntimeError(f'exact native test set not executed: missing={sorted(EXPECTED-seen)}, '
                           f'unexpected={sorted(seen-EXPECTED)}')
    return {'passed': len(cases), 'failures': 0, 'errors': 0, 'skipped': 0,
            'case_ids': [f'{c}::{n}' for c, n in sorted(seen)]}


def git(*args: str) -> str:
    p = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True,
                       timeout=30, check=False)
    if p.returncode:
        raise Blocked(f'git {args[0]} failed: {p.stderr.strip()}')
    return p.stdout.strip()


def source_identity(expected_head: str) -> dict:
    if not re.fullmatch(r'[0-9a-f]{40}', expected_head):
        raise Blocked('expected-head must be a full lowercase commit SHA')
    head = git('rev-parse', 'HEAD')
    if head != expected_head:
        raise Blocked('checkout does not match the explicitly selected commit')
    git('merge-base', '--is-ancestor', BASE_SHA, head)
    if git('status', '--porcelain', '--untracked-files=all'):
        raise Blocked('checkout is dirty; refuse stale-source qualification')
    if git('diff', '--name-only', BASE_SHA, head, '--', SUBJECT):
        raise Blocked('inherited RAB-001 subject changed; separate re-review required')
    paths = git('ls-files', '-z', '--', SUBJECT, PACKAGE).split('\x00')
    manifest = {}
    for name in paths:
        if not name:
            continue
        p = ROOT / name
        if p.is_symlink() or not p.is_file():
            raise Blocked(f'tracked source is unavailable or symlinked: {name}')
        manifest[name] = sha256(p)
    if not manifest or any(scope not in manifest for scope in SCOPES):
        raise Blocked('required test source missing from checkout')
    protocol = json.loads((ROOT / SUBJECT / 'protocol.json').read_text(encoding='utf-8'))
    if (protocol.get('G3_G5', {}).get('execution_enabled') is not False
            or protocol.get('native_transformers_version') != PINS['transformers']):
        raise Blocked('protocol boundary or native pin drifted')
    return {'head': head, 'tree': git('rev-parse', 'HEAD^{tree}'), 'base': BASE_SHA,
            'subject_tree': git('rev-parse', f'HEAD:{SUBJECT}'), 'files_sha256': manifest}


def qualify(expected_head: str, out: Path) -> int:
    out = out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise Blocked('output must be outside the checkout')
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / 'STARTED.json', {'started_at': now(), 'expected_head': expected_head,
                                     'permissions': PERMISSIONS, 'scopes': SCOPES})
    receipt = {'schema': 1, 'experiment_id': 'RAB-001-ISSUE4-NATIVE-V2',
               'evidence_kind': 'native_random_fixture_loader_mechanics',
               'expected_head': expected_head, 'permissions': PERMISSIONS,
               'status': 'BLOCKED', 'runtime': runtime(), 'scopes': SCOPES,
               'tests_started': False, 'timeout_seconds': TIMEOUT_SECONDS,
               'official_source_audit': 'NOT_REQUESTED', 'started_at': now()}
    start = time.monotonic()
    try:
        require_runtime(receipt['runtime'])
        before = source_identity(expected_head)
        receipt['source'] = before
        env = os.environ.copy()
        env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                    'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                    'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'TOKENIZERS_PARALLELISM': 'false', 'PYTHONHASHSEED': '17',
                    'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PYTEST_ADDOPTS': '',
                    'PYTEST_PLUGINS': '', 'PYTHONPATH': str(ROOT)})
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                   '--strict-config', '--strict-markers', '--noconftest',
                   f'--rootdir={ROOT}', '-o', 'addopts=', '-o', 'junit_family=xunit2', *SCOPES,
                   f'--junitxml={out / "native-junit.xml"}',
                   f'--basetemp={out / "fixtures"}']
        receipt['command'] = command
        receipt['tests_started'] = True
        with (out / 'native-tests.log').open('x', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS,
                                       check=False)
        receipt['pytest_returncode'] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(f'native test process returned {completed.returncode}')
        receipt['test_results'] = inspect_junit(out / 'native-junit.xml')
        if source_identity(expected_head) != before or runtime() != receipt['runtime']:
            raise RuntimeError('source or runtime changed during native execution')
        receipt['source_unchanged'] = True
        receipt['status'] = 'PASS'
    except (Exception, KeyboardInterrupt) as error:
        receipt['status'] = 'BLOCKED' if isinstance(error, Blocked) else 'FAIL'
        receipt['reason'] = f'{type(error).__name__}: {error}'
    receipt['wall_seconds'] = time.monotonic() - start
    receipt['finished_at'] = now()
    receipt['artifacts_sha256'] = {p.name: sha256(p) for p in out.iterdir()
                                  if p.is_file() and p.name != 'receipt.json'}
    write_once(out / 'receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'receipt': str(out / 'receipt.json'),
                      'reason': receipt.get('reason')}, indent=2))
    return 0 if receipt['status'] == 'PASS' else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-head')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.preflight_only:
        if args.expected_head or args.out:
            parser.error('preflight-only cannot be combined with execution arguments')
        observed = runtime()
        try:
            require_runtime(observed)
        except Blocked as error:
            print(json.dumps({'status': 'BLOCKED', 'runtime': observed,
                              'reason': str(error), 'native_execution': False}, indent=2))
            return 2
        print(json.dumps({'status': 'RUNTIME_PRESENT_NOT_QUALIFIED', 'runtime': observed,
                          'native_execution': False}, indent=2))
        return 0
    if not args.expected_head or args.out is None:
        parser.error('execution requires --expected-head and --out')
    try:
        return qualify(args.expected_head, args.out)
    except (OSError, Blocked) as error:
        print(f'BLOCKED: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
