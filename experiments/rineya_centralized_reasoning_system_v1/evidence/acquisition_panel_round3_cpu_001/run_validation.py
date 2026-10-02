"""Create-once coordinator validation; requires the prospectively saved PLAN."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent
sys.path.insert(0, str(PACKAGE.parent.parent))

def write(name, value):
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if len(encoded.encode('utf-8')) > 8 * 1024 * 1024:
        raise ValueError('8 MiB artifact ceiling: ' + name)
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

plan = json.loads((HERE / 'PLAN.json').read_text(encoding='utf-8'))
if sha(Path(__file__)) != plan['wrapper_sha256']:
    raise SystemExit('Frozen validation wrapper mismatch')
for relative, wanted in plan['source_sha256'].items():
    if sha(PACKAGE / relative) != wanted:
        raise SystemExit('Frozen source mismatch: ' + relative)
write('ATTEMPT.json', {'validation_id': plan['validation_id'], 'retry_allowed': False,
                       'source_sha256': plan['source_sha256'], 'started_unix': time.time()})

from experiments.rineya_centralized_reasoning_system_v1 import acquisition_data as panel
from experiments.rineya_centralized_reasoning_system_v1 import core
from experiments.rineya_centralized_reasoning_system_v1.tests import test_acquisition_data as tests

counts = {'cpu_receipt_calls': 0, 'core_calls': 0, 'receipt_rename_controls': 0}
original_receipt, original_core, original_rename = panel.cpu_receipt, core.run_case, panel.rename
captured = {}
active_receipt = False
core_outputs = []

def limited_core(*args, **kwargs):
    if counts['core_calls'] >= 32:
        raise RuntimeError('Prospective 32 core-call ceiling reached')
    counts['core_calls'] += 1
    case = args[0] if args else kwargs.get('case')
    row = {'call': counts['core_calls'], 'case_id': case.get('id') if isinstance(case, dict) else None,
           'input_sha256': panel.digest(case)}
    try:
        value = original_core(*args, **kwargs)
    except Exception as error:
        row['exception'] = {'type': type(error).__name__, 'message': str(error)}
        core_outputs.append(row)
        raise
    row['prediction'] = value
    core_outputs.append(row)
    return value

def retained_receipt(bundle):
    global active_receipt
    if counts['cpu_receipt_calls']:
        raise RuntimeError('Prospective one receipt-batch ceiling reached')
    counts['cpu_receipt_calls'] += 1
    write('PRIVATE_BUNDLE.json', bundle)
    captured['bundle'] = bundle
    active_receipt = True
    try:
        result = original_receipt(bundle)
    finally:
        active_receipt = False
    write('CPU_RECEIPT.json', result)
    captured['receipt'] = result
    return result

def counted_rename(*args, **kwargs):
    if active_receipt:
        if counts['receipt_rename_controls'] >= 32:
            raise RuntimeError('Prospective 32 receipt rename-control ceiling reached')
        counts['receipt_rename_controls'] += 1
    return original_rename(*args, **kwargs)

output = io.StringIO()
start = time.perf_counter()
with patch.object(panel, 'cpu_receipt', retained_receipt), patch.object(core, 'run_case', limited_core), patch.object(panel, 'rename', counted_rename):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests.AcquisitionTests)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
elapsed = time.perf_counter() - start
write('PRIVATE_CORE_OUTPUTS.json', core_outputs)
with (HERE / 'ROOT_TEST_OUTPUT.txt').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(output.getvalue())

receipt = captured.get('receipt', {})
bundle = captured.get('bundle', {})
costs = [row['prediction']['cost'] for row in core_outputs if 'prediction' in row]
refs = {}
for name in ('PLAN.json', 'ATTEMPT.json', 'PRIVATE_BUNDLE.json', 'PRIVATE_CORE_OUTPUTS.json', 'CPU_RECEIPT.json', 'ROOT_TEST_OUTPUT.txt', 'run_validation.py'):
    path = HERE / name
    if path.exists():
        refs[name] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
report = {
    'validation_id': plan['validation_id'], 'status': 'SOURCE_VALIDATION_EXECUTED_REVIEW_PENDING',
    'qualification_claimed': False, 'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
    'errors': len(result.errors), 'failures': len(result.failures), 'skips': len(result.skipped),
    'wall_seconds': elapsed, 'observed_call_counts': counts, 'source_sha256': plan['source_sha256'],
    'core_cost_totals': {key: sum(cost[key] for cost in costs) for key in ('operations', 'source_bytes_scanned', 'source_bytes_verification_scanned', 'extracted_events', 'wall_seconds', 'model_calls')},
    'core_peak_serialized_state_bytes': max((cost['peak_serialized_state_bytes'] for cost in costs), default=None),
    'cost_scope': 'Core row wall times are nested within validation wall time; do not add. Constructor/review costs outside this named invocation remain separate/unavailable.',
    'source_unchanged_after': all(sha(PACKAGE / relative) == wanted for relative, wanted in plan['source_sha256'].items()),
    'main_comparisons': receipt.get('main_comparisons'), 'main_passed': receipt.get('main_passed'),
    'rename_checks': receipt.get('rename_checks'), 'rename_passed': receipt.get('rename_passed'),
    'new_bank_hashes': {key: bundle.get('manifest', {}).get(key + '_sha256') for key in ('public', 'gold', 'latent', 'lineage', 'assessments')},
    'source_candidate_roots': receipt.get('source_candidate_roots'), 'unqualified_roots': receipt.get('unqualified_roots'),
    'root_raw_refs': refs, 'prior_raw_evidence_unavailable_not_reconstructed': True,
    'cumulative_core_calls_including_prior_diagnostic': plan['prior_known_core_including_diagnostic'] + counts['core_calls'],
    'cumulative_main': plan['prior_known_main'] + counts['core_calls'],
    'cumulative_rename_controls': plan['prior_known_rename'] + counts['receipt_rename_controls'],
    'cumulative_combined_comparison_control_diagnostic_invocations': plan['prior_known_combined'] + counts['core_calls'] + counts['receipt_rename_controls'],
    'original_H01_LA_I1_002_closed': False, 'model_tokenizer_framework_or_native_calls': 0,
    'native_budget_or_owner_limit_changed': False,
}
write('REPORT.json', report)
print(json.dumps({key: report[key] for key in ('validation_id', 'passed', 'tests_run', 'wall_seconds', 'observed_call_counts', 'main_passed', 'rename_passed', 'source_unchanged_after')}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() and report['source_unchanged_after'] else 1)
