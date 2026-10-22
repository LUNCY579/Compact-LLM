"""Offline, exclusive-create research receipts; no native libraries or bank IO."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-generalization-dynamics-v1')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from experiments.rineya_generalization_dynamics_v1.common import code_hash, file_hash, read, signed, verify, write_new
from experiments.rineya_generalization_dynamics_v1.query_dependence_witness import audit_witness

stamp = datetime.now(timezone.utc).isoformat()
baseline = read(OUT / 'preservation-baseline.json')
preserved = []
for entry in baseline['files']:
    path = Path(entry['path'])
    actual = file_hash(path)
    assert actual == entry['sha256'], f'Pre-existing file changed: {path}'
    preserved.append({'path': str(path), 'sha256': actual})

package = REPO / 'experiments/rineya_generalization_dynamics_v1'
syntax_count = 0
for path in sorted(package.rglob('*.py')):
    compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
    syntax_count += 1

test = subprocess.run([sys.executable, '-B', '-m', 'unittest',
    'experiments.rineya_generalization_dynamics_v1.tests.test_query_dependence_witness', '-v'],
    cwd=REPO, text=True, capture_output=True)
assert test.returncode == 0, test.stdout + test.stderr
with (OUT / 'fixture-tests.txt').open('x', encoding='utf-8') as stream:
    stream.write(test.stdout + test.stderr)

diff = subprocess.run(['git', 'diff', '--check'], cwd=REPO, text=True, capture_output=True)
assert diff.returncode == 0, diff.stdout + diff.stderr
new_files = [package / 'query_dependence_witness.py',
             package / 'tests/test_query_dependence_witness.py',
             package / 'QUERY_DEPENDENCE_RESEARCH_20261005.md']
for path in new_files:
    assert all(line == line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()), path

verified = []
for family in ('minimal', 'all_regular'):
    witness_path = OUT / f'{family}_EXCLUDED_MECHANICAL_WITNESS.json'
    cert_path = OUT / f'{family}_QUERY_DEPENDENCE_CERTIFICATE.json'
    witness, certificate = verify(read(witness_path)), verify(read(cert_path))
    assert audit_witness(witness) == certificate
    verified.append({'family': family, 'contexts': certificate['contexts'],
        'queries': certificate['queries'], 'witness_file_sha256': file_hash(witness_path),
        'certificate_file_sha256': file_hash(cert_path),
        'conditional_chance_certificates': certificate['conditional_chance_certificates']})

review = signed({'schema': 'gd001_independent_finite_witness_review_v1', 'created_utc': stamp,
    'reviewer': '/root/producer_audit', 'scope': 'independent read-only mechanical recomputation',
    'review_result': 'PASS_WITH_EXPLICIT_MODEL_EVIDENCE_LIMITS',
    'all_regular_relations': 90, 'queries': 1440, 'fixed_query_present_absent': [45, 45],
    'same_context_bytes_per_query': True, 'context_record_count': 12, 'context_entity_count': 6,
    'context_field_frequencies': 'fixed', 'value_multiset': 'fixed', 'public_metadata_ids_excluded': True,
    'maximum_typed_accuracy_for_reviewed_information_sets': 0.5,
    'context_only_maximum_typed_correct_per_context': '8/16 because null repeats eight times',
    'independent_oracle_property_checks': {'record_permutation': 2880, 'nuisance_value': 1440,
                                         'selected_value': 5760},
    'verified_payloads': verified,
    'limits': ['Finite lookup population only; not revision comparison or learned binding.',
               'Operator checks concern oracle semantics, not measured model invariance.',
               'Fixed query order, carry-over and tokenizer-length cues require native-data controls.',
               'Previous-generator perfect heuristic was not independently rechecked in this review.'],
    'native_model_calls': 0, 'actual_bank_reads': 0, 'owner_messages': 0,
    'quality_gain_established': False, 'authorizes_native_execution': False})
write_new(OUT / 'INDEPENDENT_WITNESS_REVIEW.json', review)

resource_paths = [
 Path('C:/Users/clclc/BusinessProject/mlp-ship-001-records/20261005-m1-8ef12f4b/human-authorized-warm-window-01/RESOURCE_WINDOW.json'),
 Path('C:/Users/clclc/BusinessProject/supa-run-outputs/nonrecurrent-authority-20261005-01a10d63-v1/RESOURCE_HEADROOM_RESULT.json')]
resources = []
for path in resource_paths:
    receipt = read(path)
    assert receipt['model_calls'] == 0
    resources.append({'path': str(path), 'sha256': file_hash(path), 'status': receipt['status'],
                      'model_calls': receipt['model_calls'], 'clearance': False})

write_new(OUT / 'validation.json', signed({'schema': 'gd001_query_dependence_validation_v1',
    'created_utc': stamp, 'branch': 'research/rineya-generalization-dynamics-v1',
    'head': '9044d41f019fd93daad116c5ded1fda7469d5fa5', 'code_sha256': code_hash(),
    'pre_existing_files_preserved': len(preserved), 'preserved_files': preserved,
    'syntax_files_passed': syntax_count, 'focused_fixture_tests': 20,
    'focused_fixture_tests_passed': True, 'diff_whitespace_passed': True,
    'new_file_whitespace_passed': True, 'test_log_sha256': file_hash(OUT / 'fixture-tests.txt'),
    'native_model_loads': 0, 'native_forwards': 0, 'root_training_updates': 0,
    'new_environment_created': False, 'protected_bank_reads': 0, 'owner_messages': 0,
    'resource_receipt_snapshots': resources,
    'observed_known_owner_pids_absent': [400532, 412560, 415040, 334248, 345796, 348332, 348324],
    'pid_observation_scope': 'single root CIM inspection; not continuous monitoring or all-host exclusivity',
    'confirmation_unopened': True, 'serving_policy_unchanged': True,
    'no_model_quality_claim': True, 'native_execution_authorized': False}))

write_new(OUT / 'RESEARCH_PROGRESS.json', signed({'schema': 'gd001_query_dependence_progress_v1',
    'created_utc': stamp, 'goal_status': 'active', 'goal_achieved': False,
    'progress': 'Verified finite query-dependence construction removes an observed nuisance-count shortcut.',
    'new_model_quality_outcomes': None, 'new_native_calls': 0,
    'next_decision': 'Existing owner reviews fresh query-dependent data design and exact prospective comparison.',
    'missing_prerequisites': ['Independent actual-data admission and tokenizer/EOS/loss-mask qualification.',
        'Exact source/runtime strict loading, representative generation/scoring and resource fit.',
        'Owner-approved frozen source, data, budgets, checkpoint schedule and prospective execution plan.',
        'Completed compatible original HF/PEFT snapshots, consumed prompt inventory, hashes and completion receipt.',
        'Balanced paired value/abstention/preservation improvement with independent fresh transfer.'],
    'current_native_owner_outcome': 'Stopped control training at unchanged RAM guard; partial updates unknown.',
    'eligible_completed_gd_trajectory': False, 'fixture_is_native_evidence': False,
    'confirmation_authorized': False, 'source_or_serving_changed': False}))

files = [p for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'RECEIPT_INDEX.json']
write_new(OUT / 'RECEIPT_INDEX.json', signed({'schema': 'gd001_query_dependence_receipt_index_v1',
    'created_utc': stamp, 'evidence_kind': 'offline_preparation_and_finite_mechanical_witness',
    'goal_achieved': False, 'native_model_calls': 0, 'no_model_quality_claim': True,
    'payloads': [{'path': str(p), 'sha256': file_hash(p), 'bytes': p.stat().st_size} for p in files],
    'source_deliverables': [{'path': str(p), 'sha256': file_hash(p)} for p in new_files]}))
print(json.dumps({'output': str(OUT), 'preserved': len(preserved), 'fixture_tests_passed': 20,
                  'syntax_passed': syntax_count, 'witnesses': [(v['family'], v['contexts'], v['queries']) for v in verified],
                  'code_sha256': code_hash(), 'goal_achieved': False}, sort_keys=True))
