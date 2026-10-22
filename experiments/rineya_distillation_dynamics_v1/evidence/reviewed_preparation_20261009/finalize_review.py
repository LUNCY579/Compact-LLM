"""Metadata-only final source/archive verification before Git publication."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-distillation-dynamics-v1')
PACKAGE = ROOT / 'experiments/rineya_distillation_dynamics_v1'
ARCHIVE = PACKAGE / 'evidence/reviewed_preparation_20261009'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}

def main():
    receipt_path = ARCHIVE / 'PRESERVATION_RECEIPT.json'
    record = json.loads(receipt_path.read_text(encoding='utf-8'))
    seal = record.pop('sha256')
    assert hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest() == seal
    for relative, entry in record['files'].items():
        path = (ARCHIVE / relative).resolve()
        assert path.is_relative_to(ARCHIVE.resolve())
        assert info(path) == {'bytes': entry['bytes'], 'sha256': entry['sha256']}
    source_files, changed = {}, []
    for relative, prior in record['verified_source_files'].items():
        current = info(ROOT / relative)
        source_files[relative] = current
        if current != prior:
            changed.append(relative)
    assert changed == ['experiments/rineya_distillation_dynamics_v1/README.md'], changed
    for path in [PACKAGE / 'REVIEW_HANDOFF_20261009.md', ARCHIVE / 'README.md']:
        source_files[str(path.relative_to(ROOT)).replace('\\', '/')] = info(path)
    verification = json.loads((ARCHIVE / 'current_verification/offline-verification-review002.json').read_text())
    junit = ET.parse(ARCHIVE / 'current_verification/offline-junit-review002.xml').getroot()
    suites = list(junit.iter('testsuite'))
    tests = sum(int(s.attrib['tests']) for s in suites)
    assert tests == 179
    assert all(int(s.attrib['failures']) == int(s.attrib['errors']) == int(s.attrib['skipped']) == 0 for s in suites)
    assert verification['exit_code'] == 0 and not verification['cuda_initialized']
    assert len(verification['optimizer_tests_deselected']) == 11
    assert not any(verification['tripwire_attempts'].values())
    payload = {
        'schema': 'rdd-publication-review-acceptance-v1', 'recorded_local_date': '2026-10-09',
        'branch': 'research/rineya-distillation-dynamics-v1',
        'starting_commit': '2d7be4b5c2ba8a610b09d17455d519eecaa421e6',
        'source_files': source_files, 'preservation_receipt': info(receipt_path),
        'preserved_files_verified': len(record['files']), 'preservation_body_sha256': seal,
        'archived_snapshot_changes': changed,
        'archive_change_reason': 'README alone gained final publication handoff; reviewed Python/tests unchanged',
        'independent_code_review': {'agent': '/root/rdd_code_review', 'status': 'PASS_STATIC_SCOPE',
            'findings_resolved': 2, 'additional_negative_controls': 9, 'model_or_test_execution': False},
        'cpu_verification': {'passed': tests, 'optimizer_tests_deselected': 11,
            'tripwire_attempts': verification['tripwire_attempts'], 'torch': verification['torch'],
            'native_dependency_qualification': False},
        'human_scope': 'GitHub-only review/preservation/commit/push; GPU jobs remain prioritized and reviewer inputs deferred',
        'repository_visibility_observed': 'PRIVATE', 'force_push_or_merge_authorized': False,
        'native_execution_performed': False, 'native_qualified': False,
        'model_quality_improvement_established': False, 'retention_guards_changed': False,
        'source_or_teacher_substitution_performed': False, 'protected_banks_read': False,
    }
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                                ensure_ascii=False).encode()).hexdigest()
    with (ARCHIVE / 'REVIEW_ACCEPTANCE.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'verified_copies': len(record['files']), 'tests': tests,
                      'source_files': len(source_files), 'body_sha256': payload['sha256']}))

if __name__ == '__main__':
    main()
